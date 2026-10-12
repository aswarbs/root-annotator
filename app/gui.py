"""PyQt6 front end, kept deliberately simple: drag in volumes (or use the file/folder pickers),
pick an output folder, hit Start. Built for lab users who aren't computer people -- big buttons,
plain language, no log console, a warm colour scheme instead of a default-grey dev tool.

While a volume runs, the right-hand pane shows the slices as they come out. Finished volumes can
be opened in a projection viewer (Visualise -- other volumes keep running meanwhile) and have
their noise cleaned up (Clean up).
"""
import sys
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QEvent, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QSizePolicy, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from .core.engine import RootsegEngine
from .core.jobs import JobRunner
from .core.live import render_live_frame
from .core.postprocess import CLEAN_SUFFIX, clean_path_for, CLEAN_MIN_SIZE, clean_prediction
from .core.projection import AXES, Projections
from .core.volumes import discover_volumes

STATUS_ICON = {"queued": "⏸", "running": "▶", "done": "✅", "failed": "❌", "cancelled": "⛔"}

STYLESHEET = """
QMainWindow, QWidget {
    background-color: #F5FAF6;
    color: #2E3B33;
    font-size: 13px;
}
QLabel#Heading {
    font-size: 20px;
    font-weight: 600;
    color: #2E6B45;
    padding: 4px 0 8px 0;
}
QLabel {
    color: #2E3B33;
}
QComboBox, QLineEdit {
    background: white;
    border: 1px solid #BFE0CC;
    border-radius: 6px;
    padding: 6px 8px;
    min-height: 22px;
}
QComboBox:hover, QLineEdit:hover {
    border: 1px solid #4C9A6A;
}
QListWidget {
    background: white;
    border: 1px solid #BFE0CC;
    border-radius: 8px;
    padding: 4px;
}
QListWidget::item {
    padding: 6px 4px;
    border-radius: 4px;
}
QListWidget::item:selected {
    background: #DCF0E2;
    color: #2E3B33;
}
QPushButton {
    background: white;
    border: 1px solid #BFE0CC;
    border-radius: 8px;
    padding: 8px 16px;
    color: #2E6B45;
    font-weight: 500;
}
QPushButton:hover {
    background: #EAF6EE;
}
QPushButton:disabled {
    color: #A9B7AE;
    border-color: #DDE8E0;
    background: #F5FAF6;
}
QPushButton#Primary {
    background: #4C9A6A;
    border: 1px solid #3F8459;
    color: white;
    font-weight: 600;
    padding: 10px 24px;
}
QPushButton#Primary:hover {
    background: #439065;
}
QPushButton#Primary:disabled {
    background: #BFE0CC;
    border-color: #BFE0CC;
    color: #F5FAF6;
}
QPushButton#Cancel {
    background: white;
    border: 1px solid #E0A2A2;
    color: #B24444;
}
QPushButton#Cancel:hover {
    background: #FBEAEA;
}
QLabel#Canvas {
    background: #121512;
    border-radius: 8px;
    color: #8FA397;
}
QProgressBar {
    border: 1px solid #BFE0CC;
    border-radius: 8px;
    background: white;
    text-align: center;
    min-height: 20px;
}
QProgressBar::chunk {
    background-color: #4C9A6A;
    border-radius: 7px;
}
"""


class DropListWidget(QListWidget):
    """A QListWidget that accepts dragged-in files and folders as new queue entries."""

    pathsDropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.pathsDropped.emit(paths)
        event.acceptProposedAction()


class Worker(QThread):
    job_status = pyqtSignal(object)
    progress = pyqtSignal(object, object)  # volume Path, fraction (float or None)
    finished_ok = pyqtSignal(list)
    finished_error = pyqtSignal(str)
    live = pyqtSignal(object)  # LiveInfo, after every slice (the GUI throttles redraws)

    def __init__(self, engine, out_dir, volumes):
        super().__init__()
        self.runner = JobRunner(engine, out_dir)
        self.volumes = volumes

    def cancel(self):
        self.runner.cancel()

    def run(self):
        try:
            results = self.runner.run_all(
                self.volumes,
                on_job_status=self.job_status.emit,
                on_progress=self.progress.emit,
                on_slice=self.live.emit,
            )
            self.finished_ok.emit(results)
        except Exception as exc:  # model load failure etc.
            self.finished_error.emit(str(exc))


def to_pixmap(rgb: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy())


class Canvas(QLabel):
    """Shows a pixmap scaled to fit, keeping its aspect ratio."""

    def __init__(self, placeholder=""):
        super().__init__(placeholder)
        self.setObjectName("Canvas")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(280, 280)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._pix = None

    def set_image(self, rgb: np.ndarray) -> None:
        self._pix = to_pixmap(rgb)
        self._rescale()

    def clear_image(self, text: str) -> None:
        self._pix = None
        self.setPixmap(QPixmap())
        self.setText(text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self):
        if self._pix is not None:
            self.setPixmap(self._pix.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation))


class CallWorker(QThread):
    """Runs fn(progress_cb, cancel_cb) off the GUI thread."""
    progress = pyqtSignal(str, object)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            self.done.emit(self.fn(lambda m, f=None: self.progress.emit(m, f), lambda: self._cancel))
        except Exception as exc:
            self.failed.emit("cancelled" if str(exc) == "cancelled" else str(exc))


def _split_result_path(path: Path):
    """(raw prediction path or None, cleaned mask path or None) for either file of a pair."""
    if path.name.endswith(CLEAN_SUFFIX):
        raw = path.with_name(path.name[:-len(CLEAN_SUFFIX)] + ".npy")
        return (raw if raw.is_file() else None), path
    clean = clean_path_for(path)
    return path, (clean if clean.is_file() else None)


class VisualiserWindow(QWidget):
    """Max-intensity projection of a finished prediction (and its cleaned mask, once one exists).
    Independent of the main window, so other volumes keep running while it is open."""

    def __init__(self, path: Path, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.path = Path(path)
        self.setWindowTitle(f"Visualise - {self.path.name}")
        self.resize(820, 760)
        self.projections = {}  # source key -> Projections
        self.worker = None

        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.source_combo = QComboBox()
        self.axis_combo = QComboBox()
        for key, (_ax, label) in AXES.items():
            self.axis_combo.addItem(label, userData=key)
        self.depth_check = QCheckBox("Colour by depth")
        self.thr_label = QLabel()
        self.thr_slider = QSlider(Qt.Orientation.Horizontal)
        self.thr_slider.setRange(1, 99)
        self.thr_slider.setValue(50)
        save_btn = QPushButton("Save PNG...")
        save_btn.clicked.connect(self._save_png)
        controls.addWidget(self.source_combo)
        controls.addWidget(self.axis_combo)
        controls.addWidget(self.depth_check)
        controls.addWidget(self.thr_label)
        controls.addWidget(self.thr_slider, stretch=1)
        controls.addWidget(save_btn)
        layout.addLayout(controls)

        self.canvas = Canvas("loading...")
        layout.addWidget(self.canvas, stretch=1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)

        self._img = None
        self._refreshing = False
        self.refresh_sources()
        for sig in (self.source_combo.currentIndexChanged, self.axis_combo.currentIndexChanged,
                    self.depth_check.toggled, self.thr_slider.valueChanged):
            sig.connect(self._on_control)
        self._on_control()

    def refresh_sources(self):
        raw, clean = _split_result_path(self.path)
        want = [("raw", "Raw prediction", raw), ("clean", "Cleaned mask", clean)]
        have = [(k, l, p) for k, l, p in want if p is not None]
        current = self.source_combo.currentData()
        self._refreshing = True
        self.source_combo.clear()
        self.paths = {}
        for k, label, p in have:
            self.source_combo.addItem(label, userData=k)
            self.paths[k] = p
        i = self.source_combo.findData(current)
        self.source_combo.setCurrentIndex(max(i, 0) if current else (0 if raw else 0))
        self._refreshing = False

    def event(self, e):
        if e.type() == QEvent.Type.WindowActivate and not self._refreshing:
            n = self.source_combo.count()
            self.refresh_sources()
            if self.source_combo.count() != n:
                self._on_control()
        return super().event(e)

    def _on_control(self, *_):
        if self._refreshing or self.source_combo.currentData() is None:
            return
        key = self.source_combo.currentData()
        self.thr_slider.setEnabled(key == "raw")
        self.thr_label.setText(f"threshold {self.thr_slider.value() / 100:.2f}" if key == "raw" else "binary")
        if key in self.projections:
            self._redraw()
        elif self.worker is None:
            self._load(key)

    def _load(self, key):
        path = self.paths[key]
        self.canvas.clear_image("computing projection (first open only)...")
        self.worker = CallWorker(lambda pcb, ccb: Projections.compute(path, pcb, ccb))
        self.worker.progress.connect(lambda m, f: f is not None and self.progress.setValue(int(f * 100)))
        self.worker.done.connect(lambda proj, k=key: self._loaded(k, proj))
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _loaded(self, key, proj):
        self.projections[key] = proj
        self.worker = None
        self.progress.setValue(100)
        self._on_control()

    def _failed(self, msg):
        self.worker = None
        self.canvas.clear_image(f"couldn't load: {msg}")

    def _redraw(self):
        key = self.source_combo.currentData()
        proj = self.projections.get(key)
        if proj is None:
            return
        thr = self.thr_slider.value() / 100 if key == "raw" else 0.5
        self._img = proj.render(self.axis_combo.currentData(), thr, self.depth_check.isChecked())
        self.canvas.set_image(self._img)

    def _save_png(self):
        if self._img is None:
            return
        from PIL import Image
        fn, _ = QFileDialog.getSaveFileName(self, "Save projection", f"{self.path.stem}_projection.png", "PNG (*.png)")
        if fn:
            Image.fromarray(self._img).save(fn)

    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.cancel()
            self.worker.wait(2000)
        super().closeEvent(event)


class CleanupDialog(QDialog):
    """Threshold + remove small 3D components. Non-modal and runs in a thread, so it can run
    alongside a model job."""
    cleaned = pyqtSignal(str)

    def __init__(self, pred_path: Path, parent=None):
        super().__init__(parent)
        self.pred_path = Path(pred_path)
        self.setWindowTitle(f"Clean up - {self.pred_path.name}")
        self.setMinimumWidth(460)
        self.worker = None

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.thr = QDoubleSpinBox()
        self.thr.setRange(0.05, 0.99)
        self.thr.setSingleStep(0.05)
        self.thr.setValue(0.5)
        self.min_size = QSpinBox()
        self.min_size.setRange(1, 10_000_000)
        self.min_size.setValue(CLEAN_MIN_SIZE)
        self.min_size.setSuffix(" voxels")
        form.addRow("Probability threshold:", self.thr)
        form.addRow("Remove pieces smaller than:", self.min_size)
        layout.addLayout(form)
        layout.addWidget(QLabel("Keeps a root piece only if it is at least this big in 3D (26-connected).\n"
                                "The raw prediction is left untouched; the result is saved next to it."))
        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        layout.addWidget(self.bar)
        row = QHBoxLayout()
        self.run_btn = QPushButton("Clean up")
        self.run_btn.setObjectName("Primary")
        self.run_btn.clicked.connect(self._run)
        self.vis_btn = QPushButton("Visualise result")
        self.vis_btn.setEnabled(False)
        row.addWidget(self.run_btn)
        row.addWidget(self.vis_btn)
        row.addStretch(1)
        layout.addLayout(row)

    def _run(self):
        thr, min_size = self.thr.value(), self.min_size.value()
        self.run_btn.setEnabled(False)
        self.status.setText("working...")
        self.worker = CallWorker(lambda pcb, ccb: clean_prediction(
            self.pred_path, threshold=thr, min_size=min_size, progress_cb=pcb, cancel_cb=ccb))
        self.worker.progress.connect(lambda m, f: (self.status.setText(m), f is not None and self.bar.setValue(int(f * 100))))
        self.worker.done.connect(self._done)
        self.worker.failed.connect(lambda m: (self.status.setText(f"failed: {m}"), self.run_btn.setEnabled(True)))
        self.worker.start()

    def _done(self, r):
        self.bar.setValue(100)
        self.run_btn.setEnabled(True)
        self.status.setText(
            f"Kept {r['components_kept']:,} of {r['components_total']:,} pieces "
            f"({r['voxels_after']:,} of {r['voxels_before']:,} voxels).\nSaved: {r['output']}")
        self.vis_btn.setEnabled(True)
        self.cleaned.emit(r["output"])

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(5000)
        super().closeEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Root Annotator")
        self.resize(1180, 640)

        self.worker = None
        self.items_by_path = {}
        self.outputs = {}      # volume Path -> finished prediction Path
        self.child_windows = []  # keep visualiser / clean-up windows alive
        self.live_info = None
        self.live_dirty = False

        root = QWidget()
        self.setCentralWidget(root)
        columns = QHBoxLayout(root)
        columns.setContentsMargins(20, 16, 20, 20)
        columns.setSpacing(20)
        left = QWidget()
        left.setMaximumWidth(520)
        layout = QVBoxLayout(left)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        columns.addWidget(left, stretch=0)
        self.live_pane = QWidget()
        self.live_pane.setLayout(self._build_live_pane())
        self.live_pane.hide()  # appears once a run produces its first slice
        columns.addWidget(self.live_pane, stretch=1)

        heading = QLabel("Root Annotator")
        heading.setObjectName("Heading")
        layout.addWidget(heading)

        # -- queue --
        layout.addWidget(QLabel("Volumes (drag files or a folder here, or type a path):"))
        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("type or paste a file/folder path, then press Enter")
        self.path_edit.returnPressed.connect(self._add_typed_path)
        add_path_btn = QPushButton("Add")
        add_path_btn.clicked.connect(self._add_typed_path)
        path_row.addWidget(self.path_edit, stretch=1)
        path_row.addWidget(add_path_btn)
        layout.addLayout(path_row)
        self.queue_list = DropListWidget()
        self.queue_list.pathsDropped.connect(self._add_paths)
        self.queue_list.itemSelectionChanged.connect(self._update_result_buttons)
        layout.addWidget(self.queue_list, stretch=1)

        queue_btn_row = QHBoxLayout()
        add_files_btn = QPushButton("Choose Files...")
        add_files_btn.clicked.connect(self._add_files_dialog)
        add_folder_btn = QPushButton("Choose Folder...")
        add_folder_btn.clicked.connect(self._add_folder_dialog)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._remove_selected)
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._clear_queue)
        for b in (add_files_btn, add_folder_btn, remove_btn, clear_btn):
            queue_btn_row.addWidget(b)
        layout.addLayout(queue_btn_row)

        # -- output dir --
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Save results to:"))
        self.out_dir_edit = QLineEdit()
        self.out_dir_edit.setPlaceholderText("choose a folder for the results...")
        out_row.addWidget(self.out_dir_edit, stretch=1)
        out_browse_btn = QPushButton("Browse...")
        out_browse_btn.clicked.connect(self._browse_out_dir)
        out_row.addWidget(out_browse_btn)
        layout.addLayout(out_row)

        # -- run controls --
        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Start")
        self.run_btn.setObjectName("Primary")
        self.run_btn.clicked.connect(self._run)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("Cancel")
        self.cancel_btn.clicked.connect(self._cancel)
        self.cancel_btn.setEnabled(False)
        run_row.addWidget(self.run_btn)
        run_row.addWidget(self.cancel_btn)
        run_row.addStretch(1)
        layout.addLayout(run_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        # -- results of finished volumes --
        result_row = QHBoxLayout()
        self.vis_btn = QPushButton("Visualise")
        self.vis_btn.clicked.connect(self._visualise)
        self.clean_btn = QPushButton("Clean up")
        self.clean_btn.clicked.connect(self._clean_up)
        open_btn = QPushButton("Open result...")
        open_btn.clicked.connect(self._open_result)
        for b in (self.vis_btn, self.clean_btn, open_btn):
            result_row.addWidget(b)
        layout.addLayout(result_row)
        self._update_result_buttons()

        self.live_timer = QTimer(self)
        self.live_timer.setInterval(300)
        self.live_timer.timeout.connect(self._tick_live)
        self.live_timer.start()

    # -- live view --

    def _build_live_pane(self):
        pane = QVBoxLayout()
        self.live_title = QLabel("Live view")
        self.live_title.setObjectName("Heading")
        pane.addWidget(self.live_title)
        self.live_canvas = Canvas("Start a run to watch slices come out here.")
        pane.addWidget(self.live_canvas, stretch=1)
        self.live_label = QLabel("")
        pane.addWidget(self.live_label)
        return pane

    def _on_live(self, info):
        self.live_info = info
        self.live_dirty = True

    def _tick_live(self):
        if self.live_info is None or not self.live_dirty:
            return
        self.live_dirty = False
        self.live_pane.show()
        self._draw_live()

    def _draw_live(self):
        info = self.live_info
        if info is None:
            return
        z = info.done  # always follow the latest predicted slice
        try:
            self.live_canvas.set_image(render_live_frame(info.input_path, info.pred_path, z))
        except (OSError, ValueError):
            return  # file just renamed/removed as the job finished
        self.live_label.setText(f"slice {z} of {info.shape[0] - 1}")

    # -- finished-volume actions --

    def _selected_output(self):
        for item in self.queue_list.selectedItems():
            for v, it in self.items_by_path.items():
                if it is item and v in self.outputs:
                    return self.outputs[v]
        return None

    def _update_result_buttons(self):
        has = self._selected_output() is not None
        self.vis_btn.setEnabled(has)
        self.clean_btn.setEnabled(has)

    def _track(self, window):
        self.child_windows.append(window)
        window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        window.destroyed.connect(lambda _=None, w=window: self.child_windows.remove(w) if w in self.child_windows else None)
        window.show()

    def _visualise(self, *, path=None):
        path = path or self._selected_output()
        if path is not None:
            self._track(VisualiserWindow(Path(path)))

    def _clean_up(self, *, path=None):
        path = path or self._selected_output()
        if path is None:
            return
        dlg = CleanupDialog(Path(path))
        dlg.vis_btn.clicked.connect(lambda: self._visualise(path=path))
        self._track(dlg)

    def _open_result(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Open a prediction (.npy)", "", "NumPy volumes (*.npy)")
        if not fn:
            return
        box = QMessageBox(self)
        box.setWindowTitle("Open result")
        box.setText(Path(fn).name)
        vis = box.addButton("Visualise", QMessageBox.ButtonRole.AcceptRole)
        clean = box.addButton("Clean up", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        if box.clickedButton() is vis:
            self._visualise(path=Path(fn))
        elif box.clickedButton() is clean:
            self._clean_up(path=Path(fn))

    # -- queue management --

    def _add_paths(self, raw_paths):
        try:
            volumes = discover_volumes(raw_paths)
        except FileNotFoundError as exc:
            QMessageBox.warning(self, "Hmm, that's not a volume", str(exc))
            return
        added = 0
        for v in volumes:
            if v in self.items_by_path:
                continue
            item = QListWidgetItem(f"{STATUS_ICON['queued']}  {v}")
            self.queue_list.addItem(item)
            self.items_by_path[v] = item
            added += 1
        if added and not self.out_dir_edit.text().strip():
            default_out = volumes[0].parent / "annotations"
            self.out_dir_edit.setText(str(default_out))

    def _add_typed_path(self):
        text = self.path_edit.text().strip().strip("\"'")
        if text:
            self._add_paths([str(Path(text).expanduser())])
            self.path_edit.clear()

    def _add_files_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Choose volumes", "", "NumPy volumes (*.npy)")
        if files:
            self._add_paths(files)

    def _add_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose a scan folder or a folder of volumes/scans")
        if folder:
            self._add_paths([folder])

    def _remove_selected(self):
        for item in self.queue_list.selectedItems():
            row = self.queue_list.row(item)
            self.queue_list.takeItem(row)
            self.items_by_path = {p: it for p, it in self.items_by_path.items() if it is not item}

    def _clear_queue(self):
        self.queue_list.clear()
        self.items_by_path.clear()

    def _browse_out_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose where to save results")
        if folder:
            self.out_dir_edit.setText(folder)

    # -- run --

    def _set_item_status(self, path: Path, state: str, message: str = ""):
        item = self.items_by_path.get(path)
        if item is None:
            return
        suffix = f"  ({message})" if message else ""
        item.setText(f"{STATUS_ICON.get(state, '?')}  {path}{suffix}")

    def _run(self):
        if not self.items_by_path:
            QMessageBox.information(self, "No volumes yet", "Add at least one .npy file, scan folder, or folder of either first.")
            return
        out_dir_text = self.out_dir_edit.text().strip()
        if not out_dir_text:
            QMessageBox.information(self, "Choose a save location", "Pick a folder to save results to first.")
            return

        engine = RootsegEngine()
        out_dir = Path(out_dir_text)
        volumes = list(self.items_by_path.keys())

        for v in volumes:
            self._set_item_status(v, "queued")
        self.progress_bar.setValue(0)

        self.worker = Worker(engine, out_dir, volumes)
        self.worker.job_status.connect(self._on_job_status)
        self.worker.progress.connect(self._on_progress)
        self.worker.live.connect(self._on_live)
        self.worker.finished_ok.connect(self._on_finished_ok)
        self.worker.finished_error.connect(self._on_finished_error)
        self.worker.start()

        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

    def _cancel(self):
        if self.worker:
            self.worker.cancel()
        self.cancel_btn.setEnabled(False)

    def _on_job_status(self, status):
        self._set_item_status(status.volume, status.state, status.message)
        if status.state == "running":
            self.live_info = None
            self.live_title.setText(f"Live view - {status.volume.name}")
        elif status.state == "done" and status.outputs:
            self.outputs[status.volume] = status.outputs[0]
            if self.live_info is not None:  # partial file was renamed to the final one
                self.live_info.pred_path = status.outputs[0]
                self.live_dirty = True
            self._update_result_buttons()

    def _on_progress(self, volume, frac):
        if frac is not None:
            self.progress_bar.setValue(int(frac * 100))

    def _finish_run(self):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.worker = None

    def _on_finished_ok(self, results):
        n_done = sum(1 for r in results if r.state == "done")
        n_failed = sum(1 for r in results if r.state == "failed")
        if n_failed:
            QMessageBox.warning(
                self, "Finished with some errors",
                f"{n_done} of {len(results)} volumes finished. {n_failed} failed -- "
                "check the list above for details.",
            )
        self._finish_run()

    def _on_finished_error(self, message: str):
        QMessageBox.critical(self, "Something went wrong", message)
        self._finish_run()


def run_gui() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run_gui())
