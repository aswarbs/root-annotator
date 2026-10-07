"""PyQt6 front end, kept deliberately simple: pick a model, drag in volumes (or use the file/
folder pickers), pick an output folder, hit Start. Built for lab users who aren't computer
people -- big buttons, plain language, no log console, a warm colour scheme instead of a
default-grey dev tool.
"""
import sys
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QFont
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QVBoxLayout, QWidget,
)

from .core.jobs import JobRunner
from .core.registry import MODEL_ORDER, build_registry
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

    def __init__(self, engine, model_key, out_dir, volumes):
        super().__init__()
        self.runner = JobRunner(engine, model_key, out_dir)
        self.volumes = volumes

    def cancel(self):
        self.runner.cancel()

    def run(self):
        try:
            results = self.runner.run_all(
                self.volumes,
                on_job_status=self.job_status.emit,
                on_progress=self.progress.emit,
            )
            self.finished_ok.emit(results)
        except Exception as exc:  # model load failure etc.
            self.finished_error.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Root Annotator")
        self.resize(640, 560)

        self.registry = build_registry()
        self.worker = None
        self.items_by_path = {}

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(20, 16, 20, 20)
        layout.setSpacing(12)

        heading = QLabel("\U0001F331 Root Annotator")
        heading.setObjectName("Heading")
        layout.addWidget(heading)

        # -- model selector --
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        for key in MODEL_ORDER:
            self.model_combo.addItem(self.registry[key]["short_name"], userData=key)
        model_row.addWidget(self.model_combo, stretch=1)
        layout.addLayout(model_row)

        # -- queue --
        layout.addWidget(QLabel("Volumes (drag files or a folder here):"))
        self.queue_list = DropListWidget()
        self.queue_list.pathsDropped.connect(self._add_paths)
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

        model_key = self.model_combo.currentData()
        engine = self.registry[model_key]["make_engine"]()
        out_dir = Path(out_dir_text)
        volumes = list(self.items_by_path.keys())

        for v in volumes:
            self._set_item_status(v, "queued")
        self.progress_bar.setValue(0)

        self.worker = Worker(engine, model_key, out_dir, volumes)
        self.worker.job_status.connect(self._on_job_status)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_finished_ok)
        self.worker.finished_error.connect(self._on_finished_error)
        self.worker.start()

        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.model_combo.setEnabled(False)

    def _cancel(self):
        if self.worker:
            self.worker.cancel()
        self.cancel_btn.setEnabled(False)

    def _on_job_status(self, status):
        self._set_item_status(status.volume, status.state, status.message)

    def _on_progress(self, volume, frac):
        if frac is not None:
            self.progress_bar.setValue(int(frac * 100))

    def _finish_run(self):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.model_combo.setEnabled(True)
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
