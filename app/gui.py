"""PyQt6 front end. One column, same flow as the browser UI: choose volumes, choose an output
folder, press Segment, then a viewer and the results list.

All the logic (running, loading earlier results, projections, clean-up with Keep/Undo) lives in
app.core.session.Session, shared with the browser UI; this file only draws it and forwards clicks.
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QSettings, Qt, QTimer
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

from .core.engine import expected_output
from .core.postprocess import clean_path_for
from .core.session import Session

STYLESHEET = """
QMainWindow, QWidget#Page, QScrollArea { background: #F6F6F4; }
QWidget { color: #1D2421; font-size: 14px; }
QFrame#Card { background: #FFFFFF; border: 1px solid #DCDFDB; border-radius: 8px; }
QLabel#Title { font-size: 22px; font-weight: 600; }
QLabel#Heading { font-size: 15px; font-weight: 600; }
QLabel#Mute { color: #68726D; font-size: 13px; }
QLabel#Error { color: #B3402F; font-size: 13px; }
QLabel#Canvas { background: #101412; border-radius: 6px; color: #8FA397; }
QLineEdit { background: #F6F6F4; border: 1px solid #DCDFDB; border-radius: 6px; padding: 7px 9px; }
QLineEdit:focus { border: 1px solid #1F6F5C; }
QPushButton { background: #FFFFFF; border: 1px solid #DCDFDB; border-radius: 6px; padding: 7px 16px; }
QPushButton:hover { border: 1px solid #1F6F5C; }
QPushButton:disabled { color: #A6ADA9; }
QPushButton#Primary { background: #1F6F5C; border: 1px solid #1F6F5C; color: #FFFFFF; font-weight: 600; }
QPushButton#Primary:disabled { background: #8DB5AA; border: 1px solid #8DB5AA; }
QPushButton#Link { border: none; background: transparent; color: #1F6F5C; text-decoration: underline; padding: 2px 4px; }
QPushButton#Warn { border: none; background: transparent; color: #B3402F; text-decoration: underline; padding: 2px 4px; }
QPushButton#Seg { border-radius: 0; padding: 6px 14px; }
QPushButton#Seg:checked { background: #1F6F5C; border: 1px solid #1F6F5C; color: #FFFFFF; }
QProgressBar { background: #DCDFDB; border: none; border-radius: 3px; max-height: 6px; min-height: 6px; }
QProgressBar::chunk { background: #1F6F5C; border-radius: 3px; }
QSlider::groove:horizontal { height: 4px; background: #DCDFDB; border-radius: 2px; }
QSlider::handle:horizontal { background: #1F6F5C; width: 14px; margin: -6px 0; border-radius: 7px; }
"""

PROJECTION_VIEWS = [("side", "Side"), ("front", "Front"), ("top", "Top-down")]


def to_pixmap(rgb: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy())


class Canvas(QLabel):
    """Shows a pixmap scaled to fit, keeping its aspect ratio."""

    def __init__(self):
        super().__init__()
        self.setObjectName("Canvas")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pix = None

    def set_image(self, rgb: np.ndarray) -> None:
        self._pix = to_pixmap(rgb)
        self._rescale()

    def set_text(self, text: str) -> None:
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


def card(title: str = None, number: int = None):
    frame = QFrame()
    frame.setObjectName("Card")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(20, 16, 20, 16)
    lay.setSpacing(10)
    if title:
        head = QLabel((f"{number}   " if number else "") + title)
        head.setObjectName("Heading")
        lay.addWidget(head)
    return frame, lay


def mute(text="") -> QLabel:
    label = QLabel(text)
    label.setObjectName("Mute")
    label.setWordWrap(True)
    return label


def clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
        elif item.layout():
            clear_layout(item.layout())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Root Annotator")
        self.setAcceptDrops(True)
        self.resize(860, 920)
        self.session = Session()
        self.settings = QSettings("RootAnnotator", "RootAnnotator")
        self.pool = ThreadPoolExecutor(max_workers=1)  # renders frames / projections off the UI thread

        def as_list(key):  # QSettings hands back a str for one item, None for none
            v = self.settings.value(key, [])
            return [v] if isinstance(v, str) and v else list(v or [])

        self.chosen = as_list("chosen")
        self.chosen_dirs = set(as_list("chosen_dirs"))
        self.out_auto = self.settings.value("out_auto", "true") != "false"
        self.kept = set(as_list("kept"))
        self.volumes = []          # [{path, name, done}] from Session.check
        self.snap = {"running": False, "jobs": [], "progress": 0.0}
        self.was_running = False
        self.check_error = ""
        self.run_error = ""

        # viewer state (mirrors the web page)
        self.current = None
        self.auto_follow = True
        self.mode = "projection"
        self.pview = "side"
        self.info = None
        self.fresh = True
        self.prev_running = False
        self.prev_cleaning = False
        self.shown_key = None
        self.pending = None        # (key, future)
        self.want_key = None
        self.actions_sig = None
        self.rows_sig = None

        self._build()
        if not self.out_auto:
            self.out_edit.setText(str(self.settings.value("out", "") or ""))
        self._changed(save=False)

        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    # ------------------------------------------------------------------ layout

    def _build(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        page = QWidget()
        page.setObjectName("Page")
        outer = QHBoxLayout(page)
        outer.setContentsMargins(24, 24, 24, 40)
        column = QWidget()
        column.setMaximumWidth(780)
        col = QVBoxLayout(column)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(16)
        outer.addStretch(1)
        outer.addWidget(column, 100)
        outer.addStretch(1)
        scroll.setWidget(page)
        self.setCentralWidget(scroll)

        title = QLabel("Root Annotator")
        title.setObjectName("Title")
        col.addWidget(title)

        # 1. volumes
        c1, l1 = card("Choose volumes", 1)
        self.chosen_box = QVBoxLayout()
        self.chosen_box.setSpacing(2)
        l1.addLayout(self.chosen_box)
        row = QHBoxLayout()
        add_files = QPushButton("Add volumes…")
        add_files.clicked.connect(self._add_files)
        add_folder = QPushButton("Add folder…")
        add_folder.clicked.connect(self._add_folder)
        row.addWidget(add_files)
        row.addWidget(add_folder)
        row.addWidget(mute("or drag files and folders onto this window"), 1)
        l1.addLayout(row)
        col.addWidget(c1)

        # 2. output + segment
        c2, l2 = card("Choose where to save results", 2)
        row = QHBoxLayout()
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Output folder")
        self.out_edit.textEdited.connect(self._out_edited)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_out)
        row.addWidget(self.out_edit, 1)
        row.addWidget(browse)
        l2.addLayout(row)
        row = QHBoxLayout()
        self.start_btn = QPushButton("Segment")
        self.start_btn.setObjectName("Primary")
        self.start_btn.clicked.connect(self._start)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.session.cancel)
        self.cancel_btn.hide()
        self.hint = mute()
        row.addWidget(self.start_btn)
        row.addWidget(self.cancel_btn)
        row.addWidget(self.hint, 1)
        l2.addLayout(row)
        self.run_bar = QProgressBar()
        self.run_bar.setRange(0, 1000)
        self.run_bar.setTextVisible(False)
        self.run_bar.hide()
        l2.addWidget(self.run_bar)
        self.run_text = mute()
        l2.addWidget(self.run_text)
        col.addWidget(c2)

        # viewer
        self.viewer_card, lv = card()
        self.viewer_title = QLabel("")
        self.viewer_title.setObjectName("Heading")
        lv.addWidget(self.viewer_title)
        controls = QHBoxLayout()
        self.mode_btns = {}
        self.mode_box = QWidget()
        mb = QHBoxLayout(self.mode_box)
        mb.setContentsMargins(0, 0, 0, 0)
        mb.setSpacing(0)
        for key, label in (("projection", "Projection"), ("slices", "Slices")):
            b = QPushButton(label)
            b.setObjectName("Seg")
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, k=key: self._set_mode(k))
            self.mode_btns[key] = b
            mb.addWidget(b)
        self.view_btns = {}
        self.view_box = QWidget()
        vb = QHBoxLayout(self.view_box)
        vb.setContentsMargins(0, 0, 0, 0)
        vb.setSpacing(0)
        for key, label in PROJECTION_VIEWS:
            b = QPushButton(label)
            b.setObjectName("Seg")
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, k=key: self._set_view(k))
            self.view_btns[key] = b
            vb.addWidget(b)
        controls.addWidget(self.mode_box)
        controls.addSpacing(10)
        controls.addWidget(self.view_box)
        controls.addStretch(1)
        lv.addLayout(controls)
        self.canvas = Canvas()
        lv.addWidget(self.canvas)
        self.slider_row = QWidget()
        sr = QHBoxLayout(self.slider_row)
        sr.setContentsMargins(0, 0, 0, 0)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(1, 1)
        self.slider.valueChanged.connect(lambda _=0: self._request_image())
        self.slice_label = mute()
        sr.addWidget(self.slider, 1)
        sr.addWidget(self.slice_label)
        lv.addWidget(self.slider_row)
        self.actions = QVBoxLayout()
        lv.addLayout(self.actions)
        self.viewer_card.hide()
        col.addWidget(self.viewer_card)

        # 3. results
        self.results_card, l3 = card("Results", 3)
        self.results_box = QVBoxLayout()
        self.results_box.setSpacing(4)
        l3.addLayout(self.results_box)
        self.results_card.hide()
        col.addWidget(self.results_card)
        col.addStretch(1)

    # ------------------------------------------------------------------ volumes / output

    def _save(self) -> None:
        self.settings.setValue("chosen", self.chosen)
        self.settings.setValue("chosen_dirs", sorted(self.chosen_dirs))
        self.settings.setValue("out_auto", "true" if self.out_auto else "false")
        self.settings.setValue("out", self.out_edit.text())
        self.settings.setValue("kept", sorted(self.kept))

    def _add(self, path: str, is_dir: bool) -> None:
        if path not in self.chosen:
            self.chosen.append(path)
        if is_dir:
            self.chosen_dirs.add(path)
        self._changed()

    def _add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Choose volumes", self.settings.value("last_dir", ""),
                                                "NumPy volumes (*.npy)")
        for f in files:
            self.settings.setValue("last_dir", str(Path(f).parent))
            self._add(f, False)

    def _add_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Choose a folder of volumes or a scan folder",
                                             self.settings.value("last_dir", ""))
        if d:
            self.settings.setValue("last_dir", str(Path(d).parent))
            self._add(d, True)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.is_dir() or p.suffix.lower() == ".npy":
                self._add(str(p), p.is_dir())

    def _remove(self, path: str) -> None:
        self.chosen = [p for p in self.chosen if p != path]
        self.chosen_dirs.discard(path)
        self._changed()

    def _render_chosen(self) -> None:
        clear_layout(self.chosen_box)
        for p in self.chosen:
            row = QHBoxLayout()
            label = QLabel(p)
            label.setWordWrap(True)
            rm = QPushButton("Remove")
            rm.setObjectName("Warn")
            rm.clicked.connect(lambda _=False, q=p: self._remove(q))
            row.addWidget(label, 1)
            row.addWidget(rm)
            self.chosen_box.addLayout(row)

    def _auto_out(self) -> None:
        """Until the user picks their own, the output folder is `annotations` next to the first
        volume (so earlier results are found again), or the default folder when nothing is chosen."""
        if not self.out_auto:
            if not self.out_edit.text():
                self.out_edit.setText(self.session.default_out)
            return
        p = self.chosen[0] if self.chosen else None
        if p:
            base = Path(p) if p in self.chosen_dirs else Path(p).parent
            self.out_edit.setText(str(base / "annotations"))
        else:
            self.out_edit.setText(self.session.default_out)

    def _out_edited(self, _text: str) -> None:
        self.out_auto = False
        self._save()
        self._refresh()

    def _browse_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Choose where to save results", self.out_edit.text())
        if d:
            self.out_edit.setText(d)
            self.out_auto = False
            self._save()
            self._refresh()

    def _changed(self, save=True) -> None:
        self._auto_out()
        self._render_chosen()
        if save:
            self._save()
        self._refresh()

    def _out_dir(self) -> Path:
        return Path(self.out_edit.text().strip()).expanduser()

    def _refresh(self) -> None:
        """Re-reads which chosen volumes already have a result in the output folder."""
        if not self.chosen or not self.out_edit.text().strip():
            self.volumes, self.check_error = [], ""
        else:
            try:
                self.volumes = self.session.check([Path(p).expanduser() for p in self.chosen], self._out_dir())
                self.check_error = ""
            except Exception as exc:
                self.volumes, self.check_error = [], str(exc)
        self.rows_sig = None
        self._render_all()

    # ------------------------------------------------------------------ running

    def _start(self, redo=()) -> None:
        self.run_error = ""
        try:
            self.session.start_run([Path(p).expanduser() for p in self.chosen], self._out_dir(), list(redo))
            self.auto_follow = True
        except Exception as exc:
            self.run_error = str(exc)
        self._tick()

    def _redo(self, path: str) -> None:
        if QMessageBox.question(self, "Segment again?", "The current result will be replaced.") \
                == QMessageBox.StandardButton.Yes:
            self._start(redo=[path])

    def _status(self, v: dict) -> dict:
        j = next((j for j in self.snap["jobs"] if j["path"] == v["path"]), None)
        if j and j["state"] == "running":
            return {"text": f"Segmenting… {round(self.snap['progress'] * 100)}%", "view": True}
        if j and j["state"] == "queued":
            return {"text": "Waiting"}
        if j and j["state"] == "failed":
            return {"text": "Failed: " + j["message"], "err": True}
        if j and j["state"] == "cancelled" and not v["done"]:
            return {"text": "Cancelled"}
        if v["done"]:
            return {"text": "Segmented", "view": True, "redo": not self.snap["running"]}
        return {"text": "Not segmented yet"}

    def _render_all(self) -> None:
        pending = sum(1 for v in self.volumes if not v["done"])
        running = self.snap["running"]
        self.start_btn.setEnabled(not running and pending > 0)
        self.cancel_btn.setVisible(running)
        if self.run_error:
            self.hint.setText(self.run_error)
        elif running:
            self.hint.setText("A run is in progress.")
        elif not self.chosen:
            self.hint.setText("Add at least one volume first.")
        elif not self.out_edit.text().strip():
            self.hint.setText("Choose an output folder.")
        elif self.check_error:
            self.hint.setText("Could not read the chosen volumes: " + self.check_error)
        elif not self.volumes:
            self.hint.setText("No volumes found in what you chose.")
        elif pending:
            done = len(self.volumes) - pending
            self.hint.setText(f"{pending} to segment" + (f", {done} already done" if done else ""))
        else:
            self.hint.setText("Everything chosen is already segmented. Use Redo in Results to run one again.")

        self.results_card.setVisible(bool(self.volumes))
        sig = [(v["path"], v["done"], tuple(self._status(v).items())) for v in self.volumes]
        if sig == self.rows_sig:
            return
        self.rows_sig = sig
        clear_layout(self.results_box)
        for v in self.volumes:
            st = self._status(v)
            row = QHBoxLayout()
            name = QLabel(v["name"])
            name.setWordWrap(True)
            status = mute(st["text"])
            if st.get("err"):
                status.setObjectName("Error")
            row.addWidget(name, 2)
            row.addWidget(status, 2)
            if st.get("view"):
                b = QPushButton("View")
                b.clicked.connect(lambda _=False, p=v["path"]: self._view(p))
                row.addWidget(b)
            if st.get("redo"):
                b = QPushButton("Redo")
                b.setObjectName("Link")
                b.clicked.connect(lambda _=False, p=v["path"]: self._redo(p))
                row.addWidget(b)
            self.results_box.addLayout(row)

    # ------------------------------------------------------------------ viewer

    def _view(self, path: str) -> None:
        self.current, self.auto_follow, self.mode = path, False, "projection"
        self._reset_view()

    def _reset_view(self) -> None:
        self.info, self.shown_key, self.fresh = None, None, True
        self.canvas.set_text("")
        self.actions_sig = None

    def _set_mode(self, mode: str) -> None:
        self.mode, self.shown_key = mode, None
        self._sync_view_controls()
        self._request_image()

    def _set_view(self, view: str) -> None:
        self.pview, self.shown_key = view, None
        self._sync_view_controls()
        self._request_image()

    def _sync_view_controls(self) -> None:
        for k, b in self.mode_btns.items():
            b.setChecked(k == self.mode)
        for k, b in self.view_btns.items():
            b.setChecked(k == self.pview)
        self.view_box.setVisible(self.mode == "projection")
        self.slider_row.setVisible(self.mode == "slices")

    def _diff_on(self) -> bool:
        return bool(self.info and self.info.get("cleaned") and self.current not in self.kept)

    def _viewer_tick(self) -> None:
        if not self.current:
            self.viewer_card.hide()
            return
        self.viewer_card.show()
        self.viewer_title.setText(Path(self.current).name)
        try:
            i = self.session.view_info(Path(self.current), self._out_dir())
        except Exception:
            return
        if not i.get("available"):
            self.canvas.set_text("No result yet.")
            return
        self.info = i
        ct = i.get("clean_task")
        cleaning = bool(ct and ct["state"] == "running" and ct["kind"] == "clean")
        if self.prev_running and not i["running"]:
            self.mode = "projection"
        if self.prev_cleaning and not cleaning:
            self.mode = "projection"
        self.prev_running, self.prev_cleaning = i["running"], cleaning
        if i["running"]:
            self.mode = "slices"

        at_latest = self.slider.value() >= self.slider.maximum()
        self.slider.blockSignals(True)
        self.slider.setRange(1, max(1, i["done"]))
        if self.fresh:
            self.slider.setValue(i["done"] if i["running"] else max(1, round(i["done"] / 2)))
            self.fresh = False
        elif i["running"] and at_latest:
            self.slider.setValue(i["done"])
        if cleaning and ct["written"] > 0:
            self.slider.setValue(min(self.slider.maximum(), max(1, ct["written"] - 1)))
        self.slider.blockSignals(False)
        self.mode_box.setVisible(not i["running"])
        self._sync_view_controls()
        self._request_image()
        self._render_actions()

    def _request_image(self) -> None:
        i = self.info
        if not i or not self.current:
            return
        out, vol = self._out_dir(), Path(self.current)
        ct = i.get("clean_task")
        written = ct["written"] if ct and ct["state"] == "running" else 0
        diff = self._diff_on()
        if self.mode == "projection":
            key = ("p", self.current, self.pview, i["cleaned"], diff, written)
            fn = lambda: self.session.projection_image(vol, out, self.pview, diff)  # noqa: E731
        else:
            z = min(self.slider.value(), i["done"])
            key = ("s", self.current, z, i["cleaned"], diff, i["done"] if i["running"] else 0, written)
            self.slice_label.setText(f"Slice {z}")
            fn = lambda: (None, self.session.frame(vol, out, z, diff))  # noqa: E731
        self.want_key = key
        if key != self.shown_key and self.pending is None:
            self.pending = (key, self.pool.submit(fn))

    def _collect_image(self) -> None:
        if self.pending is None or not self.pending[1].done():
            return
        key, fut = self.pending
        self.pending = None
        try:
            status, img = fut.result()
        except Exception as exc:
            self.canvas.set_text(str(exc))
            return
        if self.mode == "projection" and key[0] == "p" and img is None and status:
            if status.get("state") == "failed":
                self.canvas.set_text(status.get("message", "could not draw the projection"))
            else:
                self.canvas.set_text(f"Preparing the projection (first time only)… {round(status.get('progress', 0) * 100)}%")
            self.shown_key = None  # try again next tick
            return
        if img is not None and key == self.want_key:
            self.canvas.set_image(img)
            self.shown_key = key

    def _render_actions(self) -> None:
        i = self.info
        t = i.get("clean_task")
        busy = bool(t and t["state"] == "running")
        sig = (i["running"], i["cleaned"], t and t["state"], t and t["kind"], self.current in self.kept)
        if busy and t["kind"] == "clean" and hasattr(self, "_clean_bar"):
            pc = round(t["progress"] * 100)
            self._clean_label.setText(f"Cleaning up… {pc}%")
            self._clean_bar.setValue(pc)
        if sig == self.actions_sig:
            return
        self.actions_sig = sig
        clear_layout(self.actions)
        self.__dict__.pop("_clean_bar", None)

        def button(text, slot, name=None):
            b = QPushButton(text)
            if name:
                b.setObjectName(name)
            b.clicked.connect(lambda _=False: slot())
            return b

        row = QHBoxLayout()
        if i["running"]:
            row.addWidget(mute("Segmenting… drag the slider to look back."), 1)
        elif busy and t["kind"] == "clean":
            box = QVBoxLayout()
            self._clean_label = mute("Cleaning up… 0%")
            self._clean_bar = QProgressBar()
            self._clean_bar.setRange(0, 100)
            self._clean_bar.setTextVisible(False)
            box.addWidget(self._clean_label)
            box.addWidget(self._clean_bar)
            self.actions.addLayout(box)
            return
        elif busy:
            row.addWidget(mute("Saving the cleaned result…"), 1)
        elif i["cleaned"] and self.current not in self.kept:
            label = QLabel('Clean-up applied. <span style="color:#d98a1d">Orange</span> shows what was removed. Keep it?')
            row.addWidget(label, 1)
            row.addWidget(button("Keep", self._keep, "Primary"))
            row.addWidget(button("Undo", self._undo))
        elif i["cleaned"]:
            row.addWidget(mute("Cleaned result shown."), 1)
            row.addWidget(button("Undo clean-up", self._undo, "Warn"))
        else:
            row.addWidget(button("Clean up", self._clean))
            msg = mute("Removes small specks that are probably not roots.")
            row.addWidget(msg, 1)
            if t and t["state"] == "failed":
                err = QLabel(t["message"])
                err.setObjectName("Error")
                row.addWidget(err)
        self.actions.addLayout(row)

    def _pred(self) -> Path:
        return expected_output(Path(self.current), self._out_dir())

    def _clean(self) -> None:
        try:
            self.session.start_cleanup(self._pred())
        except Exception as exc:
            QMessageBox.warning(self, "Clean up", str(exc))
        self.actions_sig = None

    def _keep(self) -> None:
        self.kept.add(self.current)
        self._save()
        try:
            self.session.keep_cleanup(self._pred())
        except Exception as exc:
            QMessageBox.warning(self, "Keep", str(exc))
        self.actions_sig, self.shown_key = None, None

    def _undo(self) -> None:
        self.session.undo_cleanup(self._pred())
        self.kept.discard(self.current)
        self._save()
        self.actions_sig, self.shown_key = None, None

    # ------------------------------------------------------------------ polling

    def _tick(self) -> None:
        self.snap = self.session.snapshot()
        jobs = self.snap["jobs"]
        run = next((j for j in jobs if j["state"] == "running"), None)
        n, done = len(jobs), sum(1 for j in jobs if j["state"] == "done")
        self.run_bar.setVisible(self.snap["running"])
        self.run_bar.setValue(round((done + self.snap["progress"]) / max(n, 1) * 1000))
        if self.snap["running"] and run:
            dev = self.snap.get("device")
            self.run_text.setText(f"Segmenting {done + 1} of {n}: {run.get('name', '')}"
                                  + (f" ({'GPU' if dev == 'cuda' else 'CPU'})" if dev else ""))
        else:
            self.run_text.setText("")
        if self.snap["running"] and run and self.auto_follow and self.current != run["path"]:
            self.current = run["path"]
            self._reset_view()
        if self.was_running and not self.snap["running"]:
            self._refresh()
        self.was_running = self.snap["running"]
        self._render_all()
        self._viewer_tick()
        self._collect_image()

    def closeEvent(self, event) -> None:
        self._save()
        self.session.cancel()
        self.pool.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)


def run_gui() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run_gui())
