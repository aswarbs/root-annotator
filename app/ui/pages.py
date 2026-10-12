"""The four screens: Upload, Segment, Review, Export. They only draw state and emit signals; the
window (app/gui.py) owns the state and talks to app.core.session.Session."""
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFontMetrics, QPixmap
from PyQt6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QProgressBar, QScrollArea, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget,
)

from . import theme
from .widgets import (
    DropZone, ImageViewer, Segmented, SliceBar, Toggle, button, card, clear_layout, icon_button, label,
)


def _centered(page: QWidget, max_width: int, margins=(32, 24, 32, 28)):
    """A column of at most `max_width` centred in `page`; returns its layout."""
    outer = QHBoxLayout(page)
    outer.setContentsMargins(*margins)
    column = QWidget()
    column.setMaximumWidth(max_width)
    outer.addStretch(1)
    outer.addWidget(column, 100)
    outer.addStretch(1)
    lay = QVBoxLayout(column)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(16)
    return lay


def _elide(text: str, widget: QWidget, width: int) -> str:
    return QFontMetrics(widget.font()).elidedText(text, Qt.TextElideMode.ElideMiddle, width)


def _scroll() -> tuple:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    inner = QWidget()
    inner.setObjectName("Page")
    box = QVBoxLayout(inner)
    box.setContentsMargins(0, 0, 4, 0)
    box.setSpacing(10)
    box.addStretch(1)
    area.setWidget(inner)
    return area, box


# ------------------------------------------------------------------------------ Upload

class UploadPage(QWidget):
    addFiles = pyqtSignal()
    addFolder = pyqtSignal()
    dropped = pyqtSignal(list)
    removed = pyqtSignal(str)
    redo = pyqtSignal(str)
    changeOut = pyqtSignal()
    start = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setObjectName("Page")
        col = _centered(self, 780)
        col.addWidget(label("Add your root scans", "H1"))
        col.addWidget(label("Choose the CT volumes you want to segment. Your original files are never changed.",
                            "Lead"))
        self.zone = DropZone()
        self.zone.chooseFiles.connect(self.addFiles)
        self.zone.chooseFolder.connect(self.addFolder)
        self.zone.dropped.connect(self.dropped)
        col.addWidget(self.zone)

        self.list_area, self.list_box = _scroll()
        col.addWidget(self.list_area)
        self.list_area.hide()
        self._rows_sig = None

        out = QFrame()
        out.setObjectName("Row")
        ol = QHBoxLayout(out)
        ol.setContentsMargins(16, 12, 12, 12)
        ol.setSpacing(12)
        ico = QLabel()
        ico.setPixmap(theme.icon("folder", theme.GREEN, 22).pixmap(22, 22))
        text = QVBoxLayout()
        text.setSpacing(0)
        text.addWidget(label("Results will be saved to", "Mute", wrap=False))
        self.out_label = label("", wrap=False)
        text.addWidget(self.out_label)
        change = button("Change…")
        change.clicked.connect(self.changeOut)
        ol.addWidget(ico)
        ol.addLayout(text, 1)
        ol.addWidget(change)
        col.addWidget(out)

        col.addStretch(1)
        bottom = QHBoxLayout()
        self.hint = label("", "Mute")
        self.start_btn = button("Start segmentation", "Primary", "arrow_right", large=True)
        self.start_btn.clicked.connect(self.start)
        bottom.addWidget(self.hint, 1)
        bottom.addWidget(self.start_btn)
        col.addLayout(bottom)

    def set_state(self, rows, out_text, hint, hint_error, button_text, button_enabled) -> None:
        self.zone.set_compact(bool(rows))
        self.list_area.setVisible(bool(rows))
        sig = tuple((r["path"], r["chip"], r["chip_kind"], r["can_redo"], r["redo_label"]) for r in rows)
        if sig != self._rows_sig:
            self._rows_sig = sig
            clear_layout(self.list_box)
            for r in rows:
                self.list_box.addWidget(self._row(r))
            self.list_box.addStretch(1)
            self.list_area.setFixedHeight(min(len(rows) * 68 + 6, 300))
        self.out_label.setText(_elide(out_text, self.out_label, 560))
        self.out_label.setToolTip(out_text)
        self.hint.setObjectName("Error" if hint_error else "Mute")
        self.hint.style().unpolish(self.hint)
        self.hint.style().polish(self.hint)
        self.hint.setText(hint)
        self.start_btn.setText(button_text)
        self.start_btn.setEnabled(button_enabled)

    def _row(self, r) -> QFrame:
        row = QFrame()
        row.setObjectName("Row")
        lay = QHBoxLayout(row)
        lay.setContentsMargins(14, 10, 8, 10)
        lay.setSpacing(12)
        ico = QLabel()
        ico.setPixmap(theme.icon("folder" if r["is_dir"] else "file", theme.MUTED, 22).pixmap(22, 22))
        text = QVBoxLayout()
        text.setSpacing(0)
        name = label(r["name"], wrap=False)
        name.setStyleSheet("font-weight: 600;")
        sub = label(_elide(r["path"], self, 480), "Mute", wrap=False)
        sub.setToolTip(r["path"])
        text.addWidget(name)
        text.addWidget(sub)
        chip = label(r["chip"], {"good": "ChipGood", "warn": "ChipWarn", "note": "ChipNote"}.get(r["chip_kind"], "Chip"), wrap=False)
        redo = button(r["redo_label"], "Ghost", tip="Add to the list to be segmented again; the old result is replaced when you start")
        redo.setVisible(r["can_redo"])
        redo.clicked.connect(lambda _=False, p=r["path"]: self.redo.emit(p))
        rm = icon_button("close", "Remove from the list")
        rm.clicked.connect(lambda _=False, p=r["path"]: self.removed.emit(p))
        lay.addWidget(ico)
        lay.addLayout(text, 1)
        lay.addWidget(chip)
        lay.addWidget(redo)
        lay.addWidget(rm)
        return row


# ------------------------------------------------------------------------------ Segment

class SegmentPage(QWidget):
    cancel = pyqtSignal()
    review = pyqtSignal()
    overlayChanged = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.setObjectName("Page")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(32, 22, 32, 24)
        lay.setSpacing(14)
        head = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(2)
        self.title = label("Segmenting your scan", "H1")
        self.lead = label("", "Lead")
        titles.addWidget(self.title)
        titles.addWidget(self.lead)
        self.roots_toggle = Toggle("Show roots found so far")
        self.roots_toggle.setChecked(True)
        self.roots_toggle.toggled.connect(self.overlayChanged)
        head.addLayout(titles, 1)
        head.addWidget(self.roots_toggle, 0, Qt.AlignmentFlag.AlignTop)
        lay.addLayout(head)

        self.canvas = ImageViewer()
        lay.addWidget(self.canvas, 1)
        self.slicebar = SliceBar()
        lay.addWidget(self.slicebar)

        row = QHBoxLayout()
        row.setSpacing(14)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.percent = label("", wrap=False)
        self.percent.setMinimumWidth(130)
        self.percent.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.bar, 1)
        row.addWidget(self.percent)
        lay.addLayout(row)

        foot = QHBoxLayout()
        self.note = label("", "Mute")
        self.cancel_btn = button("Cancel")
        self.cancel_btn.clicked.connect(self.cancel)
        self.review_btn = button("Review results", "Primary", "arrow_right", large=True)
        self.review_btn.clicked.connect(self.review)
        foot.addWidget(self.note, 1)
        foot.addWidget(self.cancel_btn)
        foot.addWidget(self.review_btn)
        lay.addLayout(foot)

    def set_running(self, running: bool) -> None:
        self.cancel_btn.setVisible(running)
        self.review_btn.setVisible(not running)

    def set_progress(self, value, text: str) -> None:
        """`value` 0..1, or None while there is nothing meaningful to show (shows a busy bar)."""
        if value is None:
            self.bar.setRange(0, 0)
        else:
            self.bar.setRange(0, 1000)
            self.bar.setValue(round(value * 1000))
        self.percent.setText(text)


# ------------------------------------------------------------------------------ Review

class ReviewPage(QWidget):
    modeChanged = pyqtSignal(str)
    viewChanged = pyqtSignal(str)
    volumeChosen = pyqtSignal(str)
    clean = pyqtSignal()
    keep = pyqtSignal()
    undo = pyqtSignal()
    highlightChanged = pyqtSignal(bool)
    export = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setObjectName("Page")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(32, 20, 32, 22)
        lay.setSpacing(12)

        top = QHBoxLayout()
        top.setSpacing(12)
        self.title = label("", "H2", wrap=False)
        self.combo = QComboBox()
        self.combo.activated.connect(lambda _i: self.volumeChosen.emit(self.combo.currentData()))
        self.mode = Segmented([("projection", "Projection"), ("slices", "Slices"), ("original", "Original")])
        self.mode.changed.connect(self.modeChanged)
        top.addWidget(self.title)
        top.addWidget(self.combo)
        top.addStretch(1)
        top.addWidget(self.mode)
        lay.addLayout(top)

        sub = QHBoxLayout()
        sub.setSpacing(8)
        self.orient = Segmented([("side", "Side"), ("front", "Front"), ("top", "Top-down")])
        self.orient.changed.connect(self.viewChanged)
        self.zoom_out = icon_button("zoom_out", "Zoom out")
        self.zoom_in = icon_button("zoom_in", "Zoom in")
        self.fit = icon_button("fit", "Reset view")
        self.zoom_hint = label("Scroll to zoom, drag to move", "Mute", wrap=False)
        sub.addWidget(self.orient)
        sub.addStretch(1)
        sub.addWidget(self.zoom_hint)
        sub.addWidget(self.zoom_out)
        sub.addWidget(self.zoom_in)
        sub.addWidget(self.fit)
        lay.addLayout(sub)

        self.canvas = ImageViewer()
        self.zoom_out.clicked.connect(lambda: self.canvas.zoom_by(1 / 1.4))
        self.zoom_in.clicked.connect(lambda: self.canvas.zoom_by(1.4))
        self.fit.clicked.connect(self.canvas.reset_view)
        lay.addWidget(self.canvas, 1)
        self.slicebar = SliceBar()
        lay.addWidget(self.slicebar)

        # clean-up bar: one small page per state
        bar = QFrame()
        bar.setObjectName("Card")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(16, 10, 16, 10)
        self.states = QStackedWidget()
        self.states.setObjectName("Page")
        self.states.setStyleSheet("QStackedWidget { background: transparent; }")
        bl.addWidget(self.states, 1)
        self._build_states()
        lay.addWidget(bar)

        foot = QHBoxLayout()
        self.note = label("", "Mute")
        self.export_btn = button("Continue to export", "Primary", "arrow_right", large=True)
        self.export_btn.clicked.connect(self.export)
        foot.addWidget(self.note, 1)
        foot.addWidget(self.export_btn)
        lay.addLayout(foot)

    def _build_states(self) -> None:
        def page():
            w = QWidget()
            w.setObjectName("StatePage")
            w.setStyleSheet("QWidget#StatePage { background: transparent; }")
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(12)
            self.states.addWidget(w)
            return h

        # 0 idle
        h = page()
        self.clean_btn = button("Clean up", icon="sparkle")
        self.clean_btn.clicked.connect(self.clean)
        self.idle_text = label("Remove small specks: tiny disconnected pieces that are probably not roots. "
                               "You will see a preview before anything is kept.", "Mute")
        h.addWidget(self.clean_btn)
        h.addWidget(self.idle_text, 1)
        # 1 cleaning
        h = page()
        self.cleaning_text = label("Cleaning up…")
        self.cleaning_bar = QProgressBar()
        self.cleaning_bar.setRange(0, 100)
        self.cleaning_bar.setTextVisible(False)
        h.addWidget(self.cleaning_text)
        h.addWidget(self.cleaning_bar, 1)
        h.addWidget(label("Orange marks what is being removed.", "Mute", wrap=False))
        # 2 saving
        h = page()
        busy = QProgressBar()
        busy.setRange(0, 0)
        busy.setTextVisible(False)
        h.addWidget(label("Saving the cleaned result…"))
        h.addWidget(busy, 1)
        # 3 proposal
        h = page()
        self.proposal_text = label(f'Preview: <span style="color:{theme.ORANGE}; font-weight:600;">orange</span> '
                                   "shows what would be removed. Your original result is not changed.")
        self.show_removed = Toggle("Show removed")
        self.show_removed.toggled.connect(self._highlight)
        undo = button("Undo", icon="undo")
        undo.clicked.connect(self.undo)
        keep = button("Keep changes", "Primary", "check")
        keep.clicked.connect(self.keep)
        h.addWidget(self.proposal_text, 1)
        h.addWidget(self.show_removed)
        h.addWidget(undo)
        h.addWidget(keep)
        # 4 kept
        h = page()
        h.addWidget(label("Cleaned result shown. Your original result is kept safe.", "Mute"), 1)
        self.show_removed2 = Toggle("Show what was removed")
        self.show_removed2.toggled.connect(self._highlight)
        undo2 = button("Undo clean-up", "Danger")
        undo2.clicked.connect(self.undo)
        h.addWidget(self.show_removed2)
        h.addWidget(undo2)

    def _highlight(self, on: bool) -> None:
        self.highlightChanged.emit(on)

    # -- state setters used by the window
    IDLE, CLEANING, SAVING, PROPOSAL, KEPT = range(5)

    def set_action(self, state: int, highlight: bool = False, error: str = "") -> None:
        self.states.setCurrentIndex(state)
        for t in (self.show_removed, self.show_removed2):
            t.blockSignals(True)
            t.setChecked(highlight)
            t._sync(highlight)
            t.blockSignals(False)
        if state == self.IDLE:
            self.idle_text.setText(error or "Remove small specks: tiny disconnected pieces that are probably "
                                            "not roots. You will see a preview before anything is kept.")
        self.export_btn.setEnabled(state in (self.IDLE, self.KEPT))
        self.export_btn.setToolTip("" if state in (self.IDLE, self.KEPT) else "Keep or undo the clean-up first")

    def set_cleaning_progress(self, percent: int) -> None:
        self.cleaning_text.setText(f"Cleaning up… {percent}%")
        self.cleaning_bar.setValue(percent)

    def set_volumes(self, items, current: str) -> None:
        """items: [(path, name)] of volumes that have a result. One volume shows as a title."""
        multi = len(items) > 1
        self.combo.setVisible(multi)
        self.title.setVisible(not multi)
        if multi:
            sig = tuple(items)
            if getattr(self, "_items", None) != sig:
                self._items = sig
                self.combo.clear()
                for path, name in items:
                    self.combo.addItem(name, path)
            idx = self.combo.findData(current)
            if idx >= 0:
                self.combo.setCurrentIndex(idx)
        elif items:
            self.title.setText(items[0][1])


# ------------------------------------------------------------------------------ Export

class ExportPage(QWidget):
    review = pyqtSignal(str)
    reveal = pyqtSignal(str)
    redo = pyqtSignal(str)
    openFolder = pyqtSignal()
    more = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setObjectName("Page")
        col = _centered(self, 900)
        col.addWidget(label("Your results", "H1"))
        self.lead = label("", "Lead")
        col.addWidget(self.lead)

        out = QFrame()
        out.setObjectName("Row")
        ol = QHBoxLayout(out)
        ol.setContentsMargins(16, 12, 12, 12)
        ol.setSpacing(12)
        ico = QLabel()
        ico.setPixmap(theme.icon("folder", theme.GREEN, 22).pixmap(22, 22))
        text = QVBoxLayout()
        text.setSpacing(0)
        text.addWidget(label("Saved to", "Mute", wrap=False))
        self.out_label = label("", wrap=False)
        text.addWidget(self.out_label)
        open_btn = button("Open folder", icon="folder")
        open_btn.clicked.connect(self.openFolder)
        ol.addWidget(ico)
        ol.addLayout(text, 1)
        ol.addWidget(open_btn)
        col.addWidget(out)

        self.area, self.box = _scroll()
        col.addWidget(self.area, 1)
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        more = button("Segment more scans", icon="upload")
        more.clicked.connect(self.more)
        bottom.addWidget(more)
        col.addLayout(bottom)
        self._sig = None
        self._thumbs = {}

    def set_rows(self, rows, out_text: str) -> None:
        self.out_label.setText(_elide(out_text, self.out_label, 640))
        self.out_label.setToolTip(out_text)
        done = sum(1 for r in rows if r["done"])
        self.lead.setText(f"{done} of {len(rows)} volume{'s' if len(rows) != 1 else ''} segmented." if rows
                          else "Nothing has been segmented yet.")
        sig = tuple((r["path"], r["status"], r["err"], r["done"], r["cleaned"], r["can_redo"]) for r in rows)
        if sig == self._sig:
            return
        self._sig = sig
        old = self._thumbs
        self._thumbs = {}
        clear_layout(self.box)
        for r in rows:
            self.box.addWidget(self._row(r, old.get(r["path"])))
        self.box.addStretch(1)

    def _row(self, r, previous_pix) -> QFrame:
        row = QFrame()
        row.setObjectName("Row")
        lay = QHBoxLayout(row)
        lay.setContentsMargins(12, 12, 14, 12)
        lay.setSpacing(16)
        thumb = QLabel("")
        thumb.setObjectName("Thumb")
        thumb.setFixedSize(168, 100)
        thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if r["done"]:
            if previous_pix is not None and not previous_pix.isNull():
                thumb.setPixmap(previous_pix)
            else:
                thumb.setText("Preparing preview…")
            self._thumbs[r["path"]] = previous_pix
        else:
            thumb.setPixmap(theme.icon("image", "#6F8579", 30).pixmap(30, 30))
        text = QVBoxLayout()
        text.setSpacing(4)
        name = label(_elide(r["name"], self, 380), wrap=False)
        name.setToolTip(r["name"])
        name.setStyleSheet("font-weight: 600; font-size: 15px;")
        name.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        chips = QHBoxLayout()
        chips.setSpacing(6)
        kind = "ChipWarn" if r["err"] else ("ChipGood" if r["done"] else "Chip")
        chips.addWidget(label(r["status"], kind, wrap=False))
        if r["done"]:
            chips.addWidget(label("Cleaned" if r["cleaned"] else "Not cleaned", "Chip", wrap=False))
        chips.addStretch(1)
        text.addStretch(1)
        text.addWidget(name)
        text.addLayout(chips)
        text.addStretch(1)
        lay.addWidget(thumb)
        lay.addLayout(text, 1)
        if r["done"]:
            view = button("Review", "Primary", "arrow_right")
            view.clicked.connect(lambda _=False, p=r["path"]: self.review.emit(p))
            show = button("Show in folder", icon="folder")
            show.clicked.connect(lambda _=False, p=r["path"]: self.reveal.emit(p))
            lay.addWidget(view)
            lay.addWidget(show)
        if r["can_redo"]:
            again = button("Redo", "Ghost", tip="Add to the upload list to be segmented again")
            again.clicked.connect(lambda _=False, p=r["path"]: self.redo.emit(p))
            lay.addWidget(again)
        row._thumb = thumb
        row.setProperty("path", r["path"])
        return row

    def set_thumb(self, path: str, pix: QPixmap) -> None:
        self._thumbs[path] = pix
        for i in range(self.box.count()):
            w = self.box.itemAt(i).widget()
            lab = getattr(w, "_thumb", None)
            if w is not None and lab is not None and w.property("path") == path:
                lab.setPixmap(pix)
                return

    def has_thumb(self, path: str) -> bool:
        pix = self._thumbs.get(path)
        return pix is not None and not pix.isNull()
