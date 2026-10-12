"""PyQt6 front end: Upload > Segment > Review > Export.

All the logic (running, loading earlier results, projections, clean-up with Keep/Undo) lives in
app.core.session.Session; the screens in app/ui/pages.py only draw state and emit signals. This
file is the controller between the two: it owns the UI state, polls the session twice a second and
pushes what it finds into whichever screen is showing.
"""
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PyQt6.QtCore import QSettings, Qt, QTimer, QUrl
from PyQt6.QtGui import QDesktopServices, QDragEnterEvent, QDropEvent, QFont
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QStackedWidget,
    QVBoxLayout, QWidget,
)

from .core.engine import expected_output
from .core.postprocess import clean_path_for
from .core.session import Session
from .ui import theme
from .ui.pages import ExportPage, ReviewPage, SegmentPage, UploadPage
from .ui.widgets import Stepper, button, icon_button, to_pixmap

UPLOAD, SEGMENT, REVIEW, EXPORT = range(4)
STEP_NAMES = ["Upload", "Segment", "Review", "Export"]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Root Annotator")
        self.setAcceptDrops(True)
        self.resize(1180, 800)
        self.setMinimumSize(980, 680)
        self.session = Session()
        self.settings = QSettings("RootAnnotator", "RootAnnotator")
        self.pool = ThreadPoolExecutor(max_workers=1)        # renders frames / projections off the UI thread
        self.thumb_pool = ThreadPoolExecutor(max_workers=1)  # export thumbnails

        def as_list(key):  # QSettings hands back a str for one item, None for none
            v = self.settings.value(key, [])
            return [v] if isinstance(v, str) and v else list(v or [])

        self.chosen = as_list("chosen")
        self.chosen_dirs = set(as_list("chosen_dirs"))
        self.out_auto = self.settings.value("out_auto", "true") != "false"
        self.out_text = "" if self.out_auto else str(self.settings.value("out", "") or "")
        self.kept = set(as_list("kept"))
        self.volumes = []          # [{path, name, done}] from Session.check
        self.redo_marked = set()   # already-segmented volumes queued to be segmented again
        self.snap = {"running": False, "jobs": [], "progress": 0.0}
        self.was_running = False
        self.check_error = ""
        self.run_error = ""
        self.notice = ""

        self.stage = UPLOAD
        self.seg_path = None       # volume shown on the Segment screen (follows the running job)
        self.current = None        # volume shown on the Review screen
        self.mode = "projection"   # Review: projection | slices | original
        self.pview = "side"
        self.seg_overlay = True    # Segment: draw the roots found so far on the scan
        self.show_removed = True   # Review: highlight in orange what the clean-up removed

        # viewer plumbing (one image request in flight at a time)
        self.ctx = None
        self.info = None
        self.fresh = True
        self.prev_cleaning = False
        self.shown_key = None
        self.pending = None        # (key, future)
        self.want_key = None
        self.actions_sig = None
        self.thumbs = {}

        self._build()
        self._changed(save=False)

        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    # ------------------------------------------------------------------ layout

    def _build(self) -> None:
        root = QWidget()
        root.setObjectName("Root")
        lay = QVBoxLayout(root)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.setCentralWidget(root)

        header = QFrame()
        header.setObjectName("Header")
        header.setStyleSheet(
            f"QFrame#Header {{ background: {theme.SURFACE}; border-bottom: 1px solid {theme.BORDER}; }}")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(24, 10, 24, 10)
        brand = QWidget()
        brand.setObjectName("BrandBox")
        brand.setStyleSheet("QWidget#BrandBox { background: transparent; }")
        bl = QHBoxLayout(brand)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(8)
        logo = QLabel()
        logo.setPixmap(theme.icon("leaf", theme.GREEN, 26).pixmap(26, 26))
        name = QLabel("Root Annotator")
        name.setObjectName("Brand")
        bl.addWidget(logo)
        bl.addWidget(name)
        self.stepper = Stepper(STEP_NAMES)
        self.stepper.setObjectName("StepperBox")
        self.stepper.setStyleSheet("QWidget#StepperBox { background: transparent; }")
        self.stepper.stepClicked.connect(self._goto)
        balance = QWidget()
        balance.setFixedWidth(min(brand.sizeHint().width(), 120))
        hl.addWidget(brand)
        hl.addStretch(1)
        hl.addWidget(self.stepper)
        hl.addStretch(1)
        hl.addWidget(balance)
        lay.addWidget(header)

        self.notice_bar = QFrame()
        self.notice_bar.setObjectName("Notice")
        nl = QHBoxLayout(self.notice_bar)
        nl.setContentsMargins(14, 8, 8, 8)
        self.notice_label = QLabel()
        self.notice_label.setWordWrap(True)
        dismiss = icon_button("close", "Dismiss")
        dismiss.clicked.connect(self._dismiss_notice)
        nl.addWidget(self.notice_label, 1)
        nl.addWidget(dismiss)
        self.notice_bar.hide()
        wrap = QHBoxLayout()
        wrap.setContentsMargins(32, 12, 32, 0)
        wrap.addWidget(self.notice_bar)
        lay.addLayout(wrap)

        self.pages = QStackedWidget()
        self.upload_page = UploadPage()
        self.segment_page = SegmentPage()
        self.review_page = ReviewPage()
        self.export_page = ExportPage()
        for p in (self.upload_page, self.segment_page, self.review_page, self.export_page):
            self.pages.addWidget(p)
        lay.addWidget(self.pages, 1)

        up = self.upload_page
        up.addFiles.connect(self._add_files)
        up.addFolder.connect(self._add_folder)
        up.dropped.connect(self._dropped)
        up.removed.connect(self._remove)
        up.redo.connect(self._redo_pick)
        up.changeOut.connect(self._change_out)
        up.start.connect(self._upload_primary)

        sg = self.segment_page
        sg.cancel.connect(self._confirm_cancel)
        sg.review.connect(lambda: self._goto(REVIEW))
        sg.overlayChanged.connect(self._set_overlay)
        sg.slicebar.valueChanged.connect(lambda _=0: self._on_slider(sg))

        rv = self.review_page
        rv.modeChanged.connect(self._set_mode)
        rv.viewChanged.connect(self._set_view)
        rv.volumeChosen.connect(self._view)
        rv.clean.connect(self._clean)
        rv.keep.connect(self._keep)
        rv.undo.connect(self._undo)
        rv.highlightChanged.connect(self._set_highlight)
        rv.export.connect(lambda: self._goto(EXPORT))
        rv.slicebar.valueChanged.connect(lambda _=0: self._on_slider(rv))

        ex = self.export_page
        ex.review.connect(self._view)
        ex.reveal.connect(self._reveal)
        ex.redo.connect(self._redo)
        ex.openFolder.connect(self._open_out_folder)
        ex.more.connect(lambda: self._goto(UPLOAD))

    # ------------------------------------------------------------------ navigation

    def _reachable(self, stage: int) -> bool:
        if stage == UPLOAD:
            return True
        if stage == SEGMENT:
            return bool(self.snap["jobs"])
        return any(v["done"] for v in self.volumes)

    def _goto(self, stage: int) -> None:
        if not self._reachable(stage):
            return
        if stage == REVIEW:
            done = [v["path"] for v in self.volumes if v["done"]]
            if self.current not in done:
                self._set_current(done[0])
        self.stage = stage
        self.pages.setCurrentIndex(stage)
        self._render_all()

    def _set_current(self, path: str) -> None:
        self.current = path
        self.mode = "projection"
        self.show_removed = path not in self.kept

    def _view(self, path: str) -> None:
        self._set_current(path)
        self.stage = REVIEW
        self.pages.setCurrentIndex(REVIEW)
        self._render_all()

    def _dismiss_notice(self) -> None:
        self.notice = ""
        self._render_notice()

    def _render_notice(self) -> None:
        self.notice_bar.setVisible(bool(self.notice))
        self.notice_label.setText(self.notice)

    # ------------------------------------------------------------------ volumes / output

    def _save(self) -> None:
        self.settings.setValue("chosen", self.chosen)
        self.settings.setValue("chosen_dirs", sorted(self.chosen_dirs))
        self.settings.setValue("out_auto", "true" if self.out_auto else "false")
        self.settings.setValue("out", self.out_text)
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

    def _dropped(self, paths) -> None:
        for raw in paths:
            p = Path(raw)
            if p.is_dir() or p.suffix.lower() == ".npy":
                self._add(str(p), p.is_dir())

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        self._dropped([u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()])
        self._goto(UPLOAD)

    def _remove(self, path: str) -> None:
        self.chosen = [p for p in self.chosen if p != path]
        self.chosen_dirs.discard(path)
        self._changed()

    def _auto_out(self) -> None:
        """Until the user picks their own, results go to `annotations` next to the first volume (so
        earlier results are found again), or to the default folder when nothing is chosen."""
        if not self.out_auto:
            if not self.out_text:
                self.out_text = self.session.default_out
            return
        p = self.chosen[0] if self.chosen else None
        if p:
            base = Path(p) if p in self.chosen_dirs else Path(p).parent
            self.out_text = str(base / "annotations")
        else:
            self.out_text = self.session.default_out

    def _change_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Choose where to save results", self.out_text)
        if d:
            self.out_text, self.out_auto = d, False
            self._save()
            self._refresh()

    def _changed(self, save=True) -> None:
        self._auto_out()
        if save:
            self._save()
        self._refresh()

    def _out_dir(self) -> Path:
        return Path(self.out_text.strip()).expanduser()

    def _refresh(self) -> None:
        """Re-reads which chosen volumes already have a result in the output folder."""
        if not self.chosen or not self.out_text.strip():
            self.volumes, self.check_error = [], ""
        else:
            try:
                self.volumes = self.session.check([Path(p).expanduser() for p in self.chosen], self._out_dir())
                self.check_error = ""
            except Exception as exc:
                self.volumes, self.check_error = [], str(exc)
        self.redo_marked &= {v["path"] for v in self.volumes if v["done"]}
        self._render_all()

    def _open_out_folder(self) -> None:
        out = self._out_dir()
        if out.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(out)))

    def _reveal(self, path: str) -> None:
        pred = expected_output(Path(path), self._out_dir())
        clean = clean_path_for(pred)
        target = clean if clean.is_file() else pred
        if sys.platform.startswith("win") and target.is_file():
            subprocess.Popen(["explorer", f"/select,{target}"])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.parent)))

    # ------------------------------------------------------------------ running

    def _upload_primary(self) -> None:
        todo = self._todo()
        if self.snap["running"]:
            self._goto(SEGMENT)
        elif todo:
            self._start_queued()
        else:
            self._goto(REVIEW)

    def _todo(self) -> list:
        """Volumes the next run will process: not segmented yet, or queued for a redo."""
        return [v for v in self.volumes if not v["done"] or v["path"] in self.redo_marked]

    def _start_queued(self) -> None:
        redo = [v["path"] for v in self.volumes if v["path"] in self.redo_marked]
        if redo:
            n = len(redo)
            ok = QMessageBox.question(
                self, "Replace existing results?",
                (f"The current results for {n} volumes" if n > 1 else "The current result for 1 volume")
                + ", including any clean-up, will be replaced. Your scans are not changed.")
            if ok != QMessageBox.StandardButton.Yes:
                return
        self._start(redo=redo)

    def _start(self, redo=None) -> None:
        redo = list(redo or [])
        self.run_error, self.notice = "", ""
        try:
            self.session.start_run([Path(p).expanduser() for p in self.chosen], self._out_dir(), redo)
        except Exception as exc:
            self.run_error = str(exc)
            self.notice = str(exc)
            self._goto(UPLOAD)
            self._tick()
            return
        for p in redo:  # a re-run drops the old clean-up
            self.kept.discard(p)
        self.redo_marked.clear()
        self._save()
        self.seg_path, self.was_running = None, True
        self.snap = self.session.snapshot()
        self._goto(SEGMENT)
        self._tick()

    def _redo(self, path: str) -> None:
        """From the Export screen: queue the volume on the Upload list."""
        self.redo_marked.add(path)
        self._goto(UPLOAD)

    def _redo_pick(self, pick: str) -> None:
        """Queue (or un-queue) every already-segmented volume under that file or folder for a redo."""
        pp = Path(pick)
        paths = [v["path"] for v in self.volumes
                 if v["done"] and (Path(v["path"]) == pp or pp in Path(v["path"]).parents)]
        if not paths:
            return
        if all(p in self.redo_marked for p in paths):   # second click takes it off the list again
            self.redo_marked.difference_update(paths)
        else:
            self.redo_marked.update(paths)
        self._render_all()

    def _confirm_cancel(self) -> None:
        ok = QMessageBox.question(self, "Stop segmenting?",
                                  "Progress on the volume being processed will be lost. Volumes that are "
                                  "already finished are kept.")
        if ok == QMessageBox.StandardButton.Yes:
            self.session.cancel()

    def _on_run_finished(self) -> None:
        self._refresh()
        jobs = self.snap["jobs"]
        ok = [j for j in jobs if j["state"] == "done"]
        bad = [j for j in jobs if j["state"] == "failed"]
        if bad:
            self.notice = (f"{len(bad)} volume{'s' if len(bad) != 1 else ''} could not be segmented: "
                           f"{bad[0]['name']} ({bad[0]['message']})")
        if ok:
            self._set_current(ok[0]["path"])
            self._goto(REVIEW)
        else:
            if not bad:
                self.notice = "Segmentation was cancelled."
            self._goto(UPLOAD)

    def _status(self, v: dict) -> dict:
        j = next((j for j in self.snap["jobs"] if j["path"] == v["path"]), None)
        if j and j["state"] == "running":
            return {"text": f"Segmenting… {round(self.snap['progress'] * 100)}%"}
        if j and j["state"] == "queued":
            return {"text": "Waiting"}
        if j and j["state"] == "failed":
            return {"text": "Failed: " + j["message"], "err": True}
        if j and j["state"] == "cancelled" and not v["done"]:
            return {"text": "Cancelled"}
        if v["done"]:
            return {"text": "Segmented"}
        return {"text": "Not segmented yet"}

    # ------------------------------------------------------------------ rendering

    def _render_all(self) -> None:
        self.stepper.set_state(self.stage, self._reachable)
        self._render_notice()
        if self.stage == UPLOAD:
            self._render_upload()
        elif self.stage == SEGMENT:
            self._render_segment()
        elif self.stage == REVIEW:
            self._render_review()
        else:
            self._render_export()

    def _render_upload(self) -> None:
        running = self.snap["running"]
        todo = self._todo()
        pending = len(todo)
        rows = []
        for p in self.chosen:
            pp = Path(p)
            under = [v for v in self.volumes if Path(v["path"]) == pp or pp in Path(v["path"]).parents]
            n = len(under)
            done = sum(1 for v in under if v["done"] and v["path"] not in self.redo_marked)
            queued = sum(1 for v in under if v["path"] in self.redo_marked)
            if self.check_error:
                chip, kind = "Can't read", "warn"
            elif n == 0:
                chip, kind = "No scans found", "warn"
            elif n == 1:
                chip, kind = (("Will be segmented again", "note") if queued
                              else ("Already segmented", "good") if done else ("Ready", ""))
            else:
                chip = f"{n} scans" + (f" · {done} already segmented" if done else "")                     + (f" · {queued} to redo" if queued else "")
                kind = "note" if queued else ("good" if done == n else "")
            rows.append({"path": p, "name": pp.name or p, "is_dir": p in self.chosen_dirs or pp.is_dir(),
                         "chip": chip, "chip_kind": kind, "can_redo": (done + queued) > 0 and not running,
                         "redo_label": "Cancel redo" if queued and not done else "Redo"})

        err = False
        if self.run_error:
            hint, err = self.run_error, True
        elif running:
            hint = "Segmentation is in progress."
        elif not self.chosen:
            hint = "Add at least one scan to begin."
        elif not self.out_text.strip():
            hint, err = "Choose where to save the results.", True
        elif self.check_error:
            hint, err = "Could not read the chosen scans: " + self.check_error, True
        elif not self.volumes:
            hint, err = "No scans were found in what you chose.", True
        elif pending:
            done = len(self.volumes) - pending
            hint = f"{pending} to segment" + (f", {done} already done and will be skipped" if done else "")
            if self.redo_marked:
                hint += ". Results marked to redo are replaced when you start."
        else:
            hint = "These scans are already segmented. You can review them now."

        if running:
            text, enabled = "View progress", True
        elif pending:
            text, enabled = "Start segmentation", True
        else:
            text, enabled = "Review results", bool(self.volumes)
        self.upload_page.set_state(rows, self.out_text, hint, err, text, enabled)

    def _render_segment(self) -> None:
        sp, snap = self.segment_page, self.snap
        jobs = snap["jobs"]
        n, done = len(jobs), sum(1 for j in jobs if j["state"] == "done")
        run = next((j for j in jobs if j["state"] == "running"), None)
        running = snap["running"]
        sp.set_running(running)
        plural = "s" if n != 1 else ""
        if running and run:
            sp.title.setText("Segmenting your scan")
            sp.lead.setText(f"{run.get('name', '')}  ·  Processing {done + 1} of {n} volume{plural}")
            ready = snap["progress"] > 0 or done > 0
            overall = (done + snap["progress"]) / max(n, 1)
            sp.set_progress(overall if ready else None, f"{round(overall * 100)}% complete" if ready
                            else "Getting ready…")
            note = ""
            if snap.get("device") == "cpu":
                note = "No compatible graphics card was found, so this will take longer."
            sp.note.setText(note)
        else:
            sp.title.setText("Segmentation finished")
            sp.lead.setText(f"{done} of {n} volume{plural} segmented")
            sp.set_progress(1.0, "Done")
            sp.note.setText("")
            sp.review_btn.setEnabled(self._reachable(REVIEW))

    def _render_review(self) -> None:
        rp = self.review_page
        items = [(v["path"], v["name"]) for v in self.volumes if v["done"]]
        rp.set_volumes(items, self.current)
        self._sync_review_controls()

    def _sync_review_controls(self) -> None:
        rp = self.review_page
        has_original = bool(self.current) and Path(self.current).is_file()
        if self.mode == "original" and not has_original:
            self.mode = "projection"
        rp.mode.set_key_visible("original", has_original)
        rp.mode.set_value(self.mode)
        rp.orient.set_value(self.pview)
        rp.orient.setVisible(self.mode == "projection")
        rp.slicebar.setVisible(self.mode != "projection")

    def _render_export(self) -> None:
        out = self._out_dir()
        rows = []
        for v in self.volumes:
            st = self._status(v)
            cleaned = False
            if v["done"]:
                try:
                    cleaned = clean_path_for(expected_output(Path(v["path"]), out)).is_file()
                except OSError:
                    pass
            rows.append({"path": v["path"], "name": v["name"], "status": st["text"], "err": st.get("err", False),
                         "done": v["done"], "cleaned": cleaned,
                         "can_redo": v["done"] and not self.snap["running"]})
        self.export_page.set_rows(rows, self.out_text)

    # ------------------------------------------------------------------ viewer

    def _page(self):
        return self.segment_page if self.stage == SEGMENT else self.review_page

    def _active_path(self):
        return self.seg_path if self.stage == SEGMENT else self.current

    def _reset_view(self, page) -> None:
        self.info, self.shown_key, self.want_key, self.fresh = None, None, None, True
        self.actions_sig = None
        page.canvas.set_message("")
        page.canvas.reset_view()

    def _set_overlay(self, on: bool) -> None:
        self.seg_overlay, self.shown_key = on, None

    def _set_mode(self, mode: str) -> None:
        self.mode, self.shown_key = mode, None
        self.review_page.canvas.reset_view()
        self._sync_review_controls()

    def _set_view(self, view: str) -> None:
        self.pview, self.shown_key = view, None
        self.review_page.canvas.reset_view()

    def _set_highlight(self, on: bool) -> None:
        self.show_removed, self.shown_key, self.actions_sig = on, None, None

    def _on_slider(self, page) -> None:
        page.slicebar.update_caption()

    def _diff_on(self) -> bool:
        return bool(self.stage == REVIEW and self.info and self.info.get("cleaned") and self.show_removed)

    def _viewer_tick(self) -> None:
        if self.stage not in (SEGMENT, REVIEW):
            return
        page, path = self._page(), self._active_path()
        if not path:
            return
        ctx = (self.stage, path)
        if ctx != self.ctx:
            self.ctx = ctx
            self._reset_view(page)
        try:
            i = self.session.view_info(Path(path), self._out_dir())
        except Exception:
            return
        if not i.get("available"):
            self.info = None
            page.canvas.set_message("Getting ready…" if self.snap["running"] else "No result yet.")
            return
        self.info = i
        ct = i.get("clean_task")
        cleaning = bool(ct and ct["state"] == "running" and ct["kind"] == "clean")
        if self.prev_cleaning and not cleaning and self.stage == REVIEW:
            self.mode, self.show_removed, self.shown_key = "projection", True, None
            self._sync_review_controls()
        self.prev_cleaning = cleaning

        sb = page.slicebar
        at_latest = sb.slider.value() >= sb.slider.maximum()
        sb.slider.blockSignals(True)
        sb.set_range(i["done"])
        if self.fresh:
            sb.slider.setValue(i["done"] if i["running"] else max(1, round(i["done"] / 2)))
            self.fresh = False
        elif i["running"] and at_latest:
            sb.slider.setValue(i["done"])
        if cleaning and ct["written"] > 0 and self.stage == REVIEW:
            sb.slider.setValue(min(sb.slider.maximum(), max(1, ct["written"] - 1)))
        sb.slider.blockSignals(False)
        sb.update_caption()
        self._request_image(page)
        if self.stage == REVIEW:
            self._render_actions(i, ct, cleaning)

    def _request_image(self, page) -> None:
        i, path = self.info, self._active_path()
        if not i or not path:
            return
        out, vol = self._out_dir(), Path(path)
        ct = i.get("clean_task")
        written = ct["written"] if ct and ct["state"] == "running" else 0
        mode = "seg" if self.stage == SEGMENT else self.mode
        diff = self._diff_on()
        if mode == "projection":
            key = ("p", path, self.pview, i["cleaned"], diff, written)
            fn = lambda: self.session.projection_image(vol, out, self.pview, diff)  # noqa: E731
        else:
            z = min(page.slicebar.slider.value(), i["done"])
            overlay = self.seg_overlay if mode == "seg" else mode != "original"
            key = ("s", path, z, i["cleaned"], diff and overlay, i["done"] if i["running"] else 0, written, overlay)
            fn = lambda: (None, self.session.frame(vol, out, z, diff and overlay, overlay))  # noqa: E731
        self.want_key = key
        if key != self.shown_key and self.pending is None:
            self.pending = (key, self.pool.submit(fn))

    def _collect_image(self) -> None:
        if self.pending is None or not self.pending[1].done() or self.stage not in (SEGMENT, REVIEW):
            return
        key, fut = self.pending
        self.pending = None
        canvas = self._page().canvas
        try:
            status, img = fut.result()
        except Exception as exc:
            canvas.set_message(str(exc))
            return
        if key != self.want_key:
            return
        if key[0] == "p" and img is None and status:
            if status.get("state") == "failed":
                canvas.set_message(status.get("message", "The projection could not be drawn."))
            else:
                pc = round(status.get("progress", 0) * 100)
                canvas.set_message(f"Preparing the projection (first time only)… {pc}%")
            self.shown_key = None  # try again next tick
            return
        if img is not None:
            canvas.set_image(img)
            self.shown_key = key
        elif key[0] == "s" and not key[-1]:
            canvas.set_message("The original scan is not available for this result.")
            self.shown_key = key

    def _render_actions(self, i, t, cleaning) -> None:
        rp = self.review_page
        busy = bool(t and t["state"] == "running")
        if cleaning:
            state = rp.CLEANING
            rp.set_cleaning_progress(round(t["progress"] * 100))
        elif busy:
            state = rp.SAVING
        elif i["cleaned"] and self.current not in self.kept:
            state = rp.PROPOSAL
        elif i["cleaned"]:
            state = rp.KEPT
        else:
            state = rp.IDLE
        error = t["message"] if t and t["state"] == "failed" and state == rp.IDLE else ""
        sig = (state, self.show_removed, error)
        if sig != self.actions_sig:
            self.actions_sig = sig
            rp.set_action(state, self.show_removed, error)

    # ------------------------------------------------------------------ clean-up

    def _pred(self) -> Path:
        return expected_output(Path(self.current), self._out_dir())

    def _clean(self) -> None:
        try:
            self.session.start_cleanup(self._pred())
            self.show_removed = True
        except Exception as exc:
            QMessageBox.warning(self, "Clean up", str(exc))
        self.actions_sig = None

    def _keep(self) -> None:
        try:
            self.session.keep_cleanup(self._pred())
        except Exception as exc:
            QMessageBox.warning(self, "Keep changes", str(exc))
            return
        self.kept.add(self.current)
        self._save()
        self.show_removed = False
        self.actions_sig, self.shown_key = None, None

    def _undo(self) -> None:
        if self.current in self.kept:
            ok = QMessageBox.question(
                self, "Undo the clean-up?",
                "The cleaned result will be removed. Your original segmentation is not affected.")
            if ok != QMessageBox.StandardButton.Yes:
                return
        self.session.undo_cleanup(self._pred())
        self.kept.discard(self.current)
        self._save()
        self.show_removed = True
        self.actions_sig, self.shown_key = None, None

    # ------------------------------------------------------------------ thumbnails

    def _thumb_tick(self) -> None:
        if self.stage != EXPORT:
            return
        out, now = self._out_dir(), time.monotonic()
        for v in self.volumes:
            if not v["done"]:
                continue
            p = v["path"]
            pred = expected_output(Path(p), out)
            try:
                sig = (clean_path_for(pred).is_file(), pred.stat().st_mtime_ns)
            except OSError:
                continue
            t = self.thumbs.setdefault(p, {})
            fut = t.get("future")
            if fut is not None:
                if not fut.done():
                    continue
                t["future"] = None
                try:
                    _status, img = fut.result()
                except Exception:
                    img = None
                if img is not None:
                    t["sig"] = t["req_sig"]
                    self.export_page.set_thumb(p, to_pixmap(img, 100))
                else:
                    t["retry"] = now + 1.0
                continue
            if t.get("sig") == sig and self.export_page.has_thumb(p):
                continue
            if t.get("req_sig") == sig and t.get("retry", 0) > now:
                continue
            t["req_sig"] = sig
            t["future"] = self.thumb_pool.submit(
                lambda vol=Path(p): self.session.projection_image(vol, out, "side", False))

    # ------------------------------------------------------------------ polling

    def _tick(self) -> None:
        self.snap = self.session.snapshot()
        running = self.snap["running"]
        jobs = self.snap["jobs"]
        run = next((j for j in jobs if j["state"] == "running"), None)
        if running and run and run["path"] != self.seg_path:
            self.seg_path = run["path"]
        elif not running and self.seg_path is None and jobs:
            self.seg_path = next((j["path"] for j in jobs if j["state"] == "done"), jobs[0]["path"])
        if self.was_running and not running:
            self._on_run_finished()
        self.was_running = running
        if self.stage != UPLOAD and not self._reachable(self.stage):
            self.stage = UPLOAD
            self.pages.setCurrentIndex(UPLOAD)
        self._render_all()
        self._viewer_tick()
        self._collect_image()
        self._thumb_tick()

    def closeEvent(self, event) -> None:
        self._save()
        self.session.cancel()
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.thumb_pool.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)


def run_gui() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(theme.STYLESHEET)
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run_gui())
