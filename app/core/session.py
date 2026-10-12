"""Everything the front ends share: the run / view / clean-up state machine, with no HTTP or Qt in it.
The browser UI (app/web/server.py) and the desktop window (app/gui.py) are thin views over `Session`.
"""
import json
import shutil
import threading
import traceback
import uuid
from pathlib import Path

import numpy as np

from .engine import RootsegEngine, expected_output
from .jobs import JobRunner
from .live import render_live_frame
from .postprocess import CLEAN_MIN_SIZE, CLEAN_THRESHOLD, clean_path_for, clean_prediction
from .projection import AXES, Projections
from .tiffstack import tiff_dir_for, write_tiff_stack
from .volumes import discover_volumes

def volume_path(raw: str) -> Path:
    p = Path(raw).expanduser()
    if not (p.is_dir() or (p.is_file() and p.suffix == ".npy")):
        raise ValueError(f"not a volume: {raw}")
    return p


def _predicted_slices(pred_path: Path, n: int) -> int:
    """Highest slice index holding a prediction (a --max-slices run stops early)."""
    try:
        meta = json.loads(pred_path.with_name(pred_path.name + ".json").read_text())
        if meta.get("max_slices"):
            return min(n - 1, int(meta["max_slices"]))
    except (OSError, ValueError):
        pass
    return n - 1


class Session:
    def __init__(self, max_slices=None, default_out=None):
        self.max_slices = max_slices
        self.default_out = default_out or str(Path.home() / "root_annotator_results")
        self.lock = threading.RLock()
        self.engine = None
        self.runner = None
        self.thread = None
        self.jobs = []        # dicts: path, name, state, message, output
        self.progress = 0.0
        self.log = []
        self.live = None      # LiveInfo of the volume currently running
        self.live_name = ""
        self.tasks = {}       # id -> {path, state, message}  (clean-up runs)
        self.projections = {}  # path:mtime -> Projections
        self.proj_tasks = {}   # path:mtime -> {state, progress, message}

    # -- model run ---------------------------------------------------------------------

    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start_run(self, inputs, out_dir: Path, redo=()):
        """Segments every volume that has no result in `out_dir` yet, plus those listed in `redo`."""
        with self.lock:
            if self.running():
                raise RuntimeError("a run is already in progress")
            volumes = discover_volumes(inputs)
            if not volumes:
                raise ValueError("no volumes found in the chosen files and folders")
            redo = {str(Path(r)) for r in redo}
            todo = [v for v in volumes if str(v) in redo or not expected_output(v, out_dir).is_file()]
            if not todo:
                raise ValueError("everything chosen is already segmented -- open a result to view it")
            out_dir.mkdir(parents=True, exist_ok=True)
            for stale in out_dir.glob("*.partial"):  # left behind by a run that was cancelled or killed
                stale.unlink(missing_ok=True)
            for v in todo:
                if str(v) in redo:
                    self._remove_clean(expected_output(v, out_dir))
            self.jobs = [{"path": str(v), "name": v.name, "state": "queued", "message": "", "output": None}
                         for v in todo]
            self.log, self.progress, self.live = [], 0.0, None
            if self.engine is None:
                self.engine = RootsegEngine(max_slices=self.max_slices)
            self.runner = JobRunner(self.engine, out_dir)
            self.thread = threading.Thread(target=self._run, args=(todo,), daemon=True)
            self.thread.start()

    def _run(self, volumes):
        by_path = {j["path"]: j for j in self.jobs}

        def on_status(st):
            with self.lock:
                j = by_path[str(st.volume)]
                j["state"], j["message"] = st.state, st.message
                if st.state == "running":
                    self.live, self.live_name, self.progress = None, st.volume.name, 0.0
                elif st.state == "done" and st.outputs:
                    j["output"] = str(st.outputs[0])
                    if self.live is not None:  # partial file was renamed to the final one
                        self.live.pred_path = st.outputs[0]

        def on_slice(info):
            self.live = info  # a single reference swap; readers see a consistent LiveInfo

        def on_progress(_vol, frac):
            if frac is not None:
                self.progress = frac

        def on_log(msg):
            with self.lock:
                self.log = (self.log + [msg])[-200:]

        try:
            self.runner.run_all(volumes, on_job_status=on_status, on_progress=on_progress,
                                on_log=on_log, on_slice=on_slice)
        except Exception:
            on_log("ERROR: " + traceback.format_exc())

    def cancel(self):
        if self.runner:
            self.runner.cancel()

    def snapshot(self) -> dict:
        with self.lock:
            return {"default_out": self.default_out, "running": self.running(), "device": getattr(self.engine, "_device", None),
                    "progress": self.progress, "jobs": list(self.jobs)}

    # -- volumes and results -----------------------------------------------------------

    def check(self, inputs, out_dir: Path) -> list:
        """Every volume under `inputs`, with its finished prediction in `out_dir` if there is one."""
        out = []
        for v in discover_volumes(inputs):
            pred = expected_output(v, out_dir)
            out.append({"path": str(v), "name": v.name, "done": pred.is_file()})
        return out

    def _view(self, volume: Path, out_dir: Path):
        """(input_path, pred_path, n_slices, last_predicted_slice, running) for a volume, or None."""
        with self.lock:
            live = self.live
            if self.running() and live is not None and any(
                    j["state"] == "running" and j["path"] == str(volume) for j in self.jobs):
                return live.input_path, live.pred_path, live.shape[0], live.done, True
        pred = expected_output(volume, out_dir)
        if not pred.is_file():
            return None
        n = np.load(str(pred), mmap_mode="r").shape[0]
        return (volume if volume.is_file() else None), pred, n, _predicted_slices(pred, n), False

    def view_info(self, volume: Path, out_dir: Path) -> dict:
        v = self._view(volume, out_dir)
        if v is None:
            return {"available": False}
        _inp, pred, n, done, running = v
        cleaned = (not running) and clean_path_for(pred).is_file()
        task = None
        for t in self.tasks.values():
            if t["path"] == str(pred):
                task = {"state": t["state"], "message": t["message"], "progress": t.get("progress", 0.0),
                        "kind": t.get("kind", "clean"), "written": t.get("written", 0)}
        return {"available": True, "n": n, "done": done, "running": running, "cleaned": cleaned,
                "progress": self.progress if running else 1.0, "clean_task": task}

    def frame(self, volume: Path, out_dir: Path, z: int, diff: bool = False, overlay: bool = True):
        """One slice as RGB. `overlay=False` is the plain scan; None if the scan itself is gone."""
        v = self._view(volume, out_dir)
        if v is None:
            return None
        inp, pred, _n, done, running = v
        z = max(1, min(z if z is not None else done, done))
        if not overlay:
            return render_live_frame(inp, pred, z, overlay=False) if inp else None
        clean = clean_path_for(pred)
        cleaning = next((t for t in self.tasks.values() if t["path"] == str(pred) and t["state"] == "running"
                         and t.get("kind", "clean") == "clean"), None)
        if cleaning is not None:  # clean-up in progress: slices below `written` are already cleaned
            partial = clean.with_name(clean.name + ".partial")
            if cleaning.get("written", 0) and partial.is_file() and z < cleaning["written"]:
                return render_live_frame(inp, partial, z, removed_from=pred)
            return render_live_frame(inp, pred, z)
        if not running and clean.is_file():  # always show the cleaned mask once there is one
            return render_live_frame(inp, clean, z, removed_from=pred if diff else None)
        return render_live_frame(inp, pred, z)

    # -- projections -------------------------------------------------------------------

    def proj_sources(self, volume: Path, out_dir: Path, diff: bool = False):
        """Files to project: the cleaned mask if there is one, else the prediction; with `diff` and a
        cleaned mask, the prediction as well (to show what the clean-up removed)."""
        pred = expected_output(volume, out_dir)
        if not pred.is_file():
            return None
        clean = clean_path_for(pred)
        if clean.is_file():
            return [clean, pred] if diff else [clean]
        return [pred]

    def live_projection(self, volume: Path, out_dir: Path):
        """The in-progress clean-up's projection arrays (see postprocess.clean_prediction), once ready."""
        pred = expected_output(volume, out_dir)
        for t in self.tasks.values():
            if t["path"] == str(pred) and t["state"] == "running" and t.get("live", {}).get("ready"):
                return t["live"]
        return None

    def projection_image(self, volume: Path, out_dir: Path, view: str, diff: bool = False):
        """(status, image) for the viewer: image is None until the projection is ready. While a
        clean-up runs this is the live projection (white = kept, orange = removed)."""
        live = self.live_projection(volume, out_dir)
        if live is not None:
            ax = AXES[view][0]
            white, orange = live[f"w{ax}"], live[f"o{ax}"]
            img = np.full(white.shape + (3,), 18, np.uint8)
            img[white] = 255
            img[orange] = (255, 170, 40)
            return {"state": "ready", "progress": 1.0}, img
        srcs = self.proj_sources(volume, out_dir, diff)
        if srcs is None:
            return {"state": "none", "progress": 0.0}, None
        status = self.proj_status(srcs)
        if status["state"] != "ready":
            return status, None
        return status, self.proj_image(srcs, view)

    def proj_status(self, sources):
        states = [self.projection(s)[1] for s in sources]
        for st in states:
            if st["state"] == "failed":
                return st
        if all(st["state"] == "ready" for st in states):
            return {"state": "ready", "progress": 1.0}
        return {"state": "loading", "progress": sum(st["progress"] for st in states) / len(states)}

    def proj_image(self, sources, view: str):
        projs = [self.projection(s)[0] for s in sources]
        if any(p is None for p in projs):
            return None
        img = projs[0].render(view, 0.5, False)
        if len(projs) == 2:  # orange where the raw prediction had root and the clean-up removed it
            raw = projs[1].render(view, 0.5, False)
            img[(raw[..., 0] == 255) & (img[..., 0] != 255)] = (255, 170, 40)
        return img

    def projection(self, path: Path):
        """Cached Projections for `path`, or None while computing (poll proj_status)."""
        key = f"{path}:{path.stat().st_mtime_ns}"  # a re-run or a new clean-up invalidates the cache
        with self.lock:
            if key in self.projections:
                return self.projections[key], {"state": "ready", "progress": 1.0}
            task = self.proj_tasks.get(key)
            if task is None or task["state"] == "failed":
                task = self.proj_tasks[key] = {"state": "loading", "progress": 0.0, "message": ""}

                def work():
                    def pcb(msg, frac=None):
                        if frac is not None:
                            task["progress"] = frac
                    try:
                        proj = Projections.compute(path, progress_cb=pcb)
                        with self.lock:
                            self.projections[key] = proj
                        task["state"] = "ready"
                    except Exception as exc:
                        task.update(state="failed", message=str(exc))

                threading.Thread(target=work, daemon=True).start()
            return None, task

    # -- clean-up ----------------------------------------------------------------------

    def start_cleanup(self, path: Path) -> str:
        tid = uuid.uuid4().hex[:8]
        task = {"path": str(path), "state": "running", "message": "cleaning up...", "progress": 0.0, "written": 0,
                "live": {}}
        self.tasks[tid] = task

        def work():
            def pcb(msg, frac=None):
                if frac is not None:
                    task["progress"] = frac
            try:
                clean_prediction(path, threshold=CLEAN_THRESHOLD, min_size=CLEAN_MIN_SIZE, progress_cb=pcb, write_tiff=False,
                                 written_cb=lambda z: task.__setitem__("written", z), live_proj=task["live"])
                task.update(state="done", message="", progress=1.0)
            except Exception as exc:
                task.update(state="failed", message=str(exc))

        threading.Thread(target=work, daemon=True).start()
        return tid

    def keep_cleanup(self, pred: Path):
        """The user accepted the cleaned mask: write it out as a TIFF stack too."""
        clean = clean_path_for(pred)
        if not clean.is_file():
            raise ValueError("there is no cleaned result to keep")
        task = {"path": str(pred), "state": "running", "message": "saving TIFF stack...", "progress": 0.0, "kind": "tiff"}
        self.tasks[uuid.uuid4().hex[:8]] = task

        def work():
            try:
                write_tiff_stack(clean)
                task.update(state="done", message="")
            except Exception as exc:
                task.update(state="failed", message=str(exc))

        threading.Thread(target=work, daemon=True).start()

    def _remove_clean(self, pred: Path):
        clean = clean_path_for(pred)
        for f in (clean, clean.with_name(clean.name + ".json"), clean.with_name(clean.name + ".proj.npz")):
            f.unlink(missing_ok=True)
        shutil.rmtree(tiff_dir_for(clean), ignore_errors=True)
        self.tasks = {k: t for k, t in self.tasks.items() if t["path"] != str(pred)}

    def undo_cleanup(self, pred: Path):
        self._remove_clean(pred)
