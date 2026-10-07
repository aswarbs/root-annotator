"""Runs a batch of volumes through one loaded engine, sequentially, reporting per-job and
per-step progress through plain callbacks so both the GUI (Qt signals) and the CLI (print) can
drive it the same way.
"""
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from .sidecar import write_sidecar
from .volumes import stack_scan_to_npy


@dataclass
class JobStatus:
    volume: Path
    state: str = "queued"  # queued, running, done, failed, cancelled
    message: str = ""
    outputs: List[Path] = field(default_factory=list)


class JobRunner:
    def __init__(self, engine, model_key: str, out_dir: Path):
        self.engine = engine
        self.model_key = model_key
        self.out_dir = out_dir
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def _is_cancelled(self) -> bool:
        return self._cancelled

    def run_all(
        self,
        volumes: List[Path],
        on_job_status: Optional[Callable[[JobStatus], None]] = None,
        on_progress: Optional[Callable[[Path, Optional[float]], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> List[JobStatus]:
        results: List[JobStatus] = []
        loaded = False

        for volume in volumes:
            if self._cancelled:
                status = JobStatus(volume, state="cancelled")
                results.append(status)
                if on_job_status:
                    on_job_status(status)
                continue

            status = JobStatus(volume, state="running")
            if on_job_status:
                on_job_status(status)

            try:
                if not loaded:
                    if on_log:
                        on_log(f"loading {self.model_key} model...")
                    self.engine.load(progress_cb=lambda msg, frac=None: on_log and on_log(msg))
                    loaded = True

                def progress_cb(msg, frac=None, _volume=volume):
                    if on_log:
                        on_log(msg)
                    if on_progress:
                        on_progress(_volume, frac)

                run_path, stack_meta = volume, {}
                if volume.is_dir():
                    # slice-folder scan: stack to a temporary .npy the engines can load
                    run_path = self.out_dir / "_stacked" / f"{volume.name}.npy"
                    stack_meta = {"stacked_from": stack_scan_to_npy(
                        volume, run_path, progress_cb=progress_cb, cancel_cb=self._is_cancelled)}
                try:
                    result = self.engine.run(
                        run_path, self.out_dir, progress_cb=progress_cb, cancel_cb=self._is_cancelled,
                    )
                finally:
                    if volume.is_dir():
                        try:
                            run_path.unlink(missing_ok=True)
                            run_path.parent.rmdir()
                        except OSError:
                            pass

                meta = {
                    "model": self.model_key,
                    "input_volume": str(volume),
                    **stack_meta,
                    "device": result.device,
                    "elapsed_seconds": round(result.elapsed_s, 2),
                    **result.extra,
                }
                for out_path in result.outputs:
                    write_sidecar(out_path, meta)

                status.state = "done"
                status.message = f"{result.elapsed_s:.1f}s on {result.device}"
                status.outputs = result.outputs

            except Exception as exc:
                if str(exc) == "cancelled":
                    status.state = "cancelled"
                    status.message = "cancelled"
                else:
                    status.state = "failed"
                    status.message = str(exc)
                    if on_log:
                        on_log(f"ERROR on {volume.name}: {exc}\n{traceback.format_exc()}")

            results.append(status)
            if on_job_status:
                on_job_status(status)

        return results
