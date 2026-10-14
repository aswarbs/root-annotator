"""Runs a batch of volumes through one loaded engine, sequentially, reporting per-job and
per-step progress through plain callbacks so both the GUI (Qt signals) and the CLI (print) can
drive it the same way.
"""
import shutil
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from .engine_base import SliceCB
from .sidecar import write_sidecar
from .tiffstack import write_tiff_stack
from .volumes import is_vol, stack_scan_to_npy, stack_vol_to_npy, volume_voxels


@dataclass
class JobStatus:
    volume: Path
    state: str = "queued"  # queued, running, done, failed, cancelled
    message: str = ""
    outputs: List[Path] = field(default_factory=list)


class JobRunner:
    def __init__(self, engine, out_dir: Path):
        self.engine = engine
        self.out_dir = out_dir
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def _is_cancelled(self) -> bool:
        return self._cancelled

    def _check_space(self, volume: Path, needs_stack: bool) -> None:
        """Fails early, with a readable message, if the output drive cannot hold the temporary
        uint8 copy (scans and .vol files) plus the float16 prediction."""
        try:
            need = volume_voxels(volume) * (3 if needs_stack else 2)
            self.out_dir.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(self.out_dir).free
        except (OSError, ValueError):
            return  # unreadable sizes are reported by the run itself
        if need and free < need:
            raise RuntimeError(f"Not enough free disk space in {self.out_dir}: about {need / 2**30:.0f} GB "
                               f"needed, {free / 2**30:.0f} GB free. Choose another results folder.")

    def run_all(
        self,
        volumes: List[Path],
        on_job_status: Optional[Callable[[JobStatus], None]] = None,
        on_progress: Optional[Callable[[Path, Optional[float]], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
        on_slice: SliceCB = None,
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
                        on_log("loading model...")
                    self.engine.load(progress_cb=lambda msg, frac=None: on_log and on_log(msg))
                    loaded = True

                def progress_cb(msg, frac=None, _volume=volume):
                    if on_log:
                        on_log(msg)
                    if on_progress:
                        on_progress(_volume, frac)

                needs_stack = volume.is_dir() or is_vol(volume)
                self._check_space(volume, needs_stack)
                run_path, stack_meta = volume, {}
                if volume.is_dir():
                    # slice-folder scan: stack to a temporary .npy the engines can load
                    run_path = self.out_dir / "_stacked" / f"{volume.name}.npy"
                    stack_meta = {"stacked_from": stack_scan_to_npy(
                        volume, run_path, progress_cb=progress_cb, cancel_cb=self._is_cancelled)}
                elif is_vol(volume):
                    # raw .vol reconstruction: convert to a temporary uint8 .npy
                    run_path = self.out_dir / "_stacked" / f"{volume.stem}.npy"
                    stack_meta = {"converted_from": stack_vol_to_npy(
                        volume, run_path, progress_cb=progress_cb, cancel_cb=self._is_cancelled)}
                try:
                    result = self.engine.run(
                        run_path, self.out_dir, progress_cb=progress_cb, cancel_cb=self._is_cancelled,
                        slice_cb=on_slice,
                    )
                finally:
                    if needs_stack:
                        try:
                            run_path.unlink(missing_ok=True)
                            run_path.parent.rmdir()
                        except OSError:
                            pass

                tiff_dirs = [write_tiff_stack(o, progress_cb=lambda msg, frac=None: progress_cb(msg),
                                              cancel_cb=self._is_cancelled) for o in result.outputs]

                meta = {
                    "model": self.engine.display_name,
                    "input_volume": str(volume),
                    **stack_meta,
                    "tiff_stack": [str(d) for d in tiff_dirs],
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
