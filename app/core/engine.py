"""The RootSeg-LSTM engine: a ConvLSTM that walks a CT volume slice by slice along Z and
segments root in each slice, feeding its own previous prediction back in (closed loop).

The prediction is written straight into a float16 memmap on disk as slices come out (volumes are
~2 GB uint8, so a float32 prediction held in RAM would be ~9 GB). That is also what lets a viewer
show the run live: every slice below LiveInfo.done is already final in the partial file.
"""
import time
from pathlib import Path

import numpy as np
import torch

from .engine_base import CancelCB, LiveInfo, ProgressCB, RunResult, SliceCB
from .paths import checkpoint_path
from ..rootseg.model import RootTracker
from ..rootseg.norm import compute_volume_stats, normalize_slice
from ..rootseg.tiling import build_tiles

PRED_SUFFIX = "_rootseg_pred.npy"
PRED_DTYPE = np.float16


def expected_output(volume: Path, out_dir: Path) -> Path:
    """Where a finished prediction for `volume` lands (volume = .npy file or slice-folder scan)."""
    name = volume.name if volume.is_dir() else volume.stem
    return Path(out_dir) / f"{name}{PRED_SUFFIX}"


def _cuda_usable() -> bool:
    """CUDA is present *and* this PyTorch build has kernels for the GPU (an old card such as a
    GTX 1080 Ti, sm_61, is detected by torch but cannot run the model -- fall back to CPU)."""
    if not torch.cuda.is_available():
        return False
    try:
        major, minor = torch.cuda.get_device_capability(0)
        cap = major * 10 + minor
        for arch in torch.cuda.get_arch_list():
            kind, _, num = arch.partition("_")
            if num.isdigit() and ((kind == "sm" and int(num) == cap) or (kind == "compute" and int(num) <= cap)):
                return True
        return False
    except Exception:
        return True


class RootsegEngine:
    key = "rootseg"
    display_name = "RootSeg-LSTM (e4_skips_bal)"

    def __init__(self, ckpt_path: Path = None, log_every: int = 50, max_slices: int = None):
        self.ckpt_path = Path(ckpt_path) if ckpt_path else checkpoint_path()
        self.max_slices = max_slices  # only predict the first N slices (quick tests); None = all
        self.log_every = log_every
        self._model = None
        self._patch = None
        self._device = None

    def load(self, progress_cb: ProgressCB = None) -> None:
        self._device = "cuda" if _cuda_usable() else "cpu"
        if progress_cb:
            progress_cb(f"RootSeg-LSTM: loading checkpoint on {self._device}...", None)

        ckpt = torch.load(str(self.ckpt_path), map_location="cpu")
        mcfg = ckpt["config"]["model"]
        self._patch = ckpt["config"]["sequence"]["patch"]

        model = RootTracker(mcfg["base_channels"], mcfg["latent_channels"],
                            skips=mcfg.get("skips", False)).to(self._device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        self._model = model

    def run(self, volume_path: Path, out_dir: Path, progress_cb: ProgressCB = None,
            cancel_cb: CancelCB = None, slice_cb: SliceCB = None) -> RunResult:
        if self._model is None:
            self.load(progress_cb)
        out_dir.mkdir(parents=True, exist_ok=True)

        data_vol = np.load(str(volume_path), mmap_mode="r")
        D, H, W = data_vol.shape
        patch, device, model = self._patch, self._device, self._model

        stats = compute_volume_stats(data_vol)
        lo, hi = stats["p_low"], stats["p_high"]
        tiles = build_tiles(H, W, patch)
        n_tiles = len(tiles)

        out_path = out_dir / f"{volume_path.stem}{PRED_SUFFIX}"
        partial_path = out_path.with_name(out_path.name + ".partial")
        pred = np.lib.format.open_memmap(str(partial_path), mode="w+", dtype=PRED_DTYPE, shape=(D, H, W))
        # slice 0 is never predicted and stays 0

        h, c, mask_in = model.init_state(n_tiles, patch, device)
        frame_batch = torch.empty(n_tiles, 1, patch, patch, device=device)

        t0 = time.time()
        n_steps = D - 1 if not self.max_slices else min(D - 1, self.max_slices)
        try:
            with torch.no_grad():
                for z in range(n_steps):
                    if cancel_cb and cancel_cb():
                        raise RuntimeError("cancelled")

                    slice_norm = normalize_slice(data_vol[z], lo, hi)
                    for i, (y0, x0, *_rest) in enumerate(tiles):
                        frame_batch[i, 0] = torch.from_numpy(
                            np.ascontiguousarray(slice_norm[y0:y0 + patch, x0:x0 + patch]))

                    logits, (h, c) = model.step(frame_batch, mask_in, (h, c))
                    mask_in = torch.sigmoid(logits)

                    tile_probs = mask_in.cpu().numpy()
                    out_slice = np.zeros((H, W), dtype=np.float32)
                    for i, (y0, x0, oy0, oy1, ox0, ox1) in enumerate(tiles):
                        out_slice[oy0:oy1, ox0:ox1] = tile_probs[i, oy0 - y0:oy1 - y0, ox0 - x0:ox1 - x0]
                    pred[z + 1] = out_slice

                    if slice_cb:
                        slice_cb(LiveInfo(volume_path, partial_path, (D, H, W), z + 1))
                    if progress_cb and ((z + 1) % self.log_every == 0 or z == n_steps - 1):
                        progress_cb(f"{volume_path.name}: z={z + 1}/{n_steps}", (z + 1) / n_steps)

            pred.flush()
            del pred, data_vol
            partial_path.replace(out_path)
        except BaseException:
            del pred, data_vol
            partial_path.unlink(missing_ok=True)
            raise

        return RunResult(
            outputs=[out_path],
            device=device,
            elapsed_s=time.time() - t0,
            extra={"patch": patch, "checkpoint": str(self.ckpt_path),
                   "prediction_dtype": np.dtype(PRED_DTYPE).name, "slice_0": "not predicted (all zero)",
                   "max_slices": self.max_slices},
        )
