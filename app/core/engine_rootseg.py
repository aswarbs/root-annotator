"""Wraps rootseg-lstm's ConvLSTM tracker (see rootseg-lstm/predict.py) behind the Engine
interface. Imports the model code directly from the rootseg-lstm/ repo rather than duplicating
it, so a checkout update there is picked up automatically.
"""
import time
from pathlib import Path

import numpy as np
import torch

from .engine_base import CancelCB, Engine, ProgressCB, RunResult
from .paths import ensure_repo_on_path


class RootsegEngine(Engine):
    key = "rootseg"
    display_name = "RootSeg-LSTM"

    def __init__(self, repo_dir: Path, ckpt_path: Path, log_every: int = 50):
        self.repo_dir = repo_dir
        self.ckpt_path = ckpt_path
        self.log_every = log_every
        self._model = None
        self._patch = None
        self._device = None
        self._build_tiles = None
        self._compute_volume_stats = None
        self._normalize_slice = None

    def load(self, progress_cb: ProgressCB = None) -> None:
        ensure_repo_on_path(self.repo_dir)
        from eval import build_tiles  # rootseg-lstm/eval.py
        from model import RootTracker  # rootseg-lstm/model.py
        from norm import compute_volume_stats, normalize_slice  # rootseg-lstm/norm.py

        self._build_tiles = build_tiles
        self._compute_volume_stats = compute_volume_stats
        self._normalize_slice = normalize_slice

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if progress_cb:
            progress_cb(f"RootSeg-LSTM: loading checkpoint on {self._device}...", None)

        ckpt = torch.load(str(self.ckpt_path), map_location="cpu")
        cfg = ckpt["config"]
        mcfg = cfg["model"]
        self._patch = cfg["sequence"]["patch"]

        model = RootTracker(mcfg["base_channels"], mcfg["latent_channels"]).to(self._device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        self._model = model

    def run(self, volume_path: Path, out_dir: Path, progress_cb: ProgressCB = None,
            cancel_cb: CancelCB = None) -> RunResult:
        if self._model is None:
            self.load(progress_cb)
        out_dir.mkdir(parents=True, exist_ok=True)

        data_vol = np.array(np.load(str(volume_path), mmap_mode="r"))
        D, H, W = data_vol.shape
        patch = self._patch
        device = self._device

        stats = self._compute_volume_stats(data_vol)
        lo, hi = stats["p_low"], stats["p_high"]
        tiles = self._build_tiles(H, W, patch)
        n_tiles = len(tiles)

        model = self._model
        h, c, mask_in = model.init_state(n_tiles, patch, device)
        pred = np.zeros((D, H, W), dtype=np.float32)  # slice 0 is never predicted, stays 0

        t0 = time.time()
        n_steps = D - 1
        with torch.no_grad():
            for z in range(n_steps):
                if cancel_cb and cancel_cb():
                    raise RuntimeError("cancelled")

                slice_norm = self._normalize_slice(data_vol[z], lo, hi)
                frame_batch = torch.empty(n_tiles, 1, patch, patch, device=device)
                for i, (y0, x0, *_rest) in enumerate(tiles):
                    frame_batch[i, 0] = torch.from_numpy(
                        np.ascontiguousarray(slice_norm[y0:y0 + patch, x0:x0 + patch]))

                logits, (h, c) = model.step(frame_batch, mask_in, (h, c))
                mask_in = torch.sigmoid(logits)

                for i, (y0, x0, oy0, oy1, ox0, ox1) in enumerate(tiles):
                    pred[z + 1, oy0:oy1, ox0:ox1] = (
                        mask_in[i, oy0 - y0:oy1 - y0, ox0 - x0:ox1 - x0].cpu().numpy()
                    )

                if progress_cb and ((z + 1) % self.log_every == 0 or z == n_steps - 1):
                    progress_cb(f"{volume_path.name}: z={z + 1}/{n_steps}", (z + 1) / n_steps)

        elapsed = time.time() - t0
        out_path = out_dir / f"{volume_path.stem}_rootseg_pred.npy"
        np.save(str(out_path), pred)

        return RunResult(
            outputs=[out_path],
            device=device,
            elapsed_s=elapsed,
            extra={"patch": patch, "checkpoint": str(self.ckpt_path)},
        )
