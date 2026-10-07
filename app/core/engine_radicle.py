"""Wraps radicle's two-stage cascade (see radicle/inference/infer_seq.py) behind the Engine
interface, running the sequential traversal strategy only (visits every patch -- no seed
coordinates needed, unlike the BFS / vector-guided strategies).
"""
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .engine_base import CancelCB, Engine, ProgressCB, RunResult
from .paths import ensure_repo_on_path


class RadicleEngine(Engine):
    key = "radicle"
    display_name = "Radicle"

    def __init__(self, repo_dir: Path, first_ckpt: Path, second_ckpt: Path,
                 patch_size: int = 128, batch_size: int = 2, log_every: int = 10):
        self.repo_dir = repo_dir
        self.first_ckpt = first_ckpt
        self.second_ckpt = second_ckpt
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.log_every = log_every
        self._first_stage = None
        self._second_stage = None
        self._device = None
        self._downsample = None

    def load(self, progress_cb: ProgressCB = None) -> None:
        ensure_repo_on_path(self.repo_dir)
        from utils.model import DEVICE, DOWNSAMPLE, UNet3D, UNet3DRefine

        self._device = DEVICE
        self._downsample = DOWNSAMPLE

        if progress_cb:
            progress_cb(f"Radicle: loading checkpoints on {self._device}...", None)

        first_stage = UNet3D().to(self._device)
        first_stage.load_state_dict(torch.load(str(self.first_ckpt), map_location=self._device))
        first_stage.eval()

        second_stage = UNet3DRefine(in_channels=5).to(self._device)
        second_stage.load_state_dict(torch.load(str(self.second_ckpt), map_location=self._device))
        second_stage.eval()

        self._first_stage = first_stage
        self._second_stage = second_stage

    def run(self, volume_path: Path, out_dir: Path, progress_cb: ProgressCB = None,
            cancel_cb: CancelCB = None) -> RunResult:
        if self._first_stage is None:
            self.load(progress_cb)
        out_dir.mkdir(parents=True, exist_ok=True)

        volume = np.load(str(volume_path))
        D, H, W = volume.shape
        ps = self.patch_size
        stride = ps // 2
        device = self._device
        downsample = self._downsample

        mask_accum = np.zeros((1, D, H, W), dtype=np.float32)
        mask_count = np.zeros_like(mask_accum)

        vec_D, vec_H, vec_W = (math.ceil(n / downsample) for n in (D, H, W))
        vec_accum = np.zeros((3, vec_D, vec_H, vec_W), dtype=np.float32)
        vec_count = np.zeros_like(vec_accum)

        order_map = np.zeros((D, H, W), dtype=np.int32)

        coords = [
            (z, y, x)
            for z in range(0, D - ps + 1, stride)
            for y in range(0, H - ps + 1, stride)
            for x in range(0, W - ps + 1, stride)
        ]
        if not coords:
            raise ValueError(
                f"volume {volume_path.name} ({D}x{H}x{W}) is smaller than the {ps}^3 patch size"
            )

        volume_f = volume.astype(np.float32) / 255.0
        n_batches = math.ceil(len(coords) / self.batch_size)
        t0 = time.time()

        with torch.no_grad():
            for b, batch_start in enumerate(range(0, len(coords), self.batch_size)):
                if cancel_cb and cancel_cb():
                    raise RuntimeError("cancelled")

                batch_coords = coords[batch_start:batch_start + self.batch_size]
                patches = []
                for z, y, x in batch_coords:
                    p = volume_f[z:z + ps, y:y + ps, x:x + ps]
                    p = (p - p.mean()) / (p.std() + 1e-6)
                    patches.append(p)
                patches_t = torch.from_numpy(np.stack(patches)[:, None]).to(device)

                mask1, vec1 = self._first_stage(patches_t)
                vec_up = F.interpolate(vec1, size=patches_t.shape[2:], mode="trilinear", align_corners=False)
                vec_up = F.normalize(vec_up, dim=1)
                inp2 = torch.cat([patches_t, torch.sigmoid(mask1), vec_up], dim=1)
                logits2 = self._second_stage(inp2)
                mask2 = torch.sigmoid(mask1 + logits2)

                mask_np = mask2[:, 0].cpu().numpy()
                vec_np = vec1.cpu().numpy()

                for i, (z, y, x) in enumerate(batch_coords):
                    step_n = batch_start + i + 1
                    mask_accum[0, z:z + ps, y:y + ps, x:x + ps] += mask_np[i]
                    mask_count[0, z:z + ps, y:y + ps, x:x + ps] += 1

                    zd, yd, xd = z // downsample, y // downsample, x // downsample
                    pd, ph, pw = vec_np.shape[2:]
                    vec_accum[:, zd:zd + pd, yd:yd + ph, xd:xd + pw] += vec_np[i]
                    vec_count[:, zd:zd + pd, yd:yd + ph, xd:xd + pw] += 1

                    order_map[z:z + ps, y:y + ps, x:x + ps] = step_n

                if progress_cb and (b % self.log_every == 0 or b == n_batches - 1):
                    progress_cb(f"{volume_path.name}: batch {b + 1}/{n_batches}", (b + 1) / n_batches)

        refined_mask = mask_accum / (mask_count + 1e-6)
        vec_final = vec_accum / (vec_count + 1e-6)
        elapsed = time.time() - t0

        stem = volume_path.stem
        mask_path = out_dir / f"{stem}_radicle_mask.npy"
        vec_path = out_dir / f"{stem}_radicle_vector.npy"
        order_path = out_dir / f"{stem}_radicle_order.npy"
        np.save(str(mask_path), refined_mask)
        np.save(str(vec_path), vec_final)
        np.save(str(order_path), order_map)

        return RunResult(
            outputs=[mask_path, vec_path, order_path],
            device=device,
            elapsed_s=elapsed,
            extra={
                "patch_size": ps,
                "batch_size": self.batch_size,
                "strategy": "sequential",
                "first_checkpoint": str(self.first_ckpt),
                "second_checkpoint": str(self.second_ckpt),
            },
        )
