"""Renders one slice of a (possibly still-running) job: the CT input with the predicted root
probability overlaid in red. Opens the memmaps per call and closes them again, so no file handle
outlives the call (matters on Windows, where an open memmap blocks deleting the file).
"""
from pathlib import Path

import numpy as np

_MAX_SIDE = 1100


def render_live_frame(input_path, pred_path: Path, z: int, removed_from: Path = None) -> np.ndarray:
    vol = np.load(str(input_path), mmap_mode="r") if input_path else None  # None: scan folder, input gone
    pred = np.load(str(pred_path), mmap_mode="r")
    try:
        pr = np.array(pred[z], dtype=np.float32)
        img = np.array(vol[z]) if vol is not None else (np.clip(pr, 0, 1) * 255).astype(np.uint8)
        raw = np.array(np.load(str(removed_from), mmap_mode="r")[z], dtype=np.float32) if removed_from else None
    finally:
        del vol, pred
    lo, hi = np.percentile(img[::4, ::4], [1, 99])
    gray = np.clip((img.astype(np.float32) - lo) / max(float(hi - lo), 1.0), 0, 1) * 255
    rgb = np.stack([gray] * 3, axis=-1)
    a = np.clip(pr, 0, 1)[..., None] * 0.75
    rgb = rgb * (1 - a) + np.array([255, 40, 40], np.float32) * a
    if raw is not None:  # what a clean-up took out, so its effect is visible
        gone = ((raw >= 0.5) & (pr < 0.5))[..., None] * 0.8
        rgb = rgb * (1 - gone) + np.array([255, 170, 40], np.float32) * gone
    rgb = rgb.astype(np.uint8)
    step = int(np.ceil(max(rgb.shape[:2]) / _MAX_SIDE))
    return np.ascontiguousarray(rgb[::step, ::step]) if step > 1 else rgb
