"""2D projections of a prediction for the visualiser.

One streaming pass over the volume (memmap, a slab of slices at a time) records, for each of the
three axes, the maximum value along the ray and the depth at which it occurs, quantised to uint8.
Those six small 2D arrays are cached next to the prediction, so reopening a result is instant and
moving the threshold slider or switching to depth colouring never touches the volume again.
Works on both raw probability files (float16) and cleaned masks (uint8 0/1).
"""
from pathlib import Path
from typing import Optional

import numpy as np

from .engine_base import CancelCB, ProgressCB

AXES = {
    "side": (1, "Side view (max along Y)"),
    "front": (2, "Front view (max along X)"),
    "top": (0, "Top-down view (max along Z)"),
}
_CACHE_VERSION = 1
_SLAB_VOXELS = 64_000_000

# viridis control points, interpolated into a 256-entry LUT (avoids a matplotlib dependency)
_VIRIDIS = np.array([
    [68, 1, 84], [71, 44, 122], [59, 81, 139], [44, 113, 142], [33, 144, 141],
    [39, 173, 129], [92, 200, 99], [170, 220, 50], [253, 231, 37]], dtype=np.float32)
_LUT = np.stack([np.interp(np.linspace(0, 1, 256), np.linspace(0, 1, len(_VIRIDIS)), _VIRIDIS[:, c])
                 for c in range(3)], axis=1).astype(np.uint8)


def _cache_path(vol_path: Path) -> Path:
    return vol_path.with_name(vol_path.name + ".proj.npz")


def _fingerprint(vol_path: Path) -> list:
    st = vol_path.stat()
    return [_CACHE_VERSION, st.st_size, int(st.st_mtime)]


def _to_u8(chunk: np.ndarray) -> np.ndarray:
    if chunk.dtype == np.uint8:
        return chunk * 255  # uint8 inputs are 0/1 cleaned masks
    return np.rint(np.asarray(chunk, dtype=np.float32) * 255).clip(0, 255).astype(np.uint8)


class Projections:
    """max_/arg_ arrays per axis (0=Z, 1=Y, 2=X). arg_ is the index along the projected axis."""

    def __init__(self, shape, arrays):
        self.shape = tuple(shape)
        self.arrays = arrays

    @classmethod
    def compute(cls, vol_path: Path, progress_cb: ProgressCB = None, cancel_cb: CancelCB = None,
                use_cache: bool = True) -> "Projections":
        vol_path = Path(vol_path)
        cache = _cache_path(vol_path)
        fp = _fingerprint(vol_path)
        if use_cache and cache.is_file():
            try:
                z = np.load(str(cache))
                if z["fingerprint"].tolist() == fp:
                    return cls(z["shape"], {k: z[k] for k in z.files if k.startswith(("max", "arg"))})
            except Exception:
                pass

        vol = np.load(str(vol_path), mmap_mode="r")
        D, H, W = vol.shape
        max0 = np.zeros((H, W), np.uint8)
        arg0 = np.zeros((H, W), np.uint16)
        max1 = np.zeros((D, W), np.uint8)
        arg1 = np.zeros((D, W), np.uint16)
        max2 = np.zeros((D, H), np.uint8)
        arg2 = np.zeros((D, H), np.uint16)
        depth = max(1, _SLAB_VOXELS // (H * W))
        for z0 in range(0, D, depth):
            if cancel_cb and cancel_cb():
                raise RuntimeError("cancelled")
            z1 = min(z0 + depth, D)
            chunk = _to_u8(np.asarray(vol[z0:z1]))
            cm, ca = chunk.max(axis=0), chunk.argmax(axis=0)
            better = cm > max0
            max0[better], arg0[better] = cm[better], ca[better] + z0
            max1[z0:z1], arg1[z0:z1] = chunk.max(axis=1), chunk.argmax(axis=1)
            max2[z0:z1], arg2[z0:z1] = chunk.max(axis=2), chunk.argmax(axis=2)
            if progress_cb:
                progress_cb(f"projecting {vol_path.name}: z={z1}/{D}", z1 / D)
        arrays = {"max0": max0, "arg0": arg0, "max1": max1, "arg1": arg1, "max2": max2, "arg2": arg2}
        proj = cls((D, H, W), arrays)
        try:
            np.savez(str(cache), fingerprint=np.array(fp), shape=np.array([D, H, W]), **arrays)
        except OSError:
            pass  # read-only location: just recompute next time
        return proj

    def render(self, axis: str = "side", threshold: float = 0.5, depth_color: bool = False,
               max_side: Optional[int] = None) -> np.ndarray:
        """RGB uint8 image. Pixels whose max probability is below `threshold` are dark; the rest
        are white, or coloured by the depth of the maximum when `depth_color` is set."""
        ax = AXES[axis][0]
        m, a = self.arrays[f"max{ax}"], self.arrays[f"arg{ax}"]
        hit = m >= int(round(threshold * 255))
        if depth_color:
            denom = max(self.shape[ax] - 1, 1)
            img = _LUT[(a.astype(np.float32) / denom * 255).astype(np.uint8)]
        else:
            img = np.stack([m] * 3, axis=-1) if threshold <= 0 else np.full(m.shape + (3,), 255, np.uint8)
        img = np.where(hit[..., None], img, np.uint8(18))
        if max_side and max(img.shape[:2]) > max_side:
            step = int(np.ceil(max(img.shape[:2]) / max_side))
            img = img[::step, ::step]
        return np.ascontiguousarray(img)
