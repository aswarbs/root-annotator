"""Writes a prediction as a TIFF stack: a folder with one 8-bit TIFF per z-slice (0 = background,
255 = root), named slice_00000.tif, slice_00001.tif, ... -- the layout other tools expect for
annotations. The .npy the app works with is kept next to it; the stack is derived from it.
"""
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from .engine_base import CancelCB, ProgressCB

TIFF_DIR_SUFFIXES = ("_rootseg_pred_tiff", "_rootseg_pred_clean_tiff")  # see volumes.py: never treated as scans
TIFF_THRESHOLD = 0.5


def tiff_dir_for(npy_path: Path) -> Path:
    """<name>.npy -> <name>_tiff (next to it)."""
    npy_path = Path(npy_path)
    return npy_path.with_name(npy_path.stem + "_tiff")


def write_tiff_stack(npy_path: Path, out_dir: Optional[Path] = None, threshold: float = TIFF_THRESHOLD,
                     progress_cb: ProgressCB = None, cancel_cb: CancelCB = None) -> Path:
    """Thresholds a probability volume (or a 0/1 mask) and writes it as one TIFF per slice.
    Written into a temporary folder and renamed at the end, so a cancelled run leaves no half stack."""
    from PIL import Image

    npy_path = Path(npy_path)
    out_dir = Path(out_dir) if out_dir else tiff_dir_for(npy_path)
    tmp = out_dir.with_name(out_dir.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    vol = np.load(str(npy_path), mmap_mode="r")
    D = vol.shape[0]
    try:
        for z in range(D):
            if cancel_cb and cancel_cb():
                raise RuntimeError("cancelled")
            mask = (np.asarray(vol[z], dtype=np.float32) >= threshold).astype(np.uint8) * 255
            Image.fromarray(mask).save(tmp / f"slice_{z:05d}.tif", compression="tiff_lzw")
            if progress_cb and ((z + 1) % 100 == 0 or z == D - 1):
                progress_cb(f"saving TIFF stack: {z + 1}/{D}", (z + 1) / D)
        del vol
        shutil.rmtree(out_dir, ignore_errors=True)
        tmp.replace(out_dir)
    except BaseException:
        del vol
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return out_dir
