"""Turns a mix of file/folder inputs into a flat, deduplicated list of volumes, and stacks
slice-folder scans (D:\\rootCT layout) into .npy volumes the engines can load.

Two input kinds are understood:
  * a ``.npy`` volume, shape (Z, H, W), uint8
  * a scan folder: a directory of 2D slice images (.tif/.tiff/.bmp, one file per z-slice, sorted
    by name). Either the slice folder itself, or its parent containing an ``*input*`` slice folder
    next to an ``*output*`` (ground-truth) folder, which is ignored.
"""
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from .tiffstack import TIFF_DIR_SUFFIXES

import numpy as np

SLICE_EXTS = {".tif", ".tiff", ".bmp"}
_MAX_DEPTH = 3
_STATS_SLICES = 64  # slices sampled for the 16-bit -> 8-bit percentile stretch
_P_LOW, _P_HIGH = 0.5, 99.5


_DERIVED_SUFFIXES = ("_mask", "_rootseg_pred", "_rootseg_pred_clean")  # ground truth / our own outputs


def _is_derived_npy(f: Path) -> bool:
    """Masks and earlier predictions sitting next to the scans are not volumes to segment."""
    return f.stem.lower().endswith(_DERIVED_SUFFIXES)


def _is_output_name(name: str) -> bool:
    n = name.lower()
    # the dataset has an "ouput" folder typo; our own TIFF-stack annotations are not scans either
    return "output" in n or "ouput" in n or n.endswith(TIFF_DIR_SUFFIXES)


def _is_input_name(name: str) -> bool:
    return "input" in name.lower()


def slice_files(folder: Path) -> List[Path]:
    """Sorted slice images directly in `folder`. If several formats are mixed, keeps only the most
    common one (some scan folders contain a few stray files of another type)."""
    files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in SLICE_EXTS]
    if not files:
        return []
    by_ext = {}
    for f in files:
        by_ext.setdefault(f.suffix.lower(), []).append(f)
    return sorted(max(by_ext.values(), key=len))


def _input_slice_dir(scan: Path) -> Optional[Path]:
    """The slice folder for a scan: `scan` itself if it holds images, else its *input* child."""
    if slice_files(scan):
        return scan
    for child in sorted(scan.iterdir()):
        if child.is_dir() and _is_input_name(child.name) and not _is_output_name(child.name):
            if slice_files(child):
                return child
    return None


def _find_scans(d: Path, depth: int, found: set) -> None:
    for f in sorted(d.glob("*.npy")):
        if not _is_derived_npy(f):
            found.add(f.resolve())
    if _input_slice_dir(d) is not None and not _is_output_name(d.name):
        # a slice folder named *input* stands for its parent scan folder
        found.add((d.parent if slice_files(d) and _is_input_name(d.name) else d).resolve())
        return
    if depth >= _MAX_DEPTH:
        return
    for child in sorted(d.iterdir()):
        if child.is_dir() and not _is_output_name(child.name):
            _find_scans(child, depth + 1, found)


def discover_volumes(inputs: Iterable) -> List[Path]:
    """inputs: file paths or directory paths (str or Path). Returns a sorted, deduplicated list of
    resolved Paths: .npy files, and scan folders (see module docstring). Directories are searched
    for .npy files and scan folders up to a few levels deep."""
    found = set()
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            _find_scans(p, 0, found)
        elif p.is_file() and p.suffix.lower() == ".npy":
            found.add(p.resolve())
        else:
            raise FileNotFoundError(f"not a .npy file or a folder: {p}")
    return sorted(found)


def _to_uint8(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if a.dtype == np.uint8:
        return a
    scaled = (a.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1.0))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _read_slice(path: Path) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        if im.mode not in ("L", "I;16", "I;16L", "I;16B", "I"):
            im = im.convert("L")
        return np.array(im)


def stack_scan_to_npy(
    scan_dir: Path, out_npy: Path, progress_cb: Optional[Callable[[str, Optional[float]], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> dict:
    """Stacks a scan folder's slices into a uint8 (Z, H, W) .npy at `out_npy`, slice by slice
    through a memmap (scans are several GB). 16-bit slices are stretched to 0-255 using the
    0.5-99.5 percentile range of the whole scan; 8-bit slices are copied as-is.
    Returns provenance info about the conversion."""
    src = _input_slice_dir(scan_dir)
    files = slice_files(src) if src else []
    if not files:
        raise ValueError(f"no .tif/.bmp slices found in {scan_dir}")

    first = _read_slice(files[0])
    H, W = first.shape
    is_16bit = first.dtype != np.uint8
    lo = hi = 0.0
    if is_16bit:
        idx = np.linspace(0, len(files) - 1, min(_STATS_SLICES, len(files))).astype(int)
        sample = np.concatenate([_read_slice(files[i]).ravel()[::17] for i in idx])
        lo, hi = (float(v) for v in np.percentile(sample, [_P_LOW, _P_HIGH]))

    out_npy.parent.mkdir(parents=True, exist_ok=True)
    vol = np.lib.format.open_memmap(str(out_npy), mode="w+", dtype=np.uint8, shape=(len(files), H, W))
    try:
        for z, f in enumerate(files):
            if cancel_cb and cancel_cb():
                raise RuntimeError("cancelled")
            a = first if z == 0 else _read_slice(f)
            if a.shape != (H, W):
                raise ValueError(f"{f.name} is {a.shape[1]}x{a.shape[0]}, expected {W}x{H}")
            vol[z] = _to_uint8(a, lo, hi)
            if progress_cb and (z % 100 == 0 or z == len(files) - 1):
                progress_cb(f"{scan_dir.name}: stacking slices {z + 1}/{len(files)}", None)
        vol.flush()
    except BaseException:
        del vol
        out_npy.unlink(missing_ok=True)
        raise
    del vol

    return {
        "source_slice_dir": str(src),
        "n_slices": len(files),
        "source_bit_depth": 16 if is_16bit else 8,
        "stretch_percentiles": [_P_LOW, _P_HIGH] if is_16bit else None,
        "stretch_range": [lo, hi] if is_16bit else None,
    }
