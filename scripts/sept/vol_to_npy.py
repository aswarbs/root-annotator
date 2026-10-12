"""Converts a Phoenix datos|x .vol (raw voxels, z slowest) + its .pcr into a uint8 (Z, Y, X) .npy,
stretching intensities to 0-255 with the 0.5-99.5 percentile range, same as the app does for
16-bit slice stacks.

    python vol_to_npy.py <file.vol> <out.npy>
"""
import sys
from pathlib import Path

import numpy as np


def read_pcr(path: Path) -> dict:
    out = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def main(vol_path: Path, out_path: Path) -> None:
    pcr = read_pcr(vol_path.with_suffix(".pcr"))
    X, Y, Z = (int(pcr[f"Volume_Size{a}"]) for a in "XYZ")
    n = X * Y * Z
    size = vol_path.stat().st_size
    dtype = {1: np.uint8, 2: np.uint16, 4: np.float32}.get(size // n)
    if dtype is None or size % n:
        raise SystemExit(f"{vol_path.name}: {size} bytes does not fit {X}x{Y}x{Z} voxels")
    print(f"{vol_path.name}: shape (Z,Y,X)=({Z},{Y},{X}) dtype={np.dtype(dtype).name}", flush=True)
    src = np.memmap(vol_path, dtype=dtype, mode="r", shape=(Z, Y, X))

    idx = np.linspace(0, Z - 1, 64).astype(int)
    sample = np.concatenate([np.asarray(src[i]).ravel()[::17] for i in idx]).astype(np.float32)
    lo, hi = (float(v) for v in np.percentile(sample, [0.5, 99.5]))
    print(f"stretch range {lo:.4g} .. {hi:.4g}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".partial")
    out = np.lib.format.open_memmap(str(tmp), mode="w+", dtype=np.uint8, shape=(Z, Y, X))
    for z0 in range(0, Z, 32):
        chunk = np.asarray(src[z0:z0 + 32], dtype=np.float32)
        out[z0:z0 + 32] = (np.clip((chunk - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)
        if (z0 // 32) % 20 == 0:
            print(f"  z={z0}/{Z}", flush=True)
    out.flush()
    del out
    tmp.replace(out_path)
    print("done", out_path, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
