"""Turns one Sept CT volume + its Radicle output into a RootSeg-LSTM training sample:

    <train>/<group>/data/<stem>.npy    symlink to the uint8 volume
    <train>/<group>/masks/<stem>.npy   uint8 0/1: Radicle mask >= 0.5, pieces < 1000 voxels dropped
    <train>/stats/<stem>.json          per-volume p_low / p_high (as norm.py computes them)
    <train>/root_index/<stem>.npz      where the root is (as build_root_index.py computes it)

    python prepare_training.py <task_index>      # index into GROUPS below (0-9)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path.home() / "rootseg-lstm"))
from app.core.postprocess import clean_prediction  # noqa: E402
from norm import compute_volume_stats  # noqa: E402

ROOT = Path("/data/psyas17/SeptCTData")
TRAIN = ROOT / "train"
GROUPS = [  # (group, stem)
    ("rice", "DT01943_DokHein-3__Merged_Scans_1-3"),
    ("rice", "DT01945_PhanPhae-4__Merged_Scans_1-3"),
    ("rice", "DT01946_DokHein-2__Merged_Scans_1-3"),
    ("rice", "M16751_Amouk-5__Merged_Scans_1-2"),
    ("rice", "M16752_Anbaw-1__Merged_Scans_1-2"),
    ("rice", "M16753_Anbaw-3__Merged_Scans_1-2"),
    ("rice", "M16754_Anbaw-5__Merged_Scans_1-2"),
    ("rice", "M16755_Anbaw-6__Merged_Scans_1-2"),
    ("tomato_timeseries", "M16826_X20-06_R1__Merged_Scans_1-2"),
    ("tomato_timeseries", "M16838_X20-06_R1_D12_Merged_Scans_1-3"),
]
MIN_SIZE = 1000


def root_index(mask_path: Path, out_path: Path, z_stride=4, cell=32):
    m = np.load(mask_path, mmap_mode="r")
    D, H, W = m.shape
    ny, nx = H // cell, W // cell
    zs = np.arange(0, D, z_stride)
    counts = np.zeros((len(zs), ny, nx), dtype=np.uint16)
    for i, z in enumerate(zs):
        counts[i] = np.array(m[z, :ny * cell, :nx * cell]).reshape(ny, cell, nx, cell).sum(axis=(1, 3))
    np.savez(out_path, counts=counts, shape=np.array((D, H, W)), z_stride=z_stride, cell=cell)
    return int((counts > 0).sum()), counts.size


def main(i: int) -> None:
    group, stem = GROUPS[i]
    data_dir, mask_dir = TRAIN / group / "data", TRAIN / group / "masks"
    data_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    vol = ROOT / "volumes" / f"{stem}.npy"
    link = data_dir / f"{stem}.npy"
    if not link.exists():
        os.symlink(vol, link)
    out_mask = mask_dir / f"{stem}.npy"
    if not out_mask.exists():
        src = ROOT / "results" / "radicle" / f"{stem}_radicle_mask.npy"
        r = clean_prediction(src, out_path=out_mask, threshold=0.5, min_size=MIN_SIZE, write_tiff=False,
                             progress_cb=lambda m, f=None: print(" ", m, flush=True) if f is None or int(f * 100) % 25 == 0 else None)
        print(f"{stem}: pieces {r['components_kept']}/{r['components_total']} kept, "
              f"voxels {r['voxels_after']:,}/{r['voxels_before']:,}", flush=True)
        for extra in (out_mask.with_name(out_mask.name + ".json"),):
            extra.unlink(missing_ok=True)
    stats = compute_volume_stats(np.load(vol, mmap_mode="r"))
    (TRAIN / "stats" / f"{stem}.json").write_text(json.dumps(stats))
    n, total = root_index(out_mask, TRAIN / "root_index" / f"{stem}.npz")
    print(f"{stem}: group={group} stats={stats} root cells {n}/{total}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
