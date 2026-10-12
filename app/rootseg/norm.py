"""Per-volume intensity normalization: clip to [p1, p99] of a sample of slices, rescale to [0, 1].
Stats are computed once per volume and cached to JSON so training doesn't rescan the whole volume
every epoch.
"""
import json
import os

import numpy as np


def compute_volume_stats(volume, n_slabs=8, slab_depth=16, subsample=4, low_pct=1.0, high_pct=99.0):
    D = volume.shape[0]
    depth = min(slab_depth, D)
    samples = []
    for z in np.linspace(0, D - depth, n_slabs).astype(int):
        slab = np.array(volume[z:z + depth])  # materialize -- a view here turns this into scattered reads
        samples.append(slab[:, ::subsample, ::subsample].ravel())
    x = np.concatenate(samples).astype(np.float32)
    lo, hi = (float(v) for v in np.percentile(x, [low_pct, high_pct]))
    return {"p_low": lo, "p_high": hi}


def build_stats(volume_paths, out_path, n_slabs=8, slab_depth=16, subsample=4,
                 low_pct=1.0, high_pct=99.0):
    stats = {}
    for p in volume_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        vol = np.load(p, mmap_mode="r")
        stats[stem] = compute_volume_stats(vol, n_slabs, slab_depth, subsample, low_pct, high_pct)
        print(f"  [{stem}] p_low={stats[stem]['p_low']:.0f} p_high={stats[stem]['p_high']:.0f}", flush=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=1)
    return stats


def load_stats(path):
    with open(path) as f:
        return json.load(f)


def normalize_slice(slice_u8, lo, hi):
    x = slice_u8.astype(np.float32)
    return np.clip((x - lo) / max(hi - lo, 1.0), 0.0, 1.0)


class Normalizer:
    """uint8 array -> float32 [0, 1], per-volume p_low/p_high clip+rescale."""

    def __init__(self, stats_path):
        self.stats = load_stats(stats_path)

    def __call__(self, block, stem):
        if stem not in self.stats:
            raise KeyError(f"no normalization stats for volume {stem!r}; rerun norm.py")
        s = self.stats[stem]
        return normalize_slice(np.asarray(block), s["p_low"], s["p_high"])
