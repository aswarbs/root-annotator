"""Noise clean-up for a prediction: threshold the probabilities, then drop 3D connected
components smaller than `min_size` voxels (26-connectivity).

Volumes are ~2e9 voxels, so labelling is done slab by slab along Z with a union-find across slab
boundaries instead of one scipy.ndimage.label on the whole thing (int32 labels would be ~9 GB).
Peak memory is a couple of slabs. Pass 1 labels each slab and counts component sizes, pass 2
re-labels each slab and writes only the components that survived.
"""
import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from scipy import ndimage

from .engine_base import CancelCB, ProgressCB
from .sidecar import write_sidecar
from .tiffstack import write_tiff_stack

CLEAN_SUFFIX = "_clean.npy"
CLEAN_THRESHOLD = 0.5
CLEAN_MIN_SIZE = 1000  # voxels; pieces smaller than this are dropped (was 100 -- raised to clean harder)
_SLAB_VOXELS = 48_000_000
_STRUCT = np.ones((3, 3, 3), dtype=bool)  # 26-connectivity


def clean_path_for(pred_path: Path) -> Path:
    stem = pred_path.name[:-4] if pred_path.name.endswith(".npy") else pred_path.name
    return pred_path.with_name(stem + CLEAN_SUFFIX)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)

    def find(self, a: int) -> int:
        p = self.parent
        root = a
        while p[root] != root:
            root = p[root]
        while p[a] != root:
            p[a], a = root, p[a]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _slab_ranges(D: int, H: int, W: int):
    depth = max(1, _SLAB_VOXELS // (H * W))
    return [(z0, min(z0 + depth, D)) for z0 in range(0, D, depth)]


def _label_slab(pred, z0, z1, threshold):
    mask = np.asarray(pred[z0:z1]) >= threshold
    labels, n = ndimage.label(mask, structure=_STRUCT)
    return mask, labels.astype(np.int32, copy=False), n


def _boundary_pairs(prev_plane: np.ndarray, next_plane: np.ndarray):
    """(a, b) label pairs touching across a slab boundary, within the 3x3 in-plane neighbourhood."""
    H, W = prev_plane.shape
    pairs = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            a = prev_plane[max(0, dy):H + min(0, dy), max(0, dx):W + min(0, dx)]
            b = next_plane[max(0, -dy):H + min(0, -dy), max(0, -dx):W + min(0, -dx)]
            both = (a > 0) & (b > 0)
            if both.any():
                pairs.append(np.unique(np.stack([a[both], b[both]], 1), axis=0))
    return np.unique(np.concatenate(pairs), axis=0) if pairs else np.empty((0, 2), dtype=np.int64)


def clean_prediction(pred_path: Path, out_path: Path = None, threshold: float = 0.5, min_size: int = 100,
                     progress_cb: ProgressCB = None, cancel_cb: CancelCB = None, write_tiff: bool = True,
                     written_cb: Optional[Callable[[int], None]] = None, live_proj: Optional[dict] = None) -> dict:
    """Writes a uint8 0/1 mask to `out_path` (default: <pred>_clean.npy next to the input) and a
    provenance sidecar. The raw prediction is never modified. Returns summary statistics."""
    pred_path = Path(pred_path)
    out_path = Path(out_path) if out_path else clean_path_for(pred_path)
    pred = np.load(str(pred_path), mmap_mode="r")
    D, H, W = pred.shape
    slabs = _slab_ranges(D, H, W)
    # footprints of the raw mask along each axis, so a viewer can show the clean-up as it happens
    raw1 = np.zeros((D, W), dtype=bool)
    raw2 = np.zeros((D, H), dtype=bool)
    raw0_slab = []

    def check_cancel():
        if cancel_cb and cancel_cb():
            raise RuntimeError("cancelled")

    # pass 1: per-slab labels -> sizes, plus cross-boundary merges
    offsets, sizes, merges = [], [], []
    total = 0
    prev_last = None
    voxels_before = 0
    for k, (z0, z1) in enumerate(slabs):
        check_cancel()
        mask, labels, n = _label_slab(pred, z0, z1, threshold)
        voxels_before += int(mask.sum())
        raw0_slab.append(np.packbits(mask.any(axis=0).ravel()))
        raw1[z0:z1], raw2[z0:z1] = mask.any(axis=1), mask.any(axis=2)
        offsets.append(total)
        sizes.append(np.bincount(labels.ravel(), minlength=n + 1)[1:].astype(np.int64))
        if prev_last is not None and n:
            first = np.where(labels[0] > 0, labels[0] + total, 0)
            merges.append(_boundary_pairs(prev_last, first))
        prev_last = np.where(labels[-1] > 0, labels[-1] + total, 0) if n else np.zeros((H, W), np.int64)
        total += n
        if progress_cb:
            progress_cb(f"clean-up: labelling slab {k + 1}/{len(slabs)}", 0.45 * (k + 1) / len(slabs))
        del mask, labels

    uf = _UnionFind(total + 1)  # global ids are 1-based
    for pairs in merges:
        for a, b in pairs:
            uf.union(int(a), int(b))
    all_sizes = np.concatenate(sizes) if sizes else np.zeros(0, np.int64)  # index = global id - 1
    roots = np.fromiter((uf.find(i) for i in range(1, total + 1)), dtype=np.int64, count=total)
    root_size = np.zeros(total + 1, dtype=np.int64)
    np.add.at(root_size, roots, all_sizes)
    keep_root = root_size >= min_size
    n_components = int(len({int(r) for r in roots})) if total else 0
    n_kept = int(keep_root.sum())
    keep_label = np.zeros(total + 1, dtype=bool)  # global id -> survives
    keep_label[1:] = keep_root[roots]

    # live projections: white = a kept voxel (or, for slabs not cleaned yet, a raw one) lies on the ray,
    # orange = the raw mask had root there and the clean-up removed all of it
    suffix = [np.zeros_like(raw0_slab[0])] if raw0_slab else []   # suffix[k] = raw footprint of slabs >= k
    for packed in reversed(raw0_slab):
        suffix.append(suffix[-1] | packed)
    suffix.reverse()  # len(slabs) + 1 entries, last is empty
    kept0 = np.zeros((H, W), dtype=bool)
    if live_proj is not None and raw0_slab:
        raw_all0 = np.unpackbits(suffix[0])[:H * W].reshape(H, W).astype(bool)
        live = {"w0": raw_all0.copy(), "o0": np.zeros((H, W), bool), "w1": raw1.copy(), "o1": np.zeros((D, W), bool),
                "w2": raw2.copy(), "o2": np.zeros((D, H), bool)}
        live_proj.update(live)
        live_proj["ready"] = True

    # pass 2: re-label (deterministic) and write the surviving components
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".partial")
    out = np.lib.format.open_memmap(str(tmp), mode="w+", dtype=np.uint8, shape=(D, H, W))
    voxels_after = 0
    try:
        for k, (z0, z1) in enumerate(slabs):
            check_cancel()
            _, labels, n = _label_slab(pred, z0, z1, threshold)
            lut = np.zeros(n + 1, dtype=bool)
            lut[1:] = keep_label[offsets[k] + 1:offsets[k] + n + 1]
            kept = lut[labels]
            out[z0:z1] = kept
            voxels_after += int(kept.sum())
            if live_proj is not None and live_proj.get("ready"):
                kept0 |= kept.any(axis=0)
                w0 = kept0 | np.unpackbits(suffix[k + 1])[:H * W].reshape(H, W).astype(bool)
                live_proj["w0"][:], live_proj["o0"][:] = w0, raw_all0 & ~w0
                for ax, raw in ((1, raw1), (2, raw2)):
                    k_rows = kept.any(axis=ax)
                    live_proj[f"w{ax}"][z0:z1] = k_rows
                    live_proj[f"o{ax}"][z0:z1] = raw[z0:z1] & ~k_rows
            if written_cb:
                out.flush()
                written_cb(z1)  # slices below z1 of the .partial file are final
            if progress_cb:
                progress_cb(f"clean-up: writing slab {k + 1}/{len(slabs)}", 0.45 + 0.55 * (k + 1) / len(slabs))
        out.flush()
        del out
        tmp.replace(out_path)
    except BaseException:
        del out
        tmp.unlink(missing_ok=True)
        raise

    stats = {
        "source": str(pred_path), "threshold": threshold, "min_size": min_size,
        "components_total": n_components, "components_kept": n_kept,
        "voxels_before": voxels_before, "voxels_after": voxels_after,
        "connectivity": 26,
    }
    tiff_dir = None
    if write_tiff:  # the web UI defers this until the user keeps the result
        tiff_dir = write_tiff_stack(out_path, progress_cb=lambda msg, frac=None: progress_cb and progress_cb(msg, None),
                                    cancel_cb=cancel_cb)
    write_sidecar(out_path, {"kind": "cleaned mask (uint8, 0/1)", "tiff_stack": str(tiff_dir) if tiff_dir else None, **stats})
    return {**stats, "output": str(out_path), "tiff_stack": str(tiff_dir) if tiff_dir else None}
