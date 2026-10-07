"""Writes a small JSON provenance record next to each prediction .npy: which model/checkpoint
produced it, on which device, how long it took. This is what turns a bare array into a
traceable annotation.
"""
import json
from pathlib import Path


def write_sidecar(output_path: Path, meta: dict) -> Path:
    sidecar_path = output_path.parent / f"{output_path.name}.json"
    with open(sidecar_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return sidecar_path
