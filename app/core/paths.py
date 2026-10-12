"""Locates the bundled model checkpoint, in both dev and frozen (PyInstaller) modes. The
checkpoint ships inside the app package (app/rootseg/checkpoints/), so nothing else needs to be
checked out next to it.
"""
from pathlib import Path

CHECKPOINT_NAME = "e4_skips_bal.pt"


def checkpoint_path() -> Path:
    # Same relative layout in a source checkout and in the PyInstaller bundle (see the spec).
    path = Path(__file__).resolve().parents[1] / "rootseg" / "checkpoints" / CHECKPOINT_NAME
    if not path.is_file():
        raise FileNotFoundError(f"model checkpoint missing: {path}")
    return path
