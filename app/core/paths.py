"""Resolves where the radicle/ and rootseg-lstm/ model repos live, in both dev and frozen
(PyInstaller) modes, and makes a repo importable via sys.path.
"""
import sys
from pathlib import Path


def app_base_dir() -> Path:
    """Directory that contains the radicle/ and rootseg-lstm/ folders."""
    if getattr(sys, "frozen", False):
        # PyInstaller bundle: packaging/root_annotator.spec adds both repos as top-level datas.
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    # Dev mode: this file is root-annotator/app/core/paths.py; repos are siblings of root-annotator/.
    return Path(__file__).resolve().parents[3]


def repo_dir(name: str) -> Path:
    path = app_base_dir() / name
    if not path.is_dir():
        raise FileNotFoundError(
            f"expected model repo {name!r} at {path} -- is it checked out next to root-annotator/?"
        )
    return path


def ensure_repo_on_path(repo_path: Path) -> None:
    p = str(repo_path)
    if p not in sys.path:
        sys.path.insert(0, p)
