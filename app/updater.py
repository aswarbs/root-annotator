"""Over-the-air code updates for the packaged (PyInstaller) app.

The installed app bundles Python, PyTorch and Qt (the big, rarely-changing part) plus a copy of the
`app/` code. This module downloads a newer copy of just `app/` from GitHub into
`%LOCALAPPDATA%\\RootAnnotator\\code_next`. The next launch (see main.py) promotes it to `code` and
imports it in preference to the bundled copy, so a fix reaches the machine as a ~2 MB download and
needs no reinstall. A new third-party dependency still needs a rebuilt installer.

Private repos: put a read-only fine-grained token (Contents: read) in
`%LOCALAPPDATA%\\RootAnnotator\\update.token` once. Optional `update.json` overrides repo / branch.
"""
import io
import json
import os
import shutil
import sys
import threading
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Optional

DEFAULT_REPO = "aswarbs/root-annotator"
DEFAULT_BRANCH = "main"
_REQUIRED = ("app/__init__.py", "app/gui.py", "app/core/session.py")


def data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
    return Path(base) / "RootAnnotator"


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def current_version() -> str:
    """Full commit hash of the running code ('' for a source checkout without a VERSION file)."""
    return _read(Path(__file__).resolve().parents[1] / "VERSION")


def _config() -> dict:
    cfg = {"repo": DEFAULT_REPO, "branch": DEFAULT_BRANCH}
    try:
        cfg.update(json.loads(_read(data_dir() / "update.json") or "{}"))
    except ValueError:
        pass
    return cfg


def _open(url: str, token: str, accept: str, timeout: float):
    headers = {"User-Agent": "root-annotator-updater", "Accept": accept}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout)


def check_for_update(timeout: float = 20.0) -> Optional[dict]:
    """Downloads and stages the latest code if it differs from what is running.
    Returns {"version": sha} when a new version is staged, None when up to date, offline, or not
    a packaged build. Never raises."""
    if not getattr(sys, "frozen", False):
        return None
    try:
        cfg, root = _config(), data_dir()
        token = os.environ.get("ROOT_ANNOTATOR_TOKEN") or _read(root / "update.token")
        api = f"https://api.github.com/repos/{cfg['repo']}"
        with _open(f"{api}/commits/{cfg['branch']}", token, "application/vnd.github.sha", timeout) as r:
            sha = r.read().decode().strip()
        if len(sha) != 40:
            return None
        if sha in (current_version(), _read(root / "code_broken" / "VERSION")):
            return None
        if _read(root / "code_next" / "VERSION") == sha:
            return {"version": sha}

        with _open(f"{api}/zipball/{sha}", token, "application/vnd.github+json", 120) as r:
            blob = r.read()
        tmp = root / "code_next.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            for info in z.infolist():
                parts = info.filename.split("/", 1)  # "<owner>-<repo>-<sha>/app/..."
                if len(parts) < 2 or not parts[1].startswith("app/") or info.is_dir():
                    continue
                rel = Path(parts[1])
                if ".." in rel.parts:
                    continue
                dest = tmp / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(z.read(info))
        if not all((tmp / p).is_file() for p in _REQUIRED):
            shutil.rmtree(tmp, ignore_errors=True)
            return None
        (tmp / "VERSION").write_text(sha, encoding="utf-8")
        shutil.rmtree(root / "code_next", ignore_errors=True)
        tmp.replace(root / "code_next")
        return {"version": sha}
    except Exception:
        return None


def check_in_background(on_staged: Callable[[dict], None]) -> None:
    def work():
        result = check_for_update()
        if result:
            on_staged(result)

    threading.Thread(target=work, daemon=True, name="updater").start()
