# PyInstaller spec for Root Annotator.
#
# Build (from root-annotator/, with the project venv active):
#   pyinstaller packaging/root_annotator.spec --noconfirm
#
# Output lands in root-annotator/dist/root-annotator/ -- copy that whole folder to the USB
# stick. It bundles the radicle/ and rootseg-lstm/ repos (code + checkpoints) as data, so the
# target machine needs nothing checked out itself.
#
# PyInstaller cannot cross-compile: run this file on Windows to get the Windows build, and on
# Linux (or in WSL) to get the Linux build -- see build_windows.ps1 / build_linux.sh.
import sys
from pathlib import Path

block_cipher = None

SPEC_DIR = Path(SPECPATH)  # noqa: F821 -- injected by PyInstaller
PROJECT_ROOT = SPEC_DIR.parent          # root-annotator/
PHD_ROOT = PROJECT_ROOT.parent          # phd/ -- contains radicle/ and rootseg-lstm/

RADICLE_DIR = PHD_ROOT / "radicle"
ROOTSEG_DIR = PHD_ROOT / "rootseg-lstm"


def repo_datas(src: Path, dest_name: str):
    """(source, dest) pairs for every file in src, laid out under dest_name/ in the bundle,
    skipping .git and other junk we don't need at inference time."""
    skip_dirs = {".git", "media", "__pycache__", "runs", ".pytest_cache"}
    skip_suffixes = {".png", ".gif", ".ipynb"}
    pairs = []
    for f in src.rglob("*"):
        if f.is_dir():
            continue
        if any(part in skip_dirs for part in f.parts):
            continue
        if f.suffix.lower() in skip_suffixes:
            continue
        rel = f.relative_to(src)
        pairs.append((str(f), str(Path(dest_name) / rel.parent)))
    return pairs


datas = repo_datas(RADICLE_DIR, "radicle") + repo_datas(ROOTSEG_DIR, "rootseg-lstm")

a = Analysis(
    [str(PROJECT_ROOT / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=["PyQt6.sip"],
    hookspath=[],
    runtime_hooks=[],
    excludes=["matplotlib", "scipy", "skimage", "tensorboard", "plotly", "nibabel"],
    noarchive=False,
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="root-annotator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # no console flash on double-click; --cli still works when run from a terminal
    disable_windowed_traceback=False,
    icon=str(SPEC_DIR / "icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="root-annotator",
)
