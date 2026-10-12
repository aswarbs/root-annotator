# PyInstaller spec for Root Annotator.
#
# Build (from root-annotator/, with the project venv active):
#   pyinstaller packaging/root_annotator.spec --noconfirm
#
# Output lands in root-annotator/dist/root-annotator/ -- copy that whole folder to the USB
# stick. It bundles the model checkpoint and the web UI pages, so the target machine needs
# nothing checked out itself.
#
# PyInstaller cannot cross-compile: run this file on Windows to get the Windows build, and on
# Linux (or in WSL) to get the Linux build -- see build_windows.ps1 / build_linux.sh.
import sys
from pathlib import Path

block_cipher = None

SPEC_DIR = Path(SPECPATH)  # noqa: F821 -- injected by PyInstaller
PROJECT_ROOT = SPEC_DIR.parent          # root-annotator/

# The model code and checkpoint live inside the app package, so only the non-.py files need
# adding as data (same relative layout as a source checkout, which is what paths.py relies on).
datas = [
    (str(PROJECT_ROOT / "app" / "rootseg" / "checkpoints" / "e4_skips_bal.pt"), "app/rootseg/checkpoints"),
    (str(PROJECT_ROOT / "app" / "web" / "static"), "app/web/static"),
]

a = Analysis(
    [str(PROJECT_ROOT / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=["PyQt6.sip"],
    hookspath=[],
    runtime_hooks=[],
    excludes=["matplotlib", "skimage", "tensorboard", "plotly", "nibabel"]  # scipy is needed (clean-up),
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
