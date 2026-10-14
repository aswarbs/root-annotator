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
import subprocess
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

# The app's own .py files ship as plain data (not inside the compiled archive), so a newer copy
# downloaded by app/updater.py can take their place -- see main.py. The third-party packages they
# import (torch, PyQt6, numpy, scipy, PIL) are still discovered and bundled by the analysis below.
APP_DIR = PROJECT_ROOT / "app"
for f in APP_DIR.rglob("*.py"):
    datas.append((str(f), str(Path("app") / f.relative_to(APP_DIR).parent)))

# VERSION = commit this build was made from; the updater compares it with GitHub's latest commit.
try:
    _sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
except Exception:
    _sha = ""
_version_file = Path(SPECPATH) / "VERSION"  # noqa: F821
_version_file.write_text(_sha, encoding="utf-8")
datas.append((str(_version_file), "."))

a = Analysis(
    [str(PROJECT_ROOT / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    # main.py imports the app by name at run time, so list its entry modules here: the analysis then
    # follows their imports (torch, PyQt6, numpy, scipy, PIL ...). The app's own modules are removed
    # from the archive again below and shipped as plain files instead.
    hiddenimports=["PyQt6.sip", "app.gui", "app.cli", "app.web.server"],
    hookspath=[],
    runtime_hooks=[],
    excludes=["matplotlib", "skimage", "tensorboard", "plotly", "nibabel"],  # scipy is needed (clean-up)
    noarchive=False,
    cipher=block_cipher,
)
# Keep `app` out of the compiled archive so the copy on disk (bundled or downloaded) is what loads.
a.pure = [m for m in a.pure if not (m[0] == "app" or m[0].startswith("app."))]
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
