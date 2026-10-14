#!/usr/bin/env bash
# Build the Linux bundle. PyInstaller cannot cross-compile, so this must run ON a Linux
# machine (or WSL) -- the Windows build script produces the Windows .exe separately.
#
# Run from root-annotator/:
#   bash packaging/build_linux.sh
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -d ".venv312" ]; then
    echo "Creating build venv (.venv312)..."
    python3.12 -m venv .venv312 || python3 -m venv .venv312
fi

PYTHON=".venv312/bin/python"
"$PYTHON" -m pip install --upgrade pip -q
"$PYTHON" -m pip install "torch==2.14.0+cu126" --index-url https://download.pytorch.org/whl/cu126
"$PYTHON" -m pip install -r requirements.txt -r packaging/requirements-build.txt

rm -rf build dist
"$PYTHON" -m PyInstaller packaging/root_annotator.spec --noconfirm

echo ""
echo "Done. Copy the folder dist/root-annotator/ to the USB stick."
echo "On the target machine: dist/root-annotator/root-annotator"
