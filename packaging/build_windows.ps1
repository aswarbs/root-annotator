# Build the Windows bundle. Run from root-annotator/:
#   powershell -ExecutionPolicy Bypass -File packaging/build_windows.ps1
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path ".venv312")) {
    Write-Host "Creating build venv (.venv312)..."
    py -3.12 -m venv .venv312
}

$Python = ".venv312\Scripts\python.exe"
& $Python -m pip install --upgrade pip -q
& $Python -m pip install "torch==2.14.0+cu126" --index-url https://download.pytorch.org/whl/cu126
& $Python -m pip install -r requirements.txt -r packaging/requirements-build.txt

Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue
& $Python -m PyInstaller packaging\root_annotator.spec --noconfirm

Write-Host ""
Write-Host "Done. Copy the folder dist\root-annotator\ to the USB stick."
Write-Host "On the target machine: dist\root-annotator\root-annotator.exe"
