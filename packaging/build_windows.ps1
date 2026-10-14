# Build the Windows bundle. Run from root-annotator/:
#   powershell -ExecutionPolicy Bypass -File packaging/build_windows.ps1
#
# The build needs roughly 10 GB of free disk space (CUDA PyTorch download + the ~4 GB bundle).
# If the project drive is short of space, put the output on another drive:
#   powershell -ExecutionPolicy Bypass -File packaging/build_windows.ps1 -OutDir D:\ra_build
param([string]$OutDir = "dist")
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

$Work = if ($OutDir -eq "dist") { "build" } else { Join-Path $OutDir "_work" }
Remove-Item -Recurse -Force $Work, $OutDir -ErrorAction SilentlyContinue
& $Python -m PyInstaller packaging\root_annotator.spec --noconfirm --distpath $OutDir --workpath $Work
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed (exit code $LASTEXITCODE)" }

Write-Host ""
Write-Host "Done. The bundle is in $OutDir\root-annotator\ (run root-annotator.exe inside it)."
Write-Host "To make the installer, compile packaging\root_annotator.iss with Inno Setup"
Write-Host "(its Source line points at dist\root-annotator; edit it if you used -OutDir)."
