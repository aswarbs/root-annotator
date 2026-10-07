# Root Annotator

One lightweight app for running either **Radicle** or **RootSeg-LSTM** over root CT volumes on
a machine that has the data but not this dev setup (e.g. run from a USB stick). Pick the model
from a dropdown, add volumes, run, get predictions + a provenance JSON per output.

A GUI (PyQt6) is the default; a terminal client is the fallback for a headless target.

## Layout

```
root-annotator/
  app/
    core/
      paths.py       resolves the radicle/ and rootseg-lstm/ repos (dev checkout or bundled)
      volumes.py      turns file/folder picks into a flat list of .npy volumes
      engine_base.py   common Engine interface
      engine_radicle.py   wraps radicle's sequential-traversal cascade inference
      engine_rootseg.py   wraps rootseg-lstm's ConvLSTM predict.py
      registry.py      the one place both models are registered
      jobs.py          runs a batch of volumes through one engine, reports progress
      sidecar.py       writes the <output>.npy.json provenance record
    gui.py            PyQt6 window: model dropdown, drag-and-drop queue, run/cancel, log
    cli.py            terminal client, same core underneath
  main.py             entry point: GUI by default, falls back to the CLI if headless
  packaging/          PyInstaller spec + build scripts (see Packaging below)
```

The app imports the model code straight from the `radicle/` and `rootseg-lstm/` checkouts next
to this folder -- it does not fork or duplicate their model/inference code, so a checkpoint or
bugfix update in either repo is picked up automatically.

## Running from source (dev machine)

```powershell
.venv312\Scripts\python.exe main.py                 # GUI
.venv312\Scripts\python.exe main.py --cli --model radicle --input path\to\volumes --out-dir out\
```

## What each model needs

- **Radicle**: runs the *sequential* traversal strategy only (visits every patch, no seed
  coordinates needed). Radicle's BFS and vector-guided strategies need root-crown seed
  coordinates per volume and are intentionally left out of this app for now -- use radicle's own
  `inference/infer_bfs.py` / `infer_traverse.py` directly if you need those.
- **RootSeg-LSTM**: single ConvLSTM pass per volume, no configuration needed beyond the
  checkpoint (bundled).

Both expect a `.npy` volume, shape `(Z, H, W)`, dtype uint8 -- same format the underlying repos'
own scripts use.

The app also accepts **slice-folder scans** in the `D:\rootCT` layout: a folder of 2D `.tif`/`.bmp`
slices (one per z, sorted by name), either the slice folder itself or its parent containing
`input/` next to `output/`. Point it at a single scan, or at any folder above (e.g. `D:\rootCT`)
and it finds every scan underneath (up to 3 levels). `output`/`ouput` folders (ground truth) are
ignored. Each scan is stacked to a temporary uint8 `.npy` in `<out-dir>/_stacked/` (deleted after
the run); 16-bit slices are stretched to 0-255 using the scan's 0.5-99.5 percentile range, 8-bit
slices are copied as-is. Stacking takes ~2 minutes per scan, and Radicle needs RAM for the whole
volume (several GB per scan). The conversion settings are recorded in the output sidecar JSON.

## Output

For each input volume, in the chosen output folder:

- Radicle: `<stem>_radicle_mask.npy`, `<stem>_radicle_vector.npy`, `<stem>_radicle_order.npy`
- RootSeg-LSTM: `<stem>_rootseg_pred.npy` (per-voxel root probability, `[0, 1]`)

Every output file gets a `<file>.npy.json` sidecar recording which model/checkpoint produced
it, which device ran it, and how long it took -- the provenance record for that annotation.

## Packaging for the USB stick

PyInstaller bundles Python + PyTorch + PyQt6 + both model repos (code and checkpoints) into a
single folder, so the target machine needs nothing pre-installed. It **cannot cross-compile**:
build the Windows version on Windows, the Linux version on Linux (or WSL).

```powershell
# On Windows:
powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
```

```bash
# On Linux / WSL:
bash packaging/build_linux.sh
```

Each produces `dist/root-annotator/` -- copy that whole folder to the USB stick. On the target
machine, run `root-annotator.exe` (Windows) or `./root-annotator` (Linux). If no display is
found (headless Linux box), it automatically drops into the terminal client -- pass
`--model`/`--input`/`--out-dir` directly, e.g.:

```bash
./root-annotator --model radicle --input /data/ct_volumes --out-dir /data/annotations
```

**GPU**: the build scripts install PyTorch's CUDA 12.6 build (`torch==2.14.0+cu126`), which
covers most NVIDIA GPUs from the last several years given a reasonably current driver. PyTorch
auto-detects CUDA at runtime and falls back to CPU if no compatible GPU/driver is found --
nothing to configure by hand. CPU inference on these volume sizes is slow (see each repo's own
README), so a CUDA-capable GPU + driver on the target machine is strongly recommended. If the
target GPU turns out to need a different CUDA version, change the `cu126` index/version in
`packaging/build_windows.ps1` / `build_linux.sh` and rebuild.

**Size**: bundling CUDA-enabled PyTorch makes each build a few GB; expect the full USB payload
(both platform builds, if you build both) to land somewhere around 5-8GB.

## Adding a third model later

Add an `EngineXyz` class in `app/core/` implementing `load()`/`run()` like the two existing
engines, then register it in `app/core/registry.py`'s `build_registry()` and `MODEL_ORDER` --
it will show up in both the GUI dropdown and the CLI `--model` choice with no other changes.
