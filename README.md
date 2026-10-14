# Root Annotator

One lightweight app for running **RootSeg-LSTM** (the `e4_skips_bal` checkpoint) over root CT
volumes, watching the slices come out live, looking at the result as a 2D projection, and cleaning
up noise. Results are also written as **TIFF stacks**. Add volumes, run, get a prediction + a provenance JSON per output.

Three front ends over the same core (`app/core/session.py`): a PyQt6 GUI (default), a **browser
UI** for headless machines, and a terminal client. The GUI and the browser UI share the same
single-column flow: choose volumes, choose an output folder, Segment, viewer, results.

## The model

RootSeg-LSTM is a ConvLSTM that walks a volume slice by slice along Z, feeding its own previous
prediction back in. The checkpoint ships inside the app (`app/rootseg/checkpoints/e4_skips_bal.pt`,
md5 `82282a09...`, step 6000, 292K params; held-out Dice at threshold 0.5: 13-volume mean 0.536,
wheat mean 0.541), together with the model code (`app/rootseg/`, copied from the `rootseg-lstm`
repo), so nothing else needs to be checked out. Input is a `.npy` volume `(Z, H, W)` uint8, or a
folder of `.tif`/`.bmp` slices.

## Layout

```
app/
  rootseg/        model.py, norm.py, tiling.py (vendored from rootseg-lstm) + checkpoints/
  core/
    engine.py       loads the checkpoint, runs a volume, streams the prediction to disk
    session.py      run / viewer / clean-up state shared by the GUI and the browser UI
    jobs.py         runs a batch of volumes, reports progress / per-slice live events
    live.py         renders one input+prediction slice (live view)
    projection.py   max-intensity projections for the visualiser (cached next to the result)
    postprocess.py  noise clean-up: threshold + remove small 3D components
    tiffstack.py    writes a prediction as a folder of per-slice 8-bit TIFFs
    volumes.py, sidecar.py, paths.py
  gui.py          PyQt6 window (thin view over core/session.py)
  web/            browser UI (stdlib http.server + static pages)
  cli.py          terminal client
main.py           entry point
```

## Running

```bash
python main.py                                   # Qt GUI
python main.py --web [--port 8765]               # browser UI (headless machines)
python main.py --cli --input path/to/volumes --out-dir out/ [--clean --min-size 1000] [--max-slices N]
```

### Using it on a headless cluster

```bash
# on the cluster (ideally inside a GPU allocation):
python main.py --web --port 8765
# on your laptop:
ssh -L 8765:localhost:8765 <cluster-login-or-compute-node>
# then open http://localhost:8765
```

The web UI is one page, top to bottom: choose volumes, choose an output folder, press **Segment**,
then a viewer and the results list. A volume that already has a result in the output folder is
shown as *Segmented* and skipped (use **Redo** to run it again). It has a server-side file
browser, binds to localhost only by default (it can read any file your account can, so only use
`--host 0.0.0.0` on a trusted network), and takes `--max-slices N` to predict only the first N
slices (quick tests) and `--out-dir` for the default output folder.

## Viewer and Clean up (web UI)

- **While a volume runs**: the newest slice is shown with the predicted root probability in red;
  drag the slider to look back.
- **View** on a finished volume: a max-intensity projection (side / front / top-down), or the
  slices (Projection / Slices switch). Projections are computed in one streaming pass and cached
  as `<file>.proj.npz`, so reopening is instant.
- **Clean up**: one button, no parameters. It thresholds the probabilities (0.5) and removes 3D
  connected components (26-connectivity) smaller than 1000 voxels (`CLEAN_MIN_SIZE` in
  `app/core/postprocess.py`). A progress bar is shown, and the projection updates live with what
  is being removed in orange (kept = white). When it finishes you can **Keep** or **Undo**. The
  raw prediction is never modified; the cleaned mask is `<stem>_rootseg_pred_clean.npy`
  (uint8 0/1). It labels slab by slab, so it needs a few GB at most even for ~1e10-voxel volumes.
  Needs `scipy`.

Volumes are large (e.g. 1176x1416x1409 uint8 = 2.3 GB), so the prediction is written to disk as
it is produced rather than held in RAM, and is stored as **float16** (~4.7 GB for that volume).

## Input formats

The app also accepts **slice-folder scans** in the `D:\rootCT` layout: a folder of 2D `.tif`/`.bmp`
slices (one per z, sorted by name), either the slice folder itself or its parent containing
`input/` next to `output/`. Point it at a single scan, or at any folder above (e.g. `D:\rootCT`)
and it finds every scan underneath (up to 3 levels). `output`/`ouput` folders (ground truth), `*_Mask.npy` /
`*_rootseg_pred*.npy` files and our own TIFF-stack folders are ignored. Each scan is stacked to a temporary uint8 `.npy` in `<out-dir>/_stacked/` (deleted after
the run); 16-bit slices are stretched to 0-255 using the scan's 0.5-99.5 percentile range, 8-bit
slices are copied as-is. Stacking takes ~2 minutes per scan. The conversion settings are recorded in the output sidecar JSON.

The app also accepts **`.vol` reconstructions** from the CT scanner (e.g. `D:\Sept CT data`). The
volume size is read from the `.pcr` file with the same name next to it; the data type (uint8, uint16
or float32) is inferred from the file size. The `.tif`/`.pca`/`.vgl` files and `*_Annotated` folders
beside a `.vol` are ignored. Like 16-bit slice folders, the volume is converted slice by slice to a
temporary uint8 `.npy` in `<out-dir>/_stacked/` (0.5-99.5 percentile stretch, recorded in the sidecar
JSON) and deleted after the run. These files are large (the rice scans are ~15 billion voxels): the
results drive needs roughly 3 bytes per voxel free during a run, and the app checks this before it
starts.

## Output

Every prediction is also written as a **TIFF stack**: a folder `<stem>_rootseg_pred_tiff/` with one
8-bit TIFF per slice (`slice_00000.tif`, ...; 255 = root at probability >= 0.5, 0 = background).
A kept clean-up gets `<stem>_rootseg_pred_clean_tiff/`. The `.npy` is the app's working format.

For each input volume, in the chosen output folder: `<stem>_rootseg_pred.npy`, a per-voxel root
probability in `[0, 1]`, shape `(Z, H, W)`, **float16** (slice 0 is never predicted and is all
zero). Every output gets a `<file>.npy.json` sidecar recording the model/checkpoint, device and
run time, which is the provenance record for that annotation.

## Packaging for the USB stick

PyInstaller bundles Python + PyTorch + PyQt6 + the model code and checkpoint into a
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
`--input`/`--out-dir` directly (or `--web`), e.g.:

```bash
./root-annotator --input /data/ct_volumes --out-dir /data/annotations
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
