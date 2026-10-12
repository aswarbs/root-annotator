"""Max-projection PNGs (side / front / top-down) of a prediction, saved to every output dir given.

    python make_projections.py <pred.npy> <label> <out_dir> [<out_dir> ...]
-> <out_dir>/<pred stem without suffix>_<label>_<view>.png
"""
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.core.projection import AXES, Projections  # noqa: E402

pred, label, out_dirs = Path(sys.argv[1]), sys.argv[2], [Path(d) for d in sys.argv[3:]]
name = pred.name.replace("_rootseg_pred.npy", "").replace("_radicle_mask.npy", "")
proj = Projections.compute(pred, progress_cb=lambda m, f=None: print(" ", m, flush=True))
for view in AXES:
    img = Image.fromarray(proj.render(view, 0.5, False))
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        img.save(d / f"{name}_{label}_{view}.png")
print("projections saved for", name, label)
