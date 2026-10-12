"""Radicle two-stage cascade, sequential sliding window (same algorithm as
cascade_unet/inference/infer_seq.py: 128^3 patches, 50% overlap, per-patch standardisation,
mask = sigmoid(stage1 + stage2 refinement), overlapping patches averaged), but memory-lean:
the uint8 volume stays a memmap, the running sum is float32 and the hit count uint8, and the
averaged mask is written as float16. The vector field and visit-order maps are not saved.

    python radicle_lean.py <volume.npy> <out_dir>
Writes <out_dir>/<stem>_radicle_mask.npy (float16, same shape as the volume) and a TIFF stack.
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path.home() / "roots" / "code" / "models" / "cascade_unet"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.model import DEVICE, UNet3D, UNet3DRefine  # noqa: E402
from app.core.tiffstack import write_tiff_stack  # noqa: E402

PATCH, BATCH = 128, 2


def main(vol_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    first = UNet3D().to(DEVICE)
    first.load_state_dict(torch.load(REPO / "checkpoints/stage1/best.pth", map_location=DEVICE))
    first.eval()
    second = UNet3DRefine(in_channels=5).to(DEVICE)
    second.load_state_dict(torch.load(REPO / "checkpoints/stage2/best.pth", map_location=DEVICE))
    second.eval()

    vol = np.load(str(vol_path), mmap_mode="r")
    D, H, W = vol.shape
    stride = PATCH // 2
    coords = [(z, y, x) for z in range(0, D - PATCH + 1, stride)
              for y in range(0, H - PATCH + 1, stride) for x in range(0, W - PATCH + 1, stride)]
    print(f"{vol_path.name}: {D}x{H}x{W}, {len(coords)} patches on {DEVICE}", flush=True)

    acc = np.zeros((D, H, W), dtype=np.float32)
    cnt = np.zeros((D, H, W), dtype=np.uint8)
    t0 = time.time()
    with torch.no_grad():
        for b0 in range(0, len(coords), BATCH):
            bc = coords[b0:b0 + BATCH]
            patches = []
            for z, y, x in bc:
                p = np.asarray(vol[z:z + PATCH, y:y + PATCH, x:x + PATCH], dtype=np.float32) / 255.0
                patches.append((p - p.mean()) / (p.std() + 1e-6))
            t = torch.from_numpy(np.stack(patches)[:, None]).to(DEVICE)
            mask1, vec1 = first(t)
            vec_up = F.normalize(F.interpolate(vec1, size=t.shape[2:], mode="trilinear", align_corners=False), dim=1)
            logits2 = second(torch.cat([t, torch.sigmoid(mask1), vec_up], dim=1))
            m = torch.sigmoid(mask1 + logits2)[:, 0].cpu().numpy()
            for i, (z, y, x) in enumerate(bc):
                acc[z:z + PATCH, y:y + PATCH, x:x + PATCH] += m[i]
                cnt[z:z + PATCH, y:y + PATCH, x:x + PATCH] += 1
            if (b0 // BATCH) % 200 == 0:
                done = b0 + len(bc)
                el = time.time() - t0
                print(f"  {done}/{len(coords)} patches, {el / 60:.1f} min, eta {el / done * (len(coords) - done) / 60:.0f} min",
                      flush=True)

    out_path = out_dir / f"{vol_path.stem}_radicle_mask.npy"
    tmp = out_path.with_name(out_path.name + ".partial")
    out = np.lib.format.open_memmap(str(tmp), mode="w+", dtype=np.float16, shape=(D, H, W))
    for z0 in range(0, D, 64):
        out[z0:z0 + 64] = acc[z0:z0 + 64] / (cnt[z0:z0 + 64].astype(np.float32) + 1e-6)
    out.flush()
    del out, acc, cnt
    tmp.replace(out_path)
    print(f"saved {out_path} in {(time.time() - t0) / 60:.1f} min; writing TIFF stack", flush=True)
    write_tiff_stack(out_path, progress_cb=lambda msg, frac=None: print(" ", msg, flush=True))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
