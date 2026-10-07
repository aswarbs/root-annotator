"""The one place that knows about both models -- add a third model here and it shows up in
both the GUI dropdown and the CLI --model choice automatically.
"""
from .engine_radicle import RadicleEngine
from .engine_rootseg import RootsegEngine
from .paths import repo_dir


def build_registry():
    radicle_dir = repo_dir("radicle")
    rootseg_dir = repo_dir("rootseg-lstm")
    return {
        "radicle": {
            "short_name": "Radicle",
            "display_name": "Radicle (two-stage U-Net cascade, sequential traversal)",
            "output_note": "writes a mask, a vector field, and a patch visit-order array per volume",
            "make_engine": lambda: RadicleEngine(
                repo_dir=radicle_dir,
                first_ckpt=radicle_dir / "checkpoints" / "stage1" / "best.pth",
                second_ckpt=radicle_dir / "checkpoints" / "stage2" / "best.pth",
            ),
        },
        "rootseg": {
            "short_name": "RootSeg-LSTM",
            "display_name": "RootSeg-LSTM (ConvLSTM slice-by-slice tracker)",
            "output_note": "writes one root-probability mask per volume",
            "make_engine": lambda: RootsegEngine(
                repo_dir=rootseg_dir,
                ckpt_path=rootseg_dir / "checkpoints" / "best.pt",
            ),
        },
    }


MODEL_ORDER = ["radicle", "rootseg"]
