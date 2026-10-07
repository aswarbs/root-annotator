"""Terminal fallback for headless machines: pick a model, point at files/folders of .npy
volumes, get predictions + provenance JSON written to an output directory.

    python -m app.cli --model radicle  --input data/ --out-dir output/
    python -m app.cli --model rootseg  --input vol1.npy vol2.npy --out-dir output/
"""
import argparse
import sys
from pathlib import Path

from .core.jobs import JobRunner
from .core.registry import MODEL_ORDER, build_registry
from .core.volumes import discover_volumes


def _print_progress(msg: str, frac=None) -> None:
    if frac is not None:
        print(f"  [{frac * 100:5.1f}%] {msg}", flush=True)
    else:
        print(f"  {msg}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    registry = build_registry()
    parser = argparse.ArgumentParser(
        prog="root-annotator",
        description="Run a root-segmentation model over one or more CT volumes and save annotations.",
    )
    parser.add_argument(
        "--model", required=True, choices=MODEL_ORDER,
        help="; ".join(f"{k}: {registry[k]['display_name']}" for k in MODEL_ORDER),
    )
    parser.add_argument(
        "--input", required=True, nargs="+",
        help=".npy volume file(s), slice-folder scans (tif/bmp stacks, e.g. D:\\rootCT\\barley\\M16605), "
             "and/or folders containing either",
    )
    parser.add_argument("--out-dir", required=True, help="where predictions + provenance JSON are written")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    registry = build_registry()

    try:
        volumes = discover_volumes(args.input)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not volumes:
        print("error: no .npy volumes or slice-folder scans found in the given input", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    print(f"model: {registry[args.model]['display_name']}")
    print(f"{len(volumes)} volume(s) queued, output -> {out_dir}")
    for v in volumes:
        print(f"  - {v}")

    engine = registry[args.model]["make_engine"]()
    runner = JobRunner(engine, args.model, out_dir)

    def on_job_status(status):
        if status.state == "running":
            print(f"\n=== {status.volume.name} ===")
        elif status.state == "done":
            print(f"done: {status.volume.name} ({status.message})")
            for o in status.outputs:
                print(f"  -> {o}")
        elif status.state == "failed":
            print(f"FAILED: {status.volume.name}: {status.message}", file=sys.stderr)
        elif status.state == "cancelled":
            print(f"cancelled: {status.volume.name}")

    results = runner.run_all(volumes, on_job_status=on_job_status, on_progress=_print_progress)

    n_done = sum(1 for r in results if r.state == "done")
    n_failed = sum(1 for r in results if r.state == "failed")
    print(f"\n{n_done}/{len(results)} volumes done, {n_failed} failed")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
