"""Terminal client for headless machines: point at files/folders of .npy
volumes, get predictions + provenance JSON written to an output directory.

    python -m app.cli --input data/ --out-dir output/
    python -m app.cli --input vol1.npy vol2.npy --out-dir output/ --clean --min-size 200
"""
import argparse
import sys
from pathlib import Path

from .core.jobs import JobRunner
from .core.engine import RootsegEngine
from .core.postprocess import CLEAN_MIN_SIZE, CLEAN_THRESHOLD, clean_prediction
from .core.volumes import discover_volumes


def _print_progress(msg: str, frac=None) -> None:
    if frac is not None:
        print(f"  [{frac * 100:5.1f}%] {msg}", flush=True)
    else:
        print(f"  {msg}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="root-annotator",
        description="Run RootSeg-LSTM over one or more CT volumes and save root predictions.",
    )
    parser.add_argument(
        "--input", required=True, nargs="+",
        help=".npy volume file(s), slice-folder scans (tif/bmp stacks, e.g. D:\\rootCT\\barley\\M16605), "
             "and/or folders containing either",
    )
    parser.add_argument("--out-dir", required=True, help="where predictions + provenance JSON are written")
    parser.add_argument("--clean", action="store_true",
                        help="also write a noise-cleaned mask (<name>_clean.npy) for every prediction")
    parser.add_argument("--threshold", type=float, default=CLEAN_THRESHOLD, help="clean-up probability threshold")
    parser.add_argument("--min-size", type=int, default=CLEAN_MIN_SIZE,
                        help="clean-up: drop 3D components smaller than this many voxels")
    parser.add_argument("--max-slices", type=int, default=None,
                        help="only predict the first N slices of each volume (quick tests); the rest stay zero")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        volumes = discover_volumes(args.input)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not volumes:
        print("error: no .npy volumes or slice-folder scans found in the given input", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    print(f"{len(volumes)} volume(s) queued, output -> {out_dir}")
    for v in volumes:
        print(f"  - {v}")

    engine = RootsegEngine(max_slices=args.max_slices)
    runner = JobRunner(engine, out_dir)

    def on_job_status(status):
        if status.state == "running":
            print(f"\n=== {status.volume.name} ===")
        elif status.state == "done":
            print(f"done: {status.volume.name} ({status.message})")
            for o in status.outputs:
                print(f"  -> {o}")
                if args.clean:
                    r = clean_prediction(o, threshold=args.threshold, min_size=args.min_size,
                                         progress_cb=_print_progress)
                    print(f"  -> {r['output']} ({r['components_kept']}/{r['components_total']} components kept)")
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
