"""Single entry point for Root Annotator.

    python main.py                          launch the GUI (falls back to CLI if no display)
    python main.py --web [--port 8765]      browser UI served from this machine (for headless
                                             machines: forward the port over SSH and open it locally)
    python main.py --cli --input data/ --out-dir output/
                                             force the terminal client

The Windows build is packaged without a console window (see packaging/root_annotator.spec),
so a GUI failure can't just print to stderr and fall back to the CLI -- nobody would see it.
Instead it's logged to a file next to the app and shown in a plain system message box.
"""
import sys
import traceback
from pathlib import Path


def _has_display() -> bool:
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    import os
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _report_startup_failure(exc: Exception) -> None:
    log_path = _app_dir() / "error_log.txt"
    try:
        with open(log_path, "a") as f:
            f.write(traceback.format_exc() + "\n")
    except OSError:
        pass

    message = (
        "Root Annotator couldn't start.\n\n"
        f"{exc}\n\n"
        f"Details were saved to:\n{log_path}"
    )
    if sys.platform.startswith("win"):
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, message, "Root Annotator", 0x10)  # MB_ICONERROR
    else:
        print(message, file=sys.stderr)


def main() -> int:
    if "--web" in sys.argv:
        sys.argv.remove("--web")
        from app.web.server import main as web_main
        return web_main()

    if "--cli" in sys.argv:
        sys.argv.remove("--cli")
        from app.cli import main as cli_main
        return cli_main()

    if not _has_display():
        print("no display detected -- use `--web` for the browser UI, or pass --input/--out-dir for the "
              "terminal client\n")
        from app.cli import main as cli_main
        return cli_main()

    try:
        from app.gui import run_gui
        return run_gui()
    except Exception as exc:  # missing/broken Qt install, no platform plugin, etc.
        _report_startup_failure(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
