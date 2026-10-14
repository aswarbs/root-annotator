"""Single entry point for Root Annotator.

    python main.py                          launch the GUI (falls back to CLI if no display)
    python main.py --web [--port 8765]      browser UI served from this machine (for headless
                                             machines: forward the port over SSH and open it locally)
    python main.py --cli --input data/ --out-dir output/
                                             force the terminal client

The Windows build is packaged without a console window (see packaging/root_annotator.spec),
so a GUI failure can't just print to stderr and fall back to the CLI -- nobody would see it.
Instead it's logged to a file next to the app and shown in a plain system message box.

Packaged builds can update their own code: app.updater stages a newer copy of `app/` under
%LOCALAPPDATA%\\RootAnnotator\\code_next; at launch it is promoted to `code` and put in front of
the bundled copy on sys.path. If the newer code fails to import, it is parked in `code_broken`
and the bundled code is used instead.
"""
import importlib
import os
import shutil
import sys
import traceback
from pathlib import Path


def _has_display() -> bool:
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
    return Path(base) / "RootAnnotator"


def _log_path() -> Path:
    """error_log.txt next to the app, or in the user data folder when that is not writable
    (e.g. an install under Program Files)."""
    for d in (_app_dir(), _data_dir()):
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / "error_log.txt"
            with open(probe, "a"):
                pass
            return probe
        except OSError:
            continue
    return Path("error_log.txt")


def _activate_updated_code():
    """Promotes a staged update and puts the newest downloaded code first on sys.path.
    Returns the code folder in use, or None to use the bundled copy."""
    if not getattr(sys, "frozen", False):
        return None
    root = _data_dir()
    staged, live = root / "code_next", root / "code"
    try:
        if (staged / "app" / "__init__.py").is_file():
            shutil.rmtree(live, ignore_errors=True)
            staged.replace(live)
    except OSError:
        pass
    if (live / "app" / "__init__.py").is_file():
        sys.path.insert(0, str(live))
        return live
    return None


def _drop_updated_code(live: Path) -> None:
    """The downloaded code would not import: park it and fall back to the bundled copy."""
    try:
        sys.path.remove(str(live))
    except ValueError:
        pass
    for name in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        del sys.modules[name]
    broken = live.parent / "code_broken"
    shutil.rmtree(broken, ignore_errors=True)
    try:
        live.replace(broken)
    except OSError:
        shutil.rmtree(live, ignore_errors=True)


def _import(module: str, live):
    try:
        return importlib.import_module(module)
    except Exception:
        if live is None:
            raise
        with open(_log_path(), "a") as f:
            f.write("Updated code failed to load, using the bundled version:\n" + traceback.format_exc() + "\n")
        _drop_updated_code(live)
        return importlib.import_module(module)


def _report_startup_failure(exc: Exception) -> None:
    log_path = _log_path()
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
    live = _activate_updated_code()

    if "--web" in sys.argv:
        sys.argv.remove("--web")
        return _import("app.web.server", live).main()

    if "--cli" in sys.argv:
        sys.argv.remove("--cli")
        return _import("app.cli", live).main()

    if not _has_display():
        print("no display detected -- use `--web` for the browser UI, or pass --input/--out-dir for the "
              "terminal client\n")
        return _import("app.cli", live).main()

    try:
        return _import("app.gui", live).run_gui()
    except Exception as exc:  # missing/broken Qt install, no platform plugin, etc.
        _report_startup_failure(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
