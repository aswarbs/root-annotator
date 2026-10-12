"""Browser front end for headless machines: pick volumes and an output folder, press Segment, watch
the prediction come out, and clean it up -- served over plain HTTP from the stdlib (no extra packages).

    python main.py --web [--port 8765] [--host 127.0.0.1] [--max-slices N]

then on the machine you're sitting at:   ssh -L 8765:localhost:8765 <cluster>   and open
http://localhost:8765. It binds to localhost by default because it can browse and read any file
the user running it can; only pass --host 0.0.0.0 on a network you trust.
"""
import argparse
import getpass
import io
import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from ..core.engine import expected_output
from ..core.projection import AXES
from ..core.session import Session, volume_path as _volume

STATIC = Path(__file__).resolve().parent / "static"
PAGES = {"/": "index.html"}


def _png(rgb) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


_DATA_HOME = Path("/data") / getpass.getuser()
BROWSE_ROOT = str(_DATA_HOME if _DATA_HOME.is_dir() else Path.home())  # where "Add volumes" opens
STATE = Session()  # --max-slices / --out-dir are applied in main()


class Handler(BaseHTTPRequestHandler):
    server_version = "RootAnnotator"

    def log_message(self, *args):  # keep the terminal quiet; the page polls constantly
        pass

    # helpers
    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _error(self, code, msg):
        self._json({"error": msg}, code)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def _pred(self, body: dict) -> Path:
        """The finished prediction for body['vol'] in body['out_dir']."""
        pred = expected_output(_volume(body["vol"]), Path(body["out_dir"]).expanduser())
        if not pred.is_file():
            raise ValueError("this volume has no result yet")
        return pred

    # routing
    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path in PAGES:
                self._send(200, (STATIC / PAGES[url.path]).read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/state":
                self._json({**STATE.snapshot(), "browse_root": BROWSE_ROOT})
            elif url.path == "/api/browse":
                self._browse(q.get("path") or str(Path.home()))
            elif url.path == "/api/view":
                self._json(STATE.view_info(_volume(q["vol"]), Path(q["out"]).expanduser()))
            elif url.path == "/api/proj.status" and STATE.live_projection(_volume(q["vol"]), Path(q["out"]).expanduser()):
                self._json({"state": "ready", "progress": 1.0})
            elif url.path == "/api/proj.png" and STATE.live_projection(_volume(q["vol"]), Path(q["out"]).expanduser()):
                live = STATE.live_projection(_volume(q["vol"]), Path(q["out"]).expanduser())
                view = q.get("view", "side")
                if view not in AXES:
                    raise ValueError("unknown view")
                ax = AXES[view][0]
                white, orange = live[f"w{ax}"], live[f"o{ax}"]
                img = np.full(white.shape + (3,), 18, np.uint8)
                img[white] = 255
                img[orange] = (255, 170, 40)
                self._send(200, _png(img), "image/png")
            elif url.path == "/api/proj.status":
                srcs = STATE.proj_sources(_volume(q["vol"]), Path(q["out"]).expanduser(), q.get("diff") == "1")
                if srcs is None:
                    raise ValueError("no result for this volume")
                self._json(STATE.proj_status(srcs))
            elif url.path == "/api/proj.png":
                srcs = STATE.proj_sources(_volume(q["vol"]), Path(q["out"]).expanduser(), q.get("diff") == "1")
                view = q.get("view", "side")
                if view not in AXES:
                    raise ValueError("unknown view")
                img = STATE.proj_image(srcs, view) if srcs else None
                if img is None:
                    self._error(409, "projection not ready")
                else:
                    self._send(200, _png(img), "image/png")
            elif url.path == "/api/frame.png":
                rgb = STATE.frame(_volume(q["vol"]), Path(q["out"]).expanduser(),
                                  int(q["z"]) if "z" in q else None, q.get("diff") == "1")
                if rgb is None:
                    self._error(404, "no result for this volume")
                else:
                    self._send(200, _png(rgb), "image/png")
            else:
                self._error(404, "not found")
        except (KeyError, ValueError, FileNotFoundError, OSError) as exc:
            self._error(400, str(exc))
        except Exception as exc:
            traceback.print_exc()
            self._error(500, str(exc))

    def do_POST(self):
        url = urlparse(self.path)
        try:
            body = self._body()
            if url.path == "/api/check":
                self._json({"volumes": STATE.check([Path(p).expanduser() for p in body["inputs"]],
                                                   Path(body["out_dir"]).expanduser())})
            elif url.path == "/api/run":
                STATE.start_run([Path(p).expanduser() for p in body["inputs"]],
                                Path(body["out_dir"]).expanduser(), body.get("redo", []))
                self._json({"ok": True})
            elif url.path == "/api/cancel":
                STATE.cancel()
                self._json({"ok": True})
            elif url.path == "/api/cleanup":
                STATE.start_cleanup(self._pred(body))
                self._json({"ok": True})
            elif url.path == "/api/keep_cleanup":
                STATE.keep_cleanup(self._pred(body))
                self._json({"ok": True})
            elif url.path == "/api/undo_cleanup":
                STATE.undo_cleanup(self._pred(body))
                self._json({"ok": True})
            else:
                self._error(404, "not found")
        except (KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
            self._error(400, str(exc))
        except Exception as exc:
            traceback.print_exc()
            self._error(500, str(exc))

    def _browse(self, raw: str):
        p = Path(raw).expanduser().resolve()
        if p.is_file():
            p = p.parent
        entries = []
        for child in sorted(p.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower())):
            try:
                if child.name.startswith("."):
                    continue
                if child.is_dir():
                    entries.append({"name": child.name, "dir": True})
                elif child.suffix == ".npy":
                    entries.append({"name": child.name, "dir": False, "size": child.stat().st_size})
            except OSError:
                continue
        self._json({"path": str(p), "parent": str(p.parent), "entries": entries})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="root-annotator --web", description="Root Annotator browser UI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max-slices", type=int, default=None,
                    help="only predict the first N slices of each volume (quick tests)")
    ap.add_argument("--out-dir", default=None,
                    help="default output folder shown on the page (default: ~/root_annotator_results)")
    args = ap.parse_args(argv)
    STATE.max_slices = args.max_slices
    if args.out_dir:
        STATE.default_out = str(Path(args.out_dir).expanduser().resolve())
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"Root Annotator web UI on http://{args.host}:{args.port}")
    if args.host in ("127.0.0.1", "localhost"):
        print(f"from your laptop:  ssh -L {args.port}:localhost:{args.port} <this machine>   then open "
              f"http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
