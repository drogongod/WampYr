"""Loopback-only OpenAI-compatible endpoint for Lucy's BrickCluster gateway.

Same wire protocol as F:\\my-agent-vampire\\mamba3_server.py (/v1/chat/completions, /v1/feedback,
/health, /v1/models) -- a client cannot tell which substrate is behind "Lucy's own weights"
from the API shape alone. The difference is internal: BrickClusterEngine (brickcluster_engine.py)
instead of TernaryMamba3Engine. See brickcluster_engine.py's module docstring for why.

Does not host or modify donor weights. Forwards a bounded conversation to the local frozen
donor through BrickClusterEngine, while Lucy's own quinary bricks consume and persist the
same turns, exactly as specified in F:\\Wampyr.txt.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import uuid

from brickcluster_engine import BrickClusterEngine


MODEL_ID = "lucywest-brickcluster-donor-assisted"
MAX_BODY_BYTES = 16 * 1024 * 1024


def validate_bind_host(host: str) -> str:
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError("Lucy API must bind to a numeric loopback address") from exc
    if not address.is_loopback:
        raise ValueError("Lucy API is loopback-only; bind to 127.0.0.1 or ::1")
    return str(address)


class LucyHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, address, handler, engine):
        self.engine = engine
        super().__init__(address, handler)


class Handler(BaseHTTPRequestHandler):
    server_version = "LucyBrickCluster/1.0"

    def log_message(self, fmt, *args):
        print("Lucy API:", fmt % args, flush=True)

    def _send(self, status: int, payload: dict):
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path in ("/health", "/architecture"):
            try:
                self._send(200, self.server.engine.health())
            except Exception as exc:
                self._send(503, {"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            return
        if self.path == "/v1/models":
            self._send(200, {"object": "list", "data": [{
                "id": MODEL_ID, "object": "model", "owned_by": "local",
            }]})
            return
        self._send(404, {"error": {"message": "not found", "type": "not_found"}})

    def do_POST(self):
        if self.path == "/v1/feedback":
            try:
                length = int(self.headers.get("Content-Length", "-1"))
                if length < 0 or length > MAX_BODY_BYTES:
                    raise ValueError(f"Content-Length must be between 0 and {MAX_BODY_BYTES}")
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(body, dict):
                    raise ValueError("request body must be a JSON object")
                result = self.server.engine.accept_experience(
                    body.get("session_id", ""), body.get("experience_id", ""),
                    action=body.get("action", ""), answer=body.get("answer"))
                self._send(200, result)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                self._send(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
            except Exception as exc:
                self._send(502, {"error": {"message": f"{type(exc).__name__}: {exc}",
                                          "type": "lucy_gateway_error"}})
            return
        if self.path != "/v1/chat/completions":
            self._send(404, {"error": {"message": "not found", "type": "not_found"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
            if length < 0 or length > MAX_BODY_BYTES:
                raise ValueError(f"Content-Length must be between 0 and {MAX_BODY_BYTES}")
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("request body must be a JSON object")
            if body.get("stream"):
                raise ValueError("streaming is not supported; set stream=false")
            body.setdefault("vampire_session_id", body.get("conversation_id") or uuid.uuid4().hex)
            response = self.server.engine.complete(body)
            self._send(200, response)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            self._send(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
        except Exception as exc:
            self._send(502, {"error": {"message": f"{type(exc).__name__}: {exc}",
                                      "type": "lucy_gateway_error"}})


def serve(host: str = "127.0.0.1", port: int = 8105):
    """Default port 8105 -- deliberately NOT 8104 (the ternary engine's port), so both gateways
    can run side by side against the same frozen donor for comparison, per the user's "keep it
    organized" instruction: nothing here overwrites or collides with the ternary build."""
    host = validate_bind_host(host)
    # Reuse the SAME donor-identity-checking starter as the ternary engine. It never terminates
    # a process; if 8102 belongs to a different model, startup fails explicitly. NOT invoked
    # automatically by this module -- the caller (main(), or a future supervisor) decides when to
    # actually claim the GPU, per the standing shared-machine rule (check nvidia-smi first).
    from server import ensure_server
    ensure_server()
    engine = BrickClusterEngine()
    httpd = LucyHTTPServer((host, port), Handler, engine)
    print(f"Lucy BrickCluster API listening at http://{host}:{port}", flush=True)
    print("Visible prose is donor-assisted until Lucy can answer without the donor.", flush=True)
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8105)
    args = parser.parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
