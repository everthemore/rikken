# webapp/server.py
from __future__ import annotations
import os
import json
import logging
import mimetypes
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional
import numpy as np

from webapp.session import GameSession

log = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True

CURRENT_SESSION: Optional[GameSession] = None
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")


class RikkenHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Rikken WebApp."""

    def log_message(self, format, *args):
        log.info(f"{self.command} {self.path} -> {args[1]}")

    def send_json_response(self, data: dict, status: int = 200):
        body = json.dumps(data, cls=NumpyEncoder).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        global CURRENT_SESSION
        parsed = urlparse(self.path)
        path = parsed.path

        # --- API Endpoints ---
        if path == "/api/game/state":
            if CURRENT_SESSION is None:
                CURRENT_SESSION = GameSession()
            self.send_json_response(CURRENT_SESSION.get_state_payload())
            return

        elif path == "/api/game/ai_advice":
            if CURRENT_SESSION is None:
                CURRENT_SESSION = GameSession()
            self.send_json_response(CURRENT_SESSION.get_ai_advice())
            return

        elif path == "/api/game/ai_beliefs":
            if CURRENT_SESSION is None:
                CURRENT_SESSION = GameSession()
            self.send_json_response(CURRENT_SESSION.get_ai_beliefs())
            return

        # --- Static Files Serving ---
        if path == "/" or path == "/index.html":
            self.serve_file(os.path.join(STATIC_DIR, "index.html"))
            return

        if path.startswith("/docs"):
            rel_path = path[len("/docs"):].lstrip("/")
            if not rel_path or rel_path == "index.html":
                self.serve_file(os.path.join(DOCS_DIR, "index.html"))
            else:
                self.serve_file(os.path.join(DOCS_DIR, rel_path))
            return

        # Serve static file from static/
        file_path = os.path.join(STATIC_DIR, path.lstrip("/"))
        if os.path.exists(file_path) and os.path.isfile(file_path):
            self.serve_file(file_path)
            return

        # 404 fallback
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"Not Found")

    def do_POST(self):
        global CURRENT_SESSION
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        try:
            params = json.loads(body) if body else {}
        except Exception:
            params = {}

        if path == "/api/game/new":
            human_seat = int(params.get("human_seat", 0))
            ai_diff = str(params.get("ai_difficulty", "neural_master"))
            rollouts = int(params.get("rollouts", 50))
            seed = params.get("seed")
            CURRENT_SESSION = GameSession(
                human_seat=human_seat,
                ai_difficulty=ai_diff,
                rollouts=rollouts,
                seed=int(seed) if seed is not None else None,
            )
            self.send_json_response({
                "success": True,
                "message": "New game started",
                "state": CURRENT_SESSION.get_state_payload(),
            })
            return

        if CURRENT_SESSION is None:
            CURRENT_SESSION = GameSession()

        if path == "/api/game/bid":
            bid_id = int(params.get("bid_id", 0))
            ok, msg = CURRENT_SESSION.human_bid(bid_id)
            self.send_json_response({
                "success": ok,
                "message": msg,
                "state": CURRENT_SESSION.get_state_payload(),
            })
            return

        elif path == "/api/game/declare":
            trump_suit = int(params.get("trump_suit", -1))
            vraagaas_suit = int(params.get("vraagaas_suit", -1))
            ok, msg = CURRENT_SESSION.human_declare(trump_suit, vraagaas_suit)
            self.send_json_response({
                "success": ok,
                "message": msg,
                "state": CURRENT_SESSION.get_state_payload(),
            })
            return

        elif path == "/api/game/play":
            card_id = int(params.get("card_id", -1))
            ok, msg = CURRENT_SESSION.human_play(card_id)
            self.send_json_response({
                "success": ok,
                "message": msg,
                "state": CURRENT_SESSION.get_state_payload(),
            })
            return

        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"Not Found")

    def serve_file(self, filepath: str):
        if not os.path.exists(filepath) or not os.path.isfile(filepath):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"File Not Found")
            return

        mime_type, _ = mimetypes.guess_type(filepath)
        mime_type = mime_type or "application/octet-stream"

        with open(filepath, "rb") as f:
            content = f.read()

        self.send_response(200)
        self.send_header("Content-Type", f"{mime_type}; charset=utf-8" if "text" in mime_type or "javascript" in mime_type or "json" in mime_type else mime_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def run_server(host: str = "0.0.0.0", port: int = 8080):
    """Run the interactive Rikken game server."""
    server_address = (host, port)
    httpd = ReusableHTTPServer(server_address, RikkenHTTPHandler)
    print("=" * 65)
    print(f"  ♠ ♥ ♦ ♣  RIKKEN AI INTERACTIVE WEB SERVER RUNNING")
    print(f"  Local Address:  http://localhost:{port}")
    print(f"  Network Access: http://{host}:{port}")
    print(f"  Documentation:  http://localhost:{port}/docs")
    print("=" * 65)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer shutting down.")
        httpd.server_close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_server()
