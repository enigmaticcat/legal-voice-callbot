"""
Legal CallBot — ASR Worker
HTTP dummy server (Bước 1). Bước 2 sẽ chuyển sang gRPC.
Bootstrap only — logic nằm trong core/ và grpc_handler.py.
"""
import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime

from config import config
from grpc_handler import ASRServiceHandler

# ─── Logging ─────────────────────────────────────────────
logging.basicConfig(
    level=config.log_level,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("asr")

# ─── Service Handler ─────────────────────────────────────
handler = ASRServiceHandler()


class HTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {
                "status": "healthy",
                "service": "asr",
                "timestamp": datetime.utcnow().isoformat(),
            })
        elif self.path == "/":
            self._respond(200, {
                "status": "ok",
                "service": "asr",
                "version": "0.1.0",
                "mode": "dummy",
            })
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/transcribe":
            result = handler.transcriber.transcribe(b"")
            self._respond(200, {
                "text": result["text"],
                "is_final": True,
                "confidence": result["confidence"],
            })
        else:
            self._respond(404, {"error": "not found"})

    def _respond(self, status_code: int, data: dict):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def log_message(self, format, *args):
        logger.info(format % args)


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", config.port), HTTPHandler)
    logger.info(f"🎤 ASR Worker running on port {config.port}")
    server.serve_forever()
