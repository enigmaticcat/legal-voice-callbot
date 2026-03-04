"""
Legal CallBot — TTS Worker
HTTP dummy server (Bước 1). Bước 2 sẽ chuyển sang gRPC.
Bootstrap only — logic nằm trong core/ và grpc_handler.py.
"""
import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime

from config import config
from core.synthesizer import Synthesizer
from core.audio_utils import generate_silence_wav
from grpc_handler import TTSServiceHandler

# ─── Logging ─────────────────────────────────────────────
logging.basicConfig(
    level=config.log_level,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("tts")

# ─── Initialize Components ──────────────────────────────
synthesizer = Synthesizer(
    backbone_repo=config.backbone_repo,
    codec_repo=config.codec_repo,
)
handler = TTSServiceHandler(synthesizer=synthesizer)


class HTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self._respond_json(200, {
                "status": "healthy",
                "service": "tts",
                "timestamp": datetime.utcnow().isoformat(),
            })
        elif self.path == "/":
            self._respond_json(200, {
                "status": "ok",
                "service": "tts",
                "version": "0.1.0",
                "mode": "dummy",
            })
        else:
            self._respond_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/speak":
            wav_data = generate_silence_wav(duration_ms=500)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav_data)))
            self.end_headers()
            self.wfile.write(wav_data)
        else:
            self._respond_json(404, {"error": "not found"})

    def _respond_json(self, status_code: int, data: dict):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def log_message(self, format, *args):
        logger.info(format % args)


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", config.port), HTTPHandler)
    logger.info(f"TTS Worker running on port {config.port}")
    server.serve_forever()
