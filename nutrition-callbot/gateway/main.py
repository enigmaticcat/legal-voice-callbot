"""
Legal CallBot — API Gateway
FastAPI application factory.
Chỉ bootstrap + compose — không có business logic ở đây.
"""
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import settings
from routes.health import router as health_router
from routes.websocket import router as ws_router

# ─── Logging ─────────────────────────────────────────────
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("gateway")

# ─── App Factory ─────────────────────────────────────────
app = FastAPI(
    title="Legal CallBot Gateway",
    description="API Gateway cho hệ thống tư vấn pháp luật bằng giọng nói",
    version="0.1.0",
)

# ─── Middleware ──────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routes ──────────────────────────────────────────────
app.include_router(health_router)
app.include_router(ws_router)

# ─── Static files — React frontend ───────────────────────
# Serve web/dist nếu đã được build (Colab / production).
# Route handlers ở trên có priority cao hơn, nên /health và /ws/voice không bị ảnh hưởng.
_WEB_DIST = Path(__file__).parent.parent / "web" / "dist"
if _WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_WEB_DIST), html=True), name="static")
    logger.info("Serving frontend from %s", _WEB_DIST)


# ─── Startup / Shutdown ─────────────────────────────────
@app.on_event("startup")
async def startup():
    logger.info("Gateway starting up...")
    logger.info(f"   ASR  → {settings.asr_address}")
    logger.info(f"   Brain → {settings.brain_address}")
    logger.info(f"   TTS  → {settings.tts_address}")


@app.on_event("shutdown")
async def shutdown():
    logger.info("Gateway shutting down...")
