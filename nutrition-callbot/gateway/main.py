"""
Legal CallBot — API Gateway
FastAPI application factory.
Chỉ bootstrap + compose — không có business logic ở đây.
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


@app.get("/")
async def root():
    """Root endpoint — status check."""
    return {"status": "ok", "service": "gateway", "version": "0.1.0"}


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
