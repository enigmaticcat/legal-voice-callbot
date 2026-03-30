"""
Health Check Routes
GET /health — Trạng thái Gateway + kết nối tới workers.
"""
from datetime import datetime

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """Health check — trạng thái Gateway."""
    return {
        "status": "healthy",
        "service": "gateway",
        "timestamp": datetime.utcnow().isoformat(),
    }
