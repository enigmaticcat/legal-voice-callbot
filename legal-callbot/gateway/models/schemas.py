"""
Pydantic Request/Response Schemas
Định nghĩa data models cho Gateway API.
"""
from pydantic import BaseModel
from typing import Optional


# ─── Health ──────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: str


# ─── WebSocket Messages ─────────────────────────────────
class WSTranscript(BaseModel):
    """Message gửi transcript cho client."""
    type: str = "transcript"
    session_id: str
    text: str
    is_final: bool = False


class WSBotResponse(BaseModel):
    """Message gửi câu trả lời bot cho client."""
    type: str = "bot_response"
    session_id: str
    text: str
    is_final: bool = False


class WSAudioReady(BaseModel):
    """Signal báo audio sắp được gửi."""
    type: str = "audio_start"
    session_id: str
    sample_rate: int = 24000


class WSError(BaseModel):
    """Message lỗi."""
    type: str = "error"
    session_id: str
    message: str
    code: Optional[str] = None
