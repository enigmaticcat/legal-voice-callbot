"""
Session Manager
Quản lý vòng đời session: tạo, timeout, cleanup.
Bước 6 sẽ thêm Redis backend.
"""
import logging
import time
from typing import Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("gateway.services.session")


@dataclass
class Session:
    """Đại diện cho một cuộc gọi đang hoạt động."""
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    is_active: bool = True

    def touch(self):
        """Cập nhật thời gian hoạt động cuối."""
        self.last_activity = time.time()

    def is_expired(self, timeout_seconds: int = 300) -> bool:
        """Kiểm tra session đã hết hạn chưa (mặc định 5 phút)."""
        return (time.time() - self.last_activity) > timeout_seconds


class SessionManager:
    """Quản lý tất cả sessions đang hoạt động."""

    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def create(self, session_id: str) -> Session:
        session = Session(session_id=session_id)
        self._sessions[session_id] = session
        logger.info(f"Session created: {session_id}")
        return session

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def remove(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Session removed: {session_id}")

    @property
    def active_count(self) -> int:
        return len(self._sessions)
