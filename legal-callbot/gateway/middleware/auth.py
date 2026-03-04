"""
JWT Authentication Middleware
Bước 6 sẽ implement JWT verification cho WebSocket connections.
"""
import logging

logger = logging.getLogger("gateway.middleware.auth")


async def verify_token(token: str) -> bool:
    """
    Verify JWT token.
    TODO: Implement JWT verification ở Bước 6.

    Args:
        token: JWT token từ WebSocket query param.

    Returns:
        True nếu token hợp lệ.
    """
    logger.debug(f"Verifying token: {token[:20]}...")
    # Dummy — cho qua tất cả
    return True
