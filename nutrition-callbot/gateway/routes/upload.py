from __future__ import annotations

import logging

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from routes.websocket import orchestrator

router = APIRouter(tags=["upload"])
logger = logging.getLogger("gateway.routes.upload")

_ALLOWED_EXTENSIONS = {"pdf", "txt", "md"}
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB


@router.post("/upload")
async def upload_document(
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    filename = file.filename or "upload"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Định dạng không hỗ trợ: .{ext}")

    content = await file.read()
    if len(content) > _MAX_BYTES:
        raise HTTPException(status_code=400, detail="File quá lớn (tối đa 5 MB)")

    try:
        result = await orchestrator.upload_document(session_id, filename, content)
    except Exception as e:
        logger.exception("[%s] upload_document failed", session_id)
        raise HTTPException(status_code=500, detail="Không thể xử lý tài liệu.")

    return result
