from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


DISCLAIMER = "Để được tư vấn chính xác, bạn nên gặp bác sĩ dinh dưỡng."

_PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"bỏ qua\s+(tất cả\s+)?(hướng dẫn|chỉ dẫn)\s+trước", re.IGNORECASE),
    re.compile(r"system\s+prompt", re.IGNORECASE),
    re.compile(r"developer\s+message", re.IGNORECASE),
    re.compile(r"hãy\s+đóng\s+vai", re.IGNORECASE),
]

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_UNSAFE_OUTPUT_PATTERNS = [
    re.compile(r"\bngừng thuốc\b", re.IGNORECASE),
    re.compile(r"\btăng liều\b", re.IGNORECASE),
    re.compile(r"\bgiảm liều\b", re.IGNORECASE),
    re.compile(r"\bchắc chắn chữa khỏi\b", re.IGNORECASE),
    re.compile(r"\bkhỏi hoàn toàn\b", re.IGNORECASE),
]


@dataclass(frozen=True)
class EvidenceAssessment:
    sufficient: bool
    reason: str
    doc_count: int
    total_chars: int


@dataclass(frozen=True)
class OutputSafetyAssessment:
    safe: bool
    needs_disclaimer: bool
    removed_url: bool
    unsafe_claim_detected: bool


def assess_evidence(
    docs: Iterable[dict],
    min_docs: int = 1,
    min_total_chars: int = 120,
) -> EvidenceAssessment:
    docs_list = list(docs or [])
    contents = [(doc.get("content") or "").strip() for doc in docs_list]
    total_chars = sum(len(content) for content in contents)

    if len(docs_list) < min_docs:
        return EvidenceAssessment(
            sufficient=False,
            reason="no_retrieved_document",
            doc_count=len(docs_list),
            total_chars=total_chars,
        )

    if total_chars < min_total_chars:
        return EvidenceAssessment(
            sufficient=False,
            reason="retrieved_context_too_short",
            doc_count=len(docs_list),
            total_chars=total_chars,
        )

    return EvidenceAssessment(
        sufficient=True,
        reason="sufficient",
        doc_count=len(docs_list),
        total_chars=total_chars,
    )


def clean_retrieved_content(content: str) -> str:
    lines = []
    for raw_line in (content or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if any(pattern.search(line) for pattern in _PROMPT_INJECTION_PATTERNS):
            continue
        lines.append(line)
    return "\n".join(lines)


def clean_voice_output(text: str) -> tuple[str, bool]:
    cleaned = _MARKDOWN_LINK_RE.sub(r"\1", text or "")
    cleaned, removed = _URL_RE.subn("", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned, removed > 0


def assess_output_safety(text: str) -> OutputSafetyAssessment:
    cleaned, removed_url = clean_voice_output(text)
    lower_text = cleaned.lower()
    needs_disclaimer = DISCLAIMER.lower() not in lower_text
    unsafe_claim_detected = any(pattern.search(cleaned) for pattern in _UNSAFE_OUTPUT_PATTERNS)

    return OutputSafetyAssessment(
        safe=not unsafe_claim_detected,
        needs_disclaimer=needs_disclaimer,
        removed_url=removed_url,
        unsafe_claim_detected=unsafe_claim_detected,
    )


def missing_disclaimer_suffix(text: str) -> str:
    if DISCLAIMER.lower() in (text or "").lower():
        return ""
    return " " + DISCLAIMER
