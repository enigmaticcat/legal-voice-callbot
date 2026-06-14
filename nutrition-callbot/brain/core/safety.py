from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyAssessment:
    triggered: bool
    category: str = "normal"
    message: str = ""


_EMERGENCY_PATTERNS = [
    r"\bkhó thở\b",
    r"\bđau ngực\b",
    r"\bngất\b",
    r"\bco giật\b",
    r"\bđột quỵ\b",
    r"\btai biến\b",
    r"\bxuất huyết\b",
    r"\bnôn ra máu\b",
    r"\btự tử\b",
    r"\btự hại\b",
]

_CLINICAL_DECISION_PATTERNS = [
    r"\bkê đơn\b",
    r"\buống thuốc gì\b",
    r"\bliều thuốc\b",
    r"\btăng liều\b",
    r"\bgiảm liều\b",
    r"\bngừng thuốc\b",
    r"\bchẩn đoán\b",
    r"\btôi bị bệnh gì\b",
    r"\bkết quả xét nghiệm\b",
]

_OUT_OF_SCOPE_PATTERNS = [
    r"\bđầu tư\b",
    r"\bchứng khoán\b",
    r"\btiền ảo\b",
    r"\bviết code\b",
    r"\blập trình\b",
]


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def assess_safety(query: str) -> SafetyAssessment:
    text = (query or "").lower().strip()
    if not text:
        return SafetyAssessment(
            triggered=True,
            category="empty_query",
            message="Tôi chưa nghe rõ câu hỏi của bạn. Bạn vui lòng nói lại ngắn gọn hơn nhé.",
        )

    if _matches_any(text, _EMERGENCY_PATTERNS):
        return SafetyAssessment(
            triggered=True,
            category="medical_emergency",
            message=(
                "Chào bạn, các dấu hiệu bạn mô tả có thể cần được xử lý trực tiếp. "
                "Bạn nên liên hệ cấp cứu hoặc đến cơ sở y tế gần nhất ngay, đặc biệt nếu triệu chứng đang diễn ra hoặc nặng lên. "
                "Tôi không thể thay thế bác sĩ trong tình huống khẩn cấp."
            ),
        )

    if _matches_any(text, _CLINICAL_DECISION_PATTERNS):
        return SafetyAssessment(
            triggered=True,
            category="clinical_decision",
            message=(
                "Chào bạn, tôi có thể hỗ trợ thông tin dinh dưỡng chung, nhưng không thể chẩn đoán, kê đơn hoặc điều chỉnh thuốc. "
                "Bạn nên trao đổi trực tiếp với bác sĩ hoặc chuyên gia dinh dưỡng, nhất là khi có bệnh nền, đang dùng thuốc hoặc có kết quả xét nghiệm bất thường."
            ),
        )

    if _matches_any(text, _OUT_OF_SCOPE_PATTERNS):
        return SafetyAssessment(
            triggered=True,
            category="out_of_scope",
            message=(
                "Chào bạn, hệ thống này chỉ hỗ trợ các câu hỏi về dinh dưỡng và chế độ ăn. "
                "Bạn vui lòng đặt câu hỏi liên quan đến thực phẩm, vi chất, khẩu phần hoặc chế độ ăn phù hợp với tình trạng sức khỏe."
            ),
        )

    return SafetyAssessment(triggered=False)
