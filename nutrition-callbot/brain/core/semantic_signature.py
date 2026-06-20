from __future__ import annotations

import re


_AUDIENCE_PATTERNS = {
    "pregnant": [r"\bmang thai\b", r"\bthai phụ\b", r"\bbà bầu\b", r"\bmẹ bầu\b"],
    "breastfeeding": [r"\bcho con bú\b", r"\bmẹ sau sinh\b"],
    "infant": [r"\btrẻ sơ sinh\b", r"\bdưới 1 tuổi\b", r"\băn dặm\b"],
    "child": [r"\btrẻ em\b", r"\btrẻ nhỏ\b", r"\bbé\b", r"\bcon tôi\b"],
    "elderly": [r"\bngười cao tuổi\b", r"\bngười già\b"],
}

_CONDITION_PATTERNS = {
    "diabetes_type_1": [r"\bđái tháo đường (týp|type) 1\b", r"\btiểu đường (týp|type) 1\b"],
    "diabetes_type_2": [r"\bđái tháo đường (týp|type) 2\b", r"\btiểu đường (týp|type) 2\b"],
    "diabetes": [r"\bđái tháo đường\b", r"\btiểu đường\b"],
    "hypertension": [r"\btăng huyết áp\b", r"\bhuyết áp cao\b"],
    "obesity": [r"\bbéo phì\b", r"\bthừa cân\b"],
    "pregnancy": [r"\bmang thai\b", r"\bthai kỳ\b"],
    "kidney": [r"\bbệnh thận\b", r"\bsuy thận\b"],
    "cardiovascular": [r"\btim mạch\b", r"\bcholesterol\b"],
    "stomach": [r"\bdạ dày\b", r"\btrào ngược\b"],
}

_INTENT_PATTERNS = {
    "avoid_foods": [r"\bkiêng\b", r"\bkhông nên ăn\b", r"\btránh\b", r"\bcó ăn được\b"],
    "recommended_foods": [r"\bnên ăn gì\b", r"\băn gì\b", r"\blựa chọn thực phẩm\b", r"\bthực phẩm nào\b"],
    "dosage": [r"\bbao nhiêu\b", r"\bliều lượng\b", r"\bkhẩu phần\b", r"\bmỗi ngày\b"],
    "benefits": [r"\btác dụng\b", r"\blợi ích\b", r"\bcó tốt không\b"],
    "deficiency": [r"\bthiếu\b", r"\bbổ sung\b"],
    "meal_plan": [r"\bthực đơn\b", r"\bchế độ ăn\b", r"\băn uống thế nào\b", r"\băn ra sao\b"],
}


def _first_match(text: str, groups: dict[str, list[str]], default: str) -> str:
    for label, patterns in groups.items():
        if any(re.search(pattern, text) for pattern in patterns):
            return label
    return default


def semantic_signature(query: str) -> dict[str, str]:
    text = " ".join((query or "").lower().split())
    return {
        "audience": _first_match(text, _AUDIENCE_PATTERNS, "general"),
        "condition": _first_match(text, _CONDITION_PATTERNS, "general"),
        "intent": _first_match(text, _INTENT_PATTERNS, "general_information"),
    }


def signature_key(signature: dict[str, str]) -> str:
    return "|".join(
        f"{field}={signature.get(field, 'general')}"
        for field in ("audience", "condition", "intent")
    )
