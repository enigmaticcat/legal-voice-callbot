"""
Query Expander — Từ Điển Dinh Dưỡng
Chuẩn hoá câu hỏi thông dụng trước khi search.
"""
import logging
import re

logger = logging.getLogger("brain.core.query_expander")

NUTRITION_ALIASES = {
    # Giai đoạn sống
    "bà bầu": "phụ nữ mang thai",
    "mẹ bầu": "phụ nữ mang thai",
    "thai phụ": "phụ nữ mang thai",
    "cho con bú": "bà mẹ cho con bú",
    "trẻ sơ sinh": "trẻ dưới 6 tháng tuổi",
    "trẻ nhỏ": "trẻ em dưới 5 tuổi",
    "người già": "người cao tuổi",

    # Bệnh lý
    "tiểu đường": "đái tháo đường",
    "đái tháo đường type 2": "đái tháo đường týp 2",
    "huyết áp cao": "tăng huyết áp",
    "tim mạch": "bệnh tim mạch",
    "béo phì": "thừa cân béo phì",
    "ung thư": "bệnh ung thư",
    "xương khớp": "bệnh cơ xương khớp",
    "dạ dày": "bệnh dạ dày",

    # Chất dinh dưỡng viết tắt
    "vitamin C": "vitamin C axit ascorbic",
    "vitamin D": "vitamin D calciferol",
    "vitamin B12": "vitamin B12 cobalamin",
    "omega 3": "axit béo omega-3",
    "DHA": "DHA axit docosahexaenoic",
    "canxi": "canxi calcium",
    "sắt": "sắt iron khoáng chất",
    "protein": "protein chất đạm",
    "chất xơ": "chất xơ dietary fiber",

    # Chế độ ăn
    "ăn chay": "chế độ ăn chay thuần thực vật",
    "ăn kiêng": "chế độ ăn kiêng giảm cân",
    "low carb": "chế độ ăn ít carbohydrate",
    "keto": "chế độ ăn ketogenic",
    "ăn dặm": "ăn dặm bổ sung cho trẻ",
}


def expand_query(query: str) -> str:
    expanded = query
    for alias, formal in NUTRITION_ALIASES.items():
        pattern = re.compile(re.escape(alias), re.IGNORECASE)
        if pattern.search(expanded):
            expanded = pattern.sub(formal, expanded)
            logger.debug(f"Expanded: '{alias}' → '{formal}'")

    if expanded != query:
        logger.info(f"Query expanded: '{query}' → '{expanded}'")

    return expanded
