"""
Query Expander — Từ Điển Pháp Lý
Mở rộng câu hỏi bằng từ đồng nghĩa pháp lý trước khi search.
"""
import logging
import re

logger = logging.getLogger("brain.core.query_expander")

# ─── Từ điển alias pháp lý — 0ms overhead ────────────────
LEGAL_ALIASES = {
    # Giao thông
    "sổ đỏ": "Giấy chứng nhận quyền sử dụng đất",
    "GPLX": "Giấy phép lái xe",
    "bằng lái": "Giấy phép lái xe",
    "NĐ100": "Nghị định 100/2019/NĐ-CP",
    "NĐ 100": "Nghị định 100/2019/NĐ-CP",
    "nghị định 100": "Nghị định 100/2019/NĐ-CP",
    "uống bia lái xe": "nồng độ cồn điều khiển phương tiện",
    "uống rượu lái xe": "nồng độ cồn điều khiển phương tiện",
    "nhậu lái xe": "nồng độ cồn điều khiển phương tiện",
    "say xỉn lái xe": "nồng độ cồn điều khiển phương tiện",
    "vượt đèn đỏ": "phạt tiền xe mô tô xe gắn máy không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
    "đi ngược chiều": "điều khiển xe đi ngược chiều",
    "không đội mũ": "không đội mũ bảo hiểm",

    # Đất đai
    "sang tên nhà": "chuyển nhượng quyền sử dụng đất",
    "mua bán đất": "chuyển nhượng quyền sử dụng đất",
    "tách thửa": "tách thửa đất",

    # Lao động
    "nghỉ việc": "chấm dứt hợp đồng lao động",
    "đuổi việc": "sa thải người lao động",
    "bảo hiểm xã hội": "bảo hiểm xã hội bắt buộc",
    "BHXH": "bảo hiểm xã hội bắt buộc",
    "BHYT": "bảo hiểm y tế",

    # Dân sự
    "di chúc": "di chúc thừa kế",
    "chia tài sản": "phân chia tài sản",
    "ly hôn": "ly hôn giải quyết quan hệ hôn nhân",
}


def expand_query(query: str) -> str:
    """
    Thay thế các từ viết tắt/thông dụng bằng thuật ngữ pháp lý chính thức.

    Examples:
        "sổ đỏ" → "Giấy chứng nhận quyền sử dụng đất"
        "bằng lái" → "Giấy phép lái xe"
        "nghỉ việc có được trợ cấp" → "chấm dứt hợp đồng lao động có được trợ cấp"
    """
    expanded = query
    for alias, formal in LEGAL_ALIASES.items():
        pattern = re.compile(re.escape(alias), re.IGNORECASE)
        if pattern.search(expanded):
            expanded = pattern.sub(formal, expanded)
            logger.debug(f"Expanded: '{alias}' → '{formal}'")

    if expanded != query:
        logger.info(f"Query expanded: '{query}' → '{expanded}'")

    return expanded
