import re

try:
    from .query_expander import expand_query
except ImportError:
    from core.query_expander import expand_query


_COMMON_ASR_FIXES = {
    "omega ba": "omega 3",
    "omega- ba": "omega 3",
    "can xi": "canxi",
    "tiểu đường type hai": "tiểu đường type 2",
    "đái tháo đường type hai": "đái tháo đường type 2",
    "vitamin dê": "vitamin D",
    "vitamin đê": "vitamin D",
    "vitamin xê": "vitamin C",
}


def voice_query_normalization(query: str) -> str:
    """Normalize noisy ASR text before nutrition RAG search."""
    text = query.lower().strip()
    text = re.sub(r"\s+", " ", text)

    for source, target in _COMMON_ASR_FIXES.items():
        text = text.replace(source, target.lower())

    return expand_query(text)


if __name__ == "__main__":
    samples = [
        "omega ba có tác dụng gì",
        "bà bầu cần bổ sung can xi không",
        "người bị tiểu đường type hai nên ăn gì",
        "vitamin xê có trong thực phẩm nào",
    ]
    print("--- KIỂM TRA ĐẦU VÀO TỪ VOICE ASR ---")
    for sample in samples:
        print(f"User Voice : {sample}")
        print(f"RAG Input  : {voice_query_normalization(sample)}\n")
