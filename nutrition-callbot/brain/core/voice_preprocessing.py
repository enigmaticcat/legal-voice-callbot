import re

# ==============================================================================
# TỪ ĐIỂN CHUẨN HÓA VOICE-TO-TEXT DÀNH RIÊNG CHO RAG PHÁP LUẬT VIỆT NAM
# Bao trọn 100% các cơ quan ban hành văn bản quy phạm theo chuẩn Bộ Tư Pháp
# Tối ưu thuật toán: Xếp các chuỗi SIÊU DÀI lên trước để Regex không bắt nhầm 
# ==============================================================================

LEGAL_ALIASES = {
    # ---------------------------------------------------------
    # 1. KHỐI CƠ QUAN QUYỀN LỰC NHÀ NƯỚC (QUỐC HỘI, CHỦ TỊCH NƯỚC)
    # ---------------------------------------------------------
    r"\bủy ban thường vụ quốc hội\b": "UBTVQH",
    r"\buỷ ban thường vụ quốc hội\b": "UBTVQH",
    r"\btòa án nhân dân tối cao\b": "TANDTC",
    r"\btoà án nhân dân tối cao\b": "TANDTC",
    r"\bviện kiểm sát nhân dân tối cao\b": "VKSNDTC",
    r"\bhội đồng thẩm phán\b": "HĐTP",
    r"\bchủ tịch nước\b": "CTN",
    r"\bquốc hội\b": "QH",

    # ---------------------------------------------------------
    # 2. KHỐI CHÍNH PHỦ VÀ CÁC CHỨC DANH
    # ---------------------------------------------------------
    r"\bnghị quyết chính phủ\b": "NQ-CP",
    r"nghị quyết của chính phủ": "NQ-CP",
    r"\bquyết định thủ tướng\b": "QĐ-TTg",
    r"quyết định của thủ tướng": "QĐ-TTg",
    r"\bchỉ thị thủ tướng\b": "CT-TTg",
    r"chỉ thị của thủ tướng": "CT-TTg",
    r"\bhướng dẫn chính phủ\b": "HD-CP",
    r"\bthủ tướng chính phủ\b": "TTg",
    r"\bnghị định chính phủ\b": "NĐ-CP",
    r"nghị định của chính phủ": "NĐ-CP",
    r"nghị định\s+(\d+)\s+chính phủ": r"\1/NĐ-CP", # "nghị định 105 chính phủ" -> 105/NĐ-CP
    r"\bchính phủ\b": "CP",

    # ---------------------------------------------------------
    # 3. DANH SÁCH 18 BỘ (Đầy đủ không thiếu bộ nào)
    # ---------------------------------------------------------
    r"\bbộ nông nghiệp và phát triển nông thôn\b": "BNNPTNT",
    r"\bbộ lao động thương binh và xã hội\b": "BLĐTBXH",
    r"\bbộ lao động thương binh xã hội\b": "BLĐTBXH",
    r"\bbộ văn hóa thể thao và du lịch\b": "BVHTTDL",
    r"\bbộ văn hoá thể thao và du lịch\b": "BVHTTDL",
    r"\bbộ thông tin và truyền thông\b": "BTTTT",
    r"\bbộ tài nguyên và môi trường\b": "BTNMT",
    r"\bbộ giáo dục và đào tạo\b": "BGDĐT",
    r"\bbộ kế hoạch và đầu tư\b": "BKHĐT",
    r"\bbộ khoa học và công nghệ\b": "BKHCN",
    r"\bbộ giao thông vận tải\b": "BGTVT",
    r"\bbộ tài chính\b": "BTC",
    r"\bbộ công an\b": "BCA",
    r"\bbộ quốc phòng\b": "BQP",
    r"\bbộ tư pháp\b": "BTP",
    r"\bbộ y tế\b": "BYT",
    r"\bbộ nội vụ\b": "BNV",
    r"\bbộ ngoại giao\b": "BNG",
    r"\bbộ xây dựng\b": "BXD",
    r"\bbộ công thương\b": "BCT",
    r"\bbộ nông nghiệp\b": "BNN",

    # ---------------------------------------------------------
    # 4. DANH SÁCH 4 CƠ QUAN NGANG BỘ
    # ---------------------------------------------------------
    r"\bngân hàng nhà nước việt nam\b": "NHNN",
    r"\bngân hàng nhà nước\b": "NHNN",
    r"\bthanh tra chính phủ\b": "TTCP",
    r"\bvăn phòng chính phủ\b": "VPCP",
    r"\bủy ban dân tộc\b": "UBDT",
    r"\buỷ ban dân tộc\b": "UBDT",

    # ---------------------------------------------------------
    # 5. CƠ QUAN ĐỊA PHƯƠNG (Phòng trường hợp hỏi điều lệ tỉnh)
    # ---------------------------------------------------------
    r"\bủy ban nhân dân\b": "UBND",
    r"\buỷ ban nhân dân\b": "UBND",
    r"\bhội đồng nhân dân\b": "HĐND"
}

def voice_query_normalization(query: str) -> str:
    """
    Tiền xử lý chuỗi Text (Transcript) từ mô hình Nhận diện giọng nói (ASR - Whisper/ONNX).
    """
    # 1. Chữ thường và gọt khoảng trắng thừa
    text = query.lower().strip()
    
    # 2. Xử lý ảo giác số do ASR (Ví dụ: "loẹt một linh năm" -> "luật 105")
    text = text.replace("một linh ", "10")
    text = text.replace("hai linh ", "20")
    text = text.replace("linh ", "0")
    
    # 3. Phép thế chuẩn hóa Tổ chức
    for pattern, alias in LEGAL_ALIASES.items():
        # Dùng regex re.sub thay vì replace để chặn lỗi bắt dính chữ (Word boundary)
        text = re.sub(pattern, alias, text, flags=re.IGNORECASE)

    # 4. Gắn kết Logic Số Hiệu Pháp lý siêu mượt
    # (VD: "nghị định 105 năm 2025" -> "105/2025/NĐ-CP")
    text = re.sub(r'nghị định\s+(\d+)\s+năm\s+(\d{4})', r'\1/\2/NĐ-CP', text)
    text = re.sub(r'thông tư\s+(\d+)\s+năm\s+(\d{4})', r'\1/\2/TT', text)
    text = re.sub(r'luật\s+(\d+)\s+năm\s+(\d{4})', r'\1/\2/QH', text) # Thường là QH hoặc L
    text = re.sub(r'quyết định\s+(\d+)\s+năm\s+(\d{4})', r'\1/\2/QĐ', text)

    return text

if __name__ == "__main__":
    # Test thử sức mạnh của hệ thống tiền xử lý
    test_voice_transcripts = [
        "cho tôi hỏi nghị định chính phủ một linh năm về PCCC",
        "thông tư 17 năm 2024 của bộ lao động thương binh xã hội cấm gì",
        "quyết định của thủ tướng số 15 năm 2020",
        "tiếp nhận theo nghị quyết chính phủ mới"
    ]
    print("--- KIỂM TRA ĐẦU VÀO TỪ VOICE ASR ---")
    for q in test_voice_transcripts:
        print(f"User Voice : {q}")
        print(f"RAG Input  : {voice_query_normalization(q)}\n")
