LEGAL_SYSTEM_PROMPT = """Bạn là trợ lý tư vấn pháp luật Việt Nam qua giọng nói. Tuân thủ:

1. **Trích dẫn căn cứ**: Nêu số Điều, Khoản, tên văn bản trước khi phân tích.
2. **Ngắn gọn, dễ hiểu**: Trả lời cho công dân thường, không dùng thuật ngữ phức tạp.
3. **Trung thực**: Nếu không chắc → nói rõ "Tôi không có thông tin chính xác".
4. **Disclaimer**: Kết thúc bằng "Đây chỉ là tham khảo, không thay thế tư vấn pháp lý chính thức."
5. **Giọng nói**: Câu trả lời sẽ được đọc thành giọng nói, nên dùng câu ngắn, tránh bullet points dài."""

FEW_SHOT_EXAMPLES = [
    {
        "question": "Vượt đèn đỏ bị phạt bao nhiêu?",
        "answer": (
            "Theo Điều 5, Khoản 4a Nghị định 100/2019/NĐ-CP, "
            "xe ô tô vượt đèn đỏ bị phạt từ 4 đến 6 triệu đồng. "
            "Xe máy theo Khoản 3 cùng Điều, phạt từ 800 nghìn đến 1 triệu đồng. "
            "Đây chỉ là tham khảo, không thay thế tư vấn pháp lý chính thức."
        ),
    },
    {
        "question": "Uống bia rồi lái xe bị xử phạt như nào?",
        "answer": (
            "Theo Điều 5, Khoản 6 Nghị định 100/2019/NĐ-CP: "
            "Nồng độ cồn dưới 50mg thì phạt 2 đến 3 triệu cho xe máy, "
            "6 đến 8 triệu cho ô tô. "
            "Từ 50 đến 80mg phạt nặng hơn, trên 80mg phạt tối đa 40 triệu với ô tô. "
            "Đây chỉ là tham khảo, không thay thế tư vấn pháp lý chính thức."
        ),
    },
    {
        "question": "Đất không có sổ đỏ thì bán được không?",
        "answer": (
            "Theo Điều 188 Luật Đất đai 2013, muốn chuyển nhượng đất phải có "
            "Giấy chứng nhận quyền sử dụng đất. Nếu chưa có sổ đỏ, "
            "bạn cần làm thủ tục cấp Giấy chứng nhận trước khi bán. "
            "Đây chỉ là tham khảo, không thay thế tư vấn pháp lý chính thức."
        ),
    },
    {
        "question": "Nghỉ việc có được trợ cấp thất nghiệp không?",
        "answer": (
            "Theo Điều 49 Luật Việc làm 2013, bạn được hưởng trợ cấp thất nghiệp "
            "nếu đã đóng bảo hiểm thất nghiệp ít nhất 12 tháng trong 24 tháng "
            "trước khi mất việc. Bạn cần đăng ký tại Trung tâm dịch vụ việc làm "
            "trong vòng 3 tháng kể từ ngày nghỉ việc. "
            "Đây chỉ là tham khảo, không thay thế tư vấn pháp lý chính thức."
        ),
    },
]


def build_prompt(
    query: str,
    legal_context: str = "",
    conversation_history: list = None,
) -> str:
    parts = []

    parts.append("Ví dụ tư vấn:")
    for ex in FEW_SHOT_EXAMPLES[:3]:  
        parts.append(f"Hỏi: {ex['question']}")
        parts.append(f"Đáp: {ex['answer']}\n")

    if legal_context:
        parts.append("---")
        parts.append("Căn cứ pháp lý liên quan:")
        parts.append(legal_context)
        parts.append("---")
        parts.append("Hãy trả lời DỰA TRÊN các Điều luật trên. Trích dẫn chính xác số Điều, Khoản.")
    else:
        parts.append("(Chưa có dữ liệu RAG — trả lời từ kiến thức chung của bạn.)")

    if conversation_history:
        parts.append("\nLịch sử hội thoại:")
        for turn in conversation_history[-6:]:  
            role = "Người dùng" if turn["role"] == "user" else "Bot"
            text = turn["text"]
            if turn.get("interrupted"):
                text += " [bị ngắt giữa chừng]"
            parts.append(f"  {role}: {text}")

    parts.append(f"\nCâu hỏi hiện tại: {query}")

    return "\n".join(parts)
