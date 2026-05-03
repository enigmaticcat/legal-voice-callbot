NUTRITION_SYSTEM_PROMPT = """Bạn là chuyên gia tư vấn dinh dưỡng qua giọng nói. Tuân thủ:

1. **Dựa vào tài liệu**: Trả lời dựa trên thông tin dinh dưỡng được cung cấp. Không trích dẫn tên nguồn hay URL trong câu trả lời.
2. **Phong cách bác sĩ**: Bắt đầu bằng "Chào bạn,", tư vấn như chuyên gia dinh dưỡng.
3. **Ngắn gọn, dễ nghe**: Câu trả lời sẽ được đọc thành giọng nói — tối đa 80 từ, dùng câu ngắn, không dùng bullet points hay danh sách. Sau mỗi dấu chấm hoặc dấu phẩy phải có dấu cách.
4. **Trung thực**: Nếu không có thông tin → nói rõ "Tôi không có thông tin về vấn đề này".
5. **Disclaimer**: Kết thúc bằng "Để được tư vấn chính xác, bạn nên gặp bác sĩ dinh dưỡng."
"""

# Few-shot examples lấy nguyên văn từ benhvienthucuc.vn (thucuc_qa.jsonl).
# Chỉ xử lý ký tự \r\n → \n và các \n giữa câu do HTML tag crawl → space.
FEW_SHOT_EXAMPLES = [
    {
        "question": (
            "Chào bác sĩ,\n"
            "Tôi vừa mổ tuyến giáp xong, liệu sau mổ tuyến giáp có ăn được hải sản không ạ? "
            "Tôi nghe nói hải sản giàu i-ốt, nhưng không chắc lượng i-ốt có ảnh hưởng gì đến sức khỏe sau mổ hay không? "
            "Bác sĩ có thể tư vấn giúp tôi nên ăn loại hải sản nào và cần kiêng gì không ạ?"
        ),
        "answer": (
            "Chào bạn,\n"
            "Sau phẫu thuật tuyến giáp, rất nhiều người bệnh lo lắng về chế độ ăn uống, đặc biệt là câu hỏi mổ tuyến giáp có ăn được hải sản không? "
            "Đây là băn khoăn hoàn toàn chính đáng, bởi tuyến giáp liên quan trực tiếp đến chuyển hóa và việc sử dụng hormone lâu dài.\n"
            "Về vấn đề này thì câu trả lời là có, trong hầu hết các trường hợp, vì hải sản là nhóm thực phẩm giàu dinh dưỡng, giúp cơ thể phục hồi tốt hơn. "
            "Hải sản như tôm, cua, cá hồi, cá thu, cá ngừ cung cấp iốt tự nhiên, omega-3, kẽm, selen và protein chất lượng cao giúp tái tạo mô sau mổ.\n"
            "Người bệnh có thể ăn 2 – 3 bữa hải sản mỗi tuần, ưu tiên loại tươi, đánh bắt tự nhiên. "
            "Nếu còn đau họng hoặc khó nuốt, nên chế biến mềm như cháo cá, súp cua hay cá hấp xé nhỏ.\n"
            "Với bệnh nhân ung thư tuyến giáp, câu trả lời phụ thuộc vào giai đoạn điều trị. "
            "Nếu chuẩn bị điều trị bằng iod phóng xạ (I-131), cần tuân thủ chế độ ăn ít iốt (LID) và chỉ ăn lại hải sản theo hướng dẫn của bác sĩ."
        ),
    },
    {
        "question": (
            "Thưa bác sĩ, tôi muốn hỏi có nên uống Omega 3-6-9 mỗi ngày hay không, "
            "vì tôi chỉ đang bổ sung với mục đích tăng cường sức khỏe. "
            "Nếu dùng hằng ngày thì liều lượng như thế nào là phù hợp để không gây dư thừa chất béo? "
            "Ngoài ra, khi uống Omega 3-6-9 lâu dài thì có lưu ý gì không, mong được giải đáp."
        ),
        "answer": (
            "Chào bạn,\n"
            "Có nên uống Omega 3-6-9 mỗi ngày là thắc mắc của nhiều người khi muốn bổ sung chất béo tốt để bảo vệ tim mạch và não bộ. "
            "Trên thực tế, Omega 3, 6 và 9 đều đóng vai trò quan trọng đối với sức khỏe, nhưng không phải ai cũng cần bổ sung hằng ngày. "
            "Việc dùng thường xuyên hay không còn phụ thuộc vào chế độ ăn, thể trạng và mục đích sử dụng.\n"
            "Với đa số trường hợp, việc bổ sung Omega 3-6-9 kết hợp là không thật sự cần thiết. "
            "Nếu thiếu hụt, chỉ bổ sung omega-3 (EPA và DHA) riêng lẻ thường mang lại lợi ích rõ ràng hơn cho tim mạch, não bộ và thị lực.\n"
            "Thông thường, chỉ cần 1 viên mỗi ngày đã đáp ứng nhu cầu cơ bản. "
            "Thời gian bổ sung tối thiểu khoảng 2 tháng để các axit béo tích lũy vào màng tế bào, "
            "có thể duy trì liên tục hoặc theo chu kỳ 3 – 4 tháng.\n"
            "Lưu ý: phụ nữ mang thai không nên dùng Omega-3 có chứa vitamin A. "
            "Những ai đang dùng thuốc chống đông máu cần tham khảo ý kiến bác sĩ trước."
        ),
    },
    {
        "question": (
            "Thưa bác sĩ, con tôi thường xuyên chán ăn, bữa nào cũng kéo dài. "
            "Tôi được người quen khuyên nên bổ sung kẽm cho bé, "
            "vậy có thực sự cần bổ sung kẽm cho trẻ biếng ăn không ạ?"
        ),
        "answer": (
            "Chào bạn,\n"
            "Kẽm là vi chất tham gia trực tiếp vào hoạt động của vị giác. "
            "Khi trẻ thiếu kẽm, các tế bào vị giác hoạt động kém hơn, "
            "dẫn đến ăn kém, ngậm thức ăn lâu và chậm tăng cân.\n"
            "Nếu nghi ngờ thiếu kẽm, nên cho trẻ xét nghiệm vi chất để có đánh giá chính xác. "
            "Trong khi chờ, có thể bổ sung theo liều khuyến nghị cho lứa tuổi: "
            "trẻ từ 6 tháng đến 1 tuổi cần khoảng 3 – 4mg kẽm nguyên tố mỗi ngày. "
            "Không tự ý tăng liều vì dùng quá nhiều kẽm có thể ảnh hưởng đến hấp thu các khoáng chất khác.\n"
            "Kẽm nên được bổ sung theo đợt, kéo dài vài tuần đến 2 – 3 tháng tùy tình trạng của trẻ, "
            "sau đó đánh giá lại mức độ cải thiện để quyết định có tiếp tục hay không.\n"
            "Để được tư vấn chính xác, bạn nên gặp bác sĩ dinh dưỡng."
        ),
    },
]


def build_prompt(
    query: str,
    nutrition_context: str = "",
    conversation_history: list = None,
) -> str:
    parts = []

    parts.append("Ví dụ tư vấn:")
    for ex in FEW_SHOT_EXAMPLES[:2]:
        parts.append(f"Hỏi: {ex['question']}")
        parts.append(f"Đáp: {ex['answer']}\n")

    if nutrition_context:
        parts.append("---")
        parts.append("Tài liệu dinh dưỡng liên quan:")
        parts.append(nutrition_context)
        parts.append("---")
        parts.append("Hãy trả lời DỰA TRÊN các tài liệu trên. Không được nhắc tên nguồn hay URL trong câu trả lời.")
    else:
        parts.append("(Chưa có dữ liệu RAG — trả lời từ kiến thức dinh dưỡng chung của bạn.)")

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
