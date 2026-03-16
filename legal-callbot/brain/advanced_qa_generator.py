import json
import random
import asyncio
from pathlib import Path
from sklearn.model_selection import train_test_split
import google.generativeai as genai
from config import config
import re

# Cấu hình API LLM từ env / config
genai.configure(api_key=config.gemini_api_key)
# Khuyến khích dùng model 2.5 flash như config của bạn (hoặc 1.5 flash)
model_name = getattr(config, "gemini_model", "gemini-2.5-flash")
model = genai.GenerativeModel(model_name)

SYSTEM_PROMPT = """Bạn là chuyên gia thiết kế đề thi và kiểm tra pháp lý Việt Nam cấp cao. 
Nhiệm vụ của bạn là dựa vào [CĂN CỨ PHÁP LÝ] được cung cấp dưới đây, tạo ra chính xác 3 cặp Hỏi-Đáp (Q&A) theo 3 mức độ khác nhau. 

YÊU CẦU CHO TỪNG BẬC ĐỘ KHÓ:
1. "easy" (Dễ): Câu hỏi hỏi thẳng vào kiến thức cơ bản nhất của điều luật. Không cần tình huống phức tạp.
2. "medium" (Trung bình): Câu hỏi phải là MỘT TÌNH HUỐNG đời sống thực tế (có anh A, chị B, thẻ công ty X...). Người hỏi đang gặp rắc rối và cần tư vấn.
   🔴 LƯU Ý CHỐNG ẢO GIÁC: Tình huống bịa ra PHẢI KHỚP TUYỆT ĐỐI với Chủ đề/Đề mục và cơ quan pháp lý của Điều luật. Tuyệt đối KHÔNG ghép sai lĩnh vực (Ví dụ: luật thuộc Bộ TNMT / monre.gov.vn thì không được hỏi về Đăng ký kinh doanh của Sở KHĐT).
3. "hard" (Khó): Tình huống éo le, lắt léo, đánh vào các trường hợp "trừ trường hợp", "ngoại lệ", điều kiện ràng buộc khắt khe hoặc thời hạn giáp ranh có quy định trong điều luật này.

YÊU CẦU CÂU TRẢ LỜI CỦA AI (output):
- Trả lời đúng trọng tâm.
- PHẢI trích dẫn rõ tên Điều, chương, tên văn bản (áp dụng dữ liệu được cung cấp).
- 🔴 KHÔNG ĐƯỢC lặp lại các thông tin sai lệch hoặc phi logic nếu tình huống (instruction) vô tình nhắc tới. Hãy đính chính theo đúng CĂN CỨ PHÁP LÝ nếu cần.

TRẢ VỀ ĐỊNH DẠNG JSON TUYỆT ĐỐI NHƯ SAU, KHÔNG CÓ BẤT KỲ COMMENT HAY MARKDOWN NÀO KHÁC:
{
  "easy": { "question": "...", "answer": "..." },
  "medium": { "question": "...", "answer": "..." },
  "hard": { "question": "...", "answer": "..." }
}"""

async def generate_triplet_qa(law_item):
    """Gọi LLM sinh 3 câu QA cho 1 Điều luật"""
    # Lọc bỏ các điều quá ngắn
    if len(law_item.get('noidung', '')) < 50:
        return None

    context = law_item.get('llm_context')
    if not context:
        context = (
            f"Chủ đề/Đề mục: {law_item.get('chude', '')} - {law_item.get('demuc', '')}\n"
            f"Thuộc chương: {law_item.get('chuong', '')}\n"
            f"Tên điều: {law_item.get('ten', '')}\n"
            f"Nội dung: {law_item.get('noidung', '')}\n"
            f"Văn bản: {law_item.get('vbqppl', '')}"
        )
    
    prompt = f"{SYSTEM_PROMPT}\n\n[CĂN CỨ PHÁP LÝ]\n{context}"
    
    try:
        response = await model.generate_content_async(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.8
            )
        )
        
        # Clean markdown if present
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
            
        result = json.loads(raw_text.strip())
        
        # Format lại dữ liệu output
        qa_pairs = []
        for difficulty, qa in result.items():
            if difficulty in ["easy", "medium", "hard"] and isinstance(qa, dict):
                qa_pairs.append({
                    "instruction": qa.get("question", ""),
                    "output": qa.get("answer", ""),
                    "difficulty": difficulty,
                    "metadata": {
                        "source_id": law_item.get("mapc", ""),
                        "topic": law_item.get("chude", "Unknown"),
                        "subtopic": law_item.get("demuc", "Unknown")
                    }
                })
        return qa_pairs
    except Exception as e:
        print(f"⚠️ Lỗi sinh data ở điều {law_item.get('mapc', 'Unknown')}: {e}")
        return None

def stratified_split_laws(data, train_size=10, test_size=2):
    """Chia tập điều luật gốc sao cho Chủ đề (Topic) được phân bổ đều"""
    valid_data = [d for d in data if len(d.get('noidung', '')) > 50]
    
    topics = [d.get('chude', 'Other') for d in valid_data]
    topic_counts = {t: topics.count(t) for t in set(topics)}
    safe_topics = [t if topic_counts[t] > 1 else 'Rare' for t in topics]

    needed_total = train_size + test_size
    if needed_total > len(valid_data):
        needed_total = len(valid_data)
        train_size = int(needed_total * 0.85)
        test_size = needed_total - train_size

    if needed_total < len(set(safe_topics)):
        print(f"⚠️ Cảnh báo: training size quá nhỏ để chia phân tầng (số lượng classes = {len(set(safe_topics))}). Chuyển sang random split.")
        import random
        random.seed(42)
        small_sample = random.sample(valid_data, needed_total)
        return train_test_split(small_sample, test_size=test_size, random_state=42)

    sampled_data, _, sampled_topics, _ = train_test_split(
        valid_data, safe_topics, 
        train_size=needed_total, 
        stratify=safe_topics, 
        random_state=42
    )

    train_docs, test_docs = train_test_split(
        sampled_data, 
        test_size=test_size, 
        stratify=sampled_topics,
        random_state=42
    )
    return train_docs, test_docs

async def generate_dataset(docs, output_filename, batch_size=5):
    """Hàm chạy Async sinh DATA và lưu file"""
    output_path = Path(f"data/{output_filename}")
    total_generated = 0
    
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            tasks = [generate_triplet_qa(doc) for doc in batch]
            results = await asyncio.gather(*tasks)
            
            for res_list in results:
                if res_list:
                    for qa in res_list:
                        f.write(json.dumps(qa, ensure_ascii=False) + "\\n")
                        total_generated += 1
            
            print(f"🔄 Đang tiến hành {output_filename}: Xong {min(i+batch_size, len(docs))} gốc -> Đã ráp đc {total_generated} QA")
            await asyncio.sleep(2) # rate limit API

    print(f"✅ Hoàn thành! Đã tạo được {total_generated} QA cho bộ {output_filename}")

async def main():
    print("1. Đọc dữ liệu Pháp luật gốc...")
    try:
        with open("data/law_data_clean.json", "r", encoding="utf-8") as f:
            full_data = json.load(f)
    except FileNotFoundError:
        print("❌ Không tìm thấy file data/law_data_clean.json! Vui lòng chạy scripts/preprocess_data.py trước.")
        return

    print("2. Tách bộ Dữ liệu Train/Test thông minh (Stratified Split)...")
    # Sử dụng cấu hình Production: 
    # train_size=2000 -> sẽ tạo ra khoảng 6000 mẫu QA (3 độ khó) dùng để Fine-tune CỰC TỐT
    # test_size=200 -> sẽ tạo ra khoảng 600 mẫu QA (200 * 3) đủ lớn thống kê Latency Stress-test
    train_docs, test_docs = stratified_split_laws(full_data, train_size=2000, test_size=200)
    print(f"Chuẩn bị xử lý {len(train_docs)} bài cho Train và {len(test_docs)} bài cho Test.")

    print("\\n3. Bắt đầu sinh bộ TEST SET (test_qa_dataset.jsonl)...")
    await generate_dataset(test_docs, "test_qa_dataset.jsonl", batch_size=2)

    print("\\n4. Bắt đầu sinh bộ TRAIN SET (train_qa_dataset.jsonl)...")
    await generate_dataset(train_docs, "train_qa_dataset.jsonl", batch_size=5)

if __name__ == "__main__":
    asyncio.run(main())
