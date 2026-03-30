# Phân Tích Chuyên Sâu Domain Dinh Dưỡng (Nutrition) cho CallBot

Việc chuyển đổi từ pháp luật sang dinh dưỡng không chỉ là thay đổi dữ liệu mà còn là thay đổi **phương pháp tương tác**, **mục tiêu người dùng** và **rủi ro đạo đức/an toàn**. Dưới đây là phân tích chi tiết về việc xây dựng AI CallBot trong lĩnh vực Dinh dưỡng tại Việt Nam.

---

## 1. Phân Loại Nhu Cầu Người Dùng (Use Cases)

Khác với luật pháp (người dùng thường hỏi "có được làm X không?", "phạt bao nhiêu?"), trong dinh dưỡng, nhu cầu vô cùng đa dạng và mang tính cá nhân hóa cao.

| Nhóm nhu cầu | Ví dụ câu hỏi phổ biến | Mức độ khó & Rủi ro |
| :--- | :--- | :--- |
| **Dinh dưỡng giảm cân / tăng cân** | "Bữa sáng ăn gì để giảm mỡ bụng?", "Thực đơn 1500 calo rẻ tiền?" | Thấp - Trung bình |
| **Dinh dưỡng mẹ và bé** | "Bé 6 tháng tuổi ăn dặm như thế nào?", "Bầu tháng thứ 4 ăn gì để vào con không vào mẹ?" | Trung bình (cần cẩn trọng với trẻ sơ sinh) |
| **Dinh dưỡng thể thao** | "Trước khi chạy bộ nên ăn gì?", "Cần nạp bao nhiêu protein để tăng cơ?" | Thấp |
| **Kiến thức thực phẩm** | "Sầu riêng có bao nhiêu calo?", "Uống sữa đậu nành có tốt không?" | Thấp |
| **Dinh dưỡng bệnh lý (Mức độ lâm sàng)** | "Bị tiểu đường tuýp 2 thì ăn trái cây được không?", "Thực đơn cho người suy thận độ 3?" | **Rất cao (Rủi ro sức khỏe trực tiếp)** |

---

## 2. Thách Thức Kỹ Thuật (So với Domain Pháp Luật)

### 2.1. Yêu cầu với ASR (Nhận diện giọng nói)
- **Luật:** Từ khóa dài, hàn lâm (VD: "Quyết định không khởi tố hình sự", "giấy chứng nhận quyền sử dụng đất").
- **Dinh dưỡng:** 
  - Mã đơn vị đo lường cực kỳ quan trọng: `calo`, `kcal`, `gram`, `ml`, `muỗng`, `chén`. 
  - Thuật ngữ chuyên ngành y sinh: `carbohydrate`, `protein`, `cholesterol`, `glycemic index (GI)`, `insulin`.
  - Tên thực phẩm địa phương và tiếng lóng: `bún bò huế`, `cơm tấm`, `keto`, `eat clean`, `chè`.
  - *Nhiệm vụ:* Phải bổ sung bộ từ vựng dinh dưỡng vào Language Model của Whisper.

### 2.2. Yêu cầu với RAG (Retrieval-Augmented Generation)
- **Luật:** Hybrid Search hoạt động tốt vì người dùng thường trích dẫn từ khóa có trong văn bản luật (Chương, Điều, Phạt tiền).
- **Dinh dưỡng:** Yêu cầu **Semantic Search (Tìm kiếm ngữ nghĩa)** cực mạnh. Người dùng hỏi "Ăn gì cho mát gan?", bot phải hiểu "mát gan" liên quan đến "thực phẩm hỗ trợ chức năng gan", "chất chống oxy hóa", "giải độc".
- **Cross-referencing:** Hỏi về 1 thực đơn có thể phải truy xuất từ: bảng thành phần thực phẩm + hướng dẫn calo + cảnh báo dị ứng.

### 2.3. Yêu cầu về LLM Reasoning (Khả năng suy luận tính toán)
- Bot cần có khả năng tư duy toán học cơ bản: Tính BMR (Basal Metabolic Rate), TDEE (Total Daily Energy Expenditure), BMI.
- Nếu người dùng cung cấp "Tôi cao 1m60 nặng 65kg", LLM phải tính sơ bộ được BMI để khuyến nghị (Yêu cầu config Tool/Functions calling cho LLM hoặc Prompt chặt chẽ).

---

## 3. Nguồn Dữ Liệu Khuyến Nghị (Data Sources)

Dữ liệu dinh dưỡng rất dễ vướng "rác" (Blog SEO mâu thuẫn nhau). Cần thu thập từ những nguồn **Evidence-based (Dựa trên bằng chứng)**:

1. **Bảng thành phần thực phẩm Việt Nam** (Viện Dinh Dưỡng Quốc Gia ban hành) - Rất cần thiết để bot biết 1 bát phở bao nhiêu calo.
2. **Khuyến nghị dinh dưỡng cho người Việt Nam** (Bộ Y tế) - Nguồn chính thống để lấy thông số lượng vi chất, macro hằng ngày.
3. **Các Hướng dẫn chẩn đoán và điều trị bệnh lý có phần Dinh dưỡng** của Bộ Y tế (VD: Hướng dẫn dinh dưỡng điều trị đái tháo đường, tăng huyết áp).
4. **Tài liệu từ WHO, FAO** (Phiên bản tiếng Việt hoặc dịch).

---

## 4. Rủi Ro An Toàn & Đạo Đức (Safety & Ethics)

Domain dinh dưỡng là nhóm ngành thuộc y tế / sức khỏe (YMYL - Your Money or Your Life). Rủi ro cao hơn pháp luật rất nhiều.

- **Hallucination gây hại:** Nếu LLM bảo "người suy thận nên ăn nhiều chuối" (chuối nhiều Kali - gây nguy hiểm cho người suy thận), hậu quả sẽ rất nghiêm trọng.
- **Biện pháp (Guardrails):**
  1. **Bộ lọc chủ đề bệnh lý nghiêm trọng:** Nếu phát hiện các keyword "suy thận", "ung thư", "chạy thận", "hôn mê", bot PHẢI tự động nối thêm: *"Xin lưu ý, với tình trạng bệnh lý của bạn, dinh dưỡng đóng vai trò điều trị lâm sàng, xin lập tức tham khảo ý kiến bác sĩ chuyên khoa. Dưới đây chỉ là thông tin cơ bản..."*
  2. **Cấm kê đơn (No-Prescription Policy):** LLM prompt mạnh mẽ từ chối tư vấn dùng thuốc giảm cân, thực phẩm chức năng cụ thể trị bệnh.
  3. **Từ chối đối tượng nguy cơ cao nếu thiếu thông tin:** VD: Trẻ sơ sinh dưới 6 tháng tuổi. 

---

## 5. Cấu Trúc Payload RAG (Dự kiến)

Không giống cấu trúc Điều/Khoản luật, RAG dinh dưỡng nên phân mảnh (chunk) theo Chủ đề Thực phẩm / Khuyến nghị.

```json
{
  "doc_id": "ni_001",
  "category": "thanh_phan_thuc_pham",  // Hoặc "khuyen_nghi", "dinh_duong_benh_ly"
  "title": "Hàm lượng dinh dưỡng của Gạo lứt",
  "keywords": ["tinh bột", "carbohydrate", "chỉ số đường huyết", "GI"],
  "content": "Gạo lứt có chỉ số đường huyết (GI) trung bình khoảng 56-69, cung cấp khoảng 110 kcal, 2.6g protein và 1.8g chất xơ trên mỗi 100g (nấu chín). Thích hợp cho người tiểu đường hơn gạo trắng.",
  "target_audience": "chung, đái tháo đường",
  "contraindications": ["người tiêu hóa kém"], // Chống chỉ định
  "source": "Viện Dinh Dưỡng, 2020",
  "trust_level": "High"
}
```

---

## 6. Lộ Trình Triển Khai Dinh Dưỡng "Proof of Concept" (MVP)

Để chứng minh tính khả thi nhanh chóng:
1. **Scraping Mini-Data:** Lấy khoảng 200 bài viết QA chuần mực từ website Viện Dinh Dưỡng hoặc Cục Y tế Dự phòng về (1) Giảm cân, (2) Tiểu đường, (3) Thành phần phổ biến (cơm, phở, thịt, cá).
2. **Setup Prompt:** Chuyển `LEGAL_SYSTEM_PROMPT` thành `NUTRITION_SYSTEM_PROMPT` với các ràng buộc y tế.
3. **Mã hóa (Embed):** Ingest 200 bài này vào collection mới `nutrition_base`.
4. **Testing:** Chạy test với các câu như *"Tôi muốn giảm cân thì cắt hẳn tinh bột được không?"* để xem khả năng RAG và reasoning của LLM.
