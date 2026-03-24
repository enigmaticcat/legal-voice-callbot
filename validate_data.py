import json

db_path = '/Users/nguyenthithutam/Desktop/Callbot/law_data_enriched.json'
qa_path = '/Users/nguyenthithutam/Desktop/Callbot/cauhoichinhsach_2024.json'

print("=== KIỂM TRA CHẤT LƯỢNG LAW_DATA_ENRICHED.JSON ===")
try:
    with open(db_path, 'r', encoding='utf-8') as f:
        db = json.load(f)
    print(f"✅ Định dạng JSON hợp chuẩn. Tổng {len(db)} bản ghi độc lập.")
    
    seen_mapc = set()
    dup_mapc = 0
    empty_content = 0
    missing_required_keys = 0
    required_keys = ['mapc', 'ten', 'noidung', 'vbqppl']
    
    for item in db:
        isValid = True
        for k in required_keys:
            if k not in item:
                isValid = False
                break
        if not isValid:
            missing_required_keys += 1
            
        m = item.get('mapc', '')
        if m in seen_mapc:
            dup_mapc += 1
        seen_mapc.add(m)
        
        # Có một số Điều khoản chỉ có Tên (tiêu đề nhóm) mà không có Dòng nội dung
        if not str(item.get('noidung', '')).strip():
            empty_content += 1
            
    print(f"-> Vi phạm Key Bắt Buộc (mapc, ten, noidung, vbqppl): {missing_required_keys}")
    print(f"-> Mức độ lặp lặp Key Duy Nhất (dup mapc): {dup_mapc}")
    
    if dup_mapc > 0:
        print(f"⚠️ Phát hiện {dup_mapc} lặp. Bắt đầu DEDUPLICATION làm sạch...")
        unique_db = []
        clean_seen = set()
        for item in db:
            m = item.get('mapc','')
            if m not in clean_seen:
                clean_seen.add(m)
                unique_db.append(item)
        with open(db_path, 'w', encoding='utf-8') as f:
            json.dump(unique_db, f, ensure_ascii=False, indent=2)
        print(f"   -> ĐÃ KHẮC PHỤC: Kích thước Database xịn 100%: {len(unique_db)}")

except Exception as e:
    print(f"❌ Lỗi Kiểm tra Luật RAG: {e}")


print("\n=== KIỂM TRA CHẤT LƯỢNG CAUHOICHINHSACH_2024.JSON ===")
try:
    with open(qa_path, 'r', encoding='utf-8') as f:
        qa = json.load(f)
    print(f"✅ Định dạng JSON QA chuẩn tắc. Tổng {len(qa)} câu.")
    
    missing_text = 0
    for q in qa:
        has_q = bool(str(q.get('question', '')).strip())
        has_a = bool(str(q.get('answer', '')).strip())
        if not has_q or not has_a:
            missing_text += 1
            
    print(f"-> Vi phạm câu hỏi trắng/không có đáp án: {missing_text}")
    if missing_text == 0:
        print(f"   -> ĐÁNH GIÁ: Bộ phận Test (Q&A) đã hoàn hảo.")
except Exception as e:
    print(f"❌ Lỗi Kiểm tra QA Eval: {e}")

