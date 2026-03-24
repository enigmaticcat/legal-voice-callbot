import json
import re
import os
import hashlib

TEXT_DIR = '/Users/nguyenthithutam/Desktop/Callbot/extracted_texts/'
LAW_DB_PATH = '/Users/nguyenthithutam/Desktop/Callbot/law_data.json'
OUTPUT_PATH = '/Users/nguyenthithutam/Desktop/Callbot/law_data_extended.json'

def load_existing_db():
    if os.path.exists(LAW_DB_PATH):
        with open(LAW_DB_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def parse_text_file(filepath, filename):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    lines = text.split('\n')
    parsed_items = []
    
    current_chuong = ""
    current_muc = ""
    current_item = None

    document_id = filename.replace('.txt', '').replace('_', '/')

    for line in lines:
        line = line.strip()
        if not line: continue

        if line.startswith('Chương '):
            current_chuong = line
            current_muc = ""
            continue

        if line.startswith('Mục '):
            current_muc = line
            continue

        if line.startswith('Điều '):
            if current_item:
                 parsed_items.append(current_item)

            dieu_match = re.match(r'(Điều\s+[\d\w]+)\.?\s*(.*)', line)
            dieu_word = dieu_match.group(1).strip() if dieu_match else "Điều"
            stt = dieu_word.replace('Điều ', '').strip()
            rest = dieu_match.group(2).strip() if dieu_match else ""
            
            # Phân tách Tên Điều (Title) và Nội dung (Content)
            # Tìm dấu chấm câu đầu tiên (nhỏ hơn 150 ký tự) để tách title
            split_match = re.match(r'^([^.!?]{1,150}[.!?])\s+(.*)', rest)
            if split_match:
                title = f"{dieu_word}. {split_match.group(1).strip()}"
                content = split_match.group(2).strip()
            elif len(rest) > 150:
                # Nếu chuỗi quá dài (thường là Điều không có tên mà vào luôn nội dung)
                title = f"{dieu_word}. {rest[:80]}..."
                content = rest
            else:
                # Bình thường, title ngắn và nội dung rỗng (sẽ được nhặt dòng dưới)
                title = f"{dieu_word}. {rest}" if rest else dieu_word
                content = ""

            # Mã hash mapc
            m = hashlib.md5()
            m.update(f"{document_id}_{title}".encode('utf-8'))
            mapc = m.hexdigest()

            current_item = {
                'mapc': mapc,
                'ten': title,
                'noidung': content,
                'vbqppl': f'({dieu_word} {document_id})',
                'stt': stt,
                'chimuc': '',
                'chuong': current_chuong,
                'demuc': current_muc,
                'chude': '',
                'dieu_lienquan': []
            }
            continue

        if current_item:
             current_item['noidung'] = (current_item['noidung'] + '\n' + line).strip()

    if current_item:
         parsed_items.append(current_item)
         
    return parsed_items

def main():
    print("--- KHỞI ĐỘNG PARSER HỢP NHẤT ---")
    existing_data = load_existing_db()
    print(f"Dữ liệu gốc hiện có: {len(existing_data)} mục.")

    if not os.path.exists(TEXT_DIR):
        print(f"Thư mục {TEXT_DIR} không tồn tại.")
        return

    new_items_count = 0
    file_count = 0

    for filename in os.listdir(TEXT_DIR):
        if filename.endswith('.txt'):
            file_count += 1
            path = os.path.join(TEXT_DIR, filename)
            try:
                items = parse_text_file(path, filename)
                existing_data.extend(items)
                new_items_count += len(items)
            except Exception as e:
                print(f"Lỗi khi xử lý {filename}: {str(e)}")

    print(f"Đã xử lý {file_count} file văn bản thô.")
    print(f"Trích xuất bổ sung: {new_items_count} điều khoản.")
    
    # Save output
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)
        
    print(f"➡️ ĐÃ LƯU KẾT QUẢ VÀO: {OUTPUT_PATH}")
    print(f"Tổng số bản ghi mới: {len(existing_data)}")

if __name__ == "__main__":
    main()
