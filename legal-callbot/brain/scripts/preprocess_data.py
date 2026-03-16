import json
import re
from pathlib import Path
from tqdm import tqdm

def clean_dieu_title(title: str) -> str:
    """Loại bỏ các mã nội bộ như '1.1.LQ.1.' khỏi Tên Điều. Giữ nguyên nếu không match."""
    match = re.match(r'(Điều\s+\d+(?:\.\d+)?)(?:\.[A-Z]+\.\d+)?\.?\s*(.*)', title)
    if match:
        clean_title = match.group(2).strip()
        if clean_title.endswith('.'):
            clean_title = clean_title[:-1]
        return f"{match.group(1)}: {clean_title}"
    return title

def run():
    print("1. Đọc dữ liệu pháp luật gốc (law_data.json)...")
    base_dir = Path(__file__).parent.parent
    input_path = base_dir / 'data' / 'law_data.json'
    output_path = base_dir / 'data' / 'law_data_clean.json'
    
    if not input_path.exists():
        print(f"❌ Không tìm thấy file {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"✅ Đã tải: {len(data)} Điều Luật.")

    # 2. Xây dựng Hash Map (Dictionary) để tra cứu theo mapc siêu tốc
    print("2. Đang xây dựng Tra Cứu theo Mã (mapc)...")
    record_map = {item.get('mapc'): item for item in data if 'mapc' in item}
    
    # 3. Tiến hành làm sạch và nối Cross-Reference
    print("3. Đang lọc, làm sạch tên Điều và nối Điều kiện Liên quan...")
    
    for item in tqdm(data, desc="Đang xử lý"):
        # ----- 1. Làm sạch Metadata cơ bản -----
        raw_ten_dieu = item.get('ten', 'Không rõ')
        clean_ten = clean_dieu_title(raw_ten_dieu)
        item['ten_clean'] = clean_ten # Lưu riêng bản sạch để UI hoặc API khác xài nếu cần
        
        vbqppl = item.get('vbqppl', 'Không rõ').strip("() ")
        chuong = item.get('chuong', 'Không rõ')
        demuc = item.get('demuc', 'Không rõ')
        noidung = item.get('noidung', '').strip()
        
        # ----- 2. Trích xuất Điều Liên Quan (Cross-reference) -----
        related_str = ""
        dieu_lienquan = item.get('dieu_lienquan', [])
        
        if dieu_lienquan:
            rel_list = []
            for rel in dieu_lienquan:
                rel_mapc = rel.get('mapc')
                # Lấy tên sạch nếu có trong data, không thì dọn dẹp tại chỗ
                if rel_mapc and rel_mapc in record_map:
                    rel_item = record_map[rel_mapc]
                    rel_title = clean_dieu_title(rel_item.get('ten', ''))
                    # Ghép 1 đoạn ngắn Nội dung để LLM có thêm Manh mối tham chiếu
                    rel_content = rel_item.get('noidung', '').strip()
                    # Trích khoảng 500 ký tự đầu tiên để tránh phình to prompt quá đáng
                    preview_content = (rel_content[:500] + '...') if len(rel_content) > 500 else rel_content
                    # Nối vào danh sách
                    rel_list.append(f"- {rel_title}: {preview_content}")
                else:
                    # Nếu mapc liên quan KHÔNG tổn tại trong DB chính, chỉ ghép Tên (Title)
                    rel_title = clean_dieu_title(rel.get('ten', ''))
                    rel_list.append(f"- {rel_title}")
                    
            if rel_list:
                related_str = "\nCÁC ĐIỀU LIÊN QUAN ĐƯỢC THAM CHIẾU:\n" + "\n".join(rel_list)

        # ----- 3. Chuẩn bị LLM Context Chuẩn -----
        llm_context_header = (
            f"VĂN BẢN PHÁP LUẬT: {vbqppl}.\n"
            f"VỊ TRÍ: {chuong} - Đề mục: {demuc}.\n"
            f"ĐIỀU KHOẢN: {clean_ten.upper()}."
        )
        llm_context_footer = related_str
        
        item['llm_context_header'] = llm_context_header
        item['llm_context_footer'] = llm_context_footer
        
        # Full context for QA Generation (which takes the whole article)
        item['llm_context'] = (
            f"{llm_context_header}\n"
            f"NỘI DUNG CHÍNH: {noidung}\n"
            f"{llm_context_footer}"
        ).strip()

    # 4. Ghi file mới
    print(f"\n4. Ghi file dữ liệu sạch ra {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print("✅ Hoàn Thành Preprocessing! Dữ liệu đã sạch, sẵn sàng nạp Qdrant.")

if __name__ == "__main__":
    run()
