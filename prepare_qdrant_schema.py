import json
import re

db_path = '/Users/nguyenthithutam/Desktop/Callbot/law_data_enriched.json'
out_path = '/Users/nguyenthithutam/Desktop/Callbot/law_data_qdrant_payloads_by_article.json'

print("Đọc DB gốc...")
with open(db_path, 'r', encoding='utf-8') as f:
    db = json.load(f)

LOAI_VB_MAP = {
    "nghị định": "nghi-dinh",
    "thông tư":  "thong-tu",
    "luật":      "luat",
    "quyết định":"quyet-dinh",
    "hiến pháp": "hien-phap",
    "pháp lệnh": "phap-lenh",
    "nghị quyết":"nghi-quyet"
}

def parse_so_hieu(vbqppl):
    m = re.search(r'\b(\d{1,4}/\d{4}/[A-ZĐ]+(?:-[A-ZĐ]+)*)\b', vbqppl)
    if m: return m.group(1)
    m2 = re.search(r'Luật.*?số\s+([\w\d/-]+)', vbqppl, re.IGNORECASE)
    if m2: return m2.group(1).strip()
    return ""

def parse_loai_vb(vb_name):
    vb_lower = vb_name.lower()
    for kw, val in LOAI_VB_MAP.items():
        if kw in vb_lower:
            return val
    return "khac"

def parse_nam(vbqppl):
    m = re.search(r'/(\d{4})/', vbqppl)
    if m: return int(m.group(1))
    m2 = re.search(r'\b(19|20)\d{2}\b', vbqppl)
    if m2: return int(m2.group(0))
    return None

def parse_so_dieu(ten):
    m = re.search(r'Điều\s+([\d\w]+)', ten, re.IGNORECASE)
    if m: return m.group(1)
    return ""

def parse_ten_dieu(ten):
    m = re.search(r'Điều\s+[\d\w]+\.?\s*(.*)', ten, re.IGNORECASE)
    if m:
        name = m.group(1).strip()
        name = re.sub(r'^\[[^\]]+\]\s*', '', name).strip()
        name = re.sub(r'^\-\s*', '', name).strip()
        return name
    return ten.strip()

def parse_ten_vb(vbqppl):
    m = re.search(r'(Luật|Nghị định|Thông tư|Quyết định|Nghị quyết|Hiến pháp|Pháp lệnh)[^\)]*(?:số\s+)?([^\s,;\)]+)', vbqppl)
    if m:
        return m.group(0).strip().replace(' ,','')
    
    clean = re.sub(r'[,;\s]*có hiệu lực( thi hành)?\s*(kể từ[^-\]]+)?', '', vbqppl, flags=re.IGNORECASE).strip()
    return clean.replace('(','').replace(')','').strip()

payloads = []
count = 0

print(f"Xử lý {len(db)} bản ghi pháp định ở cấp độ 1 Điều = 1 Chunk...")

for doc in db:
    noidung = doc.get('noidung', '')
    
    # ---------------------------------------------
    # LOẠI BỎ CHUNKING PARAGRAPH - CHỈ ĐỂ 1 ĐIỀU NGUYÊN KHỐI
    # Lấy luôn toàn bộ trường noidung (Tùy chọn bỏ dòng tiêu đề nếu muốn vì Embed text đã chứa Title)
    # Tuy nhiên vì nội dung có mang bracket "[Điều X...]", ta có thể gọt nó đi để Embed Text lo phần context.
    # ---------------------------------------------
    
    paragraphs = noidung.split("\n")
    if paragraphs and paragraphs[0].startswith("["):
        # Gọt dòng tiên phong "[Điều ... ]"
        noidung_clean = "\n".join(paragraphs[1:]).strip()
    else:
        noidung_clean = noidung.strip()
        
    if len(noidung_clean) < 10:
        continue # skip empty articles
        
    vbqppl = doc.get('vbqppl', '')
    ten = doc.get('ten', '')
    chude = doc.get("chude", "").strip()
    chuong = doc.get("chuong", "").strip()
    demuc = doc.get("demuc", "").strip()

    if not chude:
         chude = demuc if demuc else chuong
         chude = re.sub(r'^(Chương\s+[IVXLCDMivxlcdm]+\s*[-:.]?\s*|Mục\s+\d+\s*[-:.]?\s*)', '', chude).strip()

    base = {
        "so_hieu":       parse_so_hieu(vbqppl) or parse_so_hieu(ten),
        "loai_vb":       parse_loai_vb(vbqppl),
        "so_dieu":       parse_so_dieu(ten) or str(doc.get('stt', '')),
        "ma_dieu":       doc.get("mapc", ""),
        "nam":           parse_nam(vbqppl),
        "ngay_ban_hanh": None,
        "ten_dieu":      parse_ten_dieu(ten),
        "ten_vb":        parse_ten_vb(vbqppl),
        "chu_de":        chude,
        "chuong":        chuong,
        "parent_mapc":   doc.get("mapc", ""),
        "meta": {
            "link_goc":  doc.get("url", ""), 
            "stt":       doc.get("stt", ""),
            "chimuc":    doc.get("chimuc", ""),
        }
    }

    # Gom vào làm 1 Chunk Duy Nhất
    chunk = {
        **base,
        "chunk_type":  "content",
        "chunk_index": 0,
        "chunk_text":  noidung_clean,
    }
    
    embed_txt = (
        f"{chunk['loai_vb'].replace('-', ' ').title()} {chunk['so_hieu']} "
        f"— {chunk['ten_vb']}\n"
        f"Chủ đề: {chunk['chu_de']}\n"
        f"{chunk['chuong']}\n"
        f"{chunk['ten_dieu']}\n\n"
        f"{chunk['chunk_text']}"
    )
    chunk['embed_text'] = embed_txt.strip()
    payloads.append(chunk)
        
    count += 1
    if count % 10000 == 0:
        print(f"Đã xử lý {count} / {len(db)} bản ghi...")

print("Tiến hành ghi tệp dữ liệu chunked...")
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(payloads, f, ensure_ascii=False, indent=2)

print(f"✅ HOÀN TẤT UNDO. Đã tạo ra tổng cộng: {len(payloads)} vectors (Số lượng nguyên bản 1 Điều = 1 Vector).")
print(f"File Output: {out_path}")
