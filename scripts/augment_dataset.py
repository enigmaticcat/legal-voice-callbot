import json
import re
import os

def extract_law_name_from_noidung(noidung):
    """Trích xuất tên luật từ content: [Điều X - Tên Luật]"""
    if not noidung:
        return ""
    match = re.search(r'\[.+? - (.+?)\]', noidung)
    if match:
        return match.group(1).strip()
    return ""

def augment_text(text, law_name):
    """Ghép tên luật vào sau chữ 'Điều X' nếu chưa có."""
    if not text or not isinstance(text, str) or not law_name:
        return text

    ignore_keywords = ["luật", "nghị định", "thông tư", "bộ luật", "pháp lệnh", "quyết định"]
    
    def replace_match(match):
        full_match = match.group(0)
        start_idx = match.end()
        # Kiểm tra 40 ký tự tiếp theo cho rộng rãi
        next_text = text[start_idx:start_idx+40].lower()
        
        has_law_keyword = any(kw in next_text for kw in ignore_keywords)
        
        if has_law_keyword:
            return full_match
        else:
            return f"{full_match} của {law_name}"
            
    pattern = r"(Điều\s+\d+[a-z]?)"
    return re.sub(pattern, replace_match, text, flags=re.IGNORECASE)

def process_nested_obj(obj, law_name_display):
    """Đệ quy xử lý các dictionary và list để tìm trường văn bản."""
    changed = False
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ['question', 'explanation', 'A', 'B', 'C', 'D', 'premise', 'hypothesis', 'situation', 'statement', 'reasoning']:
                if isinstance(v, str):
                    new_v = augment_text(v, law_name_display)
                    if new_v != v:
                        obj[k] = new_v
                        changed = True
            elif isinstance(v, (dict, list)):
                if process_nested_obj(v, law_name_display):
                    changed = True
    elif isinstance(obj, list):
        for item in obj:
            if process_nested_obj(item, law_name_display):
                changed = True
    return changed

def load_law_mapping(law_data_path, dataset_mapcs):
    """Xây dựng mapping mapc -> law_name."""
    mapping = {}
    if not os.path.exists(law_data_path):
        print(f"Cảnh báo: Không tìm thấy {law_data_path}")
        return mapping
        
    print(f"Đang tải dữ liệu luật từ {law_data_path}...")
    with open(law_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            mapc = item.get('mapc')
            if mapc in dataset_mapcs:
                law_name = extract_law_name_from_noidung(item.get('noidung', ''))
                if not law_name:
                    # Fallback vào field 'ten' nếu ko có trong noidung
                    law_name = item.get('ten', '')
                mapping[mapc] = law_name
    print(f"Đã xây dựng xong mapping cho {len(mapping)} mã luật.")
    return mapping

def main():
    base_dir = "/Users/nguyenthithutam/Desktop/Callbot"
    law_data_path = os.path.join(base_dir, "law_data_enriched.json")
    files_to_process = [
        os.path.join(base_dir, "task1_output.json"),
        os.path.join(base_dir, "task2_output.json"),
        os.path.join(base_dir, "task3_output.json")
    ]
    
    # Bước 1: Thu thập tất cả MAPC từ datasets
    all_dataset_mapcs = set()
    datasets = {}
    for filepath in files_to_process:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                datasets[filepath] = data
                for item in data:
                    if 'source_mapc' in item: all_dataset_mapcs.add(item['source_mapc'])
                    if 'original_id' in item: all_dataset_mapcs.add(item['original_id'])
                    if 'related_laws' in item:
                        for rl in item['related_laws']:
                            if 'mapc' in rl: all_dataset_mapcs.add(rl['mapc'])

    # Bước 2: Tải mapping
    law_map = load_law_mapping(law_data_path, all_dataset_mapcs)
    
    # Bước 3: Xử lý từng file
    for filepath, data in datasets.items():
        print(f"Đang xử lý {os.path.basename(filepath)}...")
        updated_count = 0
        for item in data:
            # Tìm law_name
            law_name = ""
            potential_mapcs = [item.get('source_mapc'), item.get('original_id')]
            if 'related_laws' in item and item['related_laws']:
                 potential_mapcs.append(item['related_laws'][0].get('mapc'))
            
            for m in potential_mapcs:
                if m in law_map and law_map[m]:
                    law_name = law_map[m]
                    break
            
            if not law_name:
                continue
            
            # Làm đẹp law_name
            if not any(kw in law_name.lower() for kw in ["luật", "nghị định", "thông tư", "quy định", "pháp lệnh", "nghị quyết", "bộ luật"]):
                law_name_display = f"văn bản về {law_name}"
            else:
                law_name_display = law_name
            
            if process_nested_obj(item, law_name_display):
                updated_count += 1
        
        print(f"Đã cập nhật {updated_count}/{len(data)} items.")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
