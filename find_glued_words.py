import json
import re

file_path = "/Users/nguyenthithutam/Desktop/Callbot/cauhoichinhsach_2024.json"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 1. Số chạm Chữ Hoa (vd: 91Luật)
pattern_num_title = re.compile(r'(\d+)([A-ZĐđ][a-zđ]+)')

# 2. Mã chạm Chữ Thường (vd: NĐ-CPđã)
pattern_caps_lower = re.compile(r'([A-ZĐ\-]{2,})([a-zđ]{2,})')

matches_num_title = []
matches_caps_lower = []

for idx, item in enumerate(data):
    text = item.get("answer", "") + " " + item.get("question", "")
    
    found1 = pattern_num_title.findall(text)
    if found1:
        matches_num_title.extend(["".join(m) for m in found1])
        
    found2 = pattern_caps_lower.findall(text)
    if found2:
        matches_caps_lower.extend(["".join(m) for m in found2])

print("--- GLUED WORD INSPECTION (P2) ---")
print(f"1. Lỗi Số + Chữ Hoa (ví dụ 91Luật): {len(matches_num_title)}")
print("Ví dụ:", list(set(matches_num_title))[:15])

print(f"\n2. Lỗi Mã + Chữ thường (ví dụ NĐ-CPđã): {len(matches_caps_lower)}")
print("Ví dụ:", list(set(matches_caps_lower))[:15])
