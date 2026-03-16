import re
import logging
from typing import List, Dict

logger = logging.getLogger("brain.core.legal_enricher")

# Patterns for in-place substitution at the start of lines
# We use lookahead/lookbehind or anchor to ensure we don't match middle of words
SUBSTITUTIONS = [
    (re.compile(r'^(\d+)\.\s'), r'Khoản \1. '),
    (re.compile(r'^([a-zđ])\)\s'), r'Điểm \1) '),
    (re.compile(r'^([ivx]+)\)\s'), r'Tiết \1) '),
    (re.compile(r'^([IVXLC]+)\.\s'), r'Mục \1. '),
]

class LegalEnricher:
    def __init__(self, use_friendly_titles: bool = True):
        self.use_friendly_titles = use_friendly_titles

    def _get_friendly_title(self, ten_entry: str, metadata: Dict) -> str:
        """Extracts a human-readable title like 'Điều 45 - Dược'."""
        vbqppl = metadata.get("vbqppl", "")
        demuc = metadata.get("demuc", "")
        
        # 1. Try to get "Điều X" from vbqppl first (it's often more accurate than ten_entry)
        dieu_match = re.search(r'Điều\s+([\d[a-zA-Z\d\.]+)', vbqppl, re.I)
        if not dieu_match:
            dieu_match = re.search(r'Điều\s+([a-zA-Z\d\.]+)', ten_entry, re.I)
            
        # 2. Get the core document name or category
        # If it's a specific law/decree, try to find it
        doc_match = re.search(r'(Nghị định|Luật|Thông tư|Quyết định)\s+số\s+[\d/]+[A-ZĐ-]+', vbqppl, re.I)
        if not doc_match:
            doc_match = re.search(r'(Bộ luật|Luật)\s+[^,.)]+', vbqppl, re.I)

        title_parts = []
        if dieu_match:
            title_parts.append(dieu_match.group(0))
        
        # Add Demuc as requested by user [Điều 45 - Dược]
        if demuc and demuc != "Unknown":
            title_parts.append(demuc)
        elif doc_match:
            title_parts.append(doc_match.group(0).strip())
            
        if title_parts:
            # Join with "-" as requested [Điều 45 - Dược]
            return " - ".join(title_parts)
        
        return ten_entry.split('.')[0] if '.' in ten_entry else ten_entry

    def enrich_content(self, ten_entry: str, noidung: str, metadata: Dict) -> str:
        """Enriches the content text in-place without splitting."""
        friendly_title = self._get_friendly_title(ten_entry, metadata)
        
        # Start with the header
        header = f"[{friendly_title}]"
        
        lines = noidung.split('\n')
        enriched_lines = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                enriched_lines.append("")
                continue
            
            # Apply substitutions for the start of the line
            new_line = stripped
            for regex, replacement in SUBSTITUTIONS:
                if regex.match(stripped):
                    new_line = regex.sub(replacement, stripped)
                    break 
            
            enriched_lines.append(new_line)
            
        return header + "\n" + "\n".join(enriched_lines)

if __name__ == "__main__":
    enricher = LegalEnricher()
    sample_text = "1. Đối tượng áp dụng...\na) Cá nhân...\nb) Tổ chức..."
    metadata = {"vbqppl": "(Điều 2 Nghị định số 127/2006/NĐ-CP, có hiệu lực...)"}
    result = enricher.enrich_content("Điều 1.1.N", sample_text, metadata)
    print(result)
