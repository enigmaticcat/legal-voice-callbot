import json
import os
import sys
import logging
from typing import List, Dict

# Add parent directory to path to import core
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.legal_enricher import LegalEnricher

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("refine_data")

def refine_dataset(input_path: str, output_path: str):
    logger.info(f"Loading raw data from {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    enricher = LegalEnricher()
    refined_data = []
    
    total = len(data)
    logger.info(f"Enriching {total} articles (1:1 mapping)...")
    
    for i, entry in enumerate(data):
        ten = entry.get("ten", "")
        noidung = entry.get("noidung", "")
        
        # Metadata
        metadata = {
            "chude": entry.get("chude", ""),
            "demuc": entry.get("demuc", ""),
            "chuong": entry.get("chuong", ""),
            "vbqppl": entry.get("vbqppl", ""),
            "link": entry.get("vbqppl_link", ""),
        }
        
        # Enrich content in-place
        enriched_noidung = enricher.enrich_content(ten, noidung, metadata)
        
        # Keep exact same structure as original law_data.json but with "noidung_enriched"
        # or overwrite "noidung". Let's create a new field to be safe, or overwrite if requested.
        # User said "đổi file law_data.json cũ thành kiểu...", implying overwrite/replace content.
        
        refined_record = dict(entry) # Copy all fields
        refined_record["noidung"] = enriched_noidung
        refined_data.append(refined_record)
        
        if (i + 1) % 5000 == 0:
            logger.info(f"Progress: {i + 1}/{total} articles enriched...")

    logger.info(f"Refinement complete. Total records: {len(refined_data)}")
    logger.info(f"Saving to {output_path}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(refined_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_file = "/Users/nguyenthithutam/Desktop/Callbot/law_data.json"
    output_file = "/Users/nguyenthithutam/Desktop/Callbot/law_data_enriched.json"
    
    refine_dataset(input_file, output_file)
