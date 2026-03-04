import sys
import json
sys.path.insert(0, '/Users/nguyenthithutam/Desktop/Callbot/legal-callbot/brain')
from core.legal_chunker import LegalChunker

with open('/Users/nguyenthithutam/Desktop/Callbot/law_data.json', 'r', encoding='utf-8') as f:
    records = json.load(f)
    for data in records:
        if '39.13.NĐ.75.6' in data.get('ten', ''):
            print(f"Found {data['ten']}")
            chunker = LegalChunker(
                min_words=80,
                max_words=500,
                overlap_words=30
            ) # matching the params in colab_ingest.py
            
            chunks = chunker.extract_chunks(
                data['mapc'], data.get('ten', ''), data.get('noidung', ''), {}
            )
            
            for c in chunks:
                if c['type'] == 'child':
                    print(f"\n--- {c['id']} ---")
                    print(f"Length: {len(c['text'])} chars")
                    print(c['text'][:200] + "...\n" + c['text'][-100:])
            break
