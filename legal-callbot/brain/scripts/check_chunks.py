import os, sys
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv

load_dotenv('/Users/nguyenthithutam/Desktop/Callbot/legal-callbot/.env')
url = os.getenv('QDRANT_URL')
api_key = os.getenv('QDRANT_API_KEY')

qdrant = QdrantClient(url=url, api_key=api_key)

print("Fetching chunks for mapc 39.13.NĐ.75.6...")
res = qdrant.scroll(
    collection_name='phap_dien_khoan',
    scroll_filter=Filter(
        must=[
            FieldCondition(
                key="mapc",
                match=MatchValue(value="39.13.NĐ.75.6")
            )
        ]
    ),
    limit=50,
    with_payload=True
)

for point in res[0]:
    label = point.payload.get("chunk_label")
    text = point.payload.get("text", "")
    print(f"\n--- {label} ---")
    print(text[:200] + "..." if len(text) > 200 else text)
    print(f"Length: {len(text)} chars")
