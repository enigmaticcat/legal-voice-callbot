import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client.http import models

import logging

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger("test_ingest")

def test_qdrant_collections():
    host = os.getenv("QDRANT_HOST", "localhost")
    port = os.getenv("QDRANT_PORT", "6333")
    
    client = QdrantClient(url=f"http://{host}:{port}")
    
    parent_coll = "phap_dien_dieu"
    child_coll = "phap_dien_khoan"
    
    try:
        parent_info = client.get_collection(parent_coll)
        child_info = client.get_collection(child_coll)
        
        logger.info(f"Collection {parent_coll} exists payload points: {parent_info.points_count}")
        logger.info(f"Collection {child_coll} exists payload points: {child_info.points_count}")
        
    except Exception as e:
        logger.error(f"Failed to fetch collection details. Did you run initialization script? Error: {e}")
        return False
        
    # Test sample search (using dummy sparse)
    logger.info("Executing sample Search logic using Dense+Sparse dummy vector...")
    
    query_dense = [0.01] * 1024
    
    results = client.query_points(
       collection_name=child_coll,
       query=query_dense, # Dense query payload 
       limit=2,
       with_payload=True
    ).points
    
    for r in results:
       logger.info(f"-> Expected score hit point: {r.id}")
       logger.info(f"   Payload Title: {r.payload.get('ten_dieu', '')}")
       logger.info(f"   Snippet: {r.payload.get('text', '')[:50]}...")
       logger.info(f"   MapC Reference: {r.payload.get('mapc', '')}")
       
    return True

if __name__ == "__main__":
    test_qdrant_collections()
