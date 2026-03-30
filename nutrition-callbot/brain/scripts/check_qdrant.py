import os, sys
from qdrant_client import QdrantClient
from dotenv import load_dotenv

import transformers.utils.import_utils
if not hasattr(transformers.utils.import_utils, 'is_torch_fx_available'):
    transformers.utils.import_utils.is_torch_fx_available = lambda: False
from FlagEmbedding import BGEM3FlagModel

load_dotenv('/Users/nguyenthithutam/Desktop/Callbot/legal-callbot/.env')
url = os.getenv('QDRANT_URL')
api_key = os.getenv('QDRANT_API_KEY')

qdrant = QdrantClient(url=url, api_key=api_key)
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)

queries = [
    'Xử phạt người điều khiển xe mô tô, xe gắn máy không chấp hành hiệu lệnh của đèn tín hiệu giao thông',
    'Phạt tiền lỗi không chấp hành hiệu lệnh của đèn tín hiệu giao thông đối với xe máy',
    'phạt tiền xe mô tô xe gắn máy không chấp hành hiệu lệnh của đèn tín hiệu giao thông'
]

for q in queries:
    print(f'\n--- Query: {q} ---')
    emb = model.encode([q], return_dense=True, return_sparse=True, return_colbert_vecs=False)
    
    res_child = qdrant.query_points(
        collection_name='phap_dien_khoan',
        query=emb['dense_vecs'][0].tolist(),
        limit=3,
        with_payload=True
    )
    for i, p in enumerate(res_child.points):
        print(f'   {i+1}. [{p.score:.4f}] {p.payload.get("ten_dieu")} - {p.payload.get("chunk_label", "Unknown")}')
        text = p.payload.get("text", "")
        print(f'      {text[:120]}...')
