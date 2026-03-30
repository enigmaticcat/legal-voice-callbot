"""
Test RAG Pipeline — Truy vấn Qdrant Cloud + Gemini 2.5 Flash
"""
import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

# Thêm thư mục brain vào sys.path để import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.llm import LLMClient
from core.prompt import build_prompt, LEGAL_SYSTEM_PROMPT
from core.query_expander import expand_query

# Load .env
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

async def main():
    if not QDRANT_URL or not QDRANT_API_KEY:
        print("Thiếu QDRANT_URL hoặc QDRANT_API_KEY trong file .env!")
        print("Vui lòng thêm vào file legal-callbot/.env:")
        print("QDRANT_URL=https://...")
        print("QDRANT_API_KEY=...")
        return
        
    print("1⃣ Khởi tạo Qdrant Client...")
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10)
    
    print("2⃣ Load mô hình BGE-M3 (để nhúng câu hỏi)...")
    # --- PATCH for modern transformers library ---
    import transformers.utils.import_utils
    if not hasattr(transformers.utils.import_utils, 'is_torch_fx_available'):
        transformers.utils.import_utils.is_torch_fx_available = lambda: False
    # ---------------------------------------------
    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False) # Local Mac dùng FP32 hoặc CPU
    
    print("3⃣ Khởi tạo Gemini LLM...")
    llm = LLMClient(api_key=GEMINI_API_KEY, model="gemini-2.5-flash")
    
    # --- TEST RAG ---
    query = "Tôi lỡ vượt đèn đỏ khi đi xe máy thì bị phạt bao nhiêu tiền?"
    print(f"\nCâu hỏi: {query}")
    
    print("\nĐang tìm kiếm trên Qdrant Cloud...")
    # 1. Embed câu hỏi
    expanded_query = expand_query(query)
    if expanded_query != query:
        print(f"Query mở rộng: {expanded_query}")
    
    q_emb = model.encode([expanded_query], return_dense=True, return_sparse=True, return_colbert_vecs=False)
    
    # 2. Hybrid search (Dense + Sparse) trên collection phap_dien_khoan
    # Prepare sparse vector
    si, sv = [], []
    for tid, w in q_emb["lexical_weights"][0].items():
        try:
            si.append(int(tid))
            sv.append(float(w))
        except: pass

    sparse_query = models.SparseVector(indices=si, values=sv)
    dense_query = q_emb["dense_vecs"][0].tolist()

    results = qdrant.query_points(
        collection_name="phap_dien_khoan",
        prefetch=[
            models.Prefetch(query=dense_query, using="", limit=20),
            models.Prefetch(query=sparse_query, using="sparse", limit=20),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=10,
        with_payload=True,
    )
    
    if not results.points:
        print("Không tìm thấy kết quả nào trong Qdrant.")
        return
        
    print(f"Tìm thấy {len(results.points)} căn cứ pháp lý:\n")
    legal_context = []
    for i, point in enumerate(results.points):
        score = point.score
        ten_dieu = point.payload.get("ten_dieu", "Unknown")
        text = point.payload.get("text", "")
        print(f"   {i+1}. [{score:.4f}] {ten_dieu}")
        print(f"      {text[:150]}...")
        legal_context.append(f"[{ten_dieu}] {text}")
        
    # 3. Build Prompt cho Gemini
    context_str = "\n".join(legal_context)
    prompt = build_prompt(query=query, legal_context=context_str)
    
    print("\nĐang gọi Gemini suy luận...")
    print("-" * 50)
    
    # 4. Stream câu trả lời từ Gemini
    async for chunk in llm.generate_stream(prompt=prompt, system_instruction=LEGAL_SYSTEM_PROMPT):
        if chunk.get("text"):
            print(chunk["text"], end="", flush=True)
            
    print("\n" + "-" * 50)
    print("Hoàn tất!")

if __name__ == "__main__":
    asyncio.run(main())
