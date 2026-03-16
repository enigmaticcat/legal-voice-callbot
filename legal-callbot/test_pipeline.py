"""
Script test kết nối chuỗi (Pipeline): ASR -> Brain (LLM)
Cho phép chạy trực tiếp trong 1 File duy nhất trên Colab (Không cần gRPC/HTTP overhead).
"""
import asyncio
import os
import sys

# Thêm đường dẫn vào Python Path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "asr")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "brain")))

from asr.core.transcriber import Transcriber
from brain.grpc_handler import BrainServiceHandler
from brain.core.llm import LLMClient
from brain.core.rag import RAGPipeline
from brain.config import config as brain_config

async def run_pipeline():
    print("⏳ 1. Đang khởi tạo ASR (Sherpa-Onnx)...")
    asr = Transcriber()
    asr_stream = asr.create_stream()

    print("⏳ 2. Đang khởi tạo Brain (LLM + RAG)...")
    llm = LLMClient(api_key=brain_config.gemini_api_key, model=brain_config.gemini_model)
    rag = RAGPipeline(
        qdrant_url=brain_config.qdrant_url,
        qdrant_api_key=brain_config.qdrant_api_key,
        collection=brain_config.qdrant_collection,
    )
    brain = BrainServiceHandler(llm=llm, rag=rag)

    print("\n✅ Hệ thống sẵn sàng!")
    print("----------------------------------------")

    # Giả lập 1 đoạn Audio byte đẩy từ Client lên
    # (Trong Colab bạn sẽ đẩy mảng pcm byte từ Client hoặc Micro ghi âm)
    dummy_audio_pcm = b'\x00' * 32000  # 1 giây im lặng để test

    print("🎤 [ASR] Đang giải mã Audio...")
    text = asr.accept_wave(asr_stream, dummy_audio_pcm)
    
    # Giả lập có chữ đổ về (Nếu mic test ra chữ)
    if not text:
        text = "Cho tôi hỏi mức phạt vượt đèn đỏ xe máy là bao nhiêu?"
        print(f"💡 [ASR Dummy/Test] Bạn nói: '{text}'")
    else:
        print(f"🗣️  [ASR] Bạn nói: '{text}'")

    print("\n🧠 [Brain] Đang suy nghĩ và tra cứu luật...")
    print("🤖 AI: ", end="", flush=True)

    async for chunk in brain.think(query=text, session_id="colab-test"):
        if chunk.get("text"):
            print(chunk["text"], end="", flush=True)

    print("\n\n✅ Hoàn tất chuỗi Inference.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
