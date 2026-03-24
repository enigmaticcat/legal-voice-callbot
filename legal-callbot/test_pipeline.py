"""
Script test kết nối chuỗi (Pipeline): ASR -> Brain (LLM)
Cho phép chạy trực tiếp trong 1 File duy nhất trên Colab (Không cần gRPC/HTTP overhead).
"""
import asyncio
import os
import sys
import time
import wave

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
    print("[1] Đang khởi tạo ASR (Sherpa-Onnx)...")
    asr = Transcriber()
    asr_stream = asr.create_stream()

    print("[2] Đang khởi tạo Brain (LLM + RAG)...")
    llm = LLMClient(api_key=brain_config.gemini_api_key, model=brain_config.gemini_model)
    rag = RAGPipeline(
        qdrant_url=brain_config.qdrant_url,
        qdrant_api_key=brain_config.qdrant_api_key,
        collection=brain_config.qdrant_collection,
    )
    brain = BrainServiceHandler(llm=llm, rag=rag)

    print("\nHệ thống sẵn sàng!")
    print("----------------------------------------")

    # Giả lập 1 đoạn Audio byte đẩy từ Client hoặc lấy từ file wav (nếu có)
    audio_file = os.getenv("TEST_AUDIO_FILE", "")
    
    if audio_file and os.path.exists(audio_file):
        print(f"[ASR] Đọc Audio từ file {audio_file}...")
        with wave.open(audio_file, 'rb') as wf:
            audio_data = wf.readframes(wf.getnframes())
    else:
        print("[ASR] Dùng audio giả lập (Dummy)...")
        audio_data = b'\x00' * 32000  # 1 giây im lặng để test

    print("[ASR] Đang giải mã Audio...")
    start_asr = time.time()
    text = asr.accept_wave(asr_stream, audio_data)
    asr_time = time.time() - start_asr
    
    # Giả lập có chữ đổ về (Nếu mic test ra chữ)
    if not text:
        text = "Cho tôi hỏi mức phạt vượt đèn đỏ xe máy là bao nhiêu?"
        print(f"[ASR Dummy/Test] Bạn nói: '{text}' (Mất {asr_time:.2f} s)")
    else:
        print(f"[ASR] Bạn nói: '{text}' (Mất {asr_time:.2f} s)")

    print("\n[Brain] Đang suy nghĩ và tra cứu luật...")
    print("AI: ", end="", flush=True)

    start_llm = time.time()
    first_token_time = None

    async for chunk in brain.think(query=text, session_id="colab-test"):
        if chunk.get("text"):
            if first_token_time is None:
                first_token_time = time.time() - start_llm
                print(f"\n[Time-To-First-Token: {first_token_time:.2f} s]\n", end="")
            
            print(chunk["text"], end="", flush=True)

    total_llm_time = time.time() - start_llm
    print(f"\n\n[Tổng thời gian Brain/LLM: {total_llm_time:.2f} s]")
    print(f"Hoàn tất chuỗi Inference. Tổng thời gian End-to-End: {asr_time + total_llm_time:.2f} s")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
