"""
Script test ASR trực tiếp với Microphone thông qua sounddevice
Yêu cầu cài đặt: 
  1. pip install sounddevice
  2. (Trên MacOS) brew install portaudio
"""
import sys
import os
import time

# Thêm thư mục hiện tại (asr/) vào PATH để load core/ và config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.transcriber import Transcriber

try:
    import numpy as np
    import sounddevice as sd
except ImportError:
    print("❌ Lỗi: Chưa cài đặt 'sounddevice' hoặc 'numpy'.")
    print("👉 Hãy chạy lệnh sau ngoài terminal:")
    print("   pip install sounddevice numpy")
    sys.exit(1)

# 1. Khởi tạo Transcriber (Load model)
print("⏳ Đang tải mô hình ASR (Sherpa-Onnx)...")
try:
    transcriber = Transcriber()
except Exception as e:
    print(f"❌ Không thể tải model: {e}")
    sys.exit(1)

# 2. Tạo Stream
stream = transcriber.create_stream()

sample_rate = 16000
chunk_size = 1024  # frames per buffer

print("\n🎤 Đang kết nối Microphone... Hãy bắt đầu nói!")
print("💡 (In kết quả Real-time. Ấn Ctrl+C để kết thúc)\n")

# Biến lưu trữ đoạn text trước đó để so sánh in đè
current_text = ""

def audio_callback(indata, frames, time_info, status):
    global current_text
    if status:
        print(f"⚠️ Status: {status}", file=sys.stderr)
        
    # sounddevice trả về float32, ta chuyển về int16 để khớp với Transcriber
    audio_float = indata[:, 0]  # Thao tác trên kênh Mono đầu tiên
    audio_int16 = (audio_float * 32767).astype(np.int16)
    pcm_bytes = audio_int16.tobytes()
    
    # Giải mã
    text = transcriber.accept_wave(stream, pcm_bytes)
    
    if text and text != current_text:
        current_text = text
        print(f"\r🗣️  Bạn nói: {text}", end="", flush=True)

try:
    # Mở Micro Input Stream
    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype='float32',
        blocksize=chunk_size,
        callback=audio_callback
    ):
        while True:
            time.sleep(0.1)
except KeyboardInterrupt:
    print("\n\n⏹️  Đã ngắt kết nối Micro.")
except Exception as e:
    print(f"\n❌ Lỗi Micro: {e}")
