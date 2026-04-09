
import sys
import os
import time
import argparse


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.transcriber import Transcriber

try:
    import numpy as np
    import sounddevice as sd
except ImportError:
    print("Lỗi: Chưa cài đặt 'sounddevice' hoặc 'numpy'.")
    print("Hãy chạy lệnh sau ngoài terminal:")
    print("   pip install sounddevice numpy")
    sys.exit(1)

# 1. Khởi tạo Transcriber (Load model)
print("Đang tải mô hình ASR (Sherpa-Onnx)...")
try:
    transcriber = Transcriber()
except Exception as e:
    print(f"Không thể tải model: {e}")
    sys.exit(1)

# 2. Tạo Stream
stream = transcriber.create_stream()

sample_rate = 16000
chunk_size = 1024  # frames per buffer


# Argument parser for input mode
parser = argparse.ArgumentParser(description="Test ASR with Microphone or Audio File")
parser.add_argument('--file', type=str, default=None, help='Path to input wav file (mono, 16kHz)')
args = parser.parse_args()

if args.file:
    print(f"Dang nhan input tu file: {args.file}")
else:
    print("\nDang ket noi Microphone... Hay bat dau noi!")
    print("(In ket qua Real-time. An Ctrl+C de ket thuc)\n")

# Biến lưu trữ đoạn text trước đó để chỉ in phần mới (delta)
current_text = ""
first_token_printed = False


def print_transcript_update(new_text: str):
    global current_text
    if not new_text:
        return

    # Trường hợp ASR trả về text mở rộng dần: chỉ in phần mới để tránh spam log
    if new_text.startswith(current_text):
        delta = new_text[len(current_text):]
        if delta:
            if not current_text:
                print("Ban noi: ", end="", flush=True)
            print(delta, end="", flush=True)
            current_text = new_text
        return

    # Fallback nếu ASR thay đổi lại toàn bộ hypothesis
    current_text = new_text
    print(f"\nBan noi (cap nhat): {new_text}", end="", flush=True)

def audio_callback(indata, frames, time_info, status):
    global first_token_printed
    if status:
        print(f"Status: {status}", file=sys.stderr)

    # sounddevice trả về float32, ta chuyển về int16 để khớp với Transcriber
    audio_float = indata[:, 0]  # Thao tác trên kênh Mono đầu tiên
    audio_int16 = (audio_float * 32767).astype(np.int16)
    pcm_bytes = audio_int16.tobytes()

    # Giải mã và đo TTFT
    text, ttft = transcriber.accept_wave_with_ttft(stream, pcm_bytes)

    if text:
        print_transcript_update(text)
        if not first_token_printed and ttft is not None:
            print(f"\nTime to first token: {ttft:.3f} giay")
            first_token_printed = True

def process_file_input(file_path):
    import wave
    global first_token_printed
    with wave.open(file_path, 'rb') as wf:
        assert wf.getnchannels() == 1, "File phai la mono"
        assert wf.getframerate() == sample_rate, f"File phai co sample rate {sample_rate}"
        chunk = wf.readframes(chunk_size)
        while chunk:
            # chunk dang la bytes int16
            text, ttft = transcriber.accept_wave_with_ttft(stream, chunk)
            if text:
                print_transcript_update(text)
                if not first_token_printed and ttft is not None:
                    print(f"\nTime to first token: {ttft:.3f} giay")
                    first_token_printed = True
            chunk = wf.readframes(chunk_size)
    print("\n\nHoan thanh nhan dien file (ASR Goc).")
    
    try:
        print("\n[VFastPunct] Đang tải mô hình khôi phục dấu câu (có thể tốn vài giây)...")
        # Fix cho PyTorch 2.6+ (vfastpunct cũ load pickle nên bị chặn)
        import torch
        import functools
        torch.load = functools.partial(torch.load, weights_only=False)
        from vfastpunct import VFastPunct
        vfp = VFastPunct("mBertPunctCap")
        # Chuyển về chữ thường để VFastPunct hoạt động đúng
        normalized_text = current_text.lower()
        restored = vfp(normalized_text)
        print(f"\n[VFastPunct] Kết quả cuối cùng:\n{restored}\n")
    except Exception as e:
        import traceback
        print(f"\n[Lỗi VFastPunct] Không thể khôi phục dấu câu: {e}")
        traceback.print_exc()

if args.file:
    try:
        process_file_input(args.file)
    except Exception as e:
        print(f"Loi khi xu ly file: {e}")
else:
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
        print("\n\nDa ngat ket noi Micro.")
    except Exception as e:
        print(f"\nLoi Micro: {e}")
