"""
Transcriber — Sherpa-Onnx Online Transducer Inference Wrapper
Nhận audio PCM qua stream → trả text tiếng Việt thời gian thực.
"""
import logging
import os
import numpy as np
import sherpa_onnx

logger = logging.getLogger("asr.core.transcriber")


class Transcriber:
    def accept_wave_with_ttft(self, stream, audio_pcm: bytes, sample_rate: int = 16000):
        """
        Đẩy chunk audio PCM vào stream, đo time to first token (TTFT).
        Returns:
            (text, ttft):
                text: văn bản đã nhận diện được
                ttft: thời gian (giây) từ lúc bắt đầu đến khi sinh ra token đầu tiên
        """
        import time
        if not audio_pcm:
            return "", None

        samples = np.frombuffer(audio_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        stream.accept_waveform(sample_rate, samples)

        ttft = None
        text = ""
        start_time = time.time()
        first_token_emitted = False

        while self.recognizer.is_ready(stream):
            self.recognizer.decode_stream(stream)
            current_text = self.recognizer.get_result(stream)
            if not first_token_emitted and current_text.strip():
                ttft = time.time() - start_time
                first_token_emitted = True
                text = current_text
        # Nếu không có token nào thì ttft = None
        if not first_token_emitted:
            text = self.recognizer.get_result(stream)
        return text, ttft
    """
    Sherpa-Onnx Streaming ASR engine.
    Hỗ trợ Online (Streaming) Transducer với encoder, decoder, joiner.
    """

    def __init__(self):
        from config import config
        
        logger.info("Initializing Sherpa-Onnx OnlineRecognizer...")
        try:
            # load online transducer via factory method
            self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=config.tokens_path,
                encoder=config.encoder_path,
                decoder=config.decoder_path,
                joiner=config.joiner_path,
                num_threads=4,
                sample_rate=config.sample_rate,
                feature_dim=80,
                enable_endpoint_detection=True,
                provider=config.provider,
            )
            logger.info("✅ Sherpa-Onnx OnlineRecognizer loaded successfully!")
        except Exception as e:
            logger.error(f"❌ Failed to load Sherpa-Onnx Recognizer: {e}")
            raise e

    def create_stream(self):
        """
        Khởi tạo một online stream cho một session/cuộc gọi mới.
        """
        return self.recognizer.create_stream()

    def accept_wave(self, stream, audio_pcm: bytes, sample_rate: int = 16000) -> str:
        """
        Đẩy chunk audio PCM (16-bit, 16kHz mono) vào stream và giải mã.
        
        Returns:
            Văn bản đã nhận diện được cho stream tới thời điểm hiện tại.
        """
        if not audio_pcm:
            return ""
            
        # Convert bytes to float32 numpy array
        samples = np.frombuffer(audio_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        stream.accept_waveform(sample_rate, samples)
        
        while self.recognizer.is_ready(stream):
            self.recognizer.decode_stream(stream)
            
        return self.recognizer.get_result(stream)


    def is_endpoint(self, stream) -> bool:
        """
        Kiểm tra đã dứt lời (Endpoint) chưa.
        """
        return self.recognizer.is_endpoint(stream)

    def transcribe(self, audio_pcm: bytes, sample_rate: int = 16000) -> dict:
        """
        Hàm fallback cho chế độ Batch (được gọi từ server.py dummy).
        """
        stream = self.create_stream()
        text = self.accept_wave(stream, audio_pcm, sample_rate)
        return {
            "text": text,
            "confidence": 0.95,
        }
