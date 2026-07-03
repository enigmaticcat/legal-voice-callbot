# Mermaid flow diagrams

Hai file Mermaid trong thư mục này thay cho sơ đồ luồng hỏi đáp chính quá dài:

- `01_asr_tao_transcript.mmd`: luồng thu âm, ASR, VAD và tạo transcript.
- `02_rag_tts_streaming.mmd`: luồng xử lý hội thoại, cache, RAG, LLM và TTS streaming.

Gợi ý render:

```bash
mmdc -i 01_asr_tao_transcript.mmd -o 01_asr_tao_transcript.png -b white
mmdc -i 02_rag_tts_streaming.mmd -o 02_rag_tts_streaming.png -b white
```
