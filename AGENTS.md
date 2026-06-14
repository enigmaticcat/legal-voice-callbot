# AGENTS.md

## Scope

This file applies to the whole repository rooted at `/Users/nguyenthithutam/Desktop/Callbot`.

## Project Summary

This repository contains a Vietnamese nutrition voice callbot project plus research/evaluation assets for a graduation thesis.

Main application code is in `nutrition-callbot/`.

Core runtime architecture:

```text
Client Web UI
  -> Gateway FastAPI/WebSocket
  -> ASR service
  -> Brain service with RAG
  -> TTS service
  -> streamed audio back to client
```

Important services:

- `nutrition-callbot/gateway/`: FastAPI gateway, WebSocket orchestration, session memory.
- `nutrition-callbot/asr/`: ASR worker, Sherpa-ONNX transcriber, VAD endpoint.
- `nutrition-callbot/brain/`: LLM + RAG worker, query expansion, prompt building, Qdrant retrieval/reranking.
- `nutrition-callbot/tts/`: VieNeu-TTS worker, text chunking, PCM streaming.
- `nutrition-callbot/web/`: React/Vite frontend.
- `nutrition-callbot/docker-compose.yml`: main app compose file.
- Root `docker-compose.yml`: alternate compose used from repo root with snapshot restore.

## Current Domain

The active product domain is **nutrition**, not legal advice.

When editing or generating content:

- Prefer “Nutrition CallBot”, “tư vấn dinh dưỡng”, “dinh dưỡng”.
- Avoid reintroducing old legal-domain names such as `Legal CallBot`, `pháp luật`, `phap_dien_khoan`, `LEGAL_SYSTEM_PROMPT`, `legal_context`, or `legal_chunker`.
- Test/demo questions should be nutrition questions, e.g. diabetes, pregnancy nutrition, omega-3, calcium, vitamins.

## Runtime Notes

The production-facing pipeline is HTTP/WebSocket based, despite some old filenames mentioning gRPC.

Key flow:

- Frontend connects to `GET /ws/voice`.
- Gateway forwards VAD audio to ASR `/ws/transcribe/vad`.
- Gateway sends transcript to Brain `/think/stream`.
- Brain streams text chunks from RAG + LLM.
- Gateway chunks and cleans text for TTS `/speak/stream`.
- TTS returns raw PCM chunks to the browser.

ASR currently uses an offline Sherpa-ONNX recognizer. Compatibility methods exist in `nutrition-callbot/asr/core/transcriber.py` for older call sites:

- `create_stream()`
- `accept_wave()`
- `accept_wave_with_ttft()`
- `is_endpoint()`

Do not remove these without updating all ASR call sites.

## Validation Commands

Use these from `nutrition-callbot/` when changing app code:

```bash
python3 -m compileall -q gateway brain asr tts tests test_pipeline.py test_full_pipeline.py
npm --prefix web run build
```

Unit tests need dev dependency:

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pytest tests/test_unit.py -q
```

If `pytest` is unavailable, the existing unit logic can still be smoke-checked with:

```bash
python3 -c "import importlib.util; spec=importlib.util.spec_from_file_location('test_unit','tests/test_unit.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); total=0; classes=[mod.TestExpandQuery, mod.TestChunkText, mod.TestBuildPrompt]; \
for cls in classes: \
    obj=cls(); \
    [getattr(obj,name)() for name in dir(obj) if name.startswith('test_')]; \
print('unit smoke passed')"
```

Frontend build:

```bash
cd nutrition-callbot/web
npm run build
```

## Environment Notes

The local shell may print a Homebrew startup error like:

```text
Error: Invalid usage: Unknown command: brew shellnev
```

This is a user shell configuration issue outside the repo. Do not treat it as a project build failure if the command exit code is otherwise successful.

Do not print real `.env` values. `.env` files are ignored and may contain API keys.

## LaTeX / Thesis Notes

Thesis-related files:

- `Tran_Quang_Minh_DATN_Outline.tex`: generated standalone thesis outline based on the DOCX outline and SOICT template.
- `Trần Quang Minh - Outline.docx`: source outline document.
- `SOICT_DATN_Research_VIE_Template/`: SOICT Vietnamese thesis template.

If creating thesis content, keep it aligned with the nutrition voicebot topic:

```text
Hệ thống tư vấn dinh dưỡng thời gian thực qua giọng nói
dựa trên Microservices và RAG
```

## Coding Guidelines

- Keep changes focused; do not rewrite unrelated notebooks/data files.
- Prefer fixing root causes over patching symptoms.
- Maintain existing Python/React style.
- Avoid adding broad dependencies unless necessary.
- Do not commit changes unless explicitly asked.
- Do not touch unrelated user-modified files such as `.xlsx`, `.docx`, or generated evaluation artifacts unless asked.

## Known Important Files

- `nutrition-callbot/brain/core/query_expander.py`: nutrition alias expansion.
- `nutrition-callbot/brain/core/prompt.py`: nutrition system prompt and few-shot examples.
- `nutrition-callbot/brain/core/rag.py`: Qdrant retrieval and reranking.
- `nutrition-callbot/gateway/services/orchestrator.py`: ASR/Brain/TTS streaming orchestration.
- `nutrition-callbot/gateway/routes/websocket.py`: browser voice WebSocket route.
- `nutrition-callbot/tts/core/chunker.py`: TTS sentence-safe chunking.
- `nutrition-callbot/tts/core/synthesizer.py`: VieNeu-TTS wrapper.
- `nutrition-callbot/asr/core/transcriber.py`: Sherpa-ONNX transcriber.

## Before Final Response

When modifying code, report:

- Files changed.
- Validation commands run.
- Any validation that could not run and why.
- Any unrelated pre-existing dirty files observed in `git status`.
