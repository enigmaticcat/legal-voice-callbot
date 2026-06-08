"""
Download 6000 samples từ doof-ferb/vlsp2020_vinai_100h
Output: evaluation/asr_benchmark_samples/  (WAV files)
        evaluation/asr_benchmark_samples.jsonl  (metadata + transcript)

Chạy:
    python scripts/download_asr_benchmark_data.py
"""

import io
import json
import numpy as np
import soundfile as sf
from pathlib import Path

N_SAMPLES  = 6000

OUT_DIR   = Path("evaluation/asr_benchmark_samples")
OUT_JSONL = Path("evaluation/asr_benchmark_samples.jsonl")

OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading dataset (streaming, first {N_SAMPLES} samples)...")
from datasets import load_dataset, Audio

ds = load_dataset(
    "doof-ferb/vlsp2020_vinai_100h",
    split="train",
    streaming=True,
    trust_remote_code=True,
)
ds = ds.cast_column("audio", Audio(decode=False))

records = []
for i, s in enumerate(ds):
    if i >= N_SAMPLES:
        break

    audio = s["audio"]
    raw = audio.get("bytes") or open(audio["path"], "rb").read()
    arr, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    dur = round(len(arr) / sr, 3)

    wav_name = f"sample_{i:04d}.wav"
    sf.write(str(OUT_DIR / wav_name), arr, sr, subtype="PCM_16")

    records.append({
        "idx":      i,
        "wav_file": wav_name,
        "duration": dur,
        "sr":       sr,
        "sentence": s.get("sentence", ""),
    })

    if (i + 1) % 500 == 0:
        print(f"  saved {i+1}/{N_SAMPLES}")

with open(OUT_JSONL, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

durations = [r["duration"] for r in records]
print(f"\nDone. {len(records)} samples")
print(f"Duration — min={min(durations):.1f}s  mean={np.mean(durations):.1f}s  max={max(durations):.1f}s")
print(f"Total   : {sum(durations)/60:.1f} min audio")
print(f"WAV     : {OUT_DIR}/")
print(f"Metadata: {OUT_JSONL}")
