# ==============================================================
# CELL 1: Cài dependencies
# ==============================================================
# !pip install -q vllm qdrant-client sentence-transformers \
#              ragas langchain-google-vertexai langchain langchain-core


# ==============================================================
# CELL 2: Xac thuc GCP — chay cell nay TRUOC, accept ALL permissions
# ==============================================================
from google.colab import auth
auth.authenticate_user(project_id=GCP_PROJECT)


# ==============================================================
# CELL 3 (phu): Mount Google Drive
# ==============================================================
from google.colab import drive
drive.mount('/content/drive')


# ==============================================================
# CELL 3: CONFIG
# ==============================================================

# Vertex AI (judge cho RAGAS)
GCP_PROJECT     = "your-gcp-project-id"
GCP_LOCATION    = "us-central1"
GEMINI_MODEL    = "gemini-2.5-pro"

# vLLM (generate answers)
VLLM_MODEL      = "Qwen/Qwen3-4B-Instruct-2507"
VLLM_PORT       = 8000

# Qdrant (subprocess, for rag_ms measurement)
SNAPSHOT_PATH   = "/content/drive/MyDrive/Nutrition data/nutrition_articles-2744933042503761-2026-03-30-08-11-07.snapshot"
COLLECTION      = "nutrition_articles"
TOP_K           = 5
QUERY_PREFIX    = "Instruct: Tim thong tin dinh duong lien quan\nQuery: "
EMBED_MODEL     = "intfloat/multilingual-e5-large-instruct"

# File paths
CONTEXTS_FILE   = "/content/drive/MyDrive/Nutrition data/eval_with_contexts_2.jsonl"
ANSWERS_FILE    = "/content/drive/MyDrive/Nutrition data/eval_answers.jsonl"
RAGAS_SUMMARY   = "/content/drive/MyDrive/Nutrition data/ragas_summary.json"
RAGAS_DETAIL    = "/content/drive/MyDrive/Nutrition data/ragas_detail.csv"
RECALL_SCREEN   = "/content/drive/MyDrive/Nutrition data/recall_screen.csv"
GAP_QUESTIONS   = "/content/drive/MyDrive/Nutrition data/gap_questions.jsonl"

RESUME = True

# Test nhanh: dat so nho (vd: 10) de check RAGAS chay dung truoc khi chay full
RAGAS_SAMPLE_LIMIT = None   # None = chay toan bo

# Nguong phan loai "answerable" vs "corpus gap"
CORPUS_GAP_THRESHOLD = 0.3  # sample co context_recall < nguong nay bi coi la corpus gap


# ==============================================================
# CELL 4: Khoi dong Qdrant tu snapshot (de do rag_ms)
# ==============================================================
import os, subprocess, requests, time

os.system("curl -L https://github.com/qdrant/qdrant/releases/latest/download/"
          "qdrant-x86_64-unknown-linux-musl.tar.gz -o /content/qdrant.tar.gz")
os.system("tar -xzf /content/qdrant.tar.gz -C /content/ qdrant")
os.system("chmod +x /content/qdrant")

qdrant_proc = subprocess.Popen(
    ["/content/qdrant"],
    stdout=open("/content/qdrant.log", "w"),
    stderr=subprocess.STDOUT,
)
time.sleep(5)

r = requests.get("http://localhost:6333/healthz")
print(f"Qdrant: {r.status_code} (PID={qdrant_proc.pid})")

requests.delete(f"http://localhost:6333/collections/{COLLECTION}")

print(f"Restore snapshot '{COLLECTION}'...")
with open(SNAPSHOT_PATH, "rb") as f:
    resp = requests.post(
        f"http://localhost:6333/collections/{COLLECTION}/snapshots/upload"
        f"?priority=snapshot",
        files={"snapshot": f},
        timeout=600,
    )
assert resp.status_code == 200, f"Upload failed: {resp.text}"

for _ in range(60):
    try:
        info   = requests.get(f"http://localhost:6333/collections/{COLLECTION}").json()
        status = info["result"]["status"]
        count  = info["result"]["points_count"]
        if status == "green":
            print(f"  '{COLLECTION}': {count:,} vectors — OK")
            break
    except:
        pass
    time.sleep(5)


# ==============================================================
# CELL 5: Khoi dong vLLM
# ==============================================================
import sys

os.system("pkill -f vllm.entrypoints 2>/dev/null")
time.sleep(2)

vllm_proc = subprocess.Popen(
    [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model",                  VLLM_MODEL,
        "--port",                   str(VLLM_PORT),
        "--max-model-len",          "8192",
        "--gpu-memory-utilization", "0.85",
    ],
    stdout=open("/content/vllm.log", "w"),
    stderr=subprocess.STDOUT,
)

print("Cho vLLM san sang (~3-5 phut)...")
for i in range(120):
    try:
        if requests.get(f"http://localhost:{VLLM_PORT}/health", timeout=2).status_code == 200:
            print(f"vLLM ready! ({i*5}s, PID={vllm_proc.pid})")
            break
    except:
        pass
    time.sleep(5)
else:
    print("TIMEOUT — xem /content/vllm.log")

os.system("tail -15 /content/vllm.log")


# ==============================================================
# CELL 6: Init clients (embed model + Qdrant + vLLM)
# ==============================================================
import json
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

embed_model   = SentenceTransformer(EMBED_MODEL)
qdrant_client = QdrantClient(url="http://localhost:6333", timeout=60)
client        = OpenAI(base_url=f"http://localhost:{VLLM_PORT}/v1", api_key="EMPTY")

print(f"Embed model loaded: {EMBED_MODEL}")
print(f"Qdrant client connected")
print(f"vLLM client ready at port {VLLM_PORT}")

NUTRITION_SYSTEM_PROMPT = (
    "Ban la chuyen gia tu van dinh duong qua giong noi. Tuan thu:\n"
    "1. Dua vao tai lieu: Tra loi dua tren thong tin dinh duong duoc cung cap.\n"
    "2. Phong cach bac si: Bat dau bang 'Chao ban,', tu van nhu chuyen gia dinh duong.\n"
    "3. Ngan gon, de nghe: Cau tra loi se duoc doc thanh giong noi — dung cau ngan.\n"
    "4. Trung thuc: Neu khong co thong tin → noi ro 'Toi khong co thong tin ve van de nay'.\n"
    "5. Disclaimer: Ket thuc bang 'De duoc tu van chinh xac, ban nen gap bac si dinh duong.'\n"
    "/no_think"
)

def build_prompt(question: str, contexts: list) -> str:
    context_str = "\n\n".join(contexts)
    return (
        f"Tai lieu dinh duong lien quan:\n{context_str}\n"
        f"---\n"
        f"Hay tra loi DUA TREN cac tai lieu tren.\n"
        f"Cau hoi: {question}"
    )


# ==============================================================
# CELL 7: Warmup — Qdrant + vLLM (dung contexts tu file)
# ==============================================================
warmup_samples = []
with open(CONTEXTS_FILE, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            warmup_samples.append(json.loads(line))
        if len(warmup_samples) >= 10:
            break

print(f"Warmup {len(warmup_samples)} cau (Qdrant + vLLM)...")
for i, s in enumerate(warmup_samples):
    try:
        # Warmup Qdrant
        q_vec = embed_model.encode(
            [QUERY_PREFIX + s["question"]], normalize_embeddings=True
        )[0].tolist()
        qdrant_client.query_points(
            collection_name=COLLECTION, query=q_vec,
            limit=TOP_K, with_payload=False,
        )
        # Warmup vLLM (dung contexts tu file, khong re-query)
        client.chat.completions.create(
            model=VLLM_MODEL,
            messages=[
                {"role": "system", "content": NUTRITION_SYSTEM_PROMPT},
                {"role": "user",   "content": build_prompt(s["question"], s["contexts"])},
            ],
            temperature=0.0,
            max_tokens=128,
        )
        print(f"  [{i+1}/{len(warmup_samples)}] warmup OK")
    except Exception as e:
        print(f"  [{i+1}/{len(warmup_samples)}] warmup ERROR: {e}")

print("Warmup xong.\n")


# ==============================================================
# CELL 8: Generate answers + do latency (full pipeline)
# - rag_ms : embed + query Qdrant live → lay contexts thuc su
# - ttft_ms: time to first token tu vLLM
# - llm_ms : tong thoi gian LLM
# - LLM dung contexts LAY TU QDRANT (khong dung pre-fetched)
# ==============================================================
from pathlib import Path

# Chi can question + reference tu CONTEXTS_FILE, contexts lay tu Qdrant
samples = [json.loads(l) for l in open(CONTEXTS_FILE, encoding="utf-8") if l.strip()]
print(f"Loaded {len(samples)} samples")

done_ids = set()
if RESUME and Path(ANSWERS_FILE).exists():
    with open(ANSWERS_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                done_ids.add(json.loads(line)["id"])
    print(f"Resume: {len(done_ids)} cau da co answer.")

pending = [s for s in samples if s["id"] not in done_ids]
print(f"Can generate: {len(pending)} cau\n")

total_written = 0
with open(ANSWERS_FILE, "a", encoding="utf-8") as fout:
    for i, sample in enumerate(pending):
        print(f"  [{i+1}/{len(pending)}] {sample['id']} ...", end="", flush=True)
        try:
            # RAG: embed + query Qdrant → lay contexts thuc su
            t_rag = time.perf_counter()
            q_vec = embed_model.encode(
                [QUERY_PREFIX + sample["question"]], normalize_embeddings=True
            )[0].tolist()
            hits = qdrant_client.query_points(
                collection_name=COLLECTION, query=q_vec,
                limit=TOP_K, with_payload=True,
            )
            rag_ms   = (time.perf_counter() - t_rag) * 1000
            contexts = [p.payload.get("text", "") for p in hits.points]

            # Do LLM + TTFT (stream=True)
            t_llm   = time.perf_counter()
            ttft_ms = None
            parts   = []

            stream = client.chat.completions.create(
                model=VLLM_MODEL,
                messages=[
                    {"role": "system", "content": NUTRITION_SYSTEM_PROMPT},
                    {"role": "user",   "content": build_prompt(sample["question"], contexts)},
                ],
                temperature=0.0,
                max_tokens=1024,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta and ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t_llm) * 1000
                parts.append(delta)

            llm_ms   = (time.perf_counter() - t_llm) * 1000
            total_ms = rag_ms + llm_ms
            answer   = "".join(parts).strip()

            record = {
                "id"       : sample["id"],
                "split"    : sample.get("split"),
                "source"   : sample.get("source"),
                "question" : sample["question"],
                "answer"   : answer,
                "contexts" : contexts,
                "reference": sample["reference"],
                "latency"  : {
                    "rag_ms"  : round(rag_ms, 1),
                    "ttft_ms" : round(ttft_ms, 1) if ttft_ms else None,
                    "llm_ms"  : round(llm_ms, 1),
                    "total_ms": round(total_ms, 1),
                },
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            done_ids.add(sample["id"])
            total_written += 1
            print(
                f" OK | rag={rag_ms:.0f}ms ttft={ttft_ms:.0f}ms"
                f" llm={llm_ms:.0f}ms total={total_ms:.0f}ms ({len(answer)} chars)"
            )

        except Exception as e:
            print(f" ERROR: {e}")
            time.sleep(2)

print(f"\nDone. {total_written} answers -> {ANSWERS_FILE}")

# os.system("pkill -f vllm.entrypoints")


# ==============================================================
# CELL 9: RAGAS Evaluation — judge bang Gemini, embed bang e5 (da load san)
# ==============================================================
import json
from ragas import evaluate
from ragas.metrics import AnswerCorrectness, Faithfulness, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import SingleTurnSample, EvaluationDataset
from ragas.run_config import RunConfig
from langchain_core.embeddings import Embeddings
from langchain_google_vertexai import ChatVertexAI

vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)

# Judge LLM
judge_llm = LangchainLLMWrapper(
    ChatVertexAI(model=GEMINI_MODEL, project=GCP_PROJECT, location=GCP_LOCATION, temperature=0)
)

# Reuse embed_model tu Cell 6 — khong load lai, nhat quan voi retrieval
class E5Embeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_model.encode(texts, normalize_embeddings=True).tolist()
    def embed_query(self, text: str) -> list[float]:
        return embed_model.encode([text], normalize_embeddings=True)[0].tolist()

judge_embeddings = LangchainEmbeddingsWrapper(E5Embeddings())

results = [json.loads(l) for l in open(ANSWERS_FILE, encoding="utf-8") if l.strip()]
if RAGAS_SAMPLE_LIMIT:
    results = results[:RAGAS_SAMPLE_LIMIT]
    print(f"TEST MODE: chi chay {len(results)} samples")
else:
    print(f"Loaded {len(results)} answers")

ragas_samples = [
    SingleTurnSample(
        user_input=r["question"],
        response=r["answer"],
        retrieved_contexts=r["contexts"],
        reference=r["reference"],
    )
    for r in results
    if r.get("answer") and r.get("contexts") and r.get("reference")
]
dataset = EvaluationDataset(samples=ragas_samples)
print(f"RAGAS dataset: {len(ragas_samples)} samples\n")

metrics = [
    AnswerCorrectness(),  # answer dung so voi ground truth (LLM + embedding)
    Faithfulness(),       # answer co bia so voi context khong (LLM)
    ContextRecall(),      # context co cover ground truth khong (LLM)
    ContextPrecision(),   # context retrieve co relevant khong (LLM)
]

print("Running RAGAS...")
eval_result = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=judge_llm,
    embeddings=judge_embeddings,
    run_config=RunConfig(timeout=120, max_retries=3, max_wait=60),
)
df = eval_result.to_pandas()

import pandas as pd

metric_cols = [
    c for c in df.columns
    if c not in ("user_input", "response", "retrieved_contexts", "reference")
]

def compute_summary(data):
    return {
        col: round(float(data[col].mean()), 4)
        for col in metric_cols if not data[col].isna().all()
    }

def print_scores(label, data, cols):
    print(f"\n=== {label} (n={len(data)}) ===")
    for col in cols:
        if data[col].isna().all():
            continue
        stats = data[col].agg(["mean", "std", "min", "max"])
        nans  = data[col].isna().sum()
        bar   = "█" * int(stats["mean"] * 20)
        print(f"  {col:<25} {stats['mean']:.4f} ± {stats['std']:.3f}"
              f"  [{stats['min']:.3f}-{stats['max']:.3f}]  {bar}"
              + (f"  ({nans} NaN)" if nans else ""))

df_answerable = df[df["context_recall"] >= CORPUS_GAP_THRESHOLD]
n_gap = (df["context_recall"] < CORPUS_GAP_THRESHOLD).sum()

print_scores("RAGAS Scores — ALL samples", df, metric_cols)
print_scores(f"RAGAS Scores — ANSWERABLE (context_recall >= {CORPUS_GAP_THRESHOLD})", df_answerable, metric_cols)
print(f"\nCorpus coverage: {len(df_answerable)}/{len(df)} samples ({len(df_answerable)/len(df)*100:.0f}%) answerable")
print(f"Corpus gap     : {n_gap}/{len(df)} samples ({n_gap/len(df)*100:.0f}%) gap")

print("\n=== 5 MAU CONTEXT_RECALL THAP NHAT ===")
worst = df.nsmallest(5, "context_recall")
for _, row in worst.iterrows():
    ac = row.get("answer_correctness")
    ac_str = f"{ac:.3f}" if not pd.isna(ac) else "NaN"
    print(f"\n  Q : {row['user_input'][:120]}")
    print(f"  Ref: {str(row['reference'])[:200]}")
    print(f"  context_recall={row['context_recall']:.3f}  answer_correctness={ac_str}")
    for i, c in enumerate(row["retrieved_contexts"]):
        print(f"  [{i+1}] {str(c)[:120]}")

summary           = compute_summary(df)
summary_answerable = compute_summary(df_answerable) if not df_answerable.empty else {}

summary_data = {
    "n_samples"        : len(ragas_samples),
    "n_answerable"     : len(df_answerable),
    "corpus_coverage"  : round(len(df_answerable) / len(df), 4),
    "gap_threshold"    : CORPUS_GAP_THRESHOLD,
    "gen_model"        : VLLM_MODEL,
    "judge_model"      : f"vertexai/{GEMINI_MODEL}",
    "embed_model"      : EMBED_MODEL,
    "ragas_all"        : summary,
    "ragas_answerable" : summary_answerable,
}
with open(RAGAS_SUMMARY, "w", encoding="utf-8") as f:
    json.dump(summary_data, f, ensure_ascii=False, indent=2)
print(f"Summary -> {RAGAS_SUMMARY}")

df.to_csv(RAGAS_DETAIL, index=False, encoding="utf-8")
print(f"Detail  -> {RAGAS_DETAIL}")

# %% [markdown]
# ## Cell 10: Context Recall Screening — lọc corpus gap trước khi chạy full RAGAS

# %% [code]
import json, pandas as pd
from ragas import evaluate
from ragas.metrics import ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas import SingleTurnSample, EvaluationDataset
from ragas.run_config import RunConfig
from langchain_google_vertexai import ChatVertexAI

# Judge LLM (embeddings không cần cho ContextRecall)
_judge_llm = LangchainLLMWrapper(
    ChatVertexAI(model=GEMINI_MODEL, project=GCP_PROJECT, location=GCP_LOCATION, temperature=0)
)

# Load answers
_results = [json.loads(l) for l in open(ANSWERS_FILE, encoding="utf-8") if l.strip()]
if RAGAS_SAMPLE_LIMIT:
    _results = _results[:RAGAS_SAMPLE_LIMIT]
    print(f"TEST MODE: {len(_results)} samples")
else:
    print(f"Loaded {len(_results)} answers")

_samples = [
    SingleTurnSample(
        user_input=r["question"],
        response=r["answer"],
        retrieved_contexts=r["contexts"],
        reference=r["reference"],
    )
    for r in _results
    if r.get("answer") and r.get("contexts") and r.get("reference")
]
print(f"Dataset: {len(_samples)} samples\n")

# Chạy chỉ ContextRecall
print("Running ContextRecall screening...")
_recall_result = evaluate(
    dataset=EvaluationDataset(samples=_samples),
    metrics=[ContextRecall()],
    llm=_judge_llm,
    run_config=RunConfig(timeout=120, max_retries=3, max_wait=60),
)
df_recall = _recall_result.to_pandas()

# Phân loại theo threshold
df_recall["is_answerable"] = df_recall["context_recall"] >= CORPUS_GAP_THRESHOLD
n_answerable = df_recall["is_answerable"].sum()
n_gap        = (~df_recall["is_answerable"]).sum()

print(f"\n=== Context Recall Screening (threshold={CORPUS_GAP_THRESHOLD}) ===")
print(f"  Mean context_recall : {df_recall['context_recall'].mean():.4f}")
print(f"  Answerable          : {n_answerable}/{len(df_recall)} ({n_answerable/len(df_recall)*100:.0f}%)")
print(f"  Corpus gap          : {n_gap}/{len(df_recall)} ({n_gap/len(df_recall)*100:.0f}%)")

# Gắn lại question/reference để dễ xem
df_recall["question"]  = [s.user_input for s in _samples]
df_recall["reference"] = [s.reference  for s in _samples]

# In gap questions để biết cần bổ sung corpus gì
df_gap_rows = df_recall[~df_recall["is_answerable"]].sort_values("context_recall")
print(f"\n=== {len(df_gap_rows)} CÂU HỎI CORPUS GAP (context_recall < {CORPUS_GAP_THRESHOLD}) ===")
for i, (_, row) in enumerate(df_gap_rows.iterrows(), 1):
    print(f"\n  [{i:>3}] recall={row['context_recall']:.3f}  Q: {row['question'][:120]}")
    print(f"         Ref: {str(row['reference'])[:150]}")

# Samples đủ điều kiện để chạy full RAGAS ở cell tiếp theo
answerable_indices = df_recall[df_recall["is_answerable"]].index.tolist()
samples_answerable = [_samples[i] for i in answerable_indices]
print(f"\n→ {len(samples_answerable)} samples sẵn sàng cho full RAGAS evaluation")

# Lưu file để bổ sung corpus sau
df_recall[["question", "reference", "context_recall", "is_answerable"]].to_csv(
    RECALL_SCREEN, index=False, encoding="utf-8"
)
with open(GAP_QUESTIONS, "w", encoding="utf-8") as f:
    for _, row in df_gap_rows.iterrows():
        json.dump({"question": row["question"], "reference": row["reference"],
                   "context_recall": round(float(row["context_recall"]), 4)}, f, ensure_ascii=False)
        f.write("\n")
print(f"\nRecall screen -> {RECALL_SCREEN}")
print(f"Gap questions -> {GAP_QUESTIONS}  ({len(df_gap_rows)} câu cần bổ sung corpus)")
