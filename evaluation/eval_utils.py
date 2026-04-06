"""
Shared utilities for RAGAS evaluation scripts.
"""
import json
from pathlib import Path
from typing import List, Dict


# ── WER / CER ────────────────────────────────────────────────────────────────

def _edit_distance(seq_a: list, seq_b: list) -> int:
    m, n = len(seq_a), len(seq_b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate (syllable/word level, split on spaces)."""
    ref = reference.lower().strip().split()
    hyp = hypothesis.lower().strip().split()
    if not ref:
        return 0.0
    return _edit_distance(ref, hyp) / len(ref)


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate (ignores spaces)."""
    ref = list(reference.lower().replace(" ", ""))
    hyp = list(hypothesis.lower().replace(" ", ""))
    if not ref:
        return 0.0
    return _edit_distance(ref, hyp) / len(ref)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_eval_split(jsonl_path: Path) -> List[Dict]:
    """
    Load eval_split_{N}.jsonl.
    Each line: {id, split, source, question, answer, ...}
    """
    samples = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def load_results(jsonl_path: Path) -> List[Dict]:
    """Load previously saved result JSONL."""
    results = []
    if not jsonl_path.exists():
        return results
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


# ── Timing statistics ─────────────────────────────────────────────────────────

def timing_stats(values: List[float]) -> Dict:
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    return {
        "avg": round(sum(s) / n, 1),
        "p50": round(s[n // 2], 1),
        "p95": round(s[min(int(0.95 * n), n - 1)], 1),
        "min": round(s[0], 1),
        "max": round(s[-1], 1),
    }


def print_timing_table(results: List[Dict], extra_keys: List[str] = None):
    """Print a timing summary table."""
    keys = ["rag_ms", "llm_ttft_ms", "total_ms"]
    if extra_keys:
        keys = extra_keys + keys

    print(f"\n{'Metric':<20} {'avg':>8} {'p50':>8} {'p95':>8} {'min':>8} {'max':>8}")
    print("-" * 60)
    for key in keys:
        vals = [r["timing"][key] for r in results if key in r.get("timing", {})]
        if not vals:
            continue
        stats = timing_stats(vals)
        print(
            f"{key:<20} {stats['avg']:>8} {stats['p50']:>8} "
            f"{stats['p95']:>8} {stats['min']:>8} {stats['max']:>8}"
        )


# ── RAGAS helpers ─────────────────────────────────────────────────────────────

def build_ragas_dataset(results: List[Dict], question_key: str = "question"):
    """
    Build RAGAS EvaluationDataset from eval results.
    Each result must have: question (or asr_transcript), answer, contexts, reference.

    question_key: "question" for LLM eval, "asr_transcript" for pipeline eval
    """
    try:
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    except ImportError:
        raise ImportError(
            "RAGAS not installed. Run:\n"
            "  pip install ragas>=0.2.0 langchain-google-genai langchain"
        )

    samples = [
        SingleTurnSample(
            user_input=r[question_key],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in results
        if r.get(question_key) and r.get("answer") and r.get("contexts") and r.get("reference")
    ]
    return EvaluationDataset(samples=samples)


def get_ragas_judge(api_key: str, model: str = "gemini-2.0-flash"):
    """Return RAGAS-wrapped LLM + embeddings using Gemini."""
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

    llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0)
    )
    embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", google_api_key=api_key
        )
    )
    return llm, embeddings


def run_ragas(results: List[Dict], api_key: str, question_key: str = "question") -> Dict:
    """
    Run RAGAS evaluation. Returns dict of metric → score.
    Metrics: answer_relevancy, faithfulness, context_precision, context_recall
    """
    from ragas import evaluate
    from ragas.metrics import AnswerRelevancy, Faithfulness, ContextPrecision, ContextRecall

    dataset = build_ragas_dataset(results, question_key=question_key)
    llm, embeddings = get_ragas_judge(api_key)

    metrics = [
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        Faithfulness(llm=llm),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]

    eval_result = evaluate(dataset=dataset, metrics=metrics)
    df = eval_result.to_pandas()

    # Aggregate: mean per metric
    metric_cols = [c for c in df.columns if c not in ("user_input", "response", "retrieved_contexts", "reference")]
    summary = {col: round(float(df[col].mean()), 4) for col in metric_cols if not df[col].isna().all()}

    # Per-sample scores (attach back to results by position)
    for i, r in enumerate(results):
        if i < len(df):
            r["ragas"] = {col: (round(float(df.iloc[i][col]), 4) if not df.iloc[i][col] != df.iloc[i][col] else None)
                          for col in metric_cols}

    return summary
