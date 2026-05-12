"""
Visualise answer_synthetic_15_5.jsonl

Metric quan trọng nhất với voice streaming bot là "time to first audio":
  perceived_latency = rag_ms + ttft_ms
  (llm_ms là tổng thời gian sinh text, nhưng audio đã phát từ chunk đầu tiên rồi)

Plots:
  1. Perceived latency (RAG + TTFT) — histogram + box plot
  2. RAG vs TTFT breakdown — stacked bar per source + overall
  3. Answer length distribution — histogram + KDE
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(__file__).parent.parent / "answer_synthetic_15_5.jsonl"
OUT  = Path(__file__).parent.parent / "scripts"

records = [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()]

rag      = np.array([r["latency"]["rag_ms"]  for r in records])
ttft     = np.array([r["latency"]["ttft_ms"] for r in records])
perceived = rag + ttft   # thời gian đến khi bot bắt đầu nói
ans_words = [len(r["answer"].split()) for r in records]

SOURCE_LABELS = {
    "suckhoedoisong": "SKDS",
    "vinmec":         "Vinmec",
    "benhvienthucuc": "Thu Cúc",
    "viendinhduong":  "VDD",
}
src_order = ["suckhoedoisong", "vinmec", "benhvienthucuc", "viendinhduong"]

COLORS = {"rag": "#4C72B0", "ttft": "#DD8452", "perceived": "#C44E52"}

# ─────────────────────────────────────────────────────────────────
# FIG 1  Perceived latency (RAG + TTFT): histogram + box plot
# ─────────────────────────────────────────────────────────────────
fig1, axes = plt.subplots(2, 3, figsize=(14, 7),
                           gridspec_kw={"height_ratios": [3, 1]})
fig1.suptitle("Latency cảm nhận = RAG + TTFT  (thời gian đến khi bot bắt đầu nói)",
              fontsize=13, fontweight="bold", y=1.01)

datasets = [
    (rag,       "RAG (Qdrant retrieval)",  COLORS["rag"]),
    (ttft,      "TTFT",                    COLORS["ttft"]),
    (perceived, "RAG + TTFT (perceived)",  COLORS["perceived"]),
]

for col, (data, label, color) in enumerate(datasets):
    ax_hist = axes[0][col]
    ax_box  = axes[1][col]

    ax_hist.hist(data, bins=28, color=color, alpha=0.82, edgecolor="white", linewidth=0.4)
    med = np.median(data)
    p90 = np.percentile(data, 90)
    ax_hist.axvline(med, color="black", lw=1.4, linestyle="--", label=f"Median {med:.0f} ms")
    ax_hist.axvline(p90, color="red",   lw=1.2, linestyle=":",  label=f"P90 {p90:.0f} ms")
    ax_hist.set_title(label, fontsize=10, fontweight="bold")
    ax_hist.set_xlabel("ms")
    ax_hist.set_ylabel("Số câu")
    ax_hist.legend(fontsize=8)
    ax_hist.grid(axis="y", alpha=0.3)

    ax_box.boxplot(data, vert=False, patch_artist=True,
                   medianprops={"color": "black", "linewidth": 2},
                   boxprops={"facecolor": color, "alpha": 0.7})
    ax_box.set_yticks([])
    ax_box.set_xlabel("ms", fontsize=8)
    ax_box.grid(axis="x", alpha=0.3)

plt.tight_layout()
out1 = OUT / "fig1_latency_distribution.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved {out1}")
plt.close(fig1)

# ─────────────────────────────────────────────────────────────────
# FIG 2  RAG + TTFT breakdown per source (stacked bar)
# ─────────────────────────────────────────────────────────────────
def mean_by(arr, src=None):
    idx = [i for i, r in enumerate(records) if src is None or r["source"] == src]
    return float(np.mean(arr[idx]))

rag_means  = [mean_by(rag,  s) for s in src_order] + [float(np.mean(rag))]
ttft_means = [mean_by(ttft, s) for s in src_order] + [float(np.mean(ttft))]

x = np.arange(len(src_order) + 1)
bar_w = 0.55

fig2, ax = plt.subplots(figsize=(10, 5))
ax.bar(x, rag_means,  bar_w, label="RAG",  color=COLORS["rag"])
ax.bar(x, ttft_means, bar_w, label="TTFT", color=COLORS["ttft"], bottom=rag_means)

totals = [r + t for r, t in zip(rag_means, ttft_means)]
for xi, (r, t, tot) in enumerate(zip(rag_means, ttft_means, totals)):
    ax.text(xi, tot + 4, f"{tot:.0f} ms", ha="center", va="bottom",
            fontsize=10, fontweight="bold")
    if r > 30:
        ax.text(xi, r / 2, f"RAG\n{r:.0f}", ha="center", va="center",
                fontsize=8, color="white")
    if t > 20:
        ax.text(xi, r + t / 2, f"TTFT\n{t:.0f}", ha="center", va="center",
                fontsize=8, color="white")

tick_labels = [SOURCE_LABELS[s] for s in src_order] + ["Overall"]
ax.set_xticks(x)
ax.set_xticklabels(tick_labels, fontsize=11)
ax.set_ylabel("Latency trung bình (ms)")
ax.set_title("Perceived Latency (RAG + TTFT) theo Nguồn\n"
             "(thời gian từ khi user xong câu → bot bắt đầu phát audio)",
             fontsize=12, fontweight="bold")
ax.legend(loc="upper right", fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(totals) * 1.25)

plt.tight_layout()
out2 = OUT / "fig2_latency_breakdown.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved {out2}")
plt.close(fig2)

# ─────────────────────────────────────────────────────────────────
# FIG 3  Answer length distribution
# ─────────────────────────────────────────────────────────────────
fig3, (ax_hist, ax_src) = plt.subplots(1, 2, figsize=(13, 5))
fig3.suptitle("Phân phối Độ dài Câu trả lời", fontsize=13, fontweight="bold")

# Left: overall histogram + KDE
color_ans = "#9467BD"
ax_hist.hist(ans_words, bins=30, color=color_ans, alpha=0.75,
             edgecolor="white", linewidth=0.4, density=True, label="Histogram")

# manual KDE
from statistics import mean, stdev
bw = 1.06 * stdev(ans_words) * len(ans_words)**(-0.2)
xs = np.linspace(min(ans_words)-20, max(ans_words)+20, 300)
kde = np.array([np.mean(np.exp(-0.5*((xs[i]-np.array(ans_words))/bw)**2)/(bw*(2*np.pi)**0.5))
                for i in range(len(xs))])
ax_hist.plot(xs, kde, color="darkviolet", lw=2, label="KDE")

med_w = np.median(ans_words)
ax_hist.axvline(med_w, color="black", lw=1.5, linestyle="--", label=f"Median {med_w:.0f} từ")
ax_hist.axvline(150, color="red", lw=1.2, linestyle=":", label="Giới hạn prompt (150 từ)")
ax_hist.set_xlabel("Số từ trong câu trả lời")
ax_hist.set_ylabel("Mật độ")
ax_hist.set_title("Toàn bộ (360 câu)")
ax_hist.legend(fontsize=9)
ax_hist.grid(alpha=0.3)

# Right: violin per source
src_data = {SOURCE_LABELS[s]: [len(r["answer"].split()) for r in records if r["source"] == s]
            for s in src_order}
src_names = list(src_data.keys())
vparts = ax_src.violinplot(list(src_data.values()), positions=range(len(src_names)),
                            showmedians=True, showextrema=True)
for pc in vparts["bodies"]:
    pc.set_facecolor(color_ans)
    pc.set_alpha(0.6)
ax_src.set_xticks(range(len(src_names)))
ax_src.set_xticklabels(src_names)
ax_src.set_ylabel("Số từ")
ax_src.set_title("Violin theo Nguồn")
ax_src.axhline(150, color="red", lw=1.2, linestyle=":", label="150 từ")
ax_src.legend(fontsize=9)
ax_src.grid(axis="y", alpha=0.3)

plt.tight_layout()
out3 = OUT / "fig3_answer_length.png"
fig3.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved {out3}")
plt.close(fig3)

print("\nDone. 3 figures saved to scripts/")
