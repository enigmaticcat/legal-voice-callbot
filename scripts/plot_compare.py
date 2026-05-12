"""
So sánh 3 cấu hình RAG:
  - fetch_k=10, top_k=3  → charts_10_3/
  - fetch_k=15, top_k=5  → charts_15_5/
  - fetch_k=20, top_k=7  → charts_20_7/
  - So sánh tổng hợp     → charts_comparison/
"""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent

CONFIGS = [
    {"label": "fetch_k=10, top_k=3", "short": "10_3", "file": "answer_synthetic_10_3.jsonl", "color": "#4C72B0"},
    {"label": "fetch_k=15, top_k=5", "short": "15_5", "file": "answer_synthetic_15_5.jsonl", "color": "#DD8452"},
    {"label": "fetch_k=20, top_k=7", "short": "20_7", "file": "answer_synthetic_20_7.jsonl", "color": "#55A868"},
]

SOURCE_LABELS = {
    "suckhoedoisong": "SKDS",
    "vinmec":         "Vinmec",
    "benhvienthucuc": "Thu Cúc",
    "viendinhduong":  "VDD",
}
SRC_ORDER = ["suckhoedoisong", "vinmec", "benhvienthucuc", "viendinhduong"]

# ── Load data ─────────────────────────────────────────────────────
datasets = []
for cfg in CONFIGS:
    path = ROOT / cfg["file"]
    records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rag  = np.array([r["latency"]["rag_ms"]  for r in records])
    ttft = np.array([r["latency"]["ttft_ms"] for r in records])
    perceived = rag + ttft
    ans_words = np.array([len(r["answer"].split()) for r in records])
    src = [r["source"] for r in records]
    datasets.append({**cfg, "records": records, "rag": rag, "ttft": ttft,
                     "perceived": perceived, "ans_words": ans_words, "src": src})


def save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path.relative_to(ROOT)}")
    plt.close(fig)


def plot_individual(ds):
    out = ROOT / "scripts" / f"charts_{ds['short']}"
    color = ds["color"]
    rag, ttft, perceived = ds["rag"], ds["ttft"], ds["perceived"]
    ans_words = ds["ans_words"]
    records = ds["records"]

    # ── Fig 1: Perceived latency distribution ─────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(14, 7),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"Latency cảm nhận — {ds['label']}", fontsize=13, fontweight="bold")
    sub = [
        (rag,       "RAG (Qdrant)",         "#4C72B0"),
        (ttft,      "TTFT",                  "#DD8452"),
        (perceived, "RAG + TTFT (perceived)", color),
    ]
    for col, (data, label, c) in enumerate(sub):
        ax_h = axes[0][col]; ax_b = axes[1][col]
        ax_h.hist(data, bins=28, color=c, alpha=0.82, edgecolor="white", linewidth=0.4)
        med = np.median(data); p90 = np.percentile(data, 90)
        ax_h.axvline(med, color="black", lw=1.4, ls="--", label=f"Median {med:.0f} ms")
        ax_h.axvline(p90, color="red",   lw=1.2, ls=":",  label=f"P90 {p90:.0f} ms")
        ax_h.set_title(label, fontsize=10, fontweight="bold")
        ax_h.set_xlabel("ms"); ax_h.set_ylabel("Số câu")
        ax_h.legend(fontsize=8); ax_h.grid(axis="y", alpha=0.3)
        ax_b.boxplot(data, vert=False, patch_artist=True,
                     medianprops={"color": "black", "linewidth": 2},
                     boxprops={"facecolor": c, "alpha": 0.7})
        ax_b.set_yticks([]); ax_b.set_xlabel("ms", fontsize=8)
        ax_b.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    save(fig, out / "fig1_latency_distribution.png")

    # ── Fig 2: Perceived latency breakdown per source ──────────────
    src = ds["src"]
    rag_means  = [np.mean(rag[[i for i,s in enumerate(src) if s == so]])  for so in SRC_ORDER]
    ttft_means = [np.mean(ttft[[i for i,s in enumerate(src) if s == so]]) for so in SRC_ORDER]
    rag_means  += [float(np.mean(rag))]
    ttft_means += [float(np.mean(ttft))]
    totals = [r+t for r,t in zip(rag_means, ttft_means)]
    x = np.arange(len(SRC_ORDER) + 1)
    bar_w = 0.55
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, rag_means,  bar_w, label="RAG",  color="#4C72B0")
    ax.bar(x, ttft_means, bar_w, label="TTFT", color="#DD8452", bottom=rag_means)
    for xi, (r, t, tot) in enumerate(zip(rag_means, ttft_means, totals)):
        ax.text(xi, tot+4, f"{tot:.0f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold")
        if r > 30: ax.text(xi, r/2,   f"RAG\n{r:.0f}",  ha="center", va="center", fontsize=8, color="white")
        if t > 20: ax.text(xi, r+t/2, f"TTFT\n{t:.0f}", ha="center", va="center", fontsize=8, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels([SOURCE_LABELS[s] for s in SRC_ORDER] + ["Overall"], fontsize=11)
    ax.set_ylabel("Latency trung bình (ms)")
    ax.set_title(f"Perceived Latency theo Nguồn — {ds['label']}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(totals) * 1.25)
    plt.tight_layout()
    save(fig, out / "fig2_latency_breakdown.png")

    # ── Fig 3: Answer length ───────────────────────────────────────
    color_ans = "#9467BD"
    fig, (ax_h, ax_v) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Phân phối Độ dài Câu trả lời — {ds['label']}", fontsize=12, fontweight="bold")
    ax_h.hist(ans_words, bins=30, color=color_ans, alpha=0.75,
              edgecolor="white", linewidth=0.4, density=True)
    bw = 1.06 * float(np.std(ans_words)) * len(ans_words)**(-0.2)
    xs = np.linspace(ans_words.min()-20, ans_words.max()+20, 300)
    kde = np.array([np.mean(np.exp(-0.5*((xs[i]-ans_words)/bw)**2)/(bw*(2*np.pi)**0.5)) for i in range(len(xs))])
    ax_h.plot(xs, kde, color="darkviolet", lw=2, label="KDE")
    med_w = float(np.median(ans_words))
    ax_h.axvline(med_w, color="black", lw=1.5, ls="--", label=f"Median {med_w:.0f} từ")
    ax_h.axvline(150, color="red", lw=1.2, ls=":", label="Giới hạn 150 từ")
    ax_h.set_xlabel("Số từ"); ax_h.set_ylabel("Mật độ")
    ax_h.set_title("Toàn bộ 360 câu"); ax_h.legend(fontsize=9); ax_h.grid(alpha=0.3)
    src_data = [ans_words[[i for i,s in enumerate(src) if s == so]] for so in SRC_ORDER]
    vp = ax_v.violinplot(src_data, positions=range(4), showmedians=True)
    for pc in vp["bodies"]: pc.set_facecolor(color_ans); pc.set_alpha(0.6)
    ax_v.set_xticks(range(4)); ax_v.set_xticklabels([SOURCE_LABELS[s] for s in SRC_ORDER])
    ax_v.axhline(150, color="red", lw=1.2, ls=":", label="150 từ")
    ax_v.set_ylabel("Số từ"); ax_v.set_title("Violin theo Nguồn")
    ax_v.legend(fontsize=9); ax_v.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save(fig, out / "fig3_answer_length.png")


# ── Vẽ từng config ───────────────────────────────────────────────
for ds in datasets:
    plot_individual(ds)

# ── So sánh tổng hợp ─────────────────────────────────────────────
out_cmp = ROOT / "scripts" / "charts_comparison"

# Fig C1: Perceived latency — 3 histogram chồng nhau
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("So sánh Perceived Latency (RAG + TTFT) — 3 cấu hình", fontsize=13, fontweight="bold")
titles = ["RAG (ms)", "TTFT (ms)", "RAG + TTFT (ms)"]
for col, key in enumerate(["rag", "ttft", "perceived"]):
    ax = axes[col]
    for ds in datasets:
        data = ds[key]
        ax.hist(data, bins=28, alpha=0.55, label=ds["label"], color=ds["color"], edgecolor="white", linewidth=0.3)
        med = np.median(data)
        ax.axvline(med, color=ds["color"], lw=2, ls="--")
    ax.set_title(titles[col], fontsize=11, fontweight="bold")
    ax.set_xlabel("ms"); ax.set_ylabel("Số câu")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
save(fig, out_cmp / "figC1_latency_histogram.png")

# Fig C2: Box plot so sánh perceived latency
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Box Plot So sánh Latency — 3 cấu hình", fontsize=13, fontweight="bold")
for col, key in enumerate(["rag", "ttft", "perceived"]):
    ax = axes[col]
    data_list = [ds[key] for ds in datasets]
    labels = [ds["short"].replace("_", "/") for ds in datasets]
    colors = [ds["color"] for ds in datasets]
    bp = ax.boxplot(data_list, patch_artist=True, vert=True,
                    medianprops={"color": "black", "linewidth": 2.5})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(titles[col], fontsize=11, fontweight="bold")
    ax.set_ylabel("ms"); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
save(fig, out_cmp / "figC2_latency_boxplot.png")

# Fig C3: Stacked bar so sánh perceived latency trung bình
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(datasets))
bar_w = 0.5
rag_means_all  = [float(np.mean(ds["rag"]))  for ds in datasets]
ttft_means_all = [float(np.mean(ds["ttft"])) for ds in datasets]
totals_all = [r+t for r,t in zip(rag_means_all, ttft_means_all)]
ax.bar(x, rag_means_all,  bar_w, label="RAG",  color="#4C72B0")
ax.bar(x, ttft_means_all, bar_w, label="TTFT", color="#DD8452", bottom=rag_means_all)
for xi, (r, t, tot) in enumerate(zip(rag_means_all, ttft_means_all, totals_all)):
    ax.text(xi, tot+3, f"{tot:.0f} ms", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.text(xi, r/2,   f"RAG\n{r:.0f}",  ha="center", va="center", fontsize=9, color="white")
    ax.text(xi, r+t/2, f"TTFT\n{t:.0f}", ha="center", va="center", fontsize=9, color="white")
ax.set_xticks(x)
ax.set_xticklabels([ds["label"] for ds in datasets], fontsize=10)
ax.set_ylabel("Latency trung bình (ms)")
ax.set_title("Perceived Latency trung bình — So sánh 3 cấu hình\n(thời gian từ user dứt câu → bot bắt đầu nói)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(totals_all) * 1.3)
plt.tight_layout()
save(fig, out_cmp / "figC3_perceived_bar.png")

# Fig C4: Answer length so sánh — violin 3 config
fig, ax = plt.subplots(figsize=(10, 5))
positions = [1, 2, 3]
vp = ax.violinplot([ds["ans_words"] for ds in datasets], positions=positions,
                   showmedians=True, showextrema=True)
for pc, ds in zip(vp["bodies"], datasets):
    pc.set_facecolor(ds["color"]); pc.set_alpha(0.65)
ax.set_xticks(positions)
ax.set_xticklabels([ds["label"] for ds in datasets], fontsize=10)
ax.axhline(150, color="red", lw=1.5, ls=":", label="Giới hạn prompt 150 từ")
for ds, pos in zip(datasets, positions):
    med = float(np.median(ds["ans_words"]))
    ax.text(pos, med+5, f"{med:.0f}", ha="center", fontsize=9, fontweight="bold", color=ds["color"])
ax.set_ylabel("Số từ trong câu trả lời")
ax.set_title("So sánh Độ dài Câu trả lời — 3 cấu hình", fontsize=12, fontweight="bold")
ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
save(fig, out_cmp / "figC4_answer_length.png")

# ── Bảng tóm tắt ─────────────────────────────────────────────────
print("\n── Tóm tắt ──────────────────────────────────────────────────")
print(f"{'Config':<22} {'RAG med':>8} {'TTFT med':>9} {'Perceived med':>14} {'P90':>7} {'Ans words med':>14}")
for ds in datasets:
    r_med = np.median(ds["rag"]); t_med = np.median(ds["ttft"])
    p_med = np.median(ds["perceived"]); p_p90 = np.percentile(ds["perceived"], 90)
    a_med = np.median(ds["ans_words"])
    print(f"{ds['label']:<22} {r_med:>7.0f}ms {t_med:>8.0f}ms {p_med:>13.0f}ms {p_p90:>6.0f}ms {a_med:>13.0f} từ")
