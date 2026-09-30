"""M6: the two figures, drawn only from results/results.json."""

import json

import matplotlib.pyplot as plt
import numpy as np

from ladder import ROOT, style
from ladder.style import BASELINE, CROSS_ENCODER, FIRST_STAGE, INK, INK_MUTED, INK_SECONDARY, JEV, SURFACE, title

style.apply()
with open(ROOT / "results" / "results.json") as f:
    results = json.load(f)
figures = ROOT / "figures"
figures.mkdir(exist_ok=True)


def dot_with_ci(ax, y, ci, color):
    ax.plot([ci["ci_low"], ci["ci_high"]], [y, y], color=color, lw=2, solid_capstyle="round", zorder=2)
    ax.scatter(ci["mean"], y, s=64, color=color, edgecolors=SURFACE, linewidths=2, zorder=3)


ladder = [
    ("BM25", "bm25", FIRST_STAGE),
    ("Dense (bge-small)", "dense", FIRST_STAGE),
    ("Hybrid (BM25 + dense, RRF)", "hybrid", FIRST_STAGE),
    ("Hybrid top 30, cross-encoder rerank", "ce", CROSS_ENCODER),
    ("Hybrid fused with cross-encoder", "hybrid+ce", CROSS_ENCODER),
    ("Hybrid top 30, Jev rerank", "jev", JEV),
    ("Hybrid fused with Jev", "hybrid+jev", JEV),
]
differences = [
    ("Jev rerank vs cross-encoder rerank", "jev - ce"),
    ("Jev rerank vs hybrid", "jev - hybrid"),
    ("Cross-encoder rerank vs hybrid", "ce - hybrid"),
    ("Fusing hybrid into Jev", "hybrid+jev - jev"),
    ("Fusing hybrid into cross-encoder", "hybrid+ce - ce"),
]

fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
title(fig, "Jev vs a local cross-encoder on SciFact (300 test queries, human relevance labels)",
      "NDCG@10, mean and 95% paired-bootstrap CI. Both rerankers reorder the same hybrid top 30.")

for i, (label, key, color) in enumerate(ladder):
    ci = results["systems"][key]["ndcg@10"]
    y = len(ladder) - 1 - i
    dot_with_ci(left, y, ci, color)
    left.text(ci["ci_high"] + 0.004, y, f"{ci['mean']:.3f}", va="center", fontsize=9, color=INK)
left.set_yticks(range(len(ladder)), [label for label, _, _ in reversed(ladder)])
left.set_xlim(0.62, 0.86)
left.set_xlabel("NDCG@10")
left.grid(axis="y", visible=False)
left.tick_params(axis="y", length=0)
handles = [plt.Line2D([], [], marker="o", color=c, lw=2, markersize=7, markeredgecolor=SURFACE)
           for c in (FIRST_STAGE, CROSS_ENCODER, JEV)]
fig.legend(handles, ["First stage", "Cross-encoder (bge-reranker-v2-m3)", "Jev (jev-1.13)"],
           loc="upper left", bbox_to_anchor=(0.015, 0.875), ncol=3, frameon=False, fontsize=9,
           labelcolor=INK_SECONDARY)

right.axvline(0, color=BASELINE, lw=1, zorder=1)
for i, (label, key) in enumerate(differences):
    ci = results["deltas"][key]["ndcg@10"]
    y = len(differences) - 1 - i
    dot_with_ci(right, y, ci, INK_SECONDARY)
    # round then add 0.0 so a tiny negative mean prints as +0.000, not -0.000
    right.text(ci["ci_high"] + 0.003, y, f"{round(ci['mean'], 3) + 0.0:+.3f}", va="center",
               fontsize=9, color=INK)
right.set_yticks(range(len(differences)), [label for label, _ in reversed(differences)])
right.set_xlim(-0.06, 0.12)
right.set_xlabel("Paired difference in NDCG@10")
right.set_title("Right of 0: the first system is better", loc="left", fontsize=9, color=INK_MUTED)
right.grid(axis="y", visible=False)
right.tick_params(axis="y", length=0)

fig.text(0.02, 0.015, "Code and cached API responses in the repo.", fontsize=8, color=INK_MUTED)
fig.subplots_adjust(left=0.2, right=0.985, top=0.78, bottom=0.14, wspace=0.62)
fig.savefig(figures / "ndcg_ladder.png", dpi=200)

thresholds = results["thresholds"]
fig, ax = plt.subplots(figsize=(8, 5.2))
title(fig, "What survives a score threshold (SciFact test, 300 queries)",
      "Precision of the kept passages vs how many are kept, pooled over queries (about 1.1 relevant per query).")
for name, color, label in [("ce", CROSS_ENCODER, "Cross-encoder"), ("jev", JEV, "Jev")]:
    curve = thresholds["curves"][name]
    kept, precision = np.array(curve["kept_per_query"]), np.array(curve["precision"])
    visible = kept <= 6
    ax.plot(kept[visible], precision[visible], color=color, lw=2, label=label)
    point = thresholds["frozen_on_dev"][name]
    ax.scatter(point["kept_per_query"], point["precision"], s=64, color=color,
               edgecolors=SURFACE, linewidths=2, zorder=3)

jev_point = thresholds["frozen_on_dev"]["jev"]
ce_point = thresholds["frozen_on_dev"]["ce"]
ax.annotate(f"Jev, threshold tuned on dev: {jev_point['precision']:.0%} precise, "
            f"{jev_point['kept_per_query']:.2f} kept/query",
            (jev_point["kept_per_query"], jev_point["precision"]), xytext=(1.3, 0.8),
            fontsize=9, color=INK, arrowprops={"arrowstyle": "-", "color": INK_MUTED, "lw": 1})
ax.annotate(f"Cross-encoder, tuned on dev: {ce_point['precision']:.0%} precise, "
            f"{ce_point['kept_per_query']:.2f} kept/query",
            (ce_point["kept_per_query"], ce_point["precision"]), xytext=(1.3, 0.62),
            fontsize=9, color=INK, arrowprops={"arrowstyle": "-", "color": INK_MUTED, "lw": 1})
gptr = thresholds["jev_at_gpt_researcher_1.5"]
ax.scatter(gptr["kept_per_query"], gptr["precision"], s=64, facecolors=SURFACE, edgecolors=JEV,
           linewidths=2, zorder=3)
ax.annotate("Jev at GPT Researcher's cutoff (1.5)", (gptr["kept_per_query"], gptr["precision"]),
            xytext=(3.4, 0.42), fontsize=9, color=INK,
            arrowprops={"arrowstyle": "-", "color": INK_MUTED, "lw": 1})
ax.set_xlim(0, 6)
ax.set_ylim(0, 1)
ax.set_xlabel("Passages kept per query")
ax.set_ylabel("Precision of kept passages")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
fig.subplots_adjust(left=0.1, right=0.97, top=0.83, bottom=0.14)
fig.savefig(figures / "threshold_curve.png", dpi=200)
print(f"wrote {figures / 'ndcg_ladder.png'} and {figures / 'threshold_curve.png'}")
