"""Shareable images for the post, 4:5 portrait (1200x1500), drawn only from results/results.json."""

import json

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from ladder import ROOT, style
from ladder.style import (BASELINE, CROSS_ENCODER, EMPTY, FIRST_STAGE, INK, INK_MUTED, INK_SECONDARY,
                          JEV, SURFACE, title)

style.apply()
plt.rcParams["font.size"] = 11
with open(ROOT / "results" / "results.json") as f:
    results = json.load(f)
systems = results["systems"]
kept_one = results["thresholds"]["matched_kept_per_query"]["1"]
jev_vs_ce = results["deltas"]["jev - ce"]["ndcg@10"]
out = ROOT / "figures" / "post"
out.mkdir(parents=True, exist_ok=True)

TITLE = "Does Jev beat standard retrieval?"
SUBTITLE = "BEIR SciFact, 300 test queries, human relevance labels"
FOOTNOTE = (f"Jev vs cross-encoder: +{jev_vs_ce['mean']:.3f} NDCG@10 "
            f"(95% CI +{jev_vs_ce['ci_low']:.3f} to +{jev_vs_ce['ci_high']:.3f}). "
            "Both rerank the same 30 candidates.")


def ndcg(key):
    return systems[key]["ndcg@10"]["mean"]


def figure(height=7.5):
    return plt.figure(figsize=(6, height))


def footnote(fig, text=FOOTNOTE):
    fig.text(0.04, 0.025, text, fontsize=8, color=INK_MUTED, va="bottom", linespacing=1.5)


def dot(ax, x, y, color, hollow=False):
    ax.scatter(x, y, s=90, color=SURFACE if hollow else color, edgecolors=color if hollow else SURFACE,
               linewidths=2, zorder=3)


def draw_ladder(ax):
    steps = [("BM25", "bm25"), ("Dense\nembeddings", "dense"), ("Hybrid\n(BM25+dense)", "hybrid")]
    ys = [ndcg(key) for _, key in steps]
    ax.plot(range(3), ys, color=FIRST_STAGE, lw=2, zorder=2)
    for x, y in enumerate(ys):
        dot(ax, x, y, FIRST_STAGE)
        ax.text(x + 0.08, y - 0.004, f"{y:.3f}", ha="left", va="top", fontsize=10, color=INK)
    for key, color, label in [("ce", CROSS_ENCODER, "Cross-encoder"), ("jev", JEV, "Jev")]:
        ax.plot([2, 3], [ys[-1], ndcg(key)], color=color, lw=2, zorder=2)
        dot(ax, 3, ndcg(key), color)
        ax.text(3.15, ndcg(key), f"{label}  {ndcg(key):.3f}", va="center", fontsize=10.5,
                color=INK, fontweight="semibold" if key == "jev" else "normal")
    dot(ax, 3, ndcg("hybrid+jev"), JEV, hollow=True)
    ax.text(3.15, ndcg("hybrid+jev"), f"Jev fused back\nwith hybrid  {ndcg('hybrid+jev'):.3f}",
            va="center", fontsize=9.5, color=INK_SECONDARY)
    ax.set_xticks(range(4), [label for label, _ in steps] + ["Rerank\nits top 30"], fontsize=9.5)
    ax.tick_params(axis="x", length=0, colors=INK_SECONDARY)
    ax.set_xlim(-0.35, 5.1)
    ax.set_ylim(0.66, 0.81)
    ax.set_ylabel("NDCG@10 (higher is better)")
    ax.grid(axis="x", visible=False)


def draw_waffle(ax, share, color, name):
    relevant = round(share * 100)
    for i in range(100):
        row, col = divmod(i, 10)
        ax.add_patch(Rectangle((col, 9 - row), 0.9, 0.9, color=color if i < relevant else EMPTY, lw=0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0, 11.9, name, fontsize=10.5, color=INK_SECONDARY)
    ax.text(0, 10.4, f"{relevant} of 100 relevant", fontsize=14, fontweight="semibold", color=INK)


def draw_waffles(fig, left_x, right_x, bottom, size):
    height = size * fig.get_figwidth() / fig.get_figheight()
    for x, key, color, name in [(left_x, "jev", JEV, "Jev"), (right_x, "ce", CROSS_ENCODER, "Cross-encoder")]:
        draw_waffle(fig.add_axes([x, bottom, size, height]), kept_one[key]["precision"], color, name)


def draw_table(ax):
    rows = [("BM25", "bm25", FIRST_STAGE), ("Dense embeddings", "dense", FIRST_STAGE),
            ("Hybrid (BM25 + dense)", "hybrid", FIRST_STAGE),
            ("Hybrid, cross-encoder rerank", "ce", CROSS_ENCODER),
            ("Hybrid, Jev rerank", "jev", JEV), ("Jev fused back with hybrid", "hybrid+jev", JEV)]
    columns = [("NDCG@10", "ndcg@10", 0.66), ("MRR@10", "mrr@10", 0.82), ("Recall@10", "recall@10", 0.98)]
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(rows) + 1)
    header = len(rows) + 0.5
    for label, _, x in columns:
        ax.text(x, header, label, ha="right", va="center", fontsize=9.5, color=INK_MUTED)
    ax.plot([0, 1], [len(rows), len(rows)], color=BASELINE, lw=0.8)
    for i, (name, key, color) in enumerate(rows):
        y = len(rows) - 0.5 - i
        if key == "jev":
            ax.add_patch(Rectangle((0, y - 0.5), 1, 1, color=JEV, alpha=0.10, lw=0))
        weight = "semibold" if key == "jev" else "normal"
        ax.scatter(0.025, y, s=45, color=color, edgecolors=SURFACE, linewidths=1.5)
        ax.text(0.06, y, name, va="center", fontsize=10, color=INK, fontweight=weight)
        for _, metric, x in columns:
            ax.text(x, y, f"{systems[key][metric]['mean']:.3f}", ha="right", va="center", fontsize=10.5,
                    color=INK, fontweight=weight)


def draw_tiles(fig, bottom, height):
    jev = results["jev"]
    tiles = [
        (f"{kept_one['jev']['precision']:.0%} vs {kept_one['ce']['precision']:.0%}",
         "Relevant among kept passages\nat ~1 per query,\nJev vs cross-encoder"),
        (f"${jev['cost_per_query_usd']:.5f}", "Jev cost per query\n(30 passages scored)"),
        (f"{jev['latency_ms_per_call']['p50']:.0f} ms", "Jev median latency\nper call (calls run\nin parallel)"),
    ]
    width, gap = 0.29, 0.025
    for i, (value, label) in enumerate(tiles):
        x = 0.04 + i * (width + gap)
        fig.patches.append(FancyBboxPatch((x, bottom), width, height, boxstyle="round,pad=0,rounding_size=0.012",
                                          transform=fig.transFigure, facecolor="#f3f2ee", edgecolor="none"))
        fig.text(x + 0.02, bottom + height - 0.025, value, fontsize=15, fontweight="semibold", color=INK, va="top")
        fig.text(x + 0.02, bottom + height - 0.07, label, fontsize=8, color=INK_SECONDARY, va="top",
                 linespacing=1.4)


def save(fig, name):
    fig.savefig(out / name, dpi=200)
    plt.close(fig)


fig = figure()
title(fig, TITLE, SUBTITLE, size=17, top=0.965, gap=0.05)
draw_ladder(fig.add_axes([0.13, 0.19, 0.84, 0.64]))
footnote(fig)
save(fig, "ladder.png")

fig = figure()
title(fig, TITLE, SUBTITLE, size=17, top=0.965, gap=0.05)
draw_table(fig.add_axes([0.04, 0.37, 0.92, 0.47]))
draw_tiles(fig, bottom=0.15, height=0.16)
footnote(fig)
save(fig, "scorecard.png")

fig = figure(height=6)
title(fig, "Keep one passage per query.\nHow many are relevant?",
      "Score cutoffs set so each reranker keeps about 1 passage per query", size=17, top=0.96, gap=0.125)
draw_waffles(fig, left_x=0.05, right_x=0.53, bottom=0.15, size=0.42)
footnote(fig, "SciFact test set, 300 queries; relevance judged by humans; colored = relevant.")
save(fig, "waffle.png")

fig = figure()
title(fig, TITLE, SUBTITLE, size=17, top=0.965, gap=0.05)
draw_ladder(fig.add_axes([0.13, 0.52, 0.84, 0.34]))
fig.text(0.04, 0.415, "Keep ~1 passage per query: how many are relevant?", fontsize=11.5,
         fontweight="semibold", color=INK)
draw_waffles(fig, left_x=0.08, right_x=0.55, bottom=0.09, size=0.28)
footnote(fig)
save(fig, "summary.png")
print(f"wrote {sorted(p.name for p in out.glob('*.png'))} to {out}")
