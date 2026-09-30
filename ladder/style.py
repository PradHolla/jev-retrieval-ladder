"""Chart palette and matplotlib defaults shared by the figure scripts.

Reference palette from the dataviz skill: the first three categorical slots in fixed order,
validated all-pairs in light mode. Aqua is under 3:1 on the surface, so its marks are always
direct-labeled.
"""

import matplotlib.pyplot as plt

FIRST_STAGE, CROSS_ENCODER, JEV = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK_SECONDARY, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, EMPTY = "#e1e0d9", "#c3c2b7", "#e1e0d9"


def apply():
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_SECONDARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
    })


def title(fig, text, subtitle, size=13, top=0.97, gap=0.055):
    fig.text(0.02, top, text, fontsize=size, fontweight="semibold", color=INK, va="top")
    fig.text(0.02, top - gap, subtitle, fontsize=size * 0.73, color=INK_SECONDARY, va="top")
