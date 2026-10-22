"""Schematic of the three-arm control experiment (Figure 11 of the report).

Arms R and C have exactly the same size, so a difference between them can
only come from label quality, never from dataset volume.
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = pathlib.Path(__file__).resolve().parents[1]
FIG = REPO / "figures"

BLUE = "#4c72b0"
RED = "#c44e52"
GREEN = "#55a868"
GREY = "#7f7f7f"


def box(ax, x, y, w, h, text, color, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.6, edgecolor=color,
                                facecolor="white", zorder=3))
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center",
            fontsize=10, fontweight="bold", color=color, zorder=4)
    ax.text(x + w / 2, y + h * 0.30, text, ha="center", va="center",
            fontsize=8.2, color="#333333", zorder=4)


def main() -> int:
    fig, ax = plt.subplots(figsize=(9.6, 4.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, 0.02, 0.52, 0.30, 0.40,
        "2,563 chunks\nall raw labels\n28.3% of them wrong",
        RED, "Arm A  (full raw pool)")
    box(ax, 0.02, 0.04, 0.30, 0.40,
        "1,837 chunks\nrandom subset of A\nlabels still 28.3% wrong",
        GREY, "Arm R  (random subset)")
    box(ax, 0.02, -0.44, 0.30, 0.40,
        "1,837 chunks\naudited concordant\n0% wrong labels",
        GREEN, "Arm C  (cleaned pool)")

    box(ax, 0.62, 0.30, 0.34, 0.36,
        "234 held-out raw chunks\n(identical for all three arms)\n"
        "scored with Macro-F1",
        BLUE, "Evaluation")

    for y in (0.72, 0.24, -0.24):
        ax.add_patch(FancyArrowPatch((0.33, y + 0.02), (0.61, 0.48),
                                     arrowstyle="-|>", mutation_scale=13,
                                     linewidth=1.3, color="#666666",
                                     connectionstyle="arc3,rad=0.0",
                                     zorder=2))
    ax.add_patch(FancyArrowPatch((0.47, 0.24), (0.47, 0.48),
                                 arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.3, color="#666666", zorder=2))

    ax.text(0.47, 0.78, "same size,\ndifferent labels", ha="center",
            va="bottom", fontsize=9, color="#333333", style="italic")
    ax.plot([0.35, 0.59], [0.26, 0.26], color="#999999", lw=0.8)
    ax.text(0.47, 0.16, "R vs C isolates label quality\nfrom dataset size",
            ha="center", va="top", fontsize=8.6, color="#333333")

    ax.set_title("The three-arm control: A gives the reference, R controls "
                 "for the smaller\ntraining set, C measures the value of "
                 "correct labels", fontsize=11.5)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig12_control_design.png", dpi=300)
    fig.savefig(FIG / "fig12_control_design.pdf")
    plt.close(fig)
    print("wrote figures/fig12_control_design.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
