"""Deliverable 1: per-species noise summary figure (P2).

Reads data/audit_triage.csv, plots one horizontal stacked bar per species
(empty / intruder / ambiguous / concordant / unlabeled), sorted by
corrupt share (empty + intruder) descending.
Outputs: figures/noise_summary.png (300 dpi) + figures/noise_summary.pdf.
No stochastic ops (seed 42 N/A here).
"""
import csv
import pathlib
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[1]
TRIAGE = REPO / "data" / "audit_triage.csv"
FIGDIR = REPO / "figures"

ORDER = ["concordant", "ambiguous", "empty", "intruder", "unlabeled"]
COLORS = {"concordant": "#2ca02c", "ambiguous": "#ffbb33", "empty": "#7f7f7f",
          "intruder": "#d62728", "unlabeled": "#c5c5e0"}
LABELS = {"concordant": "concordant (target present)",
          "ambiguous": "ambiguous", "empty": "empty / ambient",
          "intruder": "intruder / co-occurrence", "unlabeled": "unlabeled (no BirdNET label)"}


def main() -> int:
    rows = list(csv.DictReader(open(TRIAGE, encoding="utf-8")))
    per = defaultdict(Counter)
    for r in rows:
        per[r["given_label"]][r["triage"]] += 1
    species = sorted(per, key=lambda s: (
        -(per[s]["empty"] + per[s]["intruder"]) / sum(per[s].values()),
        -sum(per[s].values()), s))
    n = len(species)
    fig, ax = plt.subplots(figsize=(12, max(8, n * 0.24)))
    fig.subplots_adjust(right=0.80)
    y = list(range(n - 1, -1, -1))
    left = [0.0] * n
    totals = [sum(per[s].values()) for s in species]
    for cls in ORDER:
        frac = [per[s][cls] / totals[i] for i, s in enumerate(species)]
        ax.barh(y, frac, left=left, color=COLORS[cls], label=LABELS[cls],
                edgecolor="white", linewidth=0.4)
        left = [a + b for a, b in zip(left, frac)]
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s} (n={totals[i]})" for i, s in enumerate(species)],
                       fontsize=7)
    ax.set_xlabel("share of 5.0 s chunks")
    ax.set_title("iBC53 macro audit: BirdNET V2.4 triage per species", fontsize=12)
    ax.set_xlim(0, 1)
    ax.legend(bbox_to_anchor=(1.02, 0.0), loc="lower left", fontsize=8)
    tot = Counter(r["triage"] for r in rows)
    N = len(rows)
    ax.text(0.99, 0.01,
            f"N={N} chunks | empty {tot['empty']/N:.1%} | intruder {tot['intruder']/N:.1%} | "
            f"ambiguous {tot['ambiguous']/N:.1%} | concordant {tot['concordant']/N:.1%} | "
            f"unlabeled {tot['unlabeled']/N:.1%}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    fig.tight_layout()
    fig.subplots_adjust(right=0.80)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "noise_summary.png", dpi=300)
    fig.savefig(FIGDIR / "noise_summary.pdf")
    print(f"species={n} " + " ".join(f"{k}={tot[k]}" for k in ORDER))
    print("worst-5 corrupt share:",
          [(s, round((per[s]['empty'] + per[s]['intruder']) / sum(per[s].values()), 3))
           for s in species[:5]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
