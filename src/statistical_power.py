"""Statistical resolution of the current evaluation.

The report compares Model A (raw labels) with Model C (cleaned labels) on the
same 234 test chunks.  This script quantifies what that test set can and
cannot resolve, using an exact McNemar analysis of the paired outcomes:

  n01 = clips Model A got wrong and Model C got right
  n10 = clips Model A got right and Model C got wrong
  chi2 = (|n01 - n10| - 1)^2 / (n01 + n10)

Only the difference d = n01 - n10 is observable from the confusion matrices
(d = correct_C - correct_A = 3 clips).  The discordant count m = n01 + n10 is
not, so the script brackets the worst case (m = d, i.e. every discordant clip
favours C) and then reports how large a test set would be needed to detect
the same effect.

Outputs:
  results/statistical_power.csv
  figures/fig11_power.png (+ .pdf)
Usage: python src/statistical_power.py
"""
import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
OUTDIR = REPO / "results"
FIG = REPO / "figures"

N_TEST = 234
CORRECT_A = 103
CORRECT_C = 106
CHI2_CRIT = 3.841


def chi2_from(d, m):
    if m <= 0:
        return 0.0
    return (abs(d) - 1.0) ** 2 / m


def required_n(effect_per_clip, q):
    """Smallest n so that McNemar chi2 >= 3.841, given the observed effect
    rate (d/n) and a discordant fraction q = m/n."""
    lo, hi = 2, 10_000_000
    while lo < hi:
        mid = (lo + hi) // 2
        d = effect_per_clip * mid
        m = q * mid
        if chi2_from(d, m) >= CHI2_CRIT:
            hi = mid
        else:
            lo = mid + 1
    return lo


def main() -> int:
    d = CORRECT_C - CORRECT_A
    rows = []
    best = chi2_from(d, d)
    rows.append({
        "quantity": "observed net gain (clips)", "value": d,
        "note": "Model C correct minus Model A correct on 234 shared clips"})
    rows.append({
        "quantity": "best case McNemar chi2 on this test set", "value":
        round(best, 4),
        "note": "assumes only 3 clips are discordant and all favour C"})
    rows.append({
        "quantity": "chi2 needed for p<0.05", "value": CHI2_CRIT,
        "note": "1 degree of freedom"})
    rows.append({
        "quantity": "smallest p achievable on 234 clips", "value": 0.2482,
        "note": "significance is unreachable however the pairs fall"})
    effect = d / N_TEST
    for q in (0.05, 0.10, 0.20):
        rows.append({
            "quantity": f"test clips needed (discordant fraction {q:.0%})",
            "value": required_n(effect, q),
            "note": f"same effect size {effect:.4%}/clip, McNemar p<0.05"})

    with open(OUTDIR / "statistical_power.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["quantity", "value", "note"])
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['quantity']:52s} {str(r['value']):>10s}  {r['note']}")

    ns = np.arange(50, 6001, 5)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for q, col in ((0.05, "#c44e52"), (0.10, "#4c72b0"), (0.20, "#55a868")):
        d_n = effect * ns
        chi = (np.abs(d_n) - 1.0) ** 2 / (q * ns)
        ax.plot(ns, chi, color=col, lw=1.8,
                label=f"discordant fraction {q:.0%}")
        n_need = required_n(effect, q)
        ax.plot([n_need], [CHI2_CRIT], "o", color=col, ms=6)
        ax.annotate(f"n={n_need}", (n_need, CHI2_CRIT), fontsize=7.2,
                    textcoords="offset points", xytext=(4, 7), color=col)
    ax.axhline(CHI2_CRIT, color="#333333", ls="--", lw=1.1,
               label="significance threshold (chi2 = 3.84)")
    ax.axvline(N_TEST, color="#777777", ls=":", lw=1.2)
    ax.annotate("current test set\n234 clips", (N_TEST, ax.get_ylim()[1]),
                fontsize=7.5, ha="center", va="top", color="#555555")
    ax.set_xlim(0, 6000)
    ax.set_yscale("log")
    ax.set_xlabel("number of test clips")
    ax.set_ylabel("McNemar chi2 (log scale)")
    ax.set_title("The current 234-clip test set cannot resolve a +1 point "
                 "effect\n(how large would the test set have to be?)")
    ax.legend(fontsize=7.6, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig11_power.png", dpi=300)
    fig.savefig(FIG / "fig11_power.pdf")
    plt.close(fig)
    print("wrote figures/fig11_power.png and results/statistical_power.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
