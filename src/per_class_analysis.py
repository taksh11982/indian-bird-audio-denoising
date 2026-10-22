"""Per-class analysis: what did the cleaning actually fix?

Reads the two confusion matrices (Model A raw labels, Model C cleaned labels)
and the audit triage of the test chunks and answers, per species:
  - how many of its test chunks have a trustworthy label
  - how its recall / precision changed after cleaning
  - the hard upper and lower bound of its accuracy on the trusted subset
    (the diagonal hits can only be distributed over trusted / untrusted test
    chunks in two extreme ways, so the true value is bracketed without
    needing per-chunk predictions)

Outputs:
  results/per_class_analysis.csv
  figures/fig9_per_class_recall.png (+ .pdf)
Usage: python src/per_class_analysis.py
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
TRUSTED = "concordant"


def read(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def confusion(path):
    rows = list(csv.reader(open(path, encoding="utf-8")))
    classes = rows[0][1:]
    cm = np.array([[int(v) for v in r[1:]] for r in rows[1:]], dtype=float)
    return classes, cm


def main() -> int:
    classes, cmA = confusion(OUTDIR / "confusion_A.csv")
    _, cmC = confusion(OUTDIR / "confusion_C.csv")
    tri = {r["chunk_id"]: r for r in
           csv.DictReader(open(REPO / "data" / "audit_triage.csv",
                               encoding="utf-8"))}
    splits = read(OUTDIR / "splits_by_chunk.csv")

    n_test = {c: 0 for c in classes}
    n_trust = {c: 0 for c in classes}
    for r in splits:
        if r["split"] == "test" and r["in_raw"] == "1":
            s = r["species"]
            n_test[s] = n_test.get(s, 0) + 1
            if tri[r["chunk_id"]]["triage"] == TRUSTED:
                n_trust[s] = n_trust.get(s, 0) + 1

    rows = []
    for i, c in enumerate(classes):
        n = n_test.get(c, 0)
        u = n - n_trust.get(c, 0)
        ra = cmA[i, i] / n if n else 0.0
        rc = cmC[i, i] / n if n else 0.0
        colA = cmA[:, i].sum()
        colC = cmC[:, i].sum()
        pa = cmA[i, i] / colA if colA else 0.0
        pc = cmC[i, i] / colC if colC else 0.0
        lo = max(0.0, (cmA[i, i] - u)) / max(1, n - u)
        hi = cmA[i, i] / max(1, n - u)
        lo_c = max(0.0, (cmC[i, i] - u)) / max(1, n - u)
        hi_c = cmC[i, i] / max(1, n - u)
        rows.append({
            "species": c, "n_test": n, "n_trusted": n_trust.get(c, 0),
            "untrusted_pct": round(100.0 * u / n, 1) if n else 0.0,
            "recall_A": round(ra, 4), "recall_C": round(rc, 4),
            "delta_recall": round(rc - ra, 4),
            "precision_A": round(pa, 4), "precision_C": round(pc, 4),
            "delta_precision": round(pc - pa, 4),
            "acc_trusted_A_lo": round(lo, 4), "acc_trusted_A_hi": round(hi, 4),
            "acc_trusted_C_lo": round(lo_c, 4), "acc_trusted_C_hi": round(hi_c, 4),
        })
    rows.sort(key=lambda r: -r["delta_recall"])

    with open(OUTDIR / "per_class_analysis.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    order = [r["species"] for r in rows]
    y = np.arange(len(order))
    ra = [r["recall_A"] for r in rows]
    rc = [r["recall_C"] for r in rows]
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    h = 0.38
    b1 = ax.barh(y + h / 2, ra, h, color="#c44e52", label="Model A (raw labels)")
    b2 = ax.barh(y - h / 2, rc, h, color="#4c72b0",
                 label="Model C (cleaned labels)")
    for i, r in enumerate(rows):
        ax.text(max(ra[i], rc[i]) + 0.02, y[i],
                f"n={r['n_test']} (trusted {r['n_trusted']})",
                va="center", fontsize=6.8, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=7.6)
    ax.set_xlim(0, 1.35)
    ax.set_xlabel("recall on the shared 234-chunk raw test set")
    ax.set_title("Per-class recall after cleaning: gains concentrate in the "
                 "species whose test labels were broken")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig9_per_class_recall.png", dpi=300)
    fig.savefig(FIG / "fig9_per_class_recall.pdf")
    plt.close(fig)

    print(f"{'species':34s}{'n':>4}{'trust':>7}{'recA':>7}{'recC':>7}"
          f"{'d_rec':>8}{'d_prec':>8}")
    for r in rows:
        print(f"{r['species']:34s}{r['n_test']:>4}{r['n_trusted']:>7}"
              f"{r['recall_A']:>7.3f}{r['recall_C']:>7.3f}"
              f"{r['delta_recall']:>+8.3f}{r['delta_precision']:>+8.3f}")
    tot_a = sum(r["recall_A"] for r in rows) / len(rows)
    tot_c = sum(r["recall_C"] for r in rows) / len(rows)
    print(f"\nmacro recall A={tot_a:.4f} C={tot_c:.4f} "
          f"delta={tot_c - tot_a:+.4f}")
    print("wrote", (OUTDIR / "per_class_analysis.csv").relative_to(REPO),
          "and figures/fig9_per_class_recall.png")
    fig_mechanism(rows)
    return 0


def fig_mechanism(rows):
    """Species-level link between how dirty the training data was and how
    much cleaning moved recall (the mechanism check)."""
    x = np.array([1.0 - r["untrusted_pct"] / 100.0 for r in rows])
    y = np.array([r["delta_recall"] for r in rows])
    n = np.array([r["n_test"] for r in rows])
    if len(x) > 2:
        r_p = float(np.corrcoef(x, y)[0, 1])
        order = np.argsort(x)
        rank_x = np.empty(len(x), float)
        rank_x[order] = np.arange(len(x))
        order2 = np.argsort(y)
        rank_y = np.empty(len(y), float)
        rank_y[order2] = np.arange(len(y))
        r_s = float(np.corrcoef(rank_x, rank_y)[0, 1])
    else:
        r_p = r_s = float("nan")
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    sizes = 40 + 260 * (n / max(1, n.max()))
    ax.scatter(x * 100, y * 100, s=sizes, c=np.where(y >= 0, "#4c72b0",
                                                       "#c44e52"),
               alpha=0.85, edgecolor="white", linewidth=0.8, zorder=3)
    for r, xi, yi in zip(rows, x, y):
        ax.annotate(r["species"].split()[0], (xi * 100, yi * 100),
                    textcoords="offset points", xytext=(7, 4), fontsize=6.6,
                    color="#333333")
    z = np.polyfit(x * 100, y * 100, 1)
    xs = np.linspace(30, 100, 20)
    ax.plot(xs, np.polyval(z, xs), color="#555555", lw=1.2, ls="--",
            label=f"linear fit (Pearson r = {r_p:+.2f}, "
                  f"Spearman = {r_s:+.2f})")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xlabel("training-label quality of that species (%% clean chunks)")
    ax.set_ylabel("change in recall after cleaning (percentage points)")
    ax.set_title("Mechanism check: does cleaning help most where the labels "
                 "are worst?\n(bubble size = number of test clips)")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig10_mechanism.png", dpi=300)
    fig.savefig(FIG / "fig10_mechanism.pdf")
    plt.close(fig)
    with open(OUTDIR / "mechanism_check.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["species", "clean_label_pct", "delta_recall_pp",
                    "n_test", "pearson_r", "spearman_rho"])
        for r, xi, yi in zip(rows, x, y):
            w.writerow([r["species"], round(xi * 100, 1), round(yi * 100, 1),
                        r["n_test"], round(r_p, 4), round(r_s, 4)])
    print(f"mechanism check: Pearson r={r_p:+.3f} Spearman={r_s:+.3f} "
          "(n=11 species; indicative only)")
    print("wrote figures/fig10_mechanism.png and results/mechanism_check.csv")


if __name__ == "__main__":
    raise SystemExit(main())
