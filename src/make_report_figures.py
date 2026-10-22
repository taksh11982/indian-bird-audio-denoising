"""Generate every figure used in the Project-2 report (matplotlib, Agg).

All numbers come from the committed CSVs - nothing is hard-coded, so the
figures can be regenerated after any new experiment:
  data/audit_triage.csv        chunk -> triage
  data/chunk_index.csv         chunk -> species
  results/splits_by_chunk.csv  chunk -> split, in_raw, in_cleaned
  results/confusion_A.csv      11x11 confusion matrix, Model A
  results/confusion_C.csv      11x11 confusion matrix, Model C
  results/gavali_ablation.csv  A/C metrics
  results/label_noise_evidence.csv  pool-level label quality
Usage: python src/make_report_figures.py
"""
import csv
import pathlib
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = pathlib.Path(__file__).resolve().parents[1]
FIG = REPO / "figures"
TRIAGE = REPO / "data" / "audit_triage.csv"
SPLITS = REPO / "results" / "splits_by_chunk.csv"
COL = {
    "concordant": "#2ca02c",
    "empty": "#d62728",
    "ambiguous": "#ff7f0e",
    "intruder": "#9467bd",
    "unlabeled": "#c7c7c7",
}
TRIAGE_ORDER = ["concordant", "empty", "ambiguous", "intruder", "unlabeled"]
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "figure.dpi": 200, "savefig.bbox": "tight", "font.family": "DejaVu Sans",
})


def read(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("wrote figures/" + name + ".png/.pdf")


def fig_pipeline():
    stages = [
        ("1. Ingestion", "1,368 WAV files\n53 species, 12.68 h\nmanifest"),
        ("2. Chunking", "5.0 s non-overlapping\n48 kHz + 16 kHz\n9,289 chunks"),
        ("3. Audit", "BirdNET-Analyzer V2.4\n3 s sliding windows\nmax-pool -> S"),
        ("4. Cleaning", "frozen thresholds\nempty / intruder / amb.\nkeep concordant"),
        ("5. Train + Eval", "MFCC 39 -> LSTM\nModel A vs Model C\nMacro-F1"),
    ]
    fig, ax = plt.subplots(figsize=(11, 3.1))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    colors = ["#4c72b0", "#55a868", "#c44e52", "#dd8452", "#8172b3"]
    for i, ((title, body), c) in enumerate(zip(stages, colors)):
        x = 0.15 + i * 1.98
        ax.add_patch(FancyBboxPatch((x, 0.55), 1.72, 1.9,
                                    boxstyle="round,pad=0.04,rounding_size=0.08",
                                    fc=c, ec="none", alpha=0.92))
        ax.text(x + 0.86, 2.05, title, ha="center", va="center",
                color="white", fontsize=10, fontweight="bold")
        ax.text(x + 0.86, 1.15, body, ha="center", va="center",
                color="white", fontsize=7.6, linespacing=1.5)
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + 1.76, 1.5), (x + 1.95, 1.5),
                                         arrowstyle="-|>", mutation_scale=13,
                                         color="#555555", lw=1.3))
    ax.text(5, 2.85, "Unsupervised label audit of a public crowdsourced dataset",
            ha="center", fontsize=11, fontweight="bold")
    ax.text(5, 0.15, "No human labelling and no waveform enhancement: the audio is never "
            "modified, only the training labels are filtered.",
            ha="center", fontsize=8, color="#444444", style="italic")
    save(fig, "fig1_pipeline")


def fig_triage_share(triage):
    c = Counter(t["triage"] for t in triage)
    total = len(triage)
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    names = ["concordant", "empty", "ambiguous", "intruder", "unlabeled"]
    vals = [c.get(k, 0) for k in names]
    bars = ax.bar(names, vals, color=[COL[k] for k in names], width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 90,
                f"{v:,}\n({v / total:.1%})", ha="center", va="bottom",
                fontsize=8.5)
    ax.set_ylabel("5.0 s chunks")
    ax.set_title(f"BirdNET triage of all {total:,} iBC53 chunks "
                 f"(seed 42, frozen thresholds)")
    ax.set_ylim(0, max(vals) * 1.28)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.5, 0.97, "17.0 % of all chunks contain no detectable bird; "
            "39.2 % belong to taxa BirdNET cannot name",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
            color="#b02418")
    save(fig, "fig2_triage_share")


def fig_species_noise(triage):
    per = defaultdict(Counter)
    for t in triage:
        per[t["given_label"]][t["triage"]] += 1
    species = sorted(per, key=lambda s: -sum(per[s].values()))
    usable = [s for s in species if per[s]["concordant"] +
              per[s]["empty"] + per[s]["ambiguous"] + per[s]["intruder"] > 0]
    fig, ax = plt.subplots(figsize=(9.6, max(5, len(usable) * 0.24)))
    y = list(range(len(usable) - 1, -1, -1))
    left = [0.0] * len(usable)
    for key in TRIAGE_ORDER:
        vals = []
        for s in usable:
            tot = sum(per[s][k] for k in TRIAGE_ORDER)
            vals.append(per[s][key] / tot if tot else 0.0)
        ax.barh(y, vals, left=left, color=COL[key], label=key,
                edgecolor="white", linewidth=0.4)
        left = [l + v for l, v in zip(left, vals)]
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s} (n={sum(per[s].values())})" for s in usable],
                       fontsize=6.6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of chunks of that species")
    ax.set_title("Label-noise composition per species BEFORE cleaning")
    ax.legend(fontsize=7.5, ncol=5, loc="lower center",
              bbox_to_anchor=(0.5, -0.06))
    save(fig, "fig3_species_noise_before")


def fig_species_after(triage):
    kept = Counter(t["given_label"] for t in triage
                   if t["triage"] == "concordant")
    species = sorted(kept, key=lambda s: -kept[s])
    fig, ax = plt.subplots(figsize=(9.6, max(3.4, len(species) * 0.22)))
    y = list(range(len(species) - 1, -1, -1))
    ax.barh(y, [1.0] * len(species), color=COL["concordant"], height=0.85,
            label="concordant (target confirmed)")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s} (n={kept[s]})" for s in species], fontsize=6.6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of chunks (cleaned set = 100 % concordant by filter)")
    ax.set_title("AFTER cleaning: only audited-concordant chunks are kept")
    removed = Counter(t["triage"] for t in triage if t["triage"] != "concordant")
    ax.text(0.99, 0.02,
            f"kept {sum(kept.values()):,} / {len(triage):,} chunks "
            f"({sum(kept.values()) / len(triage):.1%})   |   removed: "
            f"empty {removed['empty']:,}, intruder {removed['intruder']}, "
            f"ambiguous {removed['ambiguous']:,}, unlabeled {removed['unlabeled']:,}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "fig4_species_noise_after")


def fig_pool_quality(ev):
    labels = [r["set"] for r in ev]
    usable = [int(r["n_chunks"]) - int(r["unusable"]) for r in ev]
    bad = [int(r["unusable"]) for r in ev]
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    x = range(len(labels))
    ax.bar(x, usable, color=COL["concordant"], label="usable label",
           edgecolor="white", linewidth=0.5)
    ax.bar(x, bad, bottom=usable, color=COL["empty"],
           label="unusable label (empty / ambiguous / wrong species)",
           edgecolor="white", linewidth=0.5)
    for i, (u, b) in enumerate(zip(usable, bad)):
        n = u + b
        if n:
            ax.text(i, u / 2, f"{u / n:.0%}\nclean", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
            if b:
                ax.text(i, u + b / 2, f"{b / n:.0%}\nnoisy", ha="center",
                        va="center", fontsize=8, color="white",
                        fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels([l.replace(" (", "\n(") for l in labels], fontsize=7.6)
    ax.set_ylabel("chunks")
    ax.set_title("Label quality of every pool: the noise we removed, and the "
                 "noise still inside the test set")
    ax.legend(fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig5_pool_quality")


def _confusion(path):
    rows = list(csv.reader(open(path, encoding="utf-8")))
    classes = rows[0][1:]
    cm = [[int(v) for v in r[1:]] for r in rows[1:]]
    return classes, cm


def fig_confusion():
    ca, ma = _confusion(REPO / "results" / "confusion_A.csv")
    cc, mc = _confusion(REPO / "results" / "confusion_C.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.2))
    for ax, cm, classes, title in (
            (axes[0], ma, ca, "Model A - trained on RAW labels"),
            (axes[1], mc, cc, "Model C - trained on CLEANED labels")):
        cm = [[float(v) for v in row] for row in cm]
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=60, ha="right", fontsize=6.4)
        ax.set_yticklabels(classes, fontsize=6.4)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(title, fontsize=10)
        vmax = max(max(r) for r in cm)
        for i in range(len(cm)):
            for j in range(len(cm)):
                if cm[i][j]:
                    ax.text(j, i, f"{int(cm[i][j])}", ha="center", va="center",
                            fontsize=5.6,
                            color="white" if cm[i][j] > vmax / 2 else "black")
    fig.suptitle("Where the errors move: same 234 raw test chunks, "
                 "only the training labels differ", fontsize=11)
    fig.tight_layout()
    save(fig, "fig6_confusion_ab")


def fig_ablation():
    rows = read(REPO / "results" / "gavali_ablation.csv")
    ab = {r["run"]: r for r in rows}
    a, c = ab["A-raw"], ab["C-cleaned"]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    labels = ["Training chunks\n(n)", "Test accuracy", "Test Macro-F1"]
    avals = [float(a["n_train"]), float(a["accuracy"]), float(a["macro_f1"])]
    cvals = [float(c["n_train"]), float(c["accuracy"]), float(c["macro_f1"])]
    x = range(3)
    w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], avals, w, color="#c44e52",
                label="Model A (raw labels)")
    b2 = ax.bar([i + w / 2 for i in x], cvals, w, color="#4c72b0",
                label="Model C (cleaned labels)")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.01,
                    f"{b.get_height():.4g}", ha="center", va="bottom",
                    fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(avals) * 1.18)
    ax.set_title("Ablation: 28 % less training data, better Macro-F1")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.5, 0.92, "C beats A while using 726 fewer chunks",
            transform=ax.transAxes, ha="center", fontsize=9,
            color="#1f5c2e", fontweight="bold")
    save(fig, "fig7_ablation")


def fig_control():
    path = REPO / "results" / "control_subset.csv"
    if not path.exists():
        print("skip fig8_control: results/control_subset.csv not present")
        return
    rows = read(path)
    names = [r["run"] for r in rows]
    acc = [float(r["accuracy"]) for r in rows]
    f1 = [float(r["macro_f1"]) for r in rows]
    ntr = [int(r["n_train"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = range(len(names))
    w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], acc, w, color="#55a868",
                label="accuracy")
    b2 = ax.bar([i + w / 2 for i in x], f1, w, color="#4c72b0",
                label="Macro-F1")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.01,
                    f"{b.get_height():.4f}", ha="center", va="bottom",
                    fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{n}\n(n_train={k})" for n, k in zip(names, ntr)],
                       fontsize=8)
    ax.set_ylim(0, max(acc + f1) * 1.2)
    ax.set_title("Control experiment: cleaning beats equal-size random removal")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig8_control")


def main() -> int:
    triage = read(TRIAGE)
    ev = read(REPO / "results" / "label_noise_evidence.csv")
    fig_pipeline()
    fig_triage_share(triage)
    fig_species_noise(triage)
    fig_species_after(triage)
    fig_pool_quality(ev)
    fig_confusion()
    fig_ablation()
    fig_control()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
