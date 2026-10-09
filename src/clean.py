"""Build cleaned set: concordant-only subset -> all-green figure (P2b).

Honest rule: the raw audit figure (figures/noise_summary.*) is a MEASUREMENT
of raw-data noise and must never be edited to look green. Green comes from
FILTERING: keep only triage == 'concordant' (argmax == label, max >= 0.50),
drop empty / intruder / ambiguous / unlabeled. The AFTER figure is then 100%
green by construction -- that is the cleaning demonstration.

Inputs: data/audit_triage.csv + data/chunk_index.csv
Outputs: data/iBC53-Cleaned-Metadata.csv
         figures/noise_summary_cleaned.png + .pdf (per-species, all green)
No stochastic ops (seed 42 kept per golden rule 1).
"""
import csv
import pathlib
import random
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[1]
TRIAGE = REPO / "data" / "audit_triage.csv"
INDEX = REPO / "data" / "chunk_index.csv"
OUT_CSV = REPO / "data" / "iBC53-Cleaned-Metadata.csv"
FIGDIR = REPO / "figures"


def main() -> int:
    random.seed(42)
    triage = {r["chunk_id"]: r for r in
              csv.DictReader(open(TRIAGE, encoding="utf-8"))}
    index = {r["chunk_id"]: r for r in
             csv.DictReader(open(INDEX, encoding="utf-8"))}
    kept = [t for t in triage.values() if t["triage"] == "concordant"]
    removed = Counter(t["triage"] for t in triage.values()
                      if t["triage"] != "concordant")
    # Join with chunk paths for a usable training manifest.
    rows = []
    for t in kept:
        ix = index.get(t["chunk_id"], {})
        rows.append({
            "chunk_id": t["chunk_id"],
            "species": t["given_label"],
            "pred_label": t["pred_label"],
            "max_conf": t["max_conf"],
            "path_48k": ix.get("path_48k", ""),
            "path_16k": ix.get("path_16k", ""),
        })
    rows.sort(key=lambda r: (r["species"], r["chunk_id"]))
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "chunk_id", "species", "pred_label", "max_conf",
            "path_48k", "path_16k"])
        w.writeheader()
        w.writerows(rows)

    # All-green per-species figure (mirrors noise_summary layout).
    per = defaultdict(int)
    for r in rows:
        per[r["species"]] += 1
    species = sorted(per, key=lambda s: (-per[s], s))
    n = len(species)
    fig, ax = plt.subplots(figsize=(12, max(6, n * 0.28)))
    y = list(range(n - 1, -1, -1))
    ax.barh(y, [1.0] * n, color="#2ca02c", edgecolor="white", linewidth=0.4,
            label="concordant (target present)")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s} (n={per[s]})" for s in species], fontsize=8)
    ax.set_xlabel("share of 5.0 s chunks (cleaned set = 100% concordant)")
    ax.set_title("iBC53 cleaned: concordant-only subset (AFTER filtering)",
                 fontsize=12)
    ax.set_xlim(0, 1)
    N_raw = len(triage)
    N_kept = len(rows)
    ax.text(0.99, 0.01,
            f"kept {N_kept}/{N_raw} ({N_kept/N_raw:.1%}) | "
            f"removed empty {removed['empty']} + intruder {removed['intruder']} "
            f"+ ambiguous {removed['ambiguous']} + unlabeled {removed['unlabeled']}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "noise_summary_cleaned.png", dpi=300)
    fig.savefig(FIGDIR / "noise_summary_cleaned.pdf")
    print(f"kept={N_kept} removed={dict(removed)} "
          f"species_kept={n} -> {OUT_CSV.relative_to(REPO).as_posix()}")
    # ponytail: Sphenocichla humei / Mystery / Chloropsis / Argya /
    # Cyornis magnirostris vanish here (no BirdNET label -> never concordant);
    # upgrade path = manual labels or a second auditor for those taxa.
    missing = sorted({t["given_label"] for t in triage.values()}
                     - set(per))
    print(f"species with 0 kept (no BirdNET label or 0 concordant): {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
