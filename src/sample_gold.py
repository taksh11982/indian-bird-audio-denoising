"""P3 gold sampler: stratified 150-chunk lock across the 12 focus species.

Focus list = top chunk counts, genera-distinct, label-usable (excludes
'Mystery mystery' (no label), 'Chloropsis' (genus-level folder), and
'Cyornis poliogenys' (same genus as C. unicolor)).
Design: 150 / 12 = 12.5 -> the 6 largest pools contribute 13 chunks,
the other 6 contribute 12. Plain uniform sampling per species (spec-literal),
seed 42. Output data/gold_test_150.csv (chunk_id,species,grade,notes) with
grade/notes left blank for human Audacity grading.
LOCKED AFTER COMMIT: never resampled, never edited (train scripts assert
zero gold chunk_id overlap).
"""
import csv
import pathlib
import random

REPO = pathlib.Path(__file__).resolve().parents[1]
INDEX = REPO / "data" / "chunk_index.csv"
GOLD = REPO / "data" / "gold_test_150.csv"

# Frozen 2026-10-02: top chunk counts x genera-distinct x label-usable.
FOCUS_12 = [
    "Pellorneum ruficeps",
    "Cuculus micropterus",
    "Arborophila torqueola",
    "Pomatorhinus ruficollis",
    "Glaucidium cuculoides",
    "Cyornis unicolor",
    "Sphenocichla humei",
    "Pnoepyga pusilla",
    "Liocichla phoenicea",
    "Rimator malacoptilus",
    "Psilopogon lineatus",
    "Psittacula eupatria",
]
N_GOLD = 150


def main() -> int:
    rng = random.Random(42)
    with open(INDEX, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    pools = {}
    for s in FOCUS_12:
        pool = sorted(r["chunk_id"] for r in rows if r["species_name"] == s)
        assert pool, f"empty pool: {s}"
        pools[s] = pool
    sizes = {s: len(p) for s, p in pools.items()}
    ranked = sorted(FOCUS_12, key=lambda s: (-sizes[s], s))
    quota = {s: (13 if i < 6 else 12) for i, s in enumerate(ranked)}
    assert sum(quota.values()) == N_GOLD
    gold = []
    for s in FOCUS_12:
        for cid in rng.sample(pools[s], quota[s]):
            gold.append({"chunk_id": cid, "species": s, "grade": "",
                         "notes": ""})
    gold.sort(key=lambda r: (r["species"], r["chunk_id"]))
    with open(GOLD, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["chunk_id", "species", "grade",
                                           "notes"])
        w.writeheader()
        w.writerows(gold)
    print(f"gold={len(gold)} " +
          " ".join(f"{s}={quota[s]}" for s in FOCUS_12))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
