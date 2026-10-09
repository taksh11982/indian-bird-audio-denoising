"""Recording-level stratified splits shared by Gavali A-vs-C (P4-Gavali).

Pools (gold excluded FIRST, so Gold-set contact is none by construction):
  RAW11     = focus-11 chunks with triage != unlabeled  (noisy labels kept)
  CLEANED11 = focus-11 chunks with triage == concordant (filtered)
Split: per species, shuffle its recording_ids with Random(42), cut 80/10/10
(train/val/test) BY RECORDING. Test is RAW chunks in test recordings for BOTH
runs (honest generalisation; testing C on cleaned-only would be circular).
Outputs: results/splits_by_recording.csv, results/splits_by_chunk.csv
Usage: python src/gavali_splits.py
"""
import csv
import json
import pathlib
import random
from collections import defaultdict

from gavali_features import FOCUS_11

REPO = pathlib.Path(__file__).resolve().parents[1]
TRIAGE = REPO / "data" / "audit_triage.csv"
INDEX = REPO / "data" / "chunk_index.csv"
GOLD = REPO / "data" / "gold_test_150.csv"
OUTDIR = REPO / "results"


def main() -> int:
    rng = random.Random(42)
    tri = {r["chunk_id"]: r for r in
           csv.DictReader(open(TRIAGE, encoding="utf-8"))}
    idx = {r["chunk_id"]: r for r in
           csv.DictReader(open(INDEX, encoding="utf-8"))}
    gold = {r["chunk_id"] for r in csv.DictReader(open(GOLD, encoding="utf-8"))}
    focus = set(FOCUS_11)

    raw_pool = [c for c, t in tri.items()
                if t["given_label"] in focus and t["triage"] != "unlabeled"
                and c not in gold and c in idx]
    cl_pool = [c for c, t in tri.items()
               if t["given_label"] in focus and t["triage"] == "concordant"
               and c not in gold and c in idx]
    cl_set = set(cl_pool)

    # Recordings per species from the RAW pool (superset of cleaned).
    recs = defaultdict(set)
    for c in raw_pool:
        recs[tri[c]["given_label"]].add(idx[c]["recording_id"])
    split_rec = {}  # (species, recording_id) -> split
    for s in sorted(recs):
        rids = sorted(recs[s])
        rng.shuffle(rids)
        n = len(rids)
        n_te = max(1, round(n * 0.10))
        n_va = max(1, round(n * 0.10))
        for i, rid in enumerate(rids):
            if i < n - n_te - n_va:
                sp = "train"
            elif i < n - n_te:
                sp = "val"
            else:
                sp = "test"
            split_rec[(s, rid)] = sp

    def split_of(cid):
        return split_rec[(tri[cid]["given_label"], idx[cid]["recording_id"])]

    OUTDIR.mkdir(parents=True, exist_ok=True)
    with open(OUTDIR / "splits_by_recording.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["species", "recording_id", "split"])
        for (s, rid), sp in sorted(split_rec.items()):
            w.writerow([s, rid, sp])
    with open(OUTDIR / "splits_by_chunk.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["chunk_id", "species", "recording_id", "split",
                    "in_raw", "in_cleaned"])
        for c in sorted(set(raw_pool) | cl_set):
            w.writerow([c, tri[c]["given_label"], idx[c]["recording_id"],
                        split_of(c), int(c in set(raw_pool)),
                        int(c in cl_set)])
    # Counts + guards.
    from collections import Counter
    raw_sp = Counter(split_of(c) for c in raw_pool)
    cl_sp = Counter(split_of(c) for c in cl_pool)
    assert not (set(raw_pool) & gold) and not (set(cl_pool) & gold), \
        "gold leakage"
    assert sum(raw_sp.values()) == len(raw_pool)
    assert sum(cl_sp.values()) == len(cl_pool)
    # ponytail: split is by recording count, so chunk counts per split vary
    # with recording length; upgrade = duration-stratified split.
    summary = {"raw11_nogold": len(raw_pool), "cleaned11_nogold": len(cl_pool),
               "raw_by_split": dict(raw_sp), "cleaned_by_split": dict(cl_sp),
               "species": sorted(focus)}
    with open(OUTDIR / "splits_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print(f"raw11_nogold={len(raw_pool)} {dict(raw_sp)}")
    print(f"cleaned11_nogold={len(cl_pool)} {dict(cl_sp)}")
    print(f"gold excluded: {len(gold)} chunk_ids, overlap in splits: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
