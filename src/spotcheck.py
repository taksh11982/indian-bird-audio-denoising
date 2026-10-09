"""30-clip human spot-check sampler (replaces 150-gold grading).

Randomly samples 30 cleaned focus-11 chunks (seed 42, gold excluded),
writes results/spotcheck_30.csv with keep/notes left blank for Audacity
listening vs data/xc_refs/. One evening of work; lets the paper state a
checked precision instead of an ungraded-gold bluff.
Usage: python src/spotcheck.py
"""
import csv
import pathlib
import random

REPO = pathlib.Path(__file__).resolve().parents[1]
N = 30


def main() -> int:
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from gavali_features import FOCUS_11
    rng = random.Random(42)
    tri = {r["chunk_id"]: r for r in
           csv.DictReader(open(REPO / "data" / "audit_triage.csv",
                                encoding="utf-8"))}
    idx = {r["chunk_id"]: r for r in
           csv.DictReader(open(REPO / "data" / "chunk_index.csv",
                                encoding="utf-8"))}
    gold = {r["chunk_id"] for r in
            csv.DictReader(open(REPO / "data" / "gold_test_150.csv",
                                 encoding="utf-8"))}
    pool = sorted(c for c, t in tri.items()
                  if t["given_label"] in set(FOCUS_11)
                  and t["triage"] == "concordant" and c not in gold
                  and c in idx)
    assert len(pool) >= N, len(pool)
    pool_by_sp = {}
    for s in FOCUS_11:
        pool_by_sp[s] = sorted(c for c in pool if tri[c]["given_label"] == s)
        assert len(pool_by_sp[s]) >= 2, (s, len(pool_by_sp[s]))
    picks = []
    for s in FOCUS_11:
        picks += rng.sample(pool_by_sp[s], 2)  # 22 guaranteed coverage
    rest = sorted(set(pool) - set(picks))
    picks += rng.sample(rest, N - len(picks))  # 8 extra, size-weighted random
    picks = sorted(picks)
    outdir = REPO / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "spotcheck_30.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["chunk_id", "species", "path_16k",
                                           "keep", "notes"])
        w.writeheader()
        for c in picks:
            w.writerow({"chunk_id": c, "species": tri[c]["given_label"],
                        "path_16k": idx[c]["path_16k"], "keep": "",
                        "notes": ""})
    from collections import Counter
    print(f"spotcheck={len(picks)} " +
          str(dict(Counter(tri[c]['given_label'] for c in picks))))
    print("grade keep/notes in Audacity vs data/xc_refs/, do NOT use for "
          "training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
