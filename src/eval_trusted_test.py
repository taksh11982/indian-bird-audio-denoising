"""Clean-test evaluation: score A and C on the test chunks we can trust.

The headline A-vs-C numbers are computed on the 234 raw test chunks, and
26.5% of those chunks do not actually contain the labelled bird
(results/label_noise_evidence.md). Scoring a model on a chunk whose label is
wrong punishes a correct prediction, so the headline metric is a lower bound
and its resolution is ~2 chunks per 1% of Macro-F1.

This script re-scores the SAME per-chunk predictions on the 172 test chunks
whose label was audited as concordant (target really present) and on the 62
untrusted ones, so the report can separate:
  - "the model is right" (trusted subset)
  - "the label is right" (how good the ground truth is)

Inputs:
  results/preds_lstm_A.csv, results/preds_lstm_C.csv   (from gavali_compare)
  results/noise_scaling_preds.csv                      (optional, per rate)
  data/audit_triage.csv
Outputs:
  results/clean_test_eval.csv
Usage: python src/eval_trusted_test.py
"""
import csv
import pathlib
from collections import defaultdict

REPO = pathlib.Path(__file__).resolve().parents[1]
TRIAGE = REPO / "data" / "audit_triage.csv"
OUTDIR = REPO / "results"
TRUSTED = "concordant"


def macro_f1(pairs):
    """pairs = [(true, pred), ...]; macro over the union of true/pred labels."""
    labels = sorted({t for t, _ in pairs} | {p for _, p in pairs})
    f1s = []
    for c in labels:
        tp = sum(1 for t, p in pairs if t == c and p == c)
        fp = sum(1 for t, p in pairs if t != c and p == c)
        fn = sum(1 for t, p in pairs if t == c and p != c)
        den = 2 * tp + fp + fn
        f1s.append(0.0 if den == 0 else 2.0 * tp / den)
    return (sum(f1s) / len(f1s)) if f1s else 0.0


def read_rows(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def score(pairs):
    if not pairs:
        return 0, 0.0, 0.0
    acc = sum(1 for t, p in pairs if t == p) / len(pairs)
    return len(pairs), round(acc, 4), round(macro_f1(pairs), 4)


def main() -> int:
    tri = {r["chunk_id"]: r for r in
           csv.DictReader(open(TRIAGE, encoding="utf-8"))}
    groups = defaultdict(list)
    for name in ("preds_lstm_A.csv", "preds_lstm_C.csv"):
        path = OUTDIR / name
        if not path.exists():
            print("missing", path.name, "- rerun src/gavali_compare.py "
                  "--backend lstm")
            continue
        for r in read_rows(path):
            groups[name.replace("preds_lstm_", "").replace(".csv", "")].append(r)
    scaling = OUTDIR / "noise_scaling_preds.csv"
    if scaling.exists():
        for r in read_rows(scaling):
            groups[f"scaling_r{r['noise_rate']}"].append(r)

    if not groups:
        print("no prediction files yet - run the Colab job first")
        return 1

    out_rows = []
    for gname in sorted(groups):
        rows = groups[gname]
        buckets = {"all": [], "trusted": [], "untrusted": []}
        for r in rows:
            t, p = r["species"], r["pred"]
            buckets["all"].append((t, p))
            triage = r.get("triage") or tri.get(r["chunk_id"], {}).get(
                "triage", "")
            key = "trusted" if triage == TRUSTED else "untrusted"
            buckets[key].append((t, p))
        for bname in ("all", "trusted", "untrusted"):
            n, acc, f1 = score(buckets[bname])
            out_rows.append({"group": gname, "subset": bname, "n": n,
                             "accuracy": acc, "macro_f1": f1})

    with open(OUTDIR / "clean_test_eval.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print("%-16s %-10s %5s %8s %9s" % ("group", "subset", "n", "acc", "macro-F1"))
    for r in out_rows:
        print("%-16s %-10s %5d %8.4f %9.4f"
              % (r["group"], r["subset"], r["n"], r["accuracy"],
                 r["macro_f1"]))
    print("\nwrote", (OUTDIR / "clean_test_eval.csv").relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
