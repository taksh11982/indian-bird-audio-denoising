"""Control experiment: is the gain caused by CLEANING or by having LESS data?

The teacher-proof question: "you removed data, so of course it got easier -
what did you actually do?" The answer needs a control that removes the same
amount of data WITHOUT cleaning:

  A-full    2,563 chunks  raw labels, no removal              (Model A)
  R-random  1,837 chunks  RANDOM subset of A's pool, labels untouched
  C-clean   1,837 chunks  the audited concordant subset       (Model C)

Same model, same seed, same test set. If C-clean > R-random, the gain comes
from the label cleaning, not from the smaller training set. R-random is the
honest baseline every "we removed the bad data" claim has to beat.

The test set is reported twice: on all 234 raw test chunks (labels are
partly wrong -> lower bound) and on the 172 audited-trusted chunks.

Usage:
  python src/control_subset.py --epochs 30 --device cuda   (Colab T4)
  python src/control_subset.py --epochs 30 --device cpu
Outputs:
  results/control_subset.csv          metrics for the three runs
  results/preds_control_{A,R,C}.csv   per-chunk predictions
  results/gavali_lstm_{A,R,C}.pt      checkpoints
"""
import argparse
import csv
import pathlib
import random
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gavali_compare import load_splits  # noqa: E402
from gavali_features import FOCUS_11  # noqa: E402
from gavali_model import (feature_matrix, macro_f1, predict,  # noqa: E402
                          train_lstm)

OUTDIR = REPO / "results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--subset-size", type=int, default=0,
                    help="0 = same size as the cleaned pool")
    args = ap.parse_args()
    random.seed(42)
    np.random.seed(42)

    import torch
    device = args.device if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

    classes = sorted(FOCUS_11)
    label_to_idx = {s: i for i, s in enumerate(classes)}
    tri, idx, rows = load_splits()

    tr_raw = [r["chunk_id"] for r in rows
              if r["split"] == "train" and r["in_raw"] == "1"]
    tr_cl = [r["chunk_id"] for r in rows
             if r["split"] == "train" and r["in_cleaned"] == "1"]
    va_cl = [r["chunk_id"] for r in rows
             if r["split"] == "val" and r["in_cleaned"] == "1"]
    va_raw = [r["chunk_id"] for r in rows
              if r["split"] == "val" and r["in_raw"] == "1"]
    te = [r["chunk_id"] for r in rows
          if r["split"] == "test" and r["in_raw"] == "1"]
    trusted = [c for c in te if tri[c]["triage"] == "concordant"]
    tset = set(trusted)

    k = args.subset_size or len(tr_cl)
    rng = random.Random(42)
    rand_subset = sorted(rng.sample(tr_raw, k))

    runs = [
        ("A-full", tr_raw, va_raw),
        ("R-random", rand_subset, va_raw),
        ("C-clean", tr_cl, va_cl),
    ]
    print(f"train sizes: " + ", ".join(f"{n}={len(ids)}" for n, ids, _ in runs),
          flush=True)

    all_ids = tr_raw + tr_cl + va_cl + va_raw + te
    all_ids = list(dict.fromkeys(all_ids))
    feats = feature_matrix(all_ids, idx)
    pos = {c: i for i, c in enumerate(all_ids)}

    def take(ids):
        return np.stack([feats[pos[c]] for c in ids])

    Xte = take(te)
    yte = np.array([label_to_idx[tri[c]["given_label"]] for c in te])
    trusted_mask = np.array([c in tset for c in te])

    out_rows, pred_rows = [], []
    for run, train_ids, val_ids in runs:
        Xtr = take(train_ids)
        ytr = np.array([label_to_idx[tri[c]["given_label"]] for c in train_ids])
        Xva = take(val_ids)
        yva = np.array([label_to_idx[tri[c]["given_label"]] for c in val_ids])
        model, best_f1, ep = train_lstm(Xtr, ytr, Xva, yva, len(classes),
                                        args.epochs, device)
        p = predict(model, Xte, device)
        acc_all = float((p == yte).mean())
        f1_all = macro_f1(yte, p)
        acc_tr = float((p[trusted_mask] == yte[trusted_mask]).mean())
        f1_tr = macro_f1(yte[trusted_mask], p[trusted_mask])
        out_rows.append({
            "run": run, "n_train": len(train_ids), "n_test": len(yte),
            "accuracy": round(acc_all, 4), "macro_f1": round(f1_all, 4),
            "accuracy_trusted": round(acc_tr, 4),
            "macro_f1_trusted": round(f1_tr, 4),
            "n_test_trusted": int(trusted_mask.sum()),
            "val_f1": round(best_f1, 4), "epochs_run": ep,
        })
        print(f"{run:9s} n_train={len(train_ids):5d} "
              f"acc_all={acc_all:.4f} f1_all={f1_all:.4f} | "
              f"acc_trusted={acc_tr:.4f} f1_trusted={f1_tr:.4f}", flush=True)
        with open(OUTDIR / f"preds_control_{run[0]}.csv", "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["chunk_id", "species", "pred", "correct", "triage"])
            for c, yi, pi in zip(te, yte, p):
                w.writerow([c, classes[int(yi)], classes[int(pi)],
                            int(yi == pi), tri[c]["triage"]])
        torch.save(model.state_dict(), OUTDIR / f"gavali_lstm_{run[0]}.pt")
        for c, yi, pi in zip(te, yte, p):
            pred_rows.append({"run": run, "chunk_id": c,
                              "species": classes[int(yi)], "pred": classes[int(pi)],
                              "triage": tri[c]["triage"]})

    OUTDIR.mkdir(parents=True, exist_ok=True)
    with open(OUTDIR / "control_subset.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print("wrote", (OUTDIR / "control_subset.csv").relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
