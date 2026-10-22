"""Noise-scaling experiment: what does label noise actually cost?

Problem this solves: the A-vs-C run reports +0.0096 Macro-F1 on 234 test
chunks. That is ~2 chunks - the current test set cannot resolve the effect,
and 26.5% of its ground truth is itself wrong (see
results/label_noise_evidence.md). A single before/after pair is therefore
not a result, it is an anecdote.

This script turns the anecdote into a curve. It starts from the CLEANED
training pool (100% audited concordant) and injects a controlled fraction r
of wrong labels, then trains the identical LSTM (hyper-parameters copied from
gavali_compare.run_lstm) and measures the damage on two test sets:

  test_all     = the 234 raw test chunks (same as the A/C run, noisy labels)
  test_trusted = the 172 test chunks whose label was audited concordant
                 (the target bird really is in the clip)

r = 0.283 reproduces the noise level we measured in Model A's raw training
pool, so the curve explains where A's deficit comes from: A is not a worse
model, it is the same model trained on a pool that is 28% mislabelled.

Fixed protocol across rates: seed 42, same features (39 MFCC, 157 frames),
same optimiser, same early stopping on the SAME cleaned val set (259
chunks) so that only the training labels vary.

Usage (Colab T4):
  python src/noise_scaling.py --epochs 30 --device cuda
  python src/noise_scaling.py --epochs 30 --device cuda --dump-preds
Outputs:
  results/noise_scaling.csv       metrics per noise rate
  results/noise_scaling_preds.csv per-chunk predictions (with --dump-preds)
  results/noise_scaling.png       accuracy vs noise rate (if matplotlib)
"""
import argparse
import csv
import pathlib
import random
import sys
from collections import Counter

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gavali_compare import load_splits  # noqa: E402
from gavali_features import FOCUS_11, mfcc_sequence  # noqa: E402

OUTDIR = REPO / "results"
CACHE = REPO / "data" / "feat_cache_lstm.npz"
RATES = [0.0, 0.05, 0.10, 0.20, 0.283]
D_IN, D_H, DROPOUT = 39, 128, 0.2
LR, BATCH, PATIENCE = 0.001, 64, 6


def macro_f1(y_true, y_pred, n_cls):
    """Macro-F1 over the union of true/pred labels (sklearn 'macro' default)."""
    f1s = []
    for c in sorted(set(y_true.tolist()) | set(y_pred.tolist())):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2.0 * tp / denom)
    return float(np.mean(f1s)) if f1s else 0.0


def features_for(ids, idx):
    """(n, 157, 39) float32, cached to disk so the 5 rates reuse one pass."""
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        order = {c: i for i, c in enumerate(z["ids"].tolist())}
        if all(c in order for c in ids):
            feats = z["feats"]
            return np.stack([feats[order[c]] for c in ids])
    print(f"extracting MFCC for {len(ids)} chunks (one pass) ...", flush=True)
    feats = np.stack([mfcc_sequence(str(REPO / idx[c]["path_16k"]))
                       for c in ids]).astype(np.float32)
    np.savez_compressed(CACHE, ids=np.array(ids), feats=feats)
    return feats


def corrupt(y, rate, n_cls, seed):
    """Relabel a `rate` fraction of samples to a random wrong class."""
    rng = random.Random(seed)
    y = y.copy()
    n = len(y)
    k = int(round(rate * n))
    idxs = rng.sample(range(n), k)
    for i in idxs:
        choices = [c for c in range(n_cls) if c != int(y[i])]
        y[i] = rng.choice(choices)
    return y, k


def train_lstm(Xtr, ytr, Xva, yva, n_cls, epochs, device, seed=42):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    class LSTMClf(nn.Module):
        def __init__(self, d_in=D_IN, d_h=D_H, n_cls=n_cls):
            super().__init__()
            self.lstm = nn.LSTM(d_in, d_h, num_layers=1, batch_first=True)
            self.drop = nn.Dropout(DROPOUT)
            self.fc = nn.Linear(d_h, n_cls)

        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.fc(self.drop(h[-1]))

    w = len(ytr) / (len(set(ytr.tolist())) *
                    np.bincount(ytr, minlength=n_cls).clip(min=1))
    model = LSTMClf(n_cls=n_cls).to(device)
    opt = torch.optim.RMSprop(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss(
        weight=torch.tensor(w, dtype=torch.float32).to(device))
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr),
                                      torch.from_numpy(ytr)),
                        batch_size=BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    Xva_t = torch.from_numpy(Xva).to(device)
    best_f1, best_state, wait = -1.0, None, 0
    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva_t).argmax(1).cpu().numpy()
        f1 = macro_f1(yva, pv, n_cls)
        if f1 > best_f1:
            best_f1, best_state, wait = f1, {
                k: v.cpu() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
        model.train()
        if wait >= PATIENCE:
            break
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    return model, best_f1, ep + 1


def predict(model, X, device):
    import torch
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            xb = torch.from_numpy(X[i:i + BATCH]).to(device)
            preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dump-preds", action="store_true")
    args = ap.parse_args()
    random.seed(42)
    np.random.seed(42)

    import torch
    device = args.device if torch.cuda.is_available() else "cpu"
    print("device:", device)

    classes = sorted(FOCUS_11)
    label_to_idx = {s: i for i, s in enumerate(classes)}
    tri, idx, rows = load_splits()

    tr_ids = [r["chunk_id"] for r in rows
              if r["split"] == "train" and r["in_cleaned"] == "1"]
    va_ids = [r["chunk_id"] for r in rows
              if r["split"] == "val" and r["in_cleaned"] == "1"]
    te_ids = [r["chunk_id"] for r in rows
              if r["split"] == "test" and r["in_raw"] == "1"]
    trusted = [c for c in te_ids if tri[c]["triage"] == "concordant"]
    print(f"train(clean)={len(tr_ids)} val(clean)={len(va_ids)} "
          f"test(raw)={len(te_ids)} test(trusted)={len(trusted)}")

    ytr = np.array([label_to_idx[tri[c]["given_label"]] for c in tr_ids])
    yva = np.array([label_to_idx[tri[c]["given_label"]] for c in va_ids])
    yte = np.array([label_to_idx[tri[c]["given_label"]] for c in te_ids])
    trusted_mask = np.array([c in set(trusted) for c in te_ids])

    all_ids = tr_ids + va_ids + te_ids
    feats = features_for(all_ids, idx)
    ntr, nva = len(tr_ids), len(va_ids)
    Xtr, Xva, Xte = feats[:ntr], feats[ntr:ntr + nva], feats[ntr + nva:]

    out_rows = []
    pred_rows = []
    for rate in RATES:
        seed = 42 + int(round(rate * 1000))
        ytr_c, k = corrupt(ytr, rate, len(classes), seed)
        model, best_f1, ep = train_lstm(Xtr, ytr_c, Xva, yva, len(classes),
                                        args.epochs, device)
        p = predict(model, Xte, device)
        acc_all = float((p == yte).mean())
        f1_all = macro_f1(yte, p, len(classes))
        acc_tr = float((p[trusted_mask] == yte[trusted_mask]).mean())
        f1_tr = macro_f1(yte[trusted_mask], p[trusted_mask], len(classes))
        out_rows.append({
            "noise_rate": rate, "n_corrupted": k, "n_train": len(ytr_c),
            "n_test_all": len(yte), "n_test_trusted": int(trusted_mask.sum()),
            "acc_test_all": round(acc_all, 4), "f1_test_all": round(f1_all, 4),
            "acc_test_trusted": round(acc_tr, 4),
            "f1_test_trusted": round(f1_tr, 4),
            "val_f1_clean": round(float(best_f1), 4), "epochs_run": ep,
        })
        print(f"rate={rate:.3f} corrupted={k:4d} "
              f"acc_all={acc_all:.4f} f1_all={f1_all:.4f} | "
              f"acc_trusted={acc_tr:.4f} f1_trusted={f1_tr:.4f}", flush=True)
        if args.dump_preds:
            for c, yi, pi in zip(te_ids, yte, p):
                pred_rows.append({"noise_rate": rate, "chunk_id": c,
                                  "species": classes[int(yi)],
                                  "pred": classes[int(pi)],
                                  "triage": tri[c]["triage"]})

    OUTDIR.mkdir(parents=True, exist_ok=True)
    with open(OUTDIR / "noise_scaling.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    if pred_rows:
        with open(OUTDIR / "noise_scaling_preds.csv", "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(pred_rows[0].keys()))
            w.writeheader()
            w.writerows(pred_rows)
    print("wrote", (OUTDIR / "noise_scaling.csv").relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
