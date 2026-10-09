"""Gavali A-vs-C: same splits/seed/test, only train data differs (P4-Gavali).

A = LSTM/MFCC trained on RAW11 train   (noisy labels kept)
C = LSTM/MFCC trained on CLEANED11 train (concordant-only)
Both evaluated on the SAME raw test (test recordings, all label-usable
chunks) -> honest generalisation; testing C on cleaned-only is circular and
is NOT done here. Seed 42 everywhere. Gold chunk_ids were removed in
gavali_splits.py and are asserted absent again here.
Backends: sklearn (local smoke, mean-pooled MFCC + LogReg, no torch) and
lstm (torch, Colab T4). Metrics: accuracy + Macro-F1 + confusion matrices.
Outputs: results/gavali_ablation.csv, results/confusion_A.csv/.png,
results/confusion_C.csv/.png (+ .pt checkpoints for lstm).
Usage:
  python src/gavali_compare.py --backend sklearn --smoke   # ~2 min local check
  python src/gavali_compare.py --backend sklearn           # full local (~50 min)
  python src/gavali_compare.py --backend lstm --epochs 30 --device cuda
"""
import argparse
import csv
import pathlib
import random

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
OUTDIR = REPO / "results"


def load_splits():
    import csv as _csv
    tri = {r["chunk_id"]: r for r in
           _csv.DictReader(open(REPO / "data" / "audit_triage.csv",
                                encoding="utf-8"))}
    idx = {r["chunk_id"]: r for r in
           _csv.DictReader(open(REPO / "data" / "chunk_index.csv",
                                encoding="utf-8"))}
    gold = {r["chunk_id"] for r in
            _csv.DictReader(open(REPO / "data" / "gold_test_150.csv",
                                 encoding="utf-8"))}
    rows = list(_csv.DictReader(open(OUTDIR / "splits_by_chunk.csv",
                                     encoding="utf-8")))
    assert not (set(r["chunk_id"] for r in rows) & gold), "gold leakage"
    return tri, idx, rows


def extract(split_rows, idx, pool_col, label_to_idx):
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from gavali_features import mfcc_sequence, mean_pool
    X_seq, X_mean, y, ids = [], [], [], []
    for r in split_rows:
        if r[pool_col] != "1":
            continue
        seq = mfcc_sequence(str(REPO / idx[r["chunk_id"]]["path_16k"]))
        X_seq.append(seq)
        X_mean.append(mean_pool(seq))
        y.append(label_to_idx[r["species"]])
        ids.append(r["chunk_id"])
    return np.stack(X_seq), np.stack(X_mean), np.array(y), ids


def save_confusion(cm, classes, path_csv, path_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with open(path_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["true/pred"] + classes)
        for i, s in enumerate(classes):
            w.writerow([s] + list(map(int, cm[i])))

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm, aspect="auto")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=11)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=6, color="white" if cm[i, j] > cm.max() / 2
                        else "black")
    fig.tight_layout()
    fig.savefig(path_png, dpi=200)
    plt.close(fig)


def cap_rows(rows, cap, seed=42):
    """Deterministic smoke subset (keeps file order stable, seed 42)."""
    if cap <= 0 or len(rows) <= cap:
        return rows
    rng = random.Random(seed)
    keep = set(rng.sample(range(len(rows)), cap))
    return [r for i, r in enumerate(rows) if i in keep]


def run_sklearn(classes, label_to_idx, tri, idx, rows, smoke=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.preprocessing import StandardScaler
    out = []
    cms = {}
    for run, pool_col in (("A-raw", "in_raw"), ("C-cleaned", "in_cleaned")):
        tr = cap_rows([r for r in rows if r["split"] == "train"],
                      200 if smoke else 0)
        va = cap_rows([r for r in rows if r["split"] == "val"],
                      60 if smoke else 0)
        te = cap_rows([r for r in rows if r["split"] == "test"
                       and r["in_raw"] == "1"], 60 if smoke else 0)
        _, Xtr, ytr, _ = extract(tr, idx, pool_col, label_to_idx)
        _, Xva, yva, _ = extract(va, idx, pool_col, label_to_idx)
        _, Xte, yte, _ = extract(te, idx, "in_raw", label_to_idx)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=500,
                                 class_weight="balanced", random_state=42)
        clf.fit(sc.transform(Xtr), ytr)
        for name, Xe, ye in (("val", Xva, yva), ("test", Xte, yte)):
            p = clf.predict(sc.transform(Xe))
            out.append({"run": run, "split": name, "n_train": len(ytr),
                        "n_eval": len(ye),
                        "accuracy": round(float(accuracy_score(ye, p)), 4),
                        "macro_f1": round(float(f1_score(ye, p,
                                                         average="macro")), 4)})
        if run.startswith("A"):
            cms["A"] = confusion_matrix(yte,
                                        clf.predict(sc.transform(Xte)),
                                        labels=list(range(len(classes))))
        else:
            cms["C"] = confusion_matrix(yte,
                                        clf.predict(sc.transform(Xte)),
                                        labels=list(range(len(classes))))
        print(f"{run}: train={len(ytr)} val_macro_f1={out[-2]['macro_f1']} "
              f"test_macro_f1={out[-1]['macro_f1']}")
    return out, cms


def run_lstm(classes, label_to_idx, tri, idx, rows, epochs, device):
    import torch
    import torch.nn as nn
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    class LSTMClf(nn.Module):
        def __init__(self, d_in=39, d_h=128, n_cls=11):
            super().__init__()
            self.lstm = nn.LSTM(d_in, d_h, num_layers=1, batch_first=True)
            self.drop = nn.Dropout(0.2)
            self.fc = nn.Linear(d_h, n_cls)

        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.fc(self.drop(h[-1]))

    out, cms = [], {}
    te_rows = [r for r in rows if r["split"] == "test" and r["in_raw"] == "1"]
    Xte_s, _, yte, _ = extract(te_rows, idx, "in_raw", label_to_idx)
    te_loader = DataLoader(TensorDataset(torch.from_numpy(Xte_s),
                                         torch.from_numpy(yte)),
                           batch_size=64)
    for run, pool_col in (("A-raw", "in_raw"), ("C-cleaned", "in_cleaned")):
        tr = [r for r in rows if r["split"] == "train"]
        va = [r for r in rows if r["split"] == "val"]
        Xs_tr, _, ytr, _ = extract(tr, idx, pool_col, label_to_idx)
        Xs_va, _, yva, _ = extract(va, idx, pool_col, label_to_idx)
        w = len(ytr) / (len(set(ytr.tolist())) *
                        np.bincount(ytr, minlength=len(classes)).clip(min=1))
        model = LSTMClf(n_cls=len(classes)).to(device)
        opt = torch.optim.RMSprop(model.parameters(), lr=0.001)
        crit = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(w, dtype=torch.float32).to(device))
        tr_loader = DataLoader(TensorDataset(torch.from_numpy(Xs_tr),
                                             torch.from_numpy(ytr)),
                               batch_size=64, shuffle=True,
                               generator=torch.Generator().manual_seed(42))
        va_t = torch.from_numpy(Xs_va).to(device)
        best_f1, best_state, wait = -1.0, None, 0
        model.train()
        for ep in range(epochs):
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                pv = model(va_t).argmax(1).cpu().numpy()
            f1 = f1_score(yva, pv, average="macro")
            if f1 > best_f1:
                best_f1, best_state, wait = f1, {
                    k: v.cpu() for k, v in model.state_dict().items()}, 0
            else:
                wait += 1
            model.train()
            if wait >= 6:
                break  # ponytail: patience 6; upgrade = full early-stop sweep
        model.load_state_dict({k: v.to(device) for k, v in
                               best_state.items()})
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in te_loader:
                preds.append(model(xb.to(device)).argmax(1).cpu().numpy())
        p = np.concatenate(preds)
        out.append({"run": run, "split": "test", "n_train": len(ytr),
                    "n_eval": len(yte),
                    "accuracy": round(float(accuracy_score(yte, p)), 4),
                    "macro_f1": round(float(f1_score(yte, p,
                                                     average="macro")), 4),
                    "epochs_run": epochs})
        cms[run[0]] = confusion_matrix(yte, p,
                                       labels=list(range(len(classes))))
        torch.save(best_state, OUTDIR / f"gavali_lstm_{run[0]}.pt")
        print(f"{run}: train={len(ytr)} test_macro_f1={out[-1]['macro_f1']}")
    return out, cms


def main() -> int:
    random.seed(42)
    np.random.seed(42)
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="sklearn", choices=["sklearn", "lstm"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--smoke", action="store_true",
                    help="fast local check: 200/60/60 train/val/test")
    args = ap.parse_args()
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from gavali_features import FOCUS_11
    classes = sorted(FOCUS_11)
    label_to_idx = {s: i for i, s in enumerate(classes)}
    tri, idx, rows = load_splits()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if args.backend == "sklearn":
        out, cms = run_sklearn(classes, label_to_idx, tri, idx, rows,
                               smoke=args.smoke)
    else:
        try:
            import torch
        except ImportError:
            print("ERROR: torch not installed. Run the lstm backend on "
                  "Colab T4: pip install torch --index-url "
                  "https://download.pytorch.org/whl/cu121 ; "
                  "python src/gavali_compare.py --backend lstm --epochs 30 "
                  "--device cuda")
            return 2
        dev = args.device if torch.cuda.is_available() or \
            args.device == "cpu" else "cpu"
        out, cms = run_lstm(classes, label_to_idx, tri, idx, rows,
                            args.epochs, dev)
    suffix = "_smoke" if args.smoke else ""
    with open(OUTDIR / f"gavali_ablation{suffix}.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "split", "n_train",
                                           "n_eval", "accuracy", "macro_f1"])
        w.writeheader()
        for r in out:
            w.writerow({k: r.get(k, "") for k in
                        ["run", "split", "n_train", "n_eval", "accuracy",
                         "macro_f1"]})
    te = [o for o in out if o["split"] == "test"]
    if len(te) == 2:
        d = round(te[1]["macro_f1"] - te[0]["macro_f1"], 4)
        print(f"DELTA test Macro-F1 (C - A) = {d}")
    save_confusion(cms["A"], classes, OUTDIR / f"confusion_A{suffix}.csv",
                   REPO / "figures" / f"confusion_gavali_A{suffix}.png",
                   f"Gavali acoustic — A (raw train) on raw test{suffix}")
    save_confusion(cms["C"], classes, OUTDIR / f"confusion_C{suffix}.csv",
                   REPO / "figures" / f"confusion_gavali_C{suffix}.png",
                   f"Gavali acoustic — C (cleaned train) on raw test{suffix}")
    print(f"wrote {OUTDIR / f'gavali_ablation{suffix}.csv'} + confusion A/C "
          f"csv+png (backend={args.backend}{' smoke' if args.smoke else ''})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
