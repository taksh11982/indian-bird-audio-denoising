"""Shared acoustic model, features and metrics for every Gavali-style run.

Single source of truth so that the A-vs-C ablation, the equal-size control
and the noise-scaling curve are trained with byte-identical code, features,
hyper-parameters and seeds (seed 42). Only the training data differs.

Features: 16 kHz 5.0 s chunk -> 39 MFCC x 157 frames, per-chunk
standardised (see src/gavali_features.py). Cached to
data/feat_cache_lstm.npz so all runs share one extraction pass.
torch and librosa are imported lazily, so the sklearn path stays torch-free.
"""
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
CACHE = REPO / "data" / "feat_cache_lstm.npz"

D_IN, D_H, DROPOUT = 39, 128, 0.2
LR, BATCH, PATIENCE = 0.001, 64, 6
SEED = 42


def feature_matrix(ids, idx, refresh=False):
    """(n, 157, 39) float32 MFCC features for the given chunk_ids."""
    if CACHE.exists() and not refresh:
        z = np.load(CACHE, allow_pickle=True)
        order = {c: i for i, c in enumerate(z["ids"].tolist())}
        if all(c in order for c in ids):
            feats = z["feats"]
            return np.stack([feats[order[c]] for c in ids])
    from gavali_features import mfcc_sequence
    print(f"extracting MFCC for {len(ids)} chunks (one pass, cached) ...",
          flush=True)
    feats = np.stack([mfcc_sequence(str(REPO / idx[c]["path_16k"]))
                       for c in ids]).astype(np.float32)
    np.savez_compressed(CACHE, ids=np.array(ids), feats=feats)
    return feats


def macro_f1(y_true, y_pred):
    """Macro-F1 over the union of true/pred labels (sklearn 'macro' default)."""
    f1s = []
    for c in sorted(set(np.asarray(y_true).tolist()) |
                    set(np.asarray(y_pred).tolist())):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        den = 2 * tp + fp + fn
        f1s.append(0.0 if den == 0 else 2.0 * tp / den)
    return float(np.mean(f1s)) if f1s else 0.0


def build_lstm(n_cls):
    import torch.nn as nn

    class LSTMClf(nn.Module):
        def __init__(self, d_in=D_IN, d_h=D_H, n_cls=n_cls):
            super().__init__()
            self.lstm = nn.LSTM(d_in, d_h, num_layers=1, batch_first=True)
            self.drop = nn.Dropout(DROPOUT)
            self.fc = nn.Linear(d_h, n_cls)

        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.fc(self.drop(h[-1]))

    return LSTMClf(n_cls=n_cls)


def class_weights(ytr, n_cls):
    y = np.asarray(ytr)
    return len(y) / (len(set(y.tolist())) *
                     np.bincount(y, minlength=n_cls).clip(min=1))


def train_lstm(Xtr, ytr, Xva, yva, n_cls, epochs, device, seed=SEED):
    """Train with early stopping on val Macro-F1; returns (model, best_f1, ep)."""
    import random

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = build_lstm(n_cls).to(device)
    opt = torch.optim.RMSprop(model.parameters(), lr=LR)
    w = class_weights(ytr, n_cls)
    crit = nn.CrossEntropyLoss(
        weight=torch.tensor(w, dtype=torch.float32).to(device))
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr),
                                      torch.from_numpy(ytr)),
                        batch_size=BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    Xva_t = torch.from_numpy(Xva).to(device)
    best_f1, best_state, wait = -1.0, None, 0
    model.train()
    ep = 0
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
        f1 = macro_f1(yva, pv)
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
    return model, float(best_f1), ep + 1


def predict(model, X, device):
    import torch
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            xb = torch.from_numpy(X[i:i + BATCH]).to(device)
            preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds)
