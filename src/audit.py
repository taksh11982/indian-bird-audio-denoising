"""P2 macro audit: BirdNET-Analyzer V2.4 (CPU) over 48 kHz chunks.

Stage 1 (scores): sliding 3.0 s windows (SIG_OVERLAP=0.0, BirdNET default),
  max-pool window outputs to one vector S in [0,1]^53 over the iBC53 species
  order (data/birdnet_species_order.csv) -> data/birdnet_scores/<chunk_id>.npz.
  Per-chunk checkpointing: existing .npz files are skipped (resume-safe).
Stage 2 (triage): data/audit_triage.csv
  (chunk_id,given_label,pred_label,max_conf,triage) with frozen rules:
  empty: max(S) < 0.20; intruder: argmax != label and max >= 0.65;
  concordant: argmax == label and max >= 0.50; else ambiguous.
  Chunks whose folder name has no BirdNET label (e.g. 'Mystery mystery')
  get triage 'unlabeled' (pred_label/max_conf still recorded).
Usage: python src/audit.py [--stage scores|triage|all]
Seed 42 set (no stochastic ops with USE_NOISE=False; kept per golden rule 1).
"""
import argparse
import csv
import pathlib
import random
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
INDEX = REPO / "data" / "chunk_index.csv"
SCORES = REPO / "data" / "birdnet_scores"
TRIAGE_CSV = REPO / "data" / "audit_triage.csv"
ORDER_CSV = REPO / "data" / "birdnet_species_order.csv"

T_EMPTY = 0.20
T_INTRUDER = 0.65
T_CONCORD = 0.50


def setup_birdnet():
    import birdnet_analyzer.config as cfg
    from birdnet_analyzer.utils import ensure_model_exists
    ensure_model_exists()
    cfg.MODEL_PATH = cfg.BIRDNET_MODEL_PATH
    cfg.LABELS_FILE = cfg.BIRDNET_LABELS_FILE
    cfg.SAMPLE_RATE = cfg.BIRDNET_SAMPLE_RATE  # 48000
    cfg.SIG_LENGTH = cfg.BIRDNET_SIG_LENGTH  # 3.0
    # ponytail: SIG_OVERLAP stays 0.0 (BirdNET default); spec pins only the
    # 3.0 s window + max-pool, so the tool default is the non-invented choice.
    cfg.BATCH_SIZE = 8
    cfg.TFLITE_THREADS = 1
    with open(cfg.LABELS_FILE, encoding="utf-8") as fh:
        cfg.LABELS = [ln.strip() for ln in fh if ln.strip()]
    return cfg


def species_order(index_rows):
    return sorted({r["species_name"] for r in index_rows})


# ponytail: folder names follow an older taxonomy than BirdNET V2.4
# (Clements 2024). Each entry matched by identical species epithet + common
# name in BirdNET_GLOBAL_6K_V2.4_Labels.txt. TODO(P5): cite Clements versions
# in the paper; 5 folders have no BirdNET counterpart at all (see below).
SYNONYMS = {
    "Alcippe cinerea": "Schoeniparus cinereus",      # Yellow-throated Fulvetta
    "Macronus gularis": "Mixornis gularis",          # Striped/Pin-striped Tit-Babbler
    "Rimator malacoptilus": "Napothera malacoptila",  # Long-billed Wren-Babbler
    "Stachyridopsis ambigua": "Cyanoderma ambiguum",  # Buff-chested Babbler
    "Iole cacharensis": "Iole viridescens",           # Cachar Bulbul split from Olive Bulbul; parent kept
}
# No BirdNET V2.4 counterpart (verified by label search): Argya longirostris,
# Cyornis magnirostris, Sphenocichla humei, Chloropsis (genus-level folder),
# Mystery mystery (no label by design). These stay 'unlabeled' in triage.


def label_index(cfg, species):
    """Index into cfg.LABELS for a folder name, or None if unmapped."""
    species = SYNONYMS.get(species, species)
    for i, lab in enumerate(cfg.LABELS):
        if lab.split("_")[0] == species:
            return i
    return None


def score_chunk(cfg, path_48k, col_idx):
    from birdnet_analyzer.analyze.utils import iterate_audio_chunks
    wins = []
    for _, _, pred in iterate_audio_chunks(path_48k):
        wins.append(np.asarray(pred, dtype=np.float64))
    if not wins:
        raise RuntimeError(f"no windows: {path_48k}")
    W = np.stack(wins)  # (n_windows, n_labels)
    assert W.shape[1] == len(cfg.LABELS), W.shape
    S = np.zeros(len(col_idx), dtype=np.float32)
    valid = col_idx >= 0  # unmapped species keep 0.0
    S[valid] = W[:, col_idx[valid]].max(axis=0).astype(np.float32)
    assert float(S.min()) >= 0.0 and float(S.max()) <= 1.0, "scores not in [0,1]"
    return S, W.shape[0]


def stage_scores(cfg, index_rows, order, col_idx):
    SCORES.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    for n, r in enumerate(index_rows, 1):
        out = SCORES / f"{r['chunk_id']}.npz"
        if out.exists():
            try:
                z = np.load(out)
                if z["S"].shape == (len(order),):
                    skipped += 1
                    continue
            except Exception:
                pass
        S, nw = score_chunk(cfg, str(REPO / r["path_48k"]), col_idx)
        np.savez_compressed(out, S=S, n_windows=np.int32(nw))
        done += 1
        if n % 200 == 0:
            print(f"scores: {n}/{len(index_rows)} (new={done} resumed={skipped})",
                  flush=True)
    print(f"scores done: new={done} resumed={skipped} total={len(index_rows)}")


def stage_triage(index_rows, order, mapped):
    with open(ORDER_CSV, "w", newline="", encoding="utf-8") as fh:
        fh.write("idx,species_name\n")
        for i, s in enumerate(order):
            fh.write(f"{i},{s}\n")
    rows = []
    for r in index_rows:
        z = np.load(SCORES / f"{r['chunk_id']}.npz")
        S = np.asarray(z["S"], dtype=np.float64)
        top = int(S.argmax())
        mc = float(S.max())
        given = r["species_name"]
        if given not in mapped:
            triage, pred = "unlabeled", order[top]
        elif mc < T_EMPTY:
            triage, pred = "empty", order[top]
        elif order[top] != given and mc >= T_INTRUDER:
            triage, pred = "intruder", order[top]
        elif order[top] == given and mc >= T_CONCORD:
            triage, pred = "concordant", order[top]
        else:
            triage, pred = "ambiguous", order[top]
        rows.append({"chunk_id": r["chunk_id"], "given_label": given,
                     "pred_label": pred, "max_conf": f"{mc:.4f}",
                     "triage": triage})
    with open(TRIAGE_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "chunk_id", "given_label", "pred_label", "max_conf", "triage"])
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print("triage:", dict(Counter(q["triage"] for q in rows)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["scores", "triage", "all"])
    args = ap.parse_args()
    random.seed(42)
    np.random.seed(42)
    with open(INDEX, newline="", encoding="utf-8") as fh:
        index_rows = list(csv.DictReader(fh))
    cfg = setup_birdnet()
    order = species_order(index_rows)
    col_idx = np.array([label_index(cfg, s) if label_index(cfg, s) is not None
                        else -1 for s in order])
    unmapped = [s for s, c in zip(order, col_idx) if c == -1]
    mapped = {s for s, c in zip(order, col_idx) if c != -1}
    print(f"species={len(order)} mapped={len(mapped)} unmapped={unmapped}")
    if args.stage in ("scores", "all"):
        stage_scores(cfg, index_rows, order, col_idx)
    if args.stage in ("triage", "all"):
        stage_triage(index_rows, order, mapped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
