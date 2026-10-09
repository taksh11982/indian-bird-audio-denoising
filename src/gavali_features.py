"""Gavali acoustic adapter: 16 kHz 5.0 s wav -> MFCC sequence (P4-Gavali).

Frozen audio grid shared with agents.md Log-Mel spec: SR=16000, n_fft=1024,
hop_length=512 -> 157 frames per 5.0 s chunk (center=True). Gavali acoustic
branch uses MFCCs + LSTM; we freeze N_MFCC=39 (their SET2) for both runs.
Per-chunk standardisation (mean 0 / std 1 per coefficient, eps=1e-6):
deterministic, no leakage across chunks. Seed 42 kept (no stochastic ops).
Usage: python src/gavali_features.py --check
"""
import argparse
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]

SR = 16000
N_MFCC = 39
N_FFT = 1024
HOP = 512
N_FRAMES = 157  # 1 + 80000//512 with center=True padding (verified)
EPS = 1e-6

FOCUS_11 = [
    "Pellorneum ruficeps",
    "Cuculus micropterus",
    "Arborophila torqueola",
    "Pomatorhinus ruficollis",
    "Glaucidium cuculoides",
    "Cyornis unicolor",
    "Pnoepyga pusilla",
    "Liocichla phoenicea",
    "Rimator malacoptilus",
    "Psilopogon lineatus",
    "Psittacula eupatria",
]
LABEL_TO_IDX = {s: i for i, s in enumerate(sorted(FOCUS_11))}


def mfcc_sequence(path_16k: str) -> np.ndarray:
    """(157, 39) float32 time-major MFCC, per-coefficient standardised."""
    import librosa
    y, sr = __import__("librosa").load(str(path_16k), sr=SR, mono=True)
    assert sr == SR, sr
    assert y.shape == (SR * 5,), y.shape  # exactly 5.0 s per chunking DoD
    m = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC,
                             n_fft=N_FFT, hop_length=HOP)  # (39, T)
    assert m.shape == (N_MFCC, N_FRAMES), m.shape
    mu = m.mean(axis=1, keepdims=True)
    sd = m.std(axis=1, keepdims=True) + EPS
    m = ((m - mu) / sd).astype(np.float32).T  # (157, 39)
    return m


def mean_pool(seq: np.ndarray) -> np.ndarray:
    """(39,) mean over time for the sklearn smoke backend."""
    return seq.mean(axis=0).astype(np.float32)


def main() -> int:
    import random
    random.seed(42)
    np.random.seed(42)
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        import csv
        rows = list(csv.DictReader(open(REPO / "data" / "chunk_index.csv",
                                        encoding="utf-8")))
        ok = 0
        for r in rows[:3]:
            p = REPO / r["path_16k"]
            s = mfcc_sequence(str(p))
            assert s.shape == (N_FRAMES, N_MFCC), s.shape
            assert np.isfinite(s).all()
            ok += 1
        print(f"features OK: {ok}/3 chunks -> (157, 39) standardised")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
