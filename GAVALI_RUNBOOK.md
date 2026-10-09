# Gavali audio-only runbook (raw vs cleaned) — read top to bottom, run in order

Scope: 11 species (focus-12 minus `Sphenocichla humei`, unauditable).
Test is RAW chunks in held-out test recordings for BOTH runs — never
cleaned-only (that would be circular). Seed 42 everywhere. Gold excluded
before splitting, asserted absent in every script.

## 0. Prereqs (once)

- Python 3.11+, local: `pip install librosa soundfile numpy pandas
  scikit-learn matplotlib` (torch NOT needed locally).
- Colab T4 for the paper model: `pip install torch librosa soundfile
  numpy pandas scikit-learn matplotlib` (torch only required for LSTM).
- Data stays local: `data/chunks_16k/`, `data/birdnet_scores/` are
  git-ignored. Small CSVs (`audit_triage`, `chunk_index`, `gold_test_150`,
  `iBC53-Cleaned-Metadata`, `results/splits_*`) commit fine.

## 1. Verify the pipeline (local, ~2 min) — must pass before Colab

```
python src/gavali_features.py --check
python src/gavali_splits.py
python src/spotcheck.py
python src/gavali_compare.py --backend sklearn --smoke
```

Expected (verified 2026-10-10):

- `features OK: 3/3 chunks -> (157, 39) standardised`
- `raw11_nogold=3209 {'train': 2563, 'val': 412, 'test': 234}`
- `cleaned11_nogold=2268 {'train': 1837, 'val': 259, 'test': 172}`
- `gold excluded: 150 chunk_ids, overlap in splits: 0`
- `spotcheck=30` with 2–3 clips per each of the 11 species
- smoke writes `results/gavali_ablation_smoke.csv` + `confusion_*_smoke.*`
  (pipeline check ONLY — never quote smoke Macro-F1 in the paper).

## 2. Full local baseline (optional, ~50 min, no torch)

```
python src/gavali_compare.py --backend sklearn
```

Writes the REAL baseline files (no `_smoke` suffix):
`results/gavali_ablation.csv`, `results/confusion_A.csv`,
`results/confusion_C.csv`, `figures/confusion_gavali_A.png`,
`figures/confusion_gavali_C.png`. LogisticRegression on mean-pooled MFCC —
weak by design; it proves the splits/features/metrics work, not your headline.

## 3. Paper model (Colab T4, the numbers that matter)

1. Transfer data: zip `data/chunks_16k/`, `data/chunk_index.csv`,
   `data/audit_triage.csv`, `data/gold_test_150.csv`,
   `data/iBC53-Cleaned-Metadata.csv`, `results/splits_by_chunk.csv` to
   Drive, unzip on Colab (or re-run ingest→chunk→audit there from Kaggle).
2. Copy `src/` + `results/splits_by_chunk.csv` unchanged (same seed/splits).
3. Run:

```
python src/gavali_compare.py --backend lstm --epochs 30 --device cuda
```

Saves `results/gavali_lstm_A.pt/.pt` (checkpoints),
overwrites `results/gavali_ablation.csv` + confusion A/C with LSTM numbers.

## 4. Human spot-check (one evening, NOT training data)

Open each of the 30 wavs in `results/spotcheck_30.csv` (`path_16k`) in
Audacity vs `data/xc_refs/`, fill `keep` (y/n) + `notes`. Report
`x/30 kept` in the paper. Never add these rows to any train split.

## 5. Outputs that matter most (in order)

1. `results/gavali_ablation.csv` → `DELTA test Macro-F1 (C - A)` printed at
   end of run. THE headline: same test, only train data differs.
2. `figures/confusion_gavali_A.png` vs `_C.png` → Fig.3 (where errors move).
3. `figures/noise_summary.png` (BEFORE) + `noise_summary_cleaned.png`
   (AFTER, all green by filtering) → Fig.1 pair. Never edit BEFORE.
4. `results/splits_summary.json` (3209 vs 2268, recording-level 80/10/10) →
   Methods proof of no leakage.
5. `results/spotcheck_30.csv` graded → "x/30 contained target" sentence.

## 6. How to present it (paper-safe wording)

- Claim: "unsupervised BirdNET triage + concordant-only training lifts the
  Gavali acoustic branch by +X Macro-F1 on a fixed raw held-out test."
- Always: label cleaning, NOT waveform enhancement; name the 9 zero-kept
  taxa as a limitation; state test is raw (noisy) for both runs.
- If DELTA ≤ 0, report it — honest negative + audit still passes workshops;
  do not retune thresholds/features to chase a target.

## 7. If something breaks

- `torch not installed` → expected locally; run `--backend sklearn --smoke`
  here, `--backend lstm` only on Colab.
- Slow MFCC (~0.5 s/chunk) → normal; full sklearn ≈ 50 min local,
  LSTM ≈ 10–20 min on T4.
- `results/` holds ONLY machine-written CSVs/PNGs — never hand-edit numbers.
