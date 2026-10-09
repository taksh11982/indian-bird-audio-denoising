# Gavali audio-only A-vs-C pipeline — 2026-10-10

- **Prompt:** build whatever is necessary + step-by-step run guide
- **Phase:** P4-Gavali (audio-only LSTM/MFCC, raw vs cleaned, no images)
- **Changed:** `src/gavali_features.py:1` — new MFCC adapter (39,1024,512 → 157×39, per-coeff std, asserts 5.0 s/16 kHz)
- **Changed:** `src/gavali_splits.py:1` — new recording-level 80/10/10 splits (raw3209: 2563/412/234; cleaned2268: 1837/259/172; gold 150 excluded, overlap 0)
- **Changed:** `src/gavali_compare.py:1` — new A-vs-C runner (sklearn smoke local + torch LSTM Colab, same raw test, Macro-F1/acc/confusions, _smoke suffix guards paper files)
- **Changed:** `src/spotcheck.py:1` — new 30-clip sampler (2/species + 8 extra, all 11 covered)
- **Changed:** `src/ibc53_loader.py:1` — new torch Dataset loader over cleaned CSV (Colab artifact)
- **Changed:** `GAVALI_RUNBOOK.md:1` — new step-by-step (verify → local baseline → Colab LSTM → spotcheck → what matters)
- **Specs touched:** audio grid kept (16 kHz/1024/512); MFCC-39 ADDED alongside frozen Log-Mel 128×157 (parallel Gavali path, CNN spec intact); thr 0.20-0.65-0.50, 5.0 s, seed 42 unchanged; torch used (pre-approved agents.md:76)
- **Gold-set contact:** none (150 excluded pre-split, asserted in splits+compare; spotcheck also excludes gold)
- **Tests:** `--check` 3/3 (157,39) OK; splits counts verified; spotcheck 11/11 covered; sklearn `--smoke` end-to-end OK; lstm-without-torch exits 2 with Colab msg; py_compile all 5 OK; full sklearn running in bg
- **For next agent:** read `results/gavali_ablation.csv` DELTA after bg job; paper numbers come ONLY from Colab `--backend lstm`; never quote `_smoke` F1
