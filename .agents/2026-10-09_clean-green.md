# Clean green via filtering, keep raw audit intact — 2026-10-09

- **Prompt:** make noise_summary all-green / show cleaning, drop CNN+gold, use Gavali model
- **Phase:** P2b cleaning demo (not P4 training)
- **Changed:** `src/clean.py:1` — new: keep triage==concordant only, join chunk paths, write cleaned CSV + all-green figure
- **Changed:** `data/iBC53-Cleaned-Metadata.csv:1` — new: 3553/9289 kept (38.2%), 44 species
- **Changed:** `figures/noise_summary_cleaned.png/.pdf:1` — new: AFTER figure, 100% green by construction; raw `noise_summary.*` untouched
- **Specs touched:** thr 0.20-0.65-0.50 unchanged, seed 42, 5.0 s chunks — none changed
- **Gold-set contact:** none (gold_test_150.csv not read, not modified, not used)
- **Tests:** `python src/clean.py` → kept=3553 removed empty=1575 intruder=22 ambiguous=501 unlabeled=3638; 9 species with 0 kept
- **For next agent:** wire Gavali et al. raw-vs-cleaned comparison on identical splits; do NOT overwrite raw audit figure; spot-check ~30 cleaned clips before paper claim
