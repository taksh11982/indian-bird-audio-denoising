# Best-outcome playbook (suggestion.md) — 2026-10-09

- **Prompt:** think hard and write suggestion.md with everything for best paper outcomes
- **Phase:** P5 planning (no training, no audit rerun)
- **Changed:** `suggestion.md:1` — new: 11-species Gavali A-vs-C plan, before/after figures, 30-clip spot-check replacing 150-gold, exact triage/scope numbers, paper+venue+risk checklists
- **Changed:** `suggestion.md:80` — fix: 11-species cleaned pool 2366 (was 2160 typo), verified from cleaned CSV
- **Specs touched:** none (thr 0.20-0.65-0.50, 5.0 s, seed 42 all unchanged; no new dep)
- **Gold-set contact:** none (gold CSV counted only: 150 rows, 0 graded; never used for tuning/training)
- **Tests:** recomputed triage 9289 = 3638/3553/1575/501/22 OK; manifest 1368 OK; cleaned-11 sum 2366 OK
- **For next agent:** needs Gavali input spec (sr/clip/feature) to write `src/gavali_features.py`; then `src/gavali_compare.py` per suggestion §4
