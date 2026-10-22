# P4.5 separation audit: 513 chunks rated, merge deferred — 2026-10-22

- **Prompt:** "rate the 523 chunks with your own tools; dominant bird vs folder label; no human rating"
- **Phase:** P4.5 separation audit (Bird-MixIT + BirdNET rescoring; additive, no spec changes)
- **Changed:** `src/separate_audit.py` (new — Bird-MixIT 4-source separation of each 5.0 s chunk,
  per-channel BirdNET V2.4 scoring (audit.py setup reused), dominant = max conf across the 4
  channels (Denton et al. 2022 recipe), verdict correct/incorrect/no_dominant vs folder label,
  target_best_conf rescue signal, resume-safe separate/score/report stages, --shard/--nshards/--threads)
- **Changed:** `.gitignore` — `data/separated/` (2,052 channel wavs, git-ignored by design)
- **Changed:** `results/separation_523.shard{0..3}of4.csv` — NEW rating of record: 129+128+128+128 = 513 rows
  (523 intruder+ambiguous − 10 gold). Stale smoke artifacts (`separation_523.csv` 3 rows,
  `_summary.json`, `figures/separation_verdicts.*`) deleted — user deferred canonical merge
- **Specs touched:** frozen thr 0.20/0.65/0.50 unchanged; NEW frozen rule: dominant = max BirdNET conf
  over separated channels, no_dominant if < 0.20; sources saved at 22.05 kHz (port native); seed 42 set
- **Deps (user-approved):** torch 2.14.1+cpu, bird-mixit-pytorch-1.0.0 (+safetensors, huggingface_hub); soxr already present
- **Gold-set contact:** none — 10 gold chunk_ids among the 523 excluded FIRST; gold CSV read only for that exclusion
- **Tests:** selection 513 asserted; sum-to-mix assert on every chunk; probe: BirdNET scores 22.05 kHz
  source (P. ruficeps 0.9803, 2 windows); shard scored totals 129/128/128/128 = 513; 3 bugs fixed en route:
  sf.write got a 2-D tensor view (libsndfile "Format not recognised"), torch 2.14 dropped
  set_interop_threads (line removed), target column used BirdNET-label-space instead of 53-order
  space (→ order.index(given); its tainted partial rows 3/3/2 wiped + rescored clean)
- **For next agent:** user deferred the merge — when asked, run
  `python src/separate_audit.py --stage report` (merges shard CSVs -> canonical sorted deduped
  `results/separation_523.csv`, `_summary.json`, `figures/separation_verdicts.png/pdf`, deletes shard
  temp files), then paper integration as an ADDITIVE third figure; never touch noise_summary pair.
