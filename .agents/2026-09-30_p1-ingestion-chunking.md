# P1 ingestion + dual-stream chunking — 2026-09-30

- **Prompt:** inspect data/raw/ibc53, run P1 pipeline, unblock ponytail
- **Phase:** P1
- **Changed:** `src/ingest.py` (new — manifest builder); `src/chunk.py` (new — 48k/16k
  chunker, fixed mid-run stream-mismatch bug); `.gitignore` (new — ignores data/raw,
  chunk/score/xc dirs); global `opencode.json` (removed dead ponytail plugin entry);
  `data/raw_manifest.csv` + `data/chunk_index.csv` (generated, committable)
- **Specs touched:** 48/16 kHz mono, 5.0 s non-overlap, tail <2.0 s dropped,
  tails [2.0,5.0) zero-padded + flagged (`padded_s`), IDs `<sp>_<rec>_c003`
- **Gold-set contact:** none (no gold set exists yet)
- **Tests:** P1 DoD sweep → 9289+9289 chunks all exactly 5.000 s, IDs unique,
  index↔dirs match, min real audio in padded chunk 2.000 s; 8-file header spot-check
- **For next agent:** P1 DONE. P2 next needs `pip install birdnet_analyzer` (Py ≥3.11)
  + `src/audit.py` (3.0 s windows, max-pool, thr 0.20/0.65/0.50). Note: only 1368
  files (not ~5000); `Mystery mystery/` (443 files) needs a label-free audit path.
