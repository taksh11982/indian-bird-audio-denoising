# P2 audit script + BirdNET weights — 2026-10-01

- **Prompt:** continue P2 macro audit
- **Phase:** P2
- **Changed:** `src/audit.py` (new — V2.4 CPU, 3.0 s windows/overlap 0.0,
  max-pool → S^53 .npz per chunk, resume-safe; triage 0.20/0.65/0.50 +
  unlabeled/ambiguous); BirdNET V2.4 weights fetched (224 MB, site-packages).
  Full `python src/audit.py` launched (background).
- **Specs touched:** BirdNET V2.4 CPU, 3 s windows, max-pool, thr 0.20/0.65/0.50.
  Overlap 0.0 = BirdNET default (spec pins window only). 5-name SYNONYMS map
  (epithet+common-name matched); 5 folders unmapped (Argya longirostris,
  Cyornis magnirostris, Sphenocichla humei, Chloropsis, Mystery) → unlabeled.
- **Gold-set contact:** none
- **Tests:** smoke on 3 chunks → 2 windows/chunk, S∈[0,1]^53, sane argmax
- **For next agent:** when audit run completes: verify audit_triage.csv,
  build figures/noise_summary (Deliverable 1), commit + push.
