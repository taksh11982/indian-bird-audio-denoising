# README rewrite: full project reference — 2026-10-22

- **Prompt:** build a perfect README encapsulating everything; use all .agents files; no mistakes; must support report building
- **Phase:** P5 support (docs; no model/threshold/data changes)
- **Changed:** `README.md` — replaced 2-line stub with full reference: pipeline (mermaid),
  triage table w/ verified counts, scope decisions, split tables, LSTM architecture
  (LSTM(39→128)→Dropout0.2→Linear128→11, RMSprop 1e-3, batch 64, 30 epochs), headline
  ablation + per-species recall table (recomputed from confusion CSVs at write time),
  P4.5 separation audit results (217 correct / 62 incorrect / 234 no_dominant, from
  shard CSVs), figure inventory incl. pending Fig.4 + three verified Fig.2 exemplar
  chunks, honest limitations, reproduction commands + QC checks, repo tree (matches
  disk: no paper/ or notebooks/ dirs), related-work citation list, changelog rules,
  status/next steps
- **Changed:** every number in §2-§6 verified this prompt from `audit_triage.csv`,
  `splits_summary.json`, `gavali_ablation*.csv`, confusion CSVs, shard CSVs, git log —
  not quoted from memory
- **Specs touched:** none (all frozen specs restated, none altered)
- **Gold-set contact:** none (README only restates its locked/unused status)
- **Tests:** verification script over all cited artifacts; table header nit fixed post-write
- **For next agent:** P5 paper draft can cite README §6/§7/§8 directly; canonical
  separation merge still deferred (one command, see README §6.3)
