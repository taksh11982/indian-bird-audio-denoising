# Evidence pack + defence brief (teacher says "you only cleaned data, +1 %") — 2026-10-10

- **Prompt:** teacher claims the work is only "cleaned the data, improved 1 %"; asked how to convince her with evidence and what to do if she agrees
- **Phase:** P4 evidence upgrade / P5 report
- **Changed:** `src/evidence_label_noise.py` (new) — measures label noise in every pool; output `results/label_noise_evidence.csv` + `.md`
- **Key numbers (measured):** A-train 2,563 chunks, 726 (28.3 %) unusable labels; C-train 1,837, 0 % unusable; val 412, 37.1 % unusable; shared test 234, 62 (26.5 %) unusable → 172 audited-trusted; Glaucidium cuculoides test n=9 with 55.6 % broken labels
- **Changed:** `src/noise_scaling.py` (new) — trains the identical LSTM on the cleaned pool with 0/5/10/20/28.3 % injected label noise, scores on raw AND trusted test → `results/noise_scaling.csv`; converts the +0.0096 anecdote into a curve
- **Changed:** `src/eval_trusted_test.py` (new) — re-scores prediction dumps on all/trusted/untrusted subsets → `results/clean_test_eval.csv`
- **Changed:** `src/gavali_compare.py` — added `dump_preds()` only (LSTM backend now writes `results/preds_lstm_A.csv` / `_C.csv`); training path untouched
- **Changed:** `paper/teacher_brief.md` (new) — handover brief: task statement, 7 deliverables, measured noise, 3 reasons the delta is small, self-flagged limitations, 5 next steps with costs, ask-list
- **Changed:** `GAVALI_RUNBOOK.md` — new section 4b (evidence-upgrade runbook, Colab ~45 min)
- **Changed:** `Report Format - Project2- Sem7 .docx` — inserted the label-noise evidence paragraph + table into §4 and renumbered tables (now Table 1 summary, Table 2 label-noise evidence, Table 3 ablation, Table 4 per-species); patch script kept OUTSIDE the repo (temp), template not otherwise modified
- **Specs touched:** none (thresholds 0.20/0.65/0.50, chunk 5.0 s, seed 42, splits unchanged)
- **Gold-set contact:** none (gold excluded before splitting, asserted in every script; `gold_test_150.csv` only named in prose)
- **Tests:** `python -m py_compile` on 4 scripts OK; `src/evidence_label_noise.py` ran and reproduced the numbers above; docx re-parsed → 336 body paragraphs, 4 tables, captions numbered 1→4 in document order, no placeholders left except the personal CV line
- **For next agent:** run on Colab T4 — `python src/gavali_compare.py --backend lstm --epochs 30 --device cuda` then `python src/noise_scaling.py --epochs 30 --device cuda --dump-preds` then `python src/eval_trusted_test.py`; then fold the trusted-subset numbers + noise curve into the report and the paper; open manual item = 30-clip spot check (`results/spotcheck_30.csv`)
