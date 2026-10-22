# Fill Project-2 report into the Word template (in place) — 2026-10-10

- **Prompt:** "you haven't done anything [in the] report format project2 word file ... undo those [codebase] changes and do report for our work done in report format"
- **Phase:** P5 (report)
- **Changed:** `Report Format - Project2- Sem7 .docx` — now the filled report (title/author/ack/abstract, §1–6, real Tables 1–3, Fig 1–2 captions). Written by copying the generated builder output over the template; a pristine copy of the blank template is kept outside the repo.
- **Reverted:** deleted `report/` (the stdlib `make_report.py` builder, `Project2_Report.md`, `Project2_Report_filled.docx`, `__pycache__`) — no builder script left in the repo
- **Changed:** `.agents/2026-10-10_report-template-fill.md` removed; this file supersedes it
- **Specs touched:** none (no data/thresholds/splits/seed changed)
- **Gold-set contact:** none (`gold_test_150.csv` named only in prose)
- **Tests:** re-parsed `Report Format - Project2- Sem7 .docx` → 334 body paragraphs / 344 children; all placeholders gone except personal "write your recent cv"
- **For next agent / user manual steps:** (1) fill Curriculum Vitae, (2) insert `figures/noise_summary.png`, `noise_summary_cleaned.png`, `confusion_gavali_A/C.png` at the caption slots, (3) rewrite the "Candidate Disclosure on AI Tools" example text
