# Report rebuilt with figures, PDF re-exported — 2026-10-22

- **Prompt:** "do the changes in the PDF file according to the Word file" (after the docx would not open).
- **Phase:** P5 (paper + artifact)
- **Changed:**
  - `src/build_report.py` — fixed an XML bug that emitted `</w:body>` twice (Word reported a parse error at line 2, column 260553); captions are now real ` SEQ Figure/Table \* ARABIC ` fields; the List of Tables / List of Figures TOC fields were retargeted to `\c "Table"` / `\c "Figure"`; the final section got an explicit empty default header (it was inheriting "3.Review of Literature"); in-text references added for Tables 1, 2, 5, 6 and Figures 2, 3, 5; the CV placeholder was replaced.
  - `src/export_report_pdf.py` (new) — opens the docx in Word, updates TOC/fields, saves the docx and exports the PDF; copies it over `Indian bird audio denoising report.pdf`.
  - `src/repair_report_root.py` (new) — restores the template's full `<w:document>` root (the earlier corruption symptom).
  - `paper/teacher_brief.md` — sections 10–11: per-class, mechanism and power analyses, report/PDF state.
- **Specs touched:** none (5.0 s chunks, triage thresholds, seed 42, 39-MFCC 157-frame features unchanged).
- **Gold-set contact:** none
- **Tests:** `python src/build_report.py ...` → rebuilt; `python src/export_report_pdf.py ...` → 35 pages, 6,281 words, 13 inline shapes, 8 tables; pypdf re-parse → figures 1–11 and tables 1–8 captioned and referenced, TOC/LoT/LoF populated with page numbers, 14 images embedded. Word opens the docx without repair.
- **For next agent:** control arm R, noise-scaling curve and fig8 still need the Colab T4 run (audio chunks are not on this machine). `Report Format - Project2- Sem7.pdf` is a generated duplicate of the deliverable PDF and is not committed.
