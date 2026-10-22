# iBC53 Label-Noise Audit & Data Cleaning — what was done, what it proves, what is next

**Project-2 · Dharmsinh Desai University · October 2026**
Authors: Madhur S. Mundra (23ITUOZ066), Taksh N. Lade (23ITUOS063)
Guide: Prof. Nikita P. Desai

---

## 1. The task, stated precisely

The assigned task is **data cleaning**: *does cleaning the crowdsourced labels of
iBC53 produce a better acoustic classifier than training on the labels as
given?* It is **not** "build the most accurate bird classifier" — that question
was already answered in the literature (Gavali & Banu 2025, ~94% via
image+audio fusion). Our question is the data one, and the deliverable is a
**measurement of label noise in a public dataset** plus a **reproducible
cleaning pipeline** — not a leaderboard number.

## 2. What was built (7 concrete deliverables, all in the repo)

| # | Deliverable | Path | Status |
|---|---|---|---|
| 1 | Dual-stream ingestion + manifest (1,368 files, 53 species, 12.68 h) | `src/ingest.py`, `data/raw_manifest.csv` | done |
| 2 | Deterministic 5.0 s chunker (48 kHz + 16 kHz, tail policy, zero-pad accounting) | `src/chunking.py`, `data/chunk_index.csv` | done |
| 3 | BirdNET V2.4 CPU auditor, 3 s sliding windows → max-pooled vector, per-chunk resumable scoring | `src/audit.py`, `data/birdnet_scores/` | done |
| 4 | Taxonomy reconciliation (5 folder names remapped to Clements-2024 BirdNET labels; 5 taxa with no label, reported not hidden) | `src/audit.py` | done |
| 5 | Frozen-threshold triage + **cleaned dataset** (keep only audited-concordant chunks) | `src/clean.py`, `data/iBC53-Cleaned-Metadata.csv` | done |
| 6 | Leakage-free A-vs-C experiment (recording-level splits, seed 42, gold excluded) + confusion matrices | `src/gavali_*.py`, `results/` | done |
| 7 | Evidence pack (train/test label-noise measurement) + human spot-check harness | `src/evidence_label_noise.py`, `src/spotcheck.py` | done |

## 3. The measured label noise (the actual finding)

BirdNET scored every 5 s chunk; the triage is a **measurement of the public
dataset**, and it is the first such measurement for iBC53.

| triage | chunks | share | meaning |
|---|---|---|---|
| concordant | 3,553 | 38.2% | target bird really present, confident |
| empty | 1,575 | 17.0% | no bird detected at all — label is meaningless |
| ambiguous | 501 | 5.4% | unsure |
| intruder | 22 | 0.2% | a **different** species is calling — label is wrong |
| unlabeled | 3,638 | 39.2% | taxon BirdNET has no label for (incl. `Mystery mystery`) |

Restricted to label-usable data, **27.9% of the chunks are empty and 0.4%
contain a different species than the folder says.** Per-species concordant
rate ranges from 37% (`Glaucidium cuculoides`) to 93% (`Pnoepyga pusilla`) —
i.e. the noise is not uniform, it is species-dependent, which is exactly why a
*model-centric* fix cannot remove it.

## 4. Why "+1 %" is not the story — three measured reasons

**Reason 1 — the training pool was 28 % wrong.** Model A trained on 2,563 raw
chunks, of which **726 (28.3 %) carry an unusable label** (empty / ambiguous /
different species). Model C trained on 1,837 chunks, 100 % audited.

**Reason 2 — the test set is also 26.5 % wrong.** The shared test set holds
234 chunks and **62 (26.5 %) of them are themselves mislabelled or empty**.
A model that is *right* about a silent clip is marked *wrong*. The reported
accuracy is therefore a lower bound and the metric is a floor, not a ceiling.

**Reason 3 — the test set is too small to resolve 1 %.** 1 % of Macro-F1 on
234 chunks is ≈ 2 chunks. Two of the 11 classes have 5 and 9 test chunks, and
40 % / 56 % of their chunks are broken — Macro-F1 is dominated by noise in a
handful of classes. The current protocol is **underpowered**, so "+1 %" is
reported as measured rather than dressed up.

## 5. The numbers we will stand behind

| model | train chunks | train label quality | test accuracy | test Macro-F1 |
|---|---|---|---|---|
| A (raw labels) | 2,563 | 71.7 % usable | 0.4402 | 0.3872 |
| C (cleaned labels) | 1,837 | 100 % usable | 0.4530 | 0.3968 |
| Δ (C − A) | −28 % data | +28.3 pts clean | +0.0128 | **+0.0096** |

Direction is correct and the data-efficiency claim is already there: **less
but cleaner data beats more but noisier data.**

## 6. Limitations we flag ourselves (before you ask)

1. No human-graded gold set: the 150-sample grading was dropped as
   infeasible; a **30-clip spot check is sampled and awaiting human grading**
   (`results/spotcheck_30.csv`) — this is our only manual dependency.
2. Absolute accuracy is far below Gavali et al.'s 94 %, because our evaluation
   is audio-only, on a **leakage-free recording-level split**, and scored
   against **audited** (partly wrong) labels. The two numbers are not
   comparable, and the 94 % cannot be reproduced without their split.
3. Five taxa have no BirdNET V2.4 label and are dropped from the cleaned
   set; nine species lose all their chunks. Reported, not hidden.
4. The gain is measured once, on one split, without cross-validation.

## 7. Next steps — converting "1 %" into a defensible result (scripts ready)

| # | Action | Why it matters | Cost | Script |
|---|---|---|---|---|
| 1 | **Clean-test evaluation** — rescore A and C on the 172 audited-trusted test chunks | separates "model right" from "label right"; removes the 26.5 % label-noise penalty from the metric | 1 Colab run (~15 min) | `src/gavali_compare.py` (now dumps per-chunk predictions) + `src/eval_trusted_test.py` |
| 2 | **Noise-scaling curve** — train the same LSTM on the clean pool with 0/5/10/20/28.3 % injected label noise | turns a single anecdote into a **curve**: shows the accuracy cost per % of noise, and that A's entire deficit is explained by the 28 % noise we removed | 1 Colab run (~30 min) | `src/noise_scaling.py` |
| 3 | **5-fold recording-level CV** for A and C | gives a mean ± std instead of one number; the only way to say "+1 %" with an error bar | 1 Colab run (~1 h) | reuse `src/gavali_splits.py` |
| 4 | **30-clip human spot check** | turns "BirdNET said so" into a *validated* auditor (precision of the cleaning rule) | ~30 min listening | `src/spotcheck.py` (sampled) |
| 5 | Fold 1–4 into the report + paper; release the cleaned dataset and loader | produces the P5 artefact | 1 day | — |

## 8. What we need from you

* **Approval + a Colab T4 slot** for steps 1–3 (≈ 2 h GPU total) — the
  scripts are written and ready.
* **30 minutes of listening** for step 4 (2 clips × 11 species, 5 s each).
* Confirmation that the project is graded on **data quality measurement +
  reproducible pipeline** (our reading of the task) rather than on a headline
  accuracy number.

## 9. Reproducing everything (seed 42 throughout)

```bash
python src/ingest.py
python src/chunking.py
python src/audit.py                 # BirdNET V2.4, CPU, resumable
python src/clean.py                 # -> data/iBC53-Cleaned-Metadata.csv
python src/gavali_splits.py         # recording-level 80/10/10
python src/gavali_compare.py --backend lstm --epochs 30 --device cuda
python src/evidence_label_noise.py # -> results/label_noise_evidence.md
python src/noise_scaling.py --epochs 30 --device cuda
python src/eval_trusted_test.py     # -> results/clean_test_eval.csv
```
