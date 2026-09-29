# agents.md — iBC53 Data-Centric Denoising (Project-2)

Mission: prove **data cleaning beats blind training** on `arghyasahoo/ibc53-indian-bird-call-dataset`
(~5,000 files, 53 species). Target result: Model C beats Model A by **4–7% Macro-F1** on a locked
150-sample gold set. Every number below is load-bearing — do not "improve" thresholds, rates, or shapes.

## 1. Frozen specs (never change without explicit user approval)

| Item | Value |
|---|---|
| Stream A (Auditor) | 48 kHz mono (BirdNET native) |
| Stream B (Downstream CNN) | 16 kHz mono (0–8 kHz Nyquist) |
| Chunking | non-overlapping 5.0 s; discard residual tail < 2.0 s |
| Chunk ID | `<species>_<recording_id>_c<chunk_idx>` (deterministic, zero-padded idx: `c003`) |
| BirdNET | BirdNET-Analyzer **V2.4**, CPU, sliding **3.0 s** windows, **max-pool** windows → `S ∈ [0,1]^53` |
| Triage: Empty / ambient | `max(S) < 0.20` |
| Triage: Intruder / co-occurrence | `argmax(S) ≠ given label AND max(S) ≥ 0.65` |
| Triage: Concordant / target present | `argmax(S) == given label AND max(S) ≥ 0.50` |
| Micro focus | 12 most frequent, genera-distinct species; 2–3 Xeno-canto Grade A refs each |
| Gold set | **150 chunks** across the 12 species, graded A/B/C in Audacity vs XC ref; **locked — no training use, ever** |
| Features | Log-Mel **128 × 157** (`n_mels=128`, `n_fft=1024`, `hop_length=512`) from 16 kHz chunks |
| CNN | custom 3-Conv baseline, Colab T4, **seed=42** everywhere; A=raw, B=pruned, C=pruned+corrected |

Grade definitions: A = pristine target; B = target present, moderate SNR; C = heavy interference / empty.

## 2. Phase plan with definitions of done

- **P1 Ingestion (Days 1–2).** `kaggle datasets download -d arghyasahoo/ibc53-indian-bird-call-dataset`.
  Parse dir paths → `data/raw_manifest.csv` (columns: `filepath,species_name,recording_id,duration_s`).
  Build both streams + chunk. DoD: every chunk is exactly 5.0 s (±10 ms), IDs unique, tails <2.0 s gone.
- **P2 Macro audit (Days 3–4).** Run BirdNET V2.4 CPU over 48 kHz chunks, store per-window confidences,
  max-pool to one vector/chunk → `data/birdnet_scores/`, triage → `data/audit_triage.csv`
  (columns: `chunk_id,given_label,pred_label,max_conf,triage`). DoD: **Deliverable 1** —
  `% corrupt vs pure per species × 53` figure (`figures/noise_summary.*`).
- **P3 Micro focus + gold (Days 5–7).** Pick 12 species, download XC refs (`data/xc_refs/`), randomly sample
  150 chunks (`data/gold_test_150.csv`, columns: `chunk_id,species,grade,notes`), grade A/B/C in Audacity.
  DoD: gold CSV committed and **never modified after lock**; assert in every training script that no gold
  `chunk_id` appears in train data.
- **P4 Proof-of-gain (Days 8–10).** Log-Mel 128×157 from 16 kHz; train A/B/C with identical seed=42;
  eval Macro-F1 on gold. DoD: **Deliverable 2** — ablation table (`results/ablation.*`), C − A ∈ [4,7]%.
- **P5 Paper + artifact (Days 11–14).** IEEEtran 5-page (Overleaf): Intro / Related (Gavali et al. 2025 as
  model-centric baseline) / Acoustic Audit Pipeline / Experiments & Findings / Conclusion & Open Science.
  Figures: false-label vs XC-Grade-A spectrogram panel; Model A vs C confusion matrices.
  DoD: `data/iBC53-Cleaned-Metadata.csv` + loaders on GitHub.
- **P6 Submission.** Guide sign-off → IEEE India regional (ICIIS / ICECSP / early-2027 ICSC / SPICES);
  fallback Springer LNNS / ICICC. Expect 4–8 wk review.

## 3. Repo layout (create only what the current phase needs)

```
data/{raw_manifest.csv,chunks_48k/,chunks_16k/,birdnet_scores/,audit_triage.csv,xc_refs/,gold_test_150.csv,iBC53-Cleaned-Metadata.csv}
src/{ingest.py,chunk.py,audit.py,features.py,train.py,evaluate.py}
notebooks/  figures/  results/  paper/  .agents/
```

## 4. Golden rules for every agent working here

1. Seed 42 in every stochastic op (`random`, `numpy`, `torch`, sklearn `random_state`).
2. Never train on, tune thresholds on, or "peek" at `gold_test_150.csv` — eval only.
3. Smallest diff that satisfies the phase DoD; stdlib over new deps (`librosa`/`soundfile` already cover audio).
4. No new dependency without asking the user first (Colab T4 / review reproducibility matter).
5. Mark deliberate shortcuts with `# ponytail: <ceiling>, <upgrade path>`.

## 5. Mandatory per-prompt changelog (no exceptions)

After **every** user prompt that changes the repo (code, data, paper, config), write a new file in `.agents/`
before finishing your reply:

- Name: `.agents/YYYY-MM-DD_<short-slug>.md` (e.g. `.agents/2026-09-30_ingest-script.md`).
- Copy `.agents/_TEMPLATE.md` and fill all sections (keep it ≤ 30 lines).
- One prompt = one file. Never append to another prompt's file, never skip — even for one-line fixes.

## 6. Packages (pip) — install only for the active phase

```
pip install kaggle librosa soundfile numpy pandas scikit-learn torch torchaudio matplotlib seaborn
pip install birdnet_analyzer   # P2, Python ≥3.11, CPU-only inference
```

No MCP server is required for this project. Do not add one. Human-side prerequisites the user handles
(see §7): Kaggle API key, Audacity grading, Colab T4 runs, Overleaf paper, GitHub release.

## 7. Things to ask the HUMAN to do (agents: surface, don't stall)

1. `kaggle.json` API key placed (`kaggle datasets download` fails without it).
2. P3 Audacity grading of the 150 gold samples (irreducibly manual — sonogram judgement).
3. P4 Colab T4 training runs (local GPU not assumed).
4. Overleaf IEEEtran draft + guide sign-off; registration fee on acceptance.
