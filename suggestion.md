# iBC53 suggestion.md — best-outcome playbook (no custom CNN, no 150-gold)

Date: 2026-10-09. Ground truth: `data/` + `src/` as committed. All numbers below
were recomputed from those files. Bring your Gavali et al. paper/code link —
its input shape is the one unknown this doc does not guess.

## 0. TL;DR — the highest-value path for a non-professional author

1. Keep the noisy `figures/noise_summary.*` as Fig.1a (BEFORE). Keep the new
   `figures/noise_summary_cleaned.*` as Fig.1b (AFTER, 100% green by filtering).
   Never overwrite 1a to look green.
2. Drop the custom 3-Conv CNN + 150 Audacity gold. Reuse Gavali et al. as a
   FIXED baseline, train it twice (raw vs cleaned) with identical splits/seed,
   report the delta. That is a valid data-centric claim and 5x less work.
3. Shrink scope from 53 to 11 species (focus-12 minus Sphenocichla humei, see
   §3). Everything — splits, figures, tables, loaders — targets those 11.
4. Replace 150-gold with: recording-level holdout + 30-clip spot-check (§5).
5. Paper claim: "unsupervised BirdNET triage removes 37% label noise and
   improves Gavali et al. by +X% Macro-F1 on a held-out set" — label cleaning,
   NOT waveform enhancement. That wording is what makes reviewers accept it.

## 1. What you actually have (do not misstate these)

- Raw: 1368 files, 53 species, 12.68 h, mean 33.4 s/file, range 1.0–362.8 s
  (`data/raw_manifest.csv`). Most skewed taxon: `Mystery mystery` = 443 files.
- Chunks: 9289 × 5.0 s, dual stream 48 kHz / 16 kHz, 849 padded (9.1%)
  (`data/chunk_index.csv`). Chunking spec intact.
- BirdNET V2.4 audit, 3.0 s windows max-pooled, thr 0.20 / 0.65 / 0.50
  (`src/audit.py:30-32`, `data/audit_triage.csv`):
  unlabeled 3638 (39.2%) | concordant 3553 (38.2%) | empty 1575 (17.0%)
  | ambiguous 501 (5.4%) | intruder 22 (0.2%).
- Labeled-only (5651 chunks, Mystery + 4 no-label taxa excluded): 62.9%
  concordant, 27.9% empty, 8.9% ambiguous, 0.4% intruder. Quote THIS when
  reviewers ask "how noisy is the label-usable data" — not the 39% unlabeled.
- Unlabeled is 5 taxa, all with no BirdNET V2.4 label: `Mystery mystery` (3033),
  `Sphenocichla humei` (251), `Chloropsis` genus folder (196),
  `Argya longirostris` (81), `Cyornis magnirostris` (77). `Mystery` alone is
  32.6% of all chunks — it must be excluded from training, explicitly.
- Cleaned (`src/clean.py:1`, `data/iBC53-Cleaned-Metadata.csv`): 3553 kept
  (38.2%), 44 species. Top: `Pellorneum ruficeps` 535, `Cuculus micropterus`
  373. Tail: `Riparia chinensis` 3, `Chrysococcyx maculatus` 4. Imbalance
  535:3 — you MUST stratify splits and use macro averaging.
- 9 species vanish after cleaning (0 kept): the 5 unlabeled above plus 4
  tiny all-empty taxa (`Iole cacharensis` 22, `Ixobrychus cinnamomeus` 13,
  `Chloropsis cochinchinensis` 10, `Acrocephalus bistrigiceps` 1). Name them
  in the paper limitations — hiding them is the fastest way to get rejected.
- Focus-12 (`src/sample_gold.py:22-35`) per-species concordant rates:
  `Pnoepyga pusilla` 197/211 (93%), `Cuculus micropterus` 373/403 (93%),
  `Arborophila torqueola` 284/347 (82%), `Pomatorhinus ruficollis` 263/335
  (79%), `Rimator malacoptilus` 135/181 (75%), `Psilopogon lineatus` 111/159
  (70%), `Pellorneum ruficeps` 535/811 (66%), `Liocichla phoenicea` 130/201
  (65%), `Cyornis unicolor` 175/276 (63%), `Psittacula eupatria` 60/147 (41%),
  `Glaucidium cuculoides` 103/276 (37%), `Sphenocichla humei` 0/251 (0%,
  unlabeled). The two low ones are your best "cleaning matters" stories.
- Gold (`data/gold_test_150.csv`): 150 sampled (13×6 + 12×6), 0 graded. It
  contains 12 `Sphenocichla humei` chunks that BirdNET can never confirm —
  another reason to drop that species and not use this gold for Gavali eval.
- XC refs (`data/xc_refs/`): 12 dirs present but names drift
  (`Napothera malacoptila` = `Rimator malacoptilus` synonym,
  `Stachyris humei` = `Sphenocichla humei` old name). Normalise before citing.
- Concrete demo pair from the SAME recording (use for spectrogram Fig.2):
  `Glaucidium cuculoides_1_c000` concordant 0.9682 vs
  `Glaucidium cuculoides_1_c002` empty 0.1363. Intruder examples:
  `Dicrurus andamanensis_18_c001` → `Cuculus micropterus` 0.9392,
  `Phylloscopus forresti_5_c001` → `Phylloscopus inornatus` 0.9222.
  Pristine anchor: `Chelidorhynx hypoxanthus_9_c001` 1.0000.

## 2. What "green" honestly means (and the one forbidden shortcut)

- Allowed: Fig.1b is green BECAUSE you deleted everything not concordant.
  Caption must say "concordant-only subset (n=3553), 100% green by
  construction; 5736 chunks removed: empty 1575 + intruder 22 + ambiguous
  501 + unlabeled 3638". Reviewers accept this — it is standard before/after.
- Forbidden: editing `audit_triage.csv`, thresholds, or `noise_summary.png`
  to inflate green. Scores in `data/birdnet_scores/` + frozen
  `src/audit.py:30-32` would contradict you. That is misconduct, not cleaning.
- Keep both files forever: `noise_summary.*` = measurement of raw noise,
  `noise_summary_cleaned.*` = proof of filtering. Your paper needs the pair.

## 3. Scope decision — go 11, not 53, not 12, not 44

- 53 is unpublishable for you: 5 taxa unratable + 4 tiny all-empty taxa +
  535:3 imbalance + Mystery dominating compute.
- 44 (all cleaned) is still too imbalanced and too much Gavali training time.
- 12 includes `Sphenocichla humei`, which contributes 0 trainable chunks
  after cleaning yet 12 gold rows — dead weight that confuses every table.
- RECOMMENDED: 11-species focus = focus-12 minus `Sphenocichla humei`.
  Trainable cleaned pool ≈ 2366 chunks (3553 minus Sphenocichla 0 minus
  non-focus species). Largest class `Pellorneum ruficeps` 535, smallest
  `Psittacula eupatria` 60 — still imbalanced but manageable with stratified
  splits + weighted loss. State the exclusion in one sentence with the reason
  (no BirdNET label → unauditable → excluded from supervised comparison).

## 4. Gavali et al. comparison — exact protocol (copy into Methods)

1. Get Gavali input spec FIRST (sample rate, clip length, feature: waveform /
   mel / MFCC, label set). If it is not 16 kHz × 5.0 s × Log-Mel 128×157,
   write ONE adapter (`src/gavali_features.py`) that reads
   `data/iBC53-Cleaned-Metadata.csv:1` paths and emits Gavali-ready tensors.
   Do not re-chunk.
2. Two runs only: Model A = Gavali trained on RAW-labeled 11-species chunks;
   Model C = Gavali trained on CLEANED (concordant-only) 11-species chunks.
   Skip Model B (pruned vs corrected needs manual labels you will not do).
3. Identical everything except data: same recording-level stratified
   80/10/10 split (train/val/test), same seed 42 (`random`, `numpy`, `torch`,
   sklearn `random_state`), same epochs/batch/lr/augmentation, same test IDs.
   Split by `recording_id`, never by chunk — same-recording chunks in both
   train and test is leakage and reviewers check this.
4. Test set = held-out RAW recordings (not cleaned-only). Report Macro-F1 +
   accuracy on that fixed test for A and C. Expected story: C beats A because
   C trained on less-but-cleaner data. If C loses, report it — negative
   result with honest audit still passes workshops; hidden split tricks do not.
5. Assert no `gold_test_150.csv` chunk appears in any Gavali split (one-line
   check, even though you are not grading gold — it proves no peeking).
6. Log everything to `results/gavali_ablation.csv` (columns: run, train_n,
   test_n, macro_f1, accuracy, seed, commit). One row per run, no hand-edited
   numbers.

## 5. Evaluation without 150-gold — 3 tiers that replace it

- Tier 1 (automatic, required): fixed held-out test (§4.3–4.4). This carries
  the paper.
- Tier 2 (manual, cheap, required): 30-clip spot-check, not 150. Randomly
  sample 30 cleaned chunks (seed 42, ~3 per species across the 11), listen +
  view spectrogram in Audacity vs `data/xc_refs/`, log
  `results/spotcheck_30.csv` (chunk_id, keep?, notes). One evening of work,
  and it lets you write "30/30 spot-checked clips contained the target"
  instead of an ungraded-gold bluff.
- Tier 3 (proxy, optional): noise-rate table — % empty/ambiguous removed per
  species (worst: `Glaucidium cuculoides` 53% empty, `Psittacula eupatria`
  52% empty; cleanest: `Pnoepyga pusilla`, `Cuculus micropterus` >92%
  concordant). Reviewers like this because it explains WHERE cleaning helps.

## 6. Figures + tables — produce exactly these, no more

- Fig.1a `noise_summary.png` (existing): BEFORE, per-species stacked bars.
- Fig.1b `noise_summary_cleaned.png` (existing): AFTER, all green.
- Fig.2 spectrogram panel (new, 3 columns): clean
  (`Glaucidium cuculoides_1_c000`) vs empty tail
  (`Glaucidium cuculoides_1_c002`) vs intruder
  (`Dicrurus andamanensis_18_c001`) each with XC Grade-A ref from `xc_refs/`.
  Generate from 16 kHz chunks, identical dB scale, seed 42 irrelevant here.
- Fig.3 confusion matrices (new): Gavali-A vs Gavali-C on the SAME test set.
- Table 1 ablation (new, `results/gavali_ablation.*`): train_n, test_n,
  Macro-F1, accuracy for A vs C + delta. No C−A ∈ [4,7]% promise — report
  whatever the fixed protocol yields; forcing a range is how papers die.
- Table 2 per-species noise removed (new): n_raw, n_kept, % removed for the
  11 species. Reuse §1 numbers.

## 7. Paper skeleton — IEEEtran 5-page, fill in this order

1. Intro (0.6 p): crowdsourced bird labels are noisy; data cleaning beats
   blind training; contribution = unsupervised BirdNET audit + cleaned
   artifact + Gavali delta on Indian birds.
2. Related (0.7 p): Gavali et al. as model-centric baseline (cite + state
   what you REUSE vs CHANGE); one paragraph on BirdNET-as-auditor prior work;
   one sentence: this is LABEL denoising, not waveform enhancement — mixing
   those terms triggers rejection.
3. Acoustic Audit Pipeline (1.2 p): dual streams, 5.0 s rule, BirdNET V2.4
   3.0 s max-pool, frozen thresholds, triage definitions, Fig.1a/1b, §1 stats,
   9-vanished-species limitation, synonym handling (`src/audit.py:60-66`).
4. Experiments (1.5 p): §4 protocol verbatim (splits, seed, adapter), Tier
   1–2 results, Fig.3, Tables 1–2. Disclose: test is from raw recordings;
   BirdNET filtered train only (train-test auditor asymmetry is a feature,
   and say why).
5. Conclusion + Open Science (0.5 p + refs): artifact
   `data/iBC53-Cleaned-Metadata.csv` + loader + commit hash; future work =
   second auditor for the 5 unlabeled taxa + manual correction of ambiguous
   501. Venue: ICIIS / ICECSP / ICSC / SPICES; fallback LNNS / ICICC (§8).

## 8. Reproducibility + venue checklist (reviewers score this)

- [ ] `src/gavali_features.py` adapter + `src/gavali_compare.py` runner,
  both seed-42, both asserting zero gold overlap and recording-level splits.
- [ ] `data/iBC53-Cleaned-Metadata.csv` + minimal loader
  (`src/ibc53_loader.py`, ~40 lines, torch Dataset reading `path_16k`) on
  GitHub with commit hash cited in paper.
- [ ] `results/` contains ONLY machine-written CSVs + figure PDFs, never
  hand-edited numbers. Keep raw `chunks_*/`, `birdnet_scores/`, `xc_refs/`
  git-ignored per `.gitignore` (CSVs stay committable).
- [ ] `.agents/` changelog per prompt (≤30 lines, existing convention).
- [ ] Guide sign-off + registration fee budget before submission; expect
  4–8 week review; workshops accept negative deltas, journals do not — submit
  the honest delta wherever it lands.

## 9. Risks — what kills this paper and the fix

1. Circular eval (train-clean + test-clean with same BirdNET filter inflates
   C). Fix: §4.4 raw held-out test.
2. Chunk leakage (same recording in train+test). Fix: split by recording_id.
3. Mystery/synonym sloppiness. Fix: name the 5 unlabeled taxa + 4 tiny taxa;
   normalise `Napothera/Stachyris` folder names before release.
4. Overclaiming "denoising". Fix: always write "label cleaning / triage";
   never imply waveform SNR enhancement you did not do.
5. Threshold tuning to hit a target delta. Fix: thresholds stay
   `src/audit.py:30-32`; any change needs explicit approval + new audit run.

## 10. Next 10 commands (in order, smallest diffs first)

1. Normalise `xc_refs/` folder names to `audit.py` synonym keys.
2. Add `src/gavali_features.py` (adapter; needs Gavali input spec from you).
3. Add `src/gavali_compare.py` (recording-level split, seed 42, A-vs-C).
4. Add `src/ibc53_loader.py` (torch Dataset over cleaned CSV).
5. Add `src/spectrogram_panel.py` (Fig.2 from §1 chunk IDs + XC refs).
6. Run 30-clip spot-check → `results/spotcheck_30.csv`.
7. Run Gavali A vs C → `results/gavali_ablation.csv` + Fig.3.
8. Freeze `results/` + `data/iBC53-Cleaned-Metadata.csv` with git tag.
9. Draft paper in Overleaf following §7; paste Tables 1–2 unedited.
10. Guide sign-off → submit (§8 venue order).
