# Colab LSTM results land in repo — 2026-10-10

- **Prompt:** extract drive zip, absorb 7 Colab files, next steps + progress
- **Phase:** P4-Gavali results (no code changes this turn)
- **Changed:** `results/gavali_ablation.csv` etc. — canonical files now LSTM (A acc 0.4402 F1 0.3872; C acc 0.4530 F1 0.3968; DELTA F1 +0.0096); sklearn-full preserved as `*_sklearn.*`
- **Changed:** `results/gavali_lstm_A.pt` + `_C.pt` — new checkpoints (355 KB each)
- **Changed:** `figures/confusion_gavali_A/C.png` — new LSTM matrices (render-checked C)
- **Specs touched:** none (splits/seed/thresholds unchanged; same 234 raw test)
- **Gold-set contact:** none (splits already gold-free)
- **Tests:** zip held all 7 files; ablation matches Colab printout; per-species recall computed (gains Pellorneum +0.14 n=70, Pnoepyga +0.23; losses Arborophila −0.46 n=13, Glaucidium −0.22 — small-n swings, net positive)
- **For next agent:** paper Table 1 + Fig.3 ready from these files; open item = write paper draft (no more runs needed unless user wants --epochs 50)
