# indian-bird-audio-denoising

Cleaning Crowdsourced Labels: Unsupervised Outlier Detection to Denoise Xeno-Canto Training Sets for Indian Bird Classification.

## Scope implemented

This repository now contains a minimal, CPU-friendly implementation scaffold for the recommended data-cleaning study:

1. choose a 15-20 species Indian subset (BirdCLEF-2024 / iBC53),
2. compute mel-spectrogram inputs,
3. train a convolutional autoencoder and score reconstruction errors,
4. prune species-wise outliers using robust MAD z-scores,
5. train/evaluate a small CNN before and after pruning,
6. report macro-mAP gain and per-species reconstruction-error variance.

Core code lives in `/home/runner/work/indian-bird-audio-denoising/indian-bird-audio-denoising/uod_pipeline.py`.

## Notes

- Species-aware pruning and mAP computation are dependency-free.
- Conv-AE and small-CNN training helpers are provided with lazy PyTorch imports (`RuntimeError` with install guidance if PyTorch is absent).
- This design keeps the denoising logic reproducible on CPU-only setups while allowing model training when dependencies are installed.

## Running tests

```bash
cd /home/runner/work/indian-bird-audio-denoising/indian-bird-audio-denoising
python -m unittest discover -s tests -p 'test_*.py'
```
