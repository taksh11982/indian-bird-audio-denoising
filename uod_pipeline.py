"""Label-noise pruning pipeline for Indian bird audio classification.

This module provides a lightweight implementation of the Collins-style
unsupervised outlier detection (UOD) workflow:
1) mel-spectrogram features
2) convolutional autoencoder reconstruction errors
3) species-aware outlier pruning
4) before/after classifier mAP comparison

The core pruning/evaluation helpers are dependency-free. Optional training
helpers for autoencoder/CNN rely on PyTorch and are imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Dict, Iterable, List, Mapping, Sequence


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    species: str


def group_indices_by_species(labels: Sequence[str]) -> Dict[str, List[int]]:
    grouped: Dict[str, List[int]] = {}
    for idx, label in enumerate(labels):
        grouped.setdefault(label, []).append(idx)
    return grouped


def _safe_variance(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mu = mean(values)
    return sum((value - mu) ** 2 for value in values) / len(values)


def species_error_variance(errors: Sequence[float], labels: Sequence[str]) -> Dict[str, float]:
    if len(errors) != len(labels):
        raise ValueError("errors and labels must have identical lengths")

    grouped = group_indices_by_species(labels)
    return {
        species: _safe_variance([errors[i] for i in indices])
        for species, indices in grouped.items()
    }


def detect_species_outliers(
    reconstruction_errors: Sequence[float],
    labels: Sequence[str],
    z_threshold: float = 3.5,
    min_group_size: int = 5,
) -> List[bool]:
    """Species-aware robust outlier detection using MAD z-scores.

    Args:
        reconstruction_errors: Per-sample reconstruction errors.
        labels: Species label for each sample.
        z_threshold: Robust z-score threshold.
        min_group_size: Minimum examples needed for pruning in a species.

    Returns:
        Boolean list where True marks an outlier to remove.
    """
    if len(reconstruction_errors) != len(labels):
        raise ValueError("reconstruction_errors and labels must have identical lengths")

    outliers = [False] * len(reconstruction_errors)
    grouped = group_indices_by_species(labels)

    for indices in grouped.values():
        if len(indices) < min_group_size:
            continue

        values = [reconstruction_errors[i] for i in indices]
        med = median(values)
        abs_dev = [abs(value - med) for value in values]
        mad = median(abs_dev)

        if mad == 0:
            # Fallback: only mark points that are strictly above median in zero-variance groups.
            for i in indices:
                outliers[i] = reconstruction_errors[i] > med
            continue

        for i in indices:
            robust_z = 0.6745 * (reconstruction_errors[i] - med) / mad
            outliers[i] = robust_z > z_threshold

    return outliers


def prune_records(records: Sequence[SampleRecord], outlier_flags: Sequence[bool]) -> List[SampleRecord]:
    if len(records) != len(outlier_flags):
        raise ValueError("records and outlier_flags must have identical lengths")
    return [record for record, is_outlier in zip(records, outlier_flags) if not is_outlier]


def average_precision_binary(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Compute binary AP without third-party dependencies."""
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have identical lengths")

    positives = sum(1 for value in y_true if value == 1)
    if positives == 0:
        return 0.0

    ranked = sorted(zip(y_score, y_true), key=lambda pair: pair[0], reverse=True)
    tp = 0
    precision_sum = 0.0

    for rank, (_, truth) in enumerate(ranked, start=1):
        if truth == 1:
            tp += 1
            precision_sum += tp / rank

    return precision_sum / positives


def macro_map(
    true_labels: Sequence[str],
    class_probabilities: Sequence[Mapping[str, float]],
    class_order: Sequence[str],
) -> float:
    """One-vs-rest macro mAP for multi-class outputs."""
    if len(true_labels) != len(class_probabilities):
        raise ValueError("true_labels and class_probabilities must have identical lengths")

    class_aps: List[float] = []
    for target_class in class_order:
        y_true = [1 if label == target_class else 0 for label in true_labels]
        y_score = [row.get(target_class, 0.0) for row in class_probabilities]
        class_aps.append(average_precision_binary(y_true, y_score))

    if not class_aps:
        return 0.0
    return sum(class_aps) / len(class_aps)


def compare_before_after_map(
    true_labels: Sequence[str],
    baseline_probabilities: Sequence[Mapping[str, float]],
    denoised_probabilities: Sequence[Mapping[str, float]],
    class_order: Sequence[str],
) -> Dict[str, float]:
    baseline = macro_map(true_labels, baseline_probabilities, class_order)
    denoised = macro_map(true_labels, denoised_probabilities, class_order)
    return {
        "baseline_map": baseline,
        "denoised_map": denoised,
        "delta_map": denoised - baseline,
    }


def train_conv_autoencoder_reconstruction_errors(
    mel_tensor,
    epochs: int = 10,
    learning_rate: float = 1e-3,
):
    """Optional PyTorch helper for Conv-AE reconstruction errors.

    Expects mel_tensor with shape [N, 1, H, W].
    """
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PyTorch is required for conv-autoencoder training. "
            "Install torch to run this part of the pipeline."
        ) from exc

    class ConvAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(8, 1, kernel_size=2, stride=2),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = ConvAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        reconstructed = model(mel_tensor)
        loss = loss_fn(reconstructed, mel_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        reconstructed = model(mel_tensor)
        sample_errors = ((reconstructed - mel_tensor) ** 2).mean(dim=(1, 2, 3))
    return sample_errors.cpu().tolist()


def train_small_cnn_probabilities(
    train_x,
    train_y,
    infer_x,
    class_order: Sequence[str],
    epochs: int = 6,
    learning_rate: float = 1e-3,
):
    """Optional PyTorch helper for before/after classifier probabilities."""
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PyTorch is required for classifier training. Install torch to run this stage."
        ) from exc

    class_to_idx = {species: idx for idx, species in enumerate(class_order)}
    y_idx = torch.tensor([class_to_idx[label] for label in train_y], dtype=torch.long)

    class SmallCNN(nn.Module):
        def __init__(self, num_classes: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Linear(16, num_classes)

        def forward(self, x):
            feat = self.features(x).flatten(1)
            return self.classifier(feat)

    model = SmallCNN(num_classes=len(class_order))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(train_x)
        loss = loss_fn(logits, y_idx)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(infer_x)
        probs = torch.softmax(logits, dim=1).cpu().tolist()

    return [
        {species: row[idx] for species, idx in class_to_idx.items()}
        for row in probs
    ]
