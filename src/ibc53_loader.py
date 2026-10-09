"""Torch Dataset over data/iBC53-Cleaned-Metadata.csv (open-science loader).

Reads chunk_id,species,path_16k; maps species -> idx over sorted focus-11
(or all 44 cleaned species when focus11=False); returns (MFCC (157,39),
label). Keeps agents.md dual-stream/5.0 s contract: asserts exactly 80000
samples at 16 kHz. Seed 42 for the optional shuffle/split helper.
Usage (Colab, torch installed):
  from ibc53_loader import IBC53Cleaned
  from gavali_features import FOCUS_11
  ds = IBC53Cleaned(focus11=True); x, y = ds[0]
"""
import csv
import pathlib
import random

REPO = pathlib.Path(__file__).resolve().parents[1]
CSV = REPO / "data" / "iBC53-Cleaned-Metadata.csv"


def _species_list(focus11=True):
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from gavali_features import FOCUS_11
    if focus11:
        return sorted(FOCUS_11)
    with open(CSV, newline="", encoding="utf-8") as fh:
        return sorted({r["species"] for r in csv.DictReader(fh)})


class IBC53Cleaned:
    def __init__(self, focus11=True, transform=None):
        import torch  # noqa: F401  (Colab artifact; needs torch)
        self.species = _species_list(focus11)
        self.to_idx = {s: i for i, s in enumerate(self.species)}
        with open(CSV, newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh)
                    if r["species"] in set(self.species)]
        rows.sort(key=lambda r: (r["species"], r["chunk_id"]))
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import sys
        sys.path.insert(0, str(REPO / "src"))
        from gavali_features import mfcc_sequence
        import torch
        r = self.rows[i]
        x = mfcc_sequence(str(REPO / r["path_16k"]))
        if self.transform is not None:
            x = self.transform(x)
        # ponytail: MFCCs computed on the fly; upgrade = disk cache (.npz).
        return torch.from_numpy(x), self.to_idx[r["species"]]


def train_val_split(rows, seed=42, val_frac=0.1):
    """Deterministic shuffle helper for notebooks (seed 42)."""
    rng = random.Random(seed)
    ids = sorted(range(len(rows)))
    rng.shuffle(ids)
    n_va = max(1, int(len(ids) * val_frac))
    return ids[n_va:], ids[:n_va]
