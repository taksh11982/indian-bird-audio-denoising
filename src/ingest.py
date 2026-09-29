"""P1 ingestion: scan data/raw/ibc53/<species>/*.wav -> data/raw_manifest.csv.

Columns: filepath,species_name,recording_id,duration_s
- species_name: parent directory name, verbatim (spaces preserved).
- recording_id: audio file stem (e.g. '1', '10').
- filepath: repo-relative POSIX path (forward slashes).
- duration_s: from WAV header (frames / samplerate), no decode.
- Deterministic order: sorted by (species_name, recording_id).
- No stochastic ops (seed 42 N/A here).
"""
import csv
import pathlib
import sys

import soundfile as sf

REPO = pathlib.Path(__file__).resolve().parents[1]
RAW_ROOT = REPO / "data" / "raw" / "ibc53"
OUT = REPO / "data" / "raw_manifest.csv"


def sort_key(stem: str):
    return (0, int(stem), stem) if stem.isdigit() else (1, 0, stem)


def main() -> int:
    if not RAW_ROOT.is_dir():
        print(f"missing dataset dir: {RAW_ROOT}", file=sys.stderr)
        return 1
    rows = []
    skipped = 0
    for species_dir in sorted(RAW_ROOT.iterdir()):
        if not species_dir.is_dir():
            continue
        species = species_dir.name
        files = sorted(
            (p for p in species_dir.iterdir()
             if p.is_file() and p.suffix.lower() == ".wav"),
            key=lambda p: sort_key(p.stem),
        )
        skipped += sum(
            1 for p in species_dir.iterdir()
            if p.is_file() and p.suffix.lower() != ".wav"
        )
        for p in files:
            info = sf.info(str(p))
            rows.append({
                "filepath": p.relative_to(REPO).as_posix(),
                "species_name": species,
                "recording_id": p.stem,
                "duration_s": f"{info.frames / float(info.samplerate):.3f}",
            })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["filepath", "species_name", "recording_id", "duration_s"])
        writer.writeheader()
        writer.writerows(rows)
    total_h = sum(float(r["duration_s"]) for r in rows) / 3600.0
    n_species = len({r["species_name"] for r in rows})
    print(f"files={len(rows)} species={n_species} hours={total_h:.2f} "
          f"manifest={OUT.relative_to(REPO).as_posix()} skipped_nonwav={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
