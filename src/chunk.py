"""P1 chunking: manifest -> dual 5.0 s streams + data/chunk_index.csv.

Streams: A = 48 kHz mono (BirdNET native), B = 16 kHz mono (0-8 kHz Nyquist).
Protocol: non-overlapping 5.0 s windows; residual tail < 2.0 s discarded.
# ponytail: tails in [2.0, 5.0) s are zero-padded to exactly 5.0 s (padded_s
#   recorded in chunk_index.csv) so every emitted chunk meets the exactly-5.0 s
#   DoD; upgrade path = drop padded chunks if the audit shows edge artifacts.
Chunk ID: <species>_<recording_id>_c<idx:03d>, deterministic per manifest order.
Outputs: data/chunks_48k/<id>.wav, data/chunks_16k/<id>.wav (PCM-16),
  data/chunk_index.csv (chunk_id,species_name,recording_id,chunk_idx,
  offset_s,padded_s,path_48k,path_16k).
No stochastic ops (seed 42 N/A here).
"""
import csv
import pathlib

import librosa
import soundfile as sf
from tqdm import tqdm

REPO = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = REPO / "data" / "raw_manifest.csv"
DIR_48K = REPO / "data" / "chunks_48k"
DIR_16K = REPO / "data" / "chunks_16k"
INDEX = REPO / "data" / "chunk_index.csv"

SR_A = 48000
SR_B = 16000
CHUNK_S = 5.0
TAIL_KEEP_S = 2.0
N_A = int(SR_A * CHUNK_S)  # 240000
N_B = int(SR_B * CHUNK_S)  # 80000


def plan_chunks(dur_s):
    """One time-based plan shared by both streams.

    Per-stream sample counts differ by a few samples after resampling, so
    chunk boundaries are derived from seconds, not per-stream lengths.
    Returns (n_full, keep_tail, rem_s).
    """
    n_full = int(dur_s // CHUNK_S)
    rem_s = dur_s - n_full * CHUNK_S
    return n_full, rem_s >= TAIL_KEEP_S, rem_s


def take_segment(y, sr, start_s):
    """Slice exactly n_win samples at start_s, zero-padding or truncating.

    Resample rounding can leave a segment a few samples short/long; the
    correction is sub-millisecond and keeps the exactly-5.0 s DoD literal.
    Returns (segment, padded_s).
    """
    import numpy as np
    n_win = int(sr * CHUNK_S)
    start = int(round(start_s * sr))
    seg = y[start:start + n_win]
    if len(seg) < n_win:
        pad_s = (n_win - len(seg)) / float(sr)
        seg = np.concatenate([seg, np.zeros(n_win - len(seg), dtype=seg.dtype)])
        return seg, pad_s
    return seg[:n_win], 0.0


def main() -> int:
    DIR_48K.mkdir(parents=True, exist_ok=True)
    DIR_16K.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    index_rows = []
    n_zero = 0
    for r in tqdm(rows, desc="chunking", unit="file"):
        src = REPO / r["filepath"]
        y, native_sr = librosa.load(str(src), sr=None, mono=True)
        dur_s = len(y) / float(native_sr)
        n_full, keep_tail, rem_s = plan_chunks(dur_s)
        y_a = librosa.resample(y, orig_sr=native_sr, target_sr=SR_A)
        y_b = librosa.resample(y, orig_sr=native_sr, target_sr=SR_B)
        n_chunks = n_full + (1 if keep_tail else 0)
        if n_chunks == 0:
            n_zero += 1
            continue
        for idx in range(n_chunks):
            start_s = idx * CHUNK_S
            seg_a, pad_a = take_segment(y_a, SR_A, start_s)
            seg_b, pad_b = take_segment(y_b, SR_B, start_s)
            assert len(seg_a) == N_A and len(seg_b) == N_B, f"bad length: {src}"
            cid = f"{r['species_name']}_{r['recording_id']}_c{idx:03d}"
            p_a = DIR_48K / f"{cid}.wav"
            p_b = DIR_16K / f"{cid}.wav"
            sf.write(str(p_a), seg_a, SR_A, subtype="PCM_16")
            sf.write(str(p_b), seg_b, SR_B, subtype="PCM_16")
            index_rows.append({
                "chunk_id": cid,
                "species_name": r["species_name"],
                "recording_id": r["recording_id"],
                "chunk_idx": str(idx),
                "offset_s": f"{start_s:.3f}",
                "padded_s": f"{pad_a:.3f}",
                "path_48k": p_a.relative_to(REPO).as_posix(),
                "path_16k": p_b.relative_to(REPO).as_posix(),
            })
    with open(INDEX, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "chunk_id", "species_name", "recording_id", "chunk_idx",
            "offset_s", "padded_s", "path_48k", "path_16k"])
        writer.writeheader()
        writer.writerows(index_rows)
    n_pad = sum(1 for q in index_rows if float(q["padded_s"]) > 0)
    print(f"chunks={len(index_rows)} padded={n_pad} "
          f"files_zero_chunks={n_zero} index={INDEX.relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
