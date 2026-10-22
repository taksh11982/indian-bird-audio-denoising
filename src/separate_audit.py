"""P4.5 separation audit: Bird-MixIT dominant-bird rating of discarded chunks.

User-approved task (2026-10-22): rate the chunks whose triage is 'intruder' or
'ambiguous' — the ~523 multi-bird chunks cleaning discards — by isolating which
bird is making the most sound, then judging correct/incorrect against the
folder label. Fully automated, no human listening: BirdNET is the rater,
exactly as in the P2 triage. Gold chunk_ids are excluded FIRST (eval-only
rule): 523 - 10 = 513 chunks rated.

Method (frozen):
  1. Bird-MixIT (Denton et al., ICASSP 2022; matijama PyTorch port, Apache
     2.0) separates each 5.0 s 16 kHz chunk into N_SRC=4 sources at 22.05 kHz
     mono; sources sum to the mix. Written to data/separated/<chunk_id>/.
  2. BirdNET V2.4 — the SAME setup as audit.py (3.0 s windows, max-pool) —
     scores every separated source, giving one S-vector per source over the
     53 folder-space species.
  3. Dominant bird = species with the highest confidence across the 4 source
     channels (max model activations over separated channels — the recipe
     Denton et al. themselves use for classification). Verdict:
       correct     = dominant species == given folder label
       incorrect   = dominant species != given folder label
       no_dominant = dominant confidence < 0.20 (same floor as triage empty)
  4. Rescue signal: target_best_conf = max confidence of the folder species
     over the channels (>= 0.50 would have passed the concordance floor).

Outputs: results/separation_523.csv (one row per chunk, per-channel detail),
         results/separation_523_summary.json,
         figures/separation_verdicts.png + .pdf (additive; the two existing
         noise_summary figures are NEVER touched).
Stages (resume-safe, like audit.py):
  --stage separate|score|report|all   (--limit N for smoke tests)
Deps: torch + bird-mixit-pytorch (installed 2026-10-22 with user approval);
      birdnet_analyzer, soundfile, numpy reused. AssertionError on anything
      suspicious instead of silent guesses.
"""
import argparse
import csv
import json
import pathlib
import random
import sys
from collections import Counter

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import audit  # noqa: E402  (reuse BirdNET setup / label mapping / scoring)

TRIAGE = REPO / "data" / "audit_triage.csv"
INDEX = REPO / "data" / "chunk_index.csv"
GOLD = REPO / "data" / "gold_test_150.csv"
SEPDIR = REPO / "data" / "separated"
OUT_CSV = REPO / "results" / "separation_523.csv"
OUT_JSON = REPO / "results" / "separation_523_summary.json"
FIGDIR = REPO / "figures"

SAVE_SR = 22050   # Bird-MixIT native output rate
N_SRC = 4
NO_BIRD = audit.T_EMPTY  # 0.20 floor, mirrors triage

FIELDS = (["chunk_id", "given_label", "orig_triage", "orig_pred_label",
           "orig_max_conf", "dominant_species", "dominant_conf",
           "dominant_channel", "dominant_rms_share", "target_best_conf",
           "verdict"]
          + sum([[f"src{c}_species", f"src{c}_conf", f"src{c}_rms_share"]
                 for c in range(N_SRC)], []))


def selection():
    """(triage map, selected rows) for intruder+ambiguous minus gold."""
    tri = {r["chunk_id"]: r for r in
           csv.DictReader(open(TRIAGE, encoding="utf-8"))}
    gold = {r["chunk_id"] for r in
            csv.DictReader(open(GOLD, encoding="utf-8"))}
    sel = sorted((r for r in tri.values()
                  if r["triage"] in ("intruder", "ambiguous")
                  and r["chunk_id"] not in gold),
                 key=lambda r: r["chunk_id"])
    n_gold_in = sum(1 for r in tri.values()
                    if r["triage"] in ("intruder", "ambiguous")
                    and r["chunk_id"] in gold)
    print(f"selection: {len(sel)} chunks "
          f"(intruder+ambiguous={len(sel) + n_gold_in}, gold excluded={n_gold_in})")
    return tri, sel


def idx_map():
    return {r["chunk_id"]: r for r in
            csv.DictReader(open(INDEX, encoding="utf-8"))}


def setup_scoring():
    """Same BirdNET/order/col_idx construction as audit.main()."""
    index_rows = list(csv.DictReader(open(INDEX, encoding="utf-8")))
    cfg = audit.setup_birdnet()
    order = audit.species_order(index_rows)
    col_idx = np.array([ci if (ci := audit.label_index(cfg, s)) is not None
                        else -1 for s in order])
    mapped = {s for s, c in zip(order, col_idx) if c != -1}
    return cfg, order, col_idx, mapped


def stage_separate(sel, idx, limit):
    import soundfile as sf
    import torch
    from bird_mixit_pytorch import BirdMixIT, load_audio

    torch.manual_seed(42)
    torch.set_grad_enabled(False)
    model = BirdMixIT.from_pretrained(num_sources=N_SRC)
    model.eval()
    done = resumed = 0
    todo = sel[:limit] if limit else sel
    for n, r in enumerate(todo, 1):
        cid = r["chunk_id"]
        out = SEPDIR / cid
        if out.is_dir() and len(list(out.glob("source*.wav"))) >= N_SRC:
            resumed += 1
            continue
        out.mkdir(parents=True, exist_ok=True)
        audio = load_audio(str(REPO / idx[cid]["path_16k"]))
        src = model(audio)                            # [1, N_SRC, samples]
        s = src[0].cpu().float().numpy()              # (N_SRC, samples)
        mix = audio[0].cpu().float().numpy() if audio.dim() == 2 \
            else audio.cpu().float().numpy()          # (samples,)
        assert s.shape[0] == N_SRC and s.shape[1] == mix.shape[0], s.shape
        diff = float(np.abs(s.sum(axis=0) - mix).max())
        assert diff < 1e-3, f"{cid}: sources do not sum to mix (max {diff})"
        for c in range(N_SRC):
            sf.write(out / f"source{c}.wav", s[c], SAVE_SR,
                     subtype="PCM_16")
        done += 1
        if n % 25 == 0:
            print(f"separate: {n}/{len(todo)} (new={done} resumed={resumed})",
                  flush=True)
    print(f"separate done: new={done} resumed={resumed} total={len(todo)}")


def _rms_share(paths):
    import soundfile as sf
    rms = []
    for p in paths:
        y, _ = sf.read(str(p), dtype="float32")
        rms.append(float(np.sqrt(np.mean(y * y)) + 1e-12))
    tot = sum(rms)
    return [float(r / tot) for r in rms]


def stage_score(sel, tri, limit, csv_path):
    cfg, order, col_idx, mapped = setup_scoring()
    existing = set()
    if csv_path.exists():
        existing = {r["chunk_id"] for r in
                    csv.DictReader(open(csv_path, encoding="utf-8"))}
    new_file = not csv_path.exists()
    f = open(csv_path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        w.writeheader()
    done = resumed = 0
    try:
        for n, r in enumerate(sel if not limit else sel, 1):
            if limit and n > limit:
                break
            cid = r["chunk_id"]
            if cid in existing:
                resumed += 1
                continue
            paths = sorted((SEPDIR / cid).glob("source*.wav"))
            assert len(paths) == N_SRC, f"{cid}: {len(paths)} sources"
            S = np.stack([audit.score_chunk(cfg, str(p), col_idx)[0]
                          for p in paths])               # (4, 53)
            top = np.unravel_index(int(np.argmax(S)), S.shape)
            dom_sp, dom_conf, dom_ch = (order[top[1]],
                                        float(S[top]), int(top[0]))
            given = tri[cid]["given_label"]
            assert given in mapped, f"{cid}: given '{given}' unmapped?!"
            tcol = order.index(given)  # target column in the 53-species space
            tgt_best = float(S[:, tcol].max())
            verdict = ("no_dominant" if dom_conf < NO_BIRD
                       else "correct" if dom_sp == given else "incorrect")
            shares = _rms_share(paths)
            row = {"chunk_id": cid, "given_label": given,
                   "orig_triage": tri[cid]["triage"],
                   "orig_pred_label": tri[cid]["pred_label"],
                   "orig_max_conf": tri[cid]["max_conf"],
                   "dominant_species": dom_sp, "dominant_conf": f"{dom_conf:.4f}",
                   "dominant_channel": dom_ch,
                   "dominant_rms_share": f"{shares[dom_ch]:.3f}",
                   "target_best_conf": f"{tgt_best:.4f}", "verdict": verdict}
            for c in range(N_SRC):
                row[f"src{c}_species"] = order[int(S[c].argmax())]
                row[f"src{c}_conf"] = f"{float(S[c].max()):.4f}"
                row[f"src{c}_rms_share"] = f"{shares[c]:.3f}"
            w.writerow(row)
            f.flush()
            done += 1
            if n % 25 == 0:
                print(f"score: {n}/{len(sel)} (new={done} resumed={resumed})",
                      flush=True)
    finally:
        f.close()
    print(f"score done: new={done} resumed={resumed} total={limit or len(sel)}")


def stage_report():
    """Merge all shard CSVs (or the single one), dedupe, rewrite canonical.

    Canonical output stays results/separation_523.csv (sorted by chunk_id);
    per-shard temp CSVs are removed after the merge so the repo keeps exactly
    one rated file.
    """
    paths = [OUT_CSV, *sorted((REPO / "results").glob("separation_523.shard*.csv"))]
    rows = {}
    for p in paths:
        if not p.exists():
            continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            rows[r["chunk_id"]] = r
    rows = [rows[k] for k in sorted(rows)]
    if not rows:
        raise SystemExit(f"no rows found from {[str(p) for p in paths]}")
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    for p in (REPO / "results").glob("separation_523.shard*.csv"):
        p.unlink()
    verdict = Counter(r["verdict"] for r in rows)
    by_tri = {}
    for t in ("intruder", "ambiguous"):
        sub = [r for r in rows if r["orig_triage"] == t]
        by_tri[t] = dict(Counter(r["verdict"] for r in sub)) | {"n": len(sub)}
    rescue_bad = sum(1 for r in rows if r["verdict"] == "incorrect"
                     and float(r["target_best_conf"]) >= 0.50)
    ch_hist = Counter(r["dominant_channel"] for r in rows)
    confs = sorted(float(r["dominant_conf"]) for r in rows)
    n = len(rows)
    summary = {
        "date": "2026-10-22",
        "rule": "dominant = max BirdNET conf over 4 separated channels "
                "(Denton et al. 2022); verdict vs folder label; "
                "no_dominant if conf < 0.20",
        "selection": "audit_triage intruder+ambiguous, gold excluded",
        "n_rated": n,
        "verdicts": dict(verdict),
        "by_orig_triage": by_tri,
        "incorrect_but_target_recoverable": rescue_bad,
        "dominant_channel_hist": dict(ch_hist),
        "dominant_conf_median": confs[n // 2] if n else None,
        "example_spread": {r["chunk_id"]: {"given": r["given_label"],
                                           "dominant": r["dominant_species"],
                                           "verdict": r["verdict"]}
                           for r in rows[:5]},
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cats = [("correct", "correct (target is dominant)", "#2ca02c"),
            ("incorrect", "incorrect (other bird dominant)", "#d62728"),
            ("no_dominant", "no dominant bird (<0.20)", "#7f7f7f")]
    tri_rows = [t for t in ("intruder", "ambiguous") if by_tri[t]["n"]]
    fig, ax = plt.subplots(figsize=(8, 0.9 + 0.5 * len(tri_rows)))
    left = np.zeros(len(tri_rows))
    for key, lab, col in cats:
        vals = np.array([by_tri[t].get(key, 0) for t in tri_rows], float)
        ax.barh(tri_rows, vals, left=left, color=col, label=lab, height=0.6)
        for i, v in enumerate(vals):
            if v > 0:
                ax.text(left[i] + v / 2, i, str(int(v)), va="center",
                        ha="center", color="white", fontsize=9)
        left += vals
    ax.set_xlabel("chunks")
    ax.set_title(f"Bird-MixIT dominant-bird rating of discarded chunks "
                 f"(n={n}, gold excluded)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "separation_verdicts.png", dpi=300)
    fig.savefig(FIGDIR / "separation_verdicts.pdf")
    print(json.dumps(summary, indent=2)[:1200])
    print(f"figure: {FIGDIR / 'separation_verdicts.png'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["separate", "score", "report", "all"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0,
                    help="process shard k of nshards (sel[k::nshards])")
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0,
                    help="torch threads for sharded workers; 0 = default")
    args = ap.parse_args()
    random.seed(42)
    np.random.seed(42)
    tri, sel = selection()
    if args.nshards > 1:
        sel = sel[args.shard::args.nshards]
        print(f"shard {args.shard}/{args.nshards}: {len(sel)} chunks")
        import torch
        torch.set_num_threads(args.threads or 1)
    idx = idx_map()
    missing = [r["chunk_id"] for r in sel if r["chunk_id"] not in idx]
    assert not missing, f"{len(missing)} selected ids missing from index"
    csv_path = (OUT_CSV if args.nshards <= 1 else
                REPO / "results" /
                f"separation_523.shard{args.shard}of{args.nshards}.csv")
    stages = (["separate", "score", "report"]
              if args.stage == "all" else [args.stage])
    if "separate" in stages:
        stage_separate(sel, idx, args.limit)
    if "score" in stages:
        stage_score(sel, tri, args.limit, csv_path)
    if "report" in stages:
        stage_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
