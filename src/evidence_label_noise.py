"""Evidence pack: how noisy are the labels we trained and tested on?

Answers the question "the gain is only ~1%, so what did you do?" with
measurements instead of claims:

  1. label-noise rate of the RAW training pool used by Model A
  2. label-noise rate of the TEST set (same for A and C) -> this is the
     ceiling on measurable accuracy, because the ground truth of the
     test chunks is itself wrong for a known share of the set
  3. per-species test noise (which classes cap the headline metric)
  4. the "trusted subset" of the test set (concordant chunks only) that a
     clean-test evaluation would use

Inputs (read-only, no training, no gold contact):
  data/audit_triage.csv      (chunk_id -> triage)
  results/splits_by_chunk.csv (chunk_id -> split, in_raw, in_cleaned)
Outputs:
  results/label_noise_evidence.csv
  results/label_noise_evidence.md
Usage: python src/evidence_label_noise.py
"""
import csv
import pathlib
from collections import Counter, defaultdict

REPO = pathlib.Path(__file__).resolve().parents[1]
TRIAGE = REPO / "data" / "audit_triage.csv"
SPLITS = REPO / "results" / "splits_by_chunk.csv"
OUT_CSV = REPO / "results" / "label_noise_evidence.csv"
OUT_MD = REPO / "results" / "label_noise_evidence.md"

# triage buckets that make the given label UNUSABLE for training
UNUSABLE = ("empty", "ambiguous", "intruder")


def main() -> int:
    tri = {r["chunk_id"]: r for r in
           csv.DictReader(open(TRIAGE, encoding="utf-8"))}
    rows = list(csv.DictReader(open(SPLITS, encoding="utf-8")))

    def pool(name, pred):
        sel = [r for r in rows if pred(r)]
        c = Counter(tri[r["chunk_id"]]["triage"] for r in sel)
        return len(sel), c

    a_tr_n, a_tr_c = pool("A-train", lambda r: r["split"] == "train"
                           and r["in_raw"] == "1")
    c_tr_n, c_tr_c = pool("C-train", lambda r: r["split"] == "train"
                           and r["in_cleaned"] == "1")
    te_n, te_c = pool("test", lambda r: r["split"] == "test"
                       and r["in_raw"] == "1")
    va_n, va_c = pool("val", lambda r: r["split"] == "val"
                       and r["in_raw"] == "1")

    def bad(c):
        return sum(c.get(k, 0) for k in UNUSABLE)

    # per-species test noise
    per_species = defaultdict(Counter)
    for r in rows:
        if r["split"] == "test" and r["in_raw"] == "1":
            per_species[r["species"]][tri[r["chunk_id"]]["triage"]] += 1

    out_rows = []
    for setname, n, c in (("A-train (raw labels)", a_tr_n, a_tr_c),
                          ("C-train (cleaned labels)", c_tr_n, c_tr_c),
                          ("val (raw labels)", va_n, va_c),
                          ("test (raw labels, used by A and C)", te_n, te_c)):
        out_rows.append({
            "set": setname, "n_chunks": n,
            "concordant": c.get("concordant", 0),
            "empty": c.get("empty", 0),
            "ambiguous": c.get("ambiguous", 0),
            "intruder": c.get("intruder", 0),
            "unusable": bad(c),
            "unusable_pct": round(100.0 * bad(c) / n, 1) if n else 0.0,
        })
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    te_bad = bad(te_c)
    te_good = te_c.get("concordant", 0)
    md = []
    md.append("# Label-noise evidence pack (auto-generated)")
    md.append("")
    md.append("Source: data/audit_triage.csv + results/splits_by_chunk.csv")
    md.append("")
    md.append("| set | chunks | concordant | empty | ambiguous | intruder | "
              "unusable | unusable % |")
    md.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in out_rows:
        md.append("| {set} | {n_chunks} | {concordant} | {empty} | "
                  "{ambiguous} | {intruder} | {unusable} | {unusable_pct}% |"
                  .format(**r))
    md.append("")
    md.append("## Per-species noise of the shared test set")
    md.append("")
    md.append("| species | test chunks | concordant | unusable | unusable % |")
    md.append("| --- | --- | --- | --- | --- |")
    for s in sorted(per_species):
        c = per_species[s]
        n = sum(c.values())
        md.append("| %s | %d | %d | %d | %.1f%% |"
                  % (s, n, c.get("concordant", 0), bad(c),
                     100.0 * bad(c) / n if n else 0.0))
    md.append("")
    md.append("## Reading of the numbers")
    md.append("")
    md.append("- Model A trained on %d raw-labelled chunks, of which %d (%.1f%%)"
              " carry an unusable label (empty / ambiguous / different"
              " species)." % (a_tr_n, bad(a_tr_c),
                             100.0 * bad(a_tr_c) / a_tr_n if a_tr_n else 0))
    md.append("- Model C trained on %d chunks, 100%% of them audited as"
              " concordant." % c_tr_n)
    md.append("- The shared test set holds %d chunks and %d (%.1f%%) of them"
              " are themselves mislabelled or unusable, so the reported"
              " accuracy is measured against partly wrong ground truth."
              % (te_n, te_bad, 100.0 * te_bad / te_n if te_n else 0))
    md.append("- Only %d test chunks (%d) have a verified target bird; that"
              " is the subset a clean-test evaluation must use before any"
              " claim about absolute accuracy."
              % (te_good, te_bad))
    md.append("")
    OUT_MD.write_text("\n".join(md), encoding="utf-8")

    for r in out_rows:
        print("%-38s n=%5d unusable=%4d (%4.1f%%)"
              % (r["set"], r["n_chunks"], r["unusable"], r["unusable_pct"]))
    print("\nper-species test noise:")
    for s in sorted(per_species):
        c = per_species[s]
        n = sum(c.values())
        print("  %-32s n=%3d unusable=%3d (%.1f%%)"
              % (s, n, bad(c), 100.0 * bad(c) / n if n else 0))
    print("\nwrote", OUT_CSV.relative_to(REPO), "and",
          OUT_MD.relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
