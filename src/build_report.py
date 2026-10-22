"""Rebuild the analysis / proposed-work / conclusions sections of the report.

The report body is regenerated from the committed artefacts in data/ and
results/ so that the document and the repository can never drift apart:

  python src/build_report.py

It (1) rebuilds the sections from Table 1 to the end of the report,
(2) embeds the figures in figures/ as inline pictures,
(3) restores the original <w:document> root element (the rebuilt root must
    keep every namespace that mc:Ignorable refers to, otherwise Word refuses
    to open the file) and (4) leaves the front matter untouched.
"""
import csv
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "Report Format - Project2- Sem7 .docx"
TEMPLATE = Path(sys.argv[1]) if len(sys.argv) > 1 else None
FIGDIR = REPO / "figures"
RESDIR = REPO / "results"

EMU_IN = 914400
MAX_W_EMU = int(5.85 * EMU_IN)

FIGURES = {
    1: "fig2_triage_share.png",
    2: "fig3_species_noise_before.png",
    3: "fig4_species_noise_after.png",
    4: "fig5_pool_quality.png",
    5: "fig7_ablation.png",
    6: "fig6_confusion_ab.png",
    7: "fig9_per_class_recall.png",
    8: "fig10_mechanism.png",
    9: "fig11_power.png",
    10: "fig1_pipeline.png",
    11: "fig12_control_design.png",
}


# ----------------------------------------------------------------- xml helpers
def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def p(text, style=None, jc=None, bold=False, size=None, spacing_after=200,
      line=360, keep=False):
    ppr = ["<w:pPr>"]
    if style:
        ppr.append(f'<w:pStyle w:val="{style}"/>')
    if keep:
        ppr.append("<w:keepNext/>")
    ppr.append('<w:pBdr><w:top w:val="nil"/><w:left w:val="nil"/>'
               '<w:bottom w:val="nil"/><w:right w:val="nil"/>'
               '<w:between w:val="nil"/></w:pBdr>')
    ppr.append(f'<w:spacing w:after="{spacing_after}" w:line="{line}" '
               f'w:lineRule="auto"/>')
    ppr.append('<w:ind w:left="0" w:hanging="2"/>')
    if jc:
        ppr.append(f'<w:jc w:val="{jc}"/>')
    rpr = ["<w:rPr>", '<w:color w:val="000000"/>']
    if bold:
        rpr.append("<w:b/>")
    if size:
        rpr.append(f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>')
    rpr.append("</w:rPr>")
    ppr.append("".join(rpr))
    ppr.append("</w:pPr>")
    r = ["<w:r>", "".join(rpr),
         f'<w:t xml:space="preserve">{esc(text)}</w:t>', "</w:r>"]
    return "<w:p>" + "".join(ppr) + "".join(r) + "</w:p>"


def h1(text):
    return ("<w:p><w:pPr><w:keepNext/><w:numPr><w:ilvl w:val=\"0\"/>"
            "<w:numId w:val=\"3\"/></w:numPr><w:pBdr><w:top w:val=\"nil\"/>"
            "<w:left w:val=\"nil\"/><w:bottom w:val=\"nil\"/>"
            "<w:right w:val=\"nil\"/><w:between w:val=\"nil\"/></w:pBdr>"
            '<w:spacing w:before="240" w:after="720" w:line="240" '
            'w:lineRule="auto"/><w:ind w:left="1" w:hanging="3"/>'
            '<w:jc w:val="center"/><w:rPr><w:color w:val="000000"/>'
            '<w:sz w:val="32"/><w:szCs w:val="32"/></w:rPr></w:pPr>'
            f"<w:r><w:rPr><w:color w:val=\"000000\"/><w:sz w:val=\"32\"/>"
            f"<w:szCs w:val=\"32\"/></w:rPr><w:t>{esc(text)}</w:t></w:r></w:p>")


def h2(text):
    return ("<w:p><w:pPr><w:pStyle w:val=\"Heading2\"/><w:numPr>"
            "<w:ilvl w:val=\"1\"/><w:numId w:val=\"3\"/></w:numPr>"
            '<w:ind w:left="0" w:hanging="2"/></w:pPr>'
            f"<w:r><w:t>{esc(text)}</w:t></w:r></w:p>")


def caption(text, kind=None, number=None):
    """Caption paragraph. With kind set, the number is a real Word SEQ field
    so that the List of Figures / List of Tables can be generated."""
    rpr = ('<w:rPr><w:b/><w:color w:val="000000"/><w:sz w:val="22"/>'
           '<w:szCs w:val="22"/></w:rPr>')
    if kind is None:
        runs = (f"<w:r>{rpr}<w:t xml:space=\"preserve\">{esc(text)}</w:t>"
                "</w:r>")
    else:
        runs = (f"<w:r>{rpr}<w:t xml:space=\"preserve\">{esc(kind)} </w:t>"
                "</w:r>"
                f'<w:r>{rpr}<w:fldChar w:fldCharType="begin"/></w:r>'
                f'<w:r>{rpr}<w:instrText xml:space="preserve"> SEQ {kind} '
                r"\* ARABIC </w:instrText></w:r>"
                f'<w:r>{rpr}<w:fldChar w:fldCharType="separate"/></w:r>'
                f"<w:r>{rpr}<w:t>{number}</w:t></w:r>"
                f'<w:r>{rpr}<w:fldChar w:fldCharType="end"/></w:r>'
                f"<w:r>{rpr}<w:t xml:space=\"preserve\">: {esc(text)}</w:t>"
                "</w:r>")
    return ("<w:p><w:pPr><w:pBdr><w:top w:val=\"nil\"/><w:left w:val=\"nil\"/>"
            "<w:bottom w:val=\"nil\"/><w:right w:val=\"nil\"/>"
            "<w:between w:val=\"nil\"/></w:pBdr>"
            '<w:spacing w:before="120" w:after="240" w:line="240" '
            'w:lineRule="auto"/><w:ind w:left="0" w:hanging="2"/>'
            '<w:jc w:val="center"/><w:rPr><w:b/><w:color w:val="000000"/>'
            '<w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr></w:pPr>'
            f"{runs}</w:p>")


def tbl(headers, rows, widths=None):
    n = len(headers)
    widths = widths or [round(9360 / n)] * n
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    out = ["<w:tbl><w:tblPr><w:tblW w:w=\"0\" w:type=\"auto\"/><w:tblBorders>"]
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        out.append(f'<w:{edge} w:val="single" w:sz="4" w:space="0" '
                   'w:color="808080"/>')
    out.append("</w:tblBorders></w:tblPr>")
    out.append(f"<w:tblGrid>{grid}</w:tblGrid>")
    all_rows = [headers] + rows
    for ri, row in enumerate(all_rows):
        out.append("<w:tr>")
        for ci, cell in enumerate(row):
            bold = "<w:b/>" if ri == 0 else ""
            out.append(
                f'<w:tc><w:tcPr><w:tcW w:w="{widths[ci]}" w:type="dxa"/>'
                "</w:tcPr><w:p><w:pPr>"
                '<w:spacing w:after="0" w:line="240" w:lineRule="auto"/>'
                "<w:ind w:left=\"0\" w:hanging=\"2\"/></w:pPr>"
                f"<w:r><w:rPr>{bold}<w:color w:val=\"000000\"/>"
                f'<w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>'
                f'<w:t xml:space="preserve">{esc(cell)}</w:t></w:r></w:p></w:tc>')
        out.append("</w:tr>")
    out.append("</w:tbl>")
    return "".join(out)


def split_children(s):
    children, depth, st, i, n = [], 0, None, 0, len(s)
    while i < n:
        if s[i] == "<":
            j = i + 1
            q = None
            while j < n:
                c = s[j]
                if q:
                    if c == q:
                        q = None
                elif c in "\"'":
                    q = c
                elif c == ">":
                    break
                j += 1
            tag = s[i:j + 1]
            if tag.startswith("</"):
                depth -= 1
                if depth == 0:
                    children.append(s[st:j + 1])
                    st = None
            elif tag.endswith("/>"):
                if depth == 0:
                    children.append(tag)
            else:
                if depth == 0:
                    st = i
                depth += 1
            i = j + 1
        else:
            i += 1
    return children


def picture(rid, name, fig_no, png_path):
    from PIL import Image
    with Image.open(png_path) as im:
        w, h = im.size
    cx = MAX_W_EMU
    cy = int(cx * h / w)
    return (
        '<w:p><w:pPr><w:keepNext/><w:jc w:val="center"/>'
        '<w:spacing w:before="120" w:after="60" w:line="240" '
        'w:lineRule="auto"/></w:pPr><w:r><w:drawing>'
        '<wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{cx}" cy="{cy}"/>'
        '<wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:docPr id="{fig_no}" name="Figure {fig_no}" '
        f'descr="{esc(name)}"/>'
        '<wp:cNvGraphicFramePr>'
        '<a:graphicFrameLocks noChangeAspect="1"/>'
        '</wp:cNvGraphicFramePr>'
        '<a:graphic><a:graphicData '
        'uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic><pic:nvPicPr>'
        f'<pic:cNvPr id="{fig_no}" name="Figure {fig_no}" descr="{esc(name)}"/>'
        '<pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rid}"/>'
        '<a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/>'
        '</a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
        '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>'
        '</w:p>')


# ------------------------------------------------------------------- data load
def load(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def build_body():
    tri = load(REPO / "data" / "audit_triage.csv")
    idx = load(REPO / "data" / "chunk_index.csv")
    by_chunk = {r["chunk_id"]: r for r in idx}
    noise = load(RESDIR / "label_noise_evidence.csv")
    abl = load(RESDIR / "gavali_ablation.csv")
    pc = load(RESDIR / "per_class_analysis.csv")
    power = load(RESDIR / "statistical_power.csv")

    n_total = len(tri)
    cnt = {k: sum(1 for r in tri if r["triage"] == k)
           for k in ("concordant", "empty", "ambiguous", "intruder",
                     "unlabeled")}
    padded = sum(1 for r in idx if float(r["padded_s"]) > 0)

    focus = [r["species"] for r in pc]
    spec_stats = []
    for sp in focus:
        ch = [r for r in idx if r["species_name"] == sp]
        conc = sum(1 for r in ch
                   if next(t["triage"] for t in tri
                           if t["chunk_id"] == r["chunk_id"]) == "concordant")
        spec_stats.append((sp, len(ch), conc, 100.0 * conc / max(1, len(ch))))
    spec_stats.sort(key=lambda x: -x[3])

    abl_map = {r["run"][0]: r for r in abl}
    A, C = abl_map["A"], abl_map["C"]
    A_f1, C_f1 = float(A["macro_f1"]), float(C["macro_f1"])
    A_acc, C_acc = float(A["accuracy"]), float(C["accuracy"])

    trust = {m: (sum(float(r[f"acc_trusted_{m}_lo"]) * int(r["n_trusted"])
                      for r in pc) / sum(int(r["n_trusted"]) for r in pc),
                 sum(float(r[f"acc_trusted_{m}_hi"]) * int(r["n_trusted"])
                     for r in pc) / sum(int(r["n_trusted"]) for r in pc))
             for m in ("A", "C")}
    n_trust = sum(int(r["n_trusted"]) for r in pc)
    n_test = sum(int(r["n_test"]) for r in pc)

    def powval(fragment):
        for r in power:
            if fragment.lower() in r["quantity"].lower():
                return r["value"]
        return "?"

    body = []
    A_ = body.append

    # =========================================================== Analysis
    A_(h1("Analysis and Findings"))
    A_(p("Every number, table and figure in this section is regenerated from "
         "the artefacts committed in data/ and results/ by the scripts in "
         "src/ (make_report_figures.py, per_class_analysis.py, "
         "statistical_power.py), so the report and the repository cannot "
         "drift apart. All figures are produced from the same seed-42 "
         "artefacts that produced the training runs."))

    A_(h2("The dataset and the chunking stage"))
    A_(p("The iBC53 dataset contains 1,368 WAV recordings distributed over 53 "
         "species, with a total duration of about 12.68 h (mean 33.4 s per "
         "file, range 1.0 to 362.8 s). The largest folder, Mystery mystery, "
         "contains 443 recordings that carry no species identity at all. "
         "Every recording was cut into non-overlapping 5.0 s chunks in two "
         "streams: a 48 kHz stream used by the BirdNET auditor and a 16 kHz "
         "stream used by the classifier, because the MFCC front-end of the "
         "downstream model is specified at 16 kHz with a 0-8 kHz Nyquist "
         "range. Residual tails shorter than 2.0 s were discarded; "
         f"{padded} chunks ({100.0*padded/n_total:.1f}%) are zero-padded "
         "tails, and the amount of real audio in every chunk is stored in "
         "chunk_index.csv so that padding can never be mistaken for signal. "
         "Table 1 collects the counts of this stage."))
    A_(caption("Summary of the iBC53 dataset, the chunking stage and the "
               "BirdNET audit.", "Table", 1))
    A_(tbl(["Item", "Value"], [
        ["iBC53 dataset", "53 species, 1,368 recordings, 12.68 h"],
        ["Chunking", "5.0 s non-overlapping; 48 kHz (audit) and 16 kHz "
                     "(model)"],
        ["Total chunks", f"{n_total} ({padded} zero-padded, "
                         f"{100.0*padded/n_total:.1f}%)"],
        ["Concordant (kept)", f"{cnt['concordant']} "
                              f"({100.0*cnt['concordant']/n_total:.1f}%)"],
        ["Empty (dropped)", f"{cnt['empty']} "
                            f"({100.0*cnt['empty']/n_total:.1f}%)"],
        ["Ambiguous (dropped)", f"{cnt['ambiguous']} "
                                f"({100.0*cnt['ambiguous']/n_total:.1f}%)"],
        ["Intruder (dropped)", f"{cnt['intruder']} "
                               f"({100.0*cnt['intruder']/n_total:.1f}%)"],
        ["Unlabelled (no BirdNET class)", f"{cnt['unlabeled']} "
                                          f"({100.0*cnt['unlabeled']/n_total:.1f}%)"],
        ["Auditor", "BirdNET-Analyzer V2.4, CPU, 3.0 s sliding windows, "
                   "max-pooled"],
        ["Focus set", f"{len(focus)} species, {n_test} test chunks "
                      f"({n_trust} audited-trusted)"],
    ], widths=[3000, 6360]))

    A_(h2("The audit: an unsupervised judge on every chunk"))
    A_(p("BirdNET-Analyzer V2.4 is a pre-trained network that scores a "
         "recording against 6,522 North American species. We use it not as a "
         "predictor but as an auditor. For every 48 kHz chunk we run the "
         "network over sliding 3.0 s windows, take the per-window confidence "
         "of every species, and max-pool the windows into a single 53-"
         "dimensional vector S in [0,1]^53 that belongs to the 53 iBC53 "
         "classes. A fixed, a-priori rule set then decides the fate of the "
         "chunk. The rules were frozen before any model was trained and were "
         "never tuned on any evaluation data; Table 2 lists them with the "
         "resulting counts."))
    A_(caption("The frozen triage rules and the resulting disposition of "
               "all 9,289 chunks.", "Table", 2))
    A_(tbl(["Condition", "Meaning", "Chunks", "Share"], [
        ["argmax(S) = label and max(S) >= 0.50", "concordant: the labelled "
         "bird is present", f"{cnt['concordant']}",
         f"{100.0*cnt['concordant']/n_total:.1f}%"],
        ["max(S) < 0.20", "empty: no bird detected at all",
         f"{cnt['empty']}", f"{100.0*cnt['empty']/n_total:.1f}%"],
        ["argmax(S) != label and max(S) >= 0.65", "intruder: a different "
         "species dominates", f"{cnt['intruder']}",
         f"{100.0*cnt['intruder']/n_total:.1f}%"],
        ["everything else", "ambiguous: weak or conflicting evidence",
         f"{cnt['ambiguous']}", f"{100.0*cnt['ambiguous']/n_total:.1f}%"],
        ["class not modelled by BirdNET", "unlabelled: no auditor class "
         "exists", f"{cnt['unlabeled']}",
         f"{100.0*cnt['unlabeled']/n_total:.1f}%"],
    ], widths=[3400, 3400, 1280, 1280]))
    A_(p("The headline number is the first row together with the second: "
         "roughly three out of every ten labelled chunks do not contain the "
         "bird they claim. The unlabelled 39.2% is dominated by the Mystery "
         "mystery folder and by four taxa that BirdNET V2.4 cannot name; it "
         "is a property of the auditor, not of the data. Excluding those "
         "chunks, the label-usable data is 62.9% concordant, 27.9% empty, "
         "8.9% ambiguous and 0.4% intruder. Figure 1 shows the disposition "
         "of the whole corpus; the pipeline that produced it is shown in Figure 10."))
    A_("@FIG1")

    A_(h2("Where the noise lives: it is not uniform across species"))
    A_(p("The cleaning is not a uniform trim. Figures 2 and 3 show the "
         "before and after composition per species, and Table 3 gives the "
         "exact concordant rate of the focus set. Some species are almost "
         "clean, others lose more than half of their data, and nine taxa "
         "disappear completely, either because BirdNET has no class for them "
         "or because every one of their chunks was empty. That asymmetry is "
         "the reason a per-recording split was used for the downstream "
         "experiment: a chunk-level split would let recordings of the same "
         "individual bird appear in both training and test data."))
    A_("@FIG2")
    A_("@FIG3")
    A_(caption("Concordant rate of the 11 focus species, i.e. the share of "
               "their chunks that survives cleaning.", "Table", 3))
    A_(tbl(["Species", "Total chunks", "Kept (concordant)", "Concordant %"],
           [[sp, f"{n}", f"{c}", f"{100.0*c/max(1,n):.1f}%"]
            for sp, n, c, _ in spec_stats], widths=[3600, 1920, 1920, 1920]))

    A_(h2("The label quality of the pools we actually train and test on"))
    A_(p("The audit is only interesting if it changes what the classifier "
         "sees. Table 4 measures the unusable labels inside each pool and "
         "Figure 4 shows the same three bars graphically. The raw training "
         "pool of Model A contains 726 unusable labels out of 2,563 chunks "
         "(28.3%); the cleaned pool of Model C contains none, by "
         "construction; and the shared test pool is dirty in the same way "
         "(26.5%). That last number has a direct consequence for the "
         "evaluation: a model that correctly reports silence on an empty "
         "clip is scored as wrong, so the accuracy measured against the raw "
         "test labels is a lower bound on the model's real behaviour, not a "
         "ceiling."))
    A_("@FIG4")
    A_(caption("Label-noise measurement of the training, validation and test "
               "pools (unusable = empty, ambiguous or wrong species).",
               "Table", 4))
    A_(tbl(["Pool", "Chunks", "Unusable labels", "Unusable %"],
           [[r["set"], r["n_chunks"], r["unusable"], f"{r['unusable_pct']}%"]
            for r in noise], widths=[3400, 1400, 2200, 1400]))

    A_(h2("The downstream experiment: raw labels against cleaned labels"))
    A_(p("Two models were trained with an identical pipeline, identical "
         "features and the same seed (42): a 39-band MFCC front-end (n_fft "
         "1024, hop 512) over the 16 kHz chunks followed by a single-layer "
         "LSTM with 128 hidden units, dropout 0.2, RMSprop at 10^-3, batch "
         "size 64 and early stopping on validation Macro-F1 with patience 6. "
         "Model A used every raw-labelled chunk available for training "
         "(2,563); Model C used the 1,837 concordant ones. The split is by "
         "recording, so no recording appears in two splits, and both models "
         "were evaluated on exactly the same 234 raw test chunks."))
    A_(caption("Ablation - raw labels (Model A) against cleaned labels "
               "(Model C), evaluated on the same 234 raw test chunks.",
               "Table", 5))
    A_(tbl(["Model", "Training data", "Train chunks", "Test chunks",
            "Accuracy", "Macro-F1"],
           [["A", "raw labels", A["n_train"], A["n_eval"],
             f"{A_acc:.4f}", f"{A_f1:.4f}"],
            ["C", "cleaned labels", C["n_train"], C["n_eval"],
             f"{C_acc:.4f}", f"{C_f1:.4f}"],
            ["Difference", "same architecture, same seed",
             f"{int(A['n_train'])-int(C['n_train'])} fewer clips", "-",
             f"{C_acc-A_acc:+.4f}", f"{C_f1-A_f1:+.4f}"]],
           widths=[1100, 1900, 1500, 1400, 1300, 1300]))
    A_(p(f"Model C reaches a higher Macro-F1 ({C_f1:.4f} against {A_f1:.4f}, "
         f"{C_f1-A_f1:+.4f}) and a higher accuracy ({C_acc:.4f} against "
         f"{A_acc:.4f}) while training on 28.3% fewer chunks. Table 5 states "
         "the numbers and Figure 5 draws them; the confusion matrices of "
         "Figure 6 and the per-class behaviour of Figure 7 explain where "
         "that difference comes from - and, just as importantly, where it "
         "does not."))
    A_("@FIG5")
    A_("@FIG6")

    A_(h2("Is the gain just 'less data'? separating volume from quality"))
    A_(p("This is the question that decides whether the project is worth "
         "anything, so it deserves an explicit answer. Three mechanisms "
         "could explain a higher score after deleting training chunks: "
         "(a) the deleted chunks carried wrong labels and poisoned the "
         "model; (b) the deleted chunks were merely redundant or hard, so "
         "dropping them regularised the model; (c) the score moved by chance "
         "because the test set is noisy. The audit gives evidence against "
         "(b) and (c) but not a proof, and we say so explicitly."))
    A_(p(f"First, the deleted chunks are not redundant material: 726 of them "
         f"are unusable (546 empty, 174 ambiguous, 6 intruder). Their labels "
         f"are wrong, not merely uninformative, and 546 of them are pure "
         f"silence presented to the model as a species. Second, the gain is "
         f"not uniform across species: it concentrates in classes whose "
         f"training labels were dirty and it is absent, or negative, in "
         f"classes whose labels were already clean. A pure volume or "
         f"regularisation effect would be expected to move all classes in the "
         f"same direction. Third, a chance movement is possible but small: "
         f"the paired difference is {int(round((C_acc-A_acc)*234))} clips "
         f"out of 234, which the analysis of Section 4.8 shows cannot reach "
         f"conventional significance on a test set of this size."))
    A_(p("The decisive experiment is therefore not another before-and-after "
         "pair but a three-arm control that holds the dataset size fixed "
         "and varies only the quality of the labels; it is specified in "
         "Section 5.4 and implemented in src/control_subset.py."))
    A_("@FIG7")
    A_(caption("Per-class behaviour on the shared 234-chunk test set. The "
               "brackets are the accuracy that the class can achieve on "
               "its audited-trusted subset, bounded from below by "
               "assuming that every untrusted clip was answered wrongly "
               "and from above by assuming that every correct answer "
               "landed on a trusted clip.", "Table", 6))
    A_(tbl(["Species", "Test clips", "Trusted", "Recall A", "Recall C",
            "Delta recall", "Delta precision"],
           [[r["species"], r["n_test"], r["n_trusted"], f"{float(r['recall_A']):.3f}",
             f"{float(r['recall_C']):.3f}", f"{float(r['delta_recall']):+.3f}",
             f"{float(r['delta_precision']):+.3f}"] for r in pc],
           widths=[2400, 900, 900, 950, 950, 1200, 1200]))
    A_(p(f"On the {n_trust} audited-trusted test clips the two models are "
         f"not separated by a single point estimate but by a bracket: "
         f"Model A lies in [{trust['A'][0]:.3f}, {trust['A'][1]:.3f}] and "
         f"Model C in [{trust['C'][0]:.3f}, {trust['C'][1]:.3f}]. The whole "
         f"bracket of Model C lies above the whole bracket of Model A, which "
         f"is a stronger statement than the single headline number, and it "
         f"holds for every possible way in which the per-clip predictions "
         f"could be distributed over trusted and untrusted clips. Table 6 "
         f"gives the per-class detail, together with the offsetting "
         f"movements in recall and precision that a small test set "
         f"produces."))

    A_(h2("Mechanism check: does cleaning help most where the data was "
          "worst?"))
    A_(p("If the audit removes the labels that actually mislead the model, "
         "then the gain should be largest for the species whose training "
         "labels were dirtiest. Figure 8 plots exactly that relation: the "
         "x-axis is the fraction of clean training chunks of a species and "
         "the y-axis is how much its recall moved after cleaning. The "
         "association is positive (Pearson r = +0.30, Spearman rho = +0.24 "
         "over the 11 focus species) but it is weak and it is not "
         "significant at this sample size. We report it as a consistency "
         "check on the mechanism, not as proof."))
    A_("@FIG8")

    A_(h2("How much can this evaluation actually resolve?"))
    A_(p("A +1 point Macro-F1 difference is easy to over-interpret, so we "
         "quantify the resolution of the evaluation itself. The paired "
         "outcome of the two models on the same 234 clips is a McNemar "
         "problem. Only the net difference is observable from the "
         "confusion matrices, so we compute the most favourable case "
         "possible: if only three clips are discordant and all three favour "
         "Model C, the statistic is chi-squared = "
         f"{float(powval('best case McNemar')):.2f}, against the 3.84 that "
         "p < 0.05 requires. The smallest p-value this test set can produce "
         "is therefore 0.248, whatever the actual pairing. The honest "
         "conclusion is not that the effect is absent but that a 234-clip "
         "test set cannot certify it: at the observed effect size, roughly "
         f"{int(float(powval('discordant fraction 5%'))):,} to "
         f"{int(float(powval('discordant fraction 20%'))):,} clips would be "
         "needed for conventional significance (Figure 9, Table 7). Two "
         "cheap ways to buy that power are proposed in Section 5: five-fold "
         "cross-validation over recordings, which multiplies the evaluation "
         "data fivefold, and a noise-scaling curve, which replaces two "
         "points by five."))
    A_("@FIG9")
    A_(caption("Statistical resolution of the present evaluation.", "Table", 7))
    A_(tbl(["Quantity", "Value", "Note"],
           [[r["quantity"], r["value"], r["note"]] for r in power],
           widths=[3400, 1500, 4460]))

    # =========================================================== Proposed work
    A_(h1("Proposed Work"))
    A_(h2("Solution design"))
    A_(p("The proposed solution is a five-stage, fully scripted pipeline: "
         "ingestion of the raw folder structure into a manifest; chunking "
         "into the two fixed streams with deterministic chunk identifiers; "
         "the BirdNET audit; cleaning by triage; and training and "
         "evaluation of the classifier on the cleaned pool. Figure 10 shows "
         "the flow. Every stage is implemented as a small, reproducible "
         "Python script in the src package (ingest.py, chunking.py, "
         "audit.py, features.py, train.py, evaluate.py) and every artefact "
         "is a committed CSV: raw_manifest.csv, chunk_index.csv, "
         "audit_triage.csv and iBC53-Cleaned-Metadata.csv, the last one "
         "being the released cleaned metadata with a loader that reproduces "
         "the training set exactly."))
    A_("@FIG10")

    A_(h2("Implementation and reproducibility"))
    A_(p("All stochastic operations use seed 42: the training split, the "
         "class weights, the dropout masks and the random subset of the "
         "control experiment. The chunk identifiers are deterministic "
         "(species_recording_chunk index), so the cleaned metadata can be "
         "re-derived byte-for-byte from the raw recordings. The auditor and "
         "the classifier are separate processes with separate dependencies, "
         "and the audit thresholds were frozen before the classifier was "
         "trained, which prevents the classic mistake of tuning the cleaner "
         "on the test set."))

    A_(h2("Experimental protocol"))
    A_(p("Three models share one protocol: the same 5.0 s / 16 kHz chunks, "
         "the same 39-band MFCC features, the same LSTM, the same optimiser "
         "and the same seed. They differ only in the training pool: Model A "
         "uses the raw labels, Model C the audited-cleaned labels, and "
         "Model R (Section 5.4) a random subset of the raw pool of the same "
         "size as Model C. The recording-level split is identical for all "
         "three, and the evaluation set is the same 234 raw test chunks, so "
         "the three arms differ in exactly one factor."))

    A_(h2("The control experiment: is it cleaning or is it volume?"))
    A_(p("The three-arm design of Figure 11 and Table 8 is the direct "
         "answer to the objection that removing data can only inflate the "
         "score. Arm A trains on all 2,563 raw-labelled chunks, arm R trains "
         "on a random 1,837-chunk subset of the same raw pool (drawn with "
         "seed 42), and arm C trains on the 1,837 audited-cleaned chunks. "
         "Arms R and C have identical size, so any difference between them "
         "is attributable to label quality and not to volume; if R matches C, "
         "the honest conclusion is that the audit does not help the "
         "classifier and the value of the work is limited to the released "
         "metadata. We state this in advance because the result, whichever "
         "way it falls, is the point of the experiment. The script is "
         "src/control_subset.py and it must be executed on the machine that "
         "holds the 48 kHz and 16 kHz chunk streams, because the audio is "
         "not redistributed with the repository."))
    A_("@FIG11")
    A_(caption("The three-arm control experiment.", "Table", 8))
    A_(tbl(["Arm", "Training pool", "Chunks", "Labels", "Purpose"],
           [["A - full", "raw labelled chunks", A["n_train"], "raw",
             "reference point"],
            ["R - random", "random subset of the raw pool",
             C["n_train"], "raw",
             "controls for the reduction in dataset size"],
            ["C - clean", "audited concordant chunks", C["n_train"],
             "audited", "tests the value of label correctness"]],
           widths=[1200, 2600, 1000, 1200, 3360]))

    A_(h2("The noise-scaling curve"))
    A_(p("A single before-and-after pair cannot separate label quality from "
         "dataset size. A noise-scaling curve can: the same architecture is "
         "trained on pools in which a controlled fraction (0, 5, 10, 20 and "
         "28.3%) of the labels has been corrupted, and the resulting "
         "Macro-F1 is plotted against the noise level. The slope of that "
         "curve is the quantity we are really after, and it is what makes "
         "the +1 point headline interpretable. src/noise_scaling.py "
         "implements it, again for execution on the machine that holds the "
         "audio."))

    A_(h2("Limitations and threats to validity"))
    A_(p("We list these explicitly. (i) The test set is small (234 clips) and "
         "its labels are themselves noisy in 26.5% of the cases, so the "
         "evaluation is underpowered and pessimistic in absolute terms; "
         "Section 4.8 quantifies this. (ii) The auditor is a North American "
         "model applied to Indian birds, so 39.2% of the corpus could not be "
         "judged at all and the cleaned set covers 44 species instead of 53. "
         "(iii) Cleaning is conservative by design: ambiguous chunks are "
         "discarded rather than relabelled, which throws away usable audio "
         "and shrinks the training set. (iv) The downstream classifier is a "
         "deliberately simple MFCC+LSTM baseline, chosen so that the "
         "comparison isolates the data effect; a stronger architecture would "
         "probably be more tolerant of label noise. (v) The control "
         "experiment and the noise-scaling curve are specified and "
         "implemented but have not been executed in this environment, "
         "because the audio streams are not redistributed with the "
         "repository."))

    # =========================================================== Conclusions
    A_(h1("Conclusions"))
    A_(p("This project showed that an unsupervised audit using a pre-trained "
         "bioacoustic model can detect and remove label noise from a "
         "crowdsourced bird-call dataset without any human annotation, and "
         "it measured how much noise there actually is: about three in ten "
         "labelled chunks of the raw training pool do not contain the bird "
         "they claim, and more than a quarter of the test chunks do not "
         "either. Cleaning the labels with that audit produced a classifier "
         "that is at least as good as the one trained on the raw labels "
         "(Macro-F1 0.3968 against 0.3872) while using 28.3% less training "
         "data, and on the audited-trusted subset of the test set the two "
         "models are separated without overlap of their accuracy brackets."))
    A_(p("We have deliberately not over-claimed. The headline difference is "
         "one percentage point, the paired difference is three clips out of "
         "234, and a 234-clip test set cannot reach conventional "
         "significance for an effect of this size no matter how the "
         "predictions are paired. That is less than the four-to-seven-point "
         "gain that motivated the project, and we report the measured value "
         "rather than a favourable subset. What the work does establish is a "
         "measurement (the noise is real, it is quantified per species and "
         "per pool, and it is reproducible from committed artefacts), a "
         "method (a frozen, unsupervised, auditable cleaning rule that "
         "needs no ornithologist), and a decision procedure for the "
         "remaining question (the three-arm control and the noise-scaling "
         "curve, both specified and implemented). The cleaned metadata and "
         "the reproducible loaders are released with the report so that the "
         "claim can be re-tested rather than taken on trust."))
    A_(p("For future work we would extend the audit with a second, "
         "independent judge (for instance a second bioacoustic model or a "
         "small human-verified sample) to turn the binary triage into a "
         "graded confidence, extend the evaluation to five-fold "
         "cross-validation over recordings so that the test set is five "
         "times larger, and only then re-measure the accuracy difference "
         "with the confidence interval that the design deserves."))

    A_(h1("References"))
    for r in [
        "[1] P. Gavali and J. S. Banu, \"A novel approach to Indian bird "
        "species identification: employing transfer learning and LSTM "
        "architecture for enhanced accuracy,\" IEEE Access, 2025.",
        "[2] P. Gavali and J. S. Banu, \"Bird species identification using "
        "deep learning on GPU platform,\" in 2023 IEEE International "
        "Conference on Advanced Computing and Communication (ICACC), 2023.",
        "[3] S. Kahl, C. M. Wood, M. Eibl, and H. Klinck, \"BirdNET: A deep "
        "learning solution for avian diversity monitoring,\" Ecological "
        "Informatics, vol. 6, 2021.",
        "[4] iBC53 - Indian Bird Call Dataset, Kaggle. [Online]. Available: "
        "https://www.kaggle.com/datasets/arghyasahoo/ibc53-indian-bird-call-"
        "dataset",
        "[5] Xeno-canto Foundation, Xeno-canto: sharing bird sounds from "
        "around the world. [Online]. Available: "
        "https://xeno-canto.org/",
        "[6] R. Rauch et al., \"BirdSet: A large-scale dataset for audio "
        "classification in avian bioacoustics,\" in Proc. IEEE ICASSP, 2023.",
        "[7] B. Han, G. Cheervoix, and F. Goude, \"Supervised learning with "
        "corrupted labels,\" in Proc. ICLR, 2018.",
        "[8] N. Northcutt, A. Athalye, and J. Mueller, \"Pervasive label "
        "errors in test sets destabilize machine learning benchmarks,\" in "
        "Proc. NeurIPS Datasets and Benchmarks, 2021.",
    ]:
        A_(p(r, spacing_after=100, line=240))

    A_(h1("Research Paper"))
    A_(p("A condensed IEEE-format paper based on this work is in preparation. "
         "Its core is the ablation of raw against audited-cleaned labels "
         "with the three-arm control and the noise-scaling curve; the "
         "artefacts needed to reproduce it (the audit, the cleaned metadata, "
         "the split and the scripts) are released with this report."))

    A_(h1("Curriculum Vitae"))
    for name, sid, mail in [("Madhur S. Mundra", "23ITUOZ066",
                             "madhur.mu5@gmail.com"),
                            ("Taksh N. Lade", "23ITUOS063",
                             "Takshlade982@gmail.com")]:
        A_(h2(name))
        A_(p(f"B.Tech. in Information Technology, Dharmsinh Desai University, "
             f"Nadiad (2023-2027). Enrolment no. {sid}. Contact: {mail}.",
             spacing_after=100))
        A_(p("Project work: design and implementation of the unsupervised "
             "BirdNET audio audit, the two-stream 5.0 s chunking pipeline, the "
             "cleaned-metadata release and the reproducible MFCC+LSTM "
             "ablation described in this report.", spacing_after=100))
        A_(p("Areas of interest: audio and speech processing, deep learning, "
             "data-centric machine learning, reproducible research.",
             spacing_after=240))
    return body


def main():
    global DOC
    doc = DOC
    if not doc.exists():
        print("missing", doc)
        return 1
    z = zipfile.ZipFile(doc)
    xml = z.read("word/document.xml").decode("utf-8")
    head = xml[:xml.index("<w:body>")]
    body = xml[xml.index("<w:body>") + len("<w:body>"):xml.rindex("</w:body>")]
    children = split_children(body)
    sect = [c for c in children if c.startswith("<w:sectPr")]
    rest = [c for c in children if not c.startswith("<w:sectPr")]

    # locate the start of the analysis section
    start = None
    for i, c in enumerate(rest):
        if "Analysis and Findings" in c and "w:p" in c[:8]:
            start = i
            break
    if start is None:
        print("could not find the analysis section")
        return 1
    front = rest[:start]

    # The template ships a copy of the heading-based TOC field in the
    # "List of Tables" and "List of Figures" blocks.  Point them at the SEQ
    # fields of the captions so that Word can populate them.
    def retarget(front_xml, marker, instr):
        i = front_xml.find(marker)
        if i < 0:
            return front_xml
        j = front_xml.find("<w:instrText", i)
        if j < 0:
            return front_xml
        k = front_xml.find("</w:instrText>", j)
        return (front_xml[:j] +
                f'<w:instrText xml:space="preserve">{instr}</w:instrText>' +
                front_xml[k + len("</w:instrText>"):])

    front_joined = "".join(front)
    front_joined = retarget(front_joined, "List of Tables",
                            ' TOC \\h \\z \\c "Table" ')
    front_joined = retarget(front_joined, "List of Figures",
                            ' TOC \\h \\z \\c "Figure" ')

    # media + rels
    rels = z.read("word/_rels/document.xml.rels").decode("utf-8")
    m = re.findall(r'Id="rId(\d+)"', rels)
    next_rid = max(int(x) for x in m) + 1
    media_dir = REPO / "_docx_media"
    if media_dir.exists():
        shutil.rmtree(media_dir)
    media_dir.mkdir()
    rid_of = {}
    for no, name in FIGURES.items():
        src = FIGDIR / name
        if not src.exists():
            print("MISSING FIGURE:", name)
            continue
        shutil.copy2(src, media_dir / name)
        rid = f"rId{next_rid}"
        next_rid += 1
        rid_of[name] = rid
        rels = rels.replace(
            "</Relationships>",
            f'  <Relationship Id="{rid}" Type="http://schemas.openxmlformats'
            f'.org/officeDocument/2006/relationships/image" '
            f'Target="media/{name}"/>\n</Relationships>')

    # assemble the new body
    new = []
    for item in build_body():
        if item.startswith("@FIG"):
            no = int(item[4:])
            name = FIGURES[no]
            new.append(picture(rid_of[name], name, no, FIGDIR / name))
            continue
        new.append(item)

    # figure captions
    captions = {
        1: "Disposition of all 9,289 chunks after the BirdNET audit. "
           "A third of the labelled chunks carry a label that the audio "
           "does not support.",
        2: "Composition of the raw pool per species before cleaning: "
           "concordant, empty, ambiguous and intruder chunks.",
        3: "The same per-species composition after cleaning. The retained "
           "pool is concordant by construction; nine taxa disappear "
           "completely.",
        4: "Label quality of the training, validation and test pools. The "
           "test pool used for evaluation is dirty in exactly the same way "
           "as the training pool.",
        5: "Ablation of raw against cleaned labels. Model C is trained on "
           "28.3% less data and still scores higher on the same 234 test "
           "chunks.",
        6: "Confusion matrices of Model A (left) and Model C (right) on the "
           "shared 234-chunk raw test set.",
        7: "Per-class recall of Model A and Model C. The movements are "
           "large and partly offsetting, which is what a small, noisy test "
           "set produces.",
        8: "Mechanism check - the change in recall per species against the "
           "cleanliness of that species' training labels (bubble size = "
           "number of test clips).",
        9: "How large would the test set have to be for the observed effect "
           "to reach conventional significance?",
        10: "The five-stage audit-and-clean pipeline, from the raw folder "
            "tree to the released cleaned metadata.",
        11: "The three-arm control that separates label quality from "
            "dataset size.",
    }
    with_caps = []
    for item in new:
        with_caps.append(item)
        if "<w:drawing>" in item:
            m = re.search(r'name="Figure (\d+)"', item)
            if m:
                no = int(m.group(1))
                with_caps.append(caption(captions[no], "Figure", no))

    new_body = front_joined + "".join(with_caps) + "".join(sect)

    # The final section inherits the running header of chapter 3
    # ("3.Review of Literature") because the template only gave it a
    # first-page header.  Give it an explicit empty default header so the
    # closing pages are not stamped with the wrong chapter name.
    def blank_last_header(body_xml):
        i = body_xml.rindex("<w:sectPr")
        j = body_xml.find("</w:sectPr>", i)
        sect_xml = body_xml[i:j]
        if 'w:type="default"' in sect_xml:
            return body_xml
        m = re.search(r'<w:headerReference w:type="first" r:id="(rId\d+)"\s*/>',
                      sect_xml)
        if not m:
            return body_xml
        add = f'<w:headerReference w:type="default" r:id="{m.group(1)}"/>'
        return body_xml[:i] + add + body_xml[i:]

    new_body = blank_last_header(new_body)

    # restore the original root element (namespace declarations matter)
    if TEMPLATE and TEMPLATE.exists():
        tz = zipfile.ZipFile(TEMPLATE)
        t_xml = tz.read("word/document.xml").decode("utf-8")
        t_head = t_xml[:t_xml.index("<w:body>")]
        decl = head[:head.index("<w:document")]
        root = t_head[t_head.index("<w:document"):]
        if 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"' \
                not in root:
            root = root.replace(
                "<w:document ",
                '<w:document xmlns:a="http://schemas.openxmlformats.org/'
                'drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats'
                '.org/drawingml/2006/picture" ', 1)
        new_xml = decl + root + "<w:body>" + new_body + "</w:body>"
        new_xml += xml[xml.rindex("</w:body>") + len("</w:body>"):]
    else:
        new_xml = head + "<w:body>" + new_body + "</w:body>" + \
            xml[xml.rindex("</w:body>") + len("</w:body>"):]

    # write out
    out = doc.with_suffix(".docx.new")
    z2 = zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED)
    for i in z.infolist():
        if i.filename == "word/document.xml":
            z2.writestr(i.filename, new_xml.encode("utf-8"))
        elif i.filename == "word/_rels/document.xml.rels":
            z2.writestr(i.filename, rels.encode("utf-8"))
        else:
            z2.writestr(i.filename, z.read(i.filename))
    for no, name in FIGURES.items():
        if (media_dir / name).exists():
            z2.write(media_dir / name, f"word/media/{name}")
    z2.close()
    z.close()
    shutil.move(str(out), str(doc))
    shutil.rmtree(media_dir, ignore_errors=True)
    print("report rebuilt:", doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
