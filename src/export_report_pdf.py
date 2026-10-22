"""Open the report in Word, refresh the TOC / figure / table lists, save the
.docx and export the matching PDF.

    python src/export_report_pdf.py

Word (Microsoft Office) must be installed.  The PDF is written next to the
.docx with the same base name and, if a second path is given, also copied to
that path.
"""
import os
import shutil
import sys
from pathlib import Path

import win32com.client as win32

REPO = Path(__file__).resolve().parents[1]
DOCX = REPO / "Report Format - Project2- Sem7 .docx"
PDF = REPO / "Report Format - Project2- Sem7.pdf"
EXTRA_PDF = Path(sys.argv[1]) if len(sys.argv) > 1 else None


def main() -> int:
    word = win32.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    doc = None
    try:
        doc = word.Documents.Open(str(DOCX), ReadOnly=False,
                                  AddToRecentFiles=False)
        # refresh the TOC and the list of figures / tables
        for _ in range(2):
            for i in range(1, doc.TablesOfContents.Count + 1):
                doc.TablesOfContents(i).Update()
            doc.Fields.Update()
        doc.Repaginate()
        pages = doc.ComputeStatistics(2)
        print(f"pages={pages} words={doc.ComputeStatistics(0)} "
              f"paragraphs={doc.Paragraphs.Count} "
              f"figures={doc.InlineShapes.Count} tables={doc.Tables.Count}")
        doc.Save()
        doc.SaveAs2(str(PDF), FileFormat=17)  # wdFormatPDF
        print("pdf written:", PDF)
        if EXTRA_PDF:
            shutil.copy2(PDF, EXTRA_PDF)
            print("pdf copied to:", EXTRA_PDF)
    except Exception as e:
        print("FAILED:", e)
        return 1
    finally:
        if doc is not None:
            doc.Close(0)
        word.Quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
