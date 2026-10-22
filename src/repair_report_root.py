"""Repair the report docx: restore the original <w:document> root element.

The rebuilt document.xml dropped the namespace declarations of the original
template (wp, w15, w16se, wp14, ...) while mc:Ignorable still references them.
Unbound prefixes in mc:Ignorable make Word refuse the file ("the file appears
to be corrupted").  This script takes the root element from the pristine
template and puts it back in front of the rebuilt body.
"""
import re
import shutil
import sys
import zipfile

TEMPLATE = sys.argv[1]      # pristine blank template
TARGET = sys.argv[2]        # docx to repair


def head_of(path):
    z = zipfile.ZipFile(path)
    x = z.read("word/document.xml").decode("utf-8")
    return x[:x.index("<w:body>")]


def repair(target, template):
    tz = zipfile.ZipFile(template)
    t_head = head_of(template)
    z = zipfile.ZipFile(target)
    doc = z.read("word/document.xml").decode("utf-8")
    old_head = doc[:doc.index("<w:body>")]
    # keep declaration line from target, use template root element
    decl = old_head[:old_head.index("<w:document")]
    new_root = t_head[t_head.index("<w:document"):] + "<w:body>"
    # make sure the drawingml namespaces used by inline pictures are declared
    if 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"' not in new_root:
        new_root = new_root.replace(
            "<w:document ",
            '<w:document xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture" ',
            1)
    new_doc = decl + new_root + doc[doc.index("<w:body>") + len("<w:body>"):]
    out = target + ".repaired"
    z2 = zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED)
    for i in z.infolist():
        if i.filename == "word/document.xml":
            z2.writestr(i.filename, new_doc.encode("utf-8"))
        else:
            z2.writestr(i.filename, z.read(i.filename))
    z2.close()
    z.close()
    shutil.move(out, target)
    print("root element restored ->", target)


if __name__ == "__main__":
    repair(TARGET, TEMPLATE)
