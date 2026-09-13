"""Compile an experiment report, not a replacement thesis or submission."""

import json
from pathlib import Path
import re
import subprocess

import pymupdf

from run_study import STUDY, RESULT

source = (STUDY / "实验报告.md").read_text()
reference = (STUDY / "文献与新颖性边界.md").read_text()
title, source = source.split("\n", 1)
source = re.sub(r"^## \d+\.\s+", "# ", source, flags=re.M)
source = re.sub(r"^### ", "## ", source, flags=re.M)
source += "\n\n# 附录：文献与新颖性\n\n" + reference.split("\n", 1)[1]
latex = subprocess.run(
    ["pandoc", "-f", "markdown+tex_math_dollars+autolink_bare_uris", "-t", "latex", "--wrap=none"],
    input=source, text=True, capture_output=True, check=True,
).stdout
latex = re.sub(r"(\\begin\{longtable\}.*?\\end\{longtable\})",
               r"{\n\\footnotesize\n\1\n}", latex, flags=re.S)
tex = r"""\documentclass[UTF8,11pt,a4paper]{ctexart}
\usepackage[left=1.7cm,right=1.7cm,top=2cm,bottom=2cm]{geometry}
\usepackage{amsmath,amssymb,graphicx,longtable,booktabs,array,calc,hyperref,xurl,seqsplit,setspace}
\setmainfont{Times New Roman}
\setCJKmainfont{Songti SC}
\hypersetup{hidelinks}
\setlength{\LTleft}{0pt plus 1fill}
\setlength{\LTright}{0pt plus 1fill}
\setlength{\emergencystretch}{3em}
\setstretch{1.15}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\pandocbounded}[1]{#1}
\begin{document}
""" + r"\begin{center}\Large\bfseries " + title.removeprefix("# ") + r"\end{center}" + latex + "\n\\end{document}\n"
(STUDY / "实验报告.tex").write_text(tex)
completed = subprocess.run(["tectonic", "--keep-logs", "实验报告.tex"],
                           cwd=STUDY, text=True, capture_output=True)
(STUDY / "logs/report_build.log").write_text(completed.stdout + completed.stderr)
completed.check_returncode()
doc = pymupdf.open(STUDY / "实验报告.pdf")
text = "\n".join(p.get_text() for p in doc)
assert "??" not in text
outside = []
for i, page in enumerate(doc):
    for b in page.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            for span in line["spans"]:
                x0, y0, x1, y1 = span["bbox"]
                if span["text"].strip() and (x0 < -1 or y0 < -1 or x1 > page.rect.width + 1 or y1 > page.rect.height + 1):
                    outside.append({"page": i + 1, "text": span["text"]})
    if i in (0, 1, 2, 3, len(doc) - 1):
        page.get_pixmap(matrix=pymupdf.Matrix(1.25, 1.25)).save(RESULT / f"report_page_{i+1}.png")
(RESULT / "pdf_check.json").write_text(json.dumps(
    {"pages": len(doc), "outside": outside, "no_unresolved_references": True}, ensure_ascii=False, indent=2))
assert not outside, outside
print("Report PDF checked:", len(doc), "pages")
