"""Compile the empirical report and mathematical appendix."""

import json
import re
import subprocess
import pymupdf
from paired_loss import ROOT

source = (ROOT / "研究报告.md").read_text()
title, source = source.split("\n", 1)
source = re.sub(r"^## \d+\.\s+", "# ", source, flags=re.M)
source = re.sub(r"^### ", "## ", source, flags=re.M)
proof = (ROOT / "方法定义与证明边界.md").read_text().split("\n", 1)[1]
proof = re.sub(r"^## \d+\.\s+", "## ", proof, flags=re.M)
source += "\n\n# 附录：方法定义与证明边界\n\n" + proof
latex = subprocess.run(
    ["pandoc", "-f", "markdown+tex_math_dollars-autolink_bare_uris", "-t", "latex", "--wrap=none", "--no-highlight"],
    input=source, text=True, capture_output=True, check=True,
).stdout
def size_table(match):
    table = match.group(0)
    preamble, rest = table.split(r"\toprule", 1)
    widths = list(re.finditer(r"\\real\{[0-9.]+\}", preamble))
    count = len(widths)
    if count in (7, 8):
        weights = ([.24] + [.76 / (count - 1)] * (count - 1))
        for item, weight in reversed(list(zip(widths, weights))):
            preamble = preamble[:item.start()] + "\\real{" + f"{weight:.6f}" + "}" + preamble[item.end():]
    return "{\n\\footnotesize\n" + preamble + r"\toprule" + rest + "\n}"

latex = re.sub(r"(\\begin\{longtable\}.*?\\end\{longtable\})",
               size_table, latex, flags=re.S)
header = r"""\documentclass[UTF8,11pt,a4paper]{ctexart}
\usepackage[left=1.6cm,right=1.6cm,top=2cm,bottom=2cm]{geometry}
\usepackage{amsmath,amssymb,graphicx,longtable,booktabs,array,calc,hyperref,xurl,setspace,fancyvrb,needspace}
\setmainfont{Times New Roman}
\setmonofont{Menlo}
\setCJKmainfont{Songti SC}
\setCJKmonofont{Songti SC}
\hypersetup{hidelinks}
\setlength{\LTleft}{0pt plus 1fill}
\setlength{\LTright}{0pt plus 1fill}
\setlength{\emergencystretch}{3em}
\setlength{\tabcolsep}{4pt}
\setstretch{1.12}
\AddToHook{cmd/section/before}{\Needspace{7\baselineskip}}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\pandocbounded}[1]{#1}
\begin{document}
"""
tex = header + r"\begin{center}\Large\bfseries " + title.removeprefix("# ") + r"\end{center}" + latex + "\n\\end{document}\n"
(ROOT / "研究报告.tex").write_text(tex)
proc = subprocess.run(["tectonic", "--keep-logs", "研究报告.tex"], cwd=ROOT, capture_output=True, text=True)
(ROOT / "logs/report_build.log").write_text(proc.stdout + proc.stderr)
proc.check_returncode()
doc = pymupdf.open(ROOT / "研究报告.pdf")
outside = []
for i, page in enumerate(doc):
    assert page.get_text().strip(), f"Empty page {i}"
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                x0, y0, x1, y1 = span["bbox"]
                if span["text"].strip() and (x0 < -1 or y0 < -1 or x1 > page.rect.width + 1 or y1 > page.rect.height + 1):
                    outside.append({"page": i + 1, "text": span["text"]})
    if i in (0, 1, 2, len(doc) - 2, len(doc) - 1):
        page.get_pixmap(matrix=pymupdf.Matrix(1.25, 1.25)).save(ROOT / "results" / f"report_page_{i+1}.png")
(ROOT / "results/pdf_check.json").write_text(json.dumps({
    "pages": len(doc), "outside": outside,
    "no_question_references": "??" not in "\n".join(p.get_text() for p in doc)},
    indent=2, ensure_ascii=False))
assert not outside, outside
print("PDF checked", len(doc), "pages")
