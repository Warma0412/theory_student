#!/usr/bin/env python3
"""Build standalone SUFE LaTeX thesis sources in the final-delivery folder."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve()
PROJECT_DIR = HERE.parents[3]
PAPER_DIR = PROJECT_DIR / "论文"
V5_DIR = PAPER_DIR / "v5"
SOURCE_MD = V5_DIR / "论文_v5.md"
TEMPLATE_DIR = PROJECT_DIR / "模板" / "应用统计专硕latex模板"
OUTPUT_DIR = V5_DIR / "latex工程" / "论文"
FIGURES_DIR = OUTPUT_DIR / "figures"
BIB_DIR = OUTPUT_DIR / "bib"


def section(text: str, start: str, end: str) -> str:
    return text.split(start, maxsplit=1)[1].split(end, maxsplit=1)[0].strip()


def normalize_markdown(markdown: str) -> str:
    lines = markdown.splitlines()
    normalized: list[str] = []
    for line in lines:
        chapter = re.match(r"^# 第\d+章\s+(.+)$", line)
        heading = re.match(r"^(#{2,4})\s+\d+(?:\.\d+){1,3}\s+(.+)$", line)
        table_caption = re.match(r"^\*\*表\d+(?:-\d+)?\s+(.+)\*\*$", line)
        if chapter:
            normalized.append("# " + chapter.group(1))
        elif heading:
            normalized.append(f"{heading.group(1)} {heading.group(2)}")
        elif table_caption:
            normalized.append("Table: " + table_caption.group(1))
        else:
            normalized.append(
                re.sub(
                    r"!\[图\d+(?:-\d+)?\s*([^\]]*)\]",
                    lambda match: f"![{match.group(1)}]",
                    line,
                )
            )
    return "\n".join(normalized)


def pandoc_latex(markdown: str, top_level_chapter: bool = False) -> str:
    command = [
        "pandoc",
        "--from=markdown+tex_math_dollars",
        "--to=latex",
        "--wrap=none",
        "--no-highlight",
    ]
    if top_level_chapter:
        command.append("--top-level-division=chapter")
    result = subprocess.run(
        command,
        input=normalize_markdown(markdown),
        text=True,
        capture_output=True,
        check=True,
    )
    latex = result.stdout
    latex = re.sub(r"\\label\{[^{}]+\}", "", latex)
    latex = re.sub(
        r"\{\\def\\LTcaptype\{none\} % do not increment counter\n"
        r"(\\begin\{longtable\}.*?\\end\{longtable\})\n\}",
        r"\1",
        latex,
        flags=re.DOTALL,
    )
    latex = latex.replace(
        r"\texttt{α\_kn=0.10}",
        r"\(\alpha_{\mathrm{kn}}=0.10\)",
    )
    latex = latex.replace(
        r"\texttt{α\_eBH=0.20}",
        r"\(\alpha_{\mathrm{eBH}}=0.20\)",
    )
    latex = re.sub(
        r"(?<![A-Za-z])([A-Za-z][A-Za-z0-9]*(?:\\_[A-Za-z0-9]+)+"
        r"(?:\.[A-Za-z0-9]+)?)",
        lambda match: r"\path{" + match.group(1).replace(r"\_", "_") + "}",
        latex,
    )
    latex = re.sub(
        r"(?<![0-9a-f])([0-9a-f]{64})(?![0-9a-f])",
        lambda match: (
            r"{\scriptsize\seqsplit{" + match.group(1) + "}}"
        ),
        latex,
    )
    return latex.strip()


def escape_reference(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    escaped = "".join(replacements.get(character, character) for character in text)
    escaped = re.sub(
        r"(https?://[^\s]+)",
        lambda match: r"\url{" + match.group(1).rstrip(".") + "}"
        + ("." if match.group(1).endswith(".") else ""),
        escaped,
    )
    return escaped


def bibliography_latex(reference_markdown: str) -> str:
    entries = re.findall(
        r"^\[(\d+)\]\s+(.+?)(?=\n\n\[\d+\]|\Z)",
        reference_markdown,
        flags=re.MULTILINE | re.DOTALL,
    )
    lines = [r"\begin{thebibliography}{99}"]
    for number, entry in entries:
        compact = " ".join(entry.split())
        lines.append(rf"\bibitem{{ref{number}}} {escape_reference(compact)}")
    lines.append(r"\end{thebibliography}")
    if len(entries) != 45:
        raise ValueError(f"Expected 45 references, found {len(entries)}")
    return "\n".join(lines)


def extract_abstracts(markdown: str) -> tuple[str, str, str, str]:
    chinese_block = section(markdown, "# 摘要\n", "# Abstract")
    chinese, keywords_cn = chinese_block.split("**关键词：**", maxsplit=1)
    english_block = section(markdown, "# Abstract\n", "# 第1章")
    english, keywords_en = english_block.split("**Key words:**", maxsplit=1)
    return (
        pandoc_latex(chinese.strip()),
        keywords_cn.strip(),
        pandoc_latex(english.strip()),
        keywords_en.strip(),
    )


def extract_body(markdown: str) -> str:
    body = "# 第1章 " + section(markdown, "# 第1章", "# 参考文献")
    return pandoc_latex(body, top_level_chapter=True)


def extract_appendices(markdown: str) -> str:
    appendix_text = markdown.split("# 附录A", maxsplit=1)[1]
    appendix_text = "# 附录A" + appendix_text
    matches = list(
        re.finditer(
            r"^# (附录[A-Z]\s+[^\n]+)\n(.*?)(?=^# 附录[A-Z]|\Z)",
            appendix_text,
            flags=re.MULTILINE | re.DOTALL,
        )
    )
    blocks = []
    for match in matches:
        title = match.group(1)
        content = pandoc_latex(match.group(2).strip())
        if title.startswith("附录A"):
            blocks.append(
                "\\begin{landscape}\n"
                + rf"\chapterx{{{title}}}"
                + "\n"
                + rf"\chapterxname{{{title}}}"
                + "\n\n\\small\n"
                + content
                + "\n\\end{landscape}"
            )
        else:
            blocks.append(
                rf"\chapterx{{{title}}}"
                + "\n"
                + rf"\chapterxname{{{title}}}"
                + "\n\n"
                + content
            )
    return "\n\n".join(blocks)


def copy_assets() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    BIB_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TEMPLATE_DIR / "sufethesismas.cls", OUTPUT_DIR)
    shutil.copy2(
        TEMPLATE_DIR / "bib" / "gbt7714-numerical.bst",
        BIB_DIR,
    )
    for name in ["SHUFEBadge.jpg", "SHUFEName.jpg", "statement.pdf"]:
        shutil.copy2(TEMPLATE_DIR / "figures" / name, FIGURES_DIR)
    used_figures = [
        "fig01_monthly_gmv.png",
        "fig02_target_distribution.png",
        "fig03_knockoff_evalues.png",
        "fig04_selection_frequency.png",
        "fig06_xgboost_shap.png",
        "fig10_predictive_models.png",
        "fig12_cluster_group.png",
        "fig13_exchangeability.png",
        "fig14_grip_importance.png",
        "fig15_weekly_gmv.png",
        "fig16_weekly_selection.png",
        "fig17_monthly_weekly_frequency.png",
        "fig18_weekly_prediction.png",
    ]
    for name in used_figures:
        shutil.copy2(V5_DIR / "figures" / name, FIGURES_DIR / name)


def build_tex() -> str:
    markdown = SOURCE_MD.read_text(encoding="utf-8")
    abstract_cn, keywords_cn, abstract_en, keywords_en = extract_abstracts(
        markdown
    )
    body = extract_body(markdown)
    references = section(markdown, "# 参考文献\n", "# 附录A")
    appendices = extract_appendices(markdown)
    return rf"""% !TEX program = xelatex
\documentclass{{sufethesismas}}

\usepackage{{fontspec}}
\usepackage{{longtable}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{calc}}
\usepackage{{url}}
\usepackage{{seqsplit}}
\usepackage{{microtype}}
\usepackage{{etoolbox}}
\usepackage{{pdflscape}}
\setlength{{\LTleft}}{{0pt}}
\setlength{{\LTright}}{{0pt}}
\providecommand{{\tightlist}}{{\setlength{{\itemsep}}{{0pt}}\setlength{{\parskip}}{{0pt}}}}
\makeatletter
\def\maxwidth{{\ifdim\Gin@nat@width>\linewidth\linewidth\else\Gin@nat@width\fi}}
\def\maxheight{{\ifdim\Gin@nat@height>.72\textheight .72\textheight\else\Gin@nat@height\fi}}
\makeatother
\setkeys{{Gin}}{{width=\maxwidth,height=\maxheight,keepaspectratio}}
\providecommand{{\pandocbounded}}[1]{{#1}}
\renewcommand{{\arraystretch}}{{1.15}}
\urlstyle{{same}}

\begin{{document}}
\classification{{}}
\confidential{{}}
\UDC{{}}
\serialnumber{{}}
\title{{基于去随机化Model-X Knockoff的电商GMV多粒度受控维度选择研究}}
\englishtitle{{Multi-granularity Controlled Dimension Selection for E-commerce GMV Using Derandomized Model-X Knockoffs}}
\author{{吴优}}
\advisor{{张吕欧}}
\major{{应用统计硕士专业学位论文}}
\completedate{{2026年9月}}
\department{{统计与数据科学学院}}
\school{{上海财经大学}}
\studentidnumber{{2025213385}}

\maketitle

\frontmatter
\begin{{abstractCN}}
{abstract_cn}
\end{{abstractCN}}
\keywordsCN{{{keywords_cn}}}

\begin{{abstractEN}}
{abstract_en}
\end{{abstractEN}}
\keywordsEN{{{keywords_en}}}

\tableofcontents

\mainmatter
{body}

\backmatter
{bibliography_latex(references)}

{appendices}

\chapterx{{致谢}}
感谢指导教师在研究选题、统计方法与论文写作方面提供的指导，感谢学院教师和同学在学习与研究过程中给予的帮助。感谢Olist公开匿名数据，使本研究能够完成可复核的实证分析。论文中的不足由作者承担。

\end{{document}}
"""


def main() -> None:
    copy_assets()
    output = OUTPUT_DIR / "论文.tex"
    output.write_text(build_tex(), encoding="utf-8")
    print(f"Generated {output}")


if __name__ == "__main__":
    main()
