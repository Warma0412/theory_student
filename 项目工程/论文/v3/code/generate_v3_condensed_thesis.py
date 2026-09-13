#!/usr/bin/env python3
"""Generate the concise V3 thesis PDF."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from generate_v3_thesis import V1_DIR, V3_DIR, build_context, markdown_table


TEMPLATE = Path(__file__).with_name("thesis_condensed_template_v3.md")
OUTPUT_PDF = V3_DIR.parent / "论文_v3_精简版.pdf"


def build_markdown() -> str:
    context = build_context()
    primary = pd.read_csv(V1_DIR / "results" / "knockoff_primary.csv")
    strict = primary.loc[
        primary["q"].eq(0.20) & primary["selected_ebh"].astype(bool)
    ].copy()
    strict = strict.sort_values(
        ["selection_frequency", "mean_evalue", "mean_w"],
        ascending=False,
    )
    strict["frequency_fmt"] = strict["selection_frequency"].map(
        lambda value: f"{value:.1%}"
    )
    context["STRICT_COMPACT_TABLE"] = markdown_table(
        strict,
        ["label_zh", "mean_w", "frequency_fmt", "mean_evalue"],
        ["变量", "平均W", "单轮入选频率", "平均e-value"],
    )
    text = TEMPLATE.read_text(encoding="utf-8")
    for key, value in context.items():
        text = text.replace("{{" + key + "}}", str(value))
    unresolved = sorted(set(re.findall(r"\{\{[A-Z0-9_]+\}\}", text)))
    if unresolved:
        raise ValueError(f"Unresolved V3 condensed placeholders: {unresolved}")
    return text


def write_css(path: Path) -> None:
    path.write_text(
        """
@page { size: A4; margin: 23mm 23mm 23mm 28mm;
  @bottom-center { content: counter(page); font-size: 9pt; } }
@page:first { @bottom-center { content: ""; } }
body { font-family: "Songti SC", "STSong", serif; font-size: 11.5pt;
  line-height: 1.65; color: #111; max-width: 164mm; margin: 0 auto; }
h1 { font-family: "Heiti SC", sans-serif; text-align: center; font-size: 18pt;
  page-break-before: always; margin-top: 0; }
h2 { font-family: "Heiti SC", sans-serif; font-size: 14pt; margin-top: 1.1em;
  break-after: avoid-page; }
h3 { font-family: "Heiti SC", sans-serif; font-size: 12pt;
  break-after: avoid-page; }
p { text-align: justify; text-indent: 2em; margin: 0.3em 0; }
p:has(+ table) { break-after: avoid-page; }
li p, td p, th p { text-indent: 0; }
ul, ol { margin-top: 0.25em; margin-bottom: 0.5em; }
a { color: #111; text-decoration: none; }
table { border-collapse: collapse; width: 100%; font-size: 8.2pt;
  margin: 0.7em 0; page-break-inside: avoid; break-inside: avoid-page; }
tr { page-break-inside: avoid; }
th, td { border: 1px solid #555; padding: 4px 6px; vertical-align: top; }
th { background: #f0f0f0; }
img { display: block; max-width: 94%; max-height: 195mm; margin: 0.8em auto;
  page-break-inside: avoid; }
.title { font-family: "Heiti SC", sans-serif; font-size: 23pt;
  text-align: center; margin-top: 52mm; }
.subtitle { text-align: center; font-size: 16pt; }
.author, .date { text-align: center; text-indent: 0; }
#TOC { page-break-before: always; page-break-after: always; }
#TOC::before { content: "目录"; display: block; font-family: "Heiti SC", sans-serif;
  font-size: 18pt; font-weight: bold; text-align: center; margin-bottom: 1.2em; }
""".strip(),
        encoding="utf-8",
    )


def main() -> None:
    text = build_markdown()
    chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    with tempfile.TemporaryDirectory(prefix="olist-v3-condensed-") as temporary:
        workdir = Path(temporary)
        markdown_path = workdir / "论文_v3_精简版.md"
        html_path = workdir / "论文_v3_精简版.html"
        pdf_path = workdir / "论文_v3_精简版.pdf"
        css_path = workdir / "condensed.css"
        markdown_path.write_text(text, encoding="utf-8")
        write_css(css_path)
        subprocess.run(
            [
                "pandoc",
                str(markdown_path),
                "--from=markdown+tex_math_dollars",
                "--standalone",
                "--toc",
                "--toc-depth=2",
                "--mathml",
                "--embed-resources",
                f"--css={css_path}",
                f"--resource-path={workdir}:{V3_DIR}:{V1_DIR}",
                "-o",
                str(html_path),
            ],
            cwd=workdir,
            check=True,
        )
        subprocess.run(
            [
                str(chrome),
                "--headless",
                "--disable-gpu",
                "--allow-file-access-from-files",
                "--no-pdf-header-footer",
                f"--print-to-pdf={pdf_path}",
                html_path.resolve().as_uri(),
            ],
            cwd=workdir,
            check=True,
        )
        shutil.copy2(pdf_path, OUTPUT_PDF)
    print(f"Generated {OUTPUT_PDF} ({OUTPUT_PDF.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
