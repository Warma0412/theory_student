#!/usr/bin/env python3
"""Validate the final SUFE thesis and proposal presentation deliverables."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve()
PROJECT_DIR = HERE.parents[3]
FINAL_DIR = PROJECT_DIR / "【最终版】" / "v6"
ENGINE_DIR = HERE.parents[1] / "latex工程"
THESIS_DIR = ENGINE_DIR / "论文"
REPORT_DIR = ENGINE_DIR / "开题报告"
SLIDES_DIR = ENGINE_DIR / "开题汇报"


def command_output(arguments: list[str]) -> str:
    return subprocess.run(
        arguments,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def pdf_metadata(path: Path) -> dict:
    info = command_output(["pdfinfo", str(path)])
    page_match = re.search(r"^Pages:\s+(\d+)$", info, re.MULTILINE)
    size_match = re.search(
        r"^Page size:\s+([\d.]+) x ([\d.]+) pts", info, re.MULTILINE
    )
    if not page_match or not size_match:
        raise ValueError(f"Unable to parse PDF metadata: {path}")
    return {
        "pages": int(page_match.group(1)),
        "width_points": float(size_match.group(1)),
        "height_points": float(size_match.group(2)),
    }


def main() -> None:
    expected = [
        ENGINE_DIR / "README.md",
        ENGINE_DIR / "build.sh",
        THESIS_DIR / "论文.tex",
        THESIS_DIR / "论文.pdf",
        THESIS_DIR / "sufethesismas.cls",
        THESIS_DIR / "figures" / "SHUFEBadge.jpg",
        THESIS_DIR / "figures" / "statement.pdf",
        REPORT_DIR / "开题报告.tex",
        REPORT_DIR / "开题报告.pdf",
        SLIDES_DIR / "开题汇报.tex",
        SLIDES_DIR / "开题汇报.pdf",
        SLIDES_DIR / "SUFE.sty",
        SLIDES_DIR / "pic" / "sufe.png",
        FINAL_DIR / "论文.tex",
        FINAL_DIR / "论文.pdf",
        FINAL_DIR / "开题报告.tex",
        FINAL_DIR / "开题报告.pdf",
        FINAL_DIR / "开题汇报.tex",
        FINAL_DIR / "开题汇报.pdf",
    ]
    assert all(path.exists() and path.stat().st_size > 0 for path in expected)
    assert sorted(path.name for path in FINAL_DIR.iterdir()) == [
        "开题报告.pdf",
        "开题报告.tex",
        "开题汇报.pdf",
        "开题汇报.tex",
        "论文.pdf",
        "论文.tex",
    ]

    thesis_source = (THESIS_DIR / "论文.tex").read_text(encoding="utf-8")
    report_source = (REPORT_DIR / "开题报告.tex").read_text(encoding="utf-8")
    slides_source = (SLIDES_DIR / "开题汇报.tex").read_text(encoding="utf-8")
    assert r"\documentclass{sufethesismas}" in thesis_source
    assert r"\documentclass[UTF8,12pt,a4paper]{ctexart}" in report_source
    assert r"\usepackage{SUFE}" in slides_source
    assert len(re.findall(r"^\\chapter\{", thesis_source, re.MULTILINE)) == 7
    assert len(re.findall(r"^\\bibitem\{", thesis_source, re.MULTILINE)) == 45
    assert len(re.findall(r"^\\bibitem\{", report_source, re.MULTILINE)) == 25
    frame_count = len(re.findall(r"\\begin\{frame", slides_source))
    assert 15 <= frame_count <= 20
    assert "2025213385" in thesis_source and "张吕欧" in thesis_source
    assert "2025213385" in report_source and "张吕欧" in report_source

    forbidden = [
        r"\bV[1-6]\b",
        r"Version\s+[1-6]",
        r"上一版",
        r"前一版",
        r"版本沿革",
    ]
    thesis_text = command_output(["pdftotext", str(FINAL_DIR / "论文.pdf"), "-"])
    report_text = command_output(
        ["pdftotext", str(FINAL_DIR / "开题报告.pdf"), "-"]
    )
    slides_text = command_output(
        ["pdftotext", str(FINAL_DIR / "开题汇报.pdf"), "-"]
    )
    for content in [
        thesis_source,
        report_source,
        slides_source,
        thesis_text,
        report_text,
        slides_text,
    ]:
        for pattern in forbidden:
            assert not re.search(pattern, content, flags=re.IGNORECASE)

    for phrase in [
        "14,677",
        "33,190",
        "Jaccard",
        "2.385",
        "2.421",
        "2.383",
        "0.054",
        "0.855",
        "0.873",
    ]:
        assert phrase in thesis_text
        assert phrase in report_text
        assert phrase in slides_text

    thesis_log = (THESIS_DIR / "论文.log").read_text(
        encoding="utf-8", errors="replace"
    )
    report_log = (REPORT_DIR / "开题报告.log").read_text(
        encoding="utf-8", errors="replace"
    )
    slides_log = (SLIDES_DIR / "开题汇报.log").read_text(
        encoding="utf-8", errors="replace"
    )
    fatal_patterns = ["LaTeX Error", "Undefined control sequence", "Emergency stop"]
    for pattern in fatal_patterns:
        assert pattern not in thesis_log
        assert pattern not in report_log
        assert pattern not in slides_log
    assert "Overfull \\vbox" not in slides_log

    thesis_pdf = pdf_metadata(FINAL_DIR / "论文.pdf")
    report_pdf = pdf_metadata(FINAL_DIR / "开题报告.pdf")
    slides_pdf = pdf_metadata(FINAL_DIR / "开题汇报.pdf")
    assert thesis_pdf["pages"] >= 60
    assert abs(thesis_pdf["width_points"] / thesis_pdf["height_points"] - 0.707) < 0.01
    assert 12 <= report_pdf["pages"] <= 16
    assert abs(report_pdf["width_points"] / report_pdf["height_points"] - 0.707) < 0.01
    assert 15 <= slides_pdf["pages"] <= 20
    assert abs(slides_pdf["width_points"] / slides_pdf["height_points"] - 16 / 9) < 0.02

    report = {
        "status": "passed",
        "compiler": command_output(["tectonic", "--version"]).strip(),
        "thesis": {
            **thesis_pdf,
            "chapters": 7,
            "references": 45,
            "template": "sufethesismas",
        },
        "proposal_slides": {
            **slides_pdf,
            "frames": frame_count,
            "template": "SUFE Beamer",
        },
        "proposal_report": {
            **report_pdf,
            "references": 25,
            "reference_layout_pages": report_pdf["pages"],
        },
        "final_folder_files": 6,
        "visible_previous_version_markers": 0,
        "fatal_latex_errors": 0,
        "slide_overfull_vboxes": 0,
    }
    (ENGINE_DIR / "验收报告.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
