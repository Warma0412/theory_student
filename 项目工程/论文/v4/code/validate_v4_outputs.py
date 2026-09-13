#!/usr/bin/env python3
"""Validate V4 experiment artifacts and three document versions."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve()
V4_DIR = HERE.parents[1]
PROJECT_DIR = HERE.parents[3]
PAPER_DIR = PROJECT_DIR / "论文"
RESULTS = V4_DIR / "results"


def pages(path: Path) -> int:
    output = subprocess.run(
        ["pdfinfo", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    match = re.search(r"^Pages:\s+(\d+)$", output, re.MULTILINE)
    if not match:
        raise ValueError(f"Unable to read PDF page count: {path}")
    return int(match.group(1))


def main() -> None:
    documents = [
        V4_DIR / "论文_v4.md",
        V4_DIR / "论文_v4.docx",
        V4_DIR / "论文_v4.html",
        V4_DIR / "论文_v4.pdf",
        PAPER_DIR / "论文_v4_最终版.md",
        PAPER_DIR / "论文_v4_精简版.pdf",
        PAPER_DIR / "论文_v4_极限精简版.pdf",
    ]
    assert all(path.exists() for path in documents)

    clustered = pd.read_csv(RESULTS / "clustered_knockoff_selection.csv")
    grouped = pd.read_csv(RESULTS / "group_knockoff_selection.csv")
    deep_w = pd.read_csv(RESULTS / "v4_adversarial_generator_w.csv")
    grip_w = pd.read_csv(RESULTS / "grip2_style_w.csv")
    diagnostics = pd.read_csv(RESULTS / "v4_exchangeability_diagnostics.csv")
    metadata = json.loads(
        (RESULTS / "v4_reproducibility.json").read_text(encoding="utf-8")
    )
    assert len(clustered) == (31 + 28) * 3
    assert len(grouped) == 12 * 3
    assert deep_w.shape == (30, 32)
    assert grip_w.shape == (20, 32)
    assert len(diagnostics) == 15
    assert diagnostics["kernel_permutations"].eq(199).all()
    assert diagnostics["kernel_minimum_attainable_p"].eq(0.005).all()
    assert diagnostics["classifier_permutations"].eq(99).all()
    assert diagnostics["classifier_minimum_attainable_p"].eq(0.01).all()
    assert metadata["marginal_calibration_fit_scope"] == (
        "generator_training_indices_only"
    )
    assert not metadata["v4_generator_passed_prespecified_diagnostics"]
    assert metadata["grip2_style_antisymmetry"]["passed_tolerance_1e_5"]

    clustered_q20 = clustered.loc[clustered["q"].eq(0.20)]
    between_count = int(
        clustered_q20.loc[
            clustered_q20["method"].eq("clustered_between_seller"),
            "selected_ebh",
        ].sum()
    )
    within_count = int(
        clustered_q20.loc[
            clustered_q20["method"].eq("clustered_within_seller"),
            "selected_ebh",
        ].sum()
    )
    group_q20_count = int(
        grouped.loc[grouped["q"].eq(0.20), "selected_ebh"].sum()
    )
    grip = pd.read_csv(RESULTS / "grip2_style_selection.csv")
    grip_q20_count = int(
        grip.loc[grip["q"].eq(0.20), "selected_ebh"].sum()
    )
    grip_q30_count = int(
        grip.loc[grip["q"].eq(0.30), "selected_ebh"].sum()
    )
    assert (between_count, within_count) == (16, 0)
    assert group_q20_count == 0
    assert (grip_q20_count, grip_q30_count) == (0, 5)

    markdown = (V4_DIR / "论文_v4.md").read_text(encoding="utf-8")
    assert not re.findall(r"\{\{[A-Z0-9_]+\}\}", markdown)
    assert "训练子集上拟合" in markdown
    assert "深度核MMD" in markdown
    assert "卖家内层" in markdown

    page_counts = {
        "complete": pages(V4_DIR / "论文_v4.pdf"),
        "condensed": pages(PAPER_DIR / "论文_v4_精简版.pdf"),
        "ultra": pages(PAPER_DIR / "论文_v4_极限精简版.pdf"),
    }
    assert page_counts["complete"] >= 55
    assert 12 <= page_counts["condensed"] <= 32
    assert page_counts["ultra"] == 5
    report = {
        "status": "passed",
        "clustered_q20": {"between": between_count, "within": within_count},
        "group_q20_count": group_q20_count,
        "v4_generator_passed": False,
        "deep_kernel_rejection_rate": 1.0,
        "grip2_style": {
            "antisymmetry_passed": True,
            "q20_count": grip_q20_count,
            "q30_count": grip_q30_count,
        },
        "calibration_scope": metadata["marginal_calibration_fit_scope"],
        "pdf_pages": page_counts,
    }
    (RESULTS / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
