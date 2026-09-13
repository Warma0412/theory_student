#!/usr/bin/env python3
"""Validate V2 analysis artifacts and thesis deliverables."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
V2_DIR = HERE.parents[1]
PROJECT_DIR = HERE.parents[3]
V1_DIR = PROJECT_DIR / "论文" / "v1"
PAPER_DIR = PROJECT_DIR / "论文"
RESULTS = V2_DIR / "results"


def pdf_pages(path: Path) -> int:
    output = subprocess.run(
        ["pdfinfo", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    match = re.search(r"^Pages:\s+(\d+)$", output, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Could not read page count for {path}")
    return int(match.group(1))


def main() -> None:
    expected = [
        V2_DIR / "论文_v2.md",
        V2_DIR / "论文_v2.docx",
        V2_DIR / "论文_v2.html",
        V2_DIR / "论文_v2.pdf",
        PAPER_DIR / "论文_v2_最终版.md",
        PAPER_DIR / "论文_v2_精简版.pdf",
        PAPER_DIR / "论文_v2_极限精简版.pdf",
        RESULTS / "deep_generator_lasso_w.csv",
        RESULTS / "paired_mlp_w.csv",
        RESULTS / "v2_predictive_model_metrics.csv",
        RESULTS / "v2_reproducibility.json",
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing V2 outputs: {missing}")

    markdown = (V2_DIR / "论文_v2.md").read_text(encoding="utf-8")
    root_markdown = (PAPER_DIR / "论文_v2_最终版.md").read_text(
        encoding="utf-8"
    )
    assert not re.findall(r"\{\{[A-Z0-9_]+\}\}", markdown)
    assert not re.findall(r"\{\{[A-Z0-9_]+\}\}", root_markdown)
    assert "14 个维度" in markdown
    assert "未通过预设联合交换性" in markdown
    assert "不作 FDR 声明" in markdown or "不作FDR声明" in markdown

    image_links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", markdown)
    missing_images = [
        link for link in image_links if not (V2_DIR / link).resolve().exists()
    ]
    if missing_images:
        raise FileNotFoundError(f"Missing thesis images: {missing_images}")

    deep_w = pd.read_csv(RESULTS / "deep_generator_lasso_w.csv")
    paired_w = pd.read_csv(RESULTS / "paired_mlp_w.csv")
    diagnostics = pd.read_csv(RESULTS / "deep_generator_diagnostics.csv")
    metrics = pd.read_csv(RESULTS / "v2_predictive_model_metrics.csv")
    assert deep_w.shape == (30, 32)
    assert paired_w.shape == (20, 32)
    assert not deep_w.isna().any().any()
    assert not paired_w.isna().any().any()
    assert set(diagnostics["method"]) == {
        "copula_mvr",
        "gaussian_mvr",
        "deep_knockoff_raw",
        "deep_knockoff_calibrated",
    }
    assert np.isfinite(
        diagnostics.select_dtypes(include=[np.number]).to_numpy()
    ).all()

    v1_metrics = pd.read_csv(V1_DIR / "results" / "predictive_model_metrics.csv")
    inherited = metrics.loc[metrics["model"].isin(v1_metrics["model"])].reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(
        inherited[v1_metrics.columns],
        v1_metrics.reset_index(drop=True),
        check_exact=True,
    )
    residual = metrics.loc[metrics["model"].eq("PyTorch-Residual-MLP")]
    assert len(residual) == 1

    v1_audit = json.loads(
        (V1_DIR / "results" / "data_audit.json").read_text(encoding="utf-8")
    )
    reproducibility = json.loads(
        (RESULTS / "v2_reproducibility.json").read_text(encoding="utf-8")
    )
    antisymmetry = json.loads(
        (RESULTS / "paired_mlp_antisymmetry.json").read_text(encoding="utf-8")
    )
    assert reproducibility["v1_panel_sha256"] == v1_audit["panel_sha256"]
    assert not reproducibility["deep_generator_passed_prespecified_diagnostics"]
    assert antisymmetry["passed_tolerance_1e_5"]

    page_counts = {
        "complete": pdf_pages(V2_DIR / "论文_v2.pdf"),
        "condensed": pdf_pages(PAPER_DIR / "论文_v2_精简版.pdf"),
        "ultra_condensed": pdf_pages(PAPER_DIR / "论文_v2_极限精简版.pdf"),
    }
    assert page_counts["complete"] >= 45
    assert 10 <= page_counts["condensed"] <= 30
    assert page_counts["ultra_condensed"] == 5

    report = {
        "status": "passed",
        "expected_files": len(expected),
        "image_links_checked": len(image_links),
        "deep_generator_repetitions": len(deep_w),
        "paired_mlp_repetitions": len(paired_w),
        "panel_sha256_matches_v1": True,
        "deep_generator_diagnostic_decision": "rejected",
        "paired_mlp_antisymmetry": "passed",
        "pdf_pages": page_counts,
    }
    (RESULTS / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
