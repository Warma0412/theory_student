#!/usr/bin/env python3
"""Validate V3 shortlist logic and document deliverables."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve()
V3_DIR = HERE.parents[1]
PROJECT_DIR = HERE.parents[3]
PAPER_DIR = PROJECT_DIR / "论文"
V1_RESULTS = PAPER_DIR / "v1" / "results"
RESULTS = V3_DIR / "results"


def pages(path: Path) -> int:
    output = subprocess.run(
        ["pdfinfo", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return int(re.search(r"^Pages:\s+(\d+)$", output, re.MULTILINE).group(1))


def main() -> None:
    outputs = [
        V3_DIR / "论文_v3.md",
        V3_DIR / "论文_v3.docx",
        V3_DIR / "论文_v3.html",
        V3_DIR / "论文_v3.pdf",
        PAPER_DIR / "论文_v3_最终版.md",
        PAPER_DIR / "论文_v3_精简版.pdf",
        PAPER_DIR / "论文_v3_极限精简版.pdf",
    ]
    assert all(path.exists() for path in outputs)
    shortlist = pd.read_csv(RESULTS / "v3_shortlist.csv")
    evidence = pd.read_csv(RESULTS / "v3_all_feature_evidence.csv")
    metadata = json.loads(
        (RESULTS / "v3_shortlist_metadata.json").read_text(encoding="utf-8")
    )
    path = pd.read_csv(RESULTS / "v3_selection_path.csv")
    default_k = int(metadata["recommended_default_k"])
    assert len(shortlist) == default_k
    assert shortlist["strict_ebh_q20"].all()
    assert evidence["strict_ebh_q20"].sum() == 14
    assert set(shortlist["feature"]) == set(metadata["shortlist_features"])
    assert metadata["shortlist_is_operational_not_new_fdr_claim"]
    assert abs(metadata["relative_rmse_change_vs_full_xgboost"]) < 0.01
    assert path["k"].tolist() == list(range(1, 15))
    assert path["recommended_default"].sum() == 1
    assert int(path.loc[path["recommended_default"], "k"].iloc[0]) == default_k
    default_row = path.loc[path["k"].eq(default_k)].iloc[0]
    assert default_row["cumulative_contribution"] >= 0.90
    assert default_row["validation_rmse_log"] <= (
        path["validation_rmse_log"].min() * 1.005
    )

    markdown = (V3_DIR / "论文_v3.md").read_text(encoding="utf-8")
    assert not re.findall(r"\{\{[A-Z0-9_]+\}\}", markdown)
    assert "参数化维度选择" in markdown
    assert "不新增FDR声明" in markdown or "不另作 FDR 声明" in markdown

    page_counts = {
        "complete": pages(V3_DIR / "论文_v3.pdf"),
        "condensed": pages(PAPER_DIR / "论文_v3_精简版.pdf"),
        "ultra": pages(PAPER_DIR / "论文_v3_极限精简版.pdf"),
    }
    assert page_counts["complete"] >= 50
    assert 10 <= page_counts["condensed"] <= 30
    assert page_counts["ultra"] == 5
    report = {
        "status": "passed",
        "formal_fdr_dimensions": 14,
        "selection_parameter": "K in [1, 14]",
        "recommended_default_k": default_k,
        "default_rule_checks": {
            "strict_ebh_q20": True,
            "cumulative_validation_contribution_ge_90pct": True,
            "validation_rmse_within_0_5pct_of_best": True,
        },
        "relative_rmse_change_vs_full_xgboost": metadata[
            "relative_rmse_change_vs_full_xgboost"
        ],
        "pdf_pages": page_counts,
    }
    (RESULTS / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
