#!/usr/bin/env python3
"""Generate the complete V3 thesis with the eight-dimension shortlist."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
from docx import Document


HERE = Path(__file__).resolve()
V3_DIR = HERE.parents[1]
PROJECT_DIR = HERE.parents[3]
V1_DIR = PROJECT_DIR / "论文" / "v1"
V2_DIR = PROJECT_DIR / "论文" / "v2"
V2_CODE = V2_DIR / "code"
RESULTS = V3_DIR / "results"
TEMPLATE = HERE.with_name("thesis_template_v3.md")
sys.path.insert(0, str(V2_CODE))

from generate_v2_thesis import (  # noqa: E402
    build_context as build_v2_context,
    labels,
    make_reference_doc,
    markdown_table,
    postprocess_docx,
    write_css,
)


ENGLISH_LABELS = {
    "order_count": "order count",
    "unique_customer_count": "unique customer count",
    "avg_price": "average product price",
    "avg_ship_days": "average approval-to-carrier days",
    "buyer_state_diversity": "buyer-state diversity",
    "avg_estimate_gap_days": "average early-delivery margin",
    "avg_delivery_days": "average order-to-delivery days",
    "item_count": "item count",
    "avg_product_name_length": "average product-title length",
    "avg_freight": "average freight",
}


def load_metadata() -> dict:
    return json.loads(
        (RESULTS / "v3_shortlist_metadata.json").read_text(encoding="utf-8")
    )


def build_context() -> dict[str, str]:
    context = build_v2_context()
    metadata = load_metadata()
    shortlist = pd.read_csv(RESULTS / "v3_shortlist.csv")
    selection_path = pd.read_csv(RESULTS / "v3_selection_path.csv")
    metrics = pd.read_csv(RESULTS / "v3_predictive_metrics.csv")
    primary = pd.read_csv(V1_DIR / "results" / "knockoff_primary.csv")
    strict = primary.loc[
        primary["q"].eq(0.20) & primary["selected_ebh"].astype(bool),
        "feature",
    ].tolist()
    features = metadata["shortlist_features"]
    remainder = [feature for feature in strict if feature not in set(features)]
    short_metric = metrics.loc[
        metrics["model"].eq(f"XGBoost-K={metadata['recommended_default_k']}")
    ].iloc[0]
    full_metric = metrics.loc[metrics["model"].eq("XGBoost")].iloc[0]
    relative_change = metadata["relative_rmse_change_vs_full_xgboost"]
    performance_retained = full_metric["rmse_log"] / short_metric["rmse_log"]

    shortlist_view = shortlist.copy()
    shortlist_view["frequency_fmt"] = shortlist_view["selection_frequency"].map(
        lambda value: f"{value:.1%}"
    )
    shortlist_view["model_rank_fmt"] = shortlist_view["model_rank"].astype(int)
    shortlist_view["contribution_fmt"] = shortlist_view[
        "contribution_share_within_fdr_set"
    ].map(lambda value: f"{value:.1%}")
    shortlist_view["cumulative_fmt"] = shortlist_view[
        "cumulative_contribution"
    ].map(lambda value: f"{value:.1%}")
    shortlist_view["evidence"] = "e-BH通过；按验证SHAP排序"
    metric_view = metrics.copy()
    metric_view["wape_fmt"] = metric_view["wape_raw"].map(
        lambda value: f"{value:.1%}"
    )
    names = {
        "XGBoost-K=4": "XGBoost（K=4极简）",
        "XGBoost-K=8": "XGBoost（K=8均衡）",
        "XGBoost-K=10": "XGBoost（K=10默认）",
        "XGBoost-K=14": "XGBoost（K=14完整）",
        "XGBoost": "XGBoost（全31维）",
        "Naive-current-GMV": "本月GMV朴素基线",
    }
    metric_view["display_model"] = metric_view["model"].map(names)
    path_view = selection_path.loc[
        selection_path["k"].isin(
            sorted(set([4, 8, metadata["recommended_default_k"], 14]))
        )
    ].copy()
    path_view["tier"] = path_view["k"].map(
        {
            4: "极简",
            8: "均衡",
            metadata["recommended_default_k"]: "默认推荐",
            14: "完整确认集",
        }
    )
    path_view["contribution_fmt"] = path_view["cumulative_contribution"].map(
        lambda value: f"{value:.1%}"
    )
    path_view["labels"] = path_view["labels_zh"]
    repro = pd.DataFrame(
        [
            ["可控参数", "K，可取1至14"],
            ["排序规则", metadata["ranking_rule"]],
            ["默认K规则", metadata["default_k_rule"]],
            ["正式FDR确认集", metadata["formal_fdr_set_count"]],
            ["模型推荐默认K", metadata["recommended_default_k"]],
            ["默认K测试RMSE", metadata["default_k_test_rmse"]],
            ["全维测试RMSE", metadata["full_xgboost_rmse"]],
            ["RMSE相对变化", f"{relative_change:.3%}"],
            ["面板SHA-256", metadata["panel_sha256"]],
            ["结论权限", "参数K仅控制展示，不新增FDR声明"],
        ],
        columns=["项目", "记录值"],
    )

    context.update(
        {
            "V3_SHORTLIST_COUNT": str(len(features)),
            "V3_DEFAULT_K": str(metadata["recommended_default_k"]),
            "V3_SHORTLIST_LIST": labels(features),
            "V3_SHORTLIST_LIST_EN": ", ".join(
                ENGLISH_LABELS[feature] for feature in features
            ),
            "V3_REMAINDER_COUNT": str(len(remainder)),
            "V3_REMAINDER_LIST": labels(remainder),
            "V3_RMSE": f"{short_metric['rmse_log']:.3f}",
            "V3_FULL_RMSE": f"{full_metric['rmse_log']:.3f}",
            "V3_RMSE_CHANGE": f"{relative_change:.3%}",
            "V3_PERFORMANCE_RETAINED": f"{performance_retained:.2%}",
            "V3_SHORTLIST_TABLE": markdown_table(
                shortlist_view,
                [
                    "shortlist_rank",
                    "label_zh",
                    "frequency_fmt",
                    "mean_evalue",
                    "model_rank_fmt",
                    "contribution_fmt",
                    "cumulative_fmt",
                    "evidence",
                ],
                [
                    "排序",
                    "维度",
                    "单轮频率",
                    "平均e-value",
                    "模型排名",
                    "贡献占比",
                    "累计贡献",
                    "入选依据",
                ],
            ),
            "V3_METRIC_TABLE": markdown_table(
                metric_view,
                [
                    "display_model",
                    "n_test",
                    "rmse_log",
                    "mae_log",
                    "r2_log",
                    "wape_fmt",
                ],
                ["模型", "测试N", "RMSE(log)", "MAE(log)", "R2", "WAPE"],
            ),
            "V3_REPRO_TABLE": markdown_table(
                repro,
                ["项目", "记录值"],
                ["项目", "记录值"],
            ),
            "V3_PATH_TABLE": markdown_table(
                path_view,
                [
                    "tier",
                    "k",
                    "contribution_fmt",
                    "validation_rmse_log",
                    "rmse_log",
                    "labels",
                ],
                [
                    "档位",
                    "K",
                    "累计贡献",
                    "验证RMSE",
                    "测试RMSE",
                    "所选维度",
                ],
            ),
        }
    )
    return context


def build_markdown() -> str:
    text = TEMPLATE.read_text(encoding="utf-8")
    for key, value in build_context().items():
        text = text.replace("{{" + key + "}}", str(value))
    unresolved = sorted(set(re.findall(r"\{\{[A-Z0-9_]+\}\}", text)))
    if unresolved:
        raise ValueError(f"Unresolved V3 placeholders: {unresolved}")
    controls = [
        character
        for character in text
        if ord(character) < 32 and character not in "\n\t\r"
    ]
    if controls:
        raise ValueError("V3 thesis contains control characters")
    return text


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=V3_DIR)


def main() -> None:
    markdown_path = V3_DIR / "论文_v3.md"
    root_markdown_path = V3_DIR.parent / "论文_v3_最终版.md"
    docx_path = V3_DIR / "论文_v3.docx"
    html_path = V3_DIR / "论文_v3.html"
    pdf_path = V3_DIR / "论文_v3.pdf"
    reference_path = V3_DIR / "reference.docx"
    css_path = V3_DIR / "thesis.css"
    text = build_markdown()
    markdown_path.write_text(text, encoding="utf-8")
    root_markdown_path.write_text(
        text.replace("](figures/", "](v3/figures/"),
        encoding="utf-8",
    )
    make_reference_doc(reference_path)
    write_css(css_path)
    run(
        [
            "pandoc",
            str(markdown_path),
            "--from=markdown+tex_math_dollars",
            "--toc",
            "--toc-depth=3",
            f"--reference-doc={reference_path}",
            f"--resource-path={V3_DIR}",
            "-o",
            str(docx_path),
        ]
    )
    postprocess_docx(docx_path)
    doc = Document(docx_path)
    doc.core_properties.title = (
        "面向电商GMV监控的可控错误发现维度选择研究（v3）"
    )
    doc.core_properties.subject = "Model-X Knockoff；八维运营短名单；XGBoost"
    doc.save(docx_path)
    run(
        [
            "pandoc",
            str(markdown_path),
            "--from=markdown+tex_math_dollars",
            "--standalone",
            "--toc",
            "--toc-depth=3",
            "--mathml",
            "--embed-resources",
            f"--css={css_path}",
            f"--resource-path={V3_DIR}",
            "-o",
            str(html_path),
        ]
    )
    chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    run(
        [
            str(chrome),
            "--headless",
            "--disable-gpu",
            "--allow-file-access-from-files",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            html_path.resolve().as_uri(),
        ]
    )
    manifest = {
        "version": "v3",
        "markdown_characters": len(text),
        "chinese_characters": len(re.findall(r"[\u4e00-\u9fff]", text)),
        "unresolved_placeholders": [],
        "outputs": {
            path.name: path.stat().st_size
            for path in [
                markdown_path,
                root_markdown_path,
                docx_path,
                html_path,
                pdf_path,
            ]
        },
    }
    (RESULTS / "thesis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
