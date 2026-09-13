#!/usr/bin/env python3
"""Generate the complete V2 thesis from audited V1 and V2 result files."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
from docx import Document


HERE = Path(__file__).resolve()
V2_DIR = HERE.parents[1]
PROJECT_DIR = HERE.parents[3]
V1_DIR = PROJECT_DIR / "论文" / "v1"
V1_CODE = V1_DIR / "code"
sys.path.insert(0, str(V1_CODE))

from build_panel import FEATURE_LABELS_ZH  # noqa: E402
from generate_thesis import (  # noqa: E402
    build_context as build_v1_context,
    make_reference_doc,
    markdown_table,
    postprocess_docx,
    write_css,
)


RESULTS = V2_DIR / "results"
TEMPLATE = HERE.with_name("thesis_template_v2.md")


def load_json(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def labels(features: list[str]) -> str:
    if not features:
        return "无"
    return "、".join(FEATURE_LABELS_ZH.get(item, item) for item in features)


def percent(value: float) -> str:
    return f"{value:.1%}"


def selected_features(frame: pd.DataFrame, q: float = 0.20) -> list[str]:
    return frame.loc[
        frame["q"].eq(q) & frame["selected_ebh"].astype(bool),
        "feature",
    ].tolist()


def build_context() -> dict[str, str]:
    context = build_v1_context()
    diagnostics = pd.read_csv(RESULTS / "deep_generator_diagnostics.csv")
    swap = pd.read_csv(RESULTS / "deep_swap_diagnostics.csv")
    deep_selection = pd.read_csv(RESULTS / "deep_generator_selection.csv")
    paired_selection = pd.read_csv(RESULTS / "paired_mlp_selection.csv")
    metrics = pd.read_csv(RESULTS / "v2_predictive_model_metrics.csv")
    posthoc = pd.read_csv(RESULTS / "posthoc_evalue_path.csv")
    history = pd.read_csv(RESULTS / "deep_generator_training_history.csv")
    antisymmetry = load_json("paired_mlp_antisymmetry.json")
    reproducibility = load_json("v2_reproducibility.json")

    diagnostic_names = {
        "copula_mvr": "Copula-MVR",
        "gaussian_mvr": "原始高斯-MVR",
        "deep_knockoff_raw": "深度生成器（原始）",
        "deep_knockoff_calibrated": "深度生成器（边际校准）",
    }
    diagnostic_view = diagnostics.copy()
    diagnostic_view["方法"] = diagnostic_view["method"].map(diagnostic_names)
    diagnostic_view["coverage_90_fmt"] = diagnostic_view["coverage_90_mean"].map(
        percent
    )
    diagnostic_view["coverage_95_fmt"] = diagnostic_view["coverage_95_mean"].map(
        percent
    )

    swap_view = swap.copy()
    swap_view["方法"] = swap_view["method"].map(diagnostic_names)
    swap_view["swap_ratio_fmt"] = swap_view["swap_ratio"].map(percent)
    swap_view["mmd_p_fmt"] = swap_view["mmd_permutation_p"].map(
        lambda value: f"{value:.3f}"
    )
    swap_view["classifier_p_fmt"] = swap_view["classifier_permutation_p"].map(
        lambda value: f"{value:.3f}"
    )

    deep_row = diagnostics.set_index("method").loc["deep_knockoff_calibrated"]
    raw_row = diagnostics.set_index("method").loc["deep_knockoff_raw"]
    deep_q = deep_selection.loc[deep_selection["q"].eq(0.20)].copy()
    deep_selected = selected_features(deep_selection)
    deep_stable = deep_q.loc[
        deep_q["selection_frequency"].ge(0.90), "feature"
    ].tolist()
    paired_selected = selected_features(paired_selection)

    metric_view = metrics.copy()
    metric_view["wape_fmt"] = metric_view["wape_raw"].map(percent)
    residual = metric_view.loc[
        metric_view["model"].eq("PyTorch-Residual-MLP")
    ].iloc[0]
    xgboost = metric_view.loc[metric_view["model"].eq("XGBoost")].iloc[0]
    rmse_delta = (
        residual["rmse_log"] - xgboost["rmse_log"]
    ) / xgboost["rmse_log"]
    if rmse_delta > 0:
        comparison = f"较 XGBoost 高 {rmse_delta:.1%}，未形成预测增益"
    else:
        comparison = f"较 XGBoost 低 {abs(rmse_delta):.1%}，形成小幅预测增益"

    posthoc_view = posthoc[
        posthoc["q_ebh"].isin([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
    ].copy()
    posthoc_view["q_fmt"] = posthoc_view["q_ebh"].map(lambda value: f"{value:.2f}")
    posthoc_view["labels"] = posthoc_view["selected_labels_zh"].fillna("无")

    deep_valid = bool(
        reproducibility["deep_generator_passed_prespecified_diagnostics"]
    )
    decision = "通过预设诊断，可进入推断"
    if not deep_valid:
        decision = "未通过预设联合交换性诊断，不作FDR声明"
    antisymmetry_passed = bool(antisymmetry["passed_tolerance_1e_5"])

    reproducibility_table = pd.DataFrame(
        [
            ["面板样本量", reproducibility["n"]],
            ["候选维度", reproducibility["p"]],
            ["随机种子", reproducibility["seed"]],
            ["计算设备", reproducibility["device"]],
            ["PyTorch", reproducibility["torch"]],
            ["深度生成重复", reproducibility["config"]["generator_repetitions"]],
            ["配对MLP重复", reproducibility["config"]["paired_mlp_repetitions"]],
            ["生成器诊断结论", decision],
            [
                "反对称性测试",
                "通过" if antisymmetry_passed else "未通过",
            ],
            ["面板SHA-256", reproducibility["v1_panel_sha256"]],
        ],
        columns=["项目", "记录值"],
    )

    context.update(
        {
            "V2_GENERATOR_MAX_EPOCHS": str(
                reproducibility["config"]["generator_epochs"]
            ),
            "V2_GENERATOR_EPOCHS_RUN": str(int(history["epoch"].max())),
            "V2_PAIRED_REPETITIONS": str(
                reproducibility["config"]["paired_mlp_repetitions"]
            ),
            "V2_DEEP_RAW_KS": f"{raw_row['mean_marginal_ks']:.3f}",
            "V2_DEEP_KS": f"{deep_row['mean_marginal_ks']:.3f}",
            "V2_DEEP_MAX_KS": f"{deep_row['max_marginal_ks']:.3f}",
            "V2_DEEP_COV_ERROR": f"{deep_row['covariance_relative_error']:.3f}",
            "V2_DEEP_CROSS_ASYM": f"{deep_row['cross_covariance_asymmetry']:.3f}",
            "V2_DEEP_CORRELATION": (
                f"{deep_row['mean_real_knockoff_correlation']:.3f}"
            ),
            "V2_DEEP_COVERAGE90": percent(deep_row["coverage_90_mean"]),
            "V2_DEEP_COVERAGE95": percent(deep_row["coverage_95_mean"]),
            "V2_DEEP_AUC": f"{deep_row['swap_classifier_auc_mean']:.3f}",
            "V2_DEEP_DECISION": decision,
            "V2_DEEP_SELECTION_SUMMARY": (
                f"固定使用 V1 的 Lasso 惩罚系数后，0.20 水平形式上得到 "
                f"{len(deep_selected)} 项：{labels(deep_selected)}；"
                f"单轮频率至少 90% 的变量有 {len(deep_stable)} 项："
                f"{labels(deep_stable)}。由于生成器未通过准入，这些数字只作诊断。"
            ),
            "V2_PAIRED_COUNT": str(len(paired_selected)),
            "V2_PAIRED_LIST": labels(paired_selected),
            "V2_ANTISYMMETRY_MAX_ERROR": (
                f"{antisymmetry['max_absolute_error']:.3e}"
            ),
            "V2_ANTISYMMETRY_REL_ERROR": (
                f"{antisymmetry['relative_l2_error']:.3e}"
            ),
            "V2_ANTISYMMETRY_DECISION": (
                "通过" if antisymmetry_passed else "未通过"
            ),
            "V2_MLP_RMSE": f"{residual['rmse_log']:.3f}",
            "V2_MLP_R2": f"{residual['r2_log']:.3f}",
            "V2_MLP_WAPE": percent(residual["wape_raw"]),
            "V2_MLP_COMPARISON": comparison,
            "V2_TORCH_VERSION": str(reproducibility["torch"]),
            "V2_LASSO_ALPHA": (
                f"{reproducibility['deep_generator_lasso_alpha']:.12f}"
            ),
            "V2_DIAGNOSTIC_TABLE": markdown_table(
                diagnostic_view,
                [
                    "方法",
                    "mean_marginal_ks",
                    "covariance_relative_error",
                    "cross_covariance_asymmetry",
                    "mean_real_knockoff_correlation",
                    "coverage_90_fmt",
                    "swap_classifier_auc_mean",
                ],
                [
                    "方法",
                    "平均KS",
                    "协方差误差",
                    "交叉非对称",
                    "平均真伪相关",
                    "90%覆盖率",
                    "swap分类AUC",
                ],
            ),
            "V2_SWAP_TABLE": markdown_table(
                swap_view,
                [
                    "方法",
                    "swap_ratio_fmt",
                    "mmd_rbf",
                    "mmd_p_fmt",
                    "classifier_auc",
                    "classifier_p_fmt",
                ],
                [
                    "方法",
                    "交换比例",
                    "RBF-MMD",
                    "MMD p值",
                    "分类AUC",
                    "分类p值",
                ],
            ),
            "V2_PREDICTIVE_TABLE": markdown_table(
                metric_view,
                [
                    "model",
                    "n_test",
                    "rmse_log",
                    "mae_log",
                    "r2_log",
                    "wape_fmt",
                ],
                ["模型", "测试N", "RMSE(log)", "MAE(log)", "R2", "WAPE"],
            ),
            "V2_POSTHOC_TABLE": markdown_table(
                posthoc_view,
                ["q_fmt", "fixed_alpha_kn", "selected_count", "labels"],
                ["最终e-BH水平", "固定单轮水平", "入选数", "入选变量"],
            ),
            "V2_REPRO_TABLE": markdown_table(
                reproducibility_table,
                ["项目", "记录值"],
                ["项目", "记录值"],
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
        raise ValueError(f"Unresolved template placeholders: {unresolved}")
    controls = [
        character
        for character in text
        if ord(character) < 32 and character not in "\n\t\r"
    ]
    if controls:
        raise ValueError("V2 thesis contains control characters")
    return text


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=V2_DIR)


def main() -> None:
    markdown_path = V2_DIR / "论文_v2.md"
    root_markdown_path = V2_DIR.parent / "论文_v2_最终版.md"
    docx_path = V2_DIR / "论文_v2.docx"
    html_path = V2_DIR / "论文_v2.html"
    pdf_path = V2_DIR / "论文_v2.pdf"
    reference_path = V2_DIR / "reference.docx"
    css_path = V2_DIR / "thesis.css"

    text = build_markdown()
    markdown_path.write_text(text, encoding="utf-8")
    root_text = text.replace("](figures/", "](v2/figures/")
    root_markdown_path.write_text(root_text, encoding="utf-8")
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
            f"--resource-path={V2_DIR}",
            "-o",
            str(docx_path),
        ]
    )
    postprocess_docx(docx_path)
    doc = Document(docx_path)
    doc.core_properties.title = (
        "面向电商GMV监控的可控错误发现维度选择研究（v2）"
    )
    doc.core_properties.subject = (
        "Olist；Model-X Knockoff；深度Knockoff；e-value；深度表格学习"
    )
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
            f"--resource-path={V2_DIR}",
            "-o",
            str(html_path),
        ]
    )
    chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if not chrome.exists():
        raise FileNotFoundError("Google Chrome is required for PDF generation")
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
        "version": "v2",
        "markdown_characters": len(text),
        "chinese_characters": len(re.findall(r"[\u4e00-\u9fff]", text)),
        "unresolved_placeholders": re.findall(r"\{\{[A-Z0-9_]+\}\}", text),
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
