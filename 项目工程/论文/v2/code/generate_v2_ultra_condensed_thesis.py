#!/usr/bin/env python3
"""Generate the fixed five-page ultra-condensed V2 thesis PDF."""

from __future__ import annotations

import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from generate_v2_thesis import (
    FEATURE_LABELS_ZH,
    RESULTS,
    V1_DIR,
    V2_DIR,
    build_context,
)


TEMPLATE = Path(__file__).with_name("thesis_ultra_condensed_template_v2.html")
OUTPUT_PDF = V2_DIR.parent / "论文_v2_极限精简版.pdf"


def label_list(features: list[str]) -> str:
    if not features:
        return "无"
    return "、".join(
        html.escape(FEATURE_LABELS_ZH.get(feature, feature))
        for feature in features
    )


def model_table(metrics: pd.DataFrame) -> str:
    preferred = [
        "XGBoost",
        "PyTorch-Residual-MLP",
        "Ridge",
        "Lasso",
        "ExtraTrees",
        "MLP",
        "Naive-current-GMV",
    ]
    available = [name for name in preferred if name in set(metrics["model"])]
    view = metrics.set_index("model").loc[available].reset_index()
    names = {
        "PyTorch-Residual-MLP": "残差MLP",
        "Naive-current-GMV": "朴素基线",
    }
    rows = []
    for _, row in view.iterrows():
        rows.append(
            "<tr>"
            f"<td>{html.escape(names.get(row['model'], row['model']))}</td>"
            f"<td>{row['rmse_log']:.3f}</td>"
            f"<td>{row['mae_log']:.3f}</td>"
            f"<td>{row['r2_log']:.3f}</td>"
            f"<td>{row['wape_raw']:.1%}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>模型</th><th>RMSE(log)</th>"
        "<th>MAE(log)</th><th>R2(log)</th><th>WAPE</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def diagnostic_table(diagnostics: pd.DataFrame) -> str:
    names = {
        "copula_mvr": "Copula-MVR",
        "deep_knockoff_raw": "深度生成（原始）",
        "deep_knockoff_calibrated": "深度生成（校准）",
    }
    preferred = list(names)
    view = diagnostics.set_index("method").loc[preferred].reset_index()
    rows = []
    for _, row in view.iterrows():
        rows.append(
            "<tr>"
            f"<td>{names[row['method']]}</td>"
            f"<td>{row['mean_marginal_ks']:.3f}</td>"
            f"<td>{row['covariance_relative_error']:.3f}</td>"
            f"<td>{row['cross_covariance_asymmetry']:.3f}</td>"
            f"<td>{row['mean_real_knockoff_correlation']:.3f}</td>"
            f"<td>{row['swap_classifier_auc_mean']:.3f}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>方法</th><th>平均KS</th><th>协方差误差</th>"
        "<th>交叉非对称</th><th>真伪相关</th><th>swap AUC</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def posthoc_table(path: pd.DataFrame) -> str:
    levels = [0.10, 0.15, 0.20, 0.30, 0.40]
    view = path.loc[path["q_ebh"].isin(levels)]
    rows = [
        "<tr>"
        f"<td>{row['q_ebh']:.2f}</td>"
        f"<td>{row['fixed_alpha_kn']:.2f}</td>"
        f"<td>{int(row['selected_count'])}</td>"
        "</tr>"
        for _, row in view.iterrows()
    ]
    return (
        "<table><thead><tr><th>最终e-BH水平</th><th>固定单轮水平</th>"
        "<th>严格入选数</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def build_html() -> str:
    context = build_context()
    primary = pd.read_csv(V1_DIR / "results" / "knockoff_primary.csv")
    deep_selection = pd.read_csv(RESULTS / "deep_generator_selection.csv")
    strict = primary.loc[
        primary["q"].eq(0.20) & primary["selected_ebh"].astype(bool),
        "feature",
    ].tolist()
    deep_formal = deep_selection.loc[
        deep_selection["q"].eq(0.20)
        & deep_selection["selected_ebh"].astype(bool),
        "feature",
    ].tolist()
    groups = {
        "STRICT_SCALE": [
            feature
            for feature in strict
            if feature
            in {
                "item_count",
                "order_count",
                "category_count",
                "unique_customer_count",
            }
        ],
        "STRICT_PRICE": [
            feature
            for feature in strict
            if feature in {"avg_price", "avg_freight"}
        ],
        "STRICT_FULFILLMENT": [
            feature
            for feature in strict
            if feature
            in {
                "avg_delivery_days",
                "avg_ship_days",
                "avg_estimate_gap_days",
            }
        ],
    }
    assigned = set(sum(groups.values(), []))
    groups["STRICT_PRODUCT_MARKET"] = [
        feature for feature in strict if feature not in assigned
    ]
    context.update({key: label_list(value) for key, value in groups.items()})
    context["V2_DEEP_FORMAL_COUNT"] = str(len(deep_formal))
    context["V2_DEEP_FORMAL_LIST"] = label_list(deep_formal)
    context["V2_SHAP_TOP10"] = label_list(
        pd.read_csv(V1_DIR / "results" / "xgboost_shap_importance.csv")
        .head(10)["feature"]
        .tolist()
    )
    fixed_effects = pd.read_csv(
        V1_DIR / "results" / "two_way_fixed_effects.csv"
    )
    context["V2_FE_SIGNIFICANT"] = label_list(
        fixed_effects.loc[fixed_effects["p_value"].lt(0.05), "feature"].tolist()
    )
    context["V2_ULTRA_MODEL_TABLE"] = model_table(
        pd.read_csv(RESULTS / "v2_predictive_model_metrics.csv")
    )
    context["V2_ULTRA_DIAGNOSTIC_TABLE"] = diagnostic_table(
        pd.read_csv(RESULTS / "deep_generator_diagnostics.csv")
    )
    context["V2_ULTRA_POSTHOC_TABLE"] = posthoc_table(
        pd.read_csv(RESULTS / "posthoc_evalue_path.csv")
    )

    document = TEMPLATE.read_text(encoding="utf-8")
    for key, value in context.items():
        document = document.replace("{{" + key + "}}", str(value))
    unresolved = sorted(set(re.findall(r"\{\{[A-Z0-9_]+\}\}", document)))
    if unresolved:
        raise ValueError(f"Unresolved placeholders: {unresolved}")
    controls = [
        character
        for character in document
        if ord(character) < 32 and character not in "\n\t\r"
    ]
    if controls:
        raise ValueError("V2 ultra-condensed document contains control characters")
    return document


def main() -> None:
    chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if not chrome.exists():
        raise FileNotFoundError("Google Chrome is required for PDF generation")
    document = build_html()
    with tempfile.TemporaryDirectory(prefix="olist-v2-ultra-") as temporary:
        workdir = Path(temporary)
        html_path = workdir / "论文_v2_极限精简版.html"
        pdf_path = workdir / "论文_v2_极限精简版.pdf"
        html_path.write_text(document, encoding="utf-8")
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
    print(
        f"Generated {OUTPUT_PDF} "
        f"({len(document)} characters, {OUTPUT_PDF.stat().st_size} bytes)"
    )


if __name__ == "__main__":
    main()
