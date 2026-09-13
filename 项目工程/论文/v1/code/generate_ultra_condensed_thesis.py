#!/usr/bin/env python3
"""Generate a fixed five-page ultra-condensed thesis PDF."""

from __future__ import annotations

import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from build_panel import FEATURE_LABELS_ZH, V1_DIR
from generate_thesis import build_context


TEMPLATE = Path(__file__).with_name("thesis_ultra_condensed_template.html")
OUTPUT_PDF = V1_DIR.parent / "论文_v1_极限精简版.pdf"


def label_list(features: list[str]) -> str:
    return "、".join(
        html.escape(FEATURE_LABELS_ZH.get(feature, feature))
        for feature in features
    )


def model_table(metrics: pd.DataFrame) -> str:
    preferred = [
        "XGBoost",
        "Ridge",
        "Lasso",
        "ExtraTrees",
        "XGBoost-Knockoff-selected",
        "MLP",
        "Naive-current-GMV",
    ]
    view = metrics.set_index("model").loc[preferred].reset_index()
    names = {
        "XGBoost-Knockoff-selected": "精简XGBoost",
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
        "<th>MAE(log)</th><th>R²(log)</th><th>WAPE</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def build_html() -> str:
    context = build_context()
    primary = pd.read_csv(V1_DIR / "results" / "knockoff_primary.csv")
    strict = primary.loc[
        primary["q"].eq(0.20) & primary["selected_ebh"].astype(bool),
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
    context["ULTRA_MODEL_TABLE"] = model_table(
        pd.read_csv(V1_DIR / "results" / "predictive_model_metrics.csv")
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
        raise ValueError("Ultra-condensed document contains control characters")
    return document


def main() -> None:
    chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if not chrome.exists():
        raise FileNotFoundError("Google Chrome is required for PDF generation")

    document = build_html()
    with tempfile.TemporaryDirectory(prefix="olist-ultra-condensed-") as temporary:
        workdir = Path(temporary)
        html_path = workdir / "论文_v1_极限精简版.html"
        pdf_path = workdir / "论文_v1_极限精简版.pdf"
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
