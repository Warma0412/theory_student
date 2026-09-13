#!/usr/bin/env python3
"""Generate the fixed five-page V3 shortlist PDF."""

from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from generate_v3_thesis import (
    RESULTS,
    V1_DIR,
    V2_DIR,
    V3_DIR,
    build_context,
)


sys.path.insert(0, str(V2_DIR / "code"))
from generate_v2_ultra_condensed_thesis import (  # noqa: E402
    diagnostic_table,
    posthoc_table,
)


TEMPLATE = Path(__file__).with_name("thesis_ultra_condensed_template_v3.html")
OUTPUT_PDF = V3_DIR.parent / "论文_v3_极限精简版.pdf"


def model_table() -> str:
    v3_metrics = pd.read_csv(RESULTS / "v3_predictive_metrics.csv")
    v2_metrics = pd.read_csv(V2_DIR / "results" / "v2_predictive_model_metrics.csv")
    residual = v2_metrics.loc[
        v2_metrics["model"].eq("PyTorch-Residual-MLP")
    ].copy()
    mlp = v2_metrics.loc[v2_metrics["model"].eq("MLP")].copy()
    view = pd.concat(
        [
            v3_metrics.loc[
                v3_metrics["model"].str.startswith("XGBoost-K=")
                | v3_metrics["model"].isin(["XGBoost", "Naive-current-GMV"])
            ],
            residual,
            mlp,
        ],
        ignore_index=True,
    )
    metadata = json.loads(
        (RESULTS / "v3_shortlist_metadata.json").read_text(encoding="utf-8")
    )
    default_name = f"XGBoost-K={metadata['recommended_default_k']}"
    order = ["XGBoost-K=4", "XGBoost-K=8", default_name, "XGBoost-K=14"]
    order += ["XGBoost", "PyTorch-Residual-MLP", "Naive-current-GMV"]
    order = list(dict.fromkeys(order))
    view = view.set_index("model").loc[order].reset_index()
    names = {
        "XGBoost-K=4": "K=4极简",
        "XGBoost-K=8": "K=8均衡",
        default_name: f"K={metadata['recommended_default_k']}默认",
        "XGBoost-K=14": "K=14完整",
        "XGBoost": "全31维XGBoost",
        "PyTorch-Residual-MLP": "残差MLP",
        "Naive-current-GMV": "朴素基线",
    }
    rows = []
    for _, row in view.iterrows():
        rows.append(
            "<tr>"
            f"<td>{html.escape(names[row['model']])}</td>"
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


def build_html() -> str:
    context = build_context()
    selection_path = pd.read_csv(RESULTS / "v3_selection_path.csv")
    for k in (4, 8):
        context[f"V3_K{k}_LIST"] = str(
            selection_path.loc[selection_path["k"].eq(k), "labels_zh"].iloc[0]
        )
    context["V3_ULTRA_MODEL_TABLE"] = model_table()
    context["V2_ULTRA_DIAGNOSTIC_TABLE"] = diagnostic_table(
        pd.read_csv(V2_DIR / "results" / "deep_generator_diagnostics.csv")
    )
    context["V2_ULTRA_POSTHOC_TABLE"] = posthoc_table(
        pd.read_csv(V2_DIR / "results" / "posthoc_evalue_path.csv")
    )
    deep = pd.read_csv(V2_DIR / "results" / "deep_generator_selection.csv")
    deep_features = deep.loc[
        deep["q"].eq(0.20) & deep["selected_ebh"].astype(bool), "feature"
    ].tolist()
    from build_panel import FEATURE_LABELS_ZH

    context["V2_DEEP_FORMAL_COUNT"] = str(len(deep_features))
    context["V2_DEEP_FORMAL_LIST"] = "、".join(
        FEATURE_LABELS_ZH[feature] for feature in deep_features
    )
    shap = pd.read_csv(V1_DIR / "results" / "xgboost_shap_importance.csv")
    context["V2_SHAP_TOP10"] = "、".join(shap.head(10)["label_zh"])
    fixed = pd.read_csv(V1_DIR / "results" / "two_way_fixed_effects.csv")
    context["V2_FE_SIGNIFICANT"] = "、".join(
        fixed.loc[fixed["p_value"].lt(0.05), "label_zh"]
    )

    document = TEMPLATE.read_text(encoding="utf-8")
    for key, value in context.items():
        document = document.replace("{{" + key + "}}", str(value))
    unresolved = sorted(set(re.findall(r"\{\{[A-Z0-9_]+\}\}", document)))
    if unresolved:
        raise ValueError(f"Unresolved V3 ultra placeholders: {unresolved}")
    return document


def main() -> None:
    chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    document = build_html()
    with tempfile.TemporaryDirectory(prefix="olist-v3-ultra-") as temporary:
        workdir = Path(temporary)
        html_path = workdir / "论文_v3_极限精简版.html"
        pdf_path = workdir / "论文_v3_极限精简版.pdf"
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
    print(f"Generated {OUTPUT_PDF} ({OUTPUT_PDF.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
