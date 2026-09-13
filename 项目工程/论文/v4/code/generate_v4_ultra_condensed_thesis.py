#!/usr/bin/env python3
"""Generate the fixed five-page V4 methods PDF."""

from __future__ import annotations

import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from generate_v4_thesis import RESULTS, V4_DIR, build_context


TEMPLATE = Path(__file__).with_name("thesis_ultra_condensed_template_v4.html")
OUTPUT_PDF = V4_DIR.parent / "论文_v4_极限精简版.pdf"


def html_table(frame: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    rows = []
    for _, row in frame.iterrows():
        cells = "".join(
            f"<td>{html.escape(str(row[column]))}</td>" for column in columns
        )
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def build_html() -> str:
    context = build_context()
    clustered = pd.read_csv(RESULTS / "clustered_knockoff_selection.csv")
    grouped = pd.read_csv(RESULTS / "group_knockoff_selection.csv")
    diagnostics = pd.read_csv(RESULTS / "v4_generator_diagnostic_summary.csv")
    grip = pd.read_csv(RESULTS / "grip2_style_selection.csv")

    cluster_rows = []
    for method, label in [
        ("clustered_between_seller", "卖家间均值层"),
        ("clustered_within_seller", "卖家内离差层"),
    ]:
        view = clustered.loc[
            clustered["method"].eq(method) & clustered["q"].eq(0.20)
        ]
        selected = view.loc[view["selected_ebh"].astype(bool), "label_zh"].tolist()
        cluster_rows.append(
            {
                "层级": label,
                "严格数": len(selected),
                "严格集合": "、".join(selected) if selected else "无",
            }
        )
    context["V4_ULTRA_CLUSTER_TABLE"] = html_table(
        pd.DataFrame(cluster_rows),
        ["层级", "严格数", "严格集合"],
        ["层级", "q=.20严格数", "严格集合"],
    )

    group_rows = []
    for q in (0.20, 0.30):
        view = grouped.loc[grouped["q"].eq(q)]
        selected = view.loc[view["selected_ebh"].astype(bool), "label_zh"].tolist()
        group_rows.append(
            {
                "水平": f"{q:.2f}",
                "组数": len(selected),
                "组": "、".join(selected) if selected else "无",
            }
        )
    context["V4_ULTRA_GROUP_TABLE"] = html_table(
        pd.DataFrame(group_rows),
        ["水平", "组数", "组"],
        ["最终水平", "严格组数", "业务组"],
    )

    names = {
        "copula_mvr": "Copula-MVR",
        "v2_deep_train_calibrated": "V2深度（训练校准）",
        "v4_worst_swap_train_calibrated": "V4最坏swap",
    }
    diagnostic_view = diagnostics.copy()
    diagnostic_view["方法"] = diagnostic_view["method"].map(names)
    diagnostic_view["KS"] = diagnostic_view["mean_marginal_ks"].map(
        lambda value: f"{value:.3f}"
    )
    diagnostic_view["深度核拒绝"] = diagnostic_view[
        "deep_kernel_rejection_rate_5pct"
    ].map(lambda value: f"{value:.0%}")
    diagnostic_view["AUC"] = diagnostic_view["classifier_auc_mean"].map(
        lambda value: f"{value:.3f}"
    )
    context["V4_ULTRA_DIAGNOSTIC_TABLE"] = html_table(
        diagnostic_view,
        ["方法", "KS", "深度核拒绝", "AUC"],
        ["方法", "平均KS", "深度核拒绝率", "分类AUC"],
    )
    copula_row = diagnostics.set_index("method").loc["copula_mvr"]
    context["V4_COPULA_AUC"] = f"{copula_row['classifier_auc_mean']:.3f}"

    grip_view = (
        grip.loc[grip["q"].eq(0.20)]
        .nlargest(10, "mean_w")
        .copy()
    )
    q30_set = set(
        grip.loc[
            grip["q"].eq(0.30) & grip["selected_ebh"].astype(bool), "feature"
        ]
    )
    grip_view["平均W"] = grip_view["mean_w"].map(lambda value: f"{value:.3f}")
    grip_view["q=.20频率"] = grip_view["selection_frequency"].map(
        lambda value: f"{value:.0%}"
    )
    grip_view["q=.30入选"] = grip_view["feature"].map(
        lambda value: "是" if value in q30_set else "否"
    )
    context["V4_ULTRA_GRIP_TABLE"] = html_table(
        grip_view,
        ["label_zh", "平均W", "q=.20频率", "q=.30入选"],
        ["维度", "平均W", "q=.20频率", "q=.30入选"],
    )

    document = TEMPLATE.read_text(encoding="utf-8")
    for key, value in context.items():
        document = document.replace("{{" + key + "}}", str(value))
    unresolved = sorted(set(re.findall(r"\{\{[A-Z0-9_]+\}\}", document)))
    if unresolved:
        raise ValueError(f"Unresolved V4 ultra placeholders: {unresolved}")
    return document


def main() -> None:
    chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    document = build_html()
    with tempfile.TemporaryDirectory(prefix="olist-v4-ultra-") as temporary:
        workdir = Path(temporary)
        html_path = workdir / "论文_v4_极限精简版.html"
        pdf_path = workdir / "论文_v4_极限精简版.pdf"
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
