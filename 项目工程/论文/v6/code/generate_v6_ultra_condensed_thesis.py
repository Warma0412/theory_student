#!/usr/bin/env python3
"""Generate the fixed five-page thesis with benchmarks and ablations."""

from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from generate_v6_thesis import PAPER_DIR, RESULTS, V6_DIR, build_context


TEMPLATE = Path(__file__).with_name("thesis_ultra_condensed_template_v6.html")
OUTPUT_PDF = PAPER_DIR / "各版本汇总" / "论文_v6_极限精简版.pdf"


def html_table(frame: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    rows = []
    for _, row in frame.iterrows():
        cells = "".join(
            f"<td>{html.escape(str(row[column]))}</td>" for column in columns
        )
        rows.append(f"<tr>{cells}</tr>")
    return (
        f"<table><thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def selected_count(path: Path) -> tuple[int, set[str]]:
    frame = pd.read_csv(path)
    view = frame.loc[
        frame["q"].eq(0.20) & frame["selected_ebh"].astype(bool)
    ]
    return len(view), set(view["feature"])


def build_html() -> str:
    context = build_context()
    v3_metadata = json.loads(
        (PAPER_DIR / "v3" / "results" / "v3_shortlist_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    context["V3_DEFAULT_K"] = str(v3_metadata["recommended_default_k"])

    monthly_audit = json.loads(
        (PAPER_DIR / "v1" / "results" / "data_audit.json").read_text(
            encoding="utf-8"
        )
    )
    weekly_audit = json.loads(
        (RESULTS / "data_audit.json").read_text(encoding="utf-8")
    )
    grain = pd.DataFrame(
        [
            {
                "口径": "月度主分析",
                "观测": f"{monthly_audit['panel_rows']:,}",
                "期间": f"{monthly_audit['panel_months']}月",
                "零值率": f"{monthly_audit['next_month_zero_rate']:.1%}",
                "q=.20": "14项",
            },
            {
                "口径": "周度稳健性",
                "观测": f"{weekly_audit['panel_rows']:,}",
                "期间": f"{weekly_audit['panel_weeks']}周",
                "零值率": f"{weekly_audit['next_week_zero_rate']:.1%}",
                "q=.20": f"{context['WEEK_STRICT_COUNT']}项",
            },
        ]
    )
    context["ULTRA_GRAIN_TABLE"] = html_table(
        grain,
        ["口径", "观测", "期间", "零值率", "q=.20"],
        ["口径", "面板行", "期间数", "下一期零值率", "严格入选"],
    )

    overlap = pd.DataFrame(
        [
            {
                "类别": "月周共同",
                "数量": context["OVERLAP_COUNT"],
                "变量": context["OVERLAP_LIST"],
            },
            {
                "类别": "仅月度",
                "数量": context["MONTH_ONLY_COUNT"],
                "变量": context["MONTH_ONLY_LIST"],
            },
            {
                "类别": "仅周度",
                "数量": context["WEEK_ONLY_COUNT"],
                "变量": context["WEEK_ONLY_LIST"],
            },
        ]
    )
    context["ULTRA_OVERLAP_TABLE"] = html_table(
        overlap,
        ["类别", "数量", "变量"],
        ["类别", "数量", "变量"],
    )

    main_count, main_set = selected_count(RESULTS / "weekly_knockoff_primary.csv")
    sensitivity_specs = [
        ("主Copula", "weekly_knockoff_primary.csv"),
        ("原始高斯", "weekly_knockoff_gaussian.csv"),
        ("同周GMV", "weekly_same_week_sensitivity.csv"),
        ("截尾期", "weekly_trimmed_sensitivity.csv"),
        ("下一周AOV", "weekly_aov_sensitivity.csv"),
        ("测试前样本", "weekly_pretest_knockoff.csv"),
        ("XGBoost W", "weekly_xgboost_knockoff.csv"),
    ]
    sensitivity_rows = []
    for label, filename in sensitivity_specs:
        count, features = selected_count(RESULTS / filename)
        sensitivity_rows.append(
            {
                "情景": label,
                "严格数": count,
                "与主集重合": len(features & main_set),
            }
        )
    assert main_count == int(context["WEEK_STRICT_COUNT"])
    context["ULTRA_SENSITIVITY_TABLE"] = html_table(
        pd.DataFrame(sensitivity_rows),
        ["情景", "严格数", "与主集重合"],
        ["情景", "q=.20严格数", "与周度主集重合"],
    )

    metrics = pd.read_csv(RESULTS / "weekly_predictive_metrics.csv")
    metric_names = {
        "ExtraTrees": "Extra Trees",
        "XGBoost": "XGBoost全31维",
        "XGBoost-weekly-FDR-set": "XGBoost周度18项",
        "Naive-current-GMV": "本周GMV基线",
    }
    metric_view = metrics.loc[metrics["model"].isin(metric_names)].copy()
    metric_view["模型"] = metric_view["model"].map(metric_names)
    metric_view["RMSE"] = metric_view["rmse_log"].map(lambda value: f"{value:.3f}")
    metric_view["R2"] = metric_view["r2_log"].map(lambda value: f"{value:.3f}")
    metric_view["WAPE"] = metric_view["wape_raw"].map(lambda value: f"{value:.1%}")
    metric_view = metric_view.sort_values("rmse_log")
    context["ULTRA_METRIC_TABLE"] = html_table(
        metric_view,
        ["模型", "RMSE", "R2", "WAPE"],
        ["模型", "RMSE(log)", "R2", "WAPE"],
    )

    selection = pd.read_csv(RESULTS / "benchmark_selection_summary.csv")
    prediction = pd.read_csv(RESULTS / "benchmark_prediction_metrics.csv")
    fair_stability = pd.read_csv(RESULTS / "fair_stability_summary.csv")
    benchmark = selection.merge(
        prediction[["method", "rmse_log"]],
        on="method",
        how="left",
    ).merge(
        fair_stability[["method", "mean_jaccard"]],
        on="method",
        how="left",
    )
    benchmark["重合"] = benchmark["overlap_with_knockoff"].map(
        lambda value: f"{int(value)}/12"
    )
    benchmark["稳定性"] = benchmark["mean_jaccard"].map(
        lambda value: f"{value:.3f}"
    )
    benchmark["RMSE"] = benchmark["rmse_log"].map(lambda value: f"{value:.3f}")
    context["ULTRA_BENCHMARK_TABLE"] = html_table(
        benchmark,
        ["method", "重合", "稳定性", "RMSE"],
        ["方法", "主集重合", "稳定性", "RMSE"],
    )

    ablation = pd.read_csv(RESULTS / "component_ablation.csv")
    ablation = ablation.loc[
        ablation["setting"].isin(
            [
                "equicorrelated",
                "M=1",
                "M=60",
                "原始高斯-MVR-Lasso",
                "Copula-MVR-XGBoost",
            ]
        )
    ].copy()
    ablation["Jaccard"] = ablation["jaccard_with_main"].map(
        lambda value: f"{value:.3f}"
    )
    context["ULTRA_ABLATION_TABLE"] = html_table(
        ablation,
        ["setting", "repetitions", "selected_count", "Jaccard"],
        ["设定", "M", "严格数", "主集Jaccard"],
    )

    simulation = pd.read_csv(RESULTS / "simulation_method_comparison.csv")
    simulation["FDP"] = simulation["mean_fdp"].map(lambda value: f"{value:.3f}")
    simulation["Power"] = simulation["mean_power"].map(
        lambda value: f"{value:.3f}"
    )
    simulation["超标率"] = simulation["probability_fdp_above_q"].map(
        lambda value: f"{value:.0%}"
    )
    context["ULTRA_SIMULATION_TABLE"] = html_table(
        simulation,
        ["method", "FDP", "超标率", "Power"],
        ["方法", "平均FDP", "FDP>.20", "Power"],
    )

    document = TEMPLATE.read_text(encoding="utf-8")
    for key, value in context.items():
        document = document.replace("{{" + key + "}}", str(value))
    unresolved = sorted(set(re.findall(r"\{\{[A-Z0-9_]+\}\}", document)))
    if unresolved:
        raise ValueError(f"Unresolved ultra-condensed placeholders: {unresolved}")
    return document


def main() -> None:
    chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    document = build_html()
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="olist-v6-ultra-") as temporary:
        workdir = Path(temporary)
        html_path = workdir / "论文_v6_极限精简版.html"
        pdf_path = workdir / "论文_v6_极限精简版.pdf"
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
