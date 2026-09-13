#!/usr/bin/env python3
"""Generate the concise V5 thesis PDF."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve()
V5_DIR = HERE.parents[1]
PAPER_DIR = V5_DIR.parent
V4_DIR = PAPER_DIR / "v4"
TEMPLATE = HERE.with_name("thesis_condensed_template_independent.md")
sys.path.insert(0, str(V4_DIR / "code"))

from generate_v4_condensed_thesis import (  # noqa: E402
    build_markdown as build_v4_markdown,
    write_css,
)
from generate_v5_thesis import build_context, markdown_table, render  # noqa: E402


OUTPUT_PDF = PAPER_DIR / "论文_v5_精简版.pdf"


def build_markdown() -> str:
    context = build_context()
    text = build_v4_markdown()
    text = text.replace(
        "Olist聚类、组级与深度诊断精简介绍版（v4）",
        "Olist月度主分析与周度稳健性精简介绍版（v5）",
        1,
    )
    text = text.replace("date: 二〇二六年八月", "date: 二〇二六年九月", 1)

    abstract = """
V5在不改写月度确认性结论的前提下，新增卖家-周完整重跑。周度面板含{{WEEK_ROWS}}行、{{WEEK_SELLERS}}个卖家和{{WEEK_COUNT}}周，下一周零成交率{{WEEK_ZERO_RATE}}。相同60轮Copula-MVR与e-BH程序在0.20水平入选{{WEEK_STRICT_COUNT}}项；与月度14项共同入选{{OVERLAP_COUNT}}项，Jaccard={{JACCARD}}。卖家内层仍为0项，周度18项XGBoost测试RMSE为{{WEEK_SELECTED_RMSE}}。因此月度继续作为主分析，周度只承担粒度稳健性和短期预警证据。
""".strip()
    marker = (
        "**关键词：** 电商 GMV；聚类Knockoff；Group Knockoff；深度核MMD；"
        "GRIP2；FDR"
    )
    text = text.replace(marker, abstract + "\n\n" + marker + "；粒度稳健性", 1)

    method = """
## 3.7 V5周度粒度稳健性

V5按周一至周日构建卖家-周面板，用本周31项特征预测下一周`log(1+GMV)`。若卖家本周活跃、下一周无已送达订单，则目标记为零。周度分析复用月度主线的Copula-MVR、Lasso、60轮Knockoff、`α_kn=0.10`与`α_eBH=0.20`，并重跑高斯生成、同周目标、截尾窗口、下一周客单价、测试期前样本、XGBoost统计量、聚类分层、业务组、GRIP2、固定效应和时间外预测。

该设计沿用“最终已送达订单按购买周归属”的回溯口径，目的是隔离时间聚合粒度的影响。物流和评价在周末未必已经可见，因此结果不能直接等同于在线部署回放。

""".strip()
    text = text.replace(
        "# 第4章 核心实证结果",
        method + "\n\n# 第4章 核心实证结果",
        1,
    )

    results = """
## 4.10 V5周度结果

**表4-11 月度与周度口径**

{{WEEK_GRAIN_TABLE}}

周度Copula平均边际KS为{{WEEK_COPULA_KS}}，原始高斯为{{WEEK_GAUSSIAN_KS}}。周度0.20水平严格入选{{WEEK_STRICT_COUNT}}项：

{{WEEK_STRICT_LIST}}。

其中单轮频率至少90%的{{WEEK_STABLE_COUNT}}项为{{WEEK_STABLE_LIST}}。

**表4-12 周度严格集合及月度复现**

{{WEEK_SELECTION_TABLE}}

![图4-4 月度与周度入选频率](figures/fig_v5_monthly_weekly_frequency.png)

月周共同{{OVERLAP_COUNT}}项：{{OVERLAP_LIST}}。月度特有{{MONTH_ONLY_COUNT}}项：{{MONTH_ONLY_LIST}}；周度特有{{WEEK_ONLY_COUNT}}项：{{WEEK_ONLY_LIST}}。共同集合覆盖规模、价格、承运准备、商品内容和市场覆盖，是V5最重要的跨粒度证据。

**表4-13 周度时间外预测**

{{WEEK_METRIC_TABLE}}

周度最佳模型{{WEEK_BEST_MODEL}}的RMSE为{{WEEK_BEST_RMSE}}，较本周GMV朴素基线改善{{WEEK_RMSE_IMPROVEMENT}}。周度18项XGBoost的RMSE为{{WEEK_SELECTED_RMSE}}，相对全31维XGBoost变化{{WEEK_SELECTED_DELTA}}。

聚类分析在卖家间层入选{{WEEK_BETWEEN_COUNT}}项、卖家内层{{WEEK_WITHIN_COUNT}}项；Group Knockoff在0.20水平入选{{WEEK_GROUP_COUNT}}组；GRIP2式统计量在0.20水平{{WEEK_GRIP_Q20_COUNT}}项、0.30水平{{WEEK_GRIP_Q30_COUNT}}项。同周GMV对照选出29项，证实次周目标对避免机械定义关系是必要的。

""".strip()
    text = text.replace(
        "# 第5章 看板应用建议",
        results + "\n\n# 第5章 看板应用建议",
        1,
    )

    application = """
## 5.4 月周双层使用

月度层继续使用14项确认池和参数K，其中默认K=10；周度层用于短期异常监测。月周共同11项可标为跨粒度核心，周度特有7项先进入观察池。实际周度部署必须按字段真实可见时间重新构造物流、评价等特征，不能直接使用本研究的最终送达回溯口径。

""".strip()
    text = text.replace(
        "# 第6章 结论与局限",
        application + "\n\n# 第6章 结论与局限",
        1,
    )
    text = text.replace(
        "第四，以反对称性测试约束深度重要性。",
        (
            "第四，以反对称性测试约束深度重要性。第五，以完整周度重跑检验"
            "结论对时间聚合粒度的敏感性。"
        ),
        1,
    )
    text = text.split("\n# 版本说明", maxsplit=1)[0].rstrip() + "\n"
    text = render(text, context)
    unresolved = re.findall(r"\{\{[A-Z0-9_]+\}\}", text)
    if unresolved:
        raise ValueError(f"Unresolved V5 condensed placeholders: {unresolved}")
    return text


def build_independent_markdown() -> str:
    context = build_context()
    primary = pd.read_csv(PAPER_DIR / "v1" / "results" / "knockoff_primary.csv")
    strict = primary.loc[
        primary["q"].eq(0.20) & primary["selected_ebh"].astype(bool)
    ].copy()
    strict = strict.sort_values(
        ["selection_frequency", "mean_evalue", "mean_w"],
        ascending=False,
    )
    strict["frequency_fmt"] = strict["selection_frequency"].map(
        lambda value: f"{value:.1%}"
    )
    context["STRICT_COMPACT_TABLE"] = markdown_table(
        strict,
        ["label_zh", "mean_w", "frequency_fmt", "mean_evalue"],
        ["变量", "平均W", "单轮入选频率", "平均e-value"],
    )
    text = render(TEMPLATE.read_text(encoding="utf-8"), context)
    unresolved = re.findall(r"\{\{[A-Z0-9_]+\}\}", text)
    if unresolved:
        raise ValueError(
            f"Unresolved standalone condensed placeholders: {unresolved}"
        )
    return text


def main() -> None:
    text = build_independent_markdown()
    chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    with tempfile.TemporaryDirectory(prefix="olist-v5-condensed-") as temporary:
        workdir = Path(temporary)
        markdown_path = workdir / "论文_v5_精简版.md"
        html_path = workdir / "论文_v5_精简版.html"
        pdf_path = workdir / "论文_v5_精简版.pdf"
        css_path = workdir / "condensed.css"
        markdown_path.write_text(text, encoding="utf-8")
        write_css(css_path)
        css_path.write_text(
            css_path.read_text(encoding="utf-8")
            + "\nh1#参考文献 ~ p { font-size: 9.3pt; line-height: 1.38; "
            + "margin: 0.15em 0; }\n",
            encoding="utf-8",
        )
        subprocess.run(
            [
                "pandoc",
                str(markdown_path),
                "--from=markdown+tex_math_dollars",
                "--standalone",
                "--toc",
                "--toc-depth=2",
                "--mathml",
                "--embed-resources",
                f"--css={css_path}",
                f"--resource-path={workdir}:{V5_DIR}:{V4_DIR}",
                "-o",
                str(html_path),
            ],
            cwd=workdir,
            check=True,
        )
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
