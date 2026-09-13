"""Build documents from audited tables, preserving failures and frozen choices."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, FEATURES, KS, SEEDS, PROTOCOL, save_json
from validate_and_summarize import LABELS, COMPARATORS, read, cinterval

PROJECT = ROOT.parents[1]
TITLE = "面向电商GMV监控的深度特征选择与稳定性评价研究"
ENGLISH = "Deep Feature Selection and Stability Evaluation for E-commerce GMV Monitoring"
LATEX = ROOT / "latex工程"
SCENARIOS = {
    "linear_sparse": "线性稀疏强信号", "linear_medium": "线性中信号",
    "linear_weak": "线性弱信号", "linear_dense": "线性较密信号",
    "nonlinear": "非线性独立误差", "nonlinear_panel": "非线性组内相关",
}
DICTIONARY = pd.read_csv(ROOT / "results/feature_dictionary.csv")
NAMES = dict(zip(DICTIONARY.feature, DICTIONARY.label_zh))


def num(value, digits=4):
    return f"{float(value):.{digits}f}"


def interval(row, bounded=False):
    lo, hi = row["ci_low"], row["ci_high"]
    if bounded:
        lo, hi = max(0, lo), min(1, hi)
    return f"{num(row['mean'])} [{num(lo)}, {num(hi)}]"


def md_table(headers, rows):
    cells = lambda row: "| " + " | ".join(str(v).replace("|", "/").replace("\n", " ") for v in row) + " |"
    return "\n".join([cells(headers), cells(["---"] * len(headers))] + [cells(row) for row in rows])


def escape(text):
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "#": r"\#",
               "_": r"\_", "{": r"\{", "}": r"\}", "$": r"\$"}
    return "".join(mapping.get(ch, ch) for ch in str(text))


def pandoc(text, chapter=False):
    text = re.sub(r"^# 第\d+章\s+", "# ", text, flags=re.M)
    text = re.sub(r"^(#{2,4})\s+\d+(?:\.\d+)+\s+", r"\1 ", text, flags=re.M)
    command = ["pandoc", "-f", "markdown+tex_math_dollars", "-t", "latex", "--wrap=none"]
    if chapter:
        command.append("--top-level-division=chapter")
    out = subprocess.run(command, input=text, text=True, capture_output=True, check=True).stdout
    out = re.sub(r"\\label\{[^{}]+\}", "", out)
    out = re.sub(r"(\\begin\{longtable\}.*?\\end\{longtable\})",
                 r"{\n\\footnotesize\n\1\n}", out, flags=re.S)
    return out


def tables():
    pred = pd.read_csv(ROOT / "results/prediction_summary.csv")
    stable = pd.read_csv(ROOT / "results/stability_summary.csv")
    pairs = pd.read_csv(ROOT / "results/complementary_pair_runs.csv")
    alg = pd.read_csv(ROOT / "results/algorithm_randomness.csv")
    sim = pd.read_csv(ROOT / "results/simulation_summary.csv")
    ci = pd.read_csv(ROOT / "results/prediction_paired_intervals.csv")
    audit = read(ROOT / "results/data_audit.json")
    result = {}
    result["METHOD_TABLE"] = md_table(["方法", "文献年份", "本研究输出/范围"], [
        ["Copula-MVR聚合", "2018/2022/2024", "生成负对照、原生名单和排名；诊断有警告"],
        ["Elastic Net", "2005", "非零集合与系数排名"],
        ["稳定性选择", "2010/2013", "Lasso频率阈值与排名，经验适配"],
        ["影子变量树", "2006/2010", "Extra Trees影子比较，不是完整Boruta"],
        ["XGBoost-SHAP", "2016/2017/2020", "树模型解释排名"],
        ["Deep Lasso", "2023", "官方输入梯度惩罚与MLP训练"],
        ["VTFS定额适配", "2024", "官方Transformer骨干；固定预算和语料适配"],
        ["TabM置换", "2025", "官方集成网络；额外置换排名"],
        ["DeepDRK适配", "2024", "三种子训练和诊断后退出完整比较"],
        ["TabPFN v2", "2025", "预训练模型；MPS兼容失败"],
        ["attention-like过滤法", "2024", "仅文献核验，未计入训练结果"],
    ])
    result["DATA_TABLE"] = md_table(["阶段", "目标月份", "行数", "卖家数", "用途"], [
        [label, months, audit["split_rows"][key], audit["split_sellers"][key], purpose]
        for key, label, months, purpose in [
            ("train", "训练", "2017-02至2017-12", "预处理与模型拟合"),
            ("tune", "调参", "2018-01至2018-02", "冻结配置"),
            ("rank", "排序", "2018-03至2018-04", "名单、早停、开发选择"),
            ("test", "测试", "2018-05至2018-07", "仅评价"),
        ]])
    config_rows = []
    for method in COMPARATORS:
        config = read(ROOT / "results/configs" / f"{method}.json")
        params = config["params"]
        desc = ", ".join(f"{k}={v}" for k, v in params.items()) or "60轮；竞争惩罚0.01"
        config_rows.append([LABELS[method], len(config["trials"]), desc])
    result["CONFIG_TABLE"] = md_table(["方法", "配置数", "选定配置"], config_rows)
    rows = []
    for method in COMPARATORS + ["all_features"]:
        part = pred.loc[pred.method.eq(method) & pred.k.eq(31 if method == "all_features" else 12)]
        a = part.loc[part.evaluator.eq("xgboost")].iloc[0]
        b = part.loc[part.evaluator.eq("tabm")].iloc[0]
        rows.append([LABELS[method], num(a.rmse_log), num(b.rmse_log),
                     num(a.mae_log), num(a.r2_log), num(a.wape_raw)])
    result["PREDICTION_TABLE"] = md_table(
        ["选择方法", "XGB-RMSE", "TabM-RMSE", "XGB-MAE", "XGB-R²", "XGB-WAPE"], rows)
    rows = []
    for evaluator in ("xgboost", "tabm"):
        for _, r in ci.loc[ci.k.eq(12) & ci.evaluator.eq(evaluator) & ci.comparator.ne("vtfs")].iterrows():
            rows.append([evaluator, LABELS[r.comparator], num(r.rmse_difference),
                         f"[{num(r.ci95_low)}, {num(r.ci95_high)}]",
                         f"[{num(r.familywise95_low)}, {num(r.familywise95_high)}]"])
    result["PAIRED_PREDICTION_TABLE"] = md_table(
        ["评价器", "对照", "VTFS减对照", "95%区间", "同组多重比较区间"], rows)
    rows = []
    for method in COMPARATORS:
        part = stable.loc[stable.method.eq(method) & stable.k.eq(12)].set_index("fraction")
        rows.append([LABELS[method], interval(part.loc[.5], True), num(part.loc[.7, "mean"]),
                     num(part.loc[.8, "mean"]), num(part.loc[.5, "nogueira"])])
    result["STABILITY_TABLE"] = md_table(
        ["方法", "50%均值[组级95%区间]", "70%均值", "80%均值", "50% Nogueira"], rows)
    rows = []
    for method in COMPARATORS:
        values = pairs.loc[pairs.method.eq(method) & pairs.k.eq(12), "complementary_jaccard"]
        rows.append([LABELS[method], interval(cinterval(values), True), len(values)])
    result["COMPLEMENTARY_TABLE"] = md_table(["方法", "互补Jaccard[95%区间]", "互补组数"], rows)
    rows = []
    for method in COMPARATORS:
        v = alg.loc[alg.method.eq(method) & alg.k.eq(12), "reference_jaccard"]
        rows.append([LABELS[method], num(v.mean()), num(v.min()), num(v.max()), len(v)])
    result["ALGORITHM_TABLE"] = md_table(["方法", "均值", "最小值", "最大值", "种子批次"], rows)
    rows = []
    for evaluator in ("xgboost", "tabm"):
        for method in COMPARATORS:
            v = pred.loc[pred.method.eq(method) & pred.evaluator.eq(evaluator)].set_index("k")
            rows.append([evaluator, LABELS[method]] + [num(v.loc[k, "rmse_log"]) for k in KS])
    result["BUDGET_TABLE"] = md_table(["评价器", "选择方法"] + [f"K={k}" for k in KS], rows)
    rows = []
    for method in COMPARATORS:
        ref = read(ROOT / "results/runs" / method / "reference.json")
        labels = [NAMES[FEATURES[j]] for j in ref["order"][:12]]
        rows.append([LABELS[method], "；".join(labels)])
    result["SELECTED_TABLE"] = md_table(["方法", "按顺序列出的前12项"], rows)
    dl = read(ROOT / "results/deep_lasso_ablation.json")
    vt = pd.read_csv(ROOT / "results/vtfs_decoder_ablation.csv")
    half = vt.loc[vt.case.str.startswith("half")]
    result["ABLATION_TABLE"] = md_table(["检查项目", "数值", "解释范围"], [
        ["Deep Lasso保留正则的半样本Jaccard", interval(dl["with_penalty"], True), "固定配置"],
        ["去掉正则的半样本Jaccard", interval(dl["without_penalty"], True), "相同60半样本"],
        ["保留减去掉正则的配对差", interval(dl["paired_with_minus_without"]), "30个组，探索性"],
        ["完整样本两种正则设置名单Jaccard", num(dl["full_top12_jaccard"]), "Top-12"],
        ["VTFS完整名单等于语料回退", "是", "不归功于深度解码"],
        ["VTFS半样本与回退相同的比例", num(half.same_top12_as_corpus_fallback.mean()), "60个半样本"],
        ["VTFS解码候选的平均开发效用增益", num(half.validation_utility_gain_from_decoded_candidates.mean()),
         "标准化响应负MSE，仅开发期"],
    ])
    rows = []
    for method in COMPARATORS:
        ref = read(ROOT / "results/runs" / method / "reference.json")
        fits = ref["metadata"].get("fits", [])
        rows.append([LABELS[method], num(ref["total_seconds"], 2),
                     fits[0]["parameters"] if fits else "不适用",
                     "/".join(str(f["epochs"]) for f in fits) or "不适用"])
    result["RUNTIME_TABLE"] = md_table(["方法", "完整选择秒数", "单网络参数量", "三种子训练轮次"], rows)
    result["SCENARIO_TABLE"] = md_table(["场景", "真信号数", "信噪比", "重复"], [
        [SCENARIOS[s["name"]], s["signals"], s["snr"], 100] for s in PROTOCOL["simulation"]["scenarios"]])
    rows = []
    comments = []
    for s in PROTOCOL["simulation"]["scenarios"]:
        sub = sim.loc[sim.scenario.eq(s["name"]) & sim.rule.eq("top12")]
        for method in COMPARATORS:
            r = sub.loc[sub.method.eq(method)].iloc[0]
            rows.append([SCENARIOS[s["name"]], LABELS[method], num(r.FDP_mean), num(r.power_mean),
                         num(r.power_se), num(r.rmse_at_12_mean)])
        best = sub.sort_values("power_mean", ascending=False).iloc[0]
        tied = sub.loc[np.isclose(sub.power_mean, best.power_mean), "method"]
        comments.append(f"{SCENARIOS[s['name']]}场景中，Top-12检出率最高的点估计为"
                        f"{num(best.power_mean)}（{'、'.join(LABELS[m] for m in tied)}）")
    result["SIMULATION_FIXED_TABLE"] = md_table(
        ["场景", "方法", "平均FDP", "Power", "Power MCSE", "共同XGB-RMSE"], rows)
    result["SIMULATION_COMMENT"] = "；".join(comments) + "。这些逐场景点排名未进行跨全部场景的多重比较推断。"
    rows = []
    for s in PROTOCOL["simulation"]["scenarios"]:
        for method in ("copula_mvr", "elastic_net", "stability_selection", "shadow_trees"):
            r = sim.loc[sim.scenario.eq(s["name"]) & sim.rule.eq("native") & sim.method.eq(method)].iloc[0]
            rows.append([SCENARIOS[s["name"]], LABELS[method], num(r.FDP_mean), num(r.FDP_se),
                         num(r.power_mean), num(r.discoveries_mean, 2), num(r["FDP_gt_0.2_mean"])])
    result["SIMULATION_NATIVE_TABLE"] = md_table(
        ["场景", "方法", "平均FDP", "FDP MCSE", "Power", "发现数", "FDP>0.20比例"], rows)
    c = sim.loc[sim.method.eq("copula_mvr") & sim.rule.eq("native")]
    result["NATIVE_COMMENT"] = (
        f"Copula-MVR原生规则在六场景中的平均FDP范围为{num(c.FDP_mean.min())}至"
        f"{num(c.FDP_mean.max())}，平均Power范围为{num(c.power_mean.min())}至"
        f"{num(c.power_mean.max())}。这是当前有限场景、固定配置的经验结果；"
        "观察均值低于目标不能证明所有分布上的FDR控制，检出率低也必须与空名单和阈值约束结合解释。")
    copula = read(ROOT / "results/copula_diagnostic.json")
    deep = read(ROOT / "results/runs/deepdrk/reference.json")["metadata"]["diagnostics"]
    rows = []
    for name, rec in [("Copula-MVR", copula)] + [(f"DeepDRK种子{s}", r) for s, r in zip(SEEDS, deep)]:
        rows.append([name, num(rec["mean_marginal_ks"])] +
                    [num(r["auc"]) for r in rec["classifier_tests"]] + ["警告"])
    result["DIAGNOSTIC_TABLE"] = md_table(["生成器", "平均边际KS", "25% AUC", "50% AUC", "100% AUC", "状态"], rows)
    rows = []
    for i, r in DICTIONARY.iterrows():
        kind = "二元" if r.feature == "has_marketing_deal" else (
            "计数" if r.feature in ("item_count", "order_count", "category_count",
                                   "unique_customer_count", "buyer_state_diversity") else "连续聚合量")
        rows.append([i + 1, r.label_zh, kind, f"{num(r['min'], 3)}至{num(r['max'], 3)}",
                     int(r["unique"]), f"{r.missing_fraction:.2%}"])
    result["DICTIONARY_TABLE"] = md_table(["编号", "中文含义", "类型", "观测范围", "取值数", "缺失率"], rows)
    result["FIDELITY_TABLE"] = md_table(["组件", "来源", "本研究差异"], [
        ["Deep Lasso", "官方正则函数", "自建两层MLP，三种子，输入损失梯度排序"],
        ["TabM", "官方0.0.3软件包", "8成员；验证置换选择器另行附加"],
        ["VTFS", "官方编码器/解码器", "随机语料、CPU、唯一token、定额解码；保留epsilon=1"],
        ["DeepDRK", "锁定官方commit", "64宽2层；修正逐列提前返回；MPS；附加多轮聚合"],
        ["Knockoff竞争", "对称系数差", "L1比例0.999的Elastic Net近似"],
        ["稳定性选择", "互补半样本思想", "40次Lasso与0.9经验阈值，无直接PFER/FDR声明"],
        ["影子树", "负对照与Extra Trees", "8次、胜率0.75，无完整Boruta复现声明"],
        ["TabPFN v2", "2.0.9软件包", "使用预训练权重；MPS失败，未下采样补结果"],
    ])
    return result


def prepare_references(source):
    registry = read(ROOT / "references/papers.json")
    lookup = {r["key"]: r for r in registry}
    keys = []
    for group in re.findall(r"\\cite\{([^}]+)\}", source):
        for key in group.split(","):
            assert key in lookup, key
            if key not in keys:
                keys.append(key)
    keys += [r["key"] for r in registry if r["key"] not in keys]
    entries = []
    tex = [r"\begin{thebibliography}{99}"]
    for i, key in enumerate(keys, 1):
        r = lookup[key]
        kind = "DB/OL" if r["key"] in ("olist2018", "funnel2018") else (
            "C" if r["key"] in ("deeplasso2023", "tabm2025", "deepdrk2024", "xgb2016", "shap2017") else "J")
        publication = f"{r['authors'].rstrip('.')}. {r['title']}[{kind}]. {r['venue']}, {r['year']}."
        address = ("https://doi.org/" + r["doi"]) if "doi" in r else r.get("url", "")
        entries.append(f"[{i}] {publication} {address}")
        tex.append(rf"\bibitem{{{key}}} {escape(publication)}" +
                   (rf" \url{{{address}}}." if address else ""))
    tex.append(r"\end{thebibliography}")
    return "\n\n".join(entries), "\n".join(tex), {key: i for i, key in enumerate(keys, 1)}


COMMON_TEX = r"""
\usepackage{fontspec,longtable,booktabs,array,calc,url,seqsplit,etoolbox,pdflscape}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\pandocbounded}[1]{#1}
\setlength{\LTleft}{0pt plus 1fill}
\setlength{\LTright}{0pt plus 1fill}
\renewcommand{\arraystretch}{1.12}
\setlength{\emergencystretch}{2em}
\makeatletter
\def\maxwidth{\ifdim\Gin@nat@width>\linewidth\linewidth\else\Gin@nat@width\fi}
\def\maxheight{\ifdim\Gin@nat@height>.70\textheight .70\textheight\else\Gin@nat@height\fi}
\makeatother
\setkeys{Gin}{width=\maxwidth,height=\maxheight,keepaspectratio}
\urlstyle{same}
"""


def copy_assets():
    template = PROJECT / "模板/应用统计专硕latex模板"
    for name in ("论文", "开题报告", "开题汇报"):
        dest = LATEX / name
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ROOT / "figures", dest / "figures", dirs_exist_ok=True)
    shutil.copy2(template / "sufethesismas.cls", LATEX / "论文")
    for name in ("SHUFEBadge.jpg", "SHUFEName.jpg", "statement.pdf"):
        shutil.copy2(template / "figures" / name, LATEX / "论文/figures")
    shutil.copy2(PROJECT / "模板/latex汇报ppt模板/SUFE.sty", LATEX / "开题汇报")
    shutil.copytree(PROJECT / "模板/latex汇报ppt模板/pic", LATEX / "开题汇报/pic", dirs_exist_ok=True)


def build_thesis(source, bibliography, numbers):
    a = source.split("# 摘要\n", 1)[1].split("# Abstract", 1)[0]
    zh, kwzh = a.split("**关键词：**")
    b = source.split("# Abstract\n", 1)[1].split("# 第1章", 1)[0]
    en, kwen = b.split("**Key words:**")
    body = "# 第1章" + source.split("# 第1章", 1)[1].split("# 参考文献", 1)[0]
    appendix = "# 附录A" + source.split("# 附录A", 1)[1]
    appendix_tex = []
    for part in re.split(r"(?=^# 附录[A-Z])", appendix, flags=re.M):
        if not part.strip():
            continue
        title, content = part.split("\n", 1)
        appendix_tex.append(r"\chapterx{" + title[2:] + "}\n" + pandoc(content))
    tex = r"\documentclass{sufethesismas}" + COMMON_TEX + rf"""
\begin{{document}}
\classification{{}}\confidential{{}}\UDC{{}}\serialnumber{{}}
\title{{面向电商GMV监控的\\深度特征选择与稳定性评价研究}}
\englishtitle{{{ENGLISH}}}
\author{{吴优}}\advisor{{张吕欧}}
\major{{应用统计硕士专业学位论文}}
\completedate{{2026年9月}}\department{{统计与数据科学学院}}
\school{{上海财经大学}}\studentidnumber{{2025213385}}
\maketitle
\frontmatter
\begin{{abstractCN}}{pandoc(zh)}\end{{abstractCN}}
\keywordsCN{{{kwzh.strip()}}}
\begin{{abstractEN}}{pandoc(en)}\end{{abstractEN}}
\keywordsEN{{{kwen.strip()}}}
\tableofcontents
\mainmatter
{pandoc(body, True)}
\backmatter
{bibliography}
{chr(10).join(appendix_tex)}
\end{{document}}
"""
    (LATEX / "论文/论文.tex").write_text(tex)
    readable = re.sub(r"\\cite\{([^}]+)\}", lambda m: "[" + ", ".join(
        str(numbers[k]) for k in m.group(1).split(",")) + "]", source)
    (ROOT / "论文_v7.md").write_text("# " + TITLE + "\n\n" + readable)


def build_report(table, bibliography):
    body = rf"""
# 选题背景与研究问题

电商经营看板包含交易、商品、支付、履约和市场覆盖等大量指标。研究拟在有限展示容量下选择值得持续监控的指标，并评价名单对卖家样本变化的敏感性。本文以本月31项指标与下一月签收确认商品金额的预测关系为评价基础，不将预测重要性解释为当期金额拆解或因果效应。

近期深度特征选择提供了输入梯度正则、序列子集搜索和参数高效集成等新工具，但其在真实电商面板上的增量价值仍需要与稀疏回归、树模型和Knockoff对照检验。研究不预设复杂方法必须胜出，而围绕信息保留、稳定性和误选边界建立统一评价。

# 文献依据与方法

Lasso与Elastic Net提供可解释的稀疏基准\cite{{lasso1996,elastic2005}}；稳定性选择与互补半样本为样本扰动设计提供依据\cite{{stability2010,cpss2013}}；Nogueira指标用于机会校正稳定性\cite{{nogueira2018}}。Model-X、MVR、多轮e-value与e-BH构成带有明确假设的Knockoff参照\cite{{modelx2018,mvr2022,derand2024,ebh2022}}。

深度候选包括2023年的Deep Lasso、2024年的VTFS与DeepDRK，以及2025年的TabM和TabPFN v2\cite{{deeplasso2023,vtfs2024,deepdrk2024,tabm2025,tabpfn2025}}。Deep Lasso调用官方输入梯度惩罚；TabM使用官方网络并附加验证置换排序；VTFS使用官方Transformer骨干进行定额适配。DeepDRK完成三种子训练后出现生成诊断警告；TabPFN在完整样本上出现MPS内存错误，均不冒充完成全部比较。

Table: 方法及完成范围

{table["METHOD_TABLE"]}

# 数据与信息时间

数据来自Olist电商与营销漏斗公开数据集\cite{{olist2018,funnel2018}}。十张原始表中包含99,441条订单和112,650条商品明细。支付、评价先按自然层级聚合，商品金额由明细价格求和，运费不计入目标，避免多对多连接放大金额。

物流与评价仅使用月末已发生的事件，营销成交按成交日期截断。最终得到13,754个卖家月、2,723名卖家、18个特征月份与31项指标，下一月零值率17.8784%。主目标为下一月按真实签收月份确认的商品金额的log(1+GMV)，不是按购买月份回溯的最终送达金额。支付与元数据没有完整修订时间，其购买时可得性仍是工作假设。

Table: 研究窗口

{table["DATA_TABLE"]}

# 核心方法与公平原则

所有方法使用相同卖家抽样清单。30组互补50%样本产生60个半样本，另运行20个70%和20个80%样本。同一卖家保留全部相应月份。主预算为12项，另比较4、8、10和14项；下游XGBoost与TabM配置对所有选择器一致。固定数据的10组种子试验与样本扰动结果分别报告。

Deep Lasso优化预测损失及输入梯度组惩罚：

$$
\mathcal L_{{DL}}=\mathcal L_B+
\lambda\sum_j\left\|\partial\mathcal L_B/\partial X_{{B,j}}\right\|_2.
$$

TabM的8个集成成员分别计算训练损失，推断时平均；验证列置换增加的MSE作为重要性。VTFS用96个随机子集效用训练编码器和解码器，在不超过128次查询内按预设K形成名单，明确保留随机语料回退与固定编号顺序的适配限制。

Knockoff采用真假系数绝对值差W，60轮证据聚合后用e-BH生成原生名单。单轮0.10、最终0.20和高频入选是不同概念；强制Top-K集合不自动继承FDR保证。所有实际FDR声明还依赖生成有效性与依赖结构条件。

# 已完成的实证与模拟

八种方法各完成111个真实选择案例和600个模拟案例，形成82组统一预测产物。测试结果没有支持近期深度模型全面领先。

Table: 主预算测试表现

{table["PREDICTION_TABLE"]}

TabM置换与Deep Lasso在XGBoost评价器下接近最优，但在TabM评价器下XGBoost-SHAP仍很强。开发期选出的VTFS适配模型没有保持样本外优势，不在测试后更换主方法以制造胜出结论。

Table: 样本稳定性

{table["STABILITY_TABLE"]}

影子变量树在半样本名单一致性上突出；固定数据种子稳定性与更换卖家的稳定性不可混同。完整样本的VTFS最终名单与随机语料回退相同，不能将这次选择收益归功于深度解码器。

模拟包括六场景、每场景100次，已知真信号后分别评价原生名单和Top-K的FDP及Power。固定12项且只有5项真信号时，FDP至少为7/12，不能据此混用原生错误率与容量约束。{table["NATIVE_COMMENT"]}

# 可行性、限制与后续计划

数据处理、官方代码适配、重复实验与结果汇总均已实际完成，后续工作的重点转向外部验证与针对性方法改进。本文的应用统计工作是将时间可见性、相同样本扰动、双评价器、容量路径和生成诊断落实到同一研究任务，并明确它们分别回答的问题。

当前需要注意三项边界：公开数据静态快照不能完全还原实时可见性；三个月测试期不足以验证长期漂移；Copula-MVR与本次DeepDRK均出现联合交换性诊断警告，真实数据不能无条件宣称FDR受控。

后续优先补充新的时间外数据；进一步研究重新调参后的全流程稳定性；对VTFS实施独立K优化与等查询预算随机搜索对照；对生成器开展卖家层诊断及独立训练；最后评估月度名单在周度业务预警中的可迁移性。这些扩展尚待验证，不计入当前完成结论。
"""
    cover = (ROOT / "report_cover.tex.in").read_text().replace("@@TITLE@@", TITLE)
    tex = r"""\documentclass[UTF8,12pt,a4paper]{ctexart}
\usepackage[left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm]{geometry}
\usepackage{amsmath,amssymb,graphicx,hyperref,fancyhdr,setspace,tabularx,enumitem}
\setCJKmainfont{Songti SC}
\setCJKsansfont{STHeiti}
\setmainfont{Times New Roman}
\hypersetup{hidelinks}
\setstretch{1.32}
\setlength{\parindent}{2em}
\setlist{nosep}
\pagestyle{fancy}\fancyhf{}\fancyhead[R]{\small\thepage}
\renewcommand{\headrulewidth}{0pt}
""" + COMMON_TEX + "\n\\begin{document}\n" + cover + pandoc(body) + bibliography + "\n\\end{document}\n"
    (LATEX / "开题报告/开题报告.tex").write_text(tex)
    (ROOT / "开题报告_v7.md").write_text("# " + TITLE + "\n\n" + body)


def small_tabular(headers, rows, spec=None):
    spec = spec or ("l" + "r" * (len(headers) - 1))
    return (r"\begin{tabular}{" + spec + "}\n\\toprule\n" +
            " & ".join(escape(c) for c in headers) + r"\\\midrule" + "\n" +
            "\n".join(" & ".join(escape(c) for c in row) + r"\\" for row in rows) +
            "\n\\bottomrule\n\\end{tabular}")


def build_slides():
    p = pd.read_csv(ROOT / "results/prediction_summary.csv")
    s = pd.read_csv(ROOT / "results/stability_summary.csv")
    sim = pd.read_csv(ROOT / "results/simulation_summary.csv")
    frames = []
    def frame(title, content):
        frames.append(r"\begin{frame}{" + title + "}\n" + content + "\n\\end{frame}")
    def items(lines):
        return "\\begin{itemize}\n" + "\n".join(r"\item " + escape(line) for line in lines) + "\n\\end{itemize}"
    frame("研究问题：有限容量的经营看板", items([
        "从31项经营指标中，选择值得持续监控的少量指标。",
        "预测任务提供信息保留标准；稳定性反映名单对卖家变化的敏感性。",
        "固定数量的推荐、原生错误控制和因果归因是不同问题。",
        "不预设复杂深度模型一定优于统计或树模型。"]))
    frame("真实文献与候选方法", r"\small " + small_tabular(
        ["方法", "年份", "出处/角色"], [
            ["Deep Lasso", "2023", "NeurIPS D&B；输入梯度正则"],
            ["VTFS", "2024", "ACM TKDD；序列生成，定额适配"],
            ["DeepDRK", "2024", "NeurIPS；深度Knockoff生成"],
            ["TabM", "2025", "ICLR；参数高效集成"],
            ["TabPFN v2", "2025", "Nature；预训练表格模型"],
        ], "lll") + r"\vspace{0.5em}" + items(["对照：Copula-MVR、Elastic Net、稳定性选择、影子树、XGBoost-SHAP。"]))
    frame("Olist数据与研究人口", items([
        "十张原始表：99,441条订单，112,650条商品明细。",
        "13,754个卖家月，2,723名卖家，18个特征月份。",
        "候选指标31项；本月有购买明细的卖家进入分析。",
        "下一月签收确认商品金额；不含运费，不等于利润或净收入。",
        "下一月目标为零的比例为17.8784%。"]))
    frame("信息时间：月末只能使用已可见事件", items([
        "审核、承运、送达与评价按发生时间截断。",
        "营销成交按成交日期截断；州频率使用当时已观察卖家。",
        "支付、商品和卖家元数据缺少完整修订历史，仍有工作假设。",
        "预测选择不等于按品类、地区拆分当期GMV的会计贡献。"]))
    frame("时间切分与名单冻结", r"\small " + small_tabular(
        ["阶段", "目标月份", "行数"], [
            ["训练", "2017-02至2017-12", "6,567"],
            ["调参", "2018-01至2018-02", "1,831"],
            ["排序", "2018-03至2018-04", "1,943"],
            ["测试", "2018-05至2018-07", "3,413"],
        ], "llr") + items([
            "测试前冻结主方法和K；预测器在全部开发数据重拟合。",
            "测试月份曾参与前期探索，不是全新外部确认样本。"]))
    frame("三种深度选择机制", items([
        "Deep Lasso：训练时惩罚输入损失梯度，再按梯度分数排序。",
        "TabM：训练8成员共享权重网络，再看验证集置换误差增量。",
        "VTFS：从子集效用语料学习Transformer表示并搜索候选序列。",
        "三者都实际训练，但不自动获得FDR或因果解释。"]))
    frame("Deep Lasso：正则与输入选择", r"""
\[
\mathcal L_{\mathrm{DL}}=\mathcal L_B+
\lambda\sum_j\|\partial\mathcal L_B/\partial X_{B,j}\|_2
\]
""" + items([
        "直接调用作者发布的正则函数，两层MLP，三个训练种子。",
        "验证早停；不使用测试集训练或选择检查点。",
        "补充去掉正则的同样本消融，检验当前配置下的局部作用。"]))
    frame("TabM：预测器与选择器分开评价", items([
        "官方2025年参数高效集成网络，8成员分别计算训练损失。",
        "每列置换3次，以MSE增加量形成排名。",
        "所有名单分别交给统一XGBoost与TabM重新预测。",
        "双评价器用于识别名单收益是否依赖特定预测模型。"]))
    frame("VTFS：适配与可解释的负结果", items([
        "使用官方编码器、解码器；96个随机子集语料。",
        "固定预算唯一token解码；效用查询不超过128次。",
        "保留公开代码中的常数重参数化，非完整原论文复现。",
        "完整样本最终名单来自语料回退，本次收益不能归功于解码器。"]))
    frame("公平稳定性：每种方法面对同一批卖家", items([
        "30组互补50%样本，共60个半样本；卖家全部月份共同保留。",
        "20个70%样本、20个80%样本，共100个外层扰动。",
        "另有10组固定数据的种子诊断，与样本稳定性分开。",
        "比较K=4、8、10、12、14；主预算预设为12。",
        "50%区间按30组计算，不把60个半样本视为独立。"]))
    frame("评价准则：各自回答不同问题", items([
        "RMSE：保留的特征能否帮助下一期预测。",
        "Jaccard：名单名称是否一致；Nogueira进行机会校正。",
        "FDP：发现中假变量的比例；Power：真变量被找到的比例。",
        "真实业务没有已知真值，不能直接计算真实FDP。",
        "固定12项而只有5项真信号时，FDP至少为7/12。"]))
    rows = []
    for m in COMPARATORS + ["all_features"]:
        part = p.loc[p.method.eq(m) & p.k.eq(31 if m == "all_features" else 12)].set_index("evaluator")
        rows.append([LABELS[m], num(part.loc["xgboost", "rmse_log"]), num(part.loc["tabm", "rmse_log"])])
    frame("同样12项：测试预测结果", r"\footnotesize " +
          small_tabular(["选择方法", "XGBoost RMSE", "TabM RMSE"], rows) +
          r"\vspace{0.4em}\par\small 开发主选择VTFS未保持优势；TabM、Deep Lasso与树模型各有竞争力。")
    rows = []
    for m in COMPARATORS:
        v = s.loc[s.method.eq(m) & s.k.eq(12)].set_index("fraction")
        rows.append([LABELS[m]] + [num(v.loc[f, "mean"], 3) for f in (.5, .7, .8)])
    frame("相同卖家扰动：Top-12稳定性", r"\footnotesize " +
          small_tabular(["方法", "50%", "70%", "80%"], rows) +
          r"\vspace{0.4em}\par\small 表中为相对完整开发名单的Jaccard；稳定性不等于正确率。")
    frame("展示容量：不能只看一条最有利结果",
          r"\centering\includegraphics[width=.94\linewidth,height=.66\textheight,keepaspectratio]{figures/05_budget_path.png}"
          r"\par\small VTFS按K=12优化顺序，其小K前缀不是独立优化结果。")
    rows = []
    for scenario in PROTOCOL["simulation"]["scenarios"]:
        part = sim.loc[sim.rule.eq("top12") & sim.scenario.eq(scenario["name"])].set_index("method")
        rows.append([SCENARIOS[scenario["name"]]] +
                    [num(part.loc[m, "power_mean"], 3) for m in ("xgboost_shap", "deep_lasso", "tabm", "vtfs")])
    frame("已知真值：每场景100次模拟", r"\footnotesize " +
          small_tabular(["Top-12 Power", "SHAP", "D.Lasso", "TabM", "VTFS"], rows) +
          items(["全部八方法共享生成数据；完整FDP、Power和原生名单见论文。",
                 "跨场景固定超参数，结果不代表各场景充分调参后的上界。"]))
    frame("生成诊断：不能无条件宣称真实FDR受控",
          r"\centering\includegraphics[width=.87\linewidth,height=.61\textheight,keepaspectratio]{figures/09_generator_diagnostic.png}"
          r"\par\small Copula-MVR与DeepDRK均有联合失配警告；诊断不通过不等于反证原论文。")
    dl = read(ROOT / "results/deep_lasso_ablation.json")
    frame("负结果与局部消融", items([
        "VTFS完整样本解码候选的开发效用增益为零，最终名单等于回退名单。",
        f"Deep Lasso保留/移除正则的半样本Jaccard：{num(dl['with_penalty']['mean'])}"
        f" / {num(dl['without_penalty']['mean'])}；仅为固定配置消融。",
        "DeepDRK只完成参考训练与诊断；TabPFN保留MPS失败记录。",
        "失败方法不填零、不补造结果，也不声称完成全部公平比较。"]))
    frame("结论与后续研究计划", items([
        "没有全面优胜者，深度结构复杂不等于名单质量更高。",
        "按新时间窗口验证TabM、Deep Lasso与强树基线。",
        "加强按K独立优化、等查询预算随机搜索和嵌套调参比较。",
        "研究混合变量与卖家依赖下的生成有效性，再讨论原生FDR。",
        "月度向周度迁移属于后续验证，不沿用未验证的多粒度结论。"]))
    selected = ["deeplasso2023", "vtfs2024", "tabm2025", "deepdrk2024", "tabpfn2025",
                "modelx2018", "derand2024", "ebh2022", "stability2010", "cpss2013", "nogueira2018", "morris2019"]
    lookup = {r["key"]: r for r in read(ROOT / "references/papers.json")}
    text = r"\begin{columns}[T]\begin{column}{.49\textwidth}\fontsize{6}{7.4}\selectfont"
    for i, key in enumerate(selected):
        if i == 6:
            text += r"\end{column}\begin{column}{.49\textwidth}\fontsize{6}{7.4}\selectfont"
        r = lookup[key]
        text += escape(f"[{i+1}] {r['authors']}. {r['title']}. {r['venue']}, {r['year']}.") + r"\par\vspace{0.4em}" + "\n"
    text += r"\end{column}\end{columns}\vfill{\scriptsize 完整26项来源及可访问链接见论文参考文献。}"
    frame("主要参考文献", text)
    assert len(frames) == 19
    tex = rf"""\documentclass[aspectratio=169]{{beamer}}
\usepackage{{ctex,hyperref,amsmath,booktabs,array,graphicx}}
\usepackage{{SUFE}}
\setbeamertemplate{{navigation symbols}}{{}}
\setbeamertemplate{{headline}}{{}}
\setbeamersize{{text margin left=7mm,text margin right=7mm}}
\setbeamerfont{{frametitle}}{{size=\large}}
\setbeamerfont{{normal text}}{{size=\small}}
\setbeamerfont{{frame number}}{{parent=author in head/foot}}
\title[深度特征选择与稳定性评价]{{{TITLE}}}
\subtitle{{相同样本扰动、双评价器与真实负结果}}
\author{{吴优\quad 2025213385}}
\institute[上海财经大学]{{上海财经大学统计与数据科学学院\\指导教师：张吕欧}}
\date{{2026年9月}}
\begin{{document}}
\begin{{frame}}
\titlepage
\centering\includegraphics[width=.19\linewidth]{{pic/sufe.png}}
\end{{frame}}
{chr(10).join(frames)}
\end{{document}}
"""
    (LATEX / "开题汇报/开题汇报.tex").write_text(tex)


def build_short_documents(rendered, bibliography):
    abstract = rendered.split("# 摘要\n", 1)[1].split("**关键词", 1)[0]
    for name, full in [("精简版", True), ("极限精简版", False)]:
        if full:
            sections = []
            for start, end in [("# 第3章", "# 第4章"), ("# 第5章", "# 第6章"),
                               ("# 第6章", "# 第7章"), ("# 第8章", "# 参考文献")]:
                sections.append(rendered.split(start, 1)[1].split(end, 1)[0])
            text = "# 研究摘要\n\n" + abstract + "\n\n" + "\n\n".join("# " + s.strip() for s in sections)
        else:
            t = tables()
            examples = []
            for method in ("deep_lasso", "tabm", "xgboost_shap"):
                ref = read(ROOT / "results/runs" / method / "reference.json")
                examples.append([LABELS[method], "；".join(NAMES[FEATURES[j]] for j in ref["order"][:12])])
            text = ("# 数据、目标与方法\n\n"
                    "研究以预测信息保留作为经营看板选维的评价标准，从31项指标中选择少量持续监控项，"
                    "不把预测性选择混同于按品类拆解当期GMV或识别因果效应。\n\n"
                    "Olist十张原始表包含99,441条订单与112,650条商品明细。按月末可见时间重建后，"
                    "得到13,754个卖家月、2,723名卖家和18个特征月份。目标是次月签收确认商品金额的"
                    "log(1+GMV)，不含运费。物流、评价和营销成交按时间遮蔽；静态快照仍有可见性假设。\n\n" +
                    t["DATA_TABLE"] + "\n\n"
                    "Deep Lasso通过官方输入梯度正则训练两层MLP；TabM训练8成员参数高效集成网络，"
                    "再用验证置换排序；VTFS使用官方Transformer骨干、随机子集语料和固定K解码。"
                    "三者均真实训练，但VTFS是明确披露的适配，不是完整原论文复现。\n\n"
                    "五种对照为Copula-MVR、Elastic Net、稳定性选择、影子变量树及XGBoost-SHAP。"
                    "八方法共享30组互补50%样本、20个70%样本、20个80%样本，并另做10个种子批次。"
                    "固定数据随机性与样本变化分开；半样本区间以30个组为单位。\n\n"
                    "\\newpage\n\n# 预测与名单稳定性\n\n"
                    "主预算为12项；所有名单使用相同XGBoost和TabM评价器，测试集不参与训练和选K。\n\n" +
                    t["PREDICTION_TABLE"] + "\n\n" + t["STABILITY_TABLE"] +
                    "\n\nTabM和Deep Lasso在树评价器中有竞争力，但没有跨评价器全面领先。"
                    "影子树名单最稳定不等于每种预算下预测最好。开发期主选择为VTFS，"
                    "其测试误差为1.8625和1.8851，未保持开发优势，也未在测试后被替换为其他主方法。"
                    "所有区间只以三个已观察测试月份为条件。\n\n"
                    "\\newpage\n\n# 模拟、误选与检出\n\n"
                    "六场景各100次，共4,800个方法案例。原生名单与固定12项名单分开评价。"
                    "FDP是发现中假变量比例，Power是真变量检出率；空名单二者均为零。"
                    "固定12项且只有5个真信号时，FDP下限为7/12，因此不能混比原生错误率与容量约束。\n\n" +
                    t["SIMULATION_NATIVE_TABLE"] +
                    "\n\nCopula-MVR在稀疏强信号与两个非线性场景均返回空原生名单，"
                    "低FDP必须同时考虑零功效。固定Top-12下，XGBoost-SHAP在两个非线性场景"
                    "的Power为0.9640和0.9700，仍高于近期网络。所有配置迁移自真实开发任务，"
                    "不是逐场景充分调参后的性能上界。\n\n"
                    "\\newpage\n\n# 具体名单、结论与边界\n\n" +
                    md_table(["选择器", "完整开发样本Top-12"], examples) +
                    "\n\n完整样本VTFS最终名单来自随机语料回退，不能归功于深度解码。"
                    "Deep Lasso正则消融的稳定性配对差仅0.0049，95%区间跨零。"
                    "Copula-MVR及DeepDRK有联合交换性诊断警告，真实名单不具无条件FDR保证；"
                    "TabPFN是MPS运行失败，不填入替代结果。\n\n"
                    "结论是多目标权衡而非复杂模型全面获胜。下一步优先使用新时间窗口复核名单，"
                    "并研究按K独立搜索、完整调参稳定性和卖家层生成诊断。"
                    "指标入选不代表永久有效或具有因果效应。\n\n"
                    "## 主要近期文献\n\n"
                    "Deep Lasso，NeurIPS Datasets and Benchmarks 2023；"
                    "VTFS，ACM TKDD 2024，DOI:10.1145/3687485；"
                    "DeepDRK，NeurIPS 2024；TabM，ICLR 2025；"
                    "TabPFN v2，Nature 2025，DOI:10.1038/s41586-024-08328-6。")
        dest = ROOT / f"论文_v7_{name}.tex"
        pre = r"""\documentclass[UTF8,11pt,a4paper]{ctexart}
\usepackage[left=2.2cm,right=2.2cm,top=2cm,bottom=2cm]{geometry}
\usepackage{amsmath,amssymb,graphicx,hyperref,setspace}
\setCJKmainfont{Songti SC}\setmainfont{Times New Roman}
\setstretch{1.18}\hypersetup{hidelinks}
"""
        dest.write_text(pre + COMMON_TEX + "\n\\begin{document}\n" +
                        r"\begin{center}\large\bfseries " + TITLE + r"\end{center}" +
                        pandoc(text) + (bibliography if full else "") + "\n\\end{document}")
        (ROOT / f"论文_v7_{name}.md").write_text("# " + TITLE + "\n\n" + text)


def main():
    assert read(ROOT / "results/final_audit.json")["passed"]
    source = (ROOT / "thesis.md.in").read_text()
    reference_md, reference_tex, numbers = prepare_references(source)
    content = tables() | {"REFERENCES": reference_md}
    for key, value in content.items():
        source = source.replace("@@" + key + "@@", value)
    assert "@@" not in source
    copy_assets()
    build_thesis(source, reference_tex, numbers)
    build_report(content, reference_tex)
    build_slides()
    build_short_documents(source, reference_tex)
    save_json(ROOT / "results/document_generation.json", {
        "title": TITLE, "reference_count": len(numbers), "slides": 20,
        "tables_from": "audited result CSV and JSON", "unresolved_placeholders": 0})
    print("Generated thesis, report, 20 slides, and two condensed sources")


if __name__ == "__main__":
    main()
