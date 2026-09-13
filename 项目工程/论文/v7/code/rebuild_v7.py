"""Rebuild standalone V7 deliverables from separate audited experiment designs."""

from pathlib import Path
import json
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ROOT, digest_file, save_json
import generate_documents as old

PROJECT = ROOT.parents[1]
SCPL = PROJECT / "论文/方法研究_20260913"
SUP = ROOT / "supplement_20260913"
MECH = ROOT / "model_improvements_20260912"
TITLE = "面向电商经营指标筛选的非线性方法与交换对称损失研究"
ENGLISH = "Nonlinear Feature Selection and Swap-Canonical Loss for E-commerce Indicators"
LABELS = {
    "lasso_pair_q20": "评分集成对Lasso", "linear_orbit_student": "匹配线性对称差",
    "xgb_shap": "XGBoost-SHAP", "xgb_pfi": "XGBoost置换", "tabm_pfi": "TabM置换",
    "deep_lasso": "Deep Lasso", "all_features": "全31项",
}
for model in ("xgb", "tabm"):
    display = "XGB" if model == "xgb" else "TabM"
    for mode, name in (("naive", "普通差"), ("midpoint", "中点差"),
                       ("orbit", "随机对称差"), ("orbit_student", "标准化对称差"),
                       ("sko_ridge", "SKO-Ridge"), ("sko_quad", "SKO-非线性均值"),
                       ("sko_scale", "SKO-尺度"), ("sko_quad_scale", "SKO-均值尺度")):
        LABELS[f"{model}_{mode}"] = f"{display}-{name}"
SCENES = {
    "global_null": "全无效", "linear_sparse": "线性稀疏", "nonlinear_strong": "强非线性",
    "nonlinear_weak": "弱非线性", "pure_interactions": "纯交互",
    "high_correlation": "高相关线性", "mixed_copula": "已知混合变换", "misspecified_t": "t生成错设",
}


def read(path):
    return json.loads(Path(path).read_text())


def fmt(v, digits=4):
    return "不定义" if pd.isna(v) else f"{float(v):.{digits}f}"


def table(headers, rows):
    return old.md_table(headers, rows)


def real_table(pred, stable, methods):
    rows = []
    for name in methods:
        r = pred.loc[pred.method.eq(name) & pred.rule.eq("top12")].set_index("evaluator")
        s = stable.loc[stable.method.eq(name) & stable.fraction.eq(.5)].iloc[0]
        rows.append([LABELS[name], fmt(r.loc["xgboost", "rmse_log"]), fmt(r.loc["tabm", "rmse_log"]),
                     fmt(s["mean"]), fmt(s.nogueira)])
    return table(["方法", "XGB-RMSE", "TabM-RMSE", "半样本J", "Nogueira"], rows)


def load_tables():
    assert read(ROOT / "results/final_audit.json")["passed"]
    assert read(MECH / "results/audit.json")["passed"]
    assert read(SCPL / "results/audit.json")["passed"]
    assert read(SCPL / "results/mvr_audit.json")["passed"]
    assert read(SUP / "results/audit.json")["passed"]
    m = pd.read_csv(MECH / "results/comparison_overview.csv")
    c = pd.read_csv(SCPL / "results/simulation_summary.csv")
    sp = pd.read_csv(SUP / "results/prediction_summary.csv")
    ss = pd.read_csv(SUP / "results/stability_summary.csv")
    si = pd.read_csv(SUP / "results/simulation_summary.csv")
    t = old.tables()
    t["DATA_TABLE"] = table(["阶段", "目标月份", "行数", "主要用途"], [
        ["训练", "2017-02至2017-12", 6567, "预处理与基础拟合"],
        ["调参", "2018-01至2018-02", 1831, "选参数/轮次后重拟合"],
        ["评分", "2018-03至2018-04", 1943, "协议A可早停；协议B固定模型评分"],
        ["测试", "2018-05至2018-07", 3413, "仅评价；已经历探索"]])
    t["METHOD_SCOPE"] = table(["家族", "代表", "本研究范围"], [
        ["统计/树基线", "Elastic Net、稳定性选择、影子树、SHAP", "公共样本与预算完整比较"],
        ["近期深度", "Deep Lasso、TabM、VTFS", "官方函数/骨干及已披露适配"],
        ["模型改动", "Jacobian、预算门、EntryPrune、序列成对损失", "十项配置完整机制实验"],
        ["对称评分", "XGB/TabM普通、中点、随机、标准化", "14种方法/消融完整实测"],
        ["条件重建", "Semi-KO Ridge及条件均值/尺度", "八项补充适配，作者数值核对"],
        ["运行后退出", "DeepDRK、TabPFN v2", "生成诊断失败/内存失败"],
        ["相邻方法", "DeepPINK、DeepPIG、GRIP2、HRT等", "文献比较；不计作完整实测基线"]])
    t["DESIGN_TABLE"] = table(["设计", "训练/评分", "实测规模", "主要边界"], [
        ["A 基础与机制", "排序期可用于早停/效用", "8基线+10改动；600份模拟", "固定配置适配，规范Top8/12复算"],
        ["B SCPL", "基础模型不见评分响应", "14评分；111真实；800模拟", "oracle与错设分开"],
        ["B 生成敏感性", "同B，仅换MVR", "111真实；30组预测", "看过参考结果后追加"],
        ["B 条件矩", "基础模型固定；条件模型使用评分响应", "8评分；111真实；800份复算", "6,400方法结果，非新数据"]])
    t["MECHANISM_TABLE"] = table(
        ["方法", "XGB误差", "TabM误差", "半样本J", "FDP@12", "Power@12"],
        [[r.label, fmt(r.rmse_xgb), fmt(r.rmse_tabm), fmt(r.jaccard_half),
          fmt(r.FDP12_mean_scenes), fmt(r.Power12_mean_scenes)] for _, r in m.iterrows()])
    trade = pd.read_csv(MECH / "results/tradeoff_evidence.csv")
    wanted = [("dl_jacobian", "deep_lasso"), ("tabm_budget_gate", "tabm"),
              ("ep_refresh", "ep_regression"), ("vtfs_cardinality", "vtfs"),
              ("vtfs_pairrank", "vtfs_cardinality")]
    labels = dict(zip(m.method, m.label))
    t["MECHANISM_PAIRED"] = table(["改动减父方法", "参考J差", "互补J差", "FDP差", "Power差"], [
        [labels[a] + "减" + labels[b], fmt(r.reference_J_diff), fmt(r.complementary_J_diff),
         fmt(r.mean_scene_FDP_diff), fmt(r.mean_scene_power_diff)]
        for a, b in wanted for _, r in trade.loc[trade.candidate.eq(a) & trade.comparator.eq(b)].iterrows()])
    p = read(SCPL / "protocol.json")
    t["SCPL_SCENES"] = table(["场景", "真信号", "相关系数", "信噪比", "生成条件"], [
        [SCENES[s["name"]], s["signals"], s["rho"], s["snr"] if s["signals"] else "不适用",
         "故意错设" if s["name"] == "misspecified_t" else "已知精确"]
        for s in p["simulation"]["scenarios"]])
    native_methods = ["linear_orbit_student", "xgb_midpoint", "tabm_midpoint", "tabm_orbit", "tabm_orbit_student"]
    t["SCPL_NATIVE"] = table(["场景", "方法", "平均FDP", "FDP-MCSE", "Power", "发现数"], [
        [SCENES[s["name"]], LABELS[name], fmt(r.FDP_mean), fmt(r.FDP_se), fmt(r.power_mean), fmt(r["size_mean"], 2)]
        for s in p["simulation"]["scenarios"] for name in native_methods
        for _, r in c.loc[c.scene.eq(s["name"]) & c.method.eq(name) & c.rule.eq("native")].iterrows()])
    cp = pd.read_csv(SCPL / "results/prediction_summary.csv")
    cs = pd.read_csv(SCPL / "results/stability_summary.csv")
    mp = pd.read_csv(SCPL / "results/mvr_prediction_summary.csv")
    ms = pd.read_csv(SCPL / "results/mvr_stability_summary.csv")
    rows = []
    for generator, pred, stable in (("等相关", cp, cs), ("MVR追加", mp, ms)):
        for name in ("xgb_shap", "deep_lasso", "tabm_midpoint", "tabm_orbit", "tabm_orbit_student"):
            r = pred.loc[pred.method.eq(name) & pred.rule.eq("top12")].set_index("evaluator")
            s = stable.loc[stable.method.eq(name) & stable.fraction.eq(.5)].iloc[0]
            rows.append([generator, LABELS[name], fmt(r.loc["xgboost","rmse_log"]),
                         fmt(r.loc["tabm","rmse_log"]), fmt(s["mean"]), fmt(s.nogueira)])
    t["SCPL_REAL"] = table(["生成条件", "方法", "XGB误差", "TabM误差", "J", "Nogueira"], rows)
    t["GENERATOR_TABLE"] = table(["条件", "25%交换", "50%交换", "100%交换"], [
        [label] + [fmt(v["seller_disjoint_auc"]) for v in read(SCPL / "results" / path)["tests"]]
        for label, path in (("等相关", "real_generator_diagnostic.json"), ("MVR追加", "mvr_generator_diagnostic.json"))])
    methods = read(SUP / "protocol.json")["methods"]
    t["SKO_REAL"] = real_table(sp, ss, ["xgb_shap", "tabm_pfi", "deep_lasso"] + methods)
    t["SKO_NATIVE"] = table(["场景", "方法", "平均FDP", "MCSE", "Power", "发现数"], [
        [SCENES[s["name"]], LABELS[name], fmt(r.FDP_mean), fmt(r.FDP_se), fmt(r.power_mean), fmt(r["size_mean"], 2)]
        for s in p["simulation"]["scenarios"] for name in methods
        for _, r in si.loc[si.scene.eq(s["name"]) & si.method.eq(name) & si.rule.eq("native")].iterrows()])
    rows = []
    for name in ["xgb_shap", "tabm_orbit", "tabm_orbit_student"] + methods:
        part = si.loc[si.rule.eq("top12") & si.method.eq(name) &
                      ~si.scene.isin(["global_null", "misspecified_t"])]
        rows.append([LABELS[name], fmt(part.FDP_mean.mean()), fmt(part.power_mean.mean()), fmt(part.rmse_mean.mean())])
    t["SKO_FIXED"] = table(["方法", "六场景FDP@12", "Power@12", "共同模拟RMSE"], rows)
    rows = []
    for source, name in ((c, "tabm_orbit_student"), (si, "tabm_sko_quad_scale")):
        for _, r in source.loc[source.method.eq(name) & source.rule.eq("native10")].iterrows():
            rows.append([LABELS[name], SCENES[r.scene], fmt(r.FDP_mean), fmt(r.power_mean), fmt(r["size_mean"], 2)])
    t["Q10_TABLE"] = table(["主候选", "场景", "平均FDP", "Power", "发现数"], rows)
    overview = pd.read_csv(SCPL / "results/comparison_overview.csv")
    t["SCPL_ALL"] = table(["方法", "XGB误差", "TabM误差", "J", "FDP@12", "Power@12"],
                         [[r.method, fmt(r.XGB_RMSE), fmt(r.TabM_RMSE), fmt(r.J), fmt(r.FDP12), fmt(r.Power12)]
                          for _, r in overview.iterrows()])
    primary = "tabm_sko_quad_scale"
    a = sp.loc[sp.method.eq(primary) & sp.evaluator.eq("xgboost") & sp.rule.eq("top12")].iloc[0]
    b = sp.loc[sp.method.eq("tabm_sko_ridge") & sp.evaluator.eq("xgboost") & sp.rule.eq("top12")].iloc[0]
    ast = ss.loc[ss.method.eq(primary) & ss.fraction.eq(.5)].iloc[0]
    strong = si.loc[si.method.eq(primary) & si.scene.eq("nonlinear_strong") & si.rule.eq("native")].iloc[0]
    baseline = si.loc[si.method.eq("tabm_sko_ridge") & si.scene.eq("nonlinear_strong") & si.rule.eq("native")].iloc[0]
    violated = si.loc[si.method.eq(primary) & si.rule.eq("native") & si.FDP_mean.gt(.2)]
    interpretation = ("；".join(f"{SCENES[r.scene]}为{fmt(r.FDP_mean)}" for _,r in violated.iterrows())
                      or "所列场景未出现高于.20的点估计，但有限模拟不证明总体控制")
    pair = pd.read_csv(SUP / "results/prediction_paired.csv")
    delta = pair.loc[pair.candidate.eq(primary) & pair.comparator.eq("xgb_shap") &
                     pair.evaluator.eq("xgboost") & pair.rule.eq("top12")].iloc[0]
    t["SKO_ABSTRACT"] = (
        f"条件矩补充中，预定TabM均值尺度版本的真实Top-12 RMSE为{fmt(a.rmse_log)}，"
        f"半样本Jaccard为{fmt(ast['mean'])}；强非线性原生Power为{fmt(strong.power_mean)}，"
        f"平均FDP为{fmt(strong.FDP_mean)}。这些结果仍需结合估计条件重建的误选风险解释，"
        "不将局部检出增加直接写为有效错误控制。")
    t["SKO_ENGLISH"] = (
        f"The prespecified conditional-mean-and-scale TabM adaptation achieves power {fmt(strong.power_mean)} "
        f"and mean FDP {fmt(strong.FDP_mean)} in the strongly nonlinear scenario. However, its real-data "
        f"Top-12 RMSE is {fmt(a.rmse_log)} with half-sample Jaccard {fmt(ast['mean'])}, "
        "which does not establish improvement over the strong tree baseline.")
    t["SKO_DISCUSSION"] = (
        f"预定主候选的XGB评价RMSE为{fmt(a.rmse_log)}，同TabM的Ridge机制适配为{fmt(b.rmse_log)}；"
        f"其半样本参考Jaccard为{fmt(ast['mean'])}。相对SHAP的RMSE差为{fmt(delta.difference)}，"
        f"卖家簇Bootstrap点态95%区间为[{fmt(delta.ci_low)},{fmt(delta.ci_high)}]。"
        "区间跨零时不能宣称已证实提升，也不能声称已证明等价。\n\n"
        f"强非线性下主候选Power为{fmt(strong.power_mean)}，Ridge适配为{fmt(baseline.power_mean)}；"
        f"主候选平均FDP为{fmt(strong.FDP_mean)}。平均FDP超过名义.20的场景检查：{interpretation}。"
        "检出增多须与误选同时解释。条件尺度建模不自动产生交换有效性，"
        "任何经验失控都必须保留，不因其真实预测较好而隐去。\n\n"
        "该实验补足了近期双侧条件重建的机制比较，但仅是固定参数、有限基函数的适配。"
        "当前没有把条件均值/尺度扩展称为已证明原创的新方法。")
    comparisons = pd.read_csv(SUP / "results/simulation_paired.csv")
    mechanism_rows = []
    for scene in ("nonlinear_strong", "nonlinear_weak", "pure_interactions", "mixed_copula", "high_correlation"):
        r = comparisons.loc[comparisons.scene.eq(scene) & comparisons.rule.eq("native") &
                            comparisons.metric.eq("power") & comparisons.candidate.eq(primary) &
                            comparisons.comparator.eq("tabm_sko_quad")].iloc[0]
        mechanism_rows.append([SCENES[scene], fmt(r["mean"]),
                               f"[{fmt(r.ci_low)},{fmt(r.ci_high)}]", f"{r.holm_p:.4g}"])
    t["SKO_DISCUSSION"] += (
        "\n\n**独立组件判断。** 强非线性中只用非线性均值的Power已为.856，联合尺度后为.866；"
        "纯交互中为.959与.966。大部分检出改进来自非线性条件均值，而不是尺度项。"
        "联合项减非线性均值的配对差如下，Holm校正限于同场景、同规则、同指标的比较族，"
        "不能解释为跨所有场景筛选后的确认结论。\n\nTable: 条件尺度相对非线性均值的检出增量\n\n" +
        table(["场景", "加入尺度的Power差", "95%区间", "Holm p"], mechanism_rows) +
        "\n\n联合TabM真实RMSE约1.8965，SHAP为1.8466，误差差区间完全为正；"
        "参考Jaccard约.4099，也低于SHAP的.7682。联合方法没有形成真实应用优势。"
        "此外，未标准化SCPL随机对称TabM在强非线性、弱非线性和纯交互中的Power仍更高，"
        "因此不能把相对Ridge适配的巨大提高写成超越所有非线性方法。")
    t["SKO_CONCLUSION"] = (
        f"双侧条件矩补充得到主候选真实RMSE {fmt(a.rmse_log)}、半样本J {fmt(ast['mean'])}，"
        f"强非线性Power {fmt(strong.power_mean)}与平均FDP {fmt(strong.FDP_mean)}。"
        "它说明估计条件矩的收益与误选代价需要共同检查；作者阈值函数的纠错仅保证实现符合公式，"
        "不能补救条件模型失配。")
    return t, sp, ss, si, m


def bibliography(source):
    registry = read(ROOT / "references/papers.json") + read(SUP / "references/additional_papers.json")
    lookup = {r["key"]: r for r in registry}
    keys = []
    for group in re.findall(r"\\cite\{([^}]+)\}", source):
        for key in group.split(","):
            assert key in lookup, key
            if key not in keys:
                keys.append(key)
    md, tex = [], [r"\begin{thebibliography}{99}"]
    for i, key in enumerate(keys, 1):
        r = lookup[key]
        kind = r.get("kind", "DB/OL" if key in ("olist2018", "funnel2018") else
                     "C" if key in ("deeplasso2023", "vtfs2024", "tabm2025", "deepdrk2024", "xgb2016", "shap2017") else "J")
        if key == "vtfs2024":
            kind = "J"
        text = f"{r['authors'].rstrip('.')}. {r['title']}[{kind}]. {r['venue']}, {r['year']}."
        url = "https://doi.org/" + r["doi"] if "doi" in r else r["url"]
        tex.append(r"\bibitem{" + key + "} " + old.escape(text) + r" \url{" + url + "}.")
        md.append(f"[{i}] {text} {url}")
    tex.append(r"\end{thebibliography}")
    return "\n\n".join(md), "\n".join(tex), {k: i for i,k in enumerate(keys)}


def pandoc(text, chapter=False):
    text = re.sub(r"^# 第\d+章\s+", "# ", text, flags=re.M)
    text = re.sub(r"^(#{2,4})\s+\d+(?:\.\d+)+\s+", r"\1 ", text, flags=re.M)
    cmd = ["pandoc", "-f", "markdown+tex_math_dollars-autolink_bare_uris", "-t", "latex", "--wrap=none", "--no-highlight"]
    if chapter:
        cmd += ["--top-level-division=chapter"]
    out = subprocess.run(cmd, input=text, text=True, capture_output=True, check=True).stdout
    out = re.sub(r"\\label\{[^{}]+\}", "", out)
    def resize(match):
        tab = match.group(0)
        pre, rest = tab.split(r"\toprule", 1)
        widths = list(re.finditer(r"\\real\{[\d.]+\}", pre))
        n = len(widths)
        if n:
            if n >= 5:
                weights = ([.25] + [.75 / (n - 1)] * (n - 1))
                if "场景" in rest[:1100] and "方法" in rest[:1100]:
                    weights = [.15, .28] + [.57 / (n - 2)] * (n - 2)
            elif n == 4:
                weights = [.21, .29, .24, .26]
            else:
                weights = [1/n] * n
            for item, weight in reversed(list(zip(widths, weights))):
                pre = pre[:item.start()] + "\\real{" + f"{weight:.6f}" + "}" + pre[item.end():]
        return "{\n\\footnotesize\n" + pre + r"\toprule" + rest + "\n}"
    return re.sub(r"\\begin\{longtable\}.*?\\end\{longtable\}", resize, out, flags=re.S)


COMMON = old.COMMON_TEX + r"""
\usepackage{xurl,needspace,fancyvrb}
\setmonofont{Menlo}
\setCJKmonofont{Songti SC}
\setlength{\tabcolsep}{4pt}
\AddToHook{cmd/section/before}{\Needspace{6\baselineskip}}
"""


def build_thesis(source, references, numbers):
    zhpart = source.split("# 摘要\n",1)[1].split("# Abstract",1)[0]
    zh, kwzh = zhpart.split("**关键词：**")
    enpart = source.split("# Abstract\n",1)[1].split("# 第1章",1)[0]
    en, kwen = enpart.split("**Key words:**")
    body = "# 第1章" + source.split("# 第1章",1)[1].split("# 参考文献",1)[0]
    appendix = "# 附录A" + source.split("# 附录A",1)[1]
    appendices = []
    for part in re.split(r"(?=^# 附录[A-Z])", appendix, flags=re.M):
        if part.strip():
            heading, content = part.split("\n",1)
            appendices.append(r"\chapterx{" + heading[2:] + "}\n" + pandoc(content))
    tex = r"\documentclass{sufethesismas}" + COMMON + rf"""
\begin{{document}}
\classification{{}}\confidential{{}}\UDC{{}}\serialnumber{{}}
\title{{面向电商经营指标筛选的\\非线性方法与交换对称损失研究}}
\englishtitle{{{ENGLISH}}}
\author{{吴优}}\advisor{{张吕欧}}\major{{应用统计硕士专业学位论文}}
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
{references}
{chr(10).join(appendices)}
\end{{document}}
"""
    (ROOT / "latex工程/论文/论文.tex").write_text(tex)
    readable = re.sub(r"\\cite\{([^}]+)\}", lambda m: "[" + ",".join(str(numbers[k]+1) for k in m[1].split(",")) + "]", source)
    (ROOT / "论文_v7.md").write_text("# " + TITLE + "\n\n" + readable)


def article(text, title, refs=""):
    return r"""\documentclass[UTF8,11pt,a4paper]{ctexart}
\usepackage[left=2cm,right=2cm,top=2cm,bottom=2cm]{geometry}
\usepackage{amsmath,amssymb,graphicx,hyperref,setspace,tabularx,enumitem}
\setCJKmainfont{Songti SC}\setmainfont{Times New Roman}
\setstretch{1.18}\hypersetup{hidelinks}
""" + COMMON + "\n\\begin{document}\n" + (
        r"\begin{center}\large\bfseries " + title + r"\end{center}") + pandoc(text) + refs + "\n\\end{document}\n"


def build_report(t, refs):
    text = rf"""# 选题背景与研究问题

研究从31项电商经营指标中筛选少量具有下一月签收确认金额预测信息的指标，同时评价名单稳定性与错误发现风险。预测筛选不同于当期金额拆分与因果归因。问题重点是近期非线性方法是否带来可检验增量，以及损失比较的对称性和生成有效性如何影响结论。

# 文献依据

Deep Lasso的输入梯度正则、TabM参数高效集成、VTFS序列生成和EntryPrune动态输入层提供不同机制\cite{{deeplasso2023,tabm2025,vtfs2024,entryprune2025}}。DeepPINK、DeepPIG和GRIP2说明成对深度选择已有先例\cite{{deeppink2018,deeppig2024,grip2026}}。CPI与Semi-knockoffs分别研究条件损失和双侧条件重建\cite{{cpi2021,semiko2026}}。本研究不把既有机制组合直接称为首创。

# 数据与公平原则

Olist十张表按自然层级聚合，并按月末已发生事件截断，形成13,754行、2,723名卖家、31项指标。目标为下一自然月签收确认商品金额log1p，不含运费。静态快照和同卖家跨期依赖仍有局限。

{t['DATA_TABLE']}

所有方法读取同一卖家抽样名单：30组互补50%、20个70%、20个80%，另10组固定数据随机性和完整参考，共111例。主预算12项、敏感性8项；共同XGBoost和TabM评价，同集合使用相同列顺序和预测。真实测试期已经探索查看，不是新外部确认。

# 拟研究的方法与性质

固定预测器SCPL构造不依赖真假命名的min/max背景，逐变量比较真假值的平方损失差。交换任意真假列对后，背景不变，被交换统计量反号，其余不变；这使其在独立评分与精确生成器条件下可调用已有Knockoff+理论，不是新FDR定理。

四种背景/评分消融接入相同XGBoost和TabM。进一步研究Semi-knockoffs的条件均值与条件尺度扩展，拆分Ridge、非线性均值、条件尺度和联合版本。估计条件矩不能自动获得oracle保证。

# 已完成结果

基础与机制实验包含八基线和十项改动；SCPL有14种评分、800份八场景模拟，MVR敏感性另有111个真实案例；条件矩补充计算八种评分、同800份模拟和32组真实预测。

{t['SCPL_REAL']}

SCPL标准化TabM在强非线性Power为.794、纯交互为.934，匹配线性对照为.087和.003。但未标准化更强，高相关线性是反例，真实稳定性也较弱。

{t['SKO_REAL']}

{t['SKO_DISCUSSION']}

# 研究计划与限制

当前已经完成数据、机制、性质与配对实验，但不将负结果写成成功创新。下一步优先补足未使用真实数据的冻结验证、近期非线性受控筛选基线，以及条件矩/生成误差对符号与FDR的影响。完成这些证据后，再判断统计贡献与适用范围。学校开题意见和签字由相关人员填写，不由研究程序预填。
"""
    cover = (ROOT / "report_cover.tex.in").read_text().replace("@@TITLE@@", TITLE)
    cover = cover.replace("论文题目：面向电商GMV监控的", "论文题目：面向电商经营指标筛选的")
    cover = cover.replace("深度特征选择与稳定性评价研究", "非线性方法与交换对称损失研究")
    tex = article(text, "", refs).replace("\\begin{document}\n", "\\begin{document}\n" + cover)
    (ROOT / "latex工程/开题报告/开题报告.tex").write_text(tex)
    (ROOT / "开题报告_v7.md").write_text("# " + TITLE + "\n\n" + text)


def short_documents(source, t, refs):
    condensed = "# 研究摘要\n\n" + source.split("# 摘要\n")[1].split("**关键词")[0]
    for chapter in (3,4,5,6,8):
        key = f"# 第{chapter}章"
        content = source.split(key,1)[1].split("# 第",1)[0].split("# 参考文献",1)[0]
        condensed += "\n\n# " + content.strip()
    brief = f"""# 问题、数据与方法

从31项月末经营指标筛选下一月签收确认商品金额的预测信息，不作因果归因。Olist面板13,754行、2,723名卖家，train/tune/rank/test为6567/1831/1943/3413行。公开快照可见性与面板依赖仍有限制。

研究包括基础八方法、十项模型机制改动、交换对称成对损失SCPL和Semi-knockoffs条件矩扩展。所有样本稳定性使用相同卖家清单，50%区间按30组互补样本计算。相同K、相同评价器和规范列顺序避免伪收益。

SCPL使用真假数值对的min/max构造不区分身份的背景。交换列对后自身损失差反号，其他统计量不变。该代数性质不证明生成器有效，真实名单不能直接称FDR受控。

\\newpage

# 真实结果与预测稳定性

{t['SCPL_REAL']}

基础规范Top-12下SHAP的XGB-RMSE为1.8466；Deep Lasso为1.8450。Jacobian的参考Jaccard为.8618，但收益未同时覆盖互补稳定性和模拟发现质量。EntryPrune刷新为负结果，VTFS主要获益来自固定K修正。

\\newpage

# 模拟与新补充

800份模拟包含全无效、线性、非线性、纯交互、高相关、混合变换与故意错设。SCPL主候选强非线性Power为.794，纯交互.934；匹配线性对照.087和.003。标准化往往弱于未标准化，不应写为普遍成功创新。

{t['SKO_REAL']}

{t['SKO_ABSTRACT']}

\\newpage

# 结论和限制

{t['SKO_CONCLUSION']}

FDP是发现中错误比例，Power是真信号检出率。全无效Power不定义，Top-K与原生规则不混比。真实测试已被探索查看，MVR是事后敏感性，不构成外部确认。中点TabM相对SHAP的RMSE差区间跨零，不能称已证实预测提升。

当前形成了有明确性质和完整机制实验的研究原型，但没有证明原创优先权或真实整体优势。需要近期非线性受控基线与第二真实数据验证。完整公式、配对区间、参考文献和审计见论文与复现工程。
"""
    for name, text, bib in (("精简版",condensed,refs), ("极限精简版",brief,"")):
        (ROOT / f"论文_v7_{name}.md").write_text("# " + TITLE + "\n\n" + text)
        (ROOT / f"论文_v7_{name}.tex").write_text(article(text,TITLE,bib))


def slides(t, sp, ss, si, mechanisms):
    frames, notes = [], []
    def items(lines):
        return "\\begin{itemize}\n" + "\n".join("\\item " + old.escape(s) for s in lines) + "\n\\end{itemize}"
    def frame(title, body, note):
        frames.append("\\begin{frame}{" + title + "}\n" + body + "\n\\end{frame}")
        notes.append(f"## {len(frames)+1}. {title}\n\n{note}")
    def tab(headers, rows):
        return r"\footnotesize\centering " + old.small_tabular(headers,rows)
    frame("研究问题：有限容量的经营指标", items([
        "31项指标中选择8或12项，保留下一月目标的预测信息。",
        "同时评价预测、样本稳定性和模拟发现质量。",
        "不把预测筛选说成当期GMV分解或因果归因。"]),
        "研究对象是经营指标的预测性筛选，而不是地区或品类的金额分解。有限看板容量要求压缩，同时要检验名单是否可靠。")
    frame("文献与方法缺口", items([
        "Deep Lasso 2023、VTFS 2024、TabM 2025提供不同非线性机制。",
        "CPI和Semi-knockoffs 2026研究条件损失及双侧重建。",
        "DeepPINK、DeepPIG、GRIP2已有成对深度研究。",
        "研究具体评分性质与组件增量，不把模型改名当创新。"]),
        "新模型的发表时间不是优势证据。本文把表达能力、评分对称性和生成有效性拆开，寻找能够独立验证的改动。")
    frame("数据与时间可见性", items([
        "Olist十张表，13,754个卖家月，2,723名卖家，31项指标。",
        "目标：下一自然月签收确认商品金额log1p，不含运费。",
        "物流、评价和营销按月末已发生事件截断。",
        "静态元数据缺少修订历史，仍有可见性假设。"]),
        "同一个GMV名称可以对应购买月或签收月。本研究按下一月签收确认定义响应，特征使用本月末已经发生的信息。")
    frame("训练、评分与测试分工", tab(["阶段","行数","用途"],[
        ["train",6567,"基础训练"],["tune",1831,"选参/轮次"],["rank",1943,"名单/条件重建"],
        ["test",3413,"仅评价"]]) + items([
            "固定预测器实验不使用rank响应训练或早停基础模型。",
            "条件重建按算法定义使用rank响应，单独披露。",
            "测试期已探索查看，不是新外部确认样本。"]),
        "两类协议分开报告：基础方法可能用排序期早停，损失评分则先冻结预测器。条件重建使用评分标签属于算法定义，不能隐藏。")
    frame("公共样本与评价规则", items([
        "30组互补50%卖家样本，共60份；另20份70%、20份80%。",
        "10组固定数据随机种子试验与样本稳定性分开。",
        "共同XGBoost与TabM评价，规范列顺序，同集合共享预测。",
        "原生q阈值与Top-K预算分别评价，空集不丢弃。"]),
        "所有方法面对相同卖家扰动。两个评价器用于区分名单的信息和模型能力，规范列顺序则避免同名单出现伪预测差。")
    rows = []
    for name in ("xgboost_shap","deep_lasso","tabm","dl_jacobian","tabm_budget_gate","vtfs_pairrank","ep_refresh"):
        r=mechanisms.loc[mechanisms.method.eq(name)].iloc[0]
        rows.append([r.label,fmt(r.rmse_xgb),fmt(r.jaccard_half),fmt(r.Power12_mean_scenes)])
    frame("模型内部改动：局部收益与失败", tab(["方法","XGB误差","J","模拟Power@12"],rows),
        "Jacobian有局部稳定性价值，但没有在所有证据上胜过父方法。预算门也有代价。EntryPrune刷新显著变差，不能只展示成功的一面。")
    frame("SCPL：背景不能偏袒真假身份", items([
        "固定训练好的预测器，评分集生成真假变量对。",
        "逐对取min/max，四组掩码及补掩码构成八背景。",
        "比较某列真假值时，其他列保持同一对称背景。",
        "比较平方损失差，再按卖家汇总。"]),
        "普通替换只比较自己，其他真实列却会影响其他变量的评分。对称背景让比较环境不随真假命名改变，这是本轮统计量的具体设计点。")
    frame("可证明的性质，不可越过的前提", items([
        "交换某列对：自身统计量反号，其他统计量不变。",
        "min/max背景逐路径不变，损失输入互换即可证明。",
        "精确生成器下，随机背景边际与X同分布。",
        "不证明预测更准、生成器正确或真实FDR已控。"]),
        "证明解决的是完整符号翻转这一项要求。还需要精确生成和适当样本条件才能调用已有Knockoff结论，这不是新提出FDR定理。")
    frame("机制消融与主候选", items([
        "同一预测器：普通、中点、随机对称、标准化四种评分。",
        "分别接入XGBoost和TabM，保留匹配线性对照。",
        "预定主候选是标准化TabM，不能事后换成最佳消融。",
        "八背景不等于八组独立Knockoff。"]),
        "拆分消融是判断收益来自哪一步的关键。主候选即使失败也保留，不能看完数据再把更好的简单版本写成原先目标。")
    frame("模拟设计：800份而非挑选案例", items([
        "八场景各100次；n=2000，p=31，训练/评分/测试1200/400/400。",
        "全无效、线性、强弱非线性、纯交互、高相关、混合变换、t错设。",
        "七场景精确生成；t场景故意错设。",
        "全无效Power不定义，FDP和Power共同解释。"]),
        "场景覆盖有利和不利机制，特别保留全无效与错设。100次只是带Monte Carlo误差的估计，不是总体FDR的证明。")
    rows=[]
    for scene in ("linear_sparse","nonlinear_strong","nonlinear_weak","pure_interactions","high_correlation"):
        a=si.loc[si.scene.eq(scene)&si.method.eq("linear_orbit_student")&si.rule.eq("native")].iloc[0]
        b=si.loc[si.scene.eq(scene)&si.method.eq("tabm_orbit_student")&si.rule.eq("native")].iloc[0]
        rows.append([SCENES[scene],fmt(a.power_mean,3),fmt(b.power_mean,3),fmt(b.FDP_mean,3)])
    frame("非线性检出增加，但存在反例",tab(["场景","线性Power","SCPL Power","SCPL FDP"],rows),
        "强非线性与纯交互中检出明显增加，说明非线性预测器有用。但高相关线性里更差，标准化也常不如未标准化，不能称普遍提升。")
    frame("真实数据：生成器影响很大",tab(["方法/生成条件","XGB误差","J"],[
        ["SHAP", "1.8466",".7682"],["SCPL主候选/等相关","2.0964",".3249"],
        ["SCPL主候选/MVR追加","1.8493",".4696"],["TabM中点/MVR追加","1.8431",".4988"]]),
        "等相关生成器导致很小的真假距离，真实评分较弱。换MVR改善预测，但稳定性仍不足。中点小幅数值优势的区间跨零，不能确认。")
    frame("真实FDR为什么不能保证",items([
        "估计Copula的交换分类AUC明显偏离.5。",
        "同卖家跨期相关，不满足简单独立评分理论。",
        "MVR在看到参考结果后追加，必须披露后验性质。",
        "名义发现数增加不等于错误率控制更好。"]),
        "生成器诊断只提供工程警告，没有授予FDR证书。两种生成器都存在失配，所以原生发现只能称名义阈值发现。")
    frame("近期原文：Semi-knockoffs",items([
        "一侧条件于其他特征，另一侧再条件于响应。",
        "分别重建特征并打乱残差，比较固定预测器损失。",
        "oracle条件矩已知时与估计条件矩的保证不同。",
        "作者代码无解阈值问题已纠正并保留反例。"]),
        "这项2026年研究提供了模型无关的双侧重建思路。原文对估计条件矩仍有额外条件和猜想，不能只读标题就说任意估计器有效。")
    frame("条件均值与尺度扩展",items([
        "Ridge均值、非线性均值、条件尺度、联合四机制。",
        "针对仅靠线性均值可能漏掉平方与交互信息的问题。",
        "同一XGBoost/TabM、同一置换、同一评分样本。",
        "8项评分复算800份已有模拟，不冒称新800份数据。"]),
        "条件尺度是探索变量信息是否通过波动而非均值体现。扩展本身属于已有条件抽样思想附近，不预先声称原创或FDR改善。")
    methods=["xgb_shap","tabm_sko_ridge","tabm_sko_quad","tabm_sko_scale","tabm_sko_quad_scale"]
    rows=[]
    for name in methods:
        r=sp.loc[sp.method.eq(name)&sp.rule.eq("top12")&sp.evaluator.eq("xgboost")].iloc[0]
        s=ss.loc[ss.method.eq(name)&ss.fraction.eq(.5)].iloc[0]
        rows.append([LABELS[name],fmt(r.rmse_log),fmt(s["mean"])])
    frame("条件矩扩展：真实结果",tab(["方法","XGB误差","J"],rows),
        "这张表来自本次补充计算。是否值得采用要同时看误差和稳定性，不用测试点估计挑一个冠军。完整配对区间见论文。")
    rows=[]
    for scene in ("global_null","nonlinear_strong","pure_interactions","high_correlation","misspecified_t"):
        r=si.loc[si.scene.eq(scene)&si.method.eq("tabm_sko_quad_scale")&si.rule.eq("native")].iloc[0]
        rows.append([SCENES[scene],fmt(r.FDP_mean),fmt(r.power_mean)])
    frame("条件矩扩展：误选与检出",tab(["场景","平均FDP","Power"],rows),
        "检出率上升不能单独算成功，要一起看FDP。估计条件模型引入的不对称风险不会因为采用了名义Knockoff阈值而消失。")
    frame("结论与下一步",items([
        "已形成明确算法问题、性质、拆分消融和完整实测。",
        "没有证据支持复杂方法全面领先或真实FDR无条件受控。",
        "补近期非线性受控基线及未使用真实数据的冻结验证。",
        "所有尝试、失败和仅参考方法独立归档。"]),
        "本研究的价值是能够明确解释哪些机制起作用、付出什么代价。当前仍是需要外部验证的方法研究，不能承诺已经满足全部创新要求。")
    frame("主要参考文献",items([
        "Candes等，Model-X，JRSS B 2018；Ren与Barber，JRSS B 2024。",
        "Watson与Wright，CPI，Machine Learning 2021。",
        "Cherepanova等，Deep Lasso，NeurIPS D&B 2023。",
        "Gorishniy等，TabM，ICLR 2025；Ying等，VTFS，TKDD 2024。",
        "Reyero-Lobo等，Semi-knockoffs，ICML 2026。",
        "Zou与Tian，GRIP2，2026预印本；完整来源见论文。"]),
        "参考文献注明正式发表与预印本状态。没有完成共同协议实测的论文只用于方法关系讨论，不作为已打败的对照。")
    tex = rf"""\documentclass[aspectratio=169]{{beamer}}
\usepackage{{ctex,hyperref,amsmath,booktabs,array,graphicx}}
\usepackage{{SUFE}}
\setbeamertemplate{{navigation symbols}}{{}}\setbeamertemplate{{headline}}{{}}
\setbeamersize{{text margin left=7mm,text margin right=7mm}}
\setbeamerfont{{frametitle}}{{size=\large}}
\setbeamerfont{{frame number}}{{parent=author in head/foot}}
\setbeamertemplate{{footline}}{{%
\leavevmode\hbox{{\begin{{beamercolorbox}}[wd=\paperwidth,ht=3ex,dp=1.2ex,leftskip=3mm,rightskip=3mm]{{author in head/foot}}
\usebeamerfont{{author in head/foot}}吴优\quad 上海财经大学
\hfill 非线性指标筛选与交换对称损失\hfill\insertframenumber/\inserttotalframenumber
\end{{beamercolorbox}}}}}}
\title[非线性指标筛选与交换对称损失]{{{TITLE}}}
\subtitle{{机制、实测结果与推断边界}}
\author{{吴优\quad 2025213385}}
\institute[上海财经大学]{{统计与数据科学学院\\指导教师：张吕欧}}
\date{{2026年9月}}
\begin{{document}}
\begin{{frame}}\titlepage\centering\includegraphics[width=.16\linewidth]{{pic/sufe.png}}\end{{frame}}
{chr(10).join(frames)}
\end{{document}}
"""
    (ROOT / "latex工程/开题汇报/开题汇报.tex").write_text(tex)
    (ROOT / "开题汇报_v7_演讲稿.md").write_text("# 开题汇报演讲稿\n\n按约10分钟准备，结果表以说明主要差异为主。\n\n" + "\n\n".join(notes))
    return len(frames)+1


def figures(sp, ss, si, mechanisms):
    plt.rcParams.update({"font.sans-serif": ["Arial Unicode MS", "PingFang SC", "DejaVu Sans"],
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 10, "savefig.dpi": 180})
    names = ["xgb_shap", "tabm_pfi", "tabm_sko_ridge", "tabm_sko_quad", "tabm_sko_scale",
             "tabm_sko_quad_scale", "xgb_sko_quad_scale"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for i, name in enumerate(names):
        p = sp.loc[sp.method.eq(name) & sp.rule.eq("top12") & sp.evaluator.eq("xgboost")].iloc[0]
        s = ss.loc[ss.method.eq(name) & ss.fraction.eq(.5)].iloc[0]
        axes[0].scatter(p.rmse_log, i, color="#237a78", s=40)
        axes[1].errorbar(s["mean"], i, xerr=[[s["mean"] - s.ci_low], [s.ci_high - s["mean"]]],
                         color="#b95353", fmt="o", capsize=3)
    for ax in axes:
        ax.set_yticks(range(len(names)), [LABELS[n] for n in names], fontsize=8)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.2)
    axes[0].set_xlabel("Top-12 / XGBoost评价RMSE")
    axes[1].set_xlabel("50%样本Jaccard / 30组区间")
    fig.tight_layout()
    fig.savefig(ROOT / "figures/10_conditional_moments_real.png", bbox_inches="tight")
    plt.close(fig)
    names = ["linear_orbit_student", "tabm_orbit", "tabm_orbit_student", "tabm_sko_ridge", "tabm_sko_quad_scale"]
    scenes = list(SCENES)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for name in names:
        part = si.loc[si.method.eq(name) & si.rule.eq("native")].set_index("scene").reindex(scenes)
        axes[0].plot(range(8), part.FDP_mean, marker="o", ms=4, label=LABELS[name])
        axes[1].plot(range(8), part.power_mean, marker="o", ms=4, label=LABELS[name])
    for ax in axes:
        ax.set_xticks(range(8), [SCENES[s] for s in scenes], rotation=30, ha="right", fontsize=8)
        ax.set_ylim(-.03, 1.03)
        ax.grid(alpha=.2)
    axes[0].axhline(.2,color="black",ls="--",lw=1)
    axes[0].set_ylabel("平均FDP / q=.20")
    axes[1].set_ylabel("Power（全无效不定义）")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / "figures/11_loss_simulation_boundaries.png", bbox_inches="tight")
    plt.close(fig)


def main():
    t, sp, ss, si, mechanisms = load_tables()
    source=(ROOT / "thesis_revised.md.in").read_text()
    md, tex, numbers=bibliography(source)
    t["REFERENCES"]=md
    for k,v in t.items():
        source=source.replace("@@"+k+"@@",v)
    assert "@@" not in source
    figures(sp,ss,si,mechanisms)
    old.copy_assets()
    build_thesis(source,tex,numbers)
    build_report(t,tex)
    short_documents(source,t,tex)
    count=slides(t,sp,ss,si,mechanisms)
    save_json(ROOT / "results/document_generation.json",{
        "title":TITLE,"reference_count":len(numbers),"slides":count,
        "source_template":"thesis_revised.md.in","source_hash":digest_file(ROOT / "thesis_revised.md.in"),
        "tables_from":"separate audited experiment designs; canonical Top8/12 predictions",
        "unresolved_placeholders":0,"supplement_audit":str(SUP / "results/audit.json")})
    print("Generated V7 thesis, report, slides and two condensed versions",count,"slides",len(numbers),"references")


if __name__=="__main__":
    main()
