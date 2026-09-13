"""Create a traceable catalogue, not a pooled ranking across incompatible studies."""

from pathlib import Path
import csv
import hashlib
import json
import os
import shutil

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[2]
ARCHIVE = PROJECT / "所有尝试与参考方法"
V7 = ROOT.parent

STUDIES = [
    ("基础与早期分支", "论文/v1", "回溯购买月口径；历史确认性措辞不再作为有效性证明"),
    ("深度生成与配对网络", "论文/v2", "旧口径；生成诊断失败，不能套用新面板数值"),
    ("参数K与SHAP截断", "论文/v3", "Top-K展示不继承原生FDR"),
    ("分层分组与GRIP2式", "论文/v4", "式实现与原论文复现有别；旧口径与阈值敏感性"),
    ("周度分支", "论文/v5", "回溯购买周口径，不是在线可见特征"),
    ("基准与配对稳定性", "论文/v6", "0.969是固定数据随机稳定性，不能与卖家抽样混比"),
    ("八方法基础实验", "论文/v7", "888个真实方法案例、4800个模拟方法案例；主预测采用后续规范列顺序复算"),
    ("梯度一致性与变量审计", "论文/v7/innovation_pilot", "gamma=0被选中；一致性惩罚未获支持；分类与去冗余不是模型创新"),
    ("四家族模型机制实验", "论文/v7/model_improvements_20260912", "十项改动；6000个新模拟方法案例；74组规范预测"),
    ("交换对称损失SCPL", "论文/方法研究_20260913", "14种评分消融、800份模拟；两套生成器真实实验分开，MVR为事后追加"),
    ("Semi-KO条件矩补充", "论文/v7/supplement_20260913", "8种评分；复用800份模拟重算，不是800份新数据；纠正作者阈值辅助函数"),
]


def link(path):
    return Path(path).resolve().as_uri()


def add(rows, name, family, status, source, evidence, conclusion):
    rows.append(dict(method=name, family=family, status=status, source=source,
                     evidence=evidence, conclusion=conclusion))


def main():
    ARCHIVE.mkdir(exist_ok=True)
    entries, rows = [], []
    for name, relative, caveat in STUDIES:
        source = PROJECT / relative
        assert source.exists()
        shortcut = ARCHIVE / "完整工程入口" / name
        shortcut.parent.mkdir(exist_ok=True)
        if not shortcut.exists():
            shortcut.symlink_to(os.path.relpath(source, shortcut.parent), target_is_directory=True)
        entries.append(f"- [{name}]({link(source)})：{caveat}。")
    for name, relative, _ in STUDIES:
        source = PROJECT / relative
        snapshot = ARCHIVE / "报告与结果快照" / name
        snapshot.mkdir(parents=True, exist_ok=True)
        for file in source.iterdir():
            if file.is_file() and file.suffix in (".md", ".pdf", ".tex"):
                shutil.copy2(file, snapshot / file.name)
        result = source / "results"
        if result.exists():
            for file in result.iterdir():
                if file.is_file() and file.suffix in (".csv", ".json") and file.stat().st_size < 2_000_000:
                    (snapshot / "results").mkdir(exist_ok=True)
                    shutil.copy2(file, snapshot / "results" / file.name)

    base = [
        ("Copula-MVR/e-value/e-BH", "Model-X 2018; MVR 2022; Derandomised knockoffs 2024", "原生错误率规则有假设；真实生成失配，不授予FDR证书"),
        ("Elastic Net", "Zou & Hastie 2005", "强线性基线；非零集合不等于FDR发现"),
        ("稳定性选择经验适配", "Meinshausen & Buhlmann 2010; Shah & Samworth 2013", "内部Lasso频率阈值；不冒用原文PFER保证"),
        ("Extra Trees影子变量树", "Geurts et al. 2006; Boruta 2010", "完整实测但不是完整Boruta；高稳定性不等于最佳预测"),
        ("XGBoost-SHAP", "XGBoost 2016; SHAP 2017; TreeSHAP 2020", "强非线性排序基线，不是FDR检验"),
        ("Deep Lasso", "NeurIPS Datasets and Benchmarks 2023", "官方正则函数；网络与任务适配明确"),
        ("TabM置换", "TabM ICLR 2025; PFI", "官方网络加置换评分，不把附加评分说成TabM原生"),
        ("VTFS定额适配", "ACM TKDD 2024, 10.1145/3687485", "官方编码器解码器；随机语料和固定K适配，参考名单来自回退"),
    ]
    for name, source, conclusion in base:
        add(rows, name, "基础选择器", "完整实测", source, "论文/v7", conclusion)
    for name, conclusion, evidence in [
        ("Ridge/Lasso预测器", "早期预测基线，非全部统一选维实验", "论文/v1/code/run_analysis.py"),
        ("Extra Trees预测器", "早期及周度预测基线", "论文/v1/code/run_analysis.py"),
        ("卖家固定效应分析", "关联分析而非因果识别，不计作选维模型", "论文/v1/code/run_analysis.py"),
        ("深度Knockoff生成器", "KS改善但联合交换性失败", "论文/v2"),
        ("配对竞争MLP", "反对称检查通过但原生空集", "论文/v2"),
        ("残差表格MLP", "旧面板预测不及XGBoost", "论文/v2"),
        ("确认集合内SHAP及K截断", "预算展示，不继承原生FDR", "论文/v3"),
        ("卖家均值/卖家内离差分层Knockoff", "均值层与组内层不同；不能混作新增面板定理", "论文/v4"),
        ("业务Group Knockoff", "组假设不等于变量假设；放宽q不算算法收益", "论文/v4"),
        ("DDLK式最坏交换生成器", "仅式实现；深度核和分类诊断失败", "论文/v4"),
        ("深度核MMD与经验边际校准", "诊断/校准组件，不是成功选择器", "论文/v4"),
        ("GRIP2式二维正则轨迹", "早期已有尝试；不是官方完整复现，q=.2无发现", "论文/v4"),
        ("周度Copula-MVR/分层/分组/GRIP2式", "周度旧口径；不能与月末可见面板数字混排", "论文/v5"),
        ("高斯/Copula与MVR/SDP/等相关消融", "生成与变换组件，不是多个原创模型", "论文/v6"),
        ("Lasso/XGBoost竞争与重复次数消融", "不同竞争器、重复预算和阈值结果分开", "论文/v6"),
        ("同月互补卖家梯度一致性Deep Lasso", "开发规则选择gamma=0，新增惩罚失败", "论文/v7/innovation_pilot"),
        ("变量删除/替代/共识审计", "变量质量分析，不是已获支持的模型创新", "论文/v7/innovation_pilot"),
        ("概念分组Top-8", "曾使用全时期相关性；只作事后探索，不是独立验证", "论文/v7/innovation_pilot"),
    ]:
        add(rows, name, "早期与局部探索", "局部实测/旧口径", "对应工程方法说明", evidence, conclusion)
    mods = [
        ("Deep Lasso-Jacobian", "稳定性局部改善；相对父方法的互补稳定性与发现质量未共同改善"),
        ("Jacobian训练+原排名", "完整Top-12与父方法相同，预测不能另算增益"),
        ("原Deep Lasso+Jacobian排名", "完整Top-12相同；局部发现改进不等于预测改进"),
        ("TabM预算门", "参考Jaccard增加，互补J与模拟发现质量未改善"),
        ("EntryPrune回归适配", "不是原论文所有分类实验复现；真实效果弱于强基线"),
        ("EntryPrune刷新", "明显负结果"),
        ("EntryPrune环境波动惩罚", "未恢复强基线水平，不把均值减标准差称置信界"),
        ("VTFS固定K", "预算匹配修正，模拟改善但真实预测没有改善"),
        ("VTFS固定K+成对效用损失", "独立增量小，不能把预算修正收益归给损失"),
        ("等查询预算随机子集搜索", "检验深度解码是否真正优于搜索预算本身"),
    ]
    for name, conclusion in mods:
        add(rows, name, "模型机制消融", "完整实测", "父方法原文与本地机制协议", "论文/v7/model_improvements_20260912", conclusion)
    for model in ("XGBoost", "TabM"):
        for mode in ("普通替换损失", "中点对称损失", "随机对称损失", "标准化随机对称损失"):
            add(rows, f"{model}-{mode}", "SCPL评分", "完整实测", "CPI; Model-X; 本地SCPL定义",
                "论文/方法研究_20260913",
                "普通替换不满足完整符号翻转；其余有代数性质。非线性模拟收益不代表真实FDR或优先权")
    for name in ("评分集成对Lasso", "匹配训练样本线性对称评分", "XGBoost置换"):
        add(rows, name, "SCPL对照", "完整实测", "标准方法/匹配信息对照", "论文/方法研究_20260913", "用于区分预测器、训练信息和评分机制")
    complete = (ROOT / "results/audit.json").exists()
    for model in ("XGBoost", "TabM"):
        for mode in ("Ridge均值", "非线性均值", "条件尺度", "非线性均值+条件尺度"):
            add(rows, f"{model}-Semi-KO-{mode}", "条件矩扩展", "完整实测" if complete else "正在计算",
                "ICML 2026 Semi-knockoffs及条件抽样扩展", "论文/v7/supplement_20260913",
                "纠正无解阈值返回值；估计条件矩不自动享有有限样本保证；不声称原创优先权")
    for name, source, evidence, conclusion in [
        ("DeepDRK", "NeurIPS 2024", "论文/v7/vendor/DeepDRK", "三种子真实训练，生成诊断失败；未完成全套模拟稳定性"),
        ("TabPFN v2", "Nature 2025", "论文/v7/logs", "MPS内存失败；不是统计效果差，也不填零"),
    ]:
        add(rows, name, "退出候选", "训练/运行后退出", source, evidence, conclusion)
    references = [
        ("MAFS attention-like过滤法", "10.1016/j.knosys.2024.112250", "仅文献核验，不是已训练的多头深度模型"),
        ("DeepPINK", "https://arxiv.org/abs/1809.01185", "参考成对深度选择；未计作本轮完整基线"),
        ("DeepPIG", "10.1038/s41598-024-66061-6", "成对层加随机门已有先例；未完整实测"),
        ("Stabl", "10.1038/s41587-023-02033-x", "参考可靠性/稀疏性评价；未完整实测"),
        ("ChronoEpilogi", "NeurIPS 2024；见局部探索文献说明", "多个等效特征集合；未完整实测"),
        ("ConsistentFeature", "https://arxiv.org/abs/2412.01476v2", "跨子集一致性已有研究；未完整实测"),
        ("Stochastic Gates", "https://proceedings.mlr.press/v119/yamada20a.html", "输入门并非本项目发明；未单独完整复现"),
        ("Structured Nonlinear Variable Selection", "https://auai.org/uai2018/proceedings/papers/17.pdf", "偏导数正则已有文献；未单独完整复现"),
        ("LassoFlexNet", "https://icml.cc/virtual/2026/poster/62560", "下载仓库只标code coming soon，没有实测结果"),
        ("Adversarial/SAM LassoNet", "本地innovation_benchmark/vendor/AdversarialLassoNet", "下载了代码；无完整结果，不能冒认论文复现"),
        ("Temporal Modulation", "本地innovation_benchmark/vendor/TemporalModulation", "下载代码但未有完整公平结果"),
        ("非参数顺序/并行残差Knockoff", "https://arxiv.org/abs/2407.06892", "阅读生成诊断与近似条件；未本轮运行该生成器"),
        ("GRIP2原论文", "https://arxiv.org/abs/2602.00218v1", "本轮阅读全文和实现细节；旧式实现不能等同官方完整复现"),
        ("HRT/dCRT/LOCO", "Semi-knockoffs 2026相关工作与作者实验代码", "只研究比较关系，未在本轮统一运行"),
    ]
    for name, source, conclusion in references:
        add(rows, name, "参考方法", "仅参考/代码已取得未完整实测", source,
            "论文/v7/supplement_20260913/references" if name in ("非参数顺序/并行残差Knockoff", "GRIP2原论文", "HRT/dCRT/LOCO") else "论文/v7", conclusion)
    with (ARCHIVE / "方法登记表.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (ARCHIVE / "方法登记表.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    table = ["| 方法 | 类别 | 状态 | 结论与限制 |", "| --- | --- | --- | --- |"]
    table += [f"| [{r['method']}]({link(PROJECT / r['evidence'])}) | {r['family']} | {r['status']} | {r['conclusion']} |" for r in rows]
    text = f"""# 所有尝试与参考方法

> 第一次进入项目请先阅读：[项目必读.md](项目必读.md)。该文件集中介绍研究问题、数据口径、实验设计、当前结果、正式交付、复现入口和结论边界。

> 压缩包不包含实际数据。数据下载地址、文件哈希、处理步骤及重建命令见：[数据来源与重建说明.md](数据来源与重建说明.md)。

> 归档包含和排除范围见：[压缩包说明.md](压缩包说明.md)与[压缩排除清单.txt](压缩排除清单.txt)。

本目录整理当前工作区可追溯的{len(rows)}项方法、适配、消融或诊断组件，不代表{len(rows)}个独立原创算法。未找到完整实验输出的方法明确标为未完整实测。正式论文不按版本沿革组织，本档案用于追踪研究过程。

## 必须先看

1. 旧面板按购买月份/周回溯最终送达状态，与月末可见、下一月签收确认的主面板不相同，不能直接比较RMSE。
2. 旧文件中的“14项确认集”等措辞是历史记录，不是本档案重新授予FDR有效性；真实生成器失配与依赖问题依然存在。
3. 0.969是固定数据改变假变量的随机稳定性，不能与70%卖家样本稳定性混比。
4. 原生阈值与Top-K分开；不能把放宽q、修复代码或增加计算预算自动算作模型创新。
5. 档案内CSV以实验分别保存，不将不同模拟场景和训练策略拼成“所有模型总冠军”。
6. 完整工程入口是本地相对符号链接，原工程未移动；报告与小型结果文件另有实际快照。单独搬走此目录不等于完整可运行备份。

## 全部登记

{chr(10).join(table)}

## 工程与复现入口

{chr(10).join(entries)}

## 文献及代码来源

`文献来源/`保存基础与补充文献登记、本次获取的原文及作者代码目录入口。`方法登记表.csv`包含每项的来源、完成状态、结果路径与结论。取得代码不等于运行成功，运行成功不等于理论有效，也不等于原创性得到确认。
"""
    (ARCHIVE / "README.md").write_text(text)
    sources = ARCHIVE / "文献来源"
    sources.mkdir(exist_ok=True)
    for file in (V7 / "references").iterdir():
        if file.is_file():
            shutil.copy2(file, sources / ("基础_" + file.name))
    for file in (ROOT / "references").iterdir():
        if file.is_file():
            shutil.copy2(file, sources / file.name)
    manifest = {}
    for directory in (V7 / "code", ROOT / "code", SCPL_PATH := PROJECT / "论文/方法研究_20260913/code"):
        for file in directory.glob("*.py"):
            manifest[str(file.relative_to(PROJECT))] = hashlib.sha256(file.read_bytes()).hexdigest()
    (ARCHIVE / "源码摘要.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Archived {len(rows)} method/variant/component entries")


if __name__ == "__main__":
    main()
