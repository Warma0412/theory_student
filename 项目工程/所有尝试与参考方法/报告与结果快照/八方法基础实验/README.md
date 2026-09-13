# V7研究工程

题目：面向电商经营指标筛选的非线性方法与交换对称损失研究。

本次重建将基础方法、模型机制改动、SCPL及Semi-knockoffs条件矩补充组织为一篇独立论文，不在正式正文中按版本沿革叙述。没有将负结果改写为方法成功或确认性发现。

## 阅读与交付

- `论文_v7.md/.pdf`：完整论文。
- `论文_v7_精简版.md/.tex/.pdf`：浓缩阅读版。
- `论文_v7_极限精简版.md/.tex/.pdf`：快速阅读版。
- `开题报告_v7.md`、`latex工程/开题报告/`：开题报告。
- `latex工程/开题汇报/`、`开题汇报_v7_演讲稿.md`：Beamer汇报与逐页讲稿。
- `../../【最终版】/v7/`：论文、开题报告、开题汇报各自的TeX/PDF六份成品。
- `../../所有尝试与参考方法/`：全方法登记、来源、失败记录、报告快照及完整工程入口。
- `文档重建备份_20260913/`：重建前的正式文件、正文源稿、生成器和模板依赖备份。

最终TeX的模板和图表依赖在`latex工程`，独立复制最终目录六个文件不等于带齐完整编译环境。开题签名与审批意见留空，供相关教师填写。

## 实验分层

| 设计 | 路径 | 完成范围 |
| --- | --- | --- |
| 基础八方法 | `results/` | 888真实方法案例、4800模拟方法案例、82原预测输出 |
| 模型机制十项改动 | `model_improvements_20260912/` | 1120真实方法案例、6000模拟方法案例、74组规范列顺序预测 |
| 梯度一致性/变量质量探索 | `innovation_pilot/` | 局部实测；gamma=0入选，新增惩罚未获支持 |
| SCPL | `../方法研究_20260913/` | 14评分，800份新模拟，111真实；MVR另111真实，合计108组预测 |
| 条件矩补充 | `supplement_20260913/` | 8评分，111真实，复算同800份模拟，32组预测 |

正式基础Top-12表采用`model_improvements_20260912/results/comparison_overview.csv`的规范列顺序结果，不再把旧列顺序造成的预测微差作为改进。SCPL与条件矩研究的基础模型不得用评分响应早停，其同名基线与基础设计不是同一拟合，分表解释。

同一份数据上重复算多个方法不能算成更多独立数据。Top-K不继承原生FDR，名义阈值发现不等于真实数据已被理论确认。均值尺度扩展的模拟收益主要来自非线性条件均值，真实预测和稳定性仍不及强基线。

## 重建文档

```bash
论文/v7/.venv/bin/python 论文/v7/supplement_20260913/code/summarize.py
论文/v7/.venv/bin/python 论文/v7/code/rebuild_v7.py
bash 论文/v7/code/build_documents.sh
论文/v7/.venv/bin/python 论文/v7/code/check_documents.py
论文/v7/.venv/bin/python 论文/v7/supplement_20260913/code/build_archive.py
```

兼容入口`code/generate_documents.py`现在转调新生成器。`thesis_revised.md.in`为当前正文模板；`thesis.md.in`是保留的基础研究源模板，不能单独生成当前完整论文。所有章节数字从各自已审计CSV/JSON读取。

需要既有Python3.11环境、Pandoc、Tectonic与中文字体。作者代码与软件依赖不随文档重建重新安装。原始十张表位于项目根目录`原始数据/论文实际使用数据汇总/01_实际读取原始表`。

## 核验入口

- `results/final_audit.json`：基础实验。
- `model_improvements_20260912/results/audit.json`：模型机制与规范预测。
- `../方法研究_20260913/results/audit.json`及`mvr_audit.json`：SCPL。
- `supplement_20260913/results/audit.json`：条件矩补充。
- `supplement_20260913/results/author_equivalence.json`：作者代码分数核对及阈值反例。
- `results/document_generation.json`：当前文档来源与引用数。
- `results/pdf_validation.json`、`results/pdf_checks/`：成品页数、边界与逐页检查。
- `references/`与`supplement_20260913/references/`：基础及追加文献原文/元数据。

研究只讨论预测性选维，不作因果归因或永久指标准入。生成器与估计条件矩的理论前提尚未充分满足；测试期已被探索查看，没有第二真实数据的独立确认。正式提交前须由作者与导师复核内容，并遵守学校AI辅助与学术诚信要求。
