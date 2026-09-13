# Olist 电商 GMV 维度选择硕士论文（v1）

本目录包含论文正文、全量数据分析代码、中间面板、统计结果、图表和运行日志。正文中的数字均由本目录代码读取原始 CSV 后生成，不使用示例数字。

## 目录

- ../论文_v1_最终版.md：位于“论文”根目录、可直接打开的最终主 Markdown。
- 论文_v1.docx：按通用中文硕士学位论文格式排版的可编辑正文。
- 论文_v1.pdf：49 页 A4 提交前预览版。
- 论文_v1.html：内嵌图表和样式的独立网页版本。
- 论文_v1.md：便于审阅和版本管理的正文源文件。
- code/build_panel.py：多表清洗、自然粒度聚合、地理距离计算与卖家-月面板构造。
- code/run_analysis.py：描述统计、Knockoff、e-value、预测模型、固定效应与稳健性检验。
- code/thesis_template.md：论文正文模板。
- code/generate_thesis.py：从结果文件生成 Markdown、DOCX、HTML 和 PDF。
- data_processed/seller_month_panel.csv：主分析面板。
- results/：机器可读的全部表格、诊断和元数据。
- figures/：论文插图。
- logs/：实际运行日志。

## 一键复现

    cd /Users/bytedance/Documents/trae_projects/theory_brazil
    python3 -m venv --system-site-packages 论文/v1/.venv
    论文/v1/.venv/bin/pip install -r 论文/v1/requirements.txt
    论文/v1/.venv/bin/python 论文/v1/code/build_panel.py
    论文/v1/.venv/bin/python 论文/v1/code/run_analysis.py
    论文/v1/.venv/bin/python 论文/v1/code/generate_thesis.py

macOS 上 XGBoost 还需要执行 brew install libomp。本研究固定随机种子为 20260823。

## 口径说明

主分析使用 2017-01 至 2018-07 的本月特征预测次月 GMV。只保留已送达订单，并在订单、支付、评价、商品明细各自的自然粒度先聚合后连接，避免多对多连接膨胀。共构造 32 个候选维度，其中申报月收入因面板内零方差被排除，31 个变量进入模型。

去随机化 Knockoff 设定最终 `alpha_eBH=0.20`、单轮 `alpha_kn=0.10`，主分析严格入选 14 个维度。Knockoff 结果识别的是控制其余候选变量后仍含次月 GMV 增量预测信息的维度，不应解释为未经实验或准实验识别的因果效应。

封面中的学校、学院、专业、作者、学号、导师和日期为待填字段；提交前应替换并套用所在学校的官方模板。
