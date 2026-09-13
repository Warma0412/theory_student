# 多粒度GMV维度选择论文

本目录包含一篇独立的统计学硕士论文及其完整计算产物。研究以卖家-月面板作为确认性主分析，以卖家-周面板检验时间聚合粒度稳健性，并使用参数K控制最终展示维度数量。

## 写作与排版依据

- 中文学位论文的章节主线、摘要闭合、实验讨论和结论结构参考`academic-writing-skills/latex-thesis-zh`；
- 通用排版参考活跃的高校论文模板项目，包括`thuthesis`、`SJTUThesis`和`ustcthesis`；
- 当前未指定学校，因此采用A4、宋体正文、黑体标题、1.5倍行距和通用硕士论文结构；正式提交时应以所在学校最新模板覆盖封面、声明页和细部格式。

## LaTeX工程与最终交付

- 完整LaTeX工程、模板资源、图片、日志、构建脚本和验收报告统一放在`latex工程/`；
- `【最终版】/v5`只发布论文、开题报告和开题汇报三组`.tex/.pdf`成品；
- 开题报告依据`模板/吴优_开题报告.pdf`的A4表单与正文结构重建。

## 核心结果

- 周度面板：33,190个卖家-周、2,834个卖家、83周；
- 下一周零成交率：40.46%；
- 周度Copula-MVR、60轮e-value、最终q=0.20：18项；
- 月度14项与周度18项交集：11项，Jaccard=0.524；
- 周度单轮频率至少90%：13项；
- 卖家间层：12项；卖家内层：0项；
- Group Knockoff q=0.20：10组；
- GRIP2式 q=0.20：0项；q=0.30：12项；
- 周度最佳模型：Extra Trees，测试RMSE=2.416；
- 周度18项XGBoost：测试RMSE=2.421。

## 运行顺序

```bash
论文/v1/.venv/bin/python 论文/v5/code/build_weekly_panel.py
论文/v1/.venv/bin/python 论文/v5/code/run_v5_weekly_analysis.py
论文/v1/.venv/bin/python 论文/v5/code/generate_v5_thesis.py
论文/v1/.venv/bin/python 论文/v5/code/generate_v5_condensed_thesis.py
论文/v1/.venv/bin/python 论文/v5/code/generate_v5_ultra_condensed_thesis.py
论文/v1/.venv/bin/python 论文/v5/code/validate_v5_outputs.py
```

## 主要交付

- `论文_v5.md`、`论文_v5.docx`、`论文_v5.html`、`论文_v5.pdf`：独立完整论文；
- `../论文_v5_最终版.md`：论文目录根层的最终Markdown；
- `../论文_v5_精简版.pdf`：精简介绍版；
- `../论文_v5_极限精简版.pdf`：无封面、固定A4五页版；
- `results/validation_report.json`：自动验收结果；
- `results/v5_weekly_reproducibility.json`：参数、版本和结果元数据；
- `results/*_w.csv`：每轮Knockoff或GRIP2统计量；
- `logs/`：面板构建和完整分析日志。

## 口径边界

周度面板沿用月度研究的回溯性口径：最终已送达订单按购买周归属。该设计保证月周统计口径尽量一致，但物流和评价信息在相应周末未必已经可见，因此V5不是严格的在线部署回放。实际部署应按字段真实可见时间重新归属或滞后。
