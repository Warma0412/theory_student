# V7 研究工程

题目：面向电商GMV监控的深度特征选择与稳定性评价研究

## 阅读入口

- `论文_v7.md`、`论文_v7.pdf`：独立论文正文。
- `论文_v7_精简版.pdf`、`论文_v7_极限精简版.pdf`：两种浓缩阅读版。
- `latex工程/`：学校模板、图表依赖、LaTeX源文件、编译日志。
- `../../【最终版】/v7/`：只存论文、开题报告、开题汇报的六份 `.tex/.pdf`。
- 开题报告的签字与审批意见留空，需要相关人员审核后填写。

## 实验范围

完整比较包含Copula-MVR、Elastic Net、稳定性选择、影子变量树、XGBoost-SHAP、Deep Lasso、TabM置换排序、VTFS定额适配八种方法。每方法111个真实数据案例、600个模拟案例；其中真实案例包含1个参考、100个公共卖家样本、10个算法种子批次。五个K、两个下游评价器及两份全维基线形成82组预测产物。

DeepDRK仅完成三种子参考训练、诊断及参考样本下游竞争，未完成全部稳定性与模拟。TabPFN v2在完整数据上发生MPS内存兼容错误。attention-like过滤方法仅完成文献核验，没有伪装成深度网络。三者均不能被视为已完成八方法表中的全部比较。

主分析是月度签收确认商品金额，而不是购买月回溯金额、净收入或因果效应。没有将旧版本的月度/周度结果混入本次比较。Copula-MVR与DeepDRK均有交换性诊断警告，因此真实Olist名单没有无条件FDR证书。

## 关键文件

| 内容 | 路径 |
| --- | --- |
| 本地协议 | `protocol.json` |
| 执行差异 | `执行修订与方法边界.md` |
| 文献登记与Crossref元数据 | `references/papers.json`、`references/verification.json` |
| 原始源表摘要与面板审计 | `results/data_audit.json` |
| 作者代码版本与软件环境 | `results/source_manifest.json` |
| 公共卖家名单 | `data_processed/sampling_manifest.json` |
| 测试前开发选择 | `results/development_choice.json` |
| 逐案例结果 | `results/runs/`、`results/simulation_runs/` |
| 测试逐行预测 | `results/predictions/` |
| 最终覆盖与一致性检查 | `results/final_audit.json` |
| 论文源稿与生成器 | `thesis.md.in`、`code/generate_documents.py` |
| 深度参考模型与训练轨迹 | `models/` |
| 完整运行日志及失败记录 | `logs/` |

原始表位于项目根目录 `原始数据/论文实际使用数据汇总/01_实际读取原始表`。`vendor/`包含锁定的作者代码及VTFS作者压缩包。压缩包中其他数据未用于本研究。

## 重建文档

在项目根目录运行：

```bash
论文/v7/.venv/bin/python 论文/v7/code/validate_and_summarize.py core
论文/v7/.venv/bin/python 论文/v7/code/validate_and_summarize.py summary
论文/v7/.venv/bin/python 论文/v7/code/supplementary_audit.py summarize
论文/v7/.venv/bin/python 论文/v7/code/supplementary_audit.py audit
论文/v7/.venv/bin/python 论文/v7/code/make_figures.py
论文/v7/.venv/bin/python 论文/v7/code/generate_documents.py
bash 论文/v7/code/build_documents.sh
论文/v7/.venv/bin/python 论文/v7/code/check_documents.py
```

需要系统提供Pandoc、Tectonic及中文字体。计算环境为Python 3.11，关键依赖锁在`requirements.txt`；全部已安装包版本另存`source_manifest.json`。最终目录中的 `.tex` 是导出源文件，图表和模板依赖保存在本工程，推荐通过构建脚本编译。

## 重跑计算

运行器以存在的案例JSON断点续跑，不覆盖已经完成的案例。完整重新计算应在工程副本中使用新的结果目录，不删除当前主结果、修订存档或失败记录。

```bash
论文/v7/.venv/bin/python 论文/v7/code/build_panel.py
论文/v7/.venv/bin/python 论文/v7/code/run_experiments.py manifest
论文/v7/.venv/bin/python 论文/v7/code/run_experiments.py reference --method deep_lasso
论文/v7/.venv/bin/python 论文/v7/code/run_experiments.py stability --method deep_lasso
论文/v7/.venv/bin/python 论文/v7/code/run_experiments.py simulate --method deep_lasso
```

将方法参数依次替换为八个完整比较方法。必须在所有参考名单完成后运行 `validate_and_summarize.py choice`，且该步骤必须先于任何测试预测。然后运行 `run_evaluation_suite.py`。不能删除现有开发选择后基于已经观察的测试结果重新冻结。

分片通过案例编号取模划分。只能为同一阶段启动互不重叠的分片；不要在旧分片进程仍运行时改用另一分片数量。训练CDF要求保持现有行排序；数值复现还依赖软件、随机种子及设备。

## 解释约定

1. 固定K的误差、FDP与稳定性只与同K对照；原生集合单独比较。
2. 固定数据种子稳定性不等于更换卖家的样本稳定性。
3. 半样本区间使用30个互补组；测试Bootstrap以卖家为单位并以三个月为条件。
4. 模拟参数迁移自真实开发配置，不代表每个场景充分调参的性能上界。
5. 论文由自动化计算和写作工具辅助形成；正式提交前仍须作者与导师复核论证、署名及学校关于AI辅助的披露要求。没有代填签名或审批结论。
