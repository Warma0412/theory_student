# 多模型算法改动实验

本目录独立于已完成的V7论文和上一次`innovation_pilot`。不修改历史论文、数据和结果。

## 比较范围

| 家族 | 新增配置 | 待检验问题 |
| --- | --- | --- |
| Deep Lasso（2023） | Jacobian训练与排名；仅改排名；改训练保留旧排名 | 去掉残差加权后，特征分数是否更可靠 |
| TabM（2025） | 共享固定K可学习输入门 | 训练输入数量与最终展示数量一致是否有收益 |
| EntryPrune（2025预印本版本） | 回归适配；刷新分数；刷新加环境波动惩罚 | 历史最高进入分数是否锁定偶然高分，刷新是否改善选择 |
| VTFS（2024） | 固定K；固定K加成对排序；等128次查询随机搜索 | 定额解码及学习到的子集效用是否真正贡献，而非靠随机语料回退 |

十个配置均运行完整主设计，不按测试表现决定是否跑完。

## 实验规模

- 每配置主预算K=12：1个完整开发参考、100个卖家样本扰动、10个算法种子批次。
- 100个卖家扰动包括30组互补50%半样本，共60次；20次70%和20次80%。
- 六种共享模拟场景各100次，共600次/配置；与V7基线读取相同NPZ。
- K=8仅做完整参考和双评价器预测，不把K=12名单稳定性写成K=8稳定性。
- 每配置两种K、两种统一评价器，4组预测。
- V7八种基线使用历史完整输出，输入和窗口核对，不计为本轮重新训练。

所有网络使用三个种子。模拟真值不参与调参；所有新方法都不承诺原生FDR控制。固定K下平均FDP和Power随真信号数一一对应。

## 调参和信息隔离

2017年训练，2018年1月早停和生成名单，2月用同一XGBoost评分并固定新增参数。最终参考和重抽样以训练加调参窗口拟合、3至4月排序。2018年5至7月测试只评价，不用于本轮参数选择。

但测试期在历史研究中已被查看，架构和数据方案也受到历史经验影响。因此这仍是探索性研究，不是独立确认。没有声称新增方法是世界首创，或已经通过第二数据集验证。

## 代码入口

在项目根目录执行，使用既有V7独立环境：

```bash
论文/v7/.venv/bin/python 论文/v7/model_improvements_20260912/code/run_study.py init
论文/v7/.venv/bin/python 论文/v7/model_improvements_20260912/code/run_study.py validate
论文/v7/.venv/bin/python 论文/v7/model_improvements_20260912/code/run_study.py tune --family deep
论文/v7/.venv/bin/python 论文/v7/model_improvements_20260912/code/run_study.py full --family deep
```

家族可替换为`deep`、`gate`、`entry`、`subset`。`full`依次运行真实数据和模拟；可用`--shards`及`--shard`运行互不重叠分片。更改分片数前先结束旧分片，已完成JSON会保留，未完成案例重算。

```bash
论文/v7/.venv/bin/python 论文/v7/model_improvements_20260912/code/predict_suite.py
论文/v7/.venv/bin/python 论文/v7/model_improvements_20260912/code/canonical_evaluation.py real --evaluator xgboost
论文/v7/.venv/bin/python 论文/v7/model_improvements_20260912/code/canonical_evaluation.py real --evaluator tabm
论文/v7/.venv/bin/python 论文/v7/model_improvements_20260912/code/canonical_evaluation.py simulate
论文/v7/.venv/bin/python 论文/v7/model_improvements_20260912/code/analyze_study.py summary
论文/v7/.venv/bin/python 论文/v7/model_improvements_20260912/code/build_report.py
```

汇总要求所有案例与预测文件完整。进度检查使用`analyze_study.py progress`，不会把缺失值填0。
规范化评价覆盖全部18种方法及全维基线，共74组真实预测和600组模拟评价；相同集合必须给出相同预测。详见`执行修订记录.md`，初始预测不删除。

## 已知实现差异

1. EntryPrune官方仅提供分类入口。本研究复用其`BaseEP.update_network`作为原分数对照，并使用共同的回归MSE、固定更新预算、CPU和最佳检查点适配。改进不是“原论文回归实验的原样复刻”。
2. EntryPrune刷新与波动版本同时重置再生列的Adam动量，因此刷新对原方法的差异包括分数和动量两项；波动版本对刷新版本才是单独波动项消融。
3. VTFS仍使用官方epsilon=1的编码器，固定K后不再有填充位置和任意前缀。保持128个不同子集的效用预算，神经候选和语料回退来源分别记录。
4. TabM的K个门是每个训练种子共享8成员的硬门；最终跨三个种子按门排序聚合再取K，不声称三份名单完全一致。
5. Jacobian正则和输入门有既有文献。即使胜过当前父实现，也只能先称“实证有效的机制改动”，不能凭实验成绩直接判定原创性。
6. 本轮使用固定K模拟对比，不将历史Copula原生小名单的低FDP与强制K=12结果混比。

## 产物

`protocol.json`保存本地计划；`results/source_manifest.json`保存起始代码、作者来源和数据摘要；每次运行日志在`logs/`；完整参考权重在`models/`；逐样本结果在`results/runs`和`results/simulation_runs`。最终汇总、配对区间、生成候选来源、完整性检查及实验报告在本目录生成。
