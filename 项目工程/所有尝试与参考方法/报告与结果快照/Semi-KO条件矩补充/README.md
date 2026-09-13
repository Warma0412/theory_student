# Semi-knockoffs条件矩补充研究

本目录包含对2026年Semi-knockoffs原文与作者代码的核验、固定Ridge机制适配，以及非线性条件均值、条件尺度和联合扩展。它是探索性方法研究，不是已证明原创的新模型。

## 范围

- 同一批XGBoost和TabM，四种条件重建，共八项评分。
- 111个真实案例：完整参考、30组互补半样本、20个70%、20个80%、10个种子批次。
- 复用SCPL的800份模拟数据，追加6400个方法案例；不是800份新独立数据。
- 8/12两种容量与两个共同评价器，共32组真实预测。
- 所有真实案例的基础预测器置换分数与SCPL父实验逐例核对。
- 基础模型只在train/tune训练，rank响应用于条件重建和评分，这是已披露的算法用途；test不参与选择。

## 原文和实现

作者：Angel Reyero-Lobo、Bertrand Thirion、Pierre Neuvial。

论文：Semi-knockoffs: a model-agnostic conditional independence testing method with finite-sample guarantees，ICML 2026，arXiv:2601.23124v2。

作者代码commit：`ff51ec239529620300e92efd9988eb69490eb3ac`，仓库位于`vendor/loss_based_KO`。原始PDF、HTML及关键源文件在`references/`，没有修改作者源文件。

`results/author_equivalence.json`记录Ridge分数与作者原代码最大差约2e-16。采用固定Ridge alpha=1，不是作者RidgeCV完整实验设置；没有冒充复现原论文全部图表。

作者`utils.knockoff_threshold`在无解时返回最后一个有限候选，`W=[1,2,3,4],q=.2`为反例。本文按论文公式改为无解返回无穷大/空集。该修正不是方法创新，数值一致性检查不等于原生阈值与错误实现一致。

## 推断边界

原文oracle条件期望与估计条件期望的保证不同。条件均值/尺度扩展在零假设下oracle函数相同时具有双侧抽样对称性，但拟合函数不自动获得有限样本FDR控制。尺度模型是log残差平方的稳定化代理，未证明是无偏条件方差。

全部变量统一按数值型重建，少数二元特征可能离开原离散支持。没有实现分类条件概率分支，真实数据也存在跨卖家月份依赖和已观察测试期，结论必须限定为适配实测。

## 复现

在项目根目录执行：

```bash
论文/v7/.venv/bin/python 论文/v7/supplement_20260913/code/run_supplement.py checks
论文/v7/.venv/bin/python 论文/v7/supplement_20260913/code/run_supplement.py full --shards 10 --shard 0
```

第二条分别执行shard 0至9。先前使用6分片，全部停止后改为10分片断点续跑，未同时运行重叠分片。已有结果JSON会跳过，失败不会自动删除或补零。

```bash
论文/v7/.venv/bin/python 论文/v7/supplement_20260913/code/run_supplement.py predict --evaluator xgboost
论文/v7/.venv/bin/python 论文/v7/supplement_20260913/code/run_supplement.py predict --evaluator tabm
论文/v7/.venv/bin/python 论文/v7/supplement_20260913/code/summarize.py
论文/v7/.venv/bin/python 论文/v7/code/rebuild_v7.py
bash 论文/v7/code/build_documents.sh
论文/v7/.venv/bin/python 论文/v7/code/check_documents.py
论文/v7/.venv/bin/python 论文/v7/supplement_20260913/code/build_archive.py
```

`checks`应在新的实验副本中运行以生成初始manifest；完整重算不要删除当前结果或改变已冻结协议。汇总CSV包含共同对照，但其源结果目录和协议分别保留，不能将所有实验拼成同一公平排名。

## 主要结果

联合TabM在强非线性模拟Power为.866、平均FDP为.131；Ridge适配Power为.129。纯交互Power为.966，Ridge为.012。但非线性均值单独版本已达到.856和.959，额外尺度增量通常较小。

联合TabM的真实Top-12 XGBoost评价RMSE约1.8965、半样本J约.4099，弱于SHAP的1.8466和.7682。未标准化SCPL在多个非线性场景Power仍更高，因此不能把相对Ridge的提高说成全面优于已有非线性方法。
