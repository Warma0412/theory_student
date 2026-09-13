# 交换对称成对损失研究

这是一轮新的方法研究，不是替换已完成的V7论文。目标是研究独立训练预测器上的成对损失统计量，是否能兼顾非线性检出和Knockoff所需的符号翻转性质。

## 工程内容

- `protocol.json`：在本轮结果产生前保存的方案；不是外部预注册。
- `方法定义与证明边界.md`：公式、符号翻转命题、先行文献和适用条件。
- `code/paired_loss.py`：对称背景、损失统计量、预测器及生成器。
- `code/run_experiment.py`：真实抽样、新随机种子模拟和统一预测。
- `code/analyze.py`：完整性核查、配对区间和全部结果汇总。
- `code/check_trained_symmetry.py`：对实际训练后的TabM验证符号性质。
- `data/`：本轮新产生的800份模拟，以及真实评分数据的真假对。
- `models/`：完整开发参考模型。
- `logs/`：运行记录，包括数值计算警告。
- `results/`：逐样本结果、汇总、诊断、审计。
- `研究报告.md/.tex/.pdf`：完成核验后生成的研究报告，不是已获认可的毕业论文。

## 复现

在项目根目录使用既有V7环境：

```bash
论文/v7/.venv/bin/python 论文/方法研究_20260913/code/run_experiment.py init
论文/v7/.venv/bin/python 论文/方法研究_20260913/code/run_experiment.py checks
论文/v7/.venv/bin/python 论文/方法研究_20260913/code/run_experiment.py full --shards 6 --shard 0
```

最后一条需分别运行shard 0至5。分片共享方法和数据，但案例互不重叠；已有JSON跳过。失败记录不自动删除。不要同时启动同一个分片。

```bash
论文/v7/.venv/bin/python 论文/方法研究_20260913/code/run_experiment.py predict --evaluator xgboost
论文/v7/.venv/bin/python 论文/方法研究_20260913/code/run_experiment.py predict --evaluator tabm
论文/v7/.venv/bin/python 论文/方法研究_20260913/code/check_trained_symmetry.py
论文/v7/.venv/bin/python 论文/方法研究_20260913/code/analyze.py summary
论文/v7/.venv/bin/python 论文/方法研究_20260913/code/build_report.py
```

## 范围与约束

1. 14种评分方法/消融共同使用111个真实数据案例、800个新模拟案例。它们不是14个独立神经网络；同一预测器供多种评分方法复用，这是控制变量设计。
2. 真实候选指标31项，模型训练为train+tune，推断为rank，最终预测为test。rank不参与基础预测器早停。
3. 模拟使用1200/400/400模型训练、评分和预测测试分割；前1000/后200用于训练内部选轮次。真值从未用于选参数。
4. 八背景是在一组Knockoff数值对上构造的八种对称取值方式，不是八次独立Knockoff生成。
5. 原生q=0.20、q=0.10，以及固定8/12项名单分开报告。普通替换损失差的名义Knockoff阈值不赋予其正式错误率保证。
6. 真实Olist没有真假标签，且存在生成器失配、同卖家跨期相关和已观察测试集。因此不能宣称真实FDR被证明受控。
7. 对称性证明是本轮统计量的代数检查；误选控制结论来自已有Knockoff理论，不自称新FDR定理。原创性还需更全面检索和导师评估。
8. 全部数字只从实际结果生成，原生空集照样保留，预测使用均值基线。原生发现数量不同不能混作同预算胜负。
9. NumPy/BLAS可能输出matmul运行时警告。所有模型分数、预处理结果、预测和协方差校验都做有限值检查；警告保留在日志，不通过关闭警告隐藏失败。

## 独立证据

新模拟种子从`91320260`开始，与V7及此前探索不同。但场景定义和方法设计仍属于当前探索研究，不能视为外部团队确认。未加载或宣称完成第二真实数据集验证。

## 事后生成器诊断

主实验完整参考暴露等相关S极小的问题后，先做参考样本MVR检查，已看到名义发现数和测试RMSE变化，随后增加了`mvr_extension_protocol.json`。只替换为已有MVR生成器，在同样111个真实样本上追加评分，两种评价器检验Top-12。不覆盖主协议，不声称追加了MVR模拟FDR验证，也不声称追加动机完全不受已见测试结果启发。

```bash
论文/v7/.venv/bin/python 论文/方法研究_20260913/code/run_experiment.py mvr-real --shards 6 --shard 0
论文/v7/.venv/bin/python 论文/方法研究_20260913/code/run_experiment.py mvr-predict --evaluator xgboost
论文/v7/.venv/bin/python 论文/方法研究_20260913/code/run_experiment.py mvr-predict --evaluator tabm
论文/v7/.venv/bin/python 论文/方法研究_20260913/code/analyze.py mvr-summary
```

第一条同样需要分别运行shard 0至5；预测需要reference先完成。`real_mvr`、`predictions_mvr`及`mvr_*`汇总与主结果独立保存。生成器不影响的四项基线排名逐案例断言保持不变。执行中曾先用3分片，全部停止后再以6分片断点续跑，没有同时运行重叠分片。

## 结果核查

`analyze.py summary`和`mvr-summary`会逐行复算预测RMSE、MAE、R²、WAPE，并检查两种原生阈值、排名、有限值、数据摘要，以及主实验和追加实验之间相同集合预测完全一致。汇总和报告代码可在结果产生后完善，模型、样本和阈值协议保持不变；初始及最终代码摘要分别保留。

`runtime_summary.csv`报告各案例总耗时，不把共享预测器训练成本虚构成各评分方法的独立计时。计算量不等于创新程度，论文可行性判断以实际机制和结果为准。
