# Olist 电商 GMV 维度选择论文（v3）

V3 完整继承 V1 的确认性 FDR 结果和 V2 的深度学习诊断，新增参数化
维度选择：

1. FDR 0.20 下保留 14 项正式确认集；
2. 用训练期模型在验证集上的 TreeSHAP 对14项排序；
3. 用户通过 `K=1...14` 决定保留前多少项；
4. 系统提供 K=4极简、K=8均衡、K=14完整，并自动推荐默认K。

默认规则为：选择累计验证SHAP贡献至少90%，且验证RMSE距离所有K
中的最优值不超过0.5%的最小K。本数据推荐 `K=10`。

默认K=10的测试RMSE为2.385338，全31维模型为2.387664，默认模型
反而低约0.097%。任意K集合都用于展示，不替代14项正式FDR确认集。

## 交付文件

- `论文_v3.pdf`、`论文_v3.docx`、`论文_v3.html`、`论文_v3.md`
- `../论文_v3_最终版.md`
- `../论文_v3_精简版.pdf`
- `../论文_v3_极限精简版.pdf`
- `results/v3_ranked_dimensions.csv`
- `results/v3_selection_path.csv`
- `results/v3_shortlist.csv`（默认K）
- `results/v3_predictive_metrics.csv`

## 复现

```bash
论文/v1/.venv/bin/python 论文/v3/code/run_v3_shortlist.py
论文/v1/.venv/bin/python 论文/v3/code/run_v3_shortlist.py --k 4
论文/v1/.venv/bin/python 论文/v3/code/run_v3_shortlist.py --k 8
论文/v1/.venv/bin/python 论文/v3/code/generate_v3_thesis.py
论文/v1/.venv/bin/python 论文/v3/code/generate_v3_condensed_thesis.py
论文/v1/.venv/bin/python 论文/v3/code/generate_v3_ultra_condensed_thesis.py
```

不传 `--k` 时采用模型推荐值。也可调整默认推荐规则：

```bash
论文/v1/.venv/bin/python 论文/v3/code/run_v3_shortlist.py \
  --contribution-target 0.90 --rmse-tolerance 0.005
```
