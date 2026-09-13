# Olist 电商 GMV 维度选择硕士论文（v2）

本目录在 V1 全量数据与确认性统计主线之上，增加真实运行的深度
Knockoff 生成、多指标交换性诊断、配对竞争 MLP 和残差表格 MLP。
V2 保留未通过诊断的深度生成结果，不将其解释为有效 FDR 发现。

## 正式结果摘要

- V1 确认性主线保持 14 项严格发现、10 项单轮频率至少 90%。
- 深度生成器校准后平均 KS 为 0.016，但 swap 分类器平均 AUC 为
  0.922，未通过预设联合交换性准入。
- 失败的深度生成分支形式上得到 11 项，但不作 FDR 声明。
- 配对竞争 MLP 的反对称性最大误差为 0，在 0.20 水平无严格发现。
- 残差表格 MLP 的时间外 RMSE 为 2.414，略逊于 XGBoost 的 2.388。

## 主要交付

- `../论文_v2_最终版.md`：论文根目录的 V2 主 Markdown。
- `论文_v2.md`、`论文_v2.docx`、`论文_v2.html`、`论文_v2.pdf`：
  V2 完整版。
- `../论文_v2_精简版.pdf`：V2 普通精简版。
- `../论文_v2_极限精简版.pdf`：V2 固定五页极限精简版。
- `code/run_deep_analysis.py`：V2 深度实验与诊断。
- `results/`：逐轮统计量、诊断、预测结果和可复现元数据。
- `models/deep_knockoff_generator.pt`：正式深度生成器 checkpoint。
- `figures/`：V1 基础图和 V2 深度实验图。

## 复现

在项目根目录执行：

```bash
论文/v1/.venv/bin/pip install -r 论文/v2/requirements.txt
论文/v1/.venv/bin/python 论文/v1/code/build_panel.py
论文/v1/.venv/bin/python 论文/v1/code/run_analysis.py
论文/v1/.venv/bin/python 论文/v2/code/run_deep_analysis.py --mode full
论文/v1/.venv/bin/python 论文/v2/code/generate_v2_thesis.py
论文/v1/.venv/bin/python 论文/v2/code/generate_v2_condensed_thesis.py
论文/v1/.venv/bin/python 论文/v2/code/generate_v2_ultra_condensed_thesis.py
```

已有正式生成器 checkpoint 时，可用
`--resume-generator` 跳过生成器训练。该选项只复用模型和训练历史，
后续诊断、逐轮统计量与预测仍会重新运行。

## 结论权限

V1 的 Copula-MVR、60 次 Knockoff 与 e-BH 是确认性主线。V2 深度
生成器只有通过预设交换性门槛后才有资格进入正式 FDR 推断；配对
MLP 是反对称性通过后的非线性稳健性分支；残差 MLP 只承担时间外
预测对照。所有分支使用同一面板哈希与随机种子记录。
