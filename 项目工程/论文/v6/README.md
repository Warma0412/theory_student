# V6研究与交付说明

## 目录职责

- `code/`：基准比较、组件消融、论文和精简版生成器、LaTeX生成与验收脚本。
- `results/`：真实数据基准、组件消融、模拟比较及继承的月周分析结果。
- `figures/`：正文图及新增基准、重复次数消融、模拟FDP/Power图。
- `data_processed/`：月度和周度建模面板。
- `logs/`：计算日志。
- `latex工程/`：论文、开题报告和开题汇报的完整LaTeX工程及编译日志。
- `V6相较V5改进说明.md`：面向版本比较的修改明细，不属于论文正文。

`【最终版】/v6`只包含论文、开题报告、开题汇报各自的`.tex`与`.pdf`六个交付文件，不放置计算脚本。

## 核心新增结果

- 统一12项预算的五方法真实数据比较。
- 30组配对的70%卖家子样本稳定性比较，所有方法使用同一批卖家。
- Copula/高斯、MVR/SDP/等相关、Lasso/XGBoost和重复次数消融。
- 20轮真实协方差、10个真信号的已知真值模拟。
- 新增图：
  - `fig19_benchmark_comparison.png`
  - `fig20_repetition_ablation.png`
  - `fig21_simulation_fdp_power.png`

## 主要复现命令

```bash
论文/v1/.venv/bin/python 论文/v6/code/run_v6_benchmarks.py
论文/v1/.venv/bin/python 论文/v6/code/run_v6_fair_stability.py
论文/v1/.venv/bin/python 论文/v6/code/generate_v6_thesis.py
论文/v1/.venv/bin/python 论文/v6/code/generate_v6_condensed_thesis.py
论文/v1/.venv/bin/python 论文/v6/code/generate_v6_ultra_condensed_thesis.py
论文/v1/.venv/bin/python 论文/v6/code/generate_final_latex.py
bash 论文/v6/latex工程/build.sh
论文/v1/.venv/bin/python 论文/v6/code/validate_final_latex.py
```

## 验收结果

- SUFE LaTeX论文：76页A4，七章，45条参考文献。
- 开题报告：16页A4，25条参考文献。
- 开题汇报：20页16:9。
- 极限精简版：5页A4。
- 正式论文、开题报告和汇报中可见历史版本标记：0。
- LaTeX致命错误：0。
