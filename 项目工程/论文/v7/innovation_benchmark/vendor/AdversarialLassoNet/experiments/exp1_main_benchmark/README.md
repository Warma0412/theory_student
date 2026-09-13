## Experiment 1: Main Benchmark

This directory contains the main benchmark experiments reported in the paper:

- Comparison between `LassoNet` and `adv-LassoNet` on SERS data
- Comparison among `LassoNet`, `adv-LassoNet`, `FISTA-Net`, and `Deep-Lasso` on benchmark datasets

Example commands:

```bash
python experiments/exp1_main_benchmark/run_sers.py
python experiments/exp1_main_benchmark/run_benchmark.py --datasets table2 --runs 5 --k 50
```

Default output directory:

- `outputs/exp1_main_benchmark/`
