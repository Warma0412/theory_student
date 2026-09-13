## Configuration Directory

This directory stores default experiment configurations. The following entry points are currently integrated:

- `exp1_main_benchmark/sers.yaml`
- `exp1_main_benchmark/table2.yaml`

Example usage:

```bash
python experiments/exp1_main_benchmark/run_sers.py
python experiments/exp1_main_benchmark/run_benchmark.py
```

These entry points first load YAML defaults and then allow command-line overrides.

The following configuration files are present as placeholders and are not yet fully integrated:

- `exp2_spurious_colored_mnist/default.yaml`
- `exp3_ablation/default.yaml`
- `exp4_alpha_sensitivity/default.yaml`

These files should currently be treated as work in progress.
