## Experiment 2: Colored MNIST Spurious Correlation

This experiment evaluates vanilla `LassoNet` and `adv-LassoNet` on Colored MNIST under spurious color-label correlations.

Example command:

```bash
python experiments/exp2_spurious_colored_mnist/run_comparison.py \
  --runs 5 \
  --k 50 \
  --save-json outputs/exp2_spurious_colored_mnist/colored_mnist_report.json
```
