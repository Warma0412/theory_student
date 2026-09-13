## Experiment 3: Ablation

This experiment contains the ablation studies reported in the paper and examines how regularization terms and adversarial-perturbation components affect performance and feature stability.

Example command:

```bash
python experiments/exp3_ablation/run_ablation.py \
  --datasets MICE \
  --runs 5 \
  --ablation all \
  --save-json outputs/exp3_ablation/ablation_summary.json
```
