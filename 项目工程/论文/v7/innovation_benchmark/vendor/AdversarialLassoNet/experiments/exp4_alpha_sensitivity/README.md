## Experiment 4: `alpha` Sensitivity

This experiment studies the sensitivity of model accuracy and feature-selection stability to `adv_alpha`.

Example command:

```bash
python experiments/exp4_alpha_sensitivity/run_alpha_sweep.py \
  --dataset MICE \
  --runs 5 \
  --alphas 0.0,0.2,0.4,0.6,0.8,1.0 \
  --output-dir outputs/exp4_alpha_sensitivity
```
