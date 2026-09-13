## Paper Experiment Entry Points

This directory is organized by experiment setting rather than by model implementation.

- `exp1_main_benchmark/`: Main benchmark comparing vanilla LassoNet, adv-LassoNet, FISTA-Net, and Deep-Lasso on SERS and benchmark datasets.
- `exp2_spurious_colored_mnist/`: Spurious correlation experiment on Colored MNIST.
- `exp3_ablation/`: Ablation experiment.
- `exp4_alpha_sensitivity/`: `alpha` sensitivity experiment.

Each subdirectory typically contains:

- `run_*.py`: Main entry point for the experiment
- `README.md`: Experiment-specific usage notes
- Optional `summarize.py` / `plot_*.py`: Result aggregation and plotting utilities
