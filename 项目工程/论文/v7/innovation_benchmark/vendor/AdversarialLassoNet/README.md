# Adversarial LassoNet

This repository contains the experimental code for Adversarial / SAM-style LassoNet. It includes the main benchmark study, the SERS binary classification experiment, the Colored MNIST spurious-correlation study, and ablation analyses.

This repository includes a vendored implementation of `lassonet`, released under the MIT License and credited to the original authors:
Ismael Lemhadri et al. The upstream project is available at <https://github.com/lasso-net/lassonet>. The corresponding license files are preserved under `src/models/lassonet/` and `src/models/lassonet_sers/`.

Part of the tabular data loading logic in `src/utils/data_utils.py` is adapted with reference to Concrete Autoencoders:
<https://github.com/mfbalin/Concrete-Autoencoders>.

## Overview

The codebase is organized around four experimental settings:

- Main benchmark experiments on SERS and tabular/image benchmarks
- Spurious-correlation evaluation on Colored MNIST
- Ablation studies for adversarial and regularization components
- Sensitivity analysis with respect to `adv_alpha`

## Repository Structure

```text
Adversarial-LassoNet/
|- configs/         # Default experiment configurations
|- data/            # Data notes and expected local directory layout
|- docs/            # Reproduction and project documentation
|- experiments/     # Primary experiment entry points
|- outputs/         # Default output location
|- scripts/         # Legacy-compatible script entry points
|- src/             # Core models, data utilities, and analysis code
|- pyproject.toml   # Package configuration
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Windows:

```bash
.venv\Scripts\activate
pip install -e .
```

Recommended environment:

- Python 3.10 or 3.11
- PyTorch 2.x
- CUDA 11.8+ (optional; CPU also works, but benchmark runs will be slower)

## Data Preparation

The default data root is `data/raw/`. It can be overridden with an environment variable:

```bash
export LASSONET_DATA_DIR=/path/to/data/raw
```

Windows PowerShell:

```powershell
$env:LASSONET_DATA_DIR = "D:\\datasets\\adversarial-lassonet"
```

Recommended directory structure:

```text
data/
|- raw/
|  |- sers/
|  |  |- HealthyControl0.csv
|  |  |- LungCancer0.csv
|  |- mice/
|  |  |- Data_Cortex_Nuclear.csv
|  |- isolet/
|  |  |- isolet1234.data
|  |  |- isolet5.data
|  |- coil-20-proc/
|  |- activity/
|  |  |- final_X_train.txt
|  |  |- final_X_test.txt
|  |  |- final_y_train.txt
|  |  |- final_y_test.txt
|  |- torchvision/
```

See [data/README.md](data/README.md) for expected files and release notes.

Notes:

- `MNIST` and `FashionMNIST` are downloaded automatically by `torchvision` into `data/raw/torchvision/`
- The repository currently includes example CSV files under `data/sers/`; redistribution constraints should be reviewed before public release

## Quick Start

Minimal smoke test:

```bash
python experiments/exp1_main_benchmark/run_sers.py --mode clean --seed 42
```

Main benchmark:

```bash
python experiments/exp1_main_benchmark/run_benchmark.py
```

Additional experiments:

```bash
python experiments/exp2_spurious_colored_mnist/run_comparison.py
python experiments/exp3_ablation/run_ablation.py --datasets MICE --ablation all --runs 5
python experiments/exp4_alpha_sensitivity/run_alpha_sweep.py --dataset MICE --runs 5
```

## Outputs

By default, results are written to:

- `outputs/exp1_main_benchmark/`
- `outputs/exp2_spurious_colored_mnist/`
- `outputs/exp3_ablation/`
- `outputs/exp4_alpha_sensitivity/`

## Reproducibility

- `configs/exp1_main_benchmark/` is currently wired into `run_sers.py` and `run_benchmark.py`
- The `scripts/` directory is retained for backward compatibility; `experiments/` is the recommended public interface
- Full reproduction should record Python, PyTorch, CUDA, random seeds, and the data directory used for each run

## Citation

If this repository is used in academic work, please cite the original LassoNet paper and acknowledge this repository as the source of the experimental implementation.

## License

The main code in this repository is released under the MIT License. See [LICENSE](LICENSE).
