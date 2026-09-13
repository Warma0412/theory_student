## Project Overview

This repository is organized around the four experimental components discussed in the paper:

1. Main experiment: SERS and benchmark comparison.
2. Spurious correlation experiment: Colored MNIST.
3. Ablation experiment.
4. `alpha` sensitivity experiment.

For readers approaching the codebase for the first time, the following order is recommended:

1. Read the root `README.md`.
2. Inspect the four experiment entry points under `experiments/`.
3. Review the core method implementations under `src/models/`.
4. Review the data, evaluation, and analysis code under `src/utils/` and `src/core/`.
