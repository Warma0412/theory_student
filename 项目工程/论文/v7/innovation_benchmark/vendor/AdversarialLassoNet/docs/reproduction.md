## Reproduction Guidelines

For reviewers and readers, the recommended reproduction workflow is:

1. Install the project dependencies.
2. Place the required raw datasets under `data/` or set `LASSONET_DATA_DIR`.
3. Launch the target experiment from `experiments/`.
4. Collect outputs from `outputs/exp*/`.

For rigorous reproduction, the following metadata should be recorded alongside the results:

- The random seeds used for each run
- The exact script used to generate each reported table or figure
- Environment information such as Python, PyTorch, and CUDA versions
