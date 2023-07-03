## Installation

1. Clone repository.
2. `cd` into cloned directory.
3. Create and activate a new virtual or miniconda python environment with python 3.10 or latest. E.g. for miniconda:
   ```bash
      conda create -n sentiment_analysis_ci_cd
      conda activate sentiment_analysis_ci_cd
   ```
4. Install package in develop mode via `pip install -e .[tests]`.
5. Install `pre-commit` via `pre-commit install`.
   * Optional: Run hooks once on all files via `pre-commit run --all-files`
