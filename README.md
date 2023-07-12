![example workflow](https://github.com/Big-Data-Programming/bdp2_apr22_exam-bdp2_apr22_group_2/actions/workflows/python-package-conda.yml/badge.svg)


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


## Run training

If running for the first time follow below steps :
1. Download dataset from [here](https://www.kaggle.com/datasets/kazanova/sentiment140/download?datasetVersionNumber=2)
2. Run `python -m spacy download en_core_web_sm`
3. `cd sa_app/src`
4. Run this cmd from the root dir of training module (i.e. sa_train/sa_train_module) : `python training/train.py --config <path to train_cfg.yml file>`


## TODOS

- [x] Add starter code
- [x] Complete Training code
- [x] Setup pre-commit
- [ ] Add more data preprocessing steps
- [ ] Create two containers :
      - [ ] For training / inferencing via Flask API
      - [ ] For dashboard APP deployment for GCP App engine.
- [ ] Complete inference code in `sa_inference_module`
- [ ] Version control on dataset, model (Explore MLFlow)
- [ ] CI / CD (Devops)
