![pypi_upload](https://github.com/Big-Data-Programming/bdp2_apr22_exam-bdp2_apr22_group_2/actions/workflows/publish-to-pypi.yml/badge.svg)
![docker_frontend](https://github.com/Big-Data-Programming/bdp2_apr22_exam-bdp2_apr22_group_2/actions/workflows/push-frontend-container-to-hub.yaml/badge.svg)
![docker_inference](https://github.com/Big-Data-Programming/bdp2_apr22_exam-bdp2_apr22_group_2/actions/workflows/push-inference-container-to-hub.yaml/badge.svg)
![docker_training](https://github.com/Big-Data-Programming/bdp2_apr22_exam-bdp2_apr22_group_2/actions/workflows/push-training-container-to-hub.yaml/badge.svg)
![pytest](https://github.com/Big-Data-Programming/bdp2_apr22_exam-bdp2_apr22_group_2/actions/workflows/pytests-all-modules.yml/badge.svg)

## A fully GKE managed Sentiment analysis application with language model training and inference schemes automated using github actions



## Architecture Overview

#### Block diagram to show the internal working of this project
![Untitled Diagram-Page-2 (1)](https://github.com/Big-Data-Programming/sentiment-analysis-gke-pipeline/assets/11462012/874ffd59-e90e-4197-9519-6385c00033f8)


#### Block diagram to show the github action automation to establish the CI/CD pipeline
![Big Data Programming 2](https://github.com/Big-Data-Programming/sentiment-analysis-gke-pipeline/assets/11462012/b58b962a-aa46-4706-be10-3ac8317b5a55)


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


## Wandb cmds

1. To upload a dataset make sure you have the data in you local, wandb account project created and then run the below cmd (you can also upload multiple files):
   * Run `python sa_app/scripts/wandb_init.py --entity <you_user_name> --project <name of wandb project> --artifact_name <artifact name> --artifact_locations <artifact local path>`
   * Example `python sa_app/scripts/wandb_init.py --entity prabhupad26 --project sa-roberta --artifact_name sentiment-dataset --artifact_locations <you can provide multple files separated by a whitespace>` 
2. To manually download the datasets run the below command with the desired file name:
   * `wandb artifact get prabhupad26/sa-roberta/sentiment-dataset:latest --root training.1600000.processed.noemoticon.csv`


## Run training

If running for the first time follow below steps :
1. Download dataset from [here](https://www.kaggle.com/datasets/kazanova/sentiment140/download?datasetVersionNumber=2)
2. Run `python -m spacy download en_core_web_sm`
3. `cd sa_app/src`
4. Run this cmd from the root dir of training module (i.e. sa_train/sa_train_module) : `python training/train.py --config <path to train_cfg.yml file>`

## Build docker image

Run below cmds from the root path of this repo
1. `docker build -t <name the image> .`
2. `docker run -p 5000:5000 <name of the image>`

## Workflow rules :
For training :
1. wildcard - `training/*` if matched will run the model-training workflow

## TODOS

- [x] **Add Starter Code:** Initial codebase setup is complete.
- [x] **Complete Training Code:** Training code for the model is implemented.
- [x] **Setup Pre-commit:** Pre-commit hooks for code quality are configured.
- [ ] **Add More Data Preprocessing Steps:** Additional data preprocessing steps are in progress.
- [x] **Create Three Containers:**
    - [x] **For Training via Flask API:** Container setup for training and inferencing is pending.
    - [x] **For Inferencing via Flask API:** Container setup for training and inferencing is pending.
    - [x] **For Dashboard APP Deployment for GCP App Engine:** Container setup for deploying the dashboard app is pending.
- [x] **Complete Inference Code in `sa_inference_module`:** Flask API for inference is not yet implemented.
- [x] **Version Control on Dataset and Model:** Explore MLFlow integration for dataset and model versioning.
- [x] **CI / CD (DevOps):** Continuous Integration (CI) and Continuous Deployment (CD) setup is pending.
  - [ ] **Maintain the version of containers in a single file** : currently it is being updated in multiple files (.github/workflows and kubernetes-manifest)
