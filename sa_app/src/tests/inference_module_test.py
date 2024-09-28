import os

import pytest
import torch
import wandb
import yaml
from sa_app.inference.inference import InferenceEngine


@pytest.fixture
def sample_config():
    config = yaml.safe_load(open("tests/test_data/test_app_cfg.yml", "r"))
    return config


def test_inference(sample_config):
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    config = sample_config
    device_in_use = "cuda" if torch.cuda.is_available() else "cpu"
    # Example usage
    ie_obj = InferenceEngine(
        inference_params=config["inference_params"],
        training_params=config["training_params"],
        dataset_params=config["dataset_params"],
        device=device_in_use,
    )
    input_sentence = "I feel so bad today . Such a bad day :( "
    predicted_labels = ie_obj.perform_inference(input_sentence)
    assert isinstance(predicted_labels, str) is True
    assert predicted_labels == "negative"
