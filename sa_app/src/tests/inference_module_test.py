import os

import pytest
import torch
import wandb
import yaml
from sa_app.inference.inference import InferenceEngine


@pytest.fixture
def sample_config():
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    config = yaml.safe_load(open("tests/test_data/test_app_cfg.yml", "r"))
    device_in_use = "cuda" if torch.cuda.is_available() else "cpu"
    # Example usage
    ie_obj = InferenceEngine(
        inference_params=config["inference_params"],
        training_params=config["training_params"],
        dataset_params=config["dataset_params"],
        device=device_in_use,
    )
    return ie_obj


def test_inference_negative(sample_config):
    engine_obj = sample_config
    input_sentence = "I feel so bad today . Such a bad day :( "
    predicted_labels = engine_obj.perform_inference(input_sentence)
    assert isinstance(predicted_labels, str) is True


def test_inference_positive(sample_config):
    engine_obj = sample_config
    input_sentence = """
    @swikey haha, okay, i feel much better now.
    let's just dye our hair paramore red!
    """
    predicted_labels = engine_obj.perform_inference(input_sentence)
    assert isinstance(predicted_labels, str) is True
