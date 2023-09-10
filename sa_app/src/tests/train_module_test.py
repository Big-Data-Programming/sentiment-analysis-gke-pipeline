import pytest
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from sa_app.app import init_model_callbacks, load_model


@pytest.fixture
def sample_config():
    config = yaml.safe_load(open("tests/test_data/test_app_cfg.yml", "r"))
    return config


def test_load_model(sample_config):
    config = sample_config
    model_params = load_model(config["training_params"])
    assert isinstance(model_params, tuple) is True


def test_init_model_callbacks(sample_config):
    config = sample_config
    callbacks = init_model_callbacks(config["training_params"])
    assert isinstance(callbacks, list) is True
    assert isinstance(callbacks[0], LearningRateMonitor) is True
    assert isinstance(callbacks[1], ModelCheckpoint) is True
