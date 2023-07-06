import pytest
from sa_app.data.data import SentimentIterableDataset
from sa_app.data.data_cleaner import StackedPreprocessor
from tqdm import tqdm
from transformers import AutoTokenizer


@pytest.fixture
def common_text_processor_config():
    config_dict = {
        "base_cleaning": {},
        "lowcase": {},
        "stem": {"language": "english"},
        "lemma": {"model_name": "en_core_web_sm"},
    }
    return config_dict


@pytest.fixture
def sample_file_name():
    return "sample.csv"


@pytest.fixture
def tokenizer_var():
    return AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base")


def test_text_preprocessor(common_text_processor_config):
    preprocessors = StackedPreprocessor(common_text_processor_config)
    sample_sentences = [
        "The quick brown fox jumps over the lazy dog. ",
        "I am running in the - park.",
        "THIs is GOING to be A wonDERFUL daY",
    ]
    expected_sentences = [
        "the quick brown fox jump over the lazi dog .",
        "I be run in the park .",
        "this be go to be a wonder day",
    ]
    results = preprocessors(sample_sentences)

    assert expected_sentences == results


def test_data_loader(tokenizer_var, sample_file_name):
    raw_dataset_file = sample_file_name
    batch_size = 3
    dataset = SentimentIterableDataset(raw_dataset_file, tokenizer_var, batch_size=batch_size)

    total_dataset_length = len(dataset)
    pbar = tqdm(total=total_dataset_length, desc="Processing batches", unit="batch")
    for batch in dataset:
        labels, sentences = batch
        assert sentences["input_ids"].shape[0] <= batch_size
        pbar.update(1)
