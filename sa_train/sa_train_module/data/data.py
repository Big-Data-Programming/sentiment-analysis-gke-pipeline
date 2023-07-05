import os.path
from typing import Dict, Generator, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import AutoTokenizer

from sa_train.sa_train_module.data.data_cleaner import StackedPreprocessor
from sa_train.sa_train_module.data.kaggle_dataset import (
    get_dataset_length,
    get_file_names,
    split_dataset,
)


def kaggle_dataset_iterator(file_name, chunk_size=1000, split_type="train") -> pd.DataFrame:
    data_files = get_file_names(file_name)
    file_map = {"train": data_files[0], "valid": data_files[1], "test": data_files[2]}
    if os.path.isfile(file_map[split_type]) is False:
        split_dataset(file_name)

    dataset_path = file_map[split_type]

    return pd.read_csv(dataset_path, encoding="latin-1", chunksize=chunk_size)


class SentimentIterableDataset(IterableDataset):
    def __init__(
        self,
        csv_file: str,
        tokenizer,
        preprocessors: Optional[Dict] = None,
        chunk_size: int = 1000,
        create_split: bool = False,
        split_type: str = "train",
        batch_size: int = 8,
        max_seq_len: int = 512,
    ):
        self.csv_file = csv_file
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.create_split = create_split
        self.split_type = split_type
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.preprocessors = StackedPreprocessor(preprocessors)

    def __len__(self):
        return get_dataset_length(self.csv_file, self.split_type) // self.batch_size

    def __iter__(self) -> Generator[Tuple[List[str], Dict[str, List[str]]], None, None]:
        for data in kaggle_dataset_iterator(self.csv_file, chunk_size=self.chunk_size, split_type=self.split_type):

            for i in range(0, self.chunk_size, self.batch_size):
                # Label mapping is also done here, 0 - negative sentiment, 1 - positive sentiment
                labels_minibatch: List[int] = list(data.iloc[:, 0].apply(lambda x: 0 if x == 0 else 1).values)[
                    i : i + 8
                ]

                # TODO: Preprocessing steps to be added here
                sentences_minibatch: Dict[str, List[str]] = self.tokenizer(
                    self.preprocessors(list(data.iloc[:, 5].values)[i : i + 8]),
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_seq_len,
                    padding=True,
                )

                labels_tensor = torch.tensor(labels_minibatch)

                sentences = {
                    "input_ids": torch.tensor(sentences_minibatch["input_ids"]),
                    "attention_mask": torch.tensor(sentences_minibatch["attention_mask"]),
                }
                yield labels_tensor, sentences


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base")
    raw_dataset_file = (
        "/home/ppradhan/Documents/my_learnings/my_uni_stuffs/sa_data_storage/training.1600000"
        ".processed.noemoticon.csv"
    )
    # ite = kaggle_dataset_iterator(raw_dataset_file, chunk_size=1000, create_split=False, split_type='train')
    dataset = SentimentIterableDataset(raw_dataset_file, tokenizer)

    total_dataset_length = len(dataset)
    pbar = tqdm(total=total_dataset_length, desc="Processing batches", unit="batch")

    for batch in dataset:
        labels, sentences = batch
        pbar.update(1)
