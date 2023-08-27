import os.path
from glob import glob
from typing import Dict, Generator, List, Optional, Tuple

import pandas as pd
import torch
import wandb
from sa_app.data.data_cleaner import StackedPreprocessor
from sa_app.data.kaggle_dataset import get_dataset_length, get_file_names, split_dataset
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import AutoTokenizer


def kaggle_dataset_iterator(file_map: dict, chunk_size=1000, split_type="train") -> pd.DataFrame:
    dataset_path = file_map[split_type]
    return pd.read_csv(dataset_path, encoding="latin-1", chunksize=chunk_size)


class InitializeDataset:
    def __init__(self, dataset_params):
        self.dataset_params = dataset_params

    def __call__(self, *args, **kwargs) -> Tuple[str, str]:
        if self.dataset_params.get("wandb_storage") is not None:
            wandb_storage = self.dataset_params.get("wandb_storage")
            wandb_user_id = wandb_storage.get("wandb_user_id")
            wandb_project_name = wandb_storage.get("wandb_project_name")
            wandb_artifact_name = wandb_storage.get("wandb_artifact_name")
            wandb_artifact_type = wandb_storage.get("wandb_artifact_type")
            wandb_file_type = wandb_storage.get("training_file_type")
            wandb_artifact_version = wandb_storage.get("wandb_artifact_version")
            labels_mapping_file_name = wandb_storage.get("labels_mapping_file_name")
            run = wandb.init(entity=wandb_user_id, project=wandb_project_name, job_type="download_dataset")
            artifact = run.use_artifact(
                f"{wandb_user_id}/{wandb_project_name}/{wandb_artifact_name}:{wandb_artifact_version}",
                type=f"{wandb_artifact_type}",
            )
            artifact_dir = artifact.download()
            assert len(glob(f"{artifact_dir}/*.{wandb_file_type}")) > 0, "CSV file download failed"
            csv_file = glob(f"{artifact_dir}/*.{wandb_file_type}")[0]
            mapping_file = os.path.join(artifact_dir, labels_mapping_file_name)
            assert os.path.isfile(mapping_file) is True, "Label mapping file download failed"
            return csv_file, mapping_file
        elif self.dataset_params.get("local_storage") is not None:
            local_storage = self.dataset_params.get("local_storage")
            csv_file = local_storage.get("raw_dataset_file")
            mapping_file = local_storage.get("labels_mapping")
            return csv_file, mapping_file
        else:
            raise NotImplementedError(f"Either of wandb_storage or local_storage should be defined in app_cfg.yml")


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
        data_files = get_file_names(self.csv_file)
        self.file_map = {"train": data_files[0], "valid": data_files[1], "test": data_files[2]}
        if os.path.isfile(self.file_map[split_type]) is False:
            print("Splitting the dataset")
            split_dataset(self.csv_file)

    def __len__(self):
        return get_dataset_length(self.csv_file, self.split_type) // self.batch_size

    def __iter__(self) -> Generator[Tuple[List[str], Dict[str, List[str]]], None, None]:
        for data in kaggle_dataset_iterator(self.file_map, chunk_size=self.chunk_size, split_type=self.split_type):
            for i in range(0, len(data), self.batch_size):
                # Label mapping is also done here, 0 - negative sentiment, 1 - positive sentiment
                labels_minibatch: List[int] = list(
                    data.iloc[i : i + self.batch_size, 0].apply(lambda x: 0 if x == 0 else 1).values
                )

                sentences_minibatch: Dict[str, List[str]] = self.tokenizer(
                    self.preprocessors(list(data.iloc[i : i + self.batch_size, 5].values)),
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
