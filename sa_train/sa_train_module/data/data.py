import pandas as pd
from torch.utils.data import Dataset


def kaggle_dataset_iterator(file_name, chunk_size=1000):
    df_iterator = pd.read_csv(file_name, encoding="latin-1", chunksize=chunk_size)
    for chunk in df_iterator:
        yield chunk


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {"input_ids": input_ids, "attention_mask": attention_mask, "label": label}


def collate_fn(batch):
    pass
