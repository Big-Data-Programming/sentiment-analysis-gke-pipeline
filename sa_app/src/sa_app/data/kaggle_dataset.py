from typing import Dict

import pandas as pd


def get_label_counts(df: pd.DataFrame) -> Dict[str, int]:
    label_count = {}
    for i in df:
        k, v = list(i[0].value_counts().to_dict().items())[0]
        if k not in label_count:
            label_count[k] = 0
        label_count[k] += v
    return label_count


def get_dataset_length(file_name: str, split_type: str) -> int:
    data_files = get_file_names(file_name)
    file_map = {"train": data_files[0], "valid": data_files[1], "test": data_files[2]}
    return sum([len(i) for i in pd.read_csv(file_map[split_type], encoding="latin-1", chunksize=1000)])


def get_file_names(file_name: str) -> tuple[str, str, str]:
    file_base_pth, file_ext = file_name.rsplit(".", 1)
    train_file = f"{file_base_pth}.train.{file_ext}"
    valid_file = f"{file_base_pth}.valid.{file_ext}"
    test_file = f"{file_base_pth}.test.{file_ext}"
    return train_file, valid_file, test_file


def split_dataset(
    file: str,
    train_ratio: float = 0.7,
    test_ratio: float = 0.15,
    valid_ratio: float = 0.15,
) -> bool:
    train_file, valid_file, test_file = get_file_names(file)

    # Load the entire dataset
    df = pd.read_csv(file, encoding="latin-1", header=None)

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Calculate the split indices
    total_samples = len(df)
    train_idx = int(train_ratio * total_samples)
    valid_idx = int(valid_ratio * total_samples)

    # Split the dataset
    train_df = df.iloc[:train_idx]
    valid_df = df.iloc[train_idx : train_idx + valid_idx]
    test_df = df.iloc[train_idx + valid_idx :]

    # Save the datasets
    train_df.to_csv(train_file, mode="w", header=False, index=False)
    valid_df.to_csv(valid_file, mode="w", header=False, index=False)
    test_df.to_csv(test_file, mode="w", header=False, index=False)

    print("Data split completed successfully!")
    return True


if __name__ == "__main__":
    raw_dataset_file = (
        "/home/ppradhan/Documents/my_learnings/my_uni_stuffs/sa_data_storage/training.1600000"
        ".processed.noemoticon.csv"
    )
    print(split_dataset(raw_dataset_file))
