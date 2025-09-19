import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from diabetes_prediction.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, BATCH_SIZE, SEED


def to_tensor_dataset(df: pd.DataFrame) -> TensorDataset:
    """Convert a DataFrame to a TensorDataset."""

    X = df.drop(columns=["Outcome"]).values
    y = df["Outcome"].values
    return TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )


def get_dataloader(
        batch_size: int = BATCH_SIZE,
        seed : int = SEED
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for training, validation and testing."""

    # Read the dataset
    dataset_path = RAW_DATA_DIR / "diabetes.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Please run dataset.py in command line first."
        )

    dataset = pd.read_csv(dataset_path)

    # Split the dataset into features and target
    rest, test = train_test_split(
        dataset,
        test_size=0.2,
        stratify=dataset["Outcome"],
        random_state=seed
    )

    train, val = train_test_split(
        rest,
        test_size=0.25,
        stratify=rest["Outcome"],
        random_state=seed
    )

    # Create TensorDatasets
    train_dataset = to_tensor_dataset(train)
    test_dataset = to_tensor_dataset(test)
    val_dataset = to_tensor_dataset(val)

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_dataloader, test_dataloader, val_dataloader
