import torch
import numpy as np
import pandas as pd
import diabetes_prediction.config as config
from typing import Tuple
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

class DiabetesDataset(Dataset):
    """Selfmade Dataset class to have the flexibility if needed."""

    def __init__(
            self,
            data : np.array,
            targets : np.array
    ):
        self.data = torch.tensor(data.astype(torch.float32))
        self.targets = torch.tensor(targets.astype(torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def get_dataloader(
        path: Path = config.RAW_DATA_DIR / 'diabetes.csv',
        batch_size : int = config.BATCH_SIZE
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """ Creates and returns dataloaders using a self defined dataloader class.

    path : path to the source of the used data. specified in config.py
    batch_size : number of items in each batch if possible. specified in config.py

    return : Tuple of Dataloader objects for the train, test and validation sets

    The data is split (70/15/15), and a StandardScaler is fit ONLY on
    the training data to prevent data leakage.
    """

    # Read the dataset
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}."
        )
    data = pd.read_csv(path)

    # Drop target from data. Train_test_split takes the order X and y come in into account.
    X = data.drop("Outcome", axis=1).values
    y = data["Outcome"].values

    # Split the data into train and a rest. Use stratify to keep a reasonable ratio of target values
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,  #70 percent train set
        stratify=y,
        random_state=config.SEED
    )

    # Scale the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_temp = scaler.transform(X_temp)

    # Split into val and test
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,  # 15 percent train and 15 percent val set
        stratify=y_temp,
        random_state=config.SEED
    )

    # Wrap it in DiabetesDatasets
    train_ds = DiabetesDataset(X_train, y_train)
    test_ds = DiabetesDataset(X_test, y_test)
    val_ds = DiabetesDataset(X_val, y_val)

    # Wrap those in DataLoaders
    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False)

    return train_loader, test_loader, val_loader

