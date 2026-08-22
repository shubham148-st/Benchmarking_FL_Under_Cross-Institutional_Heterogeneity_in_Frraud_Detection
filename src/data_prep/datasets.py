"""
PyTorch Dataset wrapper for harmonized fraud detection data.

Converts pandas DataFrames from the harmonization pipeline into
PyTorch-compatible datasets for use with DataLoader.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from .harmonizer import FEATURE_COLUMNS


class FraudDataset(Dataset):
    """
    PyTorch Dataset for harmonized fraud detection data.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonized DataFrame with columns matching HARMONIZED_COLUMNS.
    """

    def __init__(self, df: pd.DataFrame):
        self.features = torch.tensor(
            df[FEATURE_COLUMNS].values, dtype=torch.float32
        )
        self.labels = torch.tensor(
            df["label"].values, dtype=torch.float32
        )
        self.n_features = len(FEATURE_COLUMNS)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

    @property
    def fraud_rate(self) -> float:
        """Fraction of positive (fraud) labels."""
        return float(self.labels.mean())

    @property
    def pos_weight(self) -> torch.Tensor:
        """
        Weight for BCEWithLogitsLoss to handle class imbalance.
        pos_weight = n_negative / n_positive
        """
        n_pos = self.labels.sum()
        n_neg = len(self.labels) - n_pos
        if n_pos == 0:
            return torch.tensor(1.0)
        return n_neg / n_pos


def create_dataloader(
    df: pd.DataFrame,
    batch_size: int = 256,
    shuffle: bool = True,
    use_weighted_sampling: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader from a harmonized DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonized data.
    batch_size : int
        Batch size.
    shuffle : bool
        Whether to shuffle (ignored if use_weighted_sampling=True).
    use_weighted_sampling : bool
        If True, uses WeightedRandomSampler to oversample the minority class.
    num_workers : int
        DataLoader workers.

    Returns
    -------
    DataLoader
    """
    dataset = FraudDataset(df)

    sampler = None
    if use_weighted_sampling:
        # Compute per-sample weights: minority class gets higher weight
        labels = df["label"].values
        class_counts = np.bincount(labels.astype(int))
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels.astype(int)]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float64),
            num_samples=len(dataset),
            replacement=True,
        )
        shuffle = False  # Sampler and shuffle are mutually exclusive

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
