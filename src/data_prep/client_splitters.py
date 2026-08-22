"""
Client data splitting strategies for the 3-stage ablation framework.

Stage 1 – Natural Heterogeneity:
    Each dataset = one client. Natural fraud rates preserved.

Stage 2 – Base-Rate Balanced:
    Same as Stage 1, but downsample legitimate class so all clients
    share an identical fraud rate (default 1.0%).

Stage 3 – Synthetic Dirichlet Control:
    Single dataset (IEEE-CIS) split into 3 synthetic clients via
    Dirichlet distribution on the label.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from sklearn.model_selection import train_test_split

from .harmonizer import harmonize_dataset, normalize_features, FEATURE_COLUMNS


# ── Subsampling helper ───────────────────────────────────────────────────────

def _stratified_subsample(
    df: pd.DataFrame,
    max_samples: int | None,
    seed: int = 42,
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Stratified subsample to cap dataset size while preserving class distribution.

    If df has fewer than max_samples rows, returns df unchanged.
    """
    if max_samples is None or len(df) <= max_samples:
        return df

    frac = max_samples / len(df)
    sampled = df.groupby(label_col, group_keys=False).apply(
        lambda x: x.sample(frac=frac, random_state=seed)
    )
    # Adjust to hit exactly max_samples (rounding may cause small diffs)
    if len(sampled) > max_samples:
        sampled = sampled.sample(n=max_samples, random_state=seed)
    return sampled.reset_index(drop=True)


# ── Stage 1: Natural Domain Heterogeneity ────────────────────────────────────

def split_stage1_natural(
    data_root: str | Path,
    test_size: float = 0.2,
    seed: int = 42,
    max_samples_per_client: int | None = None,
) -> list[dict]:
    """
    Stage 1: Natural heterogeneity split.

    Each dataset becomes one client with its natural fraud rate.
        Client 0 = IEEE-CIS  (~3.5% fraud)
        Client 1 = Feedzai BAF (~1.1% fraud)
        Client 2 = PaySim    (~0.13% fraud)

    Parameters
    ----------
    data_root : str | Path
        Root data directory containing ieee_cis/, feedzai_baf/, paysim/.
    test_size : float
        Fraction of data reserved for testing (stratified by label).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[dict]
        List of client data dicts, each with keys:
        'name', 'train_df', 'test_df', 'train_scaler', 'fraud_rate'
    """
    data_root = Path(data_root)
    dataset_names = ["ieee_cis", "feedzai_baf", "paysim"]
    clients = []

    for name in dataset_names:
        data_dir = data_root / name
        df = harmonize_dataset(name, data_dir)

        # Stratified train/test split
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=seed,
            stratify=df["label"],
        )

        # Subsample training data if it exceeds cap
        train_df = _stratified_subsample(train_df, max_samples_per_client, seed)

        # Normalize features (fit on train only)
        train_df, scaler = normalize_features(train_df)
        test_df, _ = normalize_features(test_df, scaler=scaler)

        fraud_rate = train_df["label"].mean()

        clients.append({
            "name": name,
            "train_df": train_df.reset_index(drop=True),
            "test_df": test_df.reset_index(drop=True),
            "train_scaler": scaler,
            "fraud_rate": fraud_rate,
        })

        print(f"  [Stage 1] Client '{name}': "
              f"train={len(train_df)}, test={len(test_df)}, "
              f"fraud_rate={fraud_rate:.4f}")

    return clients


# ── Stage 2: Base-Rate Balanced ──────────────────────────────────────────────

def split_stage2_balanced(
    data_root: str | Path,
    target_fraud_rate: float = 0.01,
    test_size: float = 0.2,
    seed: int = 42,
    max_samples_per_client: int | None = None,
) -> list[dict]:
    """
    Stage 2: Base-rate balanced split.

    Same client assignment as Stage 1, but downsample the legitimate
    class per client so that all clients have an identical fraud rate.

    Parameters
    ----------
    data_root : str | Path
        Root data directory.
    target_fraud_rate : float
        Desired fraud rate for all clients (default 1.0%).
    test_size : float
        Fraction for test split (applied before resampling).
    seed : int
        Random seed.

    Returns
    -------
    list[dict]
        Same format as split_stage1_natural.
    """
    data_root = Path(data_root)
    dataset_names = ["ieee_cis", "feedzai_baf", "paysim"]
    clients = []
    rng = np.random.RandomState(seed)

    for name in dataset_names:
        data_dir = data_root / name
        df = harmonize_dataset(name, data_dir)

        # Split first, then resample train set
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=seed,
            stratify=df["label"],
        )

        # Resample train set to target fraud rate
        train_df = _resample_to_fraud_rate(train_df, target_fraud_rate, rng)

        # Subsample training data if it exceeds cap
        train_df = _stratified_subsample(train_df, max_samples_per_client, seed)

        # Normalize
        train_df, scaler = normalize_features(train_df)
        test_df, _ = normalize_features(test_df, scaler=scaler)

        fraud_rate = train_df["label"].mean()

        clients.append({
            "name": name,
            "train_df": train_df.reset_index(drop=True),
            "test_df": test_df.reset_index(drop=True),
            "train_scaler": scaler,
            "fraud_rate": fraud_rate,
        })

        print(f"  [Stage 2] Client '{name}': "
              f"train={len(train_df)}, test={len(test_df)}, "
              f"fraud_rate={fraud_rate:.4f}")

    return clients


def _resample_to_fraud_rate(
    df: pd.DataFrame,
    target_rate: float,
    rng: np.random.RandomState,
) -> pd.DataFrame:
    """
    Downsample legitimate transactions to achieve a target fraud rate.

    If the natural fraud rate is already below the target, we keep all
    fraud samples and downsample legitimates. If above, we keep all
    legitimate samples and downsample frauds (rare case).
    """
    fraud = df[df["label"] == 1]
    legit = df[df["label"] == 0]

    n_fraud = len(fraud)
    n_legit = len(legit)
    current_rate = n_fraud / len(df)

    if abs(current_rate - target_rate) < 0.001:
        return df  # Already close enough

    if current_rate < target_rate:
        # Need fewer legitimates: n_legit_new = n_fraud * (1 - target_rate) / target_rate
        n_legit_new = int(n_fraud * (1 - target_rate) / target_rate)
        n_legit_new = min(n_legit_new, n_legit)
        legit_sampled = legit.sample(n=n_legit_new, random_state=rng)
        return pd.concat([fraud, legit_sampled], ignore_index=True)
    else:
        # Need fewer frauds (unusual): n_fraud_new = n_legit * target_rate / (1 - target_rate)
        n_fraud_new = int(n_legit * target_rate / (1 - target_rate))
        n_fraud_new = min(n_fraud_new, n_fraud)
        fraud_sampled = fraud.sample(n=n_fraud_new, random_state=rng)
        return pd.concat([fraud_sampled, legit], ignore_index=True)


# ── Stage 3: Synthetic Dirichlet Control ─────────────────────────────────────

def split_stage3_dirichlet(
    data_root: str | Path,
    source_dataset: str = "ieee_cis",
    num_clients: int = 3,
    alpha: float = 0.5,
    test_size: float = 0.2,
    seed: int = 42,
) -> list[dict]:
    """
    Stage 3: Synthetic Dirichlet split of a single dataset.

    Partitions one dataset into N synthetic clients using a Dirichlet
    distribution on the label to create non-IID label skew (the standard
    FL paper setup).

    Parameters
    ----------
    data_root : str | Path
        Root data directory.
    source_dataset : str
        Which dataset to partition (default 'ieee_cis').
    num_clients : int
        Number of synthetic clients.
    alpha : float
        Dirichlet concentration parameter. Lower = more heterogeneous.
    test_size : float
        Fraction for test split.
    seed : int
        Random seed.

    Returns
    -------
    list[dict]
        Same format as split_stage1_natural.
    """
    data_root = Path(data_root)
    data_dir = data_root / source_dataset
    df = harmonize_dataset(source_dataset, data_dir)

    rng = np.random.RandomState(seed)

    # Split into per-label groups
    label_indices = {}
    for label in df["label"].unique():
        label_indices[label] = df.index[df["label"] == label].tolist()

    # Dirichlet split: for each label, distribute samples across clients
    client_indices = [[] for _ in range(num_clients)]

    for label, indices in label_indices.items():
        rng.shuffle(indices)
        # Draw proportions from Dirichlet
        proportions = rng.dirichlet([alpha] * num_clients)
        # Convert proportions to counts
        splits = (proportions * len(indices)).astype(int)
        # Distribute remainder to first clients
        remainder = len(indices) - splits.sum()
        for i in range(remainder):
            splits[i % num_clients] += 1

        # Assign indices to clients
        start = 0
        for client_id, count in enumerate(splits):
            client_indices[client_id].extend(indices[start:start + count])
            start += count

    # Build client data dicts
    clients = []
    for client_id in range(num_clients):
        client_df = df.loc[client_indices[client_id]].copy()

        train_df, test_df = train_test_split(
            client_df, test_size=test_size, random_state=seed,
            stratify=client_df["label"],
        )

        train_df, scaler = normalize_features(train_df)
        test_df, _ = normalize_features(test_df, scaler=scaler)

        fraud_rate = train_df["label"].mean()
        name = f"{source_dataset}_dirichlet_{client_id}"

        clients.append({
            "name": name,
            "train_df": train_df.reset_index(drop=True),
            "test_df": test_df.reset_index(drop=True),
            "train_scaler": scaler,
            "fraud_rate": fraud_rate,
        })

        print(f"  [Stage 3] Client '{name}': "
              f"train={len(train_df)}, test={len(test_df)}, "
              f"fraud_rate={fraud_rate:.4f}")

    return clients
