"""
Schema harmonization pipeline for fraud detection datasets.

Maps raw columns from IEEE-CIS, Feedzai BAF, and PaySim into a
Minimal Shared Schema with 7 harmonized features:

    log_amount           – log1p-scaled monetary amount
    cyclical_time_hour   – sin/cos encoded hour-of-day
    cyclical_time_day    – sin/cos encoded day-of-week
    tx_velocity_short    – short-window transaction velocity
    tx_velocity_long     – long-window transaction velocity
    account_age_days     – account/entity age in days
    label                – binary fraud label (0 or 1)

All normalization is done per-client (never cross-client) to prevent
data leakage in federated settings.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler


# ── Shared Schema Column Names ──────────────────────────────────────────────
HARMONIZED_COLUMNS = [
    "log_amount",
    "cyclical_time_hour_sin",
    "cyclical_time_hour_cos",
    "cyclical_time_day_sin",
    "cyclical_time_day_cos",
    "tx_velocity_short",
    "tx_velocity_long",
    "account_age_days",
    "label",
]

FEATURE_COLUMNS = [c for c in HARMONIZED_COLUMNS if c != "label"]


# ── IEEE-CIS Fraud Detection ────────────────────────────────────────────────

def prepare_ieee_cis(data_dir: str | Path) -> pd.DataFrame:
    """
    Harmonize IEEE-CIS Fraud Detection dataset.

    Expected files in data_dir:
        - train_transaction.csv (or transaction data with isFraud column)

    Column mappings:
        TransactionAmt → log_amount
        TransactionDT  → cyclical_time_hour, cyclical_time_day
        C1-C6 (mean)   → tx_velocity_short
        C7-C14 (mean)  → tx_velocity_long
        D1              → account_age_days
        isFraud         → label
    """
    data_dir = Path(data_dir)

    # Try common file names
    for fname in ["train_transaction.csv", "transaction.csv"]:
        fpath = data_dir / fname
        if fpath.exists():
            break
    else:
        raise FileNotFoundError(
            f"No transaction CSV found in {data_dir}. "
            f"Expected 'train_transaction.csv' or 'transaction.csv'."
        )

    # Load only the columns we need to save memory
    needed_cols = [
        "TransactionAmt", "TransactionDT", "isFraud",
        "C1", "C2", "C3", "C4", "C5", "C6",
        "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14",
        "D1",
    ]

    df = pd.read_csv(fpath, usecols=lambda c: c in needed_cols)

    result = pd.DataFrame()

    # Monetary: log1p(TransactionAmt)
    result["log_amount"] = np.log1p(df["TransactionAmt"].fillna(0).clip(lower=0))

    # Temporal: TransactionDT is delta-seconds from a reference time
    seconds_in_day = 86400
    hour_of_day = (df["TransactionDT"] % seconds_in_day) / 3600.0  # 0-24
    day_of_week = (df["TransactionDT"] / seconds_in_day) % 7        # 0-7

    result["cyclical_time_hour_sin"] = np.sin(2 * np.pi * hour_of_day / 24)
    result["cyclical_time_hour_cos"] = np.cos(2 * np.pi * hour_of_day / 24)
    result["cyclical_time_day_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    result["cyclical_time_day_cos"] = np.cos(2 * np.pi * day_of_week / 7)

    # Velocity: C1-C6 average for short window, C7-C14 for long window
    short_cols = [f"C{i}" for i in range(1, 7) if f"C{i}" in df.columns]
    long_cols = [f"C{i}" for i in range(7, 15) if f"C{i}" in df.columns]

    result["tx_velocity_short"] = df[short_cols].mean(axis=1).fillna(0)
    result["tx_velocity_long"] = df[long_cols].mean(axis=1).fillna(0)

    # Account age: D1 is in days
    result["account_age_days"] = df["D1"].fillna(df["D1"].median())

    # Label
    result["label"] = df["isFraud"].astype(int)

    return result


# ── Feedzai Bank Account Fraud (BAF) ────────────────────────────────────────

def prepare_feedzai_baf(data_dir: str | Path) -> pd.DataFrame:
    """
    Harmonize Feedzai BAF dataset (NeurIPS 2022).

    Expected files in data_dir:
        - Base.csv (or any of the variant CSVs)

    Column mappings:
        intended_balcon_amount / proposed_credit_limit → log_amount
        session_length_in_minutes → cyclical_time_hour
        month                     → cyclical_time_day
        velocity_6h               → tx_velocity_short
        velocity_24h              → tx_velocity_long
        bank_months_count         → account_age_days
        fraud_bool                → label
    """
    data_dir = Path(data_dir)

    # BAF dataset has multiple variants; prefer Base.csv
    for fname in ["Base.csv", "base.csv", "Variant I.csv", "variant_i.csv"]: # Base.csv used for simulation
        fpath = data_dir / fname
        if fpath.exists():
            break
    else:
        # Try any CSV in the directory
        csvs = list(data_dir.glob("*.csv"))
        if csvs:
            fpath = csvs[0]
        else:
            raise FileNotFoundError(
                f"No CSV files found in {data_dir}. "
                f"Expected BAF dataset files."
            )

    df = pd.read_csv(fpath)

    result = pd.DataFrame()

    # Monetary: use intended_balcon_amount or proposed_credit_limit
    amount_col = None
    for col in ["intended_balcon_amount", "proposed_credit_limit"]:
        if col in df.columns:
            amount_col = col
            break

    if amount_col:
        result["log_amount"] = np.log1p(df[amount_col].fillna(0).clip(lower=0))
    else:
        result["log_amount"] = 0.0

    # Temporal: session_length_in_minutes → approximate hour-of-day
    if "session_length_in_minutes" in df.columns:
        # Normalize session length to a 0-24 range for cyclical encoding
        session_hours = df["session_length_in_minutes"].fillna(0) / 60.0
        result["cyclical_time_hour_sin"] = np.sin(2 * np.pi * session_hours / 24)
        result["cyclical_time_hour_cos"] = np.cos(2 * np.pi * session_hours / 24)
    else:
        result["cyclical_time_hour_sin"] = 0.0
        result["cyclical_time_hour_cos"] = 0.0

    # Day-of-week from month column (approximate weekly cycle)
    if "month" in df.columns:
        result["cyclical_time_day_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        result["cyclical_time_day_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    else:
        result["cyclical_time_day_sin"] = 0.0
        result["cyclical_time_day_cos"] = 0.0

    # Velocity features
    result["tx_velocity_short"] = df.get("velocity_6h", pd.Series(0.0, index=df.index)).fillna(0)
    result["tx_velocity_long"] = df.get("velocity_24h", pd.Series(0.0, index=df.index)).fillna(0)

    # Account age: bank_months_count → days
    if "bank_months_count" in df.columns:
        result["account_age_days"] = df["bank_months_count"].fillna(0) * 30.0
    elif "customer_age" in df.columns:
        # Fallback: use customer_age as a proxy (in years → days)
        result["account_age_days"] = df["customer_age"].fillna(0) * 365.0
    else:
        result["account_age_days"] = 0.0

    # Label
    result["label"] = df["fraud_bool"].astype(int)

    return result


# ── PaySim Synthetic Mobile Money ────────────────────────────────────────────

def prepare_paysim(data_dir: str | Path) -> pd.DataFrame:
    """
    Harmonize PaySim dataset.

    Expected files in data_dir:
        - PS_20174392719_1491204439457_log.csv (or similar)

    Column mappings:
        amount    → log_amount
        step      → cyclical_time_hour (step = 1 hour), cyclical_time_day
        derived   → tx_velocity_short (rolling 6-step count)
        derived   → tx_velocity_long (rolling 24-step count)
        derived   → account_age_days (from step)
        isFraud   → label
    """
    data_dir = Path(data_dir)

    # PaySim file typically has a long name
    csvs = list(data_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. "
            f"Expected PaySim dataset CSV."
        )
    fpath = csvs[0]

    df = pd.read_csv(fpath)

    result = pd.DataFrame()

    # Monetary
    result["log_amount"] = np.log1p(df["amount"].fillna(0).clip(lower=0))

    # Temporal: step represents hours (744 steps = 30 days)
    hour_of_day = df["step"] % 24
    day_of_week = (df["step"] / 24) % 7

    result["cyclical_time_hour_sin"] = np.sin(2 * np.pi * hour_of_day / 24)
    result["cyclical_time_hour_cos"] = np.cos(2 * np.pi * hour_of_day / 24)
    result["cyclical_time_day_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    result["cyclical_time_day_cos"] = np.cos(2 * np.pi * day_of_week / 7)

    # Velocity: rolling transaction counts per origin account
    # Sort by nameOrig and step for rolling window
    df_sorted = df.sort_values(["nameOrig", "step"])

    # Short window: count transactions per account in last 6 steps
    df_sorted["_tx_count"] = 1
    velocity_short = (
        df_sorted.groupby("nameOrig")["_tx_count"]
        .rolling(window=6, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    # Long window: 24-step rolling count
    velocity_long = (
        df_sorted.groupby("nameOrig")["_tx_count"]
        .rolling(window=24, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    result["tx_velocity_short"] = velocity_short.reindex(df.index).fillna(1)
    result["tx_velocity_long"] = velocity_long.reindex(df.index).fillna(1)

    # Account age: approximate from step (each step = 1 hour → days)
    # Use first occurrence of each nameOrig as "account creation"
    first_seen = df.groupby("nameOrig")["step"].transform("min")
    result["account_age_days"] = (df["step"] - first_seen) / 24.0

    # Label
    result["label"] = df["isFraud"].astype(int)

    return result


# ── Unified Harmonization Entry Point ────────────────────────────────────────

PREPARERS = {
    "ieee_cis": prepare_ieee_cis,
    "feedzai_baf": prepare_feedzai_baf,
    "paysim": prepare_paysim,
}


def harmonize_dataset(name: str, data_dir: str | Path, use_cache: bool = True) -> pd.DataFrame:
    """
    Harmonize a raw dataset into the shared schema.

    Parameters
    ----------
    name : str
        Dataset name: 'ieee_cis', 'feedzai_baf', or 'paysim'.
    data_dir : str | Path
        Path to the dataset's raw CSV directory.
    use_cache: bool
        If True, attempts to load from/save to 'harmonized.pkl' to save time.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns matching HARMONIZED_COLUMNS.
    """
    if name not in PREPARERS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(PREPARERS.keys())}")

    data_dir = Path(data_dir)
    cache_path = data_dir / "harmonized.pkl"

    # 1. Try to load from cache
    if use_cache and cache_path.exists():
        print(f"  [Cache] Loading pre-harmonized data for {name} from {cache_path}")
        return pd.read_pickle(cache_path)

    print(f"  [Harmonize] Processing raw data for {name}. This may take a while...")
    df = PREPARERS[name](data_dir)

    # Validate output schema
    missing = set(HARMONIZED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Harmonizer for '{name}' missing columns: {missing}")

    # Ensure column order
    df = df[HARMONIZED_COLUMNS].copy()

    # 2. Save to cache for next time
    if use_cache:
        print(f"  [Cache] Saving harmonized data to {cache_path}")
        df.to_pickle(cache_path)

    return df


def normalize_features(df: pd.DataFrame, scaler: RobustScaler | None = None):
    """
    Apply RobustScaler to continuous features (per-client).

    Parameters
    ----------
    df : pd.DataFrame
        Harmonized DataFrame.
    scaler : RobustScaler | None
        Pre-fit scaler for test data. If None, fits a new scaler (training).

    Returns
    -------
    tuple[pd.DataFrame, RobustScaler]
        Scaled DataFrame and fitted scaler.
    """
    continuous_cols = ["log_amount", "tx_velocity_short", "tx_velocity_long", "account_age_days"]
    df = df.copy()

    if scaler is None:
        scaler = RobustScaler()
        df[continuous_cols] = scaler.fit_transform(df[continuous_cols])
    else:
        df[continuous_cols] = scaler.transform(df[continuous_cols])

    return df, scaler


if __name__ == "__main__":
    # Allow running this file directly to pre-process and cache all datasets
    # Resolves to the 'data' directory at the root of the project
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "data"
    
    print(f"Starting manual pre-processing/caching for data in: {data_root}")
    
    for dataset_name in PREPARERS.keys():
        dataset_dir = data_root / dataset_name
        if dataset_dir.exists():
            print(f"\n--- {dataset_name} ---")
            # This will process and save the cache, or instantly load if it already exists
            harmonize_dataset(dataset_name, dataset_dir, use_cache=True)
        else:
            print(f"\n--- {dataset_name} ---")
            print(f"Directory not found: {dataset_dir}. Skipping.")
            
    print("\nAll available datasets have been processed!")
