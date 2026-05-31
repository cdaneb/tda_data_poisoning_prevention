"""
Data loader for UNSW-NB15 and CICIDS2017 Payload-Byte datasets.

Loads the CSV files, extracts the 1500 payload byte features and labels,
and provides functions for subsampling if needed.

Column structure (1505 columns total):
  payload_byte_1 ... payload_byte_1500  — payload byte features (0-255)
  ttl, total_len, protocol, t_delta     — metadata (not used for TDA)
  label                                 — attack category / 'normal' / 'BENIGN'
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(r"C:\TDA\data")

PAYLOAD_COLUMNS = [f"payload_byte_{i}" for i in range(1, 1501)]
LABEL_COLUMN = "label"


def load_unsw(max_samples=None):
    """
    Load UNSW-NB15 Payload-Byte dataset.

    Returns:
        X: np.ndarray of shape (N, 1500), dtype=np.uint8 — payload bytes
        y: np.ndarray of shape (N,) — string labels (attack category or 'normal')
    """
    filepath = DATA_DIR / "Payload_data_UNSW.csv"
    print(f"Loading UNSW-NB15 from {filepath}...")

    cols_to_load = PAYLOAD_COLUMNS + [LABEL_COLUMN]
    df = pd.read_csv(filepath, usecols=cols_to_load)

    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"  Subsampled to {max_samples} rows")

    X = df[PAYLOAD_COLUMNS].values.astype(np.uint8)
    y = df[LABEL_COLUMN].values

    print(f"  Loaded: X.shape={X.shape}, y.shape={y.shape}")
    print(f"  Labels: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y


def load_cicids(max_samples=None):
    """
    Load CICIDS2017 Payload-Byte dataset.

    Returns:
        X: np.ndarray of shape (N, 1500), dtype=np.uint8 — payload bytes
        y: np.ndarray of shape (N,) — string labels (attack category or 'BENIGN')
    """
    filepath = DATA_DIR / "Payload_data_CICIDS2017.csv"
    print(f"Loading CICIDS2017 from {filepath}...")

    cols_to_load = PAYLOAD_COLUMNS + [LABEL_COLUMN]
    df = pd.read_csv(filepath, usecols=cols_to_load)

    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"  Subsampled to {max_samples} rows")

    X = df[PAYLOAD_COLUMNS].values.astype(np.uint8)
    y = df[LABEL_COLUMN].values

    print(f"  Loaded: X.shape={X.shape}, y.shape={y.shape}")
    print(f"  Labels: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y


if __name__ == "__main__":
    # Test loading — use a small subsample to verify it works
    print("=== Testing data loader ===\n")

    X_unsw, y_unsw = load_unsw(max_samples=1000)
    print(f"  UNSW value range: [{X_unsw.min()}, {X_unsw.max()}]")
    print(f"  UNSW dtype: {X_unsw.dtype}")
    print()

    X_cicids, y_cicids = load_cicids(max_samples=1000)
    print(f"  CICIDS value range: [{X_cicids.min()}, {X_cicids.max()}]")
    print(f"  CICIDS dtype: {X_cicids.dtype}")

    print("\n=== Data loader verification PASSED ===")
