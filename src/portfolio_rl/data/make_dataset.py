"""
Create processed datasets (returns + features) from raw price data.

Run from project root:
    python -m portfolio_rl.data.make_dataset
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


LOOKBACK_DAYS = 20  # number of past days used as features
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15  # test_frac = 1 - train_frac - val_frac


def load_prices(path: Path) -> pd.DataFrame:
    """
    Load price data from CSV.

    Assumes:
        index: date
        columns: tickers
    """
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    prices = prices.sort_index()
    prices = prices.dropna(how="any")  # drop rows with missing prices
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily simple returns from adjusted close prices.
    """
    returns = prices.pct_change().dropna(how="any")
    return returns


def build_features_from_returns(
    returns: pd.DataFrame,
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build feature and target arrays from returns.

    Features at row t contain the past `lookback` days of returns for each asset.
    Targets (returns_out) at row t are the returns for "today".

    Output shapes:
        features: (T, lookback * N_assets)
        returns_out: (T, N_assets)
    """
    returns_arr = returns.to_numpy(dtype=np.float32)
    T_full, n_assets = returns_arr.shape

    if T_full <= lookback:
        raise ValueError(
            f"Not enough data ({T_full} days) for lookback={lookback}"
        )

    T = T_full - lookback
    features = np.zeros((T, lookback * n_assets), dtype=np.float32)
    returns_out = np.zeros((T, n_assets), dtype=np.float32)

    for i in range(T):
        window = returns_arr[i: i + lookback]  # shape (lookback, n_assets)
        features[i] = window.reshape(-1)
        returns_out[i] = returns_arr[i + lookback]

    return features, returns_out


def split_time_series(
    features: np.ndarray,
    returns_arr: np.ndarray,
    train_frac: float,
    val_frac: float,
):
    """
    Split features and returns into train/val/test by time, no shuffling.
    """
    T = features.shape[0]
    train_end = int(T * train_frac)
    val_end = int(T * (train_frac + val_frac))

    X_train = features[:train_end]
    Y_train = returns_arr[:train_end]

    X_val = features[train_end:val_end]
    Y_val = returns_arr[train_end:val_end]

    X_test = features[val_end:]
    Y_test = returns_arr[val_end:]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def normalize_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
):
    """
    Standardize features using training mean and std.

    Returns:
        X_train_norm, X_val_norm, X_test_norm, mean, std
    """
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0

    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    X_test_norm = (X_test - mean) / std

    return X_train_norm, X_val_norm, X_test_norm, mean, std


def main() -> None:
    root = Path(".")
    raw_path = root / "data" / "raw" / "prices_etfs.csv"
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    prices = load_prices(raw_path)
    returns = compute_returns(prices)

    features, returns_arr = build_features_from_returns(
        returns, LOOKBACK_DAYS
    )

    (
        X_train,
        Y_train,
        X_val,
        Y_val,
        X_test,
        Y_test,
    ) = split_time_series(
        features, returns_arr, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC
    )

    X_train, X_val, X_test, mean, std = normalize_features(
        X_train, X_val, X_test
    )

    # Save processed arrays
    np.save(processed_dir / "features_train.npy", X_train)
    np.save(processed_dir / "features_val.npy", X_val)
    np.save(processed_dir / "features_test.npy", X_test)

    np.save(processed_dir / "returns_train.npy", Y_train)
    np.save(processed_dir / "returns_val.npy", Y_val)
    np.save(processed_dir / "returns_test.npy", Y_test)

    # Save normalization stats (for reproducibility / future use)
    np.save(processed_dir / "features_mean.npy", mean)
    np.save(processed_dir / "features_std.npy", std)

    print(f"Saved processed datasets to {processed_dir}")


if __name__ == "__main__":
    main()
