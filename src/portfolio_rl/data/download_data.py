"""
Download historical price data for a set of ETFs and save to data/raw/.

Run from project root:
    python -m portfolio_rl.data.download_data
"""

from pathlib import Path
from typing import List

import pandas as pd
import yfinance as yf


TICKERS: List[str] = ["SPY", "QQQ", "TLT", "GLD", "HYG", "IWM"]
START_DATE = "2005-01-01"
END_DATE = "2025-01-01"


def download_prices(
    tickers: List[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Download adjusted close prices for given tickers and date range.

    Returns a DataFrame with:
        index: DatetimeIndex
        columns: tickers
    """
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )

    # Use Adjusted Close for total-return style prices
    prices = data["Adj Close"].copy()
    prices = prices.dropna(how="all")
    prices = prices[tickers]  # ensure consistent column order
    return prices


def save_prices(prices: pd.DataFrame, path: Path) -> None:
    """
    Save prices to CSV at the given path, creating parent directories if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path, index=True)


def main() -> None:
    root = Path(".")  # assume run from project root
    raw_dir = root / "data" / "raw"
    out_path = raw_dir / "prices_etfs.csv"

    prices = download_prices(TICKERS, START_DATE, END_DATE)
    save_prices(prices, out_path)

    print(f"Saved prices for {len(TICKERS)} tickers to {out_path}")


if __name__ == "__main__":
    main()
