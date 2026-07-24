"""
Step 2: Load and preprocess the book sales data.

Handles:
- Reading the CSV
- Parsing dates and setting a monthly ("MS") DatetimeIndex
- Basic sanity checks (missing months, missing values)
- Train/test split for backtesting model accuracy
"""

from pathlib import Path

import pandas as pd


def load_book_sales(csv_path: str = "data/book_sales.csv") -> pd.Series:
    """Load the book sales CSV into a pandas Series indexed by month-start date."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {csv_path}. Run `python data/generate_dummy_data.py` "
            "first (or place your own data at this path with 'date' and 'sales' columns)."
        )

    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date")
    df = df.set_index("date")

    series = df["sales"].asfreq("MS")

    if series.isna().any():
        n_missing = int(series.isna().sum())
        print(f"Warning: {n_missing} missing month(s) detected, filling via interpolation.")
        series = series.interpolate()

    return series


def train_test_split_series(series: pd.Series, test_size: int = 12):
    """Split a time series into train/test, keeping the last `test_size` points as test."""
    if test_size >= len(series):
        raise ValueError("test_size must be smaller than the length of the series")
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    return train, test


if __name__ == "__main__":
    s = load_book_sales()
    print(s.head())
    print("...")
    print(s.tail())
    print(f"\nTotal months: {len(s)}")
