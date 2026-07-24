"""
Append-only log of backtest accuracy (RMSE/MAE/MAPE) for every pipeline run.

A single run's metrics only tell you "was this forecast accurate right now?".
Logging every run lets you answer "is accuracy stable, improving, or drifting
over time?" — the difference between checking a model once and monitoring it.
"""

import csv
import os
from datetime import datetime, timezone

import pandas as pd

METRICS_LOG_PATH = "outputs/metrics_history.csv"
METRICS_PLOT_PATH = "outputs/metrics_trend.png"

FIELDNAMES = ["timestamp", "model", "order", "seasonal_order", "rmse", "mae", "mape"]


def log_backtest_metrics(backtest_results: dict, path: str = METRICS_LOG_PATH) -> None:
    """Append one row per model (arima, sarima) from a run_backtest() result."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    file_exists = os.path.exists(path)
    timestamp = datetime.now(timezone.utc).isoformat()

    arima = backtest_results["arima"]
    sarima = backtest_results["sarima"]
    rows = [
        {
            "timestamp": timestamp,
            "model": "arima",
            "order": str(arima["order"]),
            "seasonal_order": "",
            "rmse": round(arima["metrics"]["rmse"], 4),
            "mae": round(arima["metrics"]["mae"], 4),
            "mape": round(arima["metrics"]["mape"], 4),
        },
        {
            "timestamp": timestamp,
            "model": "sarima",
            "order": str(sarima["order"]),
            "seasonal_order": str(sarima["seasonal_order"]),
            "rmse": round(sarima["metrics"]["rmse"], 4),
            "mae": round(sarima["metrics"]["mae"], 4),
            "mape": round(sarima["metrics"]["mape"], 4),
        },
    ]

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def read_metrics_history(path: str = METRICS_LOG_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=FIELDNAMES)
    return pd.read_csv(path)


def plot_metrics_history(df: pd.DataFrame, path: str = METRICS_PLOT_PATH):
    """Plot MAPE per model over time, so you can see whether accuracy is drifting."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df.empty:
        return None

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    plt.figure(figsize=(10, 5))
    for model, group in df.groupby("model"):
        group = group.sort_values("timestamp")
        plt.plot(group["timestamp"], group["mape"], marker="o", label=model.upper())
    plt.title("Backtest MAPE over time (lower = more accurate)")
    plt.xlabel("Run timestamp")
    plt.ylabel("MAPE (%)")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    return path
