"""
Shared evaluation utilities for comparing forecasts against held-out actuals.
"""

import numpy as np
import pandas as pd


def evaluate_forecast(actual: pd.Series, predicted: pd.Series) -> dict:
    """Compute RMSE, MAE, and MAPE between actual and predicted series (aligned by index)."""
    actual, predicted = actual.align(predicted, join="inner")

    errors = actual - predicted
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    mape = float(np.mean(np.abs(errors / actual))) * 100

    return {"rmse": rmse, "mae": mae, "mape": mape}


def print_metrics(name: str, metrics: dict) -> None:
    print(f"{name}: RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}  MAPE={metrics['mape']:.2f}%")
