"""
Shared pipeline logic: backtesting and future forecasting.

Used by both the CLI (main.py) and the FastAPI service (app/main.py) so the two
entry points can't drift out of sync.
"""

import pandas as pd

from src.arima_model import fit_arima, forecast_arima
from src.evaluation import evaluate_forecast
from src.sarima_model import fit_sarima, forecast_sarima


def run_backtest(train: pd.Series, test: pd.Series) -> dict:
    """Fit ARIMA and SARIMA on `train`, evaluate against `test`. Returns orders + metrics."""
    arima_fitted, arima_order = fit_arima(train)
    arima_pred = forecast_arima(arima_fitted, steps=len(test))
    arima_pred.index = test.index
    arima_metrics = evaluate_forecast(test, arima_pred)

    sarima_fitted, sarima_order, sarima_seasonal_order = fit_sarima(train)
    sarima_pred = forecast_sarima(sarima_fitted, steps=len(test))
    sarima_pred.index = test.index
    sarima_metrics = evaluate_forecast(test, sarima_pred)

    return {
        "arima": {"order": arima_order, "pred": arima_pred, "metrics": arima_metrics},
        "sarima": {
            "order": sarima_order,
            "seasonal_order": sarima_seasonal_order,
            "pred": sarima_pred,
            "metrics": sarima_metrics,
        },
    }


def run_future_forecast(
    full_series: pd.Series,
    arima_order,
    sarima_order,
    sarima_seasonal_order,
    horizons: list,
):
    """Refit both models on the full history and forecast `max(horizons)` months ahead.

    Returns (results, arima_full_forecast, sarima_full_forecast) where `results` is a
    dict {horizon: DataFrame[arima_forecast, sarima_forecast]}.
    """
    arima_fitted, _ = fit_arima(full_series, order=arima_order)
    sarima_fitted, _, _ = fit_sarima(full_series, order=sarima_order, seasonal_order=sarima_seasonal_order)

    max_horizon = max(horizons)
    arima_full_forecast = forecast_arima(arima_fitted, steps=max_horizon)
    sarima_full_forecast = forecast_sarima(sarima_fitted, steps=max_horizon)

    future_index = pd.date_range(
        start=full_series.index[-1] + pd.DateOffset(months=1), periods=max_horizon, freq="MS"
    )
    arima_full_forecast.index = future_index
    sarima_full_forecast.index = future_index

    results = {}
    for h in horizons:
        results[h] = pd.DataFrame(
            {
                "arima_forecast": arima_full_forecast.iloc[:h].round(0),
                "sarima_forecast": sarima_full_forecast.iloc[:h].round(0),
            }
        )

    return results, arima_full_forecast, sarima_full_forecast
