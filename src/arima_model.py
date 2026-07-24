"""
Step 3: ARIMA model (non-seasonal).

ARIMA(p, d, q) captures trend/autocorrelation but has no explicit seasonal term,
so it's a good baseline to compare against SARIMA below.

We pick (p, d, q) automatically via a small grid search over AIC, since that avoids
depending on the pmdarima package (which has finicky binary dependencies on Windows).
"""

import itertools
import warnings

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

P_RANGE = range(0, 4)
D_RANGE = range(0, 3)
Q_RANGE = range(0, 4)


def find_best_arima_order(train: pd.Series, p_range=P_RANGE, d_range=D_RANGE, q_range=Q_RANGE):
    """Grid search (p, d, q) combinations, return the order with the lowest AIC."""
    best_aic = float("inf")
    best_order = None

    for p, d, q in itertools.product(p_range, d_range, q_range):
        try:
            model = ARIMA(train, order=(p, d, q))
            fitted = model.fit()
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_order = (p, d, q)
        except Exception:
            continue

    return best_order, best_aic


def fit_arima(train: pd.Series, order=None):
    """Fit an ARIMA model. If order is None, search for the best one first."""
    if order is None:
        order, aic = find_best_arima_order(train)
        print(f"Best ARIMA order found: {order} (AIC={aic:.2f})")

    model = ARIMA(train, order=order)
    fitted = model.fit()
    return fitted, order


def forecast_arima(fitted_model, steps: int) -> pd.Series:
    """Return a forecasted series `steps` months ahead, indexed by future month-start dates."""
    forecast = fitted_model.get_forecast(steps=steps)
    return forecast.predicted_mean
