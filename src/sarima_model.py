"""
Step 4: SARIMA model (seasonal ARIMA).

SARIMA(p, d, q)(P, D, Q, m) extends ARIMA with a seasonal component, which matters
here because book sales have a clear yearly (m=12) pattern (holiday peak, back-to-
school bump, etc).

As with ARIMA, we grid search over AIC to pick orders automatically instead of
depending on pmdarima. The seasonal grid is kept small since SARIMAX fits are
slower than plain ARIMA.
"""

import itertools
import warnings

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

SEASONAL_PERIOD = 12

P_RANGE = range(0, 3)
D_RANGE = range(0, 2)
Q_RANGE = range(0, 3)

SP_RANGE = range(0, 2)
SD_RANGE = range(0, 2)
SQ_RANGE = range(0, 2)


def find_best_sarima_order(
    train: pd.Series,
    m: int = SEASONAL_PERIOD,
    p_range=P_RANGE,
    d_range=D_RANGE,
    q_range=Q_RANGE,
    sp_range=SP_RANGE,
    sd_range=SD_RANGE,
    sq_range=SQ_RANGE,
):
    """Grid search (p,d,q)(P,D,Q,m) combinations, return the combo with lowest AIC."""
    best_aic = float("inf")
    best_order = None
    best_seasonal_order = None

    for p, d, q in itertools.product(p_range, d_range, q_range):
        for P, D, Q in itertools.product(sp_range, sd_range, sq_range):
            if p == 0 and d == 0 and q == 0 and P == 0 and D == 0 and Q == 0:
                continue
            try:
                model = SARIMAX(
                    train,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, m),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fitted = model.fit(disp=False)
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, d, q)
                    best_seasonal_order = (P, D, Q, m)
            except Exception:
                continue

    return best_order, best_seasonal_order, best_aic


def fit_sarima(train: pd.Series, order=None, seasonal_order=None):
    """Fit a SARIMAX model. If order/seasonal_order are None, search for the best combo first."""
    if order is None or seasonal_order is None:
        order, seasonal_order, aic = find_best_sarima_order(train)
        print(f"Best SARIMA order found: {order}{seasonal_order} (AIC={aic:.2f})")

    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    return fitted, order, seasonal_order


def forecast_sarima(fitted_model, steps: int) -> pd.Series:
    """Return a forecasted series `steps` months ahead, indexed by future month-start dates."""
    forecast = fitted_model.get_forecast(steps=steps)
    return forecast.predicted_mean
