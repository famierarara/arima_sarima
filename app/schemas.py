"""Pydantic request/response models for the forecasting API."""

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel


class GenerateDataResponse(BaseModel):
    rows: int
    start_date: str
    end_date: str
    path: str


class ForecastRequest(BaseModel):
    # Which horizons (in months) to return forecasts for. Defaults to [3, 6, 9, 12].
    horizons: Optional[List[int]] = None


class ModelMetrics(BaseModel):
    rmse: float
    mae: float
    mape: float


class ArimaResult(BaseModel):
    order: Tuple[int, int, int]
    metrics: ModelMetrics


class SarimaResult(BaseModel):
    order: Tuple[int, int, int]
    seasonal_order: Tuple[int, int, int, int]
    metrics: ModelMetrics


class ForecastPoint(BaseModel):
    date: str
    arima_forecast: float
    sarima_forecast: float


class ForecastRunResponse(BaseModel):
    arima: ArimaResult
    sarima: SarimaResult
    forecasts: Dict[int, List[ForecastPoint]]
    plot_path: str
