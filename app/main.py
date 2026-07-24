"""
FastAPI microservice wrapping the ARIMA/SARIMA book sales forecasting pipeline.

Run:
    uvicorn app.main:app --reload

Then trigger the pipeline over HTTP instead of running scripts by hand:
    POST /data/generate   -> (re)generate dummy data
    POST /forecast/run    -> backtest + forecast, saves outputs/, returns JSON
    GET  /forecast/{h}    -> read back a previously saved horizon's forecast
    GET  /forecast/plot   -> the comparison PNG from the last /forecast/run
    GET  /health
"""

import os

import matplotlib

matplotlib.use("Agg")  # headless rendering, required before pyplot is imported
import matplotlib.pyplot as plt
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from data.generate_dummy_data import NUM_MONTHS, START_DATE, generate_book_sales
from src.data_loader import load_book_sales, train_test_split_series
from src.pipeline import run_backtest, run_future_forecast

from app.schemas import ForecastRequest, ForecastRunResponse, GenerateDataResponse

DATA_PATH = "data/book_sales.csv"
OUTPUT_DIR = "outputs"
TEST_SIZE = 12
DEFAULT_HORIZONS = [3, 6, 9, 12]

app = FastAPI(
    title="Book Sales Forecasting Service",
    description="Trigger ARIMA/SARIMA book sales forecasts (3/6/9/12 months ahead) over HTTP.",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/data/generate", response_model=GenerateDataResponse)
def generate_data():
    """Regenerate the dummy monthly book sales dataset."""
    os.makedirs("data", exist_ok=True)
    df = generate_book_sales(START_DATE, NUM_MONTHS)
    df.to_csv(DATA_PATH, index=False)

    return GenerateDataResponse(
        rows=len(df),
        start_date=str(df["date"].min().date()),
        end_date=str(df["date"].max().date()),
        path=DATA_PATH,
    )


@app.post("/forecast/run", response_model=ForecastRunResponse)
def run_forecast(request: ForecastRequest = ForecastRequest()):
    """Backtest ARIMA/SARIMA, then refit on full history and forecast the requested horizons.

    Saves forecast_{h}_months.csv and forecast_comparison.png to outputs/, same as main.py.
    """
    if not os.path.exists(DATA_PATH):
        raise HTTPException(
            status_code=400,
            detail=f"{DATA_PATH} not found. Call POST /data/generate first.",
        )

    horizons = sorted(request.horizons) if request.horizons else DEFAULT_HORIZONS

    series = load_book_sales(DATA_PATH)
    train, test = train_test_split_series(series, test_size=TEST_SIZE)

    backtest_results = run_backtest(train, test)

    results, arima_full_forecast, sarima_full_forecast = run_future_forecast(
        series,
        arima_order=backtest_results["arima"]["order"],
        sarima_order=backtest_results["sarima"]["order"],
        sarima_seasonal_order=backtest_results["sarima"]["seasonal_order"],
        horizons=horizons,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    forecasts_by_horizon = {}
    for h, df in results.items():
        path = f"{OUTPUT_DIR}/forecast_{h}_months.csv"
        df.to_csv(path, index_label="date")
        forecasts_by_horizon[h] = [
            {
                "date": str(idx.date()),
                "arima_forecast": row.arima_forecast,
                "sarima_forecast": row.sarima_forecast,
            }
            for idx, row in df.iterrows()
        ]

    plot_path = f"{OUTPUT_DIR}/forecast_comparison.png"
    _save_comparison_plot(series, arima_full_forecast, sarima_full_forecast, plot_path)

    return ForecastRunResponse(
        arima={
            "order": backtest_results["arima"]["order"],
            "metrics": backtest_results["arima"]["metrics"],
        },
        sarima={
            "order": backtest_results["sarima"]["order"],
            "seasonal_order": backtest_results["sarima"]["seasonal_order"],
            "metrics": backtest_results["sarima"]["metrics"],
        },
        forecasts=forecasts_by_horizon,
        plot_path=plot_path,
    )


@app.get("/forecast/plot")
def get_plot():
    """Return the historical-vs-forecast comparison plot from the last POST /forecast/run."""
    path = f"{OUTPUT_DIR}/forecast_comparison.png"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No plot available yet. Call POST /forecast/run first.")
    return FileResponse(path, media_type="image/png")


@app.get("/forecast/{horizon}")
def get_forecast(horizon: int):
    """Read back a previously saved forecast (from the last POST /forecast/run)."""
    path = f"{OUTPUT_DIR}/forecast_{horizon}_months.csv"
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail=f"No saved forecast for horizon={horizon} months. Call POST /forecast/run first.",
        )
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def _save_comparison_plot(full_series, arima_full_forecast, sarima_full_forecast, plot_path):
    plt.figure(figsize=(12, 6))
    plt.plot(full_series.index, full_series.values, label="Historical sales", color="black")
    plt.plot(arima_full_forecast.index, arima_full_forecast.values, label="ARIMA forecast", linestyle="--")
    plt.plot(sarima_full_forecast.index, sarima_full_forecast.values, label="SARIMA forecast", linestyle="--")
    plt.axvline(full_series.index[-1], color="gray", linestyle=":", linewidth=1)
    plt.title("Book Sales Forecast: ARIMA vs SARIMA")
    plt.xlabel("Date")
    plt.ylabel("Units sold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
