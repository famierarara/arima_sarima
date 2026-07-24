"""
Step 5: End-to-end CLI pipeline.

1. Load the monthly book sales data (data/book_sales.csv).
2. Split into train/test (last 12 months held out) to backtest accuracy.
3. Fit ARIMA and SARIMA on the training set, evaluate against the held-out test set.
4. Refit both models on the FULL dataset and forecast 3, 6, 9, and 12 months ahead.
5. Save forecast tables (CSV) and a comparison plot (PNG) to outputs/.

Run:
    python main.py

See app/main.py for the equivalent as a FastAPI microservice you can trigger over HTTP.
"""

import os

import matplotlib
matplotlib.use("Agg")  # safe for headless/script execution
import matplotlib.pyplot as plt

from src.data_loader import load_book_sales, train_test_split_series
from src.evaluation import print_metrics
from src.metrics_log import log_backtest_metrics, plot_metrics_history, read_metrics_history
from src.pipeline import run_backtest, run_future_forecast

DATA_PATH = "data/book_sales.csv"
OUTPUT_DIR = "outputs"
TEST_SIZE = 12
HORIZONS = [3, 6, 9, 12]


def save_outputs(results: dict, full_series, arima_full_forecast, sarima_full_forecast):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for h, df in results.items():
        path = f"{OUTPUT_DIR}/forecast_{h}_months.csv"
        df.to_csv(path, index_label="date")
        print(f"Saved {h}-month forecast -> {path}")

    # Combined plot: history + both models' forecast
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
    plot_path = f"{OUTPUT_DIR}/forecast_comparison.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved comparison plot -> {plot_path}")


def main():
    series = load_book_sales(DATA_PATH)
    train, test = train_test_split_series(series, test_size=TEST_SIZE)

    print("\n=== Backtesting on last 12 held-out months ===")
    backtest_results = run_backtest(train, test)
    print_metrics(f"ARIMA{backtest_results['arima']['order']}", backtest_results["arima"]["metrics"])
    sarima = backtest_results["sarima"]
    print_metrics(f"SARIMA{sarima['order']}{sarima['seasonal_order']}", sarima["metrics"])

    log_backtest_metrics(backtest_results)
    history = read_metrics_history()
    trend_path = plot_metrics_history(history)
    print(f"Logged this run's metrics -> outputs/metrics_history.csv ({len(history)} rows total)")
    if trend_path:
        print(f"Updated MAPE trend plot -> {trend_path}")

    print("\n=== Refitting on full history and forecasting future months ===")
    results, arima_full_forecast, sarima_full_forecast = run_future_forecast(
        series,
        arima_order=backtest_results["arima"]["order"],
        sarima_order=backtest_results["sarima"]["order"],
        sarima_seasonal_order=backtest_results["sarima"]["seasonal_order"],
        horizons=HORIZONS,
    )

    save_outputs(results, series, arima_full_forecast, sarima_full_forecast)

    print("\n=== 12-month forecast preview ===")
    print(results[12])


if __name__ == "__main__":
    main()
