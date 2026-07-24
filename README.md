# Book Sales Forecasting (ARIMA & SARIMA)

Forecasts monthly book sales 3, 6, 9, and 12 months into the future using two
classical time-series models: **ARIMA** and **SARIMA**. Includes a dummy data
generator so the whole pipeline runs out of the box, both as a CLI script and as
a **FastAPI microservice** you can trigger over HTTP.

## Project structure

```
book_sales_forecast/
├── data/
│   ├── generate_dummy_data.py   # Step 1: creates synthetic monthly sales data
│   └── book_sales.csv           # generated output (created after step 1)
├── src/
│   ├── data_loader.py           # Step 2: load CSV -> pandas Series, train/test split
│   ├── arima_model.py           # Step 3: ARIMA order search, fit, forecast
│   ├── sarima_model.py          # Step 4: SARIMA order search, fit, forecast
│   ├── evaluation.py            # RMSE / MAE / MAPE helpers
│   └── pipeline.py              # shared backtest/forecast logic (used by main.py AND the API)
├── app/
│   ├── main.py                  # FastAPI app: HTTP endpoints that trigger the pipeline
│   └── schemas.py                # request/response models
├── outputs/                     # forecast CSVs + comparison plot land here
├── main.py                      # Step 5: runs the full pipeline end-to-end (CLI)
└── requirements.txt
```

## Setup

```bash
cd book_sales_forecast
python -m venv venv
venv\Scripts\activate        # on Windows
pip install -r requirements.txt
```

## Steps

### 1. Generate dummy data

```bash
python data/generate_dummy_data.py
```

Creates `data/book_sales.csv`: 7 years (84 months) of monthly book sales with an
upward trend, yearly seasonality (holiday peak in Nov/Dec, back-to-school bump in
Sep), and random noise/promo spikes. Replace this file with your real sales data
(same two columns: `date`, `sales`) to use your own data instead.

### 2. Load & preprocess

`src/data_loader.py` reads the CSV, parses dates, sets a monthly (`MS`) frequency
index, and fills any missing months. `train_test_split_series` holds out the last
12 months for backtesting.

### 3 & 4. Fit ARIMA and SARIMA

`src/arima_model.py` and `src/sarima_model.py` each run a small grid search over
model orders — `(p, d, q)` for ARIMA, `(p, d, q)(P, D, Q, 12)` for SARIMA — picking
whichever combination minimizes AIC. SARIMA additionally models the yearly (m=12)
seasonal pattern, which ARIMA cannot capture on its own.

### 5. Run the full pipeline

```bash
python main.py
```

This will:
1. Load the data and split off the last 12 months as a test set.
2. Fit ARIMA and SARIMA on the training set and print RMSE/MAE/MAPE against the
   held-out test months, so you can see which model fits this data better.
3. Refit both models on the **entire** history (using the orders found above) and
   forecast 3, 6, 9, and 12 months ahead.
4. Save results to `outputs/`:
   - `forecast_3_months.csv`, `forecast_6_months.csv`, `forecast_9_months.csv`,
     `forecast_12_months.csv` — side-by-side ARIMA vs SARIMA forecasts.
   - `forecast_comparison.png` — plot of historical sales plus both models'
     12-month forecast.

## Run as a FastAPI microservice

Instead of running the scripts by hand, start the service once and trigger the
pipeline over HTTP:

```bash
venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then open http://127.0.0.1:8000/docs for interactive Swagger docs, or call it directly:

| Method | Endpoint            | Description                                                                 |
|--------|---------------------|-------------------------------------------------------------------------------|
| GET    | `/health`            | Liveness check                                                               |
| POST   | `/data/generate`     | (Re)generates the dummy dataset at `data/book_sales.csv`                     |
| POST   | `/forecast/run`      | Backtests ARIMA/SARIMA, refits on full history, forecasts (default 3/6/9/12) |
| GET    | `/forecast/{horizon}`| Reads back a previously saved forecast, e.g. `/forecast/12`                  |
| GET    | `/forecast/plot`     | Returns `forecast_comparison.png` from the last `/forecast/run`              |

Example:

```bash
curl -X POST http://127.0.0.1:8000/data/generate
curl -X POST http://127.0.0.1:8000/forecast/run -H "Content-Type: application/json" -d "{}"
curl http://127.0.0.1:8000/forecast/12
```

`POST /forecast/run` accepts an optional body `{"horizons": [3, 6, 9, 12]}` to
customize which horizons are computed; it defaults to `[3, 6, 9, 12]` if omitted.
It writes the same `outputs/forecast_*_months.csv` and `forecast_comparison.png`
files as the CLI (`main.py`) — both entry points share the same logic in
`src/pipeline.py`, so results are identical either way.

Note: `/forecast/run` fits both models synchronously (grid search over model
orders), so the request can take a minute or two — this is a demo/local service,
not tuned for low-latency responses.

## Notes

- Order search ranges are intentionally modest (see the `*_RANGE` constants at the
  top of `arima_model.py` / `sarima_model.py`) to keep runtime reasonable. Widen
  them if you want a more exhaustive search on your own data.
- No `pmdarima` dependency — order selection is done manually via `statsmodels` to
  avoid its sometimes-fragile Windows installation.
- Swap in real data by replacing `data/book_sales.csv` with matching `date,sales`
  columns; no code changes needed as long as the data is monthly.
