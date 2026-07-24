"""
Step 1: Generate dummy monthly book sales data.

Creates a synthetic time series of monthly book sales (units sold) with:
- an upward trend (sales growing over the years)
- yearly seasonality (peaks around Nov/Dec holiday season and back-to-school in Sep)
- random noise

Run:
    python data/generate_dummy_data.py

Output:
    data/book_sales.csv  (columns: date, sales)
"""

import numpy as np
import pandas as pd

np.random.seed(42)

START_DATE = "2018-01-01"
NUM_MONTHS = 84  # 7 years of monthly data

OUTPUT_PATH = "data/book_sales.csv"


def generate_book_sales(start_date: str, num_months: int) -> pd.DataFrame:
    dates = pd.date_range(start=start_date, periods=num_months, freq="MS")

    t = np.arange(num_months)

    # Baseline level + gradual upward trend (e.g. growing readership)
    base_level = 1500
    trend = t * 6.0

    # Yearly seasonality: month-of-year effect (index 0=Jan ... 11=Dec)
    # Bigger multipliers around Sep (back-to-school) and Nov/Dec (holidays)
    monthly_effect = {
        1: -150,   # Jan
        2: -180,   # Feb
        3: -50,
        4: 0,
        5: 50,
        6: 100,
        7: 80,
        8: 150,
        9: 300,    # back-to-school bump
        10: 200,
        11: 450,   # holiday build-up
        12: 700,   # holiday peak
    }
    seasonality = np.array([monthly_effect[d.month] for d in dates])

    # Random noise
    noise = np.random.normal(loc=0, scale=80, size=num_months)

    # A few random promotional spikes to make it more realistic
    promo_spikes = np.zeros(num_months)
    promo_months = np.random.choice(num_months, size=max(1, num_months // 15), replace=False)
    promo_spikes[promo_months] = np.random.uniform(200, 500, size=len(promo_months))

    sales = base_level + trend + seasonality + noise + promo_spikes
    sales = np.round(np.clip(sales, a_min=100, a_max=None)).astype(int)

    return pd.DataFrame({"date": dates, "sales": sales})


if __name__ == "__main__":
    df = generate_book_sales(START_DATE, NUM_MONTHS)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(df)} monthly records.")
    print(f"Saved to: {OUTPUT_PATH}")
    print(df.head())
    print("...")
    print(df.tail())
