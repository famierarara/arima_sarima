# How the forecast numbers are actually produced

This walks through every transformation between raw data and the final forecasted
sales number, in the order it happens in the code. It's meant to explain *why* the
numbers come out the way they do (e.g. why ARIMA's forecast is a near-flat line
around 2200 while SARIMA's dips to ~1830 then climbs to ~2800), not just which
function to call.

Code references point to `book_sales_forecast/` files.

---

## Explain it like I'm 5

Imagine you have a little book stand, and every month you count how many books you
sold, going back 7 years.

- **Trend**: every year you get a little more well-known, so slowly, slowly, you
  sell a few more books than last year. Not a big jump — just a gentle climb.
- **Seasonality**: every single December, without fail, you sell a LOT of books,
  because people buy them as gifts. Every January, you sell way fewer, because
  everyone already spent their money in December. This happens every year, like
  clockwork — summer, winter, summer, winter, forever.
- **Noise**: sometimes it just rains extra hard one random Tuesday, or a bus full of
  tourists randomly walks by, and that month is a little higher or lower than
  normal for no real reason. Nobody could have guessed it ahead of time.

Now imagine two robot friends who try to guess how many books you'll sell next month.

- **Robot ARIMA** only remembers the last few months, and remembers "was I guessing
  too high or too low lately?" so it can correct itself a bit. But Robot ARIMA
  doesn't own a calendar — it has no idea December is special. It just sees "sales
  have been going up a little" and guesses "a little more," over and over, forever.
  That's why its guess for next December looks the same as its guess for next June:
  it simply doesn't know December is different.
- **Robot SARIMA** remembers the same things Robot ARIMA does, *plus* it owns a
  calendar. It also checks "what did I sell this exact same month last year?" So
  when it gets to December, it remembers "oh, last December was huge, and the
  December before that was huge too" — and guesses a big number for the upcoming
  December, and a small number for the upcoming January. That's why Robot SARIMA's
  guesses go up and down through the year instead of just climbing in a straight line.

How do we pick which robot brain to use (how many months back it should remember)?
We try out a bunch of different robot brains — one that remembers 1 month, one that
remembers 2, one that also double-checks last year, and so on — and we give each one
a **score** for how good and how simple it is. We keep whichever brain got the best
score. That's the "grid search" and "AIC" bit.

But before trusting a robot's guess about the *real* future (which we can't check
yet!), we play a trick on it: we cover up the last 12 months with our hand, ask the
robot to guess those months as if it didn't already know the answer, then peek and
see how close it got. That's **backtesting** — like a practice quiz before the real
test. Whichever robot did better on the practice quiz (Robot SARIMA, it turns out) is
the one we trust more for guessing the real future.

Once we know which robot is better, we let it look at *everything* we know so far —
all 7 years — and ask: "okay, now really guess the next 3, 6, 9, and 12 months." That
final guess is the forecast number you see in the results.

---

## 0. Plain-English glossary (start here if any term below is new to you)

Skip this if you already know what ARIMA/AIC/RMSE mean — it's the same concepts as
the rest of the doc, just without the notation.

- **Time series** — just a list of numbers over time. "Sales each month for 7 years"
  is a time series.
- **Trend** — the big-picture direction, ignoring the small ups and downs: is it
  generally climbing, falling, or flat over years? Like noticing your weight has
  gone up 5 lbs over a year, even though it wiggles up and down day to day.
- **Seasonality** — a pattern that repeats on a fixed schedule. Ice cream sales spike
  every summer; book sales spike every December. It's called "seasonal" even if the
  repeating cycle isn't literally the four seasons — here it repeats every 12 months.
- **Noise** — the random wiggle left over that doesn't come from trend or
  seasonality. No pattern, can't be predicted, just chance.
- **Stationary** — a fancy word for "boring and stable": a series whose average level
  and typical wiggle-size stay the same over time, with no trend and no seasonality
  left in it. Some models need the data put into this "boring" shape before they can
  work.
- **Differencing** — instead of looking at the raw numbers, look at the *change from
  last time* (`this month minus last month`). This is the standard trick to turn a
  trending series into a stationary one — a steadily climbing line becomes a roughly
  flat line of "how much it climbed each month."
- **Lag** — "how far back" you're looking. Lag 1 = last month. Lag 12 = the same
  month one year ago.
- **AR (AutoRegressive)** — "predict the next value from recent past values."
  Example: guess next month's sales as a weighted blend of the last few months.
- **MA (Moving Average, of errors)** — "predict the next value partly from how wrong
  your recent guesses were." If the model kept under-guessing the last two months,
  it nudges the next guess up to compensate.
- **ARIMA(p, d, q)** — one model that combines all three ideas above:
  - `p` = how many past *values* it looks back at (the AR part)
  - `d` = how many times you difference the data first, to strip out trend
  - `q` = how many past *mistakes* it looks back at (the MA part)
- **SARIMA(p, d, q)(P, D, Q, m)** — the same idea as ARIMA, plus a second copy of it
  applied one full cycle back instead of one step back. `m` is the cycle length (12
  for "yearly" on monthly data). So alongside "look at last month," it also looks at
  "look at this same month last year" — which is what lets it actually learn a
  repeating seasonal shape instead of just a trend.
- **Fitting a model** — the computer trying out internal numbers (coefficients) until
  its predictions match the historical data as closely as possible. This is the
  "training" step.
- **AIC (Akaike Information Criterion)** — a single score used to compare candidate
  models: rewards a model for fitting the data well, but penalizes it for being
  needlessly complicated. Lower AIC = better trade-off. Used here to automatically
  pick `p, d, q` (and `P, D, Q`) instead of guessing by hand.
- **Grid search** — trying every reasonable combination of settings (every `p, d, q`
  in a range) and keeping whichever one scored best (lowest AIC). Brute force, not
  clever, but reliable.
- **Forecast** — once a model is fitted on known data, ask it to predict values for
  dates it has never seen (the future).
- **Backtesting** — a practice exam for the model: hide the most recent real months,
  pretend you don't know them, have the model forecast them anyway, then compare the
  guess to what actually happened. This is how you find out if a model is any good
  *before* trusting its forecast of the real, unknown future.
- **RMSE / MAE / MAPE** — three ways to score "how wrong were the guesses," all
  computed only during backtesting (where the real answer is known):
  - **MAE** — the average size of the miss, in the original units (e.g. "off by 285
    books on average").
  - **RMSE** — similar to MAE, but big misses count extra (squares the errors before
    averaging) — a high RMSE relative to MAE means the model had a few bad misses
    rather than many small ones.
  - **MAPE** — the miss size as a percentage of the real value (e.g. "off by 5% on
    average") — easiest to compare across models or datasets of different scale.

## 1. What the data actually looks like

`data/generate_dummy_data.py` builds each month's sales as:

```
sales[t] = base_level + trend[t] + seasonality[month_of(t)] + noise[t] + promo[t]
```

None of the constants below are derived from a real dataset — they're hand-picked so
the dummy series *looks and behaves* like plausible monthly book sales for a mid-size
publisher: a real trend, a real seasonal shape, and real (but bounded) noise. Here's
what each one means and why that specific value:

- **`base_level = 1500`** — the starting point in month 0 (Jan 2018), before
  anything else is added. This just sets the overall scale of the series (thousands
  of units/month, not millions or tens); the number itself carries no other meaning.

- **`trend[t] = 6.0 * t`**, where `t` is the month index (0, 1, 2, ... 83) — sales
  drift up ~6 units every month, i.e. ~72/year, i.e. ~4.8% of `base_level` per year.
  Over all 84 months that's `83 * 6 = 498` units of cumulative growth by the end —
  a slope chosen to be *visible but not dominant*: noticeably less than the
  700-unit December seasonal swing, so seasonality remains the more prominent pattern
  in the data (and the more interesting thing for SARIMA to demonstrate).

- **`monthly_effect` dict (−180 to +700)** — a fixed offset per calendar month,
  repeating every year; this *is* the seasonality. The shape was hand-drawn to mimic
  a believable retail year for books: a post-holiday slump in Jan/Feb (−150, −180),
  a gradual climb back to baseline through spring/summer, a back-to-school bump in
  Sep (+300) that tapers in Oct (+200), then a holiday build-up in Nov (+450) peaking
  in Dec (+700) — the single largest effect in the year, since gift-buying is the
  most recognizable real-world seasonal driver for this kind of product.

- **`noise ~ Normal(0, scale=80)`** — random month-to-month variation with standard
  deviation 80. Small relative to the 700-unit December effect (so seasonality still
  dominates) but large relative to the ~6/month trend (so individual months visibly
  jitter around the trend line rather than forming a perfectly smooth curve).

- **`promo_spikes`: ~1 in every 15 months gets a `Uniform(200, 500)` bump** —
  simulates occasional unplanned promotions or surprise bestsellers that don't follow
  a calendar pattern. This matters for the modeling story specifically: it's variation
  with *no learnable pattern*, so it shows up as irreducible residual error in the
  backtest metrics no matter how good the model is.

- **`np.clip(sales, a_min=100)`** — a floor so an unlucky noise draw can never push a
  simulated month to zero or negative sales; 100 was picked as "clearly still a real,
  if slow, month," comfortably below the Jan/Feb seasonal dip.

- **`np.random.seed(42)`** — makes the "random" data reproducible (same numbers every
  time you regenerate it); 42 has no significance beyond being a conventional seed.

- **`NUM_MONTHS = 84` (7 years)** — chosen so the yearly seasonal pattern repeats 7
  times. This isn't arbitrary padding: SARIMA's seasonal AR/MA terms need several
  repetitions of the cycle to estimate reliably — with only 1-2 years of history
  there wouldn't be enough repeated Novembers/Decembers for the seasonal grid search
  to find a meaningful `(P, D, Q, 12)`.

The reason this data design matters for modeling: the series has **both a trend and
a repeating 12-month cycle**, which is exactly the combination ARIMA cannot fully
describe on its own (it has no concept of "month 12 looks like month 0") but SARIMA
is built for.

## 2. Loading and preparing the series

`src/data_loader.py::load_book_sales`

- Parses `date`, sorts, sets it as the index, and calls `.asfreq("MS")` — this pins
  the series to a strict "month start" calendar so every downstream model knows the
  step size between observations is exactly one month (this is what lets SARIMA's
  `m=12` mean "12 steps = 1 year").
- Any resulting gap (a missing month) is filled via linear interpolation — models
  below need a value at every regular time step.

`train_test_split_series` then holds out the **last 12 months** as `test`, leaving
everything before that as `train`. Backtesting (step 4 below) only ever sees `train`;
`test` is used purely to check the forecast against reality afterward.

## 3. ARIMA — the non-seasonal model

`src/arima_model.py`

ARIMA(p, d, q) describes a value as a combination of three ideas:

- **I (Integrated), order `d`**: instead of modeling the raw series, model it after
  differencing `d` times (`diff[t] = sales[t] - sales[t-1]`, applied `d` times).
  Differencing removes trend — a straight upward line becomes a roughly constant
  series after one difference, a curving one after two. This is *why* `d` matters:
  ARIMA's AR/MA math assumes a stationary (no-trend) series, and this dummy data's
  trend is not perfectly linear (noise + promos wobble it), so the search often picks
  `d=2` here.
- **AR (AutoRegressive), order `p`**: today's (differenced) value as a weighted sum
  of the previous `p` (differenced) values, e.g. `y[t] = c + φ1*y[t-1] + ... + φp*y[t-p] + e[t]`.
- **MA (Moving Average), order `q`**: today's value also depends on the previous `q`
  *forecast errors*, e.g. `+ θ1*e[t-1] + ... + θq*e[t-q]`. This lets the model
  correct itself based on how wrong its recent predictions were.

**Choosing (p, d, q):** `find_best_arima_order` brute-force fits every combination in
`P_RANGE × D_RANGE × Q_RANGE` (0-3, 0-2, 0-3 by default) via `statsmodels.ARIMA`, and
keeps whichever minimizes **AIC** (Akaike Information Criterion — roughly
"goodness of fit, penalized for using more parameters", lower is better). This avoids
needing `pmdarima`'s auto-search, at the cost of being an exhaustive rather than
smart search. In our test run this picked **ARIMA(0, 2, 3)**: no AR terms, double
differencing, three MA terms.

**Fitting** (`fitted = model.fit()`) estimates the φ/θ coefficients that minimize
prediction error on `train`, via maximum likelihood — under the hood this uses a
Kalman filter over the model's state-space form, but conceptually it's "find the
coefficients that best explain what already happened."

**Forecasting** (`forecast_arima` → `fitted_model.get_forecast(steps=n)`): the model
projects its AR/MA recursion forward `n` steps with no more real data to correct it
against, then **un-differences** (integrates) the result back onto the original sales
scale. Because there's no seasonal term, and by month 12+ the AR/MA effects decay
out, this projection settles into essentially a straight continuation of the recent
trend — which is exactly why the ARIMA forecast in `outputs/forecast_12_months.csv`
is a nearly flat ~2170 → ~2266 climb with no Nov/Dec bump.

## 4. SARIMA — adding the seasonal component

`src/sarima_model.py`

SARIMA(p, d, q)(P, D, Q, m) is ARIMA plus a second, seasonal ARIMA applied at lag
`m` (here `m=12`, i.e. year-over-year):

- **Seasonal differencing, order `D`**: `sdiff[t] = y[t] - y[t-12]` — this removes
  the repeating yearly shape by comparing each month only to the *same month last
  year*, isolating the actual "growth beyond seasonality."
- **Seasonal AR, order `P`**: this month's value depends on the value(s) from
  `P` years ago (lag 12, 24, ...).
- **Seasonal MA, order `Q`**: same idea as MA above, but for errors 12 (or 24, ...)
  months back.

These seasonal terms are what let the model "know" that November and December are
reliably higher, and January/February reliably lower — information plain ARIMA
structurally cannot represent.

**Choosing the order:** `find_best_sarima_order` grid-searches `(p,d,q)` over
`P_RANGE/D_RANGE/Q_RANGE` (0-2/0-1/0-2) crossed with seasonal `(P,D,Q)` over
`SP_RANGE/SD_RANGE/SQ_RANGE` (0-1 each, `m` fixed at 12), again picking lowest AIC via
`statsmodels.SARIMAX`. In our run this picked **SARIMA(2, 1, 2)(1, 1, 1, 12)** — one
regular difference plus one *seasonal* difference, with both regular and seasonal
AR/MA terms active.

**Fitting and forecasting** work the same way as ARIMA (Kalman-filter-based maximum
likelihood fit, then `get_forecast`), except the seasonal terms mean the forward
projection actively reproduces the learned Jan-dip/Sep-bump/Dec-spike shape instead
of flattening out — which is why the SARIMA column in the same CSV dips to ~1830 in
January, climbs through the year, and spikes to ~2800 by December: it's replaying the
same seasonal curve visible in the historical data, on top of the underlying trend.

## 5. From "fitted model" to "the number in the CSV"

`src/pipeline.py` ties steps 3-4 together in two passes:

1. **`run_backtest(train, test)`** — fits ARIMA and SARIMA on `train` only (the data
   with the last 12 months hidden), forecasts those same 12 months, and compares
   against the real `test` values via `src/evaluation.py` (RMSE, MAE, MAPE — see
   below). This answers "if this model only knew data through early 2024, how close
   would its guess for the last 12 months have been to what actually happened?" In
   our run: ARIMA MAPE ≈ 14.6%, SARIMA MAPE ≈ 4.9% — concretely confirming SARIMA's
   seasonal terms make a large difference on data shaped like this.

2. **`run_future_forecast(full_series, ...)`** — refits *both* models, using the same
   `(p,d,q)` / `(P,D,Q,m)` orders already chosen in step 1 (not re-searched), but this
   time on the **entire** history (train + test combined, i.e. all real data through
   Dec 2024). It then forecasts `max(horizons)` months past the last real data point,
   and slices that single forecast down to the 3/6/9/12-month tables. Refitting on
   full history (rather than reusing the backtest-fitted model) means the coefficients
   get to use the most recent 12 months of real data too, not just data through the
   cutoff used for backtesting.

The resulting `predicted_mean` series from `get_forecast()` — already on the original
sales scale, differencing fully undone — is what gets rounded and written as
`arima_forecast` / `sarima_forecast` in `outputs/forecast_{h}_months.csv`.

## 6. The evaluation metrics

`src/evaluation.py::evaluate_forecast`, computed only during backtesting (step 1
above), where real held-out values exist to compare against:

- **RMSE** (root mean squared error): `sqrt(mean((actual - predicted)^2))` — in raw
  sales units, penalizes large misses more than small ones.
- **MAE** (mean absolute error): `mean(|actual - predicted|)` — average miss size,
  in raw sales units, treating all errors linearly.
- **MAPE** (mean absolute percentage error): `mean(|actual - predicted| / actual) * 100` —
  the same idea as MAE but as a percentage of the actual value, which is why it's the
  easiest of the three to compare across models regardless of scale.

## 7. What if the numbers change over time? (monitoring & drift)

**Simplest version:** think of the model like a kid who's gotten really good at
guessing what day the ice cream truck comes, just by watching the last few months.
If the truck starts coming on different days, the kid's old guessing habit stops
working — that's what a rising MAPE means: the guess used to match reality, and now
it doesn't.

The good news: **every time you re-run the pipeline, the "kid" re-learns from
scratch.** It's not stuck with old habits — `run_backtest` re-runs the AIC grid
search on whatever data exists *right now*, so a lot of drift fixes itself
automatically just by re-running with fresh data (see `outputs/metrics_history.csv`
and `outputs/metrics_trend.png`, from `src/metrics_log.py`, for tracking this across
runs rather than trusting a single run).

But sometimes it needs a bigger nudge, and there are exactly two dials for that:

1. **"How far back are you allowed to think?"** — the `*_RANGE` constants at the top
   of `src/arima_model.py` and `src/sarima_model.py` (e.g. `P_RANGE`, `D_RANGE`,
   `SP_RANGE`...) cap how complex a pattern the grid search is even allowed to try.
   If reality now needs a more complex pattern than the search is allowed to
   consider, widen these ranges.
2. **"How long is one full cycle?"** — `SEASONAL_PERIOD = 12` in
   `src/sarima_model.py` means "compare each month to the same month one year ago."
   If the real repeating cycle isn't yearly anymore (e.g. you switch to weekly data,
   or the business itself becomes driven by a different cycle), this is the number
   to change to match it.

**If neither dial fixes it, that's the real warning sign** — not "MAPE went up
once," but "MAPE keeps getting worse run after run even as the search re-adapts."
It means something genuinely different happened in the real world (a new product
line, a pricing change, a disruption like COVID) rather than the usual noise ARIMA/
SARIMA already expect. No dial fixes that — the real options are: exclude the
disrupted period from training data, add an external variable (SARIMAX with
exogenous regressors), or accept that a different modeling approach is needed. That's
a judgment call about what changed in the business, not a settings tweak.

**Simplest possible rule:** one bad guess = don't panic, just re-run with fresh data.
Guesses getting worse *again and again* over many re-runs = something real changed,
time to actually go look at what happened.

## Summary of the full path

```
raw CSV (date, sales)
  -> monthly-indexed Series, gaps interpolated                [data_loader.py]
  -> split into train (all but last 12mo) / test (last 12mo)  [data_loader.py]
  -> grid search (p,d,q) and (p,d,q)(P,D,Q,12) by AIC on train [arima_model.py, sarima_model.py]
  -> fit both models on train, forecast the 12 held-out months,
     compare to real test values (RMSE/MAE/MAPE)              [pipeline.py, evaluation.py]
  -> refit both models (same orders) on the FULL series       [pipeline.py]
  -> forecast max(horizons) months ahead, undo all differencing
     back to raw sales scale                                  [arima_model.py, sarima_model.py]
  -> slice into 3/6/9/12-month tables, round, save to CSV/plot [main.py / app/main.py]
```
