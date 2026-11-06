# Month landing forecast

Where does this month finish, given today? Two layers: a pre-month forecast
from five years of history (trend plus month-of-year seasonality), then a
daily pacing adjustment that blends in as the month reveals itself - so a
month running strong lifts the full-month number instead of being
under-called. Walk-forward backtested on synthetic data with known injected
effects.

![Backtest funnel](outputs/backtest_funnel.png)

## Why it exists

Every commercial team has the 4pm question: are we landing this month or
not? The common answers fail in opposite directions. A forecast set before
the month and held ignores what the month is actually doing, so strong
months get under-called. A straight run-rate chases every lumpy order and
ignores that months load toward their end. The honest answer uses both,
weighted by how much of the month has been seen.

## Method

1. **Working days only.** Weekends and England bank holidays carry no
   revenue and no weight. A month is its working days.
2. **The pre-month forecast.** Log-linear trend plus month-of-year factors,
   fitted per unit on up to five years of completed months. This is the
   number you would publish before day one.
3. **The pacing curve.** From the trailing six completed months, the
   cumulative share of a month normally landed by each point in it. Months
   load toward their end; the curve knows by how much, per unit.
4. **The blend.** Landing = share x (MTD / expected share) + (1 - share) x
   pre-month forecast. The weight is simply the share of the month observed:
   day two leans on the forecast, day eighteen leans on the evidence.
   Nothing is tuned, and the weight is stated so anyone can argue with it.
5. **Benchmarks, always.** Pacing alone, naive run-rate and the held
   forecast run alongside. A method that cannot beat the naive answers is a
   decoration.

## Backtest

Walk-forward over 34 synthetic months (2024-01 to 2026-10), every working
day, all units summed. Median absolute percentage error by point in month:

| Elapsed | Blend | Pacing alone | Run-rate | Held forecast |
|---|---|---|---|---|
| 20% | 6.4% | 6.8% | 11.1% | 5.6% |
| 40% | 4.6% | 5.9% | 8.8% | 5.5% |
| 60% | 3.7% | 4.8% | 7.7% | 5.6% |
| 80% | 3.1% | 3.0% | 3.5% | 5.6% |

The shape is the point. Early month, the pre-month forecast is still the
best single answer because two days of lumpy orders prove little. From 40%
elapsed the blend leads. The run-rate never leads at any point in the month.

**Strong months are the test that matters.** In the 10 of 34 months that
finished more than 5% above the pre-month forecast, the held forecast
under-read them by a median 10.5% at half-month; the blend by 7.2%. That is
the under-forecasting of good months, measurably reduced - not eliminated,
because at half-month the evidence is still partial, and the table says so.

## What it cannot do, shown rather than hidden

The generator injects orders pulled forward ahead of an April price
increase: March loads late, April starts hollow. A pacing curve cannot know
an order was pulled forward; it can only notice the pace has broken from
profile. The backtest prints every month it mis-read at 40% elapsed - the
pull-forward Marches are on the list alongside ordinarily lumpy months -
and the honest operational answer is a flag a human reads, not an automatic
correction chasing its own tail.

This is also not a demand model. It forecasts where the month lands given
how it is pacing; it says nothing about why.

## Data

Synthetic, with known ground truth: trend, monthly and weekday seasonality,
month-end loading, holiday gaps, lumpy large orders and pull-forward events
are injected on purpose (parameters at the top of `month_landing.py`), so
recovery is provable. Units are generic and configurable. The method class
is standard in parts and distribution businesses; this is a from-scratch
rebuild on public code and synthetic data.

## Run it

```
python3 -m venv .venv && .venv/bin/pip install matplotlib
.venv/bin/python month_landing.py
```
