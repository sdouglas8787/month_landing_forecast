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

1. **Working days, weighted.** Weekends and England bank holidays carry
   no revenue and no weight - and the working days are not equal either.
   Weekday weights are learned from the trailing twelve months per unit
   (in the synthetic data: Monday low, Tuesday best, Wednesday next,
   Thursday low, Friday third). A month is the sum of its weighted days,
   so a bank holiday changes the mix, not just the count. The method's own
   history runs the same way: the first version of this idea used calendar
   days, and treating days as equal was its first measured failure.
2. **The pre-month forecast.** Log-linear trend plus month-of-year
   factors, fitted per unit on up to five years of completed months - on
   the rate per weighted day, then priced by the month's own composition:
   how many of which weekdays actually fall in it, net of holidays. Two
   Februaries with the same seasonality can still differ, and this is
   where the difference lives.
3. **The pacing curve.** From the trailing six completed months, the
   cumulative share of a month normally landed by each point in it - with
   "point in it" measured in weighted days, not elapsed days. Months load
   toward their end; the curve knows by how much, per unit.
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
| 20% | 3.7% | 7.2% | 11.5% | 4.7% |
| 40% | 3.4% | 5.8% | 8.6% | 4.9% |
| 60% | 3.4% | 5.0% | 7.3% | 4.7% |
| 80% | 2.4% | 3.0% | 3.5% | 4.7% |

The blend leads at every point in the month, and the weekday weighting is
why: before it, raw pacing was noisy enough early on that the held
forecast won the first week. The run-rate never leads anywhere.

**Strong months are the test that matters.** In the 10 of 34 months that
finished more than 5% above the pre-month forecast, the held forecast
under-read them by a median 8.1% at half-month; the blend by 6.3%. That is
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

## The daily table

`outputs/daily_table_sample.csv` reproduces the working layout this method
grew up in: one row per calendar day - date, day, working-day and
bank-holiday flags - with the forecast standing in every future cell until
an actual lands and overwrites it. In the spreadsheet ancestor that
overwrite test was =TODAY(); here it is data presence. The interface is
part of the method: anyone can see which numbers are real and which are
still the model's.

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
