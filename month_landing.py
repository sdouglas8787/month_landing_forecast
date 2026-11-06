"""Month landing forecast: where does this month finish, given today?

Daily revenue actuals, working days only. A pacing curve learned from the
trailing six months says what share of a month is normally landed by each
point in it; dividing month-to-date by that share gives the landing
forecast, which updates every day and re-learns every month.

Everything runs on synthetic data with KNOWN injected effects (trend,
monthly and weekday seasonality, month-end loading, orders pulled forward
ahead of a price increase), so the backtest can show exactly what the
engine catches and what it structurally cannot.

Run:  python month_landing.py   ->  console report + outputs/
"""

from __future__ import annotations

import csv
import math
import random
from datetime import date, timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, INK, MUTED, LINE = "#fcfbf9", "#1c2430", "#5b6673", "#e4e0d8"
ACCENT, SECOND, THIRD = "#2a6f6a", "#b0543f", "#8a7f10"

# England bank holidays, 2021-2026.
HOLIDAYS = {d for d in map(date.fromisoformat, """
2021-01-01 2021-04-02 2021-04-05 2021-05-03 2021-05-31 2021-08-30 2021-12-27 2021-12-28
2022-01-03 2022-04-15 2022-04-18 2022-05-02 2022-06-02 2022-06-03 2022-08-29 2022-09-19 2022-12-26 2022-12-27
2023-01-02 2023-04-07 2023-04-10 2023-05-01 2023-05-08 2023-05-29 2023-08-28 2023-12-25 2023-12-26
2024-01-01 2024-03-29 2024-04-01 2024-05-06 2024-05-27 2024-08-26 2024-12-25 2024-12-26
2025-01-01 2025-04-18 2025-04-21 2025-05-05 2025-05-26 2025-08-25 2025-12-25 2025-12-26
2026-01-01 2026-04-03 2026-04-06 2026-05-04 2026-05-25 2026-08-31 2026-12-25 2026-12-28
""".split())}

UNITS = {
    # name: (base daily GBPk, annual growth, month-end loading,
    #        pull-forward share, lumpiness: P(large order), its size range)
    "Core parts":  (420.0, 0.05, 0.9, 0.12, 0.05, (1.8, 4.0)),
    "Accessories": (150.0, 0.08, 0.5, 0.04, 0.03, (1.5, 2.5)),
    "Service":     (95.0,  0.03, 0.2, 0.00, 0.01, (1.3, 1.8)),
    "Internal":    (60.0,  0.00, 1.3, 0.00, 0.08, (2.0, 5.0)),
}
MONTH_F = [0.97, 0.99, 1.08, 0.98, 1.00, 1.02, 0.96, 0.88, 1.05, 1.03, 1.02, 0.86]
WEEKDAY_F = [0.88, 1.12, 1.05, 0.92, 1.03]   # Mon low, Tue best, Wed next, Thu low, Fri third
PRICE_RISE_MONTH = 4         # increases land 1 April: March loads, April dips
START, END = date(2021, 1, 1), date(2026, 10, 31)   # five years of dailies
TRAIL_MONTHS = 6             # months the pacing curve learns from
GRID = np.linspace(0.05, 1.0, 20)                     # share curve sample points


def working_days(y, m):
    d, out = date(y, m, 1), []
    while d.month == m:
        if d.weekday() < 5 and d not in HOLIDAYS:
            out.append(d)
        d += timedelta(days=1)
    return out


def month_iter(a, b):
    y, m = a.year, a.month
    while (y, m) <= (b.year, b.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def generate():
    """Daily actuals per unit with known effects. Returns {unit: {date: £k}}."""
    rng = random.Random(7)
    out = {u: {} for u in UNITS}
    for u, (base, growth, loading, pull, p_lump, lump) in UNITS.items():
        for y, m in month_iter(START, END):
            wds = working_days(y, m)
            n = len(wds)
            for i, d in enumerate(wds):
                t_years = (d - START).days / 365.25
                level = base * (1 + growth) ** t_years * MONTH_F[m - 1] \
                        * WEEKDAY_F[d.weekday()]
                pos = (i + 1) / n                      # position in month
                level *= 1 + loading * max(0.0, pos - 0.7)   # month-end ramp
                # orders pulled ahead of the April price rise
                if pull:
                    if m == PRICE_RISE_MONTH - 1 and pos > 0.8:
                        level *= 1 + pull * 4          # March loads up
                    if m == PRICE_RISE_MONTH and pos < 0.3:
                        level *= 1 - pull * 2          # April starts hollow
                level *= rng.lognormvariate(0, 0.10)
                if rng.random() < p_lump:              # a large order lands
                    level *= rng.uniform(*lump)
                out[u][d] = level
    return out


def share_curve(actuals, unit, y, m):
    """Average cumulative-share curve from the trailing TRAIL_MONTHS months."""
    curves = []
    yy, mm = y, m
    for _ in range(TRAIL_MONTHS):
        yy, mm = (yy - 1, 12) if mm == 1 else (yy, mm - 1)
        if (yy, mm) < (START.year, START.month):
            break
        wds = working_days(yy, mm)
        vals = np.array([actuals[unit][d] for d in wds])
        cum = np.cumsum(vals) / vals.sum()
        frac, _ = weighted_fracs(wds, weekday_weights(actuals, unit, y, m))
        curves.append(np.interp(GRID, frac, cum))
    return np.mean(curves, axis=0) if curves else None


_MT, _PM, _WW = {}, {}, {}


def weekday_weights(actuals, unit, y, m):
    """Mean revenue by weekday over the 12 months before (y, m), scaled to
    average 1. Tuesday is not Monday, and the forecast should know."""
    key = (unit, y, m)
    if key not in _WW:
        yy, mm, days = y, m, []
        for _ in range(12):
            yy, mm = (yy - 1, 12) if mm == 1 else (yy, mm - 1)
            if (yy, mm) < (START.year, START.month):
                break
            days += [(d.weekday(), actuals[unit][d]) for d in working_days(yy, mm)]
        w = np.ones(5)
        for k in range(5):
            vals = [v for wd, v in days if wd == k]
            if vals:
                w[k] = np.mean(vals)
        _WW[key] = w / w.mean()
    return _WW[key]


def weighted_fracs(wds, w):
    """Cumulative weighted share of the month represented by each day."""
    ww = np.array([w[d.weekday()] for d in wds])
    return np.cumsum(ww) / ww.sum(), ww.sum()

def monthly_totals(actuals, unit):
    if unit not in _MT:
        _MT[unit] = {(y, m): sum(actuals[unit][d] for d in working_days(y, m))
                     for y, m in month_iter(START, END)}
    return _MT[unit]


def pre_month(actuals, unit, y, m):
    """The forecast made before the month starts: log-linear trend plus
    month-of-year factors, fitted on up to five years of completed months."""
    key = (unit, y, m)
    if key in _PM:
        return _PM[key]
    tot = monthly_totals(actuals, unit)
    w = weekday_weights(actuals, unit, y, m)
    wdsum = lambda yy, mm: weighted_fracs(working_days(yy, mm), w)[1]
    hist = [(yy, mm) for yy, mm in month_iter(START, END)
            if (yy, mm) < (y, m)][-60:]
    t = np.arange(len(hist))
    # fit the rate per weighted day, so the month's own day mix - how many
    # of which weekdays, net of holidays - prices the forecast explicitly
    logy = np.log([tot[k] / wdsum(*k) for k in hist])
    X = np.column_stack([np.ones(len(hist)), t] +
                        [(np.array([k[1] for k in hist]) == mm).astype(float)
                         for mm in range(2, 13)])
    beta, *_ = np.linalg.lstsq(X, logy, rcond=None)
    x = np.concatenate([[1.0, len(hist)],
                        [(m == mm) for mm in range(2, 13)]]).astype(float)
    _PM[key] = float(np.exp(x @ beta)) * wdsum(y, m)
    return _PM[key]


def landing(actuals, unit, y, m, elapsed):
    """Forecast the month's total after `elapsed` working days."""
    wds = working_days(y, m)
    mtd = sum(actuals[unit][d] for d in wds[:elapsed])
    fracs, _ = weighted_fracs(wds, weekday_weights(actuals, unit, y, m))
    frac = float(fracs[elapsed - 1])
    curve = share_curve(actuals, unit, y, m)
    expected = float(np.interp(frac, GRID, curve))
    pacing = mtd / max(expected, 1e-9)
    prior = pre_month(actuals, unit, y, m)
    # credibility blend: weight the pacing evidence by the share of the
    # month it has actually seen. Early days lean on the pre-month
    # forecast; a month pacing strong lifts the full-month number instead
    # of being under-called.
    blend = expected * pacing + (1 - expected) * prior
    return blend, pacing, mtd / frac, prior


def backtest(actuals):
    """Walk every month of 2024-01 onward, every working day, all units summed."""
    rows = []
    for y, m in month_iter(date(2024, 1, 1), END):
        wds = working_days(y, m)
        total = sum(actuals[u][d] for u in UNITS for d in wds)
        yy, mm = (y - 1, 12) if m == 1 else (y, m - 1)
        held = sum(pre_month(actuals, u, y, m) for u in UNITS)
        for e in range(2, len(wds) + 1):
            bl = prof = run = 0.0
            for u in UNITS:
                b, p, r, _ = landing(actuals, u, y, m, e)
                bl += b
                prof += p
                run += r
            rows.append({"year": y, "month": m, "elapsed": e, "of": len(wds),
                         "frac": e / len(wds), "actual": total, "blend": bl,
                         "profiled": prof, "runrate": run, "prior": held})
    return rows


def report(rows):
    ape = lambda f, a: abs(f - a) / a * 100
    buckets = [(lo, lo + 0.1) for lo in np.arange(0.1, 1.0, 0.1)]
    funnel = []
    for lo, hi in buckets:
        sub = [r for r in rows if lo < r["frac"] <= hi]
        if not sub:
            continue
        med = lambda k: float(np.median([ape(r[k], r["actual"]) for r in sub]))
        funnel.append({"elapsed": f"{int(hi*100)}%", "blend": med("blend"),
                       "profiled": med("profiled"), "runrate": med("runrate"),
                       "prior": med("prior")})

    with open("outputs/backtest_funnel.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=funnel[0].keys())
        w.writeheader()
        w.writerows(funnel)

    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    xs = range(len(funnel))
    for key, colour, label in (("prior", MUTED, "Pre-month forecast, held"),
                               ("runrate", SECOND, "Naive run-rate"),
                               ("profiled", THIRD, "Pacing alone"),
                               ("blend", ACCENT, "Blend")):
        ys = [f[key] for f in funnel]
        ax.plot(xs, ys, color=colour, linewidth=1.8)
        ax.annotate(f"{label} {ys[-1]:.1f}%", xy=(len(funnel) - 1, ys[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=colour, fontweight="bold")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f["elapsed"] for f in funnel], fontsize=8)
    ax.set_xlim(-0.5, len(funnel) * 1.35)
    ax.grid(axis="y", color=LINE, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Landing forecast error by point in month (median APE, %)",
                 loc="left", fontsize=11, color=INK, fontweight="bold")
    fig.text(0.01, 0.02, "Walk-forward over 34 synthetic months, all units, "
             "every working day. Known injected effects incl. pull-forward.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig("outputs/backtest_funnel.png", facecolor=BG)

    # where the pacing model is structurally blind: the pull-forward months
    flagged = []
    for (y, m) in {(r["year"], r["month"]) for r in rows}:
        at40 = min((r for r in rows if (r["year"], r["month"]) == (y, m)),
                   key=lambda r: abs(r["frac"] - 0.4))
        e = ape(at40["profiled"], at40["actual"])
        if e > 8:
            flagged.append((y, m, e))
    flagged.sort(key=lambda x: -x[2])

    print("MONTH LANDING FORECAST - walk-forward backtest")
    print(f"{'elapsed':>8s} {'blend':>8s} {'pacing':>8s} {'run-rate':>9s} {'held fc':>8s}")
    for f in funnel:
        print(f"{f['elapsed']:>8s} {f['blend']:>7.1f}% {f['profiled']:>7.1f}% "
              f"{f['runrate']:>8.1f}% {f['prior']:>7.1f}%")

    # his sentence, demonstrated: strong months must not be under-called
    strong = {(r["year"], r["month"]) for r in rows
              if r["actual"] > r["prior"] * 1.05}
    at = lambda k, ym: next(r[k] / r["actual"] - 1 for r in rows
                            if (r["year"], r["month"]) == ym
                            and abs(r["frac"] - 0.5) ==
                            min(abs(q["frac"] - 0.5) for q in rows
                                if (q["year"], q["month"]) == ym))
    if strong:
        h = float(np.median([at("prior", ym) for ym in strong])) * 100
        b = float(np.median([at("blend", ym) for ym in strong])) * 100
        print(f"\nStrong months (actual >5% above the pre-month forecast): "
              f"{len(strong)} of 34.")
        print(f"At half-month, the held forecast under-reads them by "
              f"{h:+.1f}%; the blend by {b:+.1f}%.")
    print("\nMonths mis-read at 40% elapsed (profiled APE > 8%):")
    for y, m, e in flagged:
        tag = " <- price-rise pull-forward" if m in (PRICE_RISE_MONTH - 1,
                                                     PRICE_RISE_MONTH) else ""
        print(f"  {y}-{m:02d}  {e:4.1f}%{tag}")
    print("\nThe pacing curve cannot know an order was pulled forward; it can "
          "only notice the pace break. That is why the flag exists.")


def daily_table(actuals, y, m, today_wd):
    """The layout the Excel ancestor used: one row per calendar day - date,
    day, workday and holiday flags - with the forecast standing in each
    future cell until an actual lands and overwrites it (the =TODAY() test,
    done by data presence)."""
    wds = working_days(y, m)
    cut = wds[today_wd - 1]
    land = rem_w = 0.0
    for u in UNITS:
        b, *_ = landing(actuals, u, y, m, today_wd)
        land += b
    mtd = sum(actuals[u][d] for u in UNITS for d in wds[:today_wd])
    w_avg = {d: np.mean([weekday_weights(actuals, u, y, m)[d.weekday()]
                         for u in UNITS]) for d in wds}
    rem_w = sum(w_avg[d] for d in wds[today_wd:])
    with open("outputs/daily_table_sample.csv", "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["date", "day", "month", "year", "working_day",
                      "bank_holiday", "total_gbpk", "source"])
        d = date(y, m, 1)
        while d.month == m:
            hol = d in HOLIDAYS
            wk = d.weekday() < 5 and not hol
            if not wk:
                val, src = "", ""
            elif d <= cut:
                val = round(sum(actuals[u][d] for u in UNITS), 1)
                src = "ACTUAL"
            else:
                val = round((land - mtd) * w_avg[d] / rem_w, 1)
                src = "FORECAST"
            wtr.writerow([d.isoformat(), d.strftime("%a").upper(), m, y,
                          "Y" if wk else "N", "Y" if hol else "N", val, src])
            d += timedelta(days=1)
    print(f"\nDaily table sample written for {y}-{m:02d} as at working day "
          f"{today_wd}: actuals to {cut}, forecast beyond, weekday-weighted.")


if __name__ == "__main__":
    actuals = generate()
    report(backtest(actuals))
    daily_table(actuals, 2026, 10, 12)
