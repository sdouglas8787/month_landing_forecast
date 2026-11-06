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

# England bank holidays, 2023-2026.
HOLIDAYS = {d for d in map(date.fromisoformat, """
2023-01-02 2023-04-07 2023-04-10 2023-05-01 2023-05-08 2023-05-29 2023-08-28 2023-12-25 2023-12-26
2024-01-01 2024-03-29 2024-04-01 2024-05-06 2024-05-27 2024-08-26 2024-12-25 2024-12-26
2025-01-01 2025-04-18 2025-04-21 2025-05-05 2025-05-26 2025-08-25 2025-12-25 2025-12-26
2026-01-01 2026-04-03 2026-04-06 2026-05-04 2026-05-25 2026-08-31 2026-12-25 2026-12-28
""".split())}

UNITS = {
    # name: (base daily £k, annual growth, month-end loading, pull-forward share)
    "Core parts":  (420.0, 0.05, 0.9, 0.12),
    "Accessories": (150.0, 0.08, 0.5, 0.04),
    "Service":     (95.0,  0.03, 0.2, 0.00),
    "Internal":    (60.0,  0.00, 1.3, 0.00),
}
MONTH_F = [0.97, 0.99, 1.08, 0.98, 1.00, 1.02, 0.96, 0.88, 1.05, 1.03, 1.02, 0.86]
WEEKDAY_F = [0.95, 1.00, 1.02, 1.03, 1.00]            # Mon..Fri
PRICE_RISE_MONTH = 4         # increases land 1 April: March loads, April dips
START, END = date(2023, 1, 1), date(2026, 10, 31)
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
    for u, (base, growth, loading, pull) in UNITS.items():
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
                out[u][d] = level * rng.lognormvariate(0, 0.06)
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
        frac = np.arange(1, len(wds) + 1) / len(wds)
        curves.append(np.interp(GRID, frac, cum))
    return np.mean(curves, axis=0) if curves else None


def landing(actuals, unit, y, m, elapsed):
    """Forecast the month's total after `elapsed` working days."""
    wds = working_days(y, m)
    mtd = sum(actuals[unit][d] for d in wds[:elapsed])
    frac = elapsed / len(wds)
    curve = share_curve(actuals, unit, y, m)
    expected = float(np.interp(frac, GRID, curve))
    return mtd / max(expected, 1e-9), mtd / frac       # profiled, naive run-rate


def backtest(actuals):
    """Walk every month of 2024-01 onward, every working day, all units summed."""
    rows = []
    for y, m in month_iter(date(2024, 1, 1), END):
        wds = working_days(y, m)
        total = sum(actuals[u][d] for u in UNITS for d in wds)
        yy, mm = (y - 1, 12) if m == 1 else (y, m - 1)
        prior = sum(actuals[u][d] for u in UNITS for d in working_days(yy, mm))
        for e in range(2, len(wds) + 1):
            prof = run = 0.0
            for u in UNITS:
                p, r = landing(actuals, u, y, m, e)
                prof += p
                run += r
            rows.append({"year": y, "month": m, "elapsed": e, "of": len(wds),
                         "frac": e / len(wds), "actual": total,
                         "profiled": prof, "runrate": run, "prior": prior})
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
        funnel.append({"elapsed": f"{int(hi*100)}%", "profiled": med("profiled"),
                       "runrate": med("runrate"), "prior": med("prior")})

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
    for key, colour, label in (("prior", MUTED, "Prior month as forecast"),
                               ("runrate", SECOND, "Naive run-rate"),
                               ("profiled", ACCENT, "Profiled pacing")):
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
    print(f"{'elapsed':>8s} {'profiled':>9s} {'run-rate':>9s} {'prior mo':>9s}")
    for f in funnel:
        print(f"{f['elapsed']:>8s} {f['profiled']:>8.1f}% {f['runrate']:>8.1f}% "
              f"{f['prior']:>8.1f}%")
    print("\nMonths mis-read at 40% elapsed (profiled APE > 8%):")
    for y, m, e in flagged:
        tag = " <- price-rise pull-forward" if m in (PRICE_RISE_MONTH - 1,
                                                     PRICE_RISE_MONTH) else ""
        print(f"  {y}-{m:02d}  {e:4.1f}%{tag}")
    print("\nThe pacing curve cannot know an order was pulled forward; it can "
          "only notice the pace break. That is why the flag exists.")


if __name__ == "__main__":
    actuals = generate()
    report(backtest(actuals))
