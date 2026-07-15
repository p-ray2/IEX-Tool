#!/usr/bin/env python3
"""
Watchlist levels chart — price history with key support/resistance for all 6 names.

Each panel shows:
  · Price line (left axis)
  · Cumulative delta from IEX data overlaid as fill + line (right axis)
  · Key support/resistance levels as horizontal dashed lines
  · High-conviction absorption signal dates as purple triangles

Usage:
    python levels_chart.py
    python levels_chart.py --since 2026-03-01
    python levels_chart.py FDS WRB
    python levels_chart.py --save                  # save PNG without showing
    python levels_chart.py --no-delta              # price + levels only
"""

import argparse
import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "cache", "market_data.duckdb")

WATCHLIST = ["VIRT", "WRB", "RPM", "FDS", "GDDY", "IBM", "OSUR", "INTU", "NVO", "CME", "MBC", "AI"]

# Derived from cumulative delta + absorption analysis, updated Jul 7 2026 (data thru Jul 6).
LEVELS = {
    "FDS": {
        "bias":       "Reversal Watch",
        "support":    [245.00, 248.00],
        "resistance": [255.00, 262.00],
        "note":       "Jul 1 earnings rip $221->$245, Jul 6 $252.07 new recovery high; but cum delta -271.8K, still net selling INTO the rally = squeeze mechanics firing on price while tape shows distribution; fragile, watch $248 hold",
    },
    "VIRT": {
        "bias":       "Confirmed Reversal",
        "support":    [58.00, 60.00],
        "resistance": [63.00, 65.00],
        "note":       "Jul 6 $62.93 recovered from the $58.19 Jun 29 dip; cum delta +262K still strong but off the +289K Jun 25 high; retesting the $63 breakout high, needs to clear $64 to extend",
    },
    "WRB": {
        "bias":       "Distribution",
        "support":    [68.00, 69.50],
        "resistance": [72.00, 74.00],
        "note":       "Jul 2 $72.07 high, Jul 6 $70.92; +8% rally on NET SELLING (cum delta -49.9K) = textbook fragile rally, distribution into strength; sellers still winning the tape despite the higher price",
    },
    "RPM": {
        "bias":       "Reversal Watch",
        "support":    [108.00, 110.00],
        "resistance": [112.00, 114.00],
        "note":       "Jul 6 $110.39; cum delta FLIPPED to +13.3K from the -82.7K Jun 15 low — sharp accumulation reversal (Jun 30 +20K, Jul 2 +19K, Jul 6 +13K buy cluster); distribution thesis broken, buyers took control",
    },
    "GDDY": {
        "bias":       "Reversal Watch",
        "support":    [82.00, 84.00],
        "resistance": [88.00, 90.00],
        "note":       "Jul 2 $88.46, Jul 6 $85.64; price recovered off the $75 Jun lows but cum delta -369.5K at a NEW LOW = classic failed-absorption, distribution into the bounce; every buy signal keeps failing",
    },
    "IBM": {
        "bias":       "Reversal Watch",
        "support":    [288.00, 292.00],
        "resistance": [300.00, 308.00],
        "note":       "Jul 6 $299.65 (+22% off Jun lows); cum delta -1.92M still deeply negative BUT recovered +630K in 3 sessions (Jul 1 +211K, Jul 6 +211K = biggest buy days in dataset) — first genuine accumulation after months of selling; watch $300",
    },
    "OSUR": {
        "bias":       "Distribution",
        "support":    [4.20, 4.30],
        "resistance": [4.50, 4.70],
        "note":       "Jul 6 $4.37; cum delta -140.9K at a new low; distribution grinds on, no buyer emerging across the dataset; price stuck $4.30-4.50",
    },
    "INTU": {
        "bias":       "Reversal Watch",
        "support":    [264.0, 268.0],
        "resistance": [276.0, 282.0],
        "note":       "Jul 1 $267 on +109K, Jul 6 $272; accumulation intact +1.14M; recovered off the $255 bottom; Jun 22-Jul 1 buy cluster (5 strong A signals) confirming the floor; thesis alive above $264",
    },
    "NVO": {
        "bias":       "Distribution",
        "support":    [47.00, 48.00],
        "resistance": [50.00, 52.00],
        "note":       "Jul 2 $50.43, Jul 6 $49.26; +35% rally on net selling (cum delta -135.3K) = fragile rally; huge-volume name, distribution persists under a rising price",
    },
    "CME": {
        "bias":       "Distribution",
        "support":    [228.00, 232.00],
        "resistance": [240.00, 248.00],
        "note":       "Jul 6 $234.64; cum delta -1.92M at a new low (Jun 17 -450K single-day sell still the record); relentless markdown $296->$220s; every absorption signal fails; deepest distribution in the dataset",
    },
    "MBC": {
        "bias":       "Squeeze Candidate",
        "support":    [9.00, 9.20],
        "resistance": [9.90, 10.30],
        "note":       "Jul 1 $10.05 then faded to $9.43 by Jul 6; cum delta -40.8K (off the -111K May low); range-bound $8.60-10.30; SPI/squeeze setup intact but no ignition, liquidity-capped",
    },
    "AI": {
        "bias":       "Squeeze Candidate",
        "support":    [8.75, 9.00],
        "resistance": [10.30, 11.30],
        "note":       "Jul 6 $9.25; faded from the $12.83 Jun 3 blow-off; cum delta +31.6K collapsed from the +152K Jun 18 high — momentum bled out; broke below $10.20; squeeze fuel drained for now",
    },
}

# High-conviction absorption signals: score ≥ 50 or large next-day follow-through
SIGNALS = {
    # score ≥ 50, or below 50 with next-day follow-through > 5%  (extended thru Jul 6 2026)
    "FDS":  ["2026-04-07",                                           # 64.4 A
             "2026-04-22",                                           # 68.5 A
             "2026-04-24",                                           # 52.3 B
             "2026-04-27",                                           # 71.0 A
             "2026-04-28",                                           # 58.8 B
             "2026-04-29",                                           # 57.1 B
             "2026-04-30",                                           # 61.6 A+B
             "2026-05-06",                                           # 71.8 A → +5.81% next day
             "2026-05-13",                                           # 62.4 B
             "2026-05-14",                                           # 47.7 B → +6.60% next day
             "2026-05-15",                                           # 49.2 A → +5.54% next day
             "2026-05-18",                                           # 77.9 B (highest in dataset)
             "2026-05-20",                                           # 52.7 A
             "2026-05-29",                                           # 62.0 B (passive absorption at $242 breakout)
             "2026-06-08",                                          # 88.9 A — HIGHEST in dataset, but → -0.15% next day; no follow-through, FDS dormant at $246
             "2026-06-11",                                          # 79.5 A — 2nd strong buyer in 4 sessions at the lows; persistent pre-earnings accumulation (same-day -2.34%, drift to $236)
             "2026-06-16",                                          # 59.8 A — 3rd active buyer in 9 sessions at $236-246; persistent accumulation into July 1 (Jun 12 +13K gave first price response)
             "2026-06-18",                                          # 76.7 A at the $221 lows — last absorption signal before the Jun 26 +33K rip; accumulation resolved into ignition
             "2026-06-29",                                           # 60.2 B, +1.05% same-day — buyer still absorbing the day before July 1 earnings; firmed into the print
             "2026-06-30",                                           # 65.6 B → +6.68% next day (the July 1 earnings rip)
             "2026-07-06"],                                          # 52.1 A+B at $252 recovery high; buyers back but cum delta still -271.8K
    "VIRT": ["2026-04-01",                                           # 51.7 A → +3.74% next day
             "2026-04-07",                                           # 38.1 B → +2.94% next day
             "2026-05-15",                                           # 93.9 A (highest score in VIRT dataset)
             "2026-05-18",                                           # 50.4 A
             "2026-06-05",                                           # 50.4 B → +3.59% next day
             "2026-06-09",                                           # 46.7 B → +3.56% next day; passive buyers working, bounce confirming
             "2026-06-22",                                           # 75.5 A — breakout-defense cluster begins
             "2026-06-24",                                           # 69.2 A
             "2026-06-25"],                                          # 59.0 A — 3 active signals defending the $60-63 breakout consolidation
    "WRB":  ["2026-04-17",                                           # 52.9 A
             "2026-04-22",                                           # 50.9 B
             "2026-04-24",                                           # 63.8 A
             "2026-04-28",                                           # 66.5 A
             "2026-05-07",                                           # 51.8 A
             "2026-05-08",                                           # 77.4 A
             "2026-05-19",                                           # 50.2 A
             "2026-05-20",                                           # 57.4 B
             "2026-05-21",                                           # 55.2 A
             "2026-05-22",                                           # 54.8 A
             "2026-05-28",                                           # 53.5 A
             "2026-05-29",                                           # 53.4 A (largest single-day buy vol at breakdown)
             "2026-06-03",                                           # 54.3 A+B
             "2026-06-04",                                           # 50.3 A — 20 active signals yet cum delta at new low (-33K): failed-absorption tilt
             "2026-06-25",                                           # 52.6 A → +2.00% next day
             "2026-06-30"],                                          # 51.4 A — buyer present but +8% rally still on net selling (cum delta -49.9K)
    "RPM":  ["2026-04-01",                                           # 50.8 A
             "2026-04-06",                                           # 63.7 A
             "2026-04-07",                                           # 41.2 A → +12.49% next day
             "2026-04-13",                                           # 70.5 A
             "2026-04-24",                                           # 52.1 B
             "2026-05-08",                                           # 70.5 A
             "2026-05-11",                                           # 57.9 B
             "2026-05-14",                                           # 60.6 A+B
             "2026-05-18",                                           # 55.8 A
             "2026-05-19",                                           # 66.1 A
             "2026-05-26",                                           # 58.9 B
             "2026-05-29",                                           # 57.3 A
             "2026-06-10",                                           # 63.2 A → +0.40% next; buyer back at $104
             "2026-06-15",                                           # 70.8 B — strong, but cum delta at new low -78.7K = RPM joining failed-absorption camp
             "2026-06-17",                                           # 53.0 A → -2.70% same-day; buyer present, sellers winning
             "2026-06-18",                                           # 51.1 A at the lows
             "2026-06-22",                                           # 54.4 A — buy cluster ahead of the delta flip
             "2026-07-01",                                           # 50.0 A
             "2026-07-02"],                                          # 68.9 A — cum delta flips +13.3K; accumulation reversal confirmed
    "GDDY": ["2026-04-08",                                           # 58.8 A
             "2026-04-29",                                           # 56.4 A
             "2026-05-06",                                           # 78.7 A
             "2026-05-07",                                           # 59.5 A
             "2026-05-13",                                           # 51.1 B
             "2026-05-15",                                           # 63.5 B
             "2026-05-18",                                           # 57.6 A+B
             "2026-05-21",                                           # 60.6 A
             "2026-05-26",                                           # 52.2 B
             "2026-06-02",                                           # 66.5 A (buyer after earnings gap; failed → -6.25% next day)
             "2026-06-08",                                           # 54.0 B (passive buyer into breakdown; failed-absorption continues)
             "2026-06-09",                                          # 59.3 A at $80.73 lows — another buyer attempt
             "2026-06-11",                                           # 75.4 A — strong buyer at the lows; same buyer-at-the-bottom cluster as IBM; watch for a turn
             "2026-06-25",                                           # 47.3 B → +5.38% next day
             "2026-07-01"],                                          # 58.9 A+B — buyer into the bounce but cum delta -369.5K at new low: failed-absorption continues
    "IBM":  ["2026-04-01",                                           # 69.0 A
             "2026-04-02",                                           # 55.4 A
             "2026-04-07",                                           # 84.2 A
             "2026-04-08",                                           # 84.6 A+B
             "2026-04-13",                                           # 57.8 A
             "2026-04-16",                                           # 58.6 B
             "2026-04-22",                                           # 76.3 A+B
             "2026-04-27",                                           # 61.9 A
             "2026-05-11",                                           # 66.7 A
             "2026-05-14",                                           # 66.9 A
             "2026-05-18",                                           # 56.6 B
             "2026-05-21",                                           # 44.8 A → +5.05% next day
             "2026-05-22",                                           # 55.1 A
             "2026-05-26",                                           # 59.8 A
             "2026-05-28",                                           # 45.7 B → +13.25% next day (pre-catalyst signal)
             "2026-06-01",                                           # 45.6 B → +2.81% next day
             "2026-06-05",                                           # 52.4 A → -1.62% next day (even active signals fail now)
             "2026-06-09",                                          # 79.8 A at $277 lows — strongest since April; possible bottom-buyer but IBM's signals have all failed
             "2026-06-10",                                          # 81.0 A → +1.16% next; strongest IBM signal in the recent cluster; 2 days of heavy buying at $277-280
             "2026-06-11",                                          # 64.3 B
             "2026-06-12",                                          # 60.9 B — buyer cluster building at the lows after -1.8M
             "2026-06-16",                                           # 71.0 B → -2.55% next; cluster RESOLVED as failed-absorption — 5 strong signals Jun 9-16, all failed, price kept bleeding to $263
             "2026-06-26"],                                          # 57.2 B → +2.39% next; the buyer that started the +630K, 3-session accumulation into $299
    "OSUR": ["2026-04-07",                                           # 59.7 A
             "2026-05-18",                                           # 61.2 A → +2.32% next day (confirmed in Jun 1 run)
             "2026-06-08",                                          # 58.9 B → +2.79% next day (rare working signal; passive buyer at $4.15)
             "2026-06-11",                                           # 66.2 B → +0.84% next; passive buyer holding $4
             "2026-07-02"],                                          # 59.6 A → -0.68% next; lone active buyer, distribution unbroken (-140.9K)
    "INTU": ["2026-04-01",                                           # 61.2 A
             "2026-04-08",                                           # 70.1 A
             "2026-04-09",                                           # 71.6 A
             "2026-04-14",                                           # 54.8 A
             "2026-04-22",                                           # 75.5 A
             "2026-04-23",                                           # 62.6 B
             "2026-05-01",                                           # 59.9 B
             "2026-05-04",                                           # 85.3 A
             "2026-05-13",                                           # 80.3 A
             "2026-05-14",                                           # 78.5 A → +3.85% next day
             "2026-05-20",                                           # 69.9 A (pre-earnings gap; signal failed)
             "2026-06-02",                                          # 72.8 A (09:00–10:00; failed → -2.25% next day, pullback to $307)
             "2026-06-04",                                          # 76.7 A (12:00–15:00; floor defense +194K — highest since mid-May, confirms accumulation)
             "2026-06-10",                                           # 52.7 A → -1.78% next; lone weak signal, big buyer faded after the failed reclaim
             "2026-06-22",                                           # 65.2 A → +1.43% next
             "2026-06-23",                                           # 82.3 A — strongest INTU signal in the recovery; +224K floor-defense day
             "2026-06-24",                                           # 59.5 A
             "2026-06-25",                                           # 71.5 A → +5.14% next day
             "2026-07-01"],                                          # 78.3 A → +3.11% next; accumulation +1.14M holding, thesis alive above $264
    "NVO":  ["2026-04-30"],                                          # 53.1 A → +3.72% next day
    "CME":  ["2026-04-01",                                           # 79.0 A+B
             "2026-04-02",                                           # 52.3 B
             "2026-04-13",                                           # 54.8 B
             "2026-04-16",                                           # 60.7 A
             "2026-04-17",                                           # 80.8 A
             "2026-04-21",                                           # 59.2 A+B
             "2026-04-22",                                           # 56.9 B
             "2026-04-24",                                           # 74.4 B
             "2026-04-28",                                           # 67.7 B
             "2026-04-30",                                           # 56.1 B
             "2026-05-04",                                           # 64.1 A+B
             "2026-05-05",                                           # 52.2 B
             "2026-05-07",                                           # 52.1 A
             "2026-05-08",                                           # 54.8 A
             "2026-05-18",                                           # 53.6 B
             "2026-05-22",                                           # 52.9 B → -3.16% next day (failed)
             "2026-05-28",                                           # 60.3 B → -1.91% next day (failed); markdown continued
             "2026-06-04",                                           # 61.6 B
             "2026-06-05",                                           # 64.9 B → -2.22% next (failed)
             "2026-06-08",                                           # 61.6 B
             "2026-06-10",                                           # 55.0 B
             "2026-06-12",                                           # 59.8 B
             "2026-06-17",                                           # 70.3 B — same day as the -450K sell record; absorbed flat then kept bleeding
             "2026-06-18",                                           # 61.3 A
             "2026-06-23",                                           # 67.9 A+B → -5.68% next (failed hard)
             "2026-06-25",                                           # 52.6 B → -3.08% next (failed)
             "2026-06-26"],                                          # 64.9 A → -1.67% next; every CME signal fails, deepest distribution in dataset (-1.92M)
    "MBC":  ["2026-04-29",                                           # 50.6 A
             "2026-05-07",                                           # 65.5 A
             "2026-05-11",                                           # 73.7 A (highest; nailed the $6.99 bottom)
             "2026-05-12",                                           # 50.3 A → preceded the +33% rip on a lag
             "2026-06-08",                                           # 50.6 A
             "2026-07-02"],                                          # 55.6 A → -4.26% next; range-bound $8.60-10.30, no ignition
    "AI":   ["2026-05-28"],                                          # 26.3 A → +5.48% next day; ONLY notable signal — near-absence is the momentum/squeeze fingerprint (no tape accumulation)
}

BIAS_COLOR = {
    "Squeeze Candidate":    "#e67e22",
    "Squeeze Active":       "#f1c40f",
    "Distribution":         "#c0392b",
    "Stealth Accumulation": "#27ae60",
    "Accumulation Watch":   "#2ecc71",
    "Bearish":              "#922b21",
    "Passive Accumulation": "#2980b9",
    "Reversal Watch":       "#8e44ad",
    "Breakout Building":    "#1abc9c",
    "Post-Earnings Floor":  "#d35400",
    "Post-Breakout Base":   "#16a085",
    "Confirmed Reversal":   "#229954",
}

DEFAULT_SINCE = "2026-02-01"


# ── Data fetchers ─────────────────────────────────────────────────────────────

def _fetch_price(sym: str, since: str) -> pd.Series:
    import yfinance as yf
    hist = yf.download(sym, start=since, progress=False, auto_adjust=True)
    if hist.empty:
        return pd.Series(dtype=float)
    if isinstance(hist.columns, pd.MultiIndex):
        return hist["Close"][sym].dropna()
    return hist["Close"].dropna()


def _fetch_cum_delta(sym: str, since: str, end: str) -> pd.Series:
    """
    Load daily cumulative delta from IEX DuckDB using Lee-Ready classification.
    Returns a pd.Series indexed by date, values = cumulative net shares.
    Returns empty Series if DB unavailable or no data for symbol.
    """
    if not os.path.exists(DB_PATH):
        return pd.Series(dtype=float)
    try:
        # Import get_daily_delta without letting its sys.exit propagate
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "cumulative_delta", os.path.join(BASE_DIR, "cumulative_delta.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        conn = mod._connect()
        df   = mod.get_daily_delta(conn, sym, since, end)
        conn.close()
        return df["cum_delta"].rename(sym)
    except SystemExit:
        return pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)


# ── Panel drawing ─────────────────────────────────────────────────────────────

def _plot_panel(ax, sym: str, close: pd.Series, cum_delta: pd.Series) -> None:
    import matplotlib.dates as mdates

    lvl        = LEVELS.get(sym, {})
    bias       = lvl.get("bias", "neutral")
    price_color = BIAS_COLOR.get(bias, "#555555")
    price_range = close.max() - close.min()
    x_end       = close.index[-1]

    # ── Price line (left axis) ────────────────────────────────────────────────
    ax.plot(close.index, close.values, color=price_color, linewidth=1.4, zorder=3)

    # ── Cumulative delta overlay (right twin axis) ────────────────────────────
    if cum_delta is not None and not cum_delta.empty:
        ax2 = ax.twinx()
        cd  = cum_delta.copy()
        cd.index = pd.to_datetime(cd.index)

        # Shade positive (net buying) green, negative (net selling) red
        ax2.fill_between(cd.index, cd.values, 0,
                         where=(cd.values >= 0), alpha=0.13, color="#27ae60", zorder=1)
        ax2.fill_between(cd.index, cd.values, 0,
                         where=(cd.values < 0),  alpha=0.13, color="#c0392b", zorder=1)
        ax2.plot(cd.index, cd.values, color="#555555", linewidth=0.9,
                 linestyle="--", alpha=0.75, zorder=2)
        ax2.axhline(0, color="#333333", linewidth=0.5, linestyle=":", zorder=2)

        # Label the most recent delta value
        last_cd = cd.iloc[-1]
        ax2.text(cd.index[-1], last_cd,
                 f" {last_cd/1e3:+.0f}K shares", fontsize=6, color="#555555",
                 va="center", ha="left")

        ax2.set_ylabel("Cumulative Delta (shares)", fontsize=6.5, color="#888888")
        ax2.tick_params(axis="y", labelsize=6, labelcolor="#888888")
        ax2.spines["top"].set_visible(False)
        # Keep delta axis from visually dominating — pad it symmetrically
        abs_max = max(abs(cd.max()), abs(cd.min())) * 1.6
        ax2.set_ylim(-abs_max, abs_max)

    # ── Support lines ────────────────────────────────────────────────────────
    for s in lvl.get("support", []):
        ax.axhline(s, color="#27ae60", linewidth=1.0, linestyle="--", alpha=0.80, zorder=4)
        ax.text(x_end, s, f" ${s:.2f}", va="bottom", ha="left", fontsize=7,
                color="#1e8449", fontweight="bold")

    # ── Resistance lines ─────────────────────────────────────────────────────
    for r in lvl.get("resistance", []):
        ax.axhline(r, color="#c0392b", linewidth=1.0, linestyle="--", alpha=0.80, zorder=4)
        ax.text(x_end, r, f" ${r:.2f}", va="top", ha="left", fontsize=7,
                color="#922b21", fontweight="bold")

    # ── Y-axis bounds to show all levels ─────────────────────────────────────
    supports    = lvl.get("support", [])
    resistances = lvl.get("resistance", [])
    if supports and resistances:
        lo = min(supports)    - price_range * 0.04
        hi = max(resistances) + price_range * 0.04
        ax.set_ylim(
            min(close.min() - price_range * 0.05, lo),
            max(close.max() + price_range * 0.05, hi),
        )

    # ── Absorption signal markers ────────────────────────────────────────────
    for ds in SIGNALS.get(sym, []):
        sig_ts = pd.Timestamp(ds)
        if sig_ts not in close.index:
            candidates = close.index[close.index >= sig_ts]
            if candidates.empty:
                continue
            sig_ts = candidates[0]
        price_at = close.loc[sig_ts]
        ax.scatter([sig_ts], [price_at + price_range * 0.025],
                   marker="v", color="purple", s=40, zorder=6, alpha=0.80)

    # ── Current price dot + label ────────────────────────────────────────────
    last_p = close.iloc[-1]
    last_t = close.index[-1]
    ax.scatter([last_t], [last_p], color=price_color, s=55, zorder=7)
    ax.text(last_t, last_p - price_range * 0.04, f"${last_p:.2f}",
            ha="center", va="top", fontsize=8, color=price_color, fontweight="bold")

    # ── Title and annotation ─────────────────────────────────────────────────
    note = lvl.get("note", "").replace("$", r"\$")
    ax.set_title(f"{sym}  ·  {bias}", fontsize=9, color=price_color,
                 fontweight="bold", pad=4)
    ax.text(0.01, 0.02, note, transform=ax.transAxes, fontsize=6.5,
            color="#666666", va="bottom")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=3))
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.grid(True, alpha=0.18, linewidth=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("Price ($)", fontsize=7.5)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Watchlist S/R levels chart with cumulative delta overlay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("symbols", nargs="*",
                        help="Ticker symbols (default: VIRT WRB RPM FDS GDDY IBM)")
    parser.add_argument("--since", default=DEFAULT_SINCE, metavar="DATE",
                        help=f"Price history start date (default: {DEFAULT_SINCE})")
    parser.add_argument("--delta-since", default="2026-04-01", metavar="DATE",
                        help="Cumulative delta start date (default: 2026-04-01)")
    parser.add_argument("--delta-end", default="2026-07-06", metavar="DATE",
                        help="Cumulative delta end date (default: 2026-06-27)")
    parser.add_argument("--no-delta", action="store_true",
                        help="Omit cumulative delta overlay")
    parser.add_argument("--save", action="store_true",
                        help="Save PNG without displaying")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.lines as mlines
    except ImportError:
        sys.exit("matplotlib not installed — run: pip install matplotlib")

    symbols = [s.upper() for s in args.symbols] if args.symbols else WATCHLIST
    n     = len(symbols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    flat_axes = axes.flatten() if n > 1 else [axes]

    delta_note = "" if args.no_delta else "  ·  Shaded fill = cumulative delta (IEX Lee-Ready)"
    fig.suptitle(
        f"Watchlist — Key Support / Resistance Levels{delta_note}\n"
        "Levels from IEX cumulative delta + absorption analysis (updated Jul 7 2026, data thru Jul 6)",
        fontsize=10, y=1.01,
    )

    print(f"Fetching price history from Yahoo Finance ({args.since} → today) ...")
    if not args.no_delta:
        if os.path.exists(DB_PATH):
            print(f"Loading cumulative delta from IEX DB "
                  f"({args.delta_since} → {args.delta_end}) ...")
        else:
            print(f"  [warn] IEX database not found at {DB_PATH} — delta overlay skipped")

    for i, sym in enumerate(symbols):
        ax    = flat_axes[i]
        close = _fetch_price(sym, args.since)
        if close.empty:
            ax.text(0.5, 0.5, f"{sym}\nNo data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="grey")
            continue

        cum_delta = None
        if not args.no_delta:
            cum_delta = _fetch_cum_delta(sym, args.delta_since, args.delta_end)
            if cum_delta.empty:
                print(f"  [{sym}] no delta data in DB for this range")

        _plot_panel(ax, sym, close, cum_delta)

    for j in range(i + 1, len(flat_axes)):
        flat_axes[j].set_visible(False)

    # ── Legend ────────────────────────────────────────────────────────────────
    leg_handles = [
        mpatches.Patch(facecolor="#27ae60", alpha=0.7, label="Support level"),
        mpatches.Patch(facecolor="#c0392b", alpha=0.7, label="Resistance level"),
        mlines.Line2D([0], [0], marker="v", color="w", markerfacecolor="purple",
                      markersize=9, label="Absorption signal"),
    ]
    if not args.no_delta:
        leg_handles += [
            mpatches.Patch(facecolor="#27ae60", alpha=0.35, label="Cum Δ > 0 (net buying)"),
            mpatches.Patch(facecolor="#c0392b", alpha=0.35, label="Cum Δ < 0 (net selling)"),
        ]
    fig.legend(handles=leg_handles, loc="lower center",
               ncol=len(leg_handles), fontsize=8,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9)

    plt.tight_layout()
    out = "watchlist_levels.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    if not args.save:
        plt.show()


if __name__ == "__main__":
    main()
