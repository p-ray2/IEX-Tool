"""
Cumulative delta (net order flow) analysis for IEX trade data.

Computes daily and cumulative buy_vol - sell_vol using quote-midpoint
classification (Lee-Ready simplified), then overlays against price.

Patterns to look for:
  Price up   + delta up    → confirmed trend
  Price up   + delta flat  → fragile move, buying came from elsewhere
  Price flat + delta up    → absorption — buyers not moving price
  Price down + delta up    → buyers fighting the move, potential support

Usage:
    python cumulative_delta.py AAPL --start 2026-04-01 --end 2026-04-30
    python cumulative_delta.py AAPL --start 2026-04-01 --end 2026-04-30 --plot
    python cumulative_delta.py AAPL --start 2026-04-01 --end 2026-04-30 --intraday 2026-04-17
    python cumulative_delta.py AAPL --start 2026-04-01 --end 2026-04-30 --hourly
    python cumulative_delta.py AAPL --start 2026-04-01 --end 2026-04-30 --hourly --min-imbalance 25
    python cumulative_delta.py AAPL --start 2026-04-01 --end 2026-04-30 --tod-profile
"""

import argparse
import os
import sys

import numpy as np
import duckdb
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "cache", "market_data.duckdb")


def _connect() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(DB_PATH, read_only=True)


def get_daily_delta(conn, symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    For each trading day: total buy_vol, sell_vol, net delta, open price, close price.
    Uses quote-midpoint classification; tick test as tiebreaker at midpoint.
    """
    trades = conn.execute("""
        SELECT trade_date, timestamp_ns, size, price
        FROM trades
        WHERE symbol = ? AND trade_date BETWEEN ? AND ?
          AND is_extended = false AND is_odd_lot = false
        ORDER BY trade_date, timestamp_ns
    """, [symbol, start, end]).df()

    if trades.empty:
        sys.exit(f"No trade data found for {symbol} between {start} and {end}.")

    quotes = conn.execute("""
        SELECT trade_date, timestamp_ns, mid_price
        FROM quotes
        WHERE symbol = ? AND trade_date BETWEEN ? AND ?
        ORDER BY trade_date, timestamp_ns
    """, [symbol, start, end]).df()

    results = []
    for date, day_trades in trades.groupby("trade_date"):
        day_trades = day_trades.sort_values("timestamp_ns").reset_index(drop=True)
        day_quotes = quotes[quotes["trade_date"] == date].sort_values("timestamp_ns")

        if not day_quotes.empty:
            merged = pd.merge_asof(
                day_trades, day_quotes[["timestamp_ns", "mid_price"]],
                on="timestamp_ns", direction="backward"
            )
        else:
            merged = day_trades.copy()
            merged["mid_price"] = float("nan")

        merged["prev_price"] = merged["price"].shift(1)

        buy_mid   = merged["price"] > merged["mid_price"]
        sell_mid  = merged["price"] < merged["mid_price"]
        at_mid    = merged["price"] == merged["mid_price"]
        no_mid    = merged["mid_price"].isna()
        buy_tick  = merged["price"] > merged["prev_price"]
        sell_tick = merged["price"] < merged["prev_price"]

        merged["buy_vol"]  = np.where(buy_mid  | ((at_mid | no_mid) & buy_tick),  merged["size"], 0)
        merged["sell_vol"] = np.where(sell_mid | ((at_mid | no_mid) & sell_tick), merged["size"], 0)

        results.append({
            "date":        pd.Timestamp(date),
            "buy_vol":     merged["buy_vol"].sum(),
            "sell_vol":    merged["sell_vol"].sum(),
            "total_vol":   merged["size"].sum(),
            "open_price":  merged["price"].iloc[0],
            "close_price": merged["price"].iloc[-1],
            "trade_count": len(merged),
        })

    df = pd.DataFrame(results).set_index("date")
    df["net_delta"]  = df["buy_vol"] - df["sell_vol"]
    df["cum_delta"]  = df["net_delta"].cumsum()
    df["price_chg"]  = df["close_price"] - df["open_price"].iloc[0]   # vs. first open
    df["delta_pct"]  = df["net_delta"] / df["total_vol"] * 100        # net as % of volume
    return df


def get_intraday_delta(conn, symbol: str, date: str) -> pd.DataFrame:
    """
    Intraday cumulative delta at 1-hour resolution for a single date.
    """
    trades = conn.execute("""
        SELECT timestamp_ns, size, price
        FROM trades
        WHERE symbol = ? AND trade_date = ?
          AND is_extended = false AND is_odd_lot = false
        ORDER BY timestamp_ns
    """, [symbol, date]).df()

    if trades.empty:
        sys.exit(f"No trade data for {symbol} on {date}.")

    quotes = conn.execute("""
        SELECT timestamp_ns, mid_price
        FROM quotes
        WHERE symbol = ? AND trade_date = ?
        ORDER BY timestamp_ns
    """, [symbol, date]).df()

    if not quotes.empty:
        merged = pd.merge_asof(trades, quotes, on="timestamp_ns", direction="backward")
    else:
        merged = trades.copy()
        merged["mid_price"] = float("nan")

    merged["prev_price"] = merged["price"].shift(1)

    buy_mid   = merged["price"] > merged["mid_price"]
    sell_mid  = merged["price"] < merged["mid_price"]
    at_mid    = merged["price"] == merged["mid_price"]
    no_mid    = merged["mid_price"].isna()
    buy_tick  = merged["price"] > merged["prev_price"]
    sell_tick = merged["price"] < merged["prev_price"]

    merged["buy_vol"]  = np.where(buy_mid  | ((at_mid | no_mid) & buy_tick),  merged["size"], 0)
    merged["sell_vol"] = np.where(sell_mid | ((at_mid | no_mid) & sell_tick), merged["size"], 0)
    merged["net"]      = merged["buy_vol"] - merged["sell_vol"]

    merged["hour_et"] = (
        pd.to_datetime(merged["timestamp_ns"], unit="ns", utc=True)
        .dt.tz_convert("US/Eastern")
        .dt.floor("h")
        .dt.tz_localize(None)
    )
    merged = merged[
        (merged["hour_et"].dt.hour >= 9) &
        (merged["hour_et"].dt.hour < 16)
    ]

    agg = merged.groupby("hour_et").agg(
        buy_vol    =("buy_vol",  "sum"),
        sell_vol   =("sell_vol", "sum"),
        total_vol  =("size",     "sum"),
        open_price =("price",    "first"),
        close_price=("price",    "last"),
    )
    agg["net_delta"] = agg["buy_vol"] - agg["sell_vol"]
    agg["cum_delta"] = agg["net_delta"].cumsum()
    agg["delta_pct"] = agg["net_delta"] / agg["total_vol"] * 100
    return agg


def get_hourly_delta(conn, symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Multi-day hourly delta across the full date range.
    Returns a DataFrame indexed by ET datetime (one row per trading hour per day)
    with a running cumulative delta that does NOT reset between days.
    """
    trades = conn.execute("""
        SELECT trade_date, timestamp_ns, size, price
        FROM trades
        WHERE symbol = ? AND trade_date BETWEEN ? AND ?
          AND is_extended = false AND is_odd_lot = false
        ORDER BY trade_date, timestamp_ns
    """, [symbol, start, end]).df()

    if trades.empty:
        sys.exit(f"No trade data found for {symbol} between {start} and {end}.")

    quotes = conn.execute("""
        SELECT trade_date, timestamp_ns, mid_price
        FROM quotes
        WHERE symbol = ? AND trade_date BETWEEN ? AND ?
        ORDER BY trade_date, timestamp_ns
    """, [symbol, start, end]).df()

    # Lee-Ready classification per day (preserves tick-test boundary correctness)
    day_frames = []
    for date, day_trades in trades.groupby("trade_date"):
        day_trades = day_trades.sort_values("timestamp_ns").reset_index(drop=True)
        day_quotes = quotes[quotes["trade_date"] == date].sort_values("timestamp_ns")

        if not day_quotes.empty:
            merged = pd.merge_asof(
                day_trades, day_quotes[["timestamp_ns", "mid_price"]],
                on="timestamp_ns", direction="backward"
            )
        else:
            merged = day_trades.copy()
            merged["mid_price"] = float("nan")

        merged["prev_price"] = merged["price"].shift(1)

        buy_mid   = merged["price"] > merged["mid_price"]
        sell_mid  = merged["price"] < merged["mid_price"]
        at_mid    = merged["price"] == merged["mid_price"]
        no_mid    = merged["mid_price"].isna()
        buy_tick  = merged["price"] > merged["prev_price"]
        sell_tick = merged["price"] < merged["prev_price"]

        merged["buy_vol"]  = np.where(buy_mid  | ((at_mid | no_mid) & buy_tick),  merged["size"], 0)
        merged["sell_vol"] = np.where(sell_mid | ((at_mid | no_mid) & sell_tick), merged["size"], 0)
        day_frames.append(merged)

    all_trades = pd.concat(day_frames).reset_index(drop=True)

    all_trades["datetime_et"] = (
        pd.to_datetime(all_trades["timestamp_ns"], unit="ns", utc=True)
        .dt.tz_convert("US/Eastern")
        .dt.floor("h")
        .dt.tz_localize(None)
    )
    all_trades = all_trades[
        (all_trades["datetime_et"].dt.hour >= 9) &
        (all_trades["datetime_et"].dt.hour < 16)
    ]

    agg = all_trades.groupby("datetime_et").agg(
        buy_vol     =("buy_vol",  "sum"),
        sell_vol    =("sell_vol", "sum"),
        total_vol   =("size",     "sum"),
        open_price  =("price",    "first"),
        close_price =("price",    "last"),
    )
    agg["net_delta"] = agg["buy_vol"] - agg["sell_vol"]
    agg["cum_delta"] = agg["net_delta"].cumsum()
    agg["delta_pct"] = agg["net_delta"] / agg["total_vol"] * 100
    return agg


def get_tod_profile(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly_df by hour-of-day to surface systematic intraday patterns.
    Shows which hours consistently lean buy-side or sell-side across all days.
    """
    h = hourly_df.copy()
    h["hour"]      = h.index.hour
    h["price_chg"] = h["close_price"] - h["open_price"]

    rows = []
    for hour, g in h.groupby("hour"):
        rows.append({
            "hour":          hour,
            "days":          len(g),
            "avg_net":       g["net_delta"].mean(),
            "avg_pct":       g["delta_pct"].mean(),
            "pct_bullish":   (g["net_delta"] > 0).mean() * 100,
            "avg_vol":       g["total_vol"].mean(),
            "avg_price_chg": g["price_chg"].mean(),
            "max_net":       g["net_delta"].max(),
            "min_net":       g["net_delta"].min(),
        })
    return pd.DataFrame(rows).set_index("hour")


def print_daily_table(df: pd.DataFrame) -> None:
    out = df[["open_price", "close_price", "buy_vol", "sell_vol",
              "net_delta", "delta_pct", "cum_delta"]].copy()
    out.index = out.index.strftime("%Y-%m-%d")
    out.columns = ["Open", "Close", "Buy Vol", "Sell Vol",
                   "Net Delta", "Delta %", "Cum Delta"]
    print(out.to_string(
        float_format=lambda x: f"{x:,.0f}",
        formatters={"Open":  lambda x: f"{x:.2f}",
                    "Close": lambda x: f"{x:.2f}",
                    "Delta %": lambda x: f"{x:+.1f}%"}
    ))
    print()


def print_intraday_table(df: pd.DataFrame, date: str) -> None:
    print(f"\nIntraday delta — {date}")
    print("-" * 65)
    out = df[["open_price", "close_price", "buy_vol", "sell_vol",
              "net_delta", "delta_pct", "cum_delta"]].copy()
    out.index = out.index.strftime("%H:%M ET")
    out.columns = ["Open", "Close", "Buy Vol", "Sell Vol",
                   "Net Delta", "Delta %", "Cum Delta"]
    print(out.to_string(
        float_format=lambda x: f"{x:,.0f}",
        formatters={"Open":  lambda x: f"{x:.2f}",
                    "Close": lambda x: f"{x:.2f}",
                    "Delta %": lambda x: f"{x:+.1f}%"}
    ))
    print()


def print_hourly_table(df: pd.DataFrame, symbol: str, min_imbalance: float = 0.0) -> None:
    """Print multi-day hourly table grouped by date, optionally filtered by |delta_pct|."""
    W = 82
    filtered = df[df["delta_pct"].abs() >= min_imbalance] if min_imbalance > 0 else df
    n_shown  = len(filtered)
    n_total  = len(df)

    print(f"\n{'═'*W}")
    print(f"  Hourly Delta — {symbol}")
    filter_label = f"  |imbalance| ≥ {min_imbalance:.0f}%  →  {n_shown}/{n_total} hours shown" if min_imbalance > 0 else f"  {n_total} hours"
    print(filter_label)
    print(f"  {'─'*W}")
    print(f"  {'Hour':>8}  {'Open':>7}  {'Close':>7}  {'Buy Vol':>8}  {'Sell Vol':>8}  "
          f"{'Net Δ':>8}  {'Δ%':>7}  {'Cum Δ':>10}")
    print(f"  {'─'*W}")

    current_date = None
    for dt, row in filtered.iterrows():
        row_date = dt.date()
        if row_date != current_date:
            current_date = row_date
            print(f"\n  ── {row_date} {'─'*(W-14)}")
        bar = "▲" if row["net_delta"] >= 0 else "▼"
        print(
            f"  {dt.strftime('%H:%M ET'):>8}  "
            f"${row['open_price']:>6.2f}  ${row['close_price']:>6.2f}  "
            f"{row['buy_vol']:>8,.0f}  {row['sell_vol']:>8,.0f}  "
            f"{row['net_delta']:>+8,.0f}  {row['delta_pct']:>+6.1f}% {bar} "
            f"{row['cum_delta']:>+10,.0f}"
        )

    print(f"\n  {'─'*W}")
    print(f"  Cum Δ = running total across all days (does not reset)\n")


def print_tod_profile(df: pd.DataFrame, symbol: str, n_days: int) -> None:
    """Print time-of-day aggregated profile — which hours systematically lean buy or sell."""
    W = 80
    print(f"\n{'═'*W}")
    print(f"  Time-of-Day Profile — {symbol}  ({n_days} trading days)")
    print(f"  {'─'*W}")
    print(f"  {'Hour':>8}  {'Days':>5}  {'Avg Net':>9}  {'Avg Δ%':>7}  "
          f"{'% Bull':>7}  {'Avg Vol':>9}  {'Avg ΔPrice':>11}  {'Range':>18}")
    print(f"  {'─'*W}")

    for hour, row in df.iterrows():
        label    = f"{hour:02d}:00 ET"
        bull_bar = "█" * int(row["pct_bullish"] / 10)
        bear_bar = "░" * (10 - int(row["pct_bullish"] / 10))
        bias = "BUY " if row["avg_pct"] > 5 else ("SELL" if row["avg_pct"] < -5 else "FLAT")
        print(
            f"  {label:>8}  {row['days']:>5.0f}  {row['avg_net']:>+9,.0f}  "
            f"{row['avg_pct']:>+6.1f}%  "
            f"[{bull_bar}{bear_bar}] {row['pct_bullish']:>4.0f}%  {bias}  "
            f"{row['avg_vol']:>9,.0f}  {row['avg_price_chg']:>+10.3f}  "
            f"[{row['min_net']:>+8,.0f} … {row['max_net']:>+8,.0f}]"
        )

    print(f"  {'─'*W}")
    print(f"  % Bull = fraction of days that hour had net buying  |  █ = bullish  ░ = bearish")
    print(f"  Bias: BUY = avg Δ% > +5%  SELL = avg Δ% < -5%  FLAT = in between\n")


def print_summary(df: pd.DataFrame, symbol: str) -> None:
    total_buy  = df["buy_vol"].sum()
    total_sell = df["sell_vol"].sum()
    total_net  = df["net_delta"].sum()
    price_move = df["close_price"].iloc[-1] - df["open_price"].iloc[0]
    price_pct  = price_move / df["open_price"].iloc[0] * 100

    # divergence: sign of cumulative delta vs sign of price move
    delta_dir = "net buying" if total_net > 0 else "net selling"
    price_dir = "up" if price_move > 0 else "down"
    if (total_net > 0 and price_move > 0) or (total_net < 0 and price_move < 0):
        pattern = "confirmed trend (delta and price agree)"
    elif total_net > 0 and abs(price_pct) < 0.5:
        pattern = "*** ABSORPTION candidate — net buying with flat price"
    elif total_net > 0 and price_move < 0:
        pattern = "buyers fighting the move — potential support"
    elif total_net < 0 and price_move > 0:
        pattern = "fragile rally — price up on net selling"
    else:
        pattern = "net selling with falling price"

    print(f"\n{'='*65}")
    print(f"  Cumulative Delta Summary: {symbol}")
    print(f"{'='*65}")
    print(f"  Period buy volume  : {total_buy:>12,.0f} shares")
    print(f"  Period sell volume : {total_sell:>12,.0f} shares")
    print(f"  Net delta          : {total_net:>+12,.0f} shares  ({delta_dir})")
    print(f"  Net as % of volume : {total_net / (total_buy + total_sell) * 100:>+.1f}%")
    print(f"  Price move         : {price_move:>+.2f}  ({price_pct:+.2f}%)  [{price_dir}]")
    print(f"  Pattern            : {pattern}")
    print(f"{'='*65}\n")


def plot_hourly(hourly_df: pd.DataFrame, symbol: str) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(15, 8))
    gs  = gridspec.GridSpec(2, 1, hspace=0.35)
    fig.suptitle(f"{symbol}  |  Hourly Delta vs Spot Price", fontsize=13)

    times = hourly_df.index

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(times, hourly_df["close_price"], color="steelblue", linewidth=1.2, label="Spot price")
    ax1.set_ylabel("Price ($)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax1r = ax1.twinx()
    ax1r.fill_between(times, hourly_df["cum_delta"], 0,
                      where=hourly_df["cum_delta"] >= 0, alpha=0.20, color="green")
    ax1r.fill_between(times, hourly_df["cum_delta"], 0,
                      where=hourly_df["cum_delta"] < 0,  alpha=0.20, color="red")
    ax1r.plot(times, hourly_df["cum_delta"], color="black", linewidth=1.0, linestyle="--",
              label="Cum delta")
    ax1r.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax1r.set_ylabel("Cumulative Delta", color="black")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    ax1.set_title("Spot Price vs. Running Cumulative Delta (hourly)")

    ax2 = fig.add_subplot(gs[1])
    colors = ["steelblue" if v >= 0 else "salmon" for v in hourly_df["net_delta"]]
    ax2.bar(times, hourly_df["net_delta"], color=colors, width=0.03)
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.set_ylabel("Hourly Net Delta (shares)")
    ax2.set_title("Hourly Net Order Flow")

    plt.tight_layout()
    plt.show()


def plot_daily(df: pd.DataFrame, symbol: str, intraday_df=None, intraday_date=None) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n_rows = 3 if intraday_df is not None else 2
    fig = plt.figure(figsize=(13, 4 * n_rows))
    gs  = gridspec.GridSpec(n_rows, 1, hspace=0.35)
    fig.suptitle(f"{symbol}  |  Cumulative Delta Analysis", fontsize=13)

    dates = df.index

    # Panel 1: price vs cumulative delta (dual axis)
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(dates, df["close_price"], color="steelblue", linewidth=1.8,
             marker="o", markersize=4, label="Close price")
    ax1.set_ylabel("Price ($)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax1r = ax1.twinx()
    ax1r.fill_between(dates, df["cum_delta"], 0,
                      where=df["cum_delta"] >= 0, alpha=0.25, color="green", label="Cum delta +")
    ax1r.fill_between(dates, df["cum_delta"], 0,
                      where=df["cum_delta"] < 0,  alpha=0.25, color="red",   label="Cum delta −")
    ax1r.plot(dates, df["cum_delta"], color="black", linewidth=1.2, linestyle="--")
    ax1r.set_ylabel("Cumulative Delta (shares)", color="black")
    ax1r.axhline(0, color="black", linewidth=0.6, linestyle=":")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    ax1.set_title("Price vs. Cumulative Delta")
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m/%d"))

    # Panel 2: daily net delta bars
    ax2 = fig.add_subplot(gs[1])
    colors = ["steelblue" if v >= 0 else "salmon" for v in df["net_delta"]]
    ax2.bar(dates, df["net_delta"], color=colors, width=0.6)
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.set_ylabel("Daily Net Delta (shares)")
    ax2.set_title("Daily Net Order Flow")
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m/%d"))

    # Panel 3: intraday (optional)
    if intraday_df is not None:
        ax3 = fig.add_subplot(gs[2])
        hours = intraday_df.index
        colors3 = ["steelblue" if v >= 0 else "salmon" for v in intraday_df["net_delta"]]
        ax3.bar(hours, intraday_df["net_delta"], color=colors3, width=0.04)
        ax3.axhline(0, color="black", linewidth=0.6)

        ax3r = ax3.twinx()
        ax3r.plot(hours, intraday_df["cum_delta"], color="black",
                  linewidth=1.2, linestyle="--", label="Cum delta")
        ax3r.set_ylabel("Cum Delta")

        ax3.set_ylabel("Hourly Net Delta")
        ax3.set_title(f"Intraday Delta — {intraday_date}")

    plt.tight_layout()
    plt.show()


def run(symbol: str, start: str, end: str, plot: bool, intraday_date: str,
        hourly: bool, min_imbalance: float, tod_profile: bool) -> None:
    conn = _connect()

    print(f"\nComputing cumulative delta for {symbol} ({start} → {end}) …")
    daily_df = get_daily_delta(conn, symbol, start, end)

    intraday_df = None
    if intraday_date:
        print(f"Computing intraday delta for {intraday_date} …")
        intraday_df = get_intraday_delta(conn, symbol, intraday_date)

    hourly_df = None
    if hourly or tod_profile:
        print(f"Computing hourly delta across range …")
        hourly_df = get_hourly_delta(conn, symbol, start, end)

    conn.close()

    print_summary(daily_df, symbol)
    print_daily_table(daily_df)

    if intraday_df is not None:
        print_intraday_table(intraday_df, intraday_date)

    if hourly and hourly_df is not None:
        print_hourly_table(hourly_df, symbol, min_imbalance)

    if tod_profile and hourly_df is not None:
        print_tod_profile(get_tod_profile(hourly_df), symbol, len(daily_df))

    if plot:
        if hourly_df is not None:
            plot_hourly(hourly_df, symbol)
        else:
            plot_daily(daily_df, symbol, intraday_df, intraday_date)


def main() -> None:
    parser = argparse.ArgumentParser(description="IEX cumulative delta / net order flow")
    parser.add_argument("symbol",  help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--start", required=True, metavar="YYYY-MM-DD", help="Start date")
    parser.add_argument("--end",   required=True, metavar="YYYY-MM-DD", help="End date")
    parser.add_argument("--plot",  action="store_true", help="Show charts")
    parser.add_argument("--intraday", metavar="YYYY-MM-DD",
                        help="Also show hourly intraday delta for this date")
    parser.add_argument("--hourly", action="store_true",
                        help="Show multi-day hourly delta table with running cumulative delta")
    parser.add_argument("--min-imbalance", type=float, default=0.0, dest="min_imbalance",
                        metavar="PCT",
                        help="With --hourly: only show hours where |delta%%| >= PCT (default: 0)")
    parser.add_argument("--tod-profile", action="store_true", dest="tod_profile",
                        help="Show time-of-day aggregated profile (which hours lean buy/sell)")
    args = parser.parse_args()

    if not os.path.exists(DB_PATH):
        sys.exit("No database found. Run 'python main.py ingest' first.")

    run(args.symbol.upper(), args.start, args.end, args.plot, args.intraday,
        args.hourly, args.min_imbalance, args.tod_profile)


if __name__ == "__main__":
    main()
