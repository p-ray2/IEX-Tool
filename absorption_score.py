"""
Absorption score detector for IEX trade + depth data.

MODE A — Active absorption:
  Buyers lifting offers over consecutive hours: positive trade imbalance,
  flat price, growing bid depth.

  Conditions (per window of >= MIN_CONSECUTIVE_HRS hours):
    1. Trade imbalance (buy_vol - sell_vol) / (buy_vol + sell_vol) > IMBALANCE_THRESHOLD
    2. Price movement across window < MAX_PRICE_CHANGE_PCT
    3. Bid depth growth > MIN_BID_DEPTH_GROWTH

MODE B — Passive absorption:
  Patient buyer resting on bid, absorbing sell flow: book heavily bid-side,
  trade imbalance negative (sellers hitting bids), yet price resilient.

  Conditions (per window of >= MIN_CONSECUTIVE_HRS hours):
    1. Book imbalance (bid_depth - ask_depth) / total > PASSIVE_BOOK_THRESHOLD
    2. Trade imbalance < -PASSIVE_TRADE_THRESHOLD  (net selling)
    3. Price change < MAX_PASSIVE_PRICE_CHG

Score (0–100) — MODE A:
    33 pts — trade imbalance strength above IMBALANCE_THRESHOLD
    33 pts — price flatness (how far below MAX_PRICE_CHANGE_PCT)
    34 pts — bid depth growth above MIN_BID_DEPTH_GROWTH

Score (0–100) — MODE B:
    33 pts — book imbalance strength above PASSIVE_BOOK_THRESHOLD
    33 pts — price resilience (0% = full, -MAX_PASSIVE_PRICE_CHG = zero)
    34 pts — sell intensity absorbed (how negative trade imbalance is)

Usage:
    python absorption_score.py AAPL --date 2026-04-02
    python absorption_score.py AAPL --start 2026-04-01 --end 2026-04-25
    python absorption_score.py AAPL --start 2026-04-01 --end 2026-04-25 --verbose
    python absorption_score.py AAPL --date 2026-04-02 --plot
"""

import argparse
import os
import sys

import numpy as np
import duckdb
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "cache", "market_data.duckdb")

# MODE A defaults — overridable via CLI
IMBALANCE_THRESHOLD  = 0.25
MIN_CONSECUTIVE_HRS  = 3
MAX_PRICE_CHANGE_PCT = 0.20
MIN_BID_DEPTH_GROWTH = 20.0

# MODE B defaults — overridable via CLI
PASSIVE_BOOK_THRESHOLD  = 0.25   # book imbalance must exceed this (bid-heavy)
PASSIVE_TRADE_THRESHOLD = 0.10   # trade imbalance must be below negative of this
MAX_PASSIVE_PRICE_CHG   = 2.0    # price allowed to drift more in passive mode


def _connect() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(DB_PATH, read_only=True)


def get_dates(conn, symbol: str, start: str, end: str) -> list:
    rows = conn.execute("""
        SELECT DISTINCT trade_date FROM trades
        WHERE symbol = ? AND trade_date BETWEEN ? AND ?
        ORDER BY trade_date
    """, [symbol, start, end]).fetchall()
    return [str(r[0]) for r in rows]


def get_hourly_trades(conn, symbol: str, date: str) -> pd.DataFrame:
    trades = conn.execute("""
        SELECT timestamp_ns, size, price
        FROM trades
        WHERE symbol = ? AND trade_date = ? AND is_extended = false AND is_odd_lot = false
        ORDER BY timestamp_ns
    """, [symbol, date]).df()

    if trades.empty:
        return pd.DataFrame()

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

    if merged.empty:
        return pd.DataFrame()

    agg = merged.groupby("hour_et").agg(
        buy_volume  =("buy_vol",  "sum"),
        sell_volume =("sell_vol", "sum"),
        total_volume=("size",     "sum"),
        trade_count =("size",     "count"),
        open_price  =("price",    "first"),
        close_price =("price",    "last"),
    )

    total_dir        = agg["buy_volume"] + agg["sell_volume"]
    agg["imbalance"] = (agg["buy_volume"] - agg["sell_volume"]) / total_dir.where(total_dir > 0)

    return agg


def get_hourly_bid_depth(conn, symbol: str, date: str) -> pd.DataFrame:
    df = conn.execute("""
        SELECT timestamp_ns, bid_depth, ask_depth, imbalance AS book_imb_raw
        FROM depth_snapshots
        WHERE symbol = ? AND trade_date = ?
        ORDER BY timestamp_ns
    """, [symbol, date]).df()

    if df.empty:
        return df

    df["hour_et"] = (
        pd.to_datetime(df["timestamp_ns"], unit="ns", utc=True)
        .dt.tz_convert("US/Eastern")
        .dt.floor("h")
        .dt.tz_localize(None)
    )
    df = df[(df["hour_et"].dt.hour >= 9) & (df["hour_et"].dt.hour < 16)]

    agg = df.groupby("hour_et").agg(
        bd_open  =("bid_depth",    "first"),
        bd_close =("bid_depth",    "last"),
        book_imb =("book_imb_raw", "mean"),   # mean book imbalance across the hour
    )
    return agg


def _find_consecutive_runs(flag_series: pd.Series) -> list:
    """Return list of (start_idx, length) for runs of True of length >= MIN_CONSECUTIVE_HRS."""
    flag = flag_series.tolist()
    runs, i = [], 0
    while i < len(flag):
        if flag[i]:
            j = i
            while j < len(flag) and flag[j]:
                j += 1
            if j - i >= MIN_CONSECUTIVE_HRS:
                runs.append((i, j - i))
            i = j
        else:
            i += 1
    return runs


def find_active_runs(imb_series: pd.Series) -> list:
    return _find_consecutive_runs(imb_series > IMBALANCE_THRESHOLD)


def find_passive_runs(book_imb_series: pd.Series, trade_imb_series: pd.Series) -> list:
    flag = (book_imb_series > PASSIVE_BOOK_THRESHOLD) & (trade_imb_series < -PASSIVE_TRADE_THRESHOLD)
    return _find_consecutive_runs(flag)


def score_window(avg_imb: float, price_chg_pct: float, bd_growth_pct: float) -> float:
    """MODE A score: trade imbalance strength + price flatness + bid depth growth."""
    imb_score   = min(1.0, max(0.0, (avg_imb - IMBALANCE_THRESHOLD) / (1 - IMBALANCE_THRESHOLD)))
    price_score = min(1.0, max(0.0, (MAX_PRICE_CHANGE_PCT - price_chg_pct) / (MAX_PRICE_CHANGE_PCT + 1.0)))
    depth_score = min(1.0, max(0.0, (bd_growth_pct - MIN_BID_DEPTH_GROWTH) / 80.0))
    return round(imb_score * 33 + price_score * 33 + depth_score * 34, 1)


def score_passive_window(avg_book_imb: float, avg_trade_imb: float, price_chg_pct: float) -> float:
    """MODE B score: book imbalance strength + price resilience + sell intensity absorbed."""
    book_score  = min(1.0, max(0.0, (avg_book_imb - PASSIVE_BOOK_THRESHOLD) / (1 - PASSIVE_BOOK_THRESHOLD)))
    price_score = min(1.0, max(0.0, (MAX_PASSIVE_PRICE_CHG + price_chg_pct) / MAX_PASSIVE_PRICE_CHG))
    sell_score  = min(1.0, max(0.0, (-avg_trade_imb - PASSIVE_TRADE_THRESHOLD) / (1 - PASSIVE_TRADE_THRESHOLD)))
    return round(book_score * 33 + price_score * 33 + sell_score * 34, 1)


def get_day_open_close(conn, symbol: str, date: str):
    """Returns (open_price, close_price) for the session, or (None, None)."""
    row = conn.execute("""
        SELECT
            FIRST(price ORDER BY timestamp_ns) AS open_price,
            LAST(price  ORDER BY timestamp_ns) AS close_price
        FROM trades
        WHERE symbol = ? AND trade_date = ? AND is_extended = false AND is_odd_lot = false
    """, [symbol, date]).fetchone()
    if row and row[0] is not None:
        return float(row[0]), float(row[1])
    return None, None


def analyse_day(conn, symbol: str, date: str) -> tuple:
    """
    Returns (signals, has_depth, df) for one date.
    signals is a list of dicts with a 'mode' key ('A' or 'B').
    """
    trades_df = get_hourly_trades(conn, symbol, date)
    if trades_df.empty:
        return [], False, pd.DataFrame()

    depth_df  = get_hourly_bid_depth(conn, symbol, date)
    has_depth = not depth_df.empty

    df = trades_df.copy()
    if has_depth:
        df = df.join(depth_df, how="left")
    else:
        df["bd_open"] = df["bd_close"] = df["book_imb"] = float("nan")

    session_close = float(df["close_price"].iloc[-1]) if not df.empty else None

    signals = []

    # MODE A — active absorption (trade imbalance positive)
    for start_idx, length in find_active_runs(df["imbalance"]):
        window = df.iloc[start_idx : start_idx + length]

        price_open  = window.iloc[0]["open_price"]
        price_close = window.iloc[-1]["close_price"]
        price_chg   = (price_close - price_open) / price_open * 100 if price_open else None

        bd_open  = window.iloc[0]["bd_open"]
        bd_close = window.iloc[-1]["bd_close"]
        if pd.notna(bd_open) and bd_open > 0 and pd.notna(bd_close):
            bd_growth = (bd_close - bd_open) / bd_open * 100
        else:
            bd_growth = None

        price_ok = price_chg is not None and price_chg < MAX_PRICE_CHANGE_PCT
        depth_ok = bd_growth is not None and bd_growth > MIN_BID_DEPTH_GROWTH

        same_day_follow = (
            (session_close - price_close) / price_close * 100
            if session_close and price_close else None
        )

        signals.append({
            "mode":            "A",
            "date":            date,
            "start":           window.index[0],
            "end":             window.index[-1],
            "hours":           length,
            "avg_imb":         window["imbalance"].mean(),
            "avg_book_imb":    window["book_imb"].mean() if "book_imb" in window else float("nan"),
            "price_chg_pct":   price_chg,
            "bd_growth_pct":   bd_growth,
            "price_ok":        price_ok,
            "depth_ok":        depth_ok,
            "full_signal":     price_ok and depth_ok,
            "score":           (
                score_window(window["imbalance"].mean(), price_chg, bd_growth)
                if price_chg is not None and bd_growth is not None else None
            ),
            "window_close":    price_close,
            "same_day_follow": same_day_follow,
            "next_day_date":   None,
            "next_day_open":   None,
            "next_day_close":  None,
        })

    # MODE B — passive absorption (book bid-heavy, trade net selling)
    if has_depth and "book_imb" in df.columns:
        book_imb_col  = df["book_imb"].fillna(0.0)
        trade_imb_col = df["imbalance"].fillna(0.0)

        for start_idx, length in find_passive_runs(book_imb_col, trade_imb_col):
            window = df.iloc[start_idx : start_idx + length]

            price_open  = window.iloc[0]["open_price"]
            price_close = window.iloc[-1]["close_price"]
            price_chg   = (price_close - price_open) / price_open * 100 if price_open else None

            bd_open  = window.iloc[0]["bd_open"]
            bd_close = window.iloc[-1]["bd_close"]
            if pd.notna(bd_open) and bd_open > 0 and pd.notna(bd_close):
                bd_growth = (bd_close - bd_open) / bd_open * 100
            else:
                bd_growth = None

            price_ok = price_chg is not None and price_chg < MAX_PASSIVE_PRICE_CHG

            same_day_follow = (
                (session_close - price_close) / price_close * 100
                if session_close and price_close else None
            )

            avg_book_imb  = window["book_imb"].mean()
            avg_trade_imb = window["imbalance"].mean()

            signals.append({
                "mode":            "B",
                "date":            date,
                "start":           window.index[0],
                "end":             window.index[-1],
                "hours":           length,
                "avg_imb":         avg_trade_imb,
                "avg_book_imb":    avg_book_imb,
                "price_chg_pct":   price_chg,
                "bd_growth_pct":   bd_growth,
                "price_ok":        price_ok,
                "depth_ok":        True,   # book imbalance replaces depth growth for passive
                "full_signal":     price_ok,
                "score":           (
                    score_passive_window(avg_book_imb, avg_trade_imb, price_chg)
                    if price_chg is not None else None
                ),
                "window_close":    price_close,
                "same_day_follow": same_day_follow,
                "next_day_date":   None,
                "next_day_open":   None,
                "next_day_close":  None,
            })

    # Sort all signals by start time
    signals.sort(key=lambda s: s["start"])
    return signals, has_depth, df


def _fmt_pct(val) -> str:
    return f"{val:+.2f}%" if val is not None else "n/a"


def print_signal_detail(s: dict) -> None:
    mode_label = "A:ACTIVE" if s["mode"] == "A" else "B:PASSIVE"
    if s["full_signal"]:
        flag = f"*** ABSORPTION [{mode_label}]"
    else:
        flag = f"    candidate  [{mode_label}]"

    print(f"  {flag}  {s['start'].strftime('%H:%M')}–{s['end'].strftime('%H:%M')} ET  ({s['hours']}h)")

    pchg = f"{s['price_chg_pct']:+.3f}%" if s["price_chg_pct"] is not None else "n/a"
    pok  = "PASS" if s["price_ok"] else ("FAIL" if s["price_chg_pct"] is not None else "n/a")
    sc   = f"{s['score']:.1f}/100" if s["score"] is not None else "n/a"

    if s["mode"] == "A":
        bdg = f"{s['bd_growth_pct']:+.1f}%" if s["bd_growth_pct"] is not None else "n/a (no DPLS)"
        dok = "PASS" if s["depth_ok"] else ("FAIL" if s["bd_growth_pct"] is not None else "n/a")
        print(f"    trade imb {s['avg_imb']:+.3f}  |  price {pchg} [{pok}]  |  "
              f"bid depth growth {bdg} [{dok}]  |  score {sc}")
    else:
        book_imb = s["avg_book_imb"]
        book_str = f"{book_imb:+.3f}" if pd.notna(book_imb) else "n/a"
        print(f"    book imb {book_str}  trade imb {s['avg_imb']:+.3f}  |  "
              f"price {pchg} [{pok}]  |  score {sc}")

    print(f"    same-day follow: {_fmt_pct(s['same_day_follow'])}  |  "
          f"next day ({s['next_day_date'] or 'n/a'}):  "
          f"open {_fmt_pct(s['next_day_open'])}  close {_fmt_pct(s['next_day_close'])}")


def print_hourly_table(df: pd.DataFrame, has_depth: bool) -> None:
    cols = ["imbalance", "buy_volume", "sell_volume", "total_volume", "open_price", "close_price"]
    if has_depth and "book_imb" in df.columns:
        cols += ["book_imb", "bd_open", "bd_close"]
    elif has_depth:
        cols += ["bd_open", "bd_close"]
    out = df[[c for c in cols if c in df.columns]].copy()
    out.index = out.index.strftime("%H:%M ET")
    out.columns = [c.replace("_", " ") for c in out.columns]
    print(out.to_string(float_format=lambda x: f"{x:,.2f}"))
    print()


def print_summary_table(all_signals: list, dates: list) -> None:
    rows = []
    for date in dates:
        day_sigs  = [s for s in all_signals if s["date"] == date]
        full      = [s for s in day_sigs if s["full_signal"]]
        full_a    = [s for s in full if s["mode"] == "A"]
        full_b    = [s for s in full if s["mode"] == "B"]
        scores    = [s["score"] for s in full if s["score"] is not None]

        same_day_vals = [s["same_day_follow"] for s in full if s["same_day_follow"] is not None]
        next_day_vals = [s["next_day_close"]  for s in full if s["next_day_close"]  is not None]

        mode_str = ""
        if full_a and full_b:
            mode_str = "A+B"
        elif full_a:
            mode_str = "A"
        elif full_b:
            mode_str = "B"
        else:
            mode_str = "-"

        rows.append({
            "date":            date,
            "windows":         len(day_sigs),
            "signals":         len(full),
            "mode":            mode_str,
            "best_score":      f"{max(scores):.1f}" if scores else "-",
            "same_day_follow": _fmt_pct(same_day_vals[0]) if same_day_vals else "—",
            "next_day_close":  _fmt_pct(next_day_vals[0]) if next_day_vals else "—",
            "signal_times":    "  ".join(
                f"{s['start'].strftime('%H:%M')}–{s['end'].strftime('%H:%M')}[{s['mode']}]"
                for s in full
            ) or "—",
        })

    df = pd.DataFrame(rows).set_index("date")
    df.columns = ["Windows", "Signals", "Mode", "Best Score",
                  "Same-Day Follow", "Next Day Close", "Signal Times (ET)"]
    print(df.to_string())
    print()


def run(symbol: str, start: str, end: str, verbose: bool, plot: bool,
        imbalance_threshold: float, min_hours: int,
        max_price_chg: float, min_depth_growth: float,
        passive_book: float, passive_trade: float, max_passive_price_chg: float) -> None:
    global IMBALANCE_THRESHOLD, MIN_CONSECUTIVE_HRS, MAX_PRICE_CHANGE_PCT, MIN_BID_DEPTH_GROWTH
    global PASSIVE_BOOK_THRESHOLD, PASSIVE_TRADE_THRESHOLD, MAX_PASSIVE_PRICE_CHG
    IMBALANCE_THRESHOLD    = imbalance_threshold
    MIN_CONSECUTIVE_HRS    = min_hours
    MAX_PRICE_CHANGE_PCT   = max_price_chg
    MIN_BID_DEPTH_GROWTH   = min_depth_growth
    PASSIVE_BOOK_THRESHOLD  = passive_book
    PASSIVE_TRADE_THRESHOLD = passive_trade
    MAX_PASSIVE_PRICE_CHG   = max_passive_price_chg

    conn  = _connect()
    dates = get_dates(conn, symbol, start, end)

    if not dates:
        conn.close()
        sys.exit(f"No trade data found for {symbol} between {start} and {end}.")

    all_signals  = []
    day_data     = {}
    warned_depth = False

    for date in dates:
        signals, has_depth, df = analyse_day(conn, symbol, date)
        all_signals.extend(signals)
        day_data[date] = (signals, has_depth, df)

        if not has_depth and not warned_depth:
            print("[warn] No DPLS depth data — bid depth and passive mode unavailable for all dates.")
            warned_depth = True

        if verbose:
            full   = [s for s in signals if s["full_signal"]]
            full_a = [s for s in full if s["mode"] == "A"]
            full_b = [s for s in full if s["mode"] == "B"]
            print(f"\n{date}  ({len(signals)} window(s), {len(full_a)} active + {len(full_b)} passive signal(s))")
            print("-" * 70)
            if signals:
                for s in signals:
                    print_signal_detail(s)
            else:
                print("  No qualifying imbalance windows.")
            print_hourly_table(df, has_depth)

    # Enrich each signal with next trading day's open/close
    for s in all_signals:
        date_idx = dates.index(s["date"])
        if date_idx + 1 < len(dates):
            next_date = dates[date_idx + 1]
            nd_open, nd_close = get_day_open_close(conn, symbol, next_date)
            wc = s["window_close"]
            s["next_day_date"]  = next_date
            s["next_day_open"]  = (nd_open  - wc) / wc * 100 if nd_open  and wc else None
            s["next_day_close"] = (nd_close - wc) / wc * 100 if nd_close and wc else None

    conn.close()

    total_full   = sum(1 for s in all_signals if s["full_signal"])
    total_active = sum(1 for s in all_signals if s["full_signal"] and s["mode"] == "A")
    total_passive= sum(1 for s in all_signals if s["full_signal"] and s["mode"] == "B")

    print(f"\n{'='*65}")
    print(f"  Absorption Summary: {symbol}  {start} → {end}")
    print(f"{'='*65}")
    print(f"  Trading days analysed  : {len(dates)}")
    print(f"  Total trigger windows  : {len(all_signals)}")
    print(f"  Full absorption signals: {total_full}  "
          f"(Active[A]: {total_active}  Passive[B]: {total_passive})")
    print()
    print_summary_table(all_signals, dates)

    if plot:
        for date in dates:
            signals, has_depth, df = day_data[date]
            if not df.empty:
                _plot(df, signals, symbol, date, has_depth)


def _plot(df: pd.DataFrame, signals: list, symbol: str, date: str, has_depth: bool) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    n_rows = 3 if has_depth else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(13, 4 * n_rows), sharex=True)
    fig.suptitle(f"{symbol}  |  Absorption Analysis  {date}", fontsize=13)
    hours = df.index

    for ax in axes:
        for s in signals:
            if s["mode"] == "A":
                color = "green" if s["full_signal"] else "yellow"
                alpha = 0.15   if s["full_signal"] else 0.08
            else:
                color = "orange" if s["full_signal"] else "lightyellow"
                alpha = 0.20    if s["full_signal"] else 0.08
            ax.axvspan(s["start"], s["end"] + pd.Timedelta(hours=1), color=color, alpha=alpha)

    ax0 = axes[0]
    colors = ["steelblue" if v > IMBALANCE_THRESHOLD else "salmon"
              for v in df["imbalance"].fillna(0)]
    ax0.bar(hours, df["imbalance"], color=colors, width=0.04)
    ax0.axhline( IMBALANCE_THRESHOLD,  color="k",      linestyle="--", linewidth=0.8,
                label=f"active threshold +{IMBALANCE_THRESHOLD}")
    ax0.axhline(-PASSIVE_TRADE_THRESHOLD, color="darkorange", linestyle="--", linewidth=0.8,
                label=f"passive threshold -{PASSIVE_TRADE_THRESHOLD}")
    ax0.set_ylabel("Trade Imbalance")
    ax0.legend(fontsize=8)

    ax1 = axes[1]
    ax1.plot(hours, df["open_price"],  marker="o", markersize=4, linewidth=1.2, label="Open")
    ax1.plot(hours, df["close_price"], marker="s", markersize=4, linewidth=1.2,
             linestyle="--", label="Close")
    ax1.set_ylabel("Price ($)")
    ax1.legend(fontsize=8)

    if has_depth and "book_imb" in df.columns:
        ax2 = axes[2]
        ax2_r = ax2.twinx()
        ax2.plot(hours, df["bd_open"],  marker="o", markersize=4, linewidth=1.2, label="Bid depth open")
        ax2.plot(hours, df["bd_close"], marker="s", markersize=4, linewidth=1.2,
                 linestyle="--", label="Bid depth close")
        ax2.set_ylabel("Bid Depth (shares)")
        ax2_r.plot(hours, df["book_imb"], color="purple", linewidth=1.2, linestyle=":", label="Book imb")
        ax2_r.axhline(PASSIVE_BOOK_THRESHOLD, color="darkorange", linestyle=":", linewidth=0.8)
        ax2_r.set_ylabel("Book Imbalance", color="purple")
        ax2_r.tick_params(axis="y", labelcolor="purple")
        ax2.legend(fontsize=8, loc="upper left")
        ax2_r.legend(fontsize=8, loc="upper right")
    elif has_depth:
        ax2 = axes[2]
        ax2.plot(hours, df["bd_open"],  marker="o", markersize=4, linewidth=1.2, label="Bid depth open")
        ax2.plot(hours, df["bd_close"], marker="s", markersize=4, linewidth=1.2,
                 linestyle="--", label="Bid depth close")
        ax2.set_ylabel("Bid Depth (shares)")
        ax2.legend(fontsize=8)

    axes[-1].set_xlabel("Hour (ET)")
    fig.legend(
        handles=[
            mpatches.Patch(color="green",  alpha=0.3, label="Active signal [A]"),
            mpatches.Patch(color="orange", alpha=0.3, label="Passive signal [B]"),
            mpatches.Patch(color="yellow", alpha=0.3, label="Partial candidate"),
        ],
        loc="upper right", fontsize=8,
    )
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="IEX absorption score detector")
    parser.add_argument("symbol", help="Ticker symbol, e.g. AAPL")

    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument("--date",  metavar="YYYY-MM-DD", help="Single trade date")
    date_group.add_argument("--start", metavar="YYYY-MM-DD", help="Start of date range")

    parser.add_argument("--end",     metavar="YYYY-MM-DD", help="End of date range (required with --start)")
    parser.add_argument("--verbose",          action="store_true",  help="Print per-hour table for each day")
    parser.add_argument("--plot",             action="store_true",  help="Show chart for each day")

    # Mode A parameters
    parser.add_argument("--imbalance",        type=float, default=IMBALANCE_THRESHOLD, metavar="N",
                        help=f"[A] Min hourly trade imbalance (default: {IMBALANCE_THRESHOLD})")
    parser.add_argument("--min-hours",        type=int,   default=MIN_CONSECUTIVE_HRS,  metavar="N",
                        help=f"[A/B] Min consecutive hours (default: {MIN_CONSECUTIVE_HRS})")
    parser.add_argument("--max-price-chg",    type=float, default=MAX_PRICE_CHANGE_PCT, metavar="PCT",
                        help=f"[A] Max price change %% across window (default: {MAX_PRICE_CHANGE_PCT})")
    parser.add_argument("--min-depth-growth", type=float, default=MIN_BID_DEPTH_GROWTH, metavar="PCT",
                        help=f"[A] Min bid depth growth %% (default: {MIN_BID_DEPTH_GROWTH})")

    # Mode B parameters
    parser.add_argument("--passive-book",     type=float, default=PASSIVE_BOOK_THRESHOLD, metavar="N",
                        help=f"[B] Min book imbalance threshold (default: {PASSIVE_BOOK_THRESHOLD})")
    parser.add_argument("--passive-trade",    type=float, default=PASSIVE_TRADE_THRESHOLD, metavar="N",
                        help=f"[B] Trade imbalance must be below negative of this (default: {PASSIVE_TRADE_THRESHOLD})")
    parser.add_argument("--max-passive-price-chg", type=float, default=MAX_PASSIVE_PRICE_CHG, metavar="PCT",
                        help=f"[B] Max price change %% for passive window (default: {MAX_PASSIVE_PRICE_CHG})")

    args = parser.parse_args()

    if args.start and not args.end:
        parser.error("--end is required when using --start")

    if not os.path.exists(DB_PATH):
        sys.exit("No database found. Run 'python main.py ingest' first.")

    start = args.date or args.start
    end   = args.date or args.end
    run(args.symbol.upper(), start, end, args.verbose, args.plot,
        args.imbalance, args.min_hours, args.max_price_chg, args.min_depth_growth,
        args.passive_book, args.passive_trade, args.max_passive_price_chg)


if __name__ == "__main__":
    main()
