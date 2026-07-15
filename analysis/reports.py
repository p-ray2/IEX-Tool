"""Microstructure reports and per-symbol queries backed by a DuckDB database."""

import sys
from typing import Optional

import duckdb
import pandas as pd

# IEX timestamps are UTC; EDT = UTC-4
_EDT_OFFSET_NS = 4 * 3_600 * 1_000_000_000
_NS_PER_HOUR = 3_600_000_000_000


def _conn(db_path: str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path, read_only=True)


def _hour_et(col: str) -> str:
    """SQL expression: convert a nanosecond-epoch column to Eastern hour (0-23)."""
    return f"CAST(({col} - {_EDT_OFFSET_NS}) / {_NS_PER_HOUR} AS INTEGER) % 24"


def _date_clause(date_filter: Optional[str], table: str = "") -> str:
    """Return a WHERE or AND clause for an optional YYYY-MM-DD date filter."""
    prefix = f"{table}." if table else ""
    if date_filter:
        return f"WHERE {prefix}trade_date = '{date_filter}'"
    return ""


def _and_date_clause(date_filter: Optional[str], table: str = "") -> str:
    prefix = f"{table}." if table else ""
    if date_filter:
        return f"AND {prefix}trade_date = '{date_filter}'"
    return ""


# ── Data summary report ───────────────────────────────────────────────────────

def report_summary(db_path: str, date_filter: Optional[str] = None) -> dict:
    """Return high-level counts: total symbols, shares, trades, and depth snapshots."""
    conn = _conn(db_path)
    wh = _date_clause(date_filter)
    trades = conn.execute(f"""
        SELECT
            COUNT(DISTINCT symbol) AS num_symbols,
            COUNT(*)               AS num_trades,
            SUM(size)              AS total_shares,
            SUM(CAST(size AS DOUBLE) * price) AS total_notional
        FROM trades {wh}
    """).df().iloc[0]
    quotes = conn.execute(f"SELECT COUNT(*) AS num_quotes FROM quotes {wh}").df().iloc[0]
    depth  = conn.execute(f"""
        SELECT COUNT(*) AS num_snapshots, COUNT(DISTINCT symbol) AS depth_symbols
        FROM depth_snapshots {wh}
    """).df().iloc[0]
    conn.close()
    return {
        "num_symbols":    int(trades["num_symbols"]),
        "num_trades":     int(trades["num_trades"]),
        "total_shares":   int(trades["total_shares"]) if trades["total_shares"] else 0,
        "total_notional": float(trades["total_notional"]) if trades["total_notional"] else 0.0,
        "num_quotes":     int(quotes["num_quotes"]),
        "num_snapshots":  int(depth["num_snapshots"]),
        "depth_symbols":  int(depth["depth_symbols"]),
    }


def report_top_notional(db_path: str, top_n: int = 25, date_filter: Optional[str] = None) -> pd.DataFrame:
    """Return top N symbols by notional value (total shares × VWAP)."""
    conn = _conn(db_path)
    wh = _date_clause(date_filter)
    df = conn.execute(f"""
        SELECT
            symbol,
            SUM(CAST(size AS DOUBLE) * price)       AS notional,
            SUM(size)                                AS total_shares,
            COUNT(*)                                 AS num_trades,
            SUM(CAST(size AS DOUBLE) * price)
                / SUM(CAST(size AS DOUBLE))          AS vwap,
            MIN(price)                               AS low,
            MAX(price)                               AS high
        FROM trades {wh}
        GROUP BY symbol
        ORDER BY notional DESC
        LIMIT {top_n}
    """).df()
    conn.close()
    return df


# ── Top-symbols volume report ─────────────────────────────────────────────────

def report_top_volume(db_path: str, top_n: int = 25, date_filter: Optional[str] = None) -> pd.DataFrame:
    """Return top N symbols by total traded shares with VWAP and trade count."""
    conn = _conn(db_path)
    wh = _date_clause(date_filter)
    df = conn.execute(f"""
        SELECT
            symbol,
            SUM(size)                            AS total_shares,
            COUNT(*)                             AS num_trades,
            SUM(CAST(size AS DOUBLE) * price)
                / SUM(CAST(size AS DOUBLE))      AS vwap,
            MIN(price)                           AS low,
            MAX(price)                           AS high
        FROM trades {wh}
        GROUP BY symbol
        ORDER BY total_shares DESC
        LIMIT {top_n}
    """).df()
    conn.close()
    return df


# ── Spread report for a given symbol list ─────────────────────────────────────

def report_spreads(db_path: str, symbols: list, date_filter: Optional[str] = None) -> pd.DataFrame:
    """Return mean/median spread and relative spread (bps) for the given symbols."""
    conn = _conn(db_path)
    sym_list = ", ".join(f"'{s}'" for s in symbols)
    ad = _and_date_clause(date_filter)
    df = conn.execute(f"""
        SELECT
            symbol,
            COUNT(*)                                           AS num_quotes,
            AVG(spread)                                        AS mean_spread,
            MEDIAN(spread)                                     AS median_spread,
            STDDEV(spread)                                     AS std_spread,
            AVG(spread / NULLIF(mid_price, 0) * 10000)        AS mean_spread_bps,
            MEDIAN(spread / NULLIF(mid_price, 0) * 10000)     AS median_spread_bps
        FROM quotes
        WHERE symbol IN ({sym_list}) {ad}
        GROUP BY symbol
        ORDER BY mean_spread DESC
    """).df()
    conn.close()
    return df


# ── Depth imbalance report ─────────────────────────────────────────────────────

def report_depth_imbalance(db_path: str, symbols: list, date_filter: Optional[str] = None) -> pd.DataFrame:
    """Return mean depth imbalance statistics for the given symbols."""
    conn = _conn(db_path)
    sym_list = ", ".join(f"'{s}'" for s in symbols)
    ad = _and_date_clause(date_filter)
    df = conn.execute(f"""
        SELECT
            symbol,
            COUNT(*)            AS num_snapshots,
            AVG(imbalance)      AS mean_imbalance,
            MEDIAN(imbalance)   AS median_imbalance,
            STDDEV(imbalance)   AS std_imbalance,
            AVG(bid_depth)      AS mean_bid_depth,
            AVG(ask_depth)      AS mean_ask_depth
        FROM depth_snapshots
        WHERE symbol IN ({sym_list}) {ad}
        GROUP BY symbol
        ORDER BY ABS(AVG(imbalance)) DESC
    """).df()
    conn.close()
    return df


# ── Per-symbol query ───────────────────────────────────────────────────────────

def query_symbol(db_path: str, symbol: str, date_filter: Optional[str] = None) -> dict:
    """Return a dict of DataFrames summarising a single symbol."""
    conn = _conn(db_path)
    sym = symbol.upper().strip()
    ad = _and_date_clause(date_filter)

    summary = conn.execute(f"""
        SELECT
            SUM(size)                             AS total_shares,
            COUNT(*)                              AS num_trades,
            SUM(CAST(size AS DOUBLE) * price)
                / SUM(CAST(size AS DOUBLE))       AS vwap,
            MIN(price)                            AS low,
            MAX(price)                            AS high,
            MIN(timestamp_ns)                     AS first_trade_ns,
            MAX(timestamp_ns)                     AS last_trade_ns
        FROM trades
        WHERE symbol = '{sym}' {ad}
    """).df()

    hourly = conn.execute(f"""
        SELECT
            trade_date,
            {_hour_et('timestamp_ns')}            AS hour_et,
            SUM(size)                             AS volume,
            COUNT(*)                              AS num_trades,
            SUM(CAST(size AS DOUBLE) * price)
                / SUM(CAST(size AS DOUBLE))       AS vwap,
            MIN(price)                            AS low,
            MAX(price)                            AS high
        FROM trades
        WHERE symbol = '{sym}' {ad}
        GROUP BY 1, 2
        ORDER BY 1, 2
    """).df()

    spread_stats = conn.execute(f"""
        SELECT
            COUNT(*)                                       AS num_quotes,
            AVG(spread)                                    AS mean_spread,
            MEDIAN(spread)                                 AS median_spread,
            STDDEV(spread)                                 AS std_spread,
            AVG(spread / NULLIF(mid_price, 0) * 10000)    AS mean_spread_bps,
            MIN(spread)                                    AS min_spread,
            MAX(spread)                                    AS max_spread
        FROM quotes
        WHERE symbol = '{sym}' {ad}
    """).df()

    depth_stats = conn.execute(f"""
        SELECT
            COUNT(*)          AS num_snapshots,
            AVG(imbalance)    AS mean_imbalance,
            STDDEV(imbalance) AS std_imbalance,
            AVG(bid_depth)    AS mean_bid_depth,
            AVG(ask_depth)    AS mean_ask_depth
        FROM depth_snapshots
        WHERE symbol = '{sym}' {ad}
    """).df()

    depth_hourly = conn.execute(f"""
        SELECT
            trade_date,
            {_hour_et('timestamp_ns')}  AS hour_et,
            AVG(imbalance)              AS mean_imbalance,
            AVG(bid_depth)              AS mean_bid_depth,
            AVG(ask_depth)              AS mean_ask_depth
        FROM depth_snapshots
        WHERE symbol = '{sym}' {ad}
        GROUP BY 1, 2
        ORDER BY 1, 2
    """).df()

    conn.close()
    return {
        "summary": summary,
        "hourly": hourly,
        "spread_stats": spread_stats,
        "depth_stats": depth_stats,
        "depth_hourly": depth_hourly,
    }


# ── Formatted printing ─────────────────────────────────────────────────────────

def _pct(val, fmt=".4f"):
    return "N/A" if val is None or (hasattr(val, "__float__") and pd.isna(val)) else f"{val:{fmt}}"


def print_market_report(db_path: str, top_n: int = 25, date_filter: Optional[str] = None) -> None:
    date_label = date_filter or "all dates"
    print("\n" + "=" * 70)
    print(f"  IEX Market Microstructure Report  —  Top {top_n} by Volume  —  {date_label}")
    print("=" * 70)

    vol_df = report_top_volume(db_path, top_n, date_filter)
    if vol_df.empty:
        print("  No trade data found. Run 'ingest --feed tops' first.")
        return

    symbols = vol_df["symbol"].tolist()

    spread_df = report_spreads(db_path, symbols, date_filter).set_index("symbol")
    depth_df  = report_depth_imbalance(db_path, symbols, date_filter).set_index("symbol")

    print(f"\n{'Symbol':<8} {'Shares':>14} {'Trades':>8} {'VWAP':>10} "
          f"{'Spread$':>9} {'Sprd bps':>9} {'Imbalance':>10}")
    print("-" * 72)

    for _, row in vol_df.iterrows():
        sym = row["symbol"]
        sp  = spread_df.loc[sym] if sym in spread_df.index else None
        dp  = depth_df.loc[sym]  if sym in depth_df.index  else None

        spread_val = _pct(sp["mean_spread"],     ".4f") if sp is not None else "  N/A"
        spread_bps = _pct(sp["mean_spread_bps"], ".1f")  if sp is not None else "  N/A"
        imbalance  = _pct(dp["mean_imbalance"],  ".3f")  if dp is not None else "  N/A"

        print(f"{sym:<8} {int(row['total_shares']):>14,} {int(row['num_trades']):>8,} "
              f"{row['vwap']:>10.4f} {spread_val:>9} {spread_bps:>9} {imbalance:>10}")

    # Top spreads (widest spread among top-volume symbols)
    if not spread_df.empty:
        print("\n── Widest Mean Spreads (among top-volume symbols) ──────────────────────")
        top_spread = spread_df.sort_values("mean_spread", ascending=False).head(10)
        print(f"\n{'Symbol':<8} {'Mean $':>9} {'Median $':>10} {'Mean bps':>10} {'Quotes':>10}")
        print("-" * 52)
        for sym, row in top_spread.iterrows():
            print(f"{sym:<8} {row['mean_spread']:>9.4f} {row['median_spread']:>10.4f} "
                  f"{row['mean_spread_bps']:>10.1f} {int(row['num_quotes']):>10,}")

    print()


def print_symbol_report(db_path: str, symbol: str, date_filter: Optional[str] = None) -> None:
    sym = symbol.upper().strip()
    date_label = f"  {date_filter}" if date_filter else "  (all dates)"
    print(f"\n{'=' * 60}")
    print(f"  Symbol Report: {sym}{date_label}")
    print(f"{'=' * 60}")

    data = query_symbol(db_path, sym, date_filter)
    s = data["summary"].iloc[0] if not data["summary"].empty else None

    if s is None or pd.isna(s["total_shares"]):
        print(f"  No trade data found for {sym}.")
        return

    from datetime import datetime, timezone
    def ns_to_et(ns):
        if pd.isna(ns): return "N/A"
        utc_s = int(ns) / 1e9
        dt = datetime.fromtimestamp(utc_s, tz=timezone.utc)
        return f"{dt.hour - 4:02d}:{dt.minute:02d}:{dt.second:02d} ET"

    print(f"\n  Total shares traded : {int(s['total_shares']):,}")
    print(f"  Number of trades    : {int(s['num_trades']):,}")
    print(f"  VWAP                : ${s['vwap']:.4f}")
    print(f"  Price range         : ${s['low']:.4f} – ${s['high']:.4f}")
    print(f"  First trade         : {ns_to_et(s['first_trade_ns'])}")
    print(f"  Last trade          : {ns_to_et(s['last_trade_ns'])}")

    sp = data["spread_stats"].iloc[0] if not data["spread_stats"].empty else None
    if sp is not None and not pd.isna(sp["num_quotes"]) and sp["num_quotes"] > 0:
        print(f"\n  ── Spread (from {int(sp['num_quotes']):,} quotes) ──────────────────────")
        print(f"  Mean spread   : ${sp['mean_spread']:.4f}  ({sp['mean_spread_bps']:.1f} bps)")
        print(f"  Median spread : ${sp['median_spread']:.4f}")
        print(f"  Std spread    : ${sp['std_spread']:.4f}")
        print(f"  Range         : ${sp['min_spread']:.4f} – ${sp['max_spread']:.4f}")

    dp = data["depth_stats"].iloc[0] if not data["depth_stats"].empty else None
    if dp is not None and not pd.isna(dp["num_snapshots"]) and dp["num_snapshots"] > 0:
        print(f"\n  ── Depth Imbalance (from {int(dp['num_snapshots']):,} snapshots) ──────────")
        print(f"  Mean imbalance : {dp['mean_imbalance']:+.4f}  "
              f"({'buy-side heavy' if dp['mean_imbalance'] > 0 else 'sell-side heavy'})")
        print(f"  Std imbalance  : {dp['std_imbalance']:.4f}")
        print(f"  Mean bid depth : {dp['mean_bid_depth']:,.0f} shares")
        print(f"  Mean ask depth : {dp['mean_ask_depth']:,.0f} shares")

    hourly = data["hourly"]
    if not hourly.empty:
        multi_day = hourly["trade_date"].nunique() > 1
        print(f"\n  ── Hourly Volume (ET) ──────────────────────────────────────────────")
        if multi_day:
            print(f"  {'Date':<12} {'Hour':>5} {'Volume':>12} {'Trades':>8} {'VWAP':>10} {'Low':>10} {'High':>10}")
            print(f"  {'-'*70}")
            for _, r in hourly.iterrows():
                h = int(r["hour_et"])
                print(f"  {str(r['trade_date']):<12} {h:02d}:00 {int(r['volume']):>12,} "
                      f"{int(r['num_trades']):>8,} {r['vwap']:>10.4f} "
                      f"{r['low']:>10.4f} {r['high']:>10.4f}")
        else:
            print(f"  {'Hour':>5} {'Volume':>12} {'Trades':>8} {'VWAP':>10} {'Low':>10} {'High':>10}")
            print(f"  {'-'*58}")
            for _, r in hourly.iterrows():
                h = int(r["hour_et"])
                print(f"  {h:02d}:00 {int(r['volume']):>12,} {int(r['num_trades']):>8,} "
                      f"{r['vwap']:>10.4f} {r['low']:>10.4f} {r['high']:>10.4f}")

    dh = data["depth_hourly"]
    if not dh.empty:
        multi_day = dh["trade_date"].nunique() > 1
        print(f"\n  ── Hourly Depth Imbalance (ET) ────────────────────────────────────")
        if multi_day:
            print(f"  {'Date':<12} {'Hour':>5} {'Imbalance':>12} {'Bid depth':>12} {'Ask depth':>12}")
            print(f"  {'-'*58}")
            for _, r in dh.iterrows():
                h = int(r["hour_et"])
                print(f"  {str(r['trade_date']):<12} {h:02d}:00 {r['mean_imbalance']:>+12.4f} "
                      f"{r['mean_bid_depth']:>12,.0f} {r['mean_ask_depth']:>12,.0f}")
        else:
            print(f"  {'Hour':>5} {'Imbalance':>12} {'Bid depth':>12} {'Ask depth':>12}")
            print(f"  {'-'*46}")
            for _, r in dh.iterrows():
                h = int(r["hour_et"])
                print(f"  {h:02d}:00 {r['mean_imbalance']:>+12.4f} {r['mean_bid_depth']:>12,.0f} "
                      f"{r['mean_ask_depth']:>12,.0f}")

    print()


def print_summary_report(db_path: str, date_filter: Optional[str] = None) -> None:
    date_label = date_filter or "all dates"
    print("\n" + "=" * 50)
    print(f"  IEX Data Summary  —  {date_label}")
    print("=" * 50)
    s = report_summary(db_path, date_filter)
    if s["num_trades"] == 0:
        print("  No trade data found. Run 'ingest --feed tops' first.")
    else:
        print(f"  Symbols traded      : {s['num_symbols']:>12,}")
        print(f"  Total trades        : {s['num_trades']:>12,}")
        print(f"  Total shares traded : {s['total_shares']:>12,}")
        print(f"  Total notional      : ${s['total_notional']:>18,.2f}")
        print(f"  Quote updates       : {s['num_quotes']:>12,}")
    if s["num_snapshots"] > 0:
        print(f"  Depth snapshots     : {s['num_snapshots']:>12,}  ({s['depth_symbols']:,} symbols)")
    print()


def print_top_notional_report(db_path: str, top_n: int = 25, date_filter: Optional[str] = None) -> None:
    date_label = date_filter or "all dates"
    print("\n" + "=" * 70)
    print(f"  Top {top_n} Symbols by Notional Value  (shares × VWAP)  —  {date_label}")
    print("=" * 70)
    df = report_top_notional(db_path, top_n, date_filter)
    if df.empty:
        print("  No trade data found. Run 'ingest --feed tops' first.")
        return
    print(f"\n{'Symbol':<8} {'Notional ($)':>18} {'Shares':>14} {'Trades':>8} "
          f"{'VWAP':>10} {'Low':>10} {'High':>10}")
    print("-" * 83)
    for _, row in df.iterrows():
        print(f"{row['symbol']:<8} {row['notional']:>18,.2f} {int(row['total_shares']):>14,} "
              f"{int(row['num_trades']):>8,} {row['vwap']:>10.4f} "
              f"{row['low']:>10.4f} {row['high']:>10.4f}")
    print()
