"""
Compare IEX exchange volume vs. Yahoo Finance total-market volume for a symbol.

Usage:
    python volume_correlation.py AAPL --start 2026-01-01 --end 2026-04-25
    python volume_correlation.py AAPL --start 2026-01-01 --end 2026-04-25 --plot
"""

import argparse
import os
import sys

import time
import requests
import duckdb
import pandas as pd
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "cache", "market_data.duckdb")


def get_iex_volume(symbol: str, start: str, end: str) -> pd.Series:
    conn = duckdb.connect(DB_PATH, read_only=True)
    df = conn.execute(
        """
        SELECT trade_date, SUM(size) AS iex_volume
        FROM trades
        WHERE symbol    = ?
          AND trade_date BETWEEN ? AND ?
        GROUP BY trade_date
        ORDER BY trade_date
        """,
        [symbol, start, end],
    ).df()
    conn.close()
    if df.empty:
        sys.exit(f"No IEX trades found for {symbol} between {start} and {end}.")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["iex_volume"]


def get_yahoo_volume(symbol: str, start: str, end: str) -> pd.Series:
    period1 = int(pd.Timestamp(start).timestamp())
    period2 = int((pd.Timestamp(end) + pd.Timedelta(days=1)).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?interval=1d&period1={period1}&period2={period2}&events=history"
    )
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=15)
    if resp.status_code != 200:
        sys.exit(f"Yahoo Finance request failed ({resp.status_code}) for {symbol}.")
    payload = resp.json()
    try:
        result = payload["chart"]["result"][0]
        timestamps = result["timestamp"]
        volume     = result["indicators"]["quote"][0]["volume"]
    except (KeyError, IndexError, TypeError):
        sys.exit(f"Unexpected Yahoo Finance response structure for {symbol}.")
    dates = pd.to_datetime(timestamps, unit="s").normalize()
    series = pd.Series(volume, index=dates, name="yahoo_volume", dtype=float)
    series = series[series.index >= pd.Timestamp(start)]
    series = series[series.index <= pd.Timestamp(end)]
    if series.empty:
        sys.exit(f"No Yahoo Finance data found for {symbol} between {start} and {end}.")
    return series


def run(symbol: str, start: str, end: str, plot: bool) -> None:
    print(f"\nFetching IEX volume for {symbol} …")
    iex = get_iex_volume(symbol, start, end)

    print(f"Fetching Yahoo Finance volume for {symbol} …")
    yahoo = get_yahoo_volume(symbol, start, end)

    df = pd.DataFrame({"iex_volume": iex, "yahoo_volume": yahoo}).dropna()

    if len(df) < 2:
        sys.exit("Not enough overlapping trading days to compute correlation.")

    pearson_r, pearson_p  = stats.pearsonr(df["iex_volume"], df["yahoo_volume"])
    spearman_r, spearman_p = stats.spearmanr(df["iex_volume"], df["yahoo_volume"])

    # IEX market share estimate
    df["iex_pct"] = df["iex_volume"] / df["yahoo_volume"] * 100

    print(f"\n{'='*55}")
    print(f"  Volume Correlation: {symbol}  ({start} → {end})")
    print(f"{'='*55}")
    print(f"  Overlapping trading days : {len(df)}")
    print(f"  Pearson  r = {pearson_r:+.4f}   p = {pearson_p:.4g}")
    print(f"  Spearman r = {spearman_r:+.4f}   p = {spearman_p:.4g}")
    print(f"\n  IEX share of total volume")
    print(f"    Mean  : {df['iex_pct'].mean():.2f}%")
    print(f"    Median: {df['iex_pct'].median():.2f}%")
    print(f"    Min   : {df['iex_pct'].min():.2f}%")
    print(f"    Max   : {df['iex_pct'].max():.2f}%")
    print(f"{'='*55}\n")

    print(df[["iex_volume", "yahoo_volume", "iex_pct"]].rename(
        columns={"iex_pct": "iex_%"}
    ).to_string())

    if plot:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
        fig.suptitle(f"{symbol}  |  IEX vs. Yahoo Finance Volume  ({start} → {end})", fontsize=13)

        # Time-series
        ax = axes[0]
        ax.bar(df.index, df["yahoo_volume"], label="Yahoo (total market)", alpha=0.6, color="steelblue")
        ax.bar(df.index, df["iex_volume"],   label="IEX",                  alpha=0.9, color="darkorange")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.tick_params(axis="x", rotation=35)
        ax.set_ylabel("Shares")
        ax.legend()
        ax.set_title("Daily Volume")

        # Scatter
        ax2 = axes[1]
        ax2.scatter(df["yahoo_volume"], df["iex_volume"], alpha=0.7, edgecolors="k", linewidths=0.4)
        m, b = df["yahoo_volume"].cov(df["iex_volume"]) / df["yahoo_volume"].var(), 0
        xs = df["yahoo_volume"].sort_values()
        # simple OLS line
        slope, intercept, *_ = stats.linregress(df["yahoo_volume"], df["iex_volume"])
        ax2.plot(xs, slope * xs + intercept, "r--", linewidth=1.2,
                 label=f"OLS fit  (r={pearson_r:.3f})")
        ax2.set_xlabel("Yahoo volume (total market)")
        ax2.set_ylabel("IEX volume")
        ax2.set_title("Scatter: IEX vs. Yahoo volume")
        ax2.legend()

        plt.tight_layout()
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="IEX vs. Yahoo Finance volume correlation")
    parser.add_argument("symbol", help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--start", required=True, metavar="YYYY-MM-DD", help="Start date (inclusive)")
    parser.add_argument("--end",   required=True, metavar="YYYY-MM-DD", help="End date (inclusive)")
    parser.add_argument("--plot",  action="store_true", help="Show time-series and scatter plots")
    args = parser.parse_args()

    if not os.path.exists(DB_PATH):
        sys.exit("No database found. Run 'python main.py ingest' first.")

    run(args.symbol.upper(), args.start, args.end, args.plot)


if __name__ == "__main__":
    main()
