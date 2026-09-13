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
    python levels_chart.py <TICKER>
    python levels_chart.py --save                  # save PNG without showing
    python levels_chart.py --no-delta              # price + levels only
"""

import argparse
import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "cache", "market_data.duckdb")

WATCHLIST = ["AAPL"]

# Derived from cumulative delta + absorption analysis, updated Jul 7 2026 (data thru Jul 6).
LEVELS = {
}

# High-conviction absorption signals: score ≥ 50 or large next-day follow-through
SIGNALS = {
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
