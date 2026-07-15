#!/usr/bin/env python3
"""
Squeeze Pressure Screener

Headline short % understates mechanical pressure when institutions hold most
of the float. This tool computes short interest as a fraction of the EFFECTIVE
float (float minus institutional holdings) to surface setups where short
covering is structurally difficult — not just large in absolute terms.

Usage:
    python squeeze_screener.py                    # default watchlist
    python squeeze_screener.py AAPL MSFT NVDA    # custom symbols
    python squeeze_screener.py FDS --detail       # full breakdown
    python squeeze_screener.py --min-real 30      # filter by real short %
"""

import argparse
import sys

DEFAULT_WATCHLIST = ["VIRT", "WRB", "RPM", "FDS", "GDDY", "IBM", "OSUR", "INTU", "NVO", "MBC", "AI"]

# Signal thresholds — all three must be met for *** flag
THRESH_REAL_SHORT = 40.0   # % of effective float
THRESH_INST_PCT   = 70.0   # % institutionally held
THRESH_DAYS_COVER =  4.0   # official days to cover


def _safe(v, default=0.0):
    if v is None:
        return default
    if isinstance(v, float) and v != v:   # NaN
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def fetch_metrics(symbol: str):
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance not installed — run: pip install yfinance")

    try:
        info = yf.Ticker(symbol).info
    except Exception as e:
        print(f"  [{symbol}] fetch error: {e}")
        return None

    price        = _safe(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"))
    float_sh     = _safe(info.get("floatShares"))
    sh_short     = _safe(info.get("sharesShort"))
    sh_short_pm  = _safe(info.get("sharesShortPriorMonth"))
    inst_pct_raw = _safe(info.get("heldPercentInstitutions"))  # may exceed 1.0 via short lending
    insider_pct  = _safe(info.get("heldPercentInsiders"))
    short_ratio  = _safe(info.get("shortRatio"))     # official days to cover
    avg_vol      = _safe(info.get("averageDailyVolume10Day") or info.get("averageVolume"))
    official_pct = _safe(info.get("shortPercentOfFloat"))   # 0.0–1.0

    if float_sh <= 0 or sh_short <= 0:
        return None

    # ── Core calculation ────────────────────────────────────────────────────
    # Institutions holding long positions are largely passive (ETFs, index
    # funds, pension funds). Stripping them out reveals the float that
    # actually trades — and what shorts must compete for when covering.
    #
    # When inst_pct > 100%: ownership data includes double-counted lent shares.
    # The excess (inst_pct − 1.0) × shares_outstanding represents the minimum
    # structural short overhang. eff_float = that overhang (the only supply
    # shorts can realistically buy from, since institutions are fully deployed).
    sh_out = _safe(info.get("sharesOutstanding")) or float_sh
    if inst_pct_raw > 1.0:
        # Structural float = overhang from double-counted lent shares
        eff_float = max((inst_pct_raw - 1.0) * sh_out, float_sh * 0.005)
        inst_pct  = 1.0   # treat as fully locked for SPI scoring
    else:
        inst_pct  = min(inst_pct_raw, 0.99)
        eff_float = float_sh * (1.0 - inst_pct)
    real_short_pct  = (sh_short / eff_float) * 100.0
    real_days       = (sh_short / avg_vol) if avg_vol > 0 else 0.0

    short_chg     = sh_short - sh_short_pm
    short_chg_pct = (short_chg / sh_short_pm * 100.0) if sh_short_pm > 0 else 0.0

    # ── Squeeze Pressure Index (0–100) ──────────────────────────────────────
    # 45%  real short as % of effective float   — primary pressure gauge
    # 25%  institutional lock-up                — how much float is frozen
    # 20%  days to cover                        — time urgency for shorts
    # 10%  short building momentum              — fuel being added
    real_score  = min(100.0, real_short_pct * 2.0)          # 50% real = 100
    inst_score  = min(100.0, inst_pct / 0.90 * 100.0)       # 90% inst = 100
    days_score  = min(100.0, short_ratio * 10.0)             # 10d cover = 100
    fuel_score  = min(100.0, max(0.0, short_chg_pct * 2.0)) # shorts adding

    spi = (real_score * 0.45 + inst_score * 0.25 +
           days_score * 0.20 + fuel_score * 0.10)

    signal = (real_short_pct >= THRESH_REAL_SHORT and
              inst_pct * 100  >= THRESH_INST_PCT   and
              short_ratio     >= THRESH_DAYS_COVER)

    return {
        "symbol":         symbol.upper(),
        "price":          price,
        "float_m":        float_sh / 1e6,
        "sh_short_m":     sh_short / 1e6,
        "official_pct":   official_pct * 100.0,
        "inst_pct":       inst_pct * 100.0,
        "insider_pct":    insider_pct * 100.0,
        "eff_float_m":    eff_float / 1e6,
        "real_short_pct": real_short_pct,
        "official_days":  short_ratio,
        "real_days":      real_days,
        "short_chg_m":    short_chg / 1e6,
        "short_chg_pct":  short_chg_pct,
        "spi":            spi,
        "signal":         signal,
    }


def print_table(rows: list, min_real: float = 0.0) -> None:
    rows = [r for r in rows if r["real_short_pct"] >= min_real]
    rows.sort(key=lambda r: r["spi"], reverse=True)

    W = 114
    print(f"\n{'═' * W}")
    print(f"  Squeeze Pressure Screener  ·  Short Interest vs Effective Float")
    print(f"  {'─' * W}")
    print(f"  {'':>5}  {'Price':>7}  {'Float':>7}  {'Short':>7}  "
          f"{'Offcl%':>7}  {'Inst%':>6}  {'EffFlt':>7}  "
          f"{'RealShrt%':>10}  {'OffDays':>8}  {'ShrtChg':>14}  {'SPI':>6}  {'':>6}")
    print(f"  {'─' * W}")

    for r in rows:
        chg = f"{r['short_chg_m']:+.2f}M ({r['short_chg_pct']:+.0f}%)"
        flag = " ***" if r["signal"] else ""
        print(
            f"  {r['symbol']:>5}  ${r['price']:>6.2f}  {r['float_m']:>6.1f}M  "
            f"{r['sh_short_m']:>6.2f}M  "
            f"{r['official_pct']:>6.1f}%  {r['inst_pct']:>5.1f}%  {r['eff_float_m']:>6.2f}M  "
            f"{r['real_short_pct']:>9.1f}%  {r['official_days']:>7.1f}d  "
            f"{chg:>14}  {r['spi']:>5.1f}{flag}"
        )

    print(f"  {'─' * W}")
    print(f"  Real Short % = Short Interest ÷ (Float × (1 − Inst%))  ←  the number that matters")
    print(f"  SPI = 45% real-short + 25% inst-lockup + 20% days-cover + 10% short-buildup")
    print(f"  *** = real short ≥{THRESH_REAL_SHORT:.0f}%  AND  inst ≥{THRESH_INST_PCT:.0f}%  "
          f"AND  days ≥{THRESH_DAYS_COVER:.0f}")
    print(f"  Data: Yahoo Finance / FINRA  (~2–4 week reporting lag on short interest)\n")


def print_detail(r: dict) -> None:
    W = 62
    covering = r["short_chg_m"] < 0
    direction = "covering ↓" if covering else "adding ↑"

    print(f"\n{'═' * W}")
    print(f"  {r['symbol']}  ·  ${r['price']:.2f}")
    print(f"  {'─' * W}")
    print(f"  FLOAT STRUCTURE")
    print(f"    Reported float         : {r['float_m']:.2f}M shares")
    print(f"    Institutional held     : {r['inst_pct']:.1f}%  "
          f"({r['float_m'] * r['inst_pct'] / 100:.2f}M shares locked)")
    print(f"    Insider held           : {r['insider_pct']:.1f}%")
    print(f"    Effective float        : {r['eff_float_m']:.2f}M shares  ← what shorts compete for")
    print()
    print(f"  SHORT INTEREST")
    print(f"    Shares short           : {r['sh_short_m']:.2f}M")
    print(f"    Official short %       : {r['official_pct']:.1f}%  (of reported float)")
    print(f"    Real short %           : {r['real_short_pct']:.1f}%  (of effective float)  ←")
    print(f"    Official days to cover : {r['official_days']:.1f}d")
    print(f"    Real days to cover     : {r['real_days']:.1f}d  (based on avg volume)")
    print()
    print(f"  SHORT CHANGE  (month-over-month)")
    print(f"    {r['short_chg_m']:+.2f}M shares  ({r['short_chg_pct']:+.1f}%)  —  {direction}")
    print()
    print(f"  SQUEEZE PRESSURE INDEX  :  {r['spi']:.1f} / 100")

    if r["signal"]:
        print(f"\n  *** MECHANICAL BREAK CONDITIONS MET ***")
        print(f"      Real short ≥ {THRESH_REAL_SHORT:.0f}%   ✓  ({r['real_short_pct']:.1f}%)")
        print(f"      Inst%  ≥ {THRESH_INST_PCT:.0f}%      ✓  ({r['inst_pct']:.1f}%)")
        print(f"      Days   ≥ {THRESH_DAYS_COVER:.0f}          ✓  ({r['official_days']:.1f}d)")
    else:
        misses = []
        if r["real_short_pct"] < THRESH_REAL_SHORT:
            misses.append(f"real short {r['real_short_pct']:.1f}% < {THRESH_REAL_SHORT:.0f}%")
        if r["inst_pct"] < THRESH_INST_PCT:
            misses.append(f"inst {r['inst_pct']:.1f}% < {THRESH_INST_PCT:.0f}%")
        if r["official_days"] < THRESH_DAYS_COVER:
            misses.append(f"days {r['official_days']:.1f} < {THRESH_DAYS_COVER:.0f}")
        print(f"  Signal not triggered: {';  '.join(misses)}")

    print(f"{'═' * W}\n")


def _fetch_daily_vol(symbol: str) -> float:
    """Annualised historical vol from 60-day close returns (yfinance)."""
    try:
        import yfinance as yf
        import math
        hist = yf.download(symbol, period="90d", progress=False, auto_adjust=True)
        if hist.empty:
            return 0.0
        if hasattr(hist.columns, "levels"):
            closes = hist["Close"][symbol].dropna()
        else:
            closes = hist["Close"].dropna()
        if len(closes) < 5:
            return 0.0
        rets = closes.pct_change().dropna()
        return float(rets.std() * math.sqrt(252))
    except Exception:
        return 0.0


def print_squeeze_impact(m: dict, gamma: float = 0.5) -> None:
    """
    Scenario grid: if X% of shorts cover over D days, what's the expected
    permanent price impact?

    Model: Almgren-Chriss square-root permanent impact
        ΔP_day = γ × σ_d × P × √(Q_day / ADV)
    where
        γ     = permanent impact coefficient (default 0.5 — moderate feedback)
        σ_d   = daily volatility (annualised_vol / √252)
        Q_day = shares covered per day
        ADV   = average daily volume

    Total ΔP = Σ over D days.  Assumes uniform covering pace.

    This is a mechanical lower-bound.  Momentum and sentiment feedback
    (the actual "squeeze" dynamic) can amplify it by 2–5×.
    """
    import math

    sym     = m["symbol"]
    price   = m["price"]
    sh_sh   = m["sh_short_m"] * 1e6
    adv     = _safe(None, 0.0)   # fetch below
    ann_vol = _fetch_daily_vol(sym)

    try:
        import yfinance as yf
        info = yf.Ticker(sym).info
        adv = _safe(info.get("averageDailyVolume10Day") or info.get("averageVolume"))
    except Exception:
        adv = sh_sh / max(m["official_days"], 1.0)

    if adv <= 0:
        print(f"  [{sym}] Cannot compute impact — no ADV data")
        return
    if ann_vol <= 0:
        ann_vol = 0.30   # fallback: 30% vol
        print(f"  [{sym}] Historical vol unavailable — using 30% default")

    sigma_d = ann_vol / math.sqrt(252)

    cover_pcts = [10, 25, 50, 75, 100]
    day_buckets = [2, 5, 10, 20]

    W = 72
    print(f"\n{'═' * W}")
    print(f"  Short Squeeze Impact Model — {sym}")
    print(f"  {'─' * W}")
    print(f"  Shares short   : {sh_sh/1e6:.2f}M")
    print(f"  Avg daily vol  : {adv/1e3:.0f}K  ({sh_sh/adv:.1f}× ADV to cover 100%)")
    print(f"  Daily vol (σ_d): {sigma_d*100:.2f}%  (annual {ann_vol*100:.0f}%)")
    print(f"  Current price  : ${price:.2f}")
    print(f"  Impact coeff γ : {gamma}  (Almgren sq-root permanent impact)")
    print(f"  {'─' * W}")
    print(f"  ΔP = γ × σ_d × P × √(Q_day / ADV)  summed over D days")
    print(f"  {'─' * W}")

    # Header
    col_w = 14
    print(f"  {'% Covered':>10}  {'Shares':>8}", end="")
    for d in day_buckets:
        header = f"{d}d target"
        print(f"  {header:>{col_w}}", end="")
    print()
    print(f"  {'─' * 10}  {'─' * 8}", end="")
    for _ in day_buckets:
        print(f"  {'─' * col_w}", end="")
    print()

    for pct in cover_pcts:
        covered = sh_sh * pct / 100.0
        print(f"  {pct:>9}%  {covered/1e6:>6.2f}M", end="")
        for d in day_buckets:
            q_day = covered / d
            participation = q_day / adv
            # square-root permanent impact per day, summed over d days
            daily_impact_pct = gamma * sigma_d * math.sqrt(participation)
            total_pct = daily_impact_pct * d
            target = price * (1 + total_pct)
            flag = " *" if participation > 1.5 else ""
            cell = f"${target:.1f} (+{total_pct*100:.1f}%){flag}"
            print(f"  {cell:>{col_w}}", end="")
        print()

    print(f"  {'─' * W}")
    print(f"  * participation > 1.5× ADV — highly disruptive; momentum amplification likely")
    print(f"  Model gives mechanical price floor.  Add 1.5–3× for full squeeze momentum.")
    print(f"\n  Official days to cover at current ADV: {m['official_days']:.1f}d")
    print(f"  If forced to cover in 2d: {sh_sh/adv/2*100:.0f}% of ADV consumed per day")
    print(f"  Effective float (excl. inst): {m['eff_float_m']:.1f}M shares")
    print(f"{'═' * W}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Squeeze pressure screener — short interest vs effective float",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("symbols", nargs="*",
                        help="Ticker symbols (default: built-in watchlist)")
    parser.add_argument("--detail", action="store_true",
                        help="Print full breakdown for each symbol")
    parser.add_argument("--min-real", type=float, default=0.0, metavar="PCT",
                        help="Only show symbols with real short %% ≥ this (default: 0)")
    parser.add_argument("--impact", action="store_true",
                        help="Print squeeze price-impact scenario grid for each symbol")
    parser.add_argument("--gamma", type=float, default=0.5, metavar="G",
                        help="Almgren impact coefficient γ (default: 0.5)")
    args = parser.parse_args()

    symbols = [s.upper() for s in args.symbols] if args.symbols else DEFAULT_WATCHLIST
    print(f"Fetching squeeze metrics for: {', '.join(symbols)} ...")

    rows = []
    for sym in symbols:
        r = fetch_metrics(sym)
        if r:
            rows.append(r)
        else:
            print(f"  [{sym}] skipped — insufficient data")

    if not rows:
        sys.exit("No data retrieved.")

    print_table(rows, min_real=args.min_real)

    if args.detail:
        for r in sorted(rows, key=lambda x: x["spi"], reverse=True):
            print_detail(r)

    if args.impact:
        for r in sorted(rows, key=lambda x: x["spi"], reverse=True):
            print_squeeze_impact(r, gamma=args.gamma)


if __name__ == "__main__":
    main()
