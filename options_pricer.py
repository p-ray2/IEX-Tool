"""
Options Pricer — Black-Scholes fair value and bid/ask spread analysis.

Solves IV from market prices, flags where fair value sits inside a wide
spread, and suggests limit order placement.

Commands:
    scout     Fetch live chain from Yahoo Finance, rank contracts, drill into any
    analyze   Full analysis given bid/ask  (IV solved from mid, or pass --iv)
    price     Price + greeks from a given IV
    solve     Back-solve IV from a single market price
    table     Strike comparison table at a given IV
    squeeze   Scenario grid: BS option value at target prices × time horizons
    (none)    Interactive prompt mode

Usage:
    python options_pricer.py scout   WRB --call --min-dte 21 --max-dte 90
    python options_pricer.py scout   FDS --call --min-delta 0.25 --max-delta 0.65
    python options_pricer.py analyze --ticker WRB260717C00070000 --stock 68.52 --bid 2.10 --ask 2.80
    python options_pricer.py price   --ticker WRB260717C00070000 --stock 68.52 --iv 0.26
    python options_pricer.py solve   --ticker FDS260717C00230000 --stock 226.00 --market-price 12.50
    python options_pricer.py table   --symbol WRB --call --stock 68.52 --expiry 2026-07-17 --iv 0.26
    python options_pricer.py squeeze FDS260718C00235000 --entry 8.50
    python options_pricer.py squeeze FDS260718C00235000 --entry 8.50 --iv-bump 0.10
    python options_pricer.py squeeze FDS260718C00235000 --stock 223.68 --iv 0.52 --targets 230,240,250,260,280
    python options_pricer.py
"""

import argparse
import math
import re
import sys
from datetime import date

from scipy.optimize import brentq
from scipy.stats import norm

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_RATE = 0.044

# Approximate annual dividend yields — override with --div
DIV_TABLE = {
    "WRB":  0.005,
    "FDS":  0.007,
    "RPM":  0.017,
    "VIRT": 0.030,
    "GDDY": 0.000,
    "IBM":  0.030,
}


# ── OCC ticker parser ─────────────────────────────────────────────────────────

def parse_ticker(ticker: str):
    """WRB260618C00067500  →  ('WRB', date(2026,6,18), 'C', 67.50)"""
    m = re.match(r'^([A-Z]+)(\d{6})([CP])(\d{8})$', ticker.strip().upper())
    if not m:
        raise ValueError(f"Cannot parse OCC ticker: {ticker!r}\n"
                         "Expected format: SYMBOL + YYMMDD + C/P + 8-digit strike*1000")
    symbol   = m.group(1)
    yy, mo, dd = int(m.group(2)[:2]), int(m.group(2)[2:4]), int(m.group(2)[4:])
    expiry   = date(2000 + yy, mo, dd)
    opt_type = m.group(3)
    strike   = int(m.group(4)) / 1000.0
    return symbol, expiry, opt_type, strike


# ── Black-Scholes core ────────────────────────────────────────────────────────

def _d1d2(S, K, T, r, sigma, q):
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return d1, d1 - sigma * math.sqrt(T)


def bs_price(S, K, T, r, sigma, q=0.0, opt_type="C") -> float:
    if T <= 0:
        return max(0.0, S - K if opt_type == "C" else K - S)
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    if opt_type == "C":
        return (S * math.exp(-q * T) * norm.cdf(d1)
                - K * math.exp(-r * T) * norm.cdf(d2))
    return (K * math.exp(-r * T) * norm.cdf(-d2)
            - S * math.exp(-q * T) * norm.cdf(-d1))


def bs_greeks(S, K, T, r, sigma, q=0.0, opt_type="C") -> dict:
    if T <= 0:
        intrinsic = max(0.0, S - K if opt_type == "C" else K - S)
        return dict(price=intrinsic, delta=float(intrinsic > 0),
                    gamma=0.0, theta=0.0, vega=0.0)
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    price  = bs_price(S, K, T, r, sigma, q, opt_type)
    delta  = (math.exp(-q * T) * norm.cdf(d1) if opt_type == "C"
              else -math.exp(-q * T) * norm.cdf(-d1))
    gamma  = math.exp(-q * T) * norm.pdf(d1) / (S * sigma * math.sqrt(T))
    base_t = -(S * math.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
    if opt_type == "C":
        theta = (base_t - r * K * math.exp(-r * T) * norm.cdf(d2)
                 + q * S * math.exp(-q * T) * norm.cdf(d1)) / 365
    else:
        theta = (base_t + r * K * math.exp(-r * T) * norm.cdf(-d2)
                 - q * S * math.exp(-q * T) * norm.cdf(-d1)) / 365
    vega = S * math.exp(-q * T) * norm.pdf(d1) * math.sqrt(T) / 100
    return dict(price=price, delta=delta, gamma=gamma, theta=theta, vega=vega)


# ── IV solver ─────────────────────────────────────────────────────────────────

def solve_iv(market_price, S, K, T, r, q=0.0, opt_type="C"):
    """Return implied vol or None if unsolvable."""
    if T <= 0:
        return None
    floor = max(0.0, (S * math.exp(-q * T) - K * math.exp(-r * T))
                if opt_type == "C" else
                (K * math.exp(-r * T) - S * math.exp(-q * T)))
    if market_price <= floor + 1e-8:
        return None
    try:
        return brentq(
            lambda v: bs_price(S, K, T, r, v, q, opt_type) - market_price,
            1e-5, 20.0, xtol=1e-6, maxiter=500
        )
    except ValueError:
        return None


# ── helpers ───────────────────────────────────────────────────────────────────

def _tte(expiry: date) -> float:
    return max((expiry - date.today()).days, 0) / 365.0


def _spread_tier(pct) -> str:
    if pct < 5:   return "tight"
    if pct < 15:  return "normal"
    if pct < 30:  return "wide"
    return "very wide"


def _suggest_limits(bid, ask, fair):
    mid    = (bid + ask) / 2
    spread = ask - bid
    pct    = spread / mid * 100 if mid > 0 else 100
    if pct < 15:
        buy_limit  = round(mid + 0.01, 2)
        sell_limit = round(mid - 0.01, 2)
    elif pct < 30:
        buy_limit  = round(min(fair + 0.05, mid + spread * 0.15), 2)
        sell_limit = round(max(fair - 0.05, mid - spread * 0.15), 2)
    else:
        buy_limit  = round(min(fair + 0.10, mid + spread * 0.20), 2)
        sell_limit = round(max(fair - 0.10, mid - spread * 0.20), 2)
    return min(buy_limit, ask), max(sell_limit, bid)


def _moneyness(S, K, opt_type) -> str:
    diff_pct = (S - K) / K * 100
    if opt_type == "C":
        if diff_pct > 3:   return f"ITM  (+{diff_pct:.1f}%)"
        if diff_pct < -3:  return f"OTM  ({diff_pct:.1f}%)"
        return f"ATM  ({diff_pct:+.1f}%)"
    if diff_pct < -3:  return f"ITM  ({diff_pct:.1f}%)"
    if diff_pct > 3:   return f"OTM  (+{diff_pct:.1f}%)"
    return f"ATM  ({diff_pct:+.1f}%)"


# ── display ───────────────────────────────────────────────────────────────────

W = 64

def _header(symbol, expiry, opt_type, strike, S, T):
    days = round(T * 365)
    type_label = "Call" if opt_type == "C" else "Put"
    print(f"\n{'═'*W}")
    print(f"  {symbol}  {type_label}  ${strike:.2f}  ·  "
          f"{expiry.strftime('%b %d %Y')}  ·  {days}d to expiry")
    print(f"  Stock: ${S:.2f}   Moneyness: {_moneyness(S, strike, opt_type)}")
    print(f"{'═'*W}")


def _print_greeks(g):
    print(f"  GREEKS")
    print(f"  {'─'*40}")
    print(f"  Delta  : {g['delta']:+.4f}   (${g['delta']:+.3f} per $1 stock move)")
    print(f"  Gamma  : {g['gamma']:+.5f}   (delta shift per $1)")
    print(f"  Theta  : {g['theta']:+.4f}/day   (${abs(g['theta']):.3f} time decay/day)")
    print(f"  Vega   : {g['vega']:+.4f}/pt   (${g['vega']:.3f} per 1% IV move)")


def _print_scenarios(g, S, K, opt_type, entry=None):
    be = (K + g["price"] if opt_type == "C" else K - g["price"])
    ref = entry if entry is not None else g["price"]
    moves = [-0.15, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.15]
    print(f"\n  SCENARIOS AT EXPIRY  (entry: ${ref:.2f}   break-even: ${be:.2f})")
    print(f"  {'─'*50}")
    print(f"  {'Stock':>8}  {'Move':>7}  {'Value':>7}  {'P&L':>8}  {'Return':>8}")
    for m in moves:
        sp  = round(S * (1 + m), 2)
        val = max(0.0, sp - K if opt_type == "C" else K - sp)
        pnl = val - ref
        ret = pnl / ref * 100 if ref > 0 else 0.0
        tag = ""
        if abs(sp - S)  < S * 0.005: tag = " ◄ now"
        if abs(sp - be) < S * 0.005: tag = " ◄ B/E"
        pct_move = f"{m*100:+.0f}%"
        print(f"  ${sp:>7.2f}  {pct_move:>7}  ${val:>6.2f}  ${pnl:>+7.2f}  {ret:>+7.1f}%{tag}")


def cmd_analyze(args):
    symbol, expiry, opt_type, strike = parse_ticker(args.ticker)
    S   = args.stock
    bid = args.bid
    ask = args.ask
    r   = args.rate
    q   = args.div if args.div is not None else DIV_TABLE.get(symbol, 0.0)
    T   = _tte(expiry)
    mid = (bid + ask) / 2

    _header(symbol, expiry, opt_type, strike, S, T)

    spread     = ask - bid
    spread_pct = spread / mid * 100
    tier       = _spread_tier(spread_pct)

    print(f"  MARKET QUOTE")
    print(f"  {'─'*40}")
    vol_str = f"   Vol: {args.volume}" if args.volume else ""
    oi_str  = f"   OI: {args.oi}"     if args.oi     else ""
    print(f"  Bid: ${bid:.2f}   Ask: ${ask:.2f}   Mid: ${mid:.2f}{vol_str}{oi_str}")
    bar_fill = min(int(spread_pct / 50 * 30), 30)
    bar = "█" * bar_fill + "░" * (30 - bar_fill)
    print(f"  Spread: ${spread:.2f}  ({spread_pct:.1f}% of mid)  [{tier}]")
    print(f"  [{bar}]  ← 0% {'':>8} 50% →")
    print()

    # Resolve IV
    iv_source = "given"
    iv = args.iv
    if iv is None:
        iv = solve_iv(mid, S, strike, T, r, q, opt_type)
        iv_source = "solved from mid"
    if iv is None:
        print("  [!] Cannot solve IV from mid — check inputs")
        return

    g         = bs_greeks(S, strike, T, r, iv, q, opt_type)
    fair      = g["price"]
    intrinsic = max(0.0, S - strike if opt_type == "C" else strike - S)
    time_val  = fair - intrinsic
    vs_mid    = fair - mid

    print(f"  FAIR VALUE  (IV: {iv*100:.2f}%  [{iv_source}])")
    print(f"  {'─'*40}")
    print(f"  BS Fair Value : ${fair:.2f}")
    print(f"    Intrinsic   : ${intrinsic:.2f}")
    print(f"    Time value  : ${time_val:.2f}")
    diff_label = ("underpriced vs mid" if vs_mid > 0.05
                  else "overpriced vs mid" if vs_mid < -0.05
                  else "fairly priced vs mid")
    print(f"  vs Mid        : ${vs_mid:+.2f}  ({diff_label})")

    # Spread position chart
    if spread > 0:
        pos = (fair - bid) / spread * 100
        pos = max(0, min(100, pos))
        bar2 = "─" * int(pos / 100 * 30)
        print(f"\n  Where fair value sits in the spread:")
        print(f"  Bid ${bid:.2f} [{bar2:30s}|] Ask ${ask:.2f}")
        print(f"  {'':>9} {'':>{int(pos/100*30)}}↑ fair ${fair:.2f}")
    print()

    buy_lim, sell_lim = _suggest_limits(bid, ask, fair)
    print(f"  LIMIT ORDER SUGGESTIONS  [{tier} spread]")
    print(f"  {'─'*40}")
    print(f"  To BUY  :  ${buy_lim:.2f}   ({(buy_lim-bid)/spread*100:.0f}% of spread from bid)")
    print(f"  To SELL :  ${sell_lim:.2f}   ({(ask-sell_lim)/spread*100:.0f}% of spread from ask)")
    if tier in ("wide", "very wide"):
        print(f"  [!] Spread is {tier} — mid fills slowly. Start at mid,")
        print(f"      improve by $0.05 increments if needed. Avoid market orders.")
    else:
        print(f"  [✓] Spread is {tier} — mid should fill.")
    print()

    _print_greeks(g)
    _print_scenarios(g, S, strike, opt_type)
    print(f"\n{'═'*W}\n")


def cmd_price(args):
    symbol, expiry, opt_type, strike = parse_ticker(args.ticker)
    S = args.stock
    r = args.rate
    q = args.div if args.div is not None else DIV_TABLE.get(symbol, 0.0)
    T = _tte(expiry)

    _header(symbol, expiry, opt_type, strike, S, T)

    g         = bs_greeks(S, strike, T, r, args.iv, q, opt_type)
    intrinsic = max(0.0, S - strike if opt_type == "C" else strike - S)

    print(f"  IV: {args.iv*100:.2f}%   Div yield: {q*100:.2f}%   Rate: {r*100:.2f}%")
    print()
    print(f"  Price         : ${g['price']:.2f}")
    print(f"    Intrinsic   : ${intrinsic:.2f}")
    print(f"    Time value  : ${g['price']-intrinsic:.2f}")
    print()
    _print_greeks(g)
    _print_scenarios(g, S, strike, opt_type)
    print(f"\n{'═'*W}\n")


def cmd_solve(args):
    symbol, expiry, opt_type, strike = parse_ticker(args.ticker)
    S  = args.stock
    r  = args.rate
    q  = args.div if args.div is not None else DIV_TABLE.get(symbol, 0.0)
    T  = _tte(expiry)
    mp = args.market_price

    iv = solve_iv(mp, S, strike, T, r, q, opt_type)
    if iv is None:
        print(f"\n  [!] Cannot solve IV for ${mp:.2f} — may be at or below intrinsic\n")
        return

    g = bs_greeks(S, strike, T, r, iv, q, opt_type)
    _header(symbol, expiry, opt_type, strike, S, T)
    print(f"  Market price  : ${mp:.2f}")
    print(f"  Implied vol   : {iv*100:.2f}%")
    print(f"  BS cross-check: ${g['price']:.2f}  (should match market price)")
    print()
    _print_greeks(g)
    print(f"\n{'═'*W}\n")


def cmd_table(args):
    symbol    = args.symbol.upper()
    opt_type  = "C" if args.call else "P"
    S         = args.stock
    iv        = args.iv
    r         = args.rate
    q         = args.div if args.div is not None else DIV_TABLE.get(symbol, 0.0)

    try:
        expiry = date.fromisoformat(args.expiry)
    except ValueError:
        sys.exit(f"Invalid expiry date: {args.expiry!r}  (use YYYY-MM-DD)")

    T    = _tte(expiry)
    days = round(T * 365)

    if args.strikes:
        strikes = [float(k) for k in args.strikes.split(",")]
    else:
        step   = 2.5 if S < 150 else 5.0
        atm    = round(S / step) * step
        strikes = [atm + i * step for i in range(-4, 7) if atm + i * step > 0]

    type_label = "CALLS" if opt_type == "C" else "PUTS"
    print(f"\n{'═'*76}")
    print(f"  {symbol}  {type_label}  ·  {expiry.strftime('%b %d %Y')}  "
          f"({days}d)  ·  S=${S:.2f}  IV={iv*100:.1f}%  q={q*100:.1f}%")
    print(f"  {'─'*76}")
    print(f"  {'Strike':>7}  {'Price':>7}  {'Delta':>7}  {'Gamma':>7}  "
          f"{'Theta/d':>8}  {'Vega/pt':>8}  {'B/E':>8}  {'Moneyness':>5}")
    print(f"  {'─'*76}")

    for K in strikes:
        g  = bs_greeks(S, K, T, r, iv, q, opt_type)
        be = K + g["price"] if opt_type == "C" else K - g["price"]
        mn = _moneyness(S, K, opt_type)
        nearest = abs(K - S) == min(abs(k - S) for k in strikes)
        marker = " ◄" if nearest else ""
        print(f"  ${K:>6.2f}  ${g['price']:>6.2f}  {g['delta']:>+7.3f}  "
              f"{g['gamma']:>7.5f}  ${g['theta']:>7.4f}  ${g['vega']:>7.4f}  "
              f"${be:>7.2f}  {mn}{marker}")

    print(f"  {'─'*76}")
    print(f"  Theta = $/day  |  Vega = $/1pt IV  |  ◄ = nearest ATM\n")


def cmd_scout(args):
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance not installed. Run: pip install yfinance")

    symbol   = args.symbol.upper()
    opt_type = "P" if getattr(args, "put", False) else "C"
    r        = args.rate
    q        = args.div if args.div is not None else DIV_TABLE.get(symbol, 0.0)
    today    = date.today()

    print(f"\n  Fetching {symbol} from Yahoo Finance...")
    yt = yf.Ticker(symbol)
    S  = None

    # 1) yf.download — most stable across yfinance versions
    try:
        dl = yf.download(symbol, period="5d", progress=False, auto_adjust=True)
        if not dl.empty:
            # MultiIndex columns when auto_adjust=True on newer builds
            close = dl["Close"]
            if hasattr(close, "iloc"):
                val = float(close.iloc[-1])
                if val and val == val:
                    S = val
    except Exception:
        pass

    # 2) Ticker.history with a wider window
    if S is None:
        try:
            hist = yt.history(period="1mo")
            if not hist.empty:
                S = float(hist["Close"].iloc[-1])
        except Exception:
            pass

    # 3) fast_info / info dict
    if S is None:
        try:
            fi = yt.fast_info
            for key in ("last_price", "regularMarketPrice"):
                val = fi.get(key)
                if val and float(val) > 0:
                    S = float(val)
                    break
        except Exception:
            pass

    if S is None or S != S:
        sys.exit(f"Could not fetch a price for {symbol}. "
                 "Check the symbol or try: pip install --upgrade yfinance")

    print(f"  {symbol} last price: ${S:.2f}  (15-min delay)")

    expiries = yt.options
    if not expiries:
        sys.exit(f"No options data found for {symbol}")

    valid = []
    for es in expiries:
        exp = date.fromisoformat(es)
        dte = (exp - today).days
        if args.min_dte <= dte <= args.max_dte:
            valid.append((es, exp, dte))

    if not valid:
        sys.exit(f"No expiries between {args.min_dte}–{args.max_dte} DTE. "
                 f"Available: {', '.join(expiries)}")

    type_label = "CALLS" if opt_type == "C" else "PUTS"
    print(f"  Scanning {type_label} across {len(valid)} expir{'y' if len(valid)==1 else 'ies'}...")

    contracts = []
    for es, exp, dte in valid:
        try:
            chain = yt.option_chain(es)
            df    = chain.calls if opt_type == "C" else chain.puts
        except Exception:
            continue

        T = dte / 365.0
        if T <= 0:
            continue

        for _, row in df.iterrows():
            bid = float(row.get("bid") or 0)
            ask = float(row.get("ask") or 0)
            if bid <= 0 or ask <= 0 or ask <= bid:
                continue

            mid    = (bid + ask) / 2.0
            strike = float(row["strike"])
            _oi    = row.get("openInterest");    oi   = 0 if not _oi  or _oi  != _oi  else int(_oi)
            _vol   = row.get("volume");          vol  = 0 if not _vol or _vol != _vol else int(_vol)
            _iv    = row.get("impliedVolatility"); iv_yh = 0.0 if not _iv or _iv != _iv else float(_iv)

            iv = iv_yh if 0.05 <= iv_yh <= 5.0 else solve_iv(mid, S, strike, T, r, q, opt_type)
            if iv is None:
                continue

            g     = bs_greeks(S, strike, T, r, iv, q, opt_type)
            fair  = g["price"]
            delta = abs(g["delta"])

            if not (args.min_delta <= delta <= args.max_delta):
                continue

            spread     = ask - bid
            spread_pct = spread / mid * 100

            # Component scores (0–100, higher = better for buying)
            spread_score = max(0.0, 100.0 - spread_pct * 3.0)
            liq_score    = min(100.0, math.log1p(oi) * 8.0 + math.log1p(vol) * 4.0)
            eff_score    = min(100.0, (delta / mid) * 200.0)   # delta per dollar paid
            vs_mid_pct   = (fair - mid) / mid * 100.0 if mid > 0 else 0.0
            value_score  = min(100.0, max(0.0, 50.0 + vs_mid_pct * 2.0))

            composite = (spread_score * 0.30 + liq_score * 0.25
                         + eff_score * 0.25 + value_score * 0.20)

            contracts.append({
                "ticker":        str(row["contractSymbol"]),
                "expiry":        exp,
                "dte":           dte,
                "strike":        strike,
                "bid":           bid,
                "ask":           ask,
                "mid":           mid,
                "fair":          fair,
                "iv":            iv,
                "delta":         g["delta"],
                "theta":         g["theta"],
                "oi":            oi,
                "volume":        vol,
                "spread_pct":    spread_pct,
                "spread_score":  spread_score,
                "liq_score":     liq_score,
                "eff_score":     eff_score,
                "value_score":   value_score,
                "composite":     composite,
            })

    if not contracts:
        sys.exit("No contracts matched the filters. Try widening --min-delta/--max-delta or DTE range.")

    contracts.sort(key=lambda x: x["composite"], reverse=True)
    show_n = min(args.top, len(contracts))

    # ATM strike in visible slice
    atm_strike = min((c["strike"] for c in contracts[:show_n]),
                     key=lambda k: abs(k - S))

    W2 = 92
    print(f"\n{'═'*W2}")
    print(f"  {symbol} {type_label}  ·  S=${S:.2f}  ·  "
          f"δ {args.min_delta:.2f}–{args.max_delta:.2f}  ·  "
          f"{args.min_dte}–{args.max_dte} DTE  ·  top {show_n} of {len(contracts)}")
    print(f"  {'─'*W2}")
    print(f"  {'#':>3}  {'Expiry':>11}  {'DTE':>4}  {'Strike':>7}  "
          f"{'Bid':>5}  {'Ask':>5}  {'Fair':>5}  {'IV':>6}  {'Δ':>5}  "
          f"{'θ/d':>6}  {'Sprd%':>6}  {'OI':>6}  {'Score':>6}")
    print(f"  {'─'*W2}")

    for i, c in enumerate(contracts[:show_n]):
        atm = " ◄" if c["strike"] == atm_strike else ""
        print(f"  {i+1:>3}  {c['expiry'].strftime('%b %d %Y'):>11}  {c['dte']:>4}  "
              f"${c['strike']:>6.2f}  "
              f"${c['bid']:>4.2f}  ${c['ask']:>4.2f}  ${c['fair']:>4.2f}  "
              f"{c['iv']*100:>5.1f}%  {c['delta']:>+5.2f}  "
              f"${c['theta']:>5.3f}  {c['spread_pct']:>5.1f}%  "
              f"{c['oi']:>6,}  {c['composite']:>6.1f}{atm}")

    print(f"  {'─'*W2}")
    print(f"  Score = 30% spread tightness + 25% OI/volume + 25% delta-per-dollar + 20% fair-vs-mid")
    print(f"  ◄ = nearest ATM  |  Data: Yahoo Finance (~15-min delay) — verify live before ordering.\n")

    while True:
        sel = input(f"  Row # to analyze (1–{show_n}), or q to quit: ").strip()
        if sel.lower() in ("q", ""):
            break
        try:
            idx = int(sel) - 1
            if not 0 <= idx < show_n:
                print(f"  Enter 1–{show_n}")
                continue
        except ValueError:
            print("  Invalid input")
            continue

        c = contracts[idx]

        class _A:
            pass
        a         = _A()
        a.ticker  = c["ticker"]
        a.stock   = S
        a.bid     = c["bid"]
        a.ask     = c["ask"]
        a.iv      = c["iv"]
        a.volume  = c["volume"] or None
        a.oi      = c["oi"] or None
        a.rate    = r
        a.div     = q
        cmd_analyze(a)

        again = input("  Analyze another contract? (y/n): ").strip().lower()
        if again != "y":
            break


def cmd_squeeze(args):
    """Scenario grid: BS option value at target prices × time horizons (not just expiry intrinsic)."""
    symbol, expiry, opt_type, strike = parse_ticker(args.ticker)
    r = args.rate
    q = args.div if args.div is not None else DIV_TABLE.get(symbol, 0.0)
    days_to_exp = (expiry - date.today()).days
    T_now = max(days_to_exp, 0) / 365.0
    type_label = "Call" if opt_type == "C" else "Put"

    # Resolve stock price
    S = getattr(args, "stock", None)
    if S is None:
        try:
            import yfinance as yf
            dl = yf.download(symbol, period="5d", progress=False, auto_adjust=True)
            if not dl.empty:
                close = dl["Close"]
                if hasattr(close, "columns"):
                    close = close.iloc[:, 0]
                val = float(close.iloc[-1])
                if val and val == val:
                    S = val
        except Exception:
            pass
        if S is None:
            sys.exit(f"Cannot fetch price for {symbol}. Use --stock.")

    # Resolve IV
    iv = getattr(args, "iv", None)
    entry = getattr(args, "entry", None)
    if iv is None and entry is not None:
        iv = solve_iv(entry, S, strike, T_now, r, q, opt_type)
        if iv is None:
            sys.exit(f"Cannot solve IV from entry ${entry:.2f} — check inputs.")
    if iv is None:
        try:
            import yfinance as yf
            yt = yf.Ticker(symbol)
            chain = yt.option_chain(expiry.isoformat())
            df = chain.calls if opt_type == "C" else chain.puts
            mask = abs(df["strike"] - strike) < 0.01
            if mask.any():
                _iv = df[mask].iloc[0].get("impliedVolatility")
                if _iv and _iv == _iv and 0.05 <= float(_iv) <= 5.0:
                    iv = float(_iv)
                elif float(df[mask].iloc[0].get("bid") or 0) > 0:
                    mid = (float(df[mask].iloc[0]["bid"]) + float(df[mask].iloc[0]["ask"])) / 2
                    iv = solve_iv(mid, S, strike, T_now, r, q, opt_type)
        except Exception:
            pass
    if iv is None:
        sys.exit("Cannot determine IV. Provide --iv or --entry.")

    iv_bump = getattr(args, "iv_bump", 0.0) or 0.0
    ref = entry if entry is not None else bs_price(S, strike, T_now, r, iv, q, opt_type)
    be = strike + ref if opt_type == "C" else strike - ref

    # Build target list
    if getattr(args, "targets", None):
        targets = sorted(float(t) for t in args.targets.split(","))
    else:
        moves = [-0.08, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
        targets = [round(S * (1 + m), 2) for m in moves]

    # Build horizon list (days forward from today, must be < days_to_exp)
    if getattr(args, "horizons", None):
        horizons = sorted(int(h) for h in args.horizons.split(","))
        horizons = [h for h in horizons if 0 < h < days_to_exp]
    else:
        candidates = [10, 20, days_to_exp // 2]
        horizons = sorted(set(h for h in candidates if 0 < h < days_to_exp))

    # Layout
    ncols = len(horizons) + 1
    col_w = 14
    W2 = max(64, 18 + ncols * col_w)

    print(f"\n{'═'*W2}")
    print(f"  {symbol}  {type_label}  ${strike:.2f}  ·  "
          f"{expiry.strftime('%b %d %Y')}  ·  {days_to_exp}d to expiry")
    iv_label = f"IV {iv*100:.1f}%"
    if iv_bump:
        iv_label += f" + {iv_bump*100:.0f}%pt bump → {(iv+iv_bump)*100:.1f}%"
    print(f"  Stock: ${S:.2f}   Entry: ${ref:.2f}   {iv_label}   B/E: ${be:.2f}")
    print(f"{'═'*W2}")

    print(f"\n  SCENARIO GRID  (entry ${ref:.2f}  ·  mid-term = BS  ·  expiry = intrinsic)")
    print(f"  {'─'*W2}")

    # Header
    hdr = f"  {'Stock':>8}  {'Move':>6}"
    for h in horizons:
        label = f"{h}d ({days_to_exp-h}DTE)"
        hdr += f"  {label:^{col_w-2}}"
    hdr += f"  {'At expiry':^{col_w-2}}"
    print(hdr)
    print(f"  {'─'*W2}")

    for target in targets:
        move_pct = (target - S) / S * 100
        marker = "  ◄ now" if abs(target - S) < S * 0.005 else ""
        row = f"  ${target:>7.2f}  {move_pct:>+5.1f}%"
        for h in horizons:
            T_fwd = max(days_to_exp - h, 0) / 365.0
            val = bs_price(target, strike, T_fwd, r, iv + iv_bump, q, opt_type)
            pnl_pct = (val - ref) / ref * 100 if ref > 0 else 0.0
            row += f"  ${val:>5.2f} {pnl_pct:>+6.0f}%"
        intrinsic = max(0.0, target - strike if opt_type == "C" else strike - target)
        pnl_pct = (intrinsic - ref) / ref * 100 if ref > 0 else 0.0
        row += f"  ${intrinsic:>5.2f} {pnl_pct:>+6.0f}%"
        row += marker
        print(row)

    print(f"  {'─'*W2}")
    print(f"  P&L% vs entry ${ref:.2f}  |  Mid-term uses BS with T remaining  |  Expiry = intrinsic")

    # IV sensitivity at +10% stock, 10d from now
    h_sense = min(10, days_to_exp - 1)
    T_sense = max(days_to_exp - h_sense, 0) / 365.0
    sense_tgt = round(S * 1.10, 2)
    print(f"\n  IV SENSITIVITY  (stock +10% at ${sense_tgt:.2f}, {h_sense}d from now)")
    print(f"  {'─'*46}")
    for bump in [-0.10, -0.05, 0.0, 0.10, 0.20]:
        iv_s = max(iv + bump, 0.01)
        val = bs_price(sense_tgt, strike, T_sense, r, iv_s, q, opt_type)
        pnl_pct = (val - ref) / ref * 100 if ref > 0 else 0.0
        label = f"IV {iv_s*100:.0f}% ({bump*100:+.0f}pt)"
        marker = "  ◄ base case" if bump == 0 else ("  ← squeeze expansion" if bump > 0 else "")
        print(f"  {label:>18}  →  ${val:>6.2f}  ({pnl_pct:>+5.0f}%){marker}")

    print(f"\n{'═'*W2}\n")


def interactive():
    print("\n══ Options Pricer ══════════════════════════════════════════")
    print("  Enter option details. Ctrl-C to quit.\n")

    ticker_raw = input("  Option ticker (e.g. WRB260717C00070000), or 'manual': ").strip()

    if ticker_raw.lower() == "manual":
        symbol   = input("  Symbol: ").strip().upper()
        expiry   = date.fromisoformat(input("  Expiry (YYYY-MM-DD): ").strip())
        opt_type = input("  Type (C/P): ").strip().upper()
        strike   = float(input("  Strike: ").strip())
    else:
        symbol, expiry, opt_type, strike = parse_ticker(ticker_raw)
        print(f"  → {symbol}  {'Call' if opt_type=='C' else 'Put'}  "
              f"${strike:.2f}  {expiry.strftime('%b %d %Y')}")

    S = float(input(f"\n  Current stock price for {symbol}: $").strip())
    q = DIV_TABLE.get(symbol, 0.0)
    q_input = input(f"  Dividend yield (default {q*100:.1f}%): ").strip()
    if q_input:
        q = float(q_input.rstrip("%")) / 100

    r_input = input(f"  Risk-free rate (default {DEFAULT_RATE*100:.1f}%): ").strip()
    r = float(r_input.rstrip("%")) / 100 if r_input else DEFAULT_RATE

    has_quote = input("\n  Have a market quote? (y/n): ").strip().lower() == "y"
    bid = ask = iv = None
    volume = oi = None

    if has_quote:
        bid    = float(input("  Bid: $").strip())
        ask    = float(input("  Ask: $").strip())
        v_str  = input("  Volume (optional): ").strip()
        oi_str = input("  Open interest (optional): ").strip()
        volume = int(v_str) if v_str else None
        oi     = int(oi_str) if oi_str else None
        iv_str = input("  IV from broker (optional, e.g. 0.26 or 26%): ").strip()
        if iv_str:
            iv = float(iv_str.rstrip("%"))
            if iv > 1.0:
                iv /= 100
    else:
        iv_str = input("  Implied vol (e.g. 0.26 or 26%): ").strip()
        iv = float(iv_str.rstrip("%"))
        if iv > 1.0:
            iv /= 100

    # Build minimal args namespace and dispatch
    class _A:
        pass

    a = _A()
    a.ticker, a.stock, a.rate, a.div = ticker_raw if ticker_raw.lower() != "manual" else \
        f"{symbol}{expiry.strftime('%y%m%d')}{opt_type}{int(strike*1000):08d}", S, r, q

    if has_quote:
        a.bid, a.ask, a.volume, a.oi, a.iv = bid, ask, volume, oi, iv
        # Manually reconstruct ticker if manual entry
        if ticker_raw.lower() == "manual":
            a.ticker = f"{symbol}{expiry.strftime('%y%m%d')}{opt_type}{int(strike*1000):08d}"
        cmd_analyze(a)
    else:
        a.iv = iv
        if ticker_raw.lower() == "manual":
            a.ticker = f"{symbol}{expiry.strftime('%y%m%d')}{opt_type}{int(strike*1000):08d}"
        cmd_price(a)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Options pricer — fair value and bid/ask spread analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    def add_common(p):
        p.add_argument("--rate", type=float, default=DEFAULT_RATE,
                       metavar="R", help=f"Risk-free rate (default {DEFAULT_RATE})")
        p.add_argument("--div",  type=float, default=None,
                       metavar="Q", help="Dividend yield (default: per-symbol table)")

    # scout
    p_sc = sub.add_parser("scout", help="Fetch live chain from Yahoo, rank and select contracts")
    p_sc.add_argument("symbol",      help="Ticker symbol e.g. WRB")
    p_sc.add_argument("--call",      action="store_true", default=True)
    p_sc.add_argument("--put",       action="store_true")
    p_sc.add_argument("--min-dte",   type=int,   default=21,   metavar="N",
                      help="Min days to expiry (default 21)")
    p_sc.add_argument("--max-dte",   type=int,   default=90,   metavar="N",
                      help="Max days to expiry (default 90)")
    p_sc.add_argument("--min-delta", type=float, default=0.20, metavar="D",
                      help="Min absolute delta (default 0.20)")
    p_sc.add_argument("--max-delta", type=float, default=0.75, metavar="D",
                      help="Max absolute delta (default 0.75)")
    p_sc.add_argument("--top",       type=int,   default=20,   metavar="N",
                      help="Rows to display (default 20)")
    add_common(p_sc)

    # analyze
    p_an = sub.add_parser("analyze", help="Full analysis given bid/ask")
    p_an.add_argument("--ticker", required=True, metavar="OCC")
    p_an.add_argument("--stock",  required=True, type=float, metavar="S")
    p_an.add_argument("--bid",    required=True, type=float)
    p_an.add_argument("--ask",    required=True, type=float)
    p_an.add_argument("--iv",     type=float, default=None,
                      help="Override IV instead of solving from mid")
    p_an.add_argument("--volume", type=int, default=None)
    p_an.add_argument("--oi",     type=int, default=None)
    add_common(p_an)

    # price
    p_pr = sub.add_parser("price", help="Price from IV")
    p_pr.add_argument("--ticker", required=True, metavar="OCC")
    p_pr.add_argument("--stock",  required=True, type=float, metavar="S")
    p_pr.add_argument("--iv",     required=True, type=float)
    add_common(p_pr)

    # solve
    p_sv = sub.add_parser("solve", help="Back-solve IV from a market price")
    p_sv.add_argument("--ticker",       required=True, metavar="OCC")
    p_sv.add_argument("--stock",        required=True, type=float, metavar="S")
    p_sv.add_argument("--market-price", required=True, type=float, dest="market_price")
    add_common(p_sv)

    # squeeze
    p_sq = sub.add_parser("squeeze",
                          help="Scenario grid: BS P&L at target prices × time horizons")
    p_sq.add_argument("ticker", metavar="OCC",
                      help="OCC ticker e.g. FDS260718C00235000")
    p_sq.add_argument("--stock",    type=float, default=None, metavar="S",
                      help="Current stock price (default: fetch from Yahoo)")
    p_sq.add_argument("--entry",    type=float, default=None, metavar="P",
                      help="Option entry price — used to solve IV and as P&L reference")
    p_sq.add_argument("--iv",       type=float, default=None,
                      help="Override IV (e.g. 0.52 for 52%%)")
    p_sq.add_argument("--iv-bump",  type=float, default=0.0, dest="iv_bump", metavar="B",
                      help="IV shift to apply in all scenarios (e.g. 0.10 for squeeze expansion)")
    p_sq.add_argument("--targets",  metavar="P1,P2,...", default=None,
                      help="Comma-separated target stock prices (default: auto ±8%%..+50%%)")
    p_sq.add_argument("--horizons", metavar="D1,D2,...", default=None,
                      help="Days-forward horizons (default: 10,20,half-DTE)")
    add_common(p_sq)

    # table
    p_tb = sub.add_parser("table", help="Strike comparison table")
    p_tb.add_argument("--symbol",  required=True)
    p_tb.add_argument("--stock",   required=True, type=float, metavar="S")
    p_tb.add_argument("--expiry",  required=True, metavar="YYYY-MM-DD")
    p_tb.add_argument("--iv",      required=True, type=float)
    p_tb.add_argument("--call",    action="store_true", default=True)
    p_tb.add_argument("--put",     action="store_true")
    p_tb.add_argument("--strikes", metavar="K1,K2,...",
                      help="Comma-separated strikes (default: auto-range)")
    add_common(p_tb)

    args = parser.parse_args()

    if args.command is None:
        interactive()
    elif args.command == "scout":
        if getattr(args, "put", False):
            args.call = False
        cmd_scout(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "price":
        cmd_price(args)
    elif args.command == "solve":
        cmd_solve(args)
    elif args.command == "squeeze":
        cmd_squeeze(args)
    elif args.command == "table":
        if args.put:
            args.call = False
        cmd_table(args)


if __name__ == "__main__":
    main()
