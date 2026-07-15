#!/usr/bin/env python3
"""Morning review bundle: cumulative delta + absorption summaries for the full
watchlist, plus a freshly regenerated levels chart.

Reads whatever is currently in the DuckDB (populated by nightly_ingest.py) and
writes a dated markdown report to logs/morning_report_YYYY-MM-DD.md, then
refreshes watchlist_levels.png.

Designed to be run unattended (launchd) after the nightly ingest, but safe to
run by hand any time:

    python morning_report.py
    python morning_report.py --end 2026-06-10
    python morning_report.py --start 2026-04-01 --end 2026-06-10
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import date

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\b")

# Headless matplotlib for the chart regen (no display under launchd).
os.environ.setdefault("MPLBACKEND", "Agg")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR  = os.path.join(BASE_DIR, "logs")
PY       = sys.executable  # the venv python running this script

# Keep in sync with levels_chart.WATCHLIST / squeeze_screener.DEFAULT_WATCHLIST
WATCHLIST = ["VIRT", "WRB", "RPM", "FDS", "GDDY", "IBM", "OSUR", "INTU", "NVO", "CME", "MBC", "AI"]

# Same tuning we use for the manual absorption runs.
ABSORB_FLAGS = ["--imbalance", "0.15", "--min-hours", "2",
                "--max-price-chg", "0.5", "--min-depth-growth", "5"]


def run(cmd) -> str:
    """Run a command in BASE_DIR, return stdout+stderr. Never raises."""
    try:
        res = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True,
                             text=True, timeout=1800)
        return (res.stdout or "") + (res.stderr or "")
    except Exception as e:  # noqa: BLE001 - report job must not crash
        return f"[error running {' '.join(cmd)}]: {e}\n"


def fenced(title: str, body: str) -> str:
    return f"### {title}\n\n```\n{body.rstrip()}\n```\n\n"


def last_row(delta_output: str):
    """Parse the final data row of a cumulative_delta table.

    Row layout: date open close buy_vol sell_vol net_delta delta% cum_delta
    Returns (date, close, net_delta, cum_delta) or None if unparseable.
    """
    last = None
    for line in delta_output.splitlines():
        if _DATE_RE.match(line.strip()):
            last = line.strip()
    if not last:
        return None
    t = last.split()
    try:
        return t[0], float(t[2]), int(t[5]), int(t[7])
    except (IndexError, ValueError):
        return None


def append_daily_log(end: str, latest: dict) -> None:
    """Append one factual line to the project DAILY_LOG.md.

    `latest` maps symbol -> (date, close, net_delta, cum_delta).
    The standouts are the biggest net buyer / seller by share flow.
    """
    log_path = os.path.join(BASE_DIR, "DAILY_LOG.md")
    report_link = f"logs/morning_report_{end}.md"

    if latest:
        sess_date = max(v[0] for v in latest.values())
        b_sym, (_, _, b_net, _) = max(latest.items(), key=lambda kv: kv[1][2])
        s_sym, (_, _, s_net, _) = min(latest.items(), key=lambda kv: kv[1][2])
        line = (f"- **{sess_date}** — top net buy {b_sym} {b_net/1e3:+.0f}K · "
                f"top net sell {s_sym} {s_net/1e3:+.0f}K · "
                f"[report]({report_link})\n")
    else:
        line = f"- **{end}** — no data parsed · [report]({report_link})\n"

    header = (
        "# IEX Tool — Daily Log\n\n"
        "One auto-generated line per session: standout net-delta names (by share "
        "flow) + link to that day's full morning report. Interpretation happens in "
        "review, not here.\n\n"
    )
    exists = os.path.exists(log_path)
    with open(log_path, "a") as f:
        if not exists:
            f.write(header)
        f.write(line)
    print(f"Appended daily log -> {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Morning watchlist report + chart")
    parser.add_argument("--start", default="2026-04-01", metavar="YYYY-MM-DD")
    parser.add_argument("--end", default=date.today().isoformat(), metavar="YYYY-MM-DD",
                        help="End date (default: today; resolves to through prior trading day)")
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    report_path = os.path.join(LOG_DIR, f"morning_report_{args.end}.md")

    parts = [
        f"# Morning Watchlist Report — {args.end}\n",
        f"_Window: {args.start} → {args.end}  ·  chart: watchlist_levels.png_\n\n",
        "## Cumulative Delta\n\n",
    ]
    latest = {}  # sym -> (date, close, net_delta, cum_delta)
    for sym in WATCHLIST:
        out = run([PY, "cumulative_delta.py", sym, "--start", args.start, "--end", args.end])
        parts.append(fenced(sym, out))
        row = last_row(out)
        if row:
            latest[sym] = row

    parts.append("## Absorption Scores\n\n")
    for sym in WATCHLIST:
        out = run([PY, "absorption_score.py", sym,
                   "--start", args.start, "--end", args.end] + ABSORB_FLAGS)
        parts.append(fenced(sym, out))

    with open(report_path, "w") as f:
        f.write("".join(parts))
    print(f"Wrote report -> {report_path}")

    append_daily_log(args.end, latest)

    # Regenerate the chart with the matching end date.
    chart_out = run([PY, "levels_chart.py", "--save", "--delta-end", args.end])
    print(chart_out.strip())
    print("Done.")


if __name__ == "__main__":
    main()
