#!/usr/bin/env python3
"""Backfill missed nightly IEX ingests over a date range — sequentially.

Runs `nightly_ingest.py --date YYYYMMDD` for each weekday in [start, end], ONE AT
A TIME. DuckDB is single-writer, so the ingests cannot overlap. Weekends are
skipped automatically; market holidays (no data on the server, e.g. Juneteenth)
fail with a 404 inside nightly_ingest and are reported + skipped here. Days
already in the DB are skipped by nightly_ingest itself, so this script is safe to
re-run if it's interrupted.

Each day downloads ~14 GB (TOPS) + ~70 GB (DPLS), extracts, ingests, then deletes
the PCAPs before moving on — so disk use stays bounded to one day at a time, but
expect this to run for hours across the full range.

Usage:
    python backfill_ingest.py --start 2026-06-19 --end 2026-06-26
    python backfill_ingest.py --dates 20260622,20260623,20260624
    python backfill_ingest.py --start 2026-06-19 --end 2026-06-26 --overwrite
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable  # the venv python running this script


def weekdays(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:  # 0-4 = Mon-Fri
            yield d
        d += timedelta(days=1)


def main() -> None:
    p = argparse.ArgumentParser(description="Sequentially backfill missed nightly ingests")
    p.add_argument("--start", metavar="YYYY-MM-DD", help="Range start (inclusive)")
    p.add_argument("--end", metavar="YYYY-MM-DD", help="Range end (inclusive)")
    p.add_argument("--dates", help="Explicit comma-separated YYYYMMDD list (overrides start/end)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-ingest even if the day is already in the DB")
    args = p.parse_args()

    if args.dates:
        targets = [datetime.strptime(d.strip(), "%Y%m%d").date() for d in args.dates.split(",")]
    elif args.start and args.end:
        s = datetime.strptime(args.start, "%Y-%m-%d").date()
        e = datetime.strptime(args.end, "%Y-%m-%d").date()
        targets = list(weekdays(s, e))
    else:
        p.error("provide --start and --end, or --dates")

    print(f"Backfill plan — {len(targets)} weekday(s), run sequentially:")
    for d in targets:
        print(f"  {d:%Y-%m-%d} ({d:%a})")
    print("\n(weekends skipped; holidays will 404 and be reported as FAILED — that's expected)\n")

    results = []
    t0 = time.time()
    for d in targets:
        ds = d.strftime("%Y%m%d")
        print(f"\n{'='*64}\n  Ingesting {d:%Y-%m-%d} ({d:%a})\n{'='*64}")
        cmd = [PY, "nightly_ingest.py", "--date", ds]
        if args.overwrite:
            cmd.append("--overwrite")
        rc = subprocess.call(cmd, cwd=BASE_DIR)
        status = "OK" if rc == 0 else f"FAILED (rc={rc} — holiday / no data / error)"
        results.append((d, status))
        print(f"  -> {d:%Y-%m-%d}: {status}")

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*64}\n  Backfill summary  ({elapsed:.0f} min total)\n{'='*64}")
    for d, status in results:
        print(f"  {d:%Y-%m-%d} ({d:%a})  {status}")


if __name__ == "__main__":
    main()
