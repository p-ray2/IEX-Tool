#!/usr/bin/env python3
"""Download, extract, and ingest IEX PCAP data for the previous trading day.

Usage:
    python nightly_ingest.py [--date YYYYMMDD] [--overwrite] [--keep]
"""

import argparse
import gzip
import os
import shutil
import sys
import time
from datetime import date, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH  = os.path.join(BASE_DIR, "cache", "market_data.duckdb")

GCS_BASE = "https://storage.googleapis.com/iex/data/feeds"

FEEDS = {
    "tops": "IEXTP1_TOPS1.6",
    "dpls": "IEXTP1_DPLS1.0",
}


def prev_trading_day(today: date) -> date:
    """Return the most recent weekday before today (does not account for US holidays)."""
    d = today - timedelta(days=1)
    while d.weekday() >= 5:  # 5=Sat, 6=Sun
        d -= timedelta(days=1)
    return d


def gcs_url(trade_date: date, feed_key: str) -> str:
    ds = trade_date.strftime("%Y%m%d")
    return f"{GCS_BASE}/{ds}/{ds}_{FEEDS[feed_key]}.pcap.gz"


def download_file(url: str, dest: str) -> None:
    import urllib.request
    print(f"Downloading {os.path.basename(dest)} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "iex-tool/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            if resp.status == 404:
                raise FileNotFoundError(f"File not found: {url}")
            total = int(resp.headers.get("Content-Length") or 0)
            downloaded = 0
            chunk_size = 1 << 20  # 1 MB
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r  {downloaded/1e9:.2f} / {total/1e9:.2f} GB ({pct:.1f}%)", end="", flush=True)
        print()
    except Exception as e:
        if os.path.exists(dest):
            os.remove(dest)
        raise


def extract_gz(gz_path: str, out_path: str) -> None:
    print(f"Extracting {os.path.basename(gz_path)} ...")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out, length=1 << 20)


def check_already_ingested(trade_date: date) -> bool:
    """Return True if both tops and dpls are already in the ingest log for this date."""
    try:
        import duckdb
        if not os.path.exists(DB_PATH):
            return False
        conn = duckdb.connect(DB_PATH, read_only=True)
        result = conn.execute(
            "SELECT COUNT(DISTINCT feed) FROM ingest_log WHERE trade_date = ?",
            [trade_date],
        ).fetchone()
        conn.close()
        return result[0] >= 2
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Nightly IEX PCAP download and ingest")
    parser.add_argument("--date", metavar="YYYYMMDD",
                        help="Target date (default: previous trading day)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-ingest even if this date is already in the database")
    parser.add_argument("--keep", action="store_true",
                        help="Keep PCAP files on disk after ingestion")
    args = parser.parse_args()

    if args.date:
        ds = args.date
        trade_date = date(int(ds[:4]), int(ds[4:6]), int(ds[6:8]))
    else:
        trade_date = prev_trading_day(date.today())

    print(f"=== IEX nightly ingest: {trade_date} ===")

    if not args.overwrite and check_already_ingested(trade_date):
        print(f"Already ingested {trade_date} — use --overwrite to re-run.")
        sys.exit(0)

    os.makedirs(DATA_DIR, exist_ok=True)

    pcap_paths = {}

    # Download + extract each feed
    for feed_key in ("tops", "dpls"):
        ds = trade_date.strftime("%Y%m%d")
        gz_name   = f"{ds}_{FEEDS[feed_key]}.pcap.gz"
        pcap_name = f"{ds}_{FEEDS[feed_key]}.pcap"
        gz_path   = os.path.join(DATA_DIR, gz_name)
        pcap_path = os.path.join(DATA_DIR, pcap_name)
        pcap_paths[feed_key] = pcap_path

        if os.path.exists(pcap_path):
            print(f"PCAP already on disk, skipping download: {pcap_name}")
            continue

        url = gcs_url(trade_date, feed_key)
        try:
            download_file(url, gz_path)
        except FileNotFoundError:
            print(f"ERROR: {feed_key.upper()} data not available for {trade_date}.")
            print("This may be a US market holiday or the data hasn't posted yet (T+1).")
            sys.exit(1)

        extract_gz(gz_path, pcap_path)
        os.remove(gz_path)
        size_gb = os.path.getsize(pcap_path) / 1e9
        print(f"  -> {pcap_name}  ({size_gb:.2f} GB)")

    # Ingest
    sys.path.insert(0, BASE_DIR)
    from analysis.ingest import ingest_tops, ingest_dpls

    ingest_tops(pcap_paths["tops"], DB_PATH, overwrite=args.overwrite)
    ingest_dpls(pcap_paths["dpls"], DB_PATH, overwrite=args.overwrite)

    # Cleanup
    if not args.keep:
        for feed_key, pcap_path in pcap_paths.items():
            if os.path.exists(pcap_path):
                os.remove(pcap_path)
                print(f"Deleted {os.path.basename(pcap_path)}")

    print(f"\nDone: {trade_date}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Total elapsed: {time.time() - t0:.0f}s")
