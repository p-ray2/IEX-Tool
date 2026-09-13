# IEX Market Microstructure Analyzer

Parses IEX exchange PCAP files (TOPS 1.6 and DPLS 1.0) into a local DuckDB database and provides CLI commands for market microstructure analysis. Started this project in early April 2026. Still working on how to build out additional cloud storage solutions and automated pipelining. Volume correlation tool provided to test IEX's volume against data from yahoo finance to test viability of exchange data as a proxy. 

---

## The Two Data Feeds

### TOPS 1.6 — Trade and Quote Data
`data_feeds_YYYYMMDD_YYYYMMDD_IEXTP1_TOPS1.6.pcap`

TOPS is the primary market data feed. It contains:
- **Quote Updates** — best bid/ask prices and sizes for every symbol, updated in real time
- **Trade Reports** — every executed trade with price, size, and trade ID

This is the feed you need for spread analysis, VWAP, volume, and price range data.

### DPLS 1.0 — Depth of Book Data
`data_feeds_YYYYMMDD_YYYYMMDD_IEXTP1_DPLS1.0.pcap`

DPLS is the depth feed. It contains **Price Level Updates** — incremental changes to the full order book on both the bid and ask sides. It does not contain trades or top-of-book quotes.

This feed is used to compute **order book depth imbalance**: the running ratio of total bid shares to total ask shares across all price levels. A positive imbalance means more buy-side interest; negative means more sell-side.

Neither feed contains broker identifiers — all data is anonymized by the exchange.

---

## Setup

### Requirements
Python 3.8+ is required. Install dependencies into a virtual environment:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### PCAP File Placement
Place the PCAP files in the project root directory (same folder as `main.py`). The expected filenames are:

```
data_feeds_20260417_20260417_IEXTP1_TOPS1.6.pcap
data_feeds_20260417_20260417_IEXTP1_DPLS1.0.pcap
```

If your filenames differ, update `TOPS_PCAP` and `DPLS_PCAP` at the top of `main.py`.

---

## Building the Database

The database is stored at `cache/market_data.duckdb` and is created automatically on first ingest.

Parse both feeds (recommended — runs sequentially):
```bash
python main.py ingest --feed both
```

Parse only one feed:
```bash
python main.py ingest --feed tops
python main.py ingest --feed dpls
```

For a quick test run with a subset of packets:
```bash
python main.py ingest --feed tops --limit 200000
```

To clear existing data and re-ingest from scratch:
```bash
python main.py ingest --feed both --overwrite
```

> **Note:** Full ingestion of a ~50 GB file takes roughly 15–60 minutes depending on disk speed. Progress is shown with a packet counter (TOPS) or byte progress bar (DPLS).

---

## Commands

All display commands accept an optional `--date YYYY-MM-DD` flag to filter results to a single trading day. Without it, all ingested dates are combined.

### `summary [--date YYYY-MM-DD]`
High-level counts across the database (or a single day).
```bash
python main.py summary
python main.py summary --date 2026-04-17
```
Shows: total symbols, total trades, total shares, total notional value, quote updates, and depth snapshots.

### `report [--top N] [--date YYYY-MM-DD]`
Microstructure report for the top N symbols by share volume (default: 25).
```bash
python main.py report
python main.py report --top 50 --date 2026-04-17
```
Shows: shares traded, trade count, VWAP, mean spread ($ and bps), and mean depth imbalance per symbol.

### `notional [--top N] [--date YYYY-MM-DD]`
Top N symbols ranked by notional value — total shares traded × VWAP (default: 25).
```bash
python main.py notional
python main.py notional --top 10 --date 2026-04-17
```
Shows: notional value, shares, trade count, VWAP, low, and high price.

### `query <SYMBOL> [--date YYYY-MM-DD]`
Detailed breakdown for a single symbol, optionally filtered to one day.
```bash
python main.py query AAPL
python main.py query SPY --date 2026-04-17
```
Shows: total shares, trades, VWAP, price range, spread statistics, depth imbalance stats, hourly volume, and hourly depth imbalance. When multiple dates are loaded, hourly tables include a date column. All times are displayed in Eastern Time.

---

## Switching to a New Day's Data

Each day of IEX data is a separate pair of PCAP files. To analyze a different day:

1. **Place the new PCAP files** in the project root. The filenames follow the same pattern with a different date, e.g.:
   ```
   data_feeds_20260418_20260418_IEXTP1_TOPS1.6.pcap
   data_feeds_20260418_20260418_IEXTP1_DPLS1.0.pcap
   ```

2. **Update the paths** in `main.py` (lines 16–17):
   ```python
   TOPS_PCAP = os.path.join(BASE_DIR, "data_feeds_20260418_20260418_IEXTP1_TOPS1.6.pcap")
   DPLS_PCAP = os.path.join(BASE_DIR, "data_feeds_20260418_20260418_IEXTP1_DPLS1.0.pcap")
   ```

3. **Delete the existing database** (the schema changed to add a `trade_date` column — required once if you have an existing `cache/market_data.duckdb`):
   ```bash
   rm cache/market_data.duckdb
   ```

4. **Re-ingest** all days you want to analyze:
   ```bash
   python main.py ingest --feed both --overwrite
   ```

   Or delete the database file entirely and start fresh:
   ```bash
   rm cache/market_data.duckdb
   python main.py ingest --feed both
   ```

> If you want to keep multiple days in the same database, remove the `--overwrite` flag. The tables will accumulate rows across all ingested days, and all queries will reflect the combined dataset. The `ingest_log` table records each completed ingest with a timestamp.

---

## Database Schema

| Table | Description |
|---|---|
| `trades` | Every executed trade: symbol, timestamp, price, size, flags |
| `quotes` | Best bid/ask snapshots: symbol, timestamp, prices, sizes, spread, mid |
| `depth_snapshots` | Per-symbol order book state sampled every 60 seconds: bid depth, ask depth, imbalance |
| `ingest_log` | One row per completed ingest run |

You can query the database directly with any DuckDB-compatible tool:
```bash
python -c "import duckdb; print(duckdb.connect('cache/market_data.duckdb').execute('SELECT COUNT(*) FROM trades').df())"
```
