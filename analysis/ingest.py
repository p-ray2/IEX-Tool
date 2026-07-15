"""Parse IEX PCAP files and write market data into a DuckDB database.

Usage:
    from analysis.ingest import ingest_tops, ingest_dpls
    ingest_tops("path/to/TOPS.pcap", "cache/market_data.duckdb")
    ingest_dpls("path/to/DPLS.pcap", "cache/market_data.duckdb")
"""

import os
import re
import struct
import time
from datetime import date
from typing import Optional

import duckdb
from tqdm import tqdm

from iex_parser.pcapng import extract_udp_payload, iter_packets
from iex_parser.iextp import PROTO_TOPS16, PROTO_DPLS10, iter_messages
from iex_parser.messages import (
    MSG_QUOTE_UPDATE, MSG_TRADE_REPORT,
    MSG_ADD_ORDER, MSG_DELETE_ORDER,
    decode_quote, decode_trade,
    _ADD_ORDER_FMT, _ADD_ORDER_SIZE,
    _DEL_ORDER_FMT, _DEL_ORDER_SIZE,
    ORDER_SIDE_BUY, _sym,
)

BATCH_SIZE = 500_000
DEPTH_SNAPSHOT_INTERVAL_NS = 60_000_000_000  # 60 seconds per symbol

EDT_OFFSET_NS = 4 * 3_600 * 1_000_000_000  # Eastern Daylight Time = UTC-4


# ── Schema ────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS trades (
    trade_date    DATE    NOT NULL,
    timestamp_ns  BIGINT  NOT NULL,
    symbol        VARCHAR NOT NULL,
    size          INTEGER NOT NULL,
    price         DOUBLE  NOT NULL,
    trade_id      BIGINT,
    flags         SMALLINT,
    is_extended   BOOLEAN,
    is_odd_lot    BOOLEAN
);

CREATE TABLE IF NOT EXISTS quotes (
    trade_date    DATE    NOT NULL,
    timestamp_ns  BIGINT  NOT NULL,
    symbol        VARCHAR NOT NULL,
    bid_price     DOUBLE  NOT NULL,
    ask_price     DOUBLE  NOT NULL,
    bid_size      INTEGER NOT NULL,
    ask_size      INTEGER NOT NULL,
    spread        DOUBLE  NOT NULL,
    mid_price     DOUBLE  NOT NULL
);

CREATE TABLE IF NOT EXISTS depth_snapshots (
    trade_date    DATE    NOT NULL,
    timestamp_ns  BIGINT  NOT NULL,
    symbol        VARCHAR NOT NULL,
    bid_depth     BIGINT  NOT NULL,
    ask_depth     BIGINT  NOT NULL,
    imbalance     DOUBLE  NOT NULL
);

CREATE TABLE IF NOT EXISTS ingest_log (
    feed          VARCHAR NOT NULL,
    trade_date    DATE,
    completed_at  DOUBLE  NOT NULL,
    trade_rows    BIGINT,
    quote_rows    BIGINT,
    depth_rows    BIGINT
);
"""


def _date_from_path(pcap_path: str) -> date:
    """Extract trade date from the PCAP filename (first 8-digit run = YYYYMMDD)."""
    m = re.search(r"(\d{8})", os.path.basename(pcap_path))
    if m:
        d = m.group(1)
        return date(int(d[:4]), int(d[4:6]), int(d[6:8]))
    raise ValueError(f"Cannot extract date from filename: {pcap_path}")


def _open_db(db_path: str) -> duckdb.DuckDBPyConnection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = duckdb.connect(db_path)
    conn.execute(_DDL)
    return conn


def _flush(conn: duckdb.DuckDBPyConnection, table: str, rows: list, columns: list) -> None:
    if not rows:
        return
    import pandas as pd
    df = pd.DataFrame(rows, columns=columns)
    conn.execute(f"INSERT INTO {table} SELECT * FROM df")


# ── TOPS ingestion ─────────────────────────────────────────────────────────────

_TRADE_COLS = ["trade_date", "timestamp_ns", "symbol", "size", "price", "trade_id", "flags", "is_extended", "is_odd_lot"]
_QUOTE_COLS = ["trade_date", "timestamp_ns", "symbol", "bid_price", "ask_price", "bid_size", "ask_size", "spread", "mid_price"]


def ingest_tops(pcap_path: str, db_path: str, limit: Optional[int] = None, overwrite: bool = False) -> None:
    """Parse TOPS 1.6 pcap → trades + quotes tables."""
    conn = _open_db(db_path)
    trade_date = _date_from_path(pcap_path)

    if overwrite:
        conn.execute("DELETE FROM trades WHERE trade_date = ?", [trade_date])
        conn.execute("DELETE FROM quotes WHERE trade_date = ?", [trade_date])

    trade_rows: list = []
    quote_rows: list = []
    trade_total = 0
    quote_total = 0
    pkt_count = 0

    file_size = os.path.getsize(pcap_path)
    print(f"Ingesting TOPS {trade_date}: {pcap_path}  ({file_size/1e9:.1f} GB)")
    t0 = time.time()

    conn.execute("BEGIN")
    with tqdm(total=None, unit=" pkts", desc="packets") as pbar:
        for _ts_cap, packet in iter_packets(pcap_path):
            udp = extract_udp_payload(packet)
            if udp is None or len(udp) < 40:
                continue

            proto_id = struct.unpack_from("<H", udp, 2)[0]
            if proto_id != PROTO_TOPS16:
                continue

            for _send_time, _proto, msg_type, body in iter_messages(udp):
                if msg_type == MSG_TRADE_REPORT:
                    t = decode_trade(body)
                    if t:
                        trade_rows.append((
                            trade_date,
                            t["timestamp_ns"], t["symbol"], t["size"], t["price"],
                            t["trade_id"], t["flags"], t["is_extended_hours"], t["is_odd_lot"],
                        ))

                elif msg_type == MSG_QUOTE_UPDATE:
                    q = decode_quote(body)
                    if q:
                        quote_rows.append((
                            trade_date,
                            q["timestamp_ns"], q["symbol"], q["bid_price"], q["ask_price"],
                            q["bid_size"], q["ask_size"], q["spread"], q["mid_price"],
                        ))

            if len(trade_rows) >= BATCH_SIZE:
                _flush(conn, "trades", trade_rows, _TRADE_COLS)
                trade_total += len(trade_rows)
                trade_rows = []

            if len(quote_rows) >= BATCH_SIZE:
                _flush(conn, "quotes", quote_rows, _QUOTE_COLS)
                quote_total += len(quote_rows)
                quote_rows = []

            pkt_count += 1
            pbar.update(1)
            if limit and pkt_count >= limit:
                break

    _flush(conn, "trades", trade_rows, _TRADE_COLS)
    _flush(conn, "quotes", quote_rows, _QUOTE_COLS)
    trade_total += len(trade_rows)
    quote_total += len(quote_rows)
    conn.execute("COMMIT")

    conn.execute(
        "INSERT INTO ingest_log VALUES (?, ?, ?, ?, ?, ?)",
        ["tops", trade_date, time.time(), trade_total, quote_total, 0],
    )
    conn.close()
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s — {trade_total:,} trades, {quote_total:,} quotes")


# ── DPLS ingestion ─────────────────────────────────────────────────────────────

_DEPTH_COLS = ["trade_date", "timestamp_ns", "symbol", "bid_depth", "ask_depth", "imbalance"]


class _OrderBook:
    """Per-symbol order book tracking bid/ask depth at the individual order level."""

    __slots__ = ("orders", "bid_depth", "ask_depth")

    def __init__(self):
        self.orders: dict = {}   # order_id → (side, shares)  — 'bid' or 'ask'
        self.bid_depth: int = 0
        self.ask_depth: int = 0

    def add_order(self, order_id: int, side: str, shares: int) -> None:
        existing = self.orders.get(order_id)
        if existing is not None:
            ex_side, ex_shares = existing
            if ex_side == "bid":
                self.bid_depth -= ex_shares
            else:
                self.ask_depth -= ex_shares
        self.orders[order_id] = (side, shares)
        if side == "bid":
            self.bid_depth += shares
        else:
            self.ask_depth += shares

    def delete_order(self, order_id: int) -> None:
        entry = self.orders.pop(order_id, None)
        if entry is None:
            return
        side, shares = entry
        if side == "bid":
            self.bid_depth -= shares
        else:
            self.ask_depth -= shares

    @property
    def imbalance(self) -> float:
        total = self.bid_depth + self.ask_depth
        return (self.bid_depth - self.ask_depth) / total if total > 0 else 0.0


def ingest_dpls(pcap_path: str, db_path: str, limit: Optional[int] = None, overwrite: bool = False) -> None:
    """Parse DPLS 1.0 pcap → depth_snapshots table."""
    conn = _open_db(db_path)
    trade_date = _date_from_path(pcap_path)

    if overwrite:
        conn.execute("DELETE FROM depth_snapshots WHERE trade_date = ?", [trade_date])

    order_books: dict = {}           # symbol → _OrderBook
    last_snapshot: dict = {}         # symbol → last snapshot send_time_ns
    depth_rows: list = []
    depth_total = 0
    pkt_count = 0

    file_size = os.path.getsize(pcap_path)
    print(f"Ingesting DPLS {trade_date}: {pcap_path}  ({file_size/1e9:.1f} GB)")
    t0 = time.time()

    conn.execute("BEGIN")
    with tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024, desc="DPLS") as pbar:
        for _ts_cap, packet in iter_packets(pcap_path):
            udp = extract_udp_payload(packet)
            if udp is None or len(udp) < 40:
                pbar.update(len(packet))
                continue

            proto_id = struct.unpack_from("<H", udp, 2)[0]
            if proto_id != PROTO_DPLS10:
                pbar.update(len(packet))
                continue

            for send_time, _proto, msg_type, body in iter_messages(udp):
                # Inline ADD_ORDER decode — avoids creating a dict per event
                if msg_type == MSG_ADD_ORDER:
                    if len(body) < _ADD_ORDER_SIZE:
                        continue
                    side_byte, _ts, sym_raw, order_id, size, _price = struct.unpack_from(_ADD_ORDER_FMT, body)
                    if size == 0:
                        continue
                    sym = _sym(sym_raw)
                    book = order_books.get(sym)
                    if book is None:
                        book = _OrderBook()
                        order_books[sym] = book
                    side = "bid" if side_byte[0] == ORDER_SIDE_BUY else "ask"
                    book.add_order(order_id, side, size)

                # Inline DELETE_ORDER decode
                elif msg_type == MSG_DELETE_ORDER:
                    if len(body) < _DEL_ORDER_SIZE:
                        continue
                    _, _ts, sym_raw, order_id_ref = struct.unpack_from(_DEL_ORDER_FMT, body)
                    sym = _sym(sym_raw)
                    book = order_books.get(sym)
                    if book is None:
                        continue
                    book.delete_order(order_id_ref)

                else:
                    continue

                last = last_snapshot.get(sym, 0)
                if send_time - last >= DEPTH_SNAPSHOT_INTERVAL_NS:
                    last_snapshot[sym] = send_time
                    depth_rows.append((
                        trade_date, send_time, sym,
                        book.bid_depth, book.ask_depth, book.imbalance,
                    ))

            if len(depth_rows) >= BATCH_SIZE:
                _flush(conn, "depth_snapshots", depth_rows, _DEPTH_COLS)
                depth_total += len(depth_rows)
                depth_rows = []

            pkt_count += 1
            pbar.update(len(packet))
            if limit and pkt_count >= limit:
                break

    _flush(conn, "depth_snapshots", depth_rows, _DEPTH_COLS)
    depth_total += len(depth_rows)
    conn.execute("COMMIT")

    conn.execute(
        "INSERT INTO ingest_log VALUES (?, ?, ?, ?, ?, ?)",
        ["dpls", trade_date, time.time(), 0, 0, depth_total],
    )
    conn.close()
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s — {depth_total:,} depth snapshots")
