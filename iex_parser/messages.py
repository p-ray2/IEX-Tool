"""IEX message type decoders for TOPS 1.6 and DPLS 1.0.

Prices are stored as 8-byte integers in units of $0.0001 (divide by 10_000 for USD).
Timestamps are nanoseconds since Unix epoch (UTC).
Symbols are 8 ASCII bytes, space-padded on the right.
"""

import struct
from typing import Optional

# ── TOPS 1.6 message types ────────────────────────────────────────────────────
MSG_SYSTEM_EVENT    = 0x53  # S
MSG_SECURITY_DIR    = 0x44  # D
MSG_TRADING_STATUS  = 0x48  # H
MSG_OP_HALT         = 0x4F  # O
MSG_SSP_STATUS      = 0x50  # P
MSG_SECURITY_EVENT  = 0x45  # E
MSG_QUOTE_UPDATE    = 0x51  # Q
MSG_TRADE_REPORT    = 0x54  # T
MSG_OFFICIAL_PRICE  = 0x58  # X
MSG_TRADE_BREAK     = 0x42  # B
MSG_AUCTION_INFO    = 0x41  # A

# ── DPLS 1.0 message types ────────────────────────────────────────────────────
# 'a' (0x61) = Add Order — displayed order added to IEX Book (buy or sell)
# 'R' (0x52) = Order Delete — displayed order removed from IEX Book
MSG_ADD_ORDER    = 0x61  # a
MSG_DELETE_ORDER = 0x52  # R

# Add Order side indicators
ORDER_SIDE_BUY  = 0x38  # '8'
ORDER_SIDE_SELL = 0x35  # '5'

# Trade sale condition flags (bitmask)
TRADE_FLAG_EXTENDED_HOURS = 0x08
TRADE_FLAG_ODD_LOT        = 0x10

# ── Struct formats ─────────────────────────────────────────────────────────────
# Quote Update:  flags(1)+ts(8)+sym(8)+bid_sz(4)+bid_px(8)+ask_px(8)+ask_sz(4) = 41
_QUOTE_FMT = "<BQ8sIQQI"
_QUOTE_SIZE = 41

# Trade Report:  flags(1)+ts(8)+sym(8)+size(4)+price(8)+trade_id(8) = 37
_TRADE_FMT = "<BQ8sIQQ"
_TRADE_SIZE = 37

# Add Order ('a', 0x61) body = 37 bytes:
#   side(1)+ts(8)+sym(8)+order_id(8)+size(4)+price(8)
# side: 0x38='8'=Buy, 0x35='5'=Sell
# price is 8-byte uint64 in $0.0001 units (same scale as TOPS)
_ADD_ORDER_FMT  = "<cQ8sQIQ"
_ADD_ORDER_SIZE = 37

# Order Delete ('R', 0x52) body = 25 bytes:
#   reserved(1)+ts(8)+sym(8)+order_id_ref(8)
_DEL_ORDER_FMT  = "<BQ8sQ"
_DEL_ORDER_SIZE = 25

_PRICE_SCALE = 10_000  # IEX price units → USD

_SYM_CACHE: dict = {}


def _sym(raw: bytes) -> str:
    s = _SYM_CACHE.get(raw)
    if s is None:
        s = raw.rstrip(b" ").decode("ascii", errors="replace")
        _SYM_CACHE[raw] = s
    return s


def decode_quote(body: bytes) -> Optional[dict]:
    """Decode a Quote Update message body. Returns None if too short."""
    if len(body) < _QUOTE_SIZE:
        return None
    flags, ts, sym_raw, bid_sz, bid_px, ask_px, ask_sz = struct.unpack_from(_QUOTE_FMT, body)
    if bid_px == 0 or ask_px == 0:
        return None  # no active quote
    spread = (ask_px - bid_px) / _PRICE_SCALE
    mid = (bid_px + ask_px) / (2 * _PRICE_SCALE)
    return {
        "timestamp_ns": ts,
        "symbol": _sym(sym_raw),
        "bid_price": bid_px / _PRICE_SCALE,
        "ask_price": ask_px / _PRICE_SCALE,
        "bid_size": bid_sz,
        "ask_size": ask_sz,
        "spread": spread,
        "mid_price": mid,
        "flags": flags,
    }


def decode_trade(body: bytes) -> Optional[dict]:
    """Decode a Trade Report message body. Returns None if too short."""
    if len(body) < _TRADE_SIZE:
        return None
    flags, ts, sym_raw, size, price, trade_id = struct.unpack_from(_TRADE_FMT, body)
    if size == 0 or price == 0:
        return None
    return {
        "timestamp_ns": ts,
        "symbol": _sym(sym_raw),
        "size": size,
        "price": price / _PRICE_SCALE,
        "trade_id": trade_id,
        "flags": flags,
        "is_extended_hours": bool(flags & TRADE_FLAG_EXTENDED_HOURS),
        "is_odd_lot": bool(flags & TRADE_FLAG_ODD_LOT),
    }


def decode_add_order(body: bytes) -> Optional[dict]:
    """Decode a DPLS Add Order message ('a', 0x61). Body must be 37 bytes.

    Side byte: 0x38 ('8') = Buy, 0x35 ('5') = Sell.
    Order ID is the unique key used to track this order in subsequent Delete messages.
    """
    if len(body) < _ADD_ORDER_SIZE:
        return None
    side_byte, ts, sym_raw, order_id, size, price_raw = struct.unpack_from(_ADD_ORDER_FMT, body)
    if size == 0:
        return None
    side_char = side_byte[0] if isinstance(side_byte, (bytes, bytearray)) else side_byte
    return {
        "timestamp_ns": ts,
        "symbol": _sym(sym_raw),
        "order_id": order_id,
        "side": "bid" if side_char == ORDER_SIDE_BUY else "ask",
        "shares": size,
        "price_raw": price_raw,
        "price": price_raw / _PRICE_SCALE,
    }


def decode_delete_order(body: bytes) -> Optional[dict]:
    """Decode a DPLS Order Delete message ('R', 0x52). Body must be 25 bytes.

    order_id_ref references the Order ID from a prior Add Order message.
    The referenced order is removed from the book regardless of side.
    """
    if len(body) < _DEL_ORDER_SIZE:
        return None
    _, ts, sym_raw, order_id_ref = struct.unpack_from(_DEL_ORDER_FMT, body)
    return {
        "timestamp_ns": ts,
        "symbol": _sym(sym_raw),
        "order_id_ref": order_id_ref,
    }
