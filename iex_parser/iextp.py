"""IEXTP transport protocol parser.

IEXTP header (40 bytes, little-endian):
  version(1) reserved(1) protocol_id(2) channel_id(4) session_id(4)
  payload_len(2) msg_count(2) stream_offset(8) first_msg_seq(8) send_time(8)

Each message:
  msg_len(2) msg_type(1) body(msg_len-1)
"""

import struct
from typing import Iterator, Tuple

PROTO_TOPS16 = 0x8003
PROTO_DPLS10 = 0x8005

_HDR_SIZE = 40
_HDR_FMT = "<BBHIIHHQQQ"  # 1+1+2+4+4+2+2+8+8+8 = 40


def iter_messages(payload: bytes) -> Iterator[Tuple[int, int, int, bytes]]:
    """Yield (send_time_ns, protocol_id, msg_type, msg_body) for every message in an IEXTP packet."""
    if len(payload) < _HDR_SIZE:
        return

    (
        _version, _reserved, proto_id,
        _channel, _session,
        _payload_len, msg_count,
        _stream_offset, _first_seq, send_time,
    ) = struct.unpack_from(_HDR_FMT, payload, 0)

    offset = _HDR_SIZE
    for _ in range(msg_count):
        if offset + 2 > len(payload):
            break
        msg_len = struct.unpack_from("<H", payload, offset)[0]
        offset += 2
        if msg_len == 0 or offset + msg_len > len(payload):
            break
        msg_type = payload[offset]
        msg_body = payload[offset + 1 : offset + msg_len]
        offset += msg_len
        yield send_time, proto_id, msg_type, msg_body
