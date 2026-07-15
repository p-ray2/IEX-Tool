"""Streaming PCAPNG reader — yields (timestamp_ns, raw_packet_bytes) per Enhanced Packet Block."""

import os
import struct
from typing import Iterator, Optional, Tuple

BLOCK_SHB = 0x0A0D0D0A
BLOCK_IDB = 0x00000001
BLOCK_EPB = 0x00000006
BLOCK_OPB = 0x00000002

# Default timestamp resolution: microseconds (10^-6 s) → multiply by 1000 to get ns
_DEFAULT_NS_PER_UNIT = 1000


def _parse_idb_ts_resol(body: bytes) -> int:
    """Extract nanoseconds-per-unit from an Interface Description Block body."""
    off = 8  # skip link_type(2) + reserved(2) + snap_len(4)
    while off + 4 <= len(body):
        opt_code, opt_len = struct.unpack_from("<HH", body, off)
        off += 4
        if opt_code == 0:
            break
        val = body[off : off + opt_len]
        off += (opt_len + 3) & ~3  # pad to 4-byte boundary
        if opt_code == 9 and opt_len == 1:  # if_tsresol
            b = val[0]
            if b & 0x80:
                ns_per_unit = 10**9 // (2 ** (b & 0x7F))
            else:
                ns_per_unit = 10**9 // (10 ** (b & 0x7F))
            return ns_per_unit
    return _DEFAULT_NS_PER_UNIT


def iter_packets(filepath: str) -> Iterator[Tuple[int, bytes]]:
    """Yield (timestamp_ns, raw_ethernet_frame) for every captured packet."""
    ts_resols: dict = {}
    iface_count = 0
    file_size = os.path.getsize(filepath)
    bytes_read = 0

    with open(filepath, "rb") as f:
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            block_type, block_len = struct.unpack_from("<II", hdr)
            body_len = block_len - 12  # total minus 8-byte hdr and 4-byte trailing len
            body = f.read(body_len)
            f.read(4)  # trailing block length (ignored)
            bytes_read += block_len

            if block_type == BLOCK_SHB:
                iface_count = 0
                ts_resols.clear()

            elif block_type == BLOCK_IDB:
                ts_resols[iface_count] = _parse_idb_ts_resol(body)
                iface_count += 1

            elif block_type == BLOCK_EPB:
                if len(body) < 20:
                    continue
                iface_id, ts_high, ts_low, cap_len, _orig = struct.unpack_from("<IIIII", body, 0)
                ts_units = (ts_high << 32) | ts_low
                ns_per_unit = ts_resols.get(iface_id, _DEFAULT_NS_PER_UNIT)
                timestamp_ns = ts_units * ns_per_unit
                packet = body[20 : 20 + cap_len]
                yield timestamp_ns, packet

            elif block_type == BLOCK_OPB:
                # Obsolete Packet Block: iface_id(2)+drops(2)+ts_hi(4)+ts_lo(4)+cap(4)+orig(4)
                if len(body) < 16:
                    continue
                iface_id = struct.unpack_from("<H", body, 0)[0]
                ts_high, ts_low, cap_len = struct.unpack_from("<III", body, 4)
                ts_units = (ts_high << 32) | ts_low
                ns_per_unit = ts_resols.get(iface_id, _DEFAULT_NS_PER_UNIT)
                timestamp_ns = ts_units * ns_per_unit
                packet = body[16 : 16 + cap_len]
                yield timestamp_ns, packet


def extract_udp_payload(packet: bytes) -> Optional[bytes]:
    """Extract UDP payload from a raw Ethernet frame. Returns None if not IPv4/UDP."""
    if len(packet) < 14:
        return None

    ethertype = (packet[12] << 8) | packet[13]
    ip_start = 14

    # Unwrap VLAN tags (802.1Q / 802.1ad)
    while ethertype in (0x8100, 0x88A8, 0x9100):
        if len(packet) < ip_start + 4:
            return None
        ethertype = (packet[ip_start + 2] << 8) | packet[ip_start + 3]
        ip_start += 4

    if ethertype != 0x0800:
        return None

    if len(packet) < ip_start + 20:
        return None

    ihl = (packet[ip_start] & 0x0F) * 4
    protocol = packet[ip_start + 9]

    if protocol != 17:  # UDP
        return None

    udp_start = ip_start + ihl
    if len(packet) < udp_start + 8:
        return None

    udp_len = (packet[udp_start + 4] << 8) | packet[udp_start + 5]
    payload = packet[udp_start + 8 : udp_start + udp_len]
    return payload if payload else None
