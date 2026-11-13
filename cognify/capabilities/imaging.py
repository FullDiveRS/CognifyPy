"""Pure-stdlib PNG/APNG encode & decode (8-bit, non-interlaced)."""
from __future__ import annotations

import struct
import zlib
from typing import List, Sequence, Tuple

PNG_SIG = b"\x89PNG\r\n\x1a\n"
Pixel = Tuple[int, int, int]
Rows = List[List[Pixel]]


def _chunk(ctype: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + ctype + data +
            struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF))


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


# --------------------------------------------------------------------- encode
def encode_png(width: int, height: int, rows: Rows) -> bytes:
    """Encode RGB rows (each pixel a 3-tuple 0..255) into a PNG byte string."""
    if len(rows) != height or any(len(r) != width for r in rows):
        raise ValueError("rows must match width x height")
    raw = bytearray()
    for row in rows:
        raw.append(0)                       # filter type 0 (None)
        for px in row:
            raw += bytes((px[0], px[1], px[2]))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (PNG_SIG + _chunk(b"IHDR", ihdr) +
            _chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + _chunk(b"IEND", b""))


def encode_apng(frames: Sequence[Tuple[int, int, Rows]], delay_ms: int = 50) -> bytes:
    """Encode multiple same-size RGB frames into an animated PNG (APNG).

    ``frames`` is a sequence of (width, height, rows). Plays in a loop in
    every modern browser. delay_ms is the per-frame delay.
    """
    if not frames:
        raise ValueError("need at least one frame")
    width, height = frames[0][0], frames[0][1]
    if any(f[0] != width or f[1] != height for f in frames):
        raise ValueError("all frames must share the same size")
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    out = bytearray(PNG_SIG + _chunk(b"IHDR", ihdr))
    out += _chunk(b"acTL", struct.pack(">II", len(frames), 0))  # loop forever
    seq = 0
    delay = (max(1, delay_ms), 1000)            # numerator, denominator
    for i, (_, _, rows) in enumerate(frames):
        fctl = struct.pack(">IIIIIIIII", seq, width, height, 0, 0,
                           delay[0], delay[1], 0, 0)
        seq += 1
        raw = bytearray()
        for row in rows:
            raw.append(0)
            for px in row:
                raw += bytes((px[0], px[1], px[2]))
        comp = zlib.compress(bytes(raw), 6)
        if i == 0:
            out += _chunk(b"fcTL", fctl) + _chunk(b"IDAT", comp)
        else:
            out += _chunk(b"fcTL", fctl) + _chunk(b"fdAT", struct.pack(">I", seq) + comp)
            seq += 1
    out += _chunk(b"IEND", b"")
    return bytes(out)


# --------------------------------------------------------------------- decode
def decode_png(data: bytes) -> Tuple[int, int, Rows]:
    """Decode a PNG (or the first frame of an APNG) into RGB rows."""
    if data[:8] != PNG_SIG:
        raise ValueError("not a PNG stream")
    pos, idat, ihdr, plte = 8, bytearray(), None, None
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        ctype = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        if ctype == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", body)
        elif ctype == b"IDAT":
            idat += body
        elif ctype == b"PLTE":
            plte = [tuple(body[i:i + 3]) for i in range(0, len(body), 3)]
        elif ctype == b"IEND":
            break
        pos += 12 + length
    if ihdr is None:
        raise ValueError("missing IHDR chunk")
    width, height, depth, color, _, _, interlace = ihdr
    if depth != 8 or interlace != 0:
        raise ValueError("only 8-bit non-interlaced PNG is supported")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color)
    if channels is None:
        raise ValueError(f"unsupported color type {color}")
    raw = zlib.decompress(bytes(idat))
    stride = width * channels
    if len(raw) < height * (stride + 1):
        raise ValueError("truncated PNG data")
    decoded: List[bytes] = []
    prev = bytearray(stride)
    i = 0
    for _ in range(height):
        ftype = raw[i]
        i += 1
        line = bytearray(raw[i:i + stride])
        i += stride
        if ftype == 1:                                  # Sub
            for x in range(channels, stride):
                line[x] = (line[x] + line[x - channels]) & 0xFF
        elif ftype == 2:                                # Up
            for x in range(stride):
                line[x] = (line[x] + prev[x]) & 0xFF
        elif ftype == 3:                                # Average
            for x in range(stride):
                left = line[x - channels] if x >= channels else 0
                line[x] = (line[x] + ((left + prev[x]) >> 1)) & 0xFF
        elif ftype == 4:                                # Paeth
            for x in range(stride):
                left = line[x - channels] if x >= channels else 0
                upleft = prev[x - channels] if x >= channels else 0
                line[x] = (line[x] + _paeth(left, prev[x], upleft)) & 0xFF
        elif ftype != 0:
            raise ValueError(f"unknown PNG filter {ftype}")
        prev = line
        decoded.append(bytes(line))
    rows: Rows = []
    for line in decoded:
        px_row: List[Pixel] = []
        for x in range(width):
            if color == 2:
                px_row.append((line[3 * x], line[3 * x + 1], line[3 * x + 2]))
            elif color == 6:
                px_row.append((line[4 * x], line[4 * x + 1], line[4 * x + 2]))
            elif color == 0:
                g = line[x]
                px_row.append((g, g, g))
            elif color == 4:
                g = line[2 * x]
                px_row.append((g, g, g))
            else:                                       # palette
                px_row.append(plte[line[x]] if plte else (0, 0, 0))
        rows.append(px_row)
    return width, height, rows


def is_png(data: bytes) -> bool:
    return bool(data) and data[:8] == PNG_SIG


def has_actl(data: bytes) -> bool:
    """True if the PNG carries an acTL chunk (i.e. it is animated)."""
    return b"acTL" in data[:200]
