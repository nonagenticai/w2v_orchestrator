"""A minimal, VALID in-memory PNG for tests that need real raster bytes.

Built byte-by-byte rather than shipped as a binary fixture so the tests stay
self-contained (and so ``detect-secrets``/diff review never has to reason about
an opaque blob). Used by the storyboard PDF-board export tests to prove that a
frame is genuinely embedded, as opposed to always falling through to the
placeholder.
"""

from __future__ import annotations

import struct
import zlib


def tiny_png(width: int = 8, height: int = 6) -> bytes:
    """A ``width`` by ``height`` mid-grey truecolour PNG."""

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x7f\x7f\x7f" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
