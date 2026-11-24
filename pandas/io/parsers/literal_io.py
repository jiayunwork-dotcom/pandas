"""
Byte source adapter used only by ``read_csv(sep_mode="literal")``.

The C tokenizer already copes with arbitrarily short reads (it re-enters the
tokenizer with the same withheld separator-prefix state).  This adapter adds the
two things the literal path guarantees on top of that:

* a leading UTF-8 BOM is stripped even when the source hands out one byte at a
  time (the tokenizer's own BOM check needs three bytes in one buffer);
* text sources are normalised to UTF-8 bytes with the same codec the C reader
  would use, so a character-level short read behaves like a byte-level one.

Only a fixed amount (a handful of bytes) is buffered up front; the rest is
streamed straight through and never retained.
"""
from __future__ import annotations

from typing import Any

_BOM = b"\xef\xbb\xbf"


class LiteralSource:
    """Minimal ``read(n)`` adapter over a text or binary readable object."""

    def __init__(self, obj: Any, encoding_errors: str = "strict") -> None:
        self._obj = obj
        self._errors = encoding_errors
        self._started = False
        self._prefix = b""

    def _read_bytes(self, nbytes: int) -> bytes:
        data = self._obj.read(nbytes)
        if data is None:
            return b""
        if isinstance(data, str):
            return data.encode("utf-8", self._errors)
        return bytes(data)

    def _strip_bom(self) -> None:
        # Read until we either have three bytes or hit EOF, so a BOM split over
        # 1-byte reads is still recognised.  This buffers at most 3 bytes.
        buf = b""
        while len(buf) < len(_BOM):
            chunk = self._read_bytes(len(_BOM) - len(buf))
            if not chunk:
                break
            buf += chunk
        if buf.startswith(_BOM):
            buf = buf[len(_BOM) :]
        self._prefix = buf
        self._started = True

    def read(self, nbytes: int = -1) -> bytes:
        if not self._started:
            self._strip_bom()
        if self._prefix:
            if nbytes is None or nbytes < 0:
                out = self._prefix + self._read_bytes(-1)
                self._prefix = b""
                return out
            out = self._prefix[:nbytes]
            self._prefix = self._prefix[nbytes:]
            if len(out) < nbytes:
                out += self._read_bytes(nbytes - len(out))
            return out
        if nbytes is None:
            nbytes = -1
        return self._read_bytes(nbytes)
