"""
Resource-check for ``sep_mode="literal"``.

Generates the required scale of rows with ``sep="aba"`` and fields dense in
partial prefixes (``ab``, ``a``), then measures:

* the read-side peak Python allocation while streaming fixed-size chunks and
  discarding each batch immediately;
* the write-side peak Python allocation while writing to a discarding sink (the
  input DataFrame is prepared before measurement, so it is excluded).

Also verifies that a non-seekable source is read once (no re-reads) and that the
underlying source may return 1, 7, or 4096 units per ``read`` call.

Usage::

    python scripts/verify_literal_sep_resources.py
"""
from __future__ import annotations

import io
import sys
import tracemalloc

import pandas as pd

SEP = "aba"
CHUNKSIZE = 2048
SCALES = (100_000, 1_000_000)
LIMIT_MIB = 16


def _field_a(i: int) -> str:
    return ("ab" * 60)[:200] + f"{i:06d}"


def _field_b() -> str:
    return (("a" * 3 + "ab") * 20)[:250]


def _field_c(i: int) -> str:
    return "中文" + ("ab" * 8) + f"{i:04d}"


def _generate_text(n: int) -> bytes:
    parts = [f"{_field_a(i)}{SEP}{_field_b()}{SEP}{_field_c(i)}\n" for i in range(n)]
    return "".join(parts).encode("utf-8")


class _NullSink:
    def write(self, s):
        return len(s)

    def flush(self):
        pass

    @property
    def mode(self):
        return "w"


class _ShortRead:
    """Non-seekable source returning at most ``n`` units per read."""

    def __init__(self, data, n):
        self._data = data
        self._n = n
        self._pos = 0
        self.bytes_read = 0

    def read(self, size=-1):
        if size is None or size < 0:
            size = self._n
        size = min(size, self._n)
        out = self._data[self._pos : self._pos + size]
        self._pos += len(out)
        self.bytes_read += len(out)
        return out

    def seekable(self):
        return False


def measure_read(n: int) -> int:
    payload = _generate_text(n)
    src = io.BytesIO(payload)
    reader = pd.read_csv(
        src,
        sep=SEP,
        sep_mode="literal",
        chunksize=CHUNKSIZE,
        header=None,
        dtype=str,
        keep_default_na=False,
    )
    tracemalloc.start()
    total = 0
    for chunk in reader:
        total += len(chunk)
        del chunk
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert total == n, (total, n)
    return peak


def measure_write(n: int) -> int:
    rows = [[_field_a(i), _field_b(), _field_c(i)] for i in range(n)]
    df = pd.DataFrame(rows, columns=["a", "b", "c"])  # excluded from measurement
    del rows
    sink = _NullSink()
    tracemalloc.start()
    df.to_csv(sink, sep=SEP, sep_mode="literal", index=False, chunksize=CHUNKSIZE)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def check_no_reread() -> None:
    payload = _generate_text(500)
    for n in (1, 7, 4096):
        src = _ShortRead(payload, n)
        df = pd.read_csv(src, sep=SEP, sep_mode="literal", header=None, dtype=str)
        assert len(df) == 500, (n, len(df))
        assert src.bytes_read == len(payload), (n, src.bytes_read, len(payload))
    print("short-read 1/7/4096: each byte read exactly once")


def main() -> int:
    read_peaks = {n: measure_read(n) for n in SCALES}
    write_peaks = {n: measure_write(n) for n in SCALES}
    for label, peaks in (("read", read_peaks), ("write", write_peaks)):
        small, large = peaks[SCALES[0]], peaks[SCALES[1]]
        delta = large - small
        print(
            f"{label}: peak@{SCALES[0]}={small} B, "
            f"peak@{SCALES[1]}={large} B, delta={delta} B "
            f"({delta / 2**20:.3f} MiB)"
        )
        if delta > LIMIT_MIB * 2**20:
            print(f"FAIL: {label} peak grew by more than {LIMIT_MIB} MiB")
            return 1
    check_no_reread()
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
