"""
Writer used by ``DataFrame.to_csv(sep_mode="literal")``.

It presents the same tiny surface as ``csv.writer`` (``writerow`` /
``writerows``) so the existing chunked row batch code in
``pandas/_libs/writers.pyx`` and ``CSVFormatter`` can keep driving it.  Per-cell
encoding is delegated to ``pandas._libs.writers.literal_encode_field``, which
uses the same boundary primitives as the C reader.
"""
from __future__ import annotations

import csv as csvlib
from typing import (
    Any,
    Iterable,
)

from pandas._libs import writers as libwriters


class LiteralWriter:
    """A ``csv.writer``-compatible writer for literal multi-character separators."""

    def __init__(
        self,
        handle: Any,
        *,
        sep: str,
        quoting: int = csvlib.QUOTE_MINIMAL,
        quotechar: str | None = '"',
        doublequote: bool = True,
        escapechar: str | None = None,
        lineterminator: str = "\n",
    ) -> None:
        self._handle = handle
        self._sep = sep
        self._sep_bytes = sep.encode("ascii")
        self._quoting = int(quoting)
        self._quotechar = quotechar
        self._doublequote = bool(doublequote)
        self._escapechar = escapechar
        self._lineterminator = lineterminator
        self._binary = _handle_is_binary(handle)

    def _feed(self, line: bytes) -> None:
        if self._binary:
            self._handle.write(line)
        else:
            self._handle.write(line.decode("utf-8"))

    def writerow(self, row: Iterable[Any]) -> None:
        row = list(row)
        last = len(row) - 1
        cells = [
            libwriters.literal_encode_field(
                value,
                self._sep_bytes,
                self._quoting,
                self._quotechar,
                self._doublequote,
                self._escapechar,
                i == last,
            )
            for i, value in enumerate(row)
        ]
        line = self._sep_bytes.join(cells) + self._lineterminator.encode("utf-8")
        self._feed(line)

    def writerows(self, rows: Iterable[Iterable[Any]]) -> None:
        for row in rows:
            self.writerow(row)


def _handle_is_binary(handle: Any) -> bool:
    mode = getattr(handle, "mode", None)
    if isinstance(mode, str):
        return "b" in mode
    # BytesIO and friends
    try:
        import io

        if isinstance(handle, (io.RawIOBase, io.BufferedIOBase)):
            return True
    except Exception:  # pragma: no cover - defensive
        pass
    return False
