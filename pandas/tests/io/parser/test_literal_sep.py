"""
Tests for ``sep_mode="literal"``: whole-string multi-character separators that
work identically on the ``read_csv`` and ``to_csv`` sides.
"""
from __future__ import annotations

import io

import numpy as np
import pytest

from pandas import (
    DataFrame,
    MultiIndex,
    concat,
    read_csv,
)
from pandas.errors import ParserError

pytestmark = pytest.mark.filterwarnings("error::pandas.errors.ParserWarning")


# ---------------------------------------------------------------------------
# Independent reference implementation (deliberately not using pandas) used to
# cross-check both directions.  It implements the documented rule: a complete
# separator, matched left to right outside quotes, consuming the whole string
# without reusing an overlap; partial prefixes are content.
# ---------------------------------------------------------------------------
def ref_parse(
    text: str,
    sep: str,
    quotechar: str = '"',
    doublequote: bool = True,
    escapechar: str | None = None,
    quoting: int = 0,
) -> list[list[str]]:
    rows: list[list[str]] = []
    row: list[str] = []
    cur: list[str] = []
    state = "start"
    started = False
    i, n = 0, len(text)

    def end_field() -> None:
        nonlocal cur
        row.append("".join(cur))
        cur = []

    def end_row() -> None:
        nonlocal row, started
        end_field()
        rows.append(row)
        row = []
        started = False

    while i < n:
        c = text[i]
        if state in ("start", "field"):
            if state == "start" and not started:
                started = True
            if state == "start" and c == quotechar and quoting != 3:
                state = "quoted"
                i += 1
                continue
            if escapechar and c == escapechar:
                state = "escaped"
                i += 1
                continue
            if text.startswith(sep, i):
                end_field()
                i += len(sep)
                state = "start"
                continue
            if c == "\r":
                if i + 1 < n and text[i + 1] == "\n":
                    end_row()
                    i += 2
                else:
                    end_row()
                    i += 1
                state = "start"
                continue
            if c == "\n":
                end_row()
                i += 1
                state = "start"
                continue
            cur.append(c)
            state = "field"
            i += 1
            continue
        if state == "escaped":
            cur.append(c)
            state = "field"
            i += 1
            continue
        if state == "quoted":
            if escapechar and c == escapechar:
                state = "esc_quoted"
                i += 1
                continue
            if c == quotechar:
                state = "quote_in_quoted" if doublequote else "field"
                i += 1
                continue
            cur.append(c)
            i += 1
            continue
        if state == "esc_quoted":
            cur.append(c)
            state = "quoted"
            i += 1
            continue
        # quote_in_quoted
        if c == quotechar:
            cur.append(c)
            state = "quoted"
            i += 1
            continue
        if text.startswith(sep, i):
            end_field()
            i += len(sep)
            state = "start"
            continue
        if c in ("\r", "\n"):
            if c == "\r" and i + 1 < n and text[i + 1] == "\n":
                end_row()
                i += 2
            else:
                end_row()
                i += 1
            state = "start"
            continue
        cur.append(c)
        state = "field"
        i += 1

    if state in ("quoted", "esc_quoted"):
        raise ValueError("EOF inside quoted field")
    if state == "escaped":
        raise ValueError("EOF after escapechar")
    if state in ("start", "field", "quote_in_quoted"):
        if state != "start" or row or cur:
            end_row()
    return rows


def _read(text: bytes | str, sep: str, **kw):
    buf = io.BytesIO(text) if isinstance(text, bytes) else io.StringIO(text)
    return read_csv(buf, sep=sep, sep_mode="literal", **kw)


# ---------------------------------------------------------------------------
# Hand-computed cell splits
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,sep,expected",
    [
        # a||b||c => 3 cells
        ("a||b||c\n", "||", [["a", "b", "c"]]),
        # leading separator => empty first cell
        ("||a\n", "||", [["", "a"]]),
        # trailing separator => empty last cell
        ("a||\n", "||", [["a", ""]]),
        # adjacent separators => empty middle cell
        ("a||||b\n", "||", [["a", "", "b"]]),
        # overlapping prefix: "aaba" = "a" + "aba" -> ["a", ""]
        ("aaba\n", "aba", [["a", ""]]),
        # "abaab" = "aba" + "ab" -> ["", "ab"]
        ("abaab\n", "aba", [["", "ab"]]),
        # "ababa" = "aba" + "ba" -> ["", "ba"]
        ("ababa\n", "aba", [["", "ba"]]),
        # separator inside quotes is content
        ('"a||b"||c\n', "||", [["a||b", "c"]]),
        # doubled quotes
        ('"a""q"||c\n', "||", [['a"q', "c"]]),
        # regex metacharacters are literal
        ("a|.|b\n", "|.|", [["a", "b"]]),
        ("a$^b\n", "$^", [["a", "b"]]),
        ("a<->b\n", "<->", [["a", "b"]]),
        # partial prefix is content, then a full separator ends the field
        ("axab||c\n", "ab||", [["ax", "c"]]),
        # newline inside quotes is content
        ('"a\nb"||c\n', "||", [["a\nb", "c"]]),
        ('"a\r\nb"||c\n', "||", [["a\r\nb", "c"]]),
        # UTF-8 content
        ("中||文\n", "||", [["中", "文"]]),
        # no trailing newline on last row
        ("a||b", "||", [["a", "b"]]),
    ],
)
def test_hand_computed_splits(text, sep, expected):
    df = _read(text, sep, header=None, dtype=str, keep_default_na=False)
    assert df.values.tolist() == expected


def test_escaped_separator_is_content():
    # escapechar makes the following character lose structural meaning, so the
    # first "||" is not a separator here.
    df = _read(
        "a\\||b||c\n",
        "||",
        header=None,
        escapechar="\\",
        dtype=str,
        keep_default_na=False,
    )
    assert df.values.tolist() == [["a||b", "c"]]


def test_quote_only_opens_at_field_start():
    df = _read('a"b||c\n', "||", header=None, dtype=str, keep_default_na=False)
    assert df.values.tolist() == [['a"b', "c"]]


def test_quote_none_quotechar_does_not_open():
    df = _read(
        '"a||b"\n', "||", header=None, quoting=3, dtype=str, keep_default_na=False
    )
    assert df.values.tolist() == [['"a', 'b"']]


# ---------------------------------------------------------------------------
# Short-read / boundary sweep
# ---------------------------------------------------------------------------
class _ShortRead:
    """Non-seekable reader that returns at most ``n`` units per ``read`` call."""

    def __init__(self, data, n):
        self._data = data
        self._n = n
        self._pos = 0

    def read(self, size=-1):
        if size is None or size < 0:
            size = self._n
        size = min(size, self._n)
        out = self._data[self._pos : self._pos + size]
        self._pos += len(out)
        return out

    def seekable(self):
        return False


_BOUNDARY_TEXTS = [
    (b"a||b||c\n", "||"),
    (b"||a||\n", "||"),
    (b"a||||b\n", "||"),
    (b"aaba\n", "aba"),
    (b"abaab\n", "aba"),
    (b"ababa\n", "aba"),
    (b'"a||b"||c\n', "||"),
    (b'"a""q"||c\n', "||"),
    (b'"a\nb"||c\n', "||"),
    ("中||文\n".encode(), "||"),
    (b"a||b", "||"),
    (b"x<->y<->z\n", "<->"),
    (b"p|.|q\n", "|.|"),
    (b"a$^b\n", "$^"),
    (b"pre||post||\n", "||"),
    (b"aba||aba\n", "aba"),
]


@pytest.mark.parametrize("text,sep", _BOUNDARY_TEXTS)
def test_short_read_boundary_sweep(text, sep):
    expected = None
    for nbytes in (1, 2, 3, 4, 5, 6, 7, 8, 13, 64, 4096):
        src = _ShortRead(text, nbytes)
        df = read_csv(
            src,
            sep=sep,
            sep_mode="literal",
            header=None,
            dtype=str,
            keep_default_na=False,
        )
        got = df.values.tolist()
        if expected is None:
            expected = got
        else:
            assert got == expected, (sep, nbytes, got, expected)
    # and it matches the independent reference over the decoded text
    assert expected == ref_parse(text.decode(), sep)


@pytest.mark.parametrize("text,sep", _BOUNDARY_TEXTS)
def test_chunked_read_matches_full(text, sep):
    full = read_csv(
        io.BytesIO(text),
        sep=sep,
        sep_mode="literal",
        header=None,
        dtype=str,
        keep_default_na=False,
    )
    for chunksize in (1, 2, 3, 7, 64):
        pieces = list(
            read_csv(
                io.BytesIO(text),
                sep=sep,
                sep_mode="literal",
                header=None,
                dtype=str,
                keep_default_na=False,
                chunksize=chunksize,
            )
        )
        df = concat(pieces, ignore_index=True) if pieces else DataFrame()
        assert df.values.tolist() == full.values.tolist(), chunksize


# ---------------------------------------------------------------------------
# Sources, encodings, non-seekable input
# ---------------------------------------------------------------------------
def test_text_and_binary_sources_match():
    good = "a||b\n1||x\n2||y\n"
    from_text = _read(good, "||")
    from_bytes = _read(good.encode(), "||")
    assert from_text.equals(from_bytes)
    # a genuinely inconsistent later row raises identically from both sources
    bad = "a||b\n1||x\n2||y||z\n"
    with pytest.raises(ParserError):
        _read(bad, "||")
    with pytest.raises(ParserError):
        _read(bad.encode(), "||")


@pytest.mark.parametrize("nbytes", [1, 7, 4096])
def test_short_binary_and_text_reads(nbytes):
    payload = "a||b\n1||x\n2||y\n"
    df_bytes = read_csv(
        _ShortRead(payload.encode(), nbytes), sep="||", sep_mode="literal"
    )
    df_text = read_csv(_ShortRead(payload, nbytes), sep="||", sep_mode="literal")
    expected = DataFrame({"a": [1, 2], "b": ["x", "y"]})
    assert df_bytes.equals(expected)
    assert df_text.equals(expected)


def test_no_reread_of_non_seekable_source():
    class Counting:
        def __init__(self, data):
            self._data = data
            self._pos = 0
            self.total = 0

        def read(self, size=-1):
            if size is None or size < 0:
                size = 1 << 20
            out = self._data[self._pos : self._pos + size]
            self._pos += len(out)
            self.total += len(out)
            return out

        def seekable(self):
            return False

    payload = b"a||b\n" + b"".join(b"%d||v%d\n" % (i, i) for i in range(50))
    src = Counting(payload)
    df = read_csv(src, sep="||", sep_mode="literal")
    assert len(df) == 50
    # each byte is requested exactly once
    assert src.total == len(payload)


def test_utf8_sig_bom_one_byte_reads():
    payload = b"\xef\xbb\xbfa||b\n1||2\n"
    for n in (1, 2, 3, 7, 4096):
        df = read_csv(
            _ShortRead(payload, n),
            sep="||",
            sep_mode="literal",
            encoding="utf-8-sig",
            dtype=str,
            keep_default_na=False,
        )
        assert df.columns.tolist() == ["a", "b"]
        assert df.values.tolist() == [["1", "2"]]


def test_read_utf8_path_and_buffer(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("k||v\n中||文\n", encoding="utf-8")
    assert _read("k||v\n中||文\n", "||", dtype=str).values.tolist() == [["中", "文"]]
    assert read_csv(p, sep="||", sep_mode="literal", dtype=str).values.tolist() == [
        ["中", "文"]
    ]


# ---------------------------------------------------------------------------
# Column conversion / projection / identity
# ---------------------------------------------------------------------------
_GRID = "a||b||c\n1||x||3.5\n2||y||NA\n3||z||1.0\n"


def test_dtype_and_na_handling():
    df = _read(_GRID, "||", dtype={"a": int})
    assert df["a"].dtype == np.int64
    assert pd_isna(df.loc[1, "c"])


def pd_isna(value):
    import pandas as pd

    return pd.isna(value)


def test_keep_default_na_toggle():
    df = _read(_GRID, "||", keep_default_na=False, dtype=str)
    assert df.loc[1, "c"] == "NA"


def test_custom_na_values():
    df = _read(_GRID, "||", na_values=["x"])
    assert pd_isna(df.loc[0, "b"])


def test_usecols_names_and_ints():
    by_name = _read(_GRID, "||", usecols=["a", "c"])
    by_int = _read(_GRID, "||", usecols=[0, 2])
    assert by_name.columns.tolist() == ["a", "c"]
    assert by_name.equals(by_int)


def test_index_col_single():
    by_name = _read(_GRID, "||", index_col="a")
    by_int = _read(_GRID, "||", index_col=0)
    assert by_name.index.tolist() == [1, 2, 3]
    assert by_name.equals(by_int)


def test_full_chunks_iterator_identity():
    full = _read(_GRID, "||", dtype={"a": int})

    chunked = concat(
        read_csv(
            io.StringIO(_GRID),
            sep="||",
            sep_mode="literal",
            dtype={"a": int},
            chunksize=1,
        ),
        ignore_index=True,
    )
    assert full.equals(chunked)

    it = read_csv(
        io.StringIO(_GRID),
        sep="||",
        sep_mode="literal",
        dtype={"a": int},
        iterator=True,
    )
    parts = []
    while True:
        try:
            parts.append(it.read(1))
        except StopIteration:
            break
    iterated = concat(parts, ignore_index=True)
    assert full.equals(iterated)
    assert iterated.index.tolist() == full.index.tolist()


def test_bad_width_reports_line():
    # An inconsistent row after the first one must be a hard error, and the
    # message names the offending line (mirrors the C engine's legacy report).
    with pytest.raises(ParserError, match="line 3"):
        _read("a||b\n1||x\n2||y||3\n", "||")


def test_bad_encoding_still_errors():
    with pytest.raises(UnicodeDecodeError):
        _read(b"a||b\n\xff||2\n", "||")


def test_eof_inside_quote_errors():
    with pytest.raises(ParserError):
        _read('"a||b\n', "||")


def test_eof_after_escape_errors():
    with pytest.raises(ParserError):
        _read(b"a\\", "||", escapechar="\\")


def test_premature_close():
    src = _ShortRead(b"a||b\n1||2\n", 1)
    it = read_csv(src, sep="||", sep_mode="literal", iterator=True)
    it.read(1)
    it.close()


# ---------------------------------------------------------------------------
# Rejected configurations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"sep": "|", "sep_mode": "literal"},  # too short
        {"sep": "|||||||||", "sep_mode": "literal"},  # too long
        {"sep": "| |", "sep_mode": "literal"},  # space
        {"sep": "|\t", "sep_mode": "literal"},  # tab (not printable)
        {"sep": '|"', "sep_mode": "literal"},  # quotechar
        {"sep": "|\\", "sep_mode": "literal", "escapechar": "\\"},
        {"sep": "||", "sep_mode": "literal", "engine": "python"},
        {"sep": "||", "sep_mode": "literal", "engine": "pyarrow"},
        {"sep": "||", "sep_mode": "literal", "skiprows": 1},
        {"sep": "||", "sep_mode": "literal", "skipfooter": 1},
        {"sep": "||", "sep_mode": "literal", "parse_dates": ["a"]},
        {"sep": "||", "sep_mode": "literal", "converters": {"a": str}},
        {"sep": "||", "sep_mode": "literal", "usecols": lambda c: True},
        {"sep": "||", "sep_mode": "literal", "comment": "#"},
        {"sep": "||", "sep_mode": "literal", "on_bad_lines": "skip"},
        {"sep": "||", "sep_mode": "literal", "header": 1},
        {"sep": "||", "sep_mode": "literal", "encoding": "latin-1"},
        {"sep": None, "sep_mode": "literal"},
        {"sep": "||", "sep_mode": "bogus"},
        {"sep": "||", "sep_mode": "literal", "memory_map": True},
    ],
)
def test_rejected_configurations(kwargs):
    data = "a||b\n1||2\n"
    with pytest.raises((ValueError, TypeError)):
        read_csv(io.StringIO(data), **kwargs)


def test_compression_inferred_rejected(tmp_path):
    import gzip

    p = tmp_path / "x.csv.gz"
    with gzip.open(p, "wb") as f:
        f.write(b"a||b\n1||2\n")
    with pytest.raises(ValueError, match="compression"):
        read_csv(p, sep="||", sep_mode="literal")


def test_legacy_mode_unchanged():
    # single-character separators and regex fallbacks are untouched
    assert read_csv(io.StringIO("a,b\n1,2\n")).values.tolist() == [[1, 2]]
    # sep_mode defaults to legacy; a multi-char sep is a regex handled by the
    # Python engine, which splits on each "|" (so it yields empty cells)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = read_csv(io.StringIO("a||b\n1||2\n"), sep="||")
    # legacy regex splits each "|", producing extra empty-ish columns
    assert "|" in df.columns.tolist()


# ---------------------------------------------------------------------------
# Writer: hand-checked bytes, reference comparison, round trips
# ---------------------------------------------------------------------------
def _ref_encode_field(
    value,
    sep,
    quoting=0,
    quotechar='"',
    doublequote=True,
    escapechar=None,
    is_last=False,
):
    field = "" if value is None else str(value)
    contains_sep = sep in field
    border = False
    if field and not is_last:
        joined = field + sep
        first = joined.find(sep)
        border = first != len(field)
    contains_quote = quotechar is not None and quotechar in field
    contains_nl = "\r" in field or "\n" in field
    if quoting == 3:  # QUOTE_NONE
        if (contains_sep or border) and not escapechar:
            raise ValueError("cannot represent")
        out = []
        i = 0
        while i < len(field):
            if field.startswith(sep, i):
                out.extend(escapechar + c for c in sep)
                i += len(sep)
            else:
                c = field[i]
                if escapechar and c == escapechar:
                    out.append(escapechar + c)
                else:
                    out.append(c)
                i += 1
        if border:
            raise ValueError("border cannot be represented without more work")
        return "".join(out)
    need = quoting == 1 or contains_sep or border or contains_quote or contains_nl
    if not need:
        if escapechar:
            return field.replace(escapechar, escapechar + escapechar)
        return field
    body = []
    for c in field:
        if escapechar and c == escapechar:
            body.append(escapechar)
            body.append(c)
        elif c == quotechar:
            if doublequote:
                body.append(quotechar)
                body.append(quotechar)
            else:
                if not escapechar:
                    raise ValueError("need escape")
                body.append(escapechar)
                body.append(c)
        else:
            body.append(c)
    return quotechar + "".join(body) + quotechar


def _ref_write(rows, sep, **kw):
    lines = []
    for row in rows:
        cells = [
            _ref_encode_field(
                v,
                sep,
                kw.get("quoting", 0),
                kw.get("quotechar", '"'),
                kw.get("doublequote", True),
                kw.get("escapechar"),
                i == len(row) - 1,
            )
            for i, v in enumerate(row)
        ]
        lines.append(sep.join(cells))
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize(
    "row,sep",
    [
        (["a", "b"], "||"),
        (["x||y", "z"], "||"),
        (["ab", "c"], "aba"),  # tail overlaps the separator
        (["aab", "c"], "aba"),
        (["", ""], "||"),
        (["a", ""], "||"),
        (["", "b"], "||"),
        (["中||文", "ascii"], "||"),
        (["a\nb", "c"], "||"),
        (['a"b', "c"], "||"),
        (["a<->b", "<->"], "<->"),
        (["a|.|b", "c"], "|.|"),
        (["a$^b", "c"], "$^"),
    ],
)
def test_writer_bytes_match_reference(row, sep):
    df = DataFrame({"p": [row[0]], "q": [row[1]]})
    got = df.to_csv(sep=sep, sep_mode="literal", index=False)
    expected = _ref_write([["p", "q"], row], sep)
    assert got == expected


def test_writer_quote_all():
    df = DataFrame({"a": ["x", "y"], "b": ["||", "z"]})
    got = df.to_csv(sep="||", sep_mode="literal", index=False, quoting=1)
    expected = _ref_write([["a", "b"], ["x", "||"], ["y", "z"]], "||", quoting=1)
    assert got == expected
    out = read_csv(
        io.StringIO(got), sep="||", sep_mode="literal", dtype=str, keep_default_na=False
    )
    assert out.values.tolist() == [["x", "||"], ["y", "z"]]


def test_writer_quote_none_with_escapechar():
    df = DataFrame({"a": ["x||y", "n"], "b": ["ab", "z"]})
    got = df.to_csv(
        sep="||", sep_mode="literal", index=False, quoting=3, escapechar="\\"
    )
    out = read_csv(
        io.StringIO(got),
        sep="||",
        sep_mode="literal",
        quoting=3,
        escapechar="\\",
        dtype=str,
        keep_default_na=False,
    )
    assert out.values.tolist() == [["x||y", "ab"], ["n", "z"]]


def test_writer_quote_none_without_escapechar_errors():
    df = DataFrame({"a": ["x||y"], "b": ["c"]})
    with pytest.raises(ValueError):
        df.to_csv(sep="||", sep_mode="literal", index=False, quoting=3)


def test_writer_quote_none_border_escapechar():
    df = DataFrame({"a": ["ab"], "b": ["c"]})
    got = df.to_csv(
        sep="aba", sep_mode="literal", index=False, quoting=3, escapechar="\\"
    )
    out = read_csv(
        io.StringIO(got),
        sep="aba",
        sep_mode="literal",
        quoting=3,
        escapechar="\\",
        dtype=str,
        keep_default_na=False,
    )
    assert out.values.tolist() == [["ab", "c"]]


def test_writer_doublequote_false():
    df = DataFrame({"a": ['x"y'], "b": ["z"]})
    got = df.to_csv(
        sep="||",
        sep_mode="literal",
        index=False,
        quoting=1,
        doublequote=False,
        escapechar="\\",
    )
    out = read_csv(
        io.StringIO(got),
        sep="||",
        sep_mode="literal",
        doublequote=False,
        escapechar="\\",
        dtype=str,
        keep_default_na=False,
    )
    assert out.values.tolist() == [['x"y', "z"]]


def test_writer_border_disambiguation():
    # sep="aba", field "ab" then separator: naive join would give "ababa"
    # which the reader splits as "", "ba".  The writer must avoid that.
    df = DataFrame({"a": ["ab"], "b": ["c"]})
    s = df.to_csv(sep="aba", sep_mode="literal", index=False)
    rows = ref_parse(s, "aba")
    assert rows[0] == ["a", "b"]
    assert rows[1] == ["ab", "c"]


@pytest.mark.parametrize("sep", ["||", "aba", "<->", "|.|", "$^", "abab"])
def test_writer_roundtrip_randomized(sep):
    import random

    random.seed(0)
    alphabet = list("ab|<>-中$^x. ")
    for _ in range(60):
        rows = []
        for _ in range(random.randint(1, 4)):
            cells = [
                "".join(random.choice(alphabet) for _ in range(random.randint(1, 5)))
                for _ in range(random.randint(1, 3))
            ]
            if any(c.strip() == "" for c in cells):
                continue
            rows.append(cells)
        if not rows:
            continue
        ncols = len(rows[0])
        rows = [r for r in rows if len(r) == ncols]
        if not rows:
            continue
        cols = [f"c{i}" for i in range(ncols)]
        df = DataFrame(rows, columns=cols)
        for quoting in (0, 1):
            esc = None if quoting == 0 else "\\"
            s = df.to_csv(
                sep=sep,
                sep_mode="literal",
                index=False,
                quoting=quoting,
                escapechar=esc,
            )
            # independent parse
            parsed = ref_parse(s, sep, escapechar=esc)
            assert parsed[0] == cols
            assert parsed[1:] == rows
            # pandas parse at every chunk size
            for cs in (1, 2, 7, 64):
                out = concat(
                    read_csv(
                        io.StringIO(s),
                        sep=sep,
                        sep_mode="literal",
                        chunksize=cs,
                        dtype=str,
                        keep_default_na=False,
                        escapechar=esc,
                    ),
                    ignore_index=True,
                )
                assert out.values.tolist() == rows, (sep, quoting, cs)


def test_writer_lineterminators_roundtrip():
    df = DataFrame({"a": ["x||y", "中"], "b": ["z", ""]})
    for term in ("\n", "\r\n"):
        s = df.to_csv(sep="||", sep_mode="literal", index=False, lineterminator=term)
        out = read_csv(
            io.StringIO(s),
            sep="||",
            sep_mode="literal",
            dtype=str,
            keep_default_na=False,
        )
        assert out.values.tolist() == df.values.tolist(), term


def test_writer_path_binary_text_targets(tmp_path):
    df = DataFrame({"a": ["x||y"], "b": ["中"]})
    expected = df.to_csv(sep="||", sep_mode="literal", index=False)

    p = tmp_path / "out.csv"
    df.to_csv(p, sep="||", sep_mode="literal", index=False)
    assert p.read_text(encoding="utf-8") == expected

    b = io.BytesIO()
    df.to_csv(b, sep="||", sep_mode="literal", index=False)
    assert b.getvalue().decode("utf-8") == expected

    t = io.StringIO()
    df.to_csv(t, sep="||", sep_mode="literal", index=False)
    assert t.getvalue() == expected


def test_writer_utf8_sig(tmp_path):
    df = DataFrame({"a": ["x||y"], "b": ["中"]})
    p = tmp_path / "out.csv"
    df.to_csv(p, sep="||", sep_mode="literal", index=False, encoding="utf-8-sig")
    raw = p.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    out = read_csv(p, sep="||", sep_mode="literal", encoding="utf-8-sig")
    assert out.values.tolist() == [["x||y", "中"]]


def test_writer_multiindex_and_dupes_rejected():
    mi_cols = DataFrame(
        [[1, 2]], columns=MultiIndex.from_tuples([("a", "x"), ("b", "y")])
    )
    with pytest.raises(TypeError):
        mi_cols.to_csv(sep="||", sep_mode="literal")
    mi_rows = DataFrame({"a": [1]}, index=MultiIndex.from_tuples([(1, 2)]))
    with pytest.raises(TypeError):
        mi_rows.to_csv(sep="||", sep_mode="literal")
    dup = DataFrame([[1, 2]], columns=["a", "a"])
    with pytest.raises(ValueError):
        dup.to_csv(sep="||", sep_mode="literal")


def test_writer_empty_and_header_only():
    empty = DataFrame({"a": [], "b": []})
    assert empty.to_csv(sep="||", sep_mode="literal", index=False) == "a||b\n"
    header_only = DataFrame({"a": [], "b": []})
    assert (
        header_only.to_csv(sep="||", sep_mode="literal", index=False, header=True)
        == "a||b\n"
    )
    assert (
        DataFrame({"a": [1], "b": [2]}).to_csv(
            sep="||", sep_mode="literal", index=False, header=False
        )
        == "1||2\n"
    )


def test_writer_compression_rejected(tmp_path):
    df = DataFrame({"a": [1]})
    with pytest.raises(ValueError):
        df.to_csv(tmp_path / "x.gz", sep="||", sep_mode="literal", compression="gzip")
