"""
Literal (whole-string) multi-character separator handling shared by
``read_csv(sep_mode="literal")`` and ``DataFrame.to_csv(sep_mode="literal")``.

Both endpoints use the same ``pd_literal_*`` primitives from
``pandas/_libs/include/pandas/parser/literal.h``; only the thin byte-level
plumbing lives here.
"""
from __future__ import annotations

from typing import Final

#: Separator length bounds, in bytes (printable ASCII => bytes == characters).
LITERAL_SEP_MIN: Final = 2
LITERAL_SEP_MAX: Final = 8

#: Modes accepted by ``sep_mode``.  ``"legacy"`` keeps the historical
#: (single-character / regex) behaviour untouched.
LEGACY: Final = "legacy"
LITERAL: Final = "literal"
_VALID_SEP_MODES: Final = (LEGACY, LITERAL)


def validate_sep_mode(sep_mode: object) -> str:
    """
    Check the ``sep_mode`` argument itself.

    Raises
    ------
    ValueError
        If ``sep_mode`` is not one of ``"legacy"`` / ``"literal"``.
    """
    if sep_mode not in _VALID_SEP_MODES:
        raise ValueError(
            f"sep_mode must be one of {_VALID_SEP_MODES!r}, got {sep_mode!r}"
        )
    return sep_mode


def _is_ascii_printable(ch: str) -> bool:
    code = ord(ch)
    # Printable ASCII excluding space (0x20): 0x21..0x7E.
    return 0x21 <= code <= 0x7E


def validate_literal_sep(
    sep: object, *, quotechar: object = '"', escapechar: object = None
) -> str:
    """
    Validate a literal separator.

    The separator must be a string of 2 to 8 printable ASCII characters with no
    spaces, no CR/LF, and must not contain the active quote character or the
    escape character.  It may contain regular-expression metacharacters and may
    have overlapping prefixes/suffixes (``"||"``, ``"aba"``, ``"<->"``).

    Raises
    ------
    ValueError
        If the separator is invalid.
    """
    if not isinstance(sep, str):
        raise ValueError(
            f"sep must be a string when sep_mode='literal', got {type(sep).__name__}"
        )
    if not LITERAL_SEP_MIN <= len(sep) <= LITERAL_SEP_MAX:
        raise ValueError(
            f"sep must be between {LITERAL_SEP_MIN} and {LITERAL_SEP_MAX} "
            f"characters when sep_mode='literal', got {len(sep)} characters"
        )
    for ch in sep:
        if not _is_ascii_printable(ch):
            raise ValueError(
                "sep must contain only printable ASCII characters "
                f"(no spaces, CR, LF or non-ASCII) when sep_mode='literal', "
                f"got {sep!r}"
            )
        if ch in ("\r", "\n"):
            raise ValueError(
                f"sep must not contain CR or LF when sep_mode='literal', got {sep!r}"
            )
    if quotechar is not None and isinstance(quotechar, str) and quotechar:
        if len(quotechar) == 1 and quotechar in sep:
            raise ValueError(
                f"sep must not contain the quotechar {quotechar!r} when "
                "sep_mode='literal'"
            )
    if escapechar is not None and isinstance(escapechar, str) and escapechar:
        if len(escapechar) == 1 and escapechar in sep:
            raise ValueError(
                f"sep must not contain the escapechar {escapechar!r} when "
                "sep_mode='literal'"
            )
    return sep


#: Options that the literal reader/writer deliberately does not accept yet.
LITERAL_READ_UNSUPPORTED: Final = (
    "skiprows",
    "skipfooter",
    "parse_dates",
    "converters",
)

#: Encodings accepted by the literal reader / writer.
LITERAL_ENCODINGS: Final = ("utf-8", "utf-8-sig")


def normalize_encoding(encoding: object) -> str:
    """Return the canonical lower-case form of a literal-mode encoding."""
    if encoding is None:
        return "utf-8"
    return str(encoding).lower()


def validate_literal_encoding(encoding: object) -> str:
    """
    Check that ``encoding`` is one of UTF-8 / UTF-8-SIG.

    Raises
    ------
    ValueError
        If another codec is requested.
    """
    norm = normalize_encoding(encoding)
    if norm not in LITERAL_ENCODINGS:
        raise ValueError(
            "sep_mode='literal' supports only UTF-8 and UTF-8-SIG encodings, "
            f"got {encoding!r}"
        )
    return norm
