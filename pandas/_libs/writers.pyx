cimport cython
from cython cimport Py_ssize_t
import numpy as np

from cpython.bytes cimport PyBytes_AsString
from cpython cimport (
    PyBytes_GET_SIZE,
    PyUnicode_GET_LENGTH,
)
from libc.stdlib cimport (
    free,
    malloc,
)
from libc.string cimport memcmp
from numpy cimport (
    int64_t,
    ndarray,
    uint8_t,
)

cdef extern from "pandas/parser/literal.h":
    int pd_literal_kmp_step(const char *sep, int64_t n, const int *fail,
                            int *state, unsigned char ch) nogil
    void pd_literal_build_failure(const char *sep, int64_t n, int *fail) nogil
    int pd_literal_border_ambiguous(const char *field, int64_t len,
                                    const char *sep, int64_t sep_len,
                                    char *scratch) nogil

ctypedef fused pandas_string:
    str
    bytes


@cython.boundscheck(False)
@cython.wraparound(False)
def write_csv_rows(
    list data,
    ndarray data_index,
    Py_ssize_t nlevels,
    ndarray cols,
    object writer
) -> None:
    """
    Write the given data to the writer object, pre-allocating where possible
    for performance improvements.

    Parameters
    ----------
    data : list[ArrayLike]
    data_index : ndarray
    nlevels : int
    cols : ndarray
    writer : _csv.writer
    """
    # In crude testing, N>100 yields little marginal improvement
    cdef:
        Py_ssize_t i, j = 0, k = len(data_index), N = 100, ncols = len(cols)
        list rows

    # pre-allocate rows
    rows = [[None] * (nlevels + ncols) for _ in range(N)]

    if nlevels == 1:
        for j in range(k):
            row = rows[j % N]
            row[0] = data_index[j]
            for i in range(ncols):
                row[1 + i] = data[i][j]

            if j >= N - 1 and j % N == N - 1:
                writer.writerows(rows)
    elif nlevels > 1:
        for j in range(k):
            row = rows[j % N]
            row[:nlevels] = list(data_index[j])
            for i in range(ncols):
                row[nlevels + i] = data[i][j]

            if j >= N - 1 and j % N == N - 1:
                writer.writerows(rows)
    else:
        for j in range(k):
            row = rows[j % N]
            for i in range(ncols):
                row[i] = data[i][j]

            if j >= N - 1 and j % N == N - 1:
                writer.writerows(rows)

    if j >= 0 and (j < N - 1 or (j % N) != N - 1):
        writer.writerows(rows[:((j + 1) % N)])


@cython.boundscheck(False)
@cython.wraparound(False)
def literal_encode_field(
    object field,
    bytes sep,
    int quoting,
    object quotechar,
    bint doublequote,
    object escapechar,
    bint is_last,
):
    """
    Encode one cell for the literal multi-character separator writer.

    Returns ``bytes``.  ``quoting`` uses the stdlib ``csv`` constants
    (QUOTE_MINIMAL=0, QUOTE_ALL=1, QUOTE_NONNUMERIC=2, QUOTE_NONE=3).

    Whether the field contains a separator, and whether its tail overlaps the
    following separator, is decided with the shared ``pd_literal_*`` primitives,
    so writer and reader agree on the boundary rule by construction.
    """
    cdef:
        bytes raw
        const char *buf
        int64_t length
        const char *sepbuf
        int64_t sep_len = PyBytes_GET_SIZE(sep)
        int fail[8]
        int state = 0
        int64_t i
        bint contains_sep = False
        bint contains_quote = False
        bint contains_newline = False
        unsigned char ch
        int quote_ord = -1
        int escape_ord = -1
        char *scratch = NULL
        bint border_ambiguous = False

    if field is None:
        field = ""
    if isinstance(field, bytes):
        raw = field
    elif isinstance(field, str):
        raw = (<str>field).encode("utf-8")
    else:
        raw = str(field).encode("utf-8")

    buf = PyBytes_AsString(raw)
    length = PyBytes_GET_SIZE(raw)
    sepbuf = PyBytes_AsString(sep)

    if quotechar is not None and len(quotechar) == 1:
        quote_ord = ord(quotechar)
    if escapechar is not None and len(escapechar) == 1:
        escape_ord = ord(escapechar)

    pd_literal_build_failure(sepbuf, sep_len, fail)

    for i in range(length):
        ch = <unsigned char>buf[i]
        if ch == 0x0D or ch == 0x0A:
            contains_newline = True
        if quote_ord >= 0 and <int>ch == quote_ord:
            contains_quote = True
        if pd_literal_kmp_step(sepbuf, sep_len, fail, &state, ch):
            contains_sep = True
            state = 0

    if length > 0 and not is_last:
        scratch = <char *>malloc(<size_t>(length + sep_len))
        if scratch == NULL:
            raise MemoryError()
        border_ambiguous = pd_literal_border_ambiguous(
            buf, length, sepbuf, sep_len, scratch
        ) != 0
        free(scratch)

    if quoting == 3:  # QUOTE_NONE
        if (contains_sep or border_ambiguous) and escape_ord < 0:
            raise ValueError(
                "sep_mode='literal': a field contains the separator (or its "
                "tail overlaps the next separator) and quoting=QUOTE_NONE "
                "without an escapechar cannot represent it"
            )
        return _literal_encode_unquoted(
            buf, length, sepbuf, sep_len, escape_ord,
            protect_sep=True, border_ambiguous=border_ambiguous,
        )

    if quoting == 1:  # QUOTE_ALL
        return _literal_encode_quoted(
            buf, length, quote_ord, doublequote, escape_ord
        )

    # QUOTE_MINIMAL / QUOTE_NONNUMERIC: quote only when needed.  A separator
    # inside the field, a separator straddling the closing quote boundary, a
    # quote char or a line break all require quoting.  QUOTE_NONNUMERIC has no
    # extra meaning here: the C parser already converted numeric columns.
    need_quote = (
        contains_sep
        or border_ambiguous
        or contains_quote
        or contains_newline
    )
    if need_quote:
        return _literal_encode_quoted(
            buf, length, quote_ord, doublequote, escape_ord
        )
    return _literal_encode_unquoted(
        buf, length, sepbuf, sep_len, escape_ord,
        protect_sep=False, border_ambiguous=False,
    )


cdef bytes _literal_encode_quoted(const char *buf, int64_t length,
                                  int quote_ord, bint doublequote,
                                  int escape_ord):
    cdef:
        Py_ssize_t i
        object chunk = bytearray()
        unsigned char ch
        bint has_escape = escape_ord >= 0

    if quote_ord < 0:
        raise ValueError(
            "sep_mode='literal': quoting requested but quotechar is unset"
        )
    chunk.append(quote_ord)
    for i in range(length):
        ch = <unsigned char>buf[i]
        if has_escape and <int>ch == escape_ord:
            chunk.append(escape_ord)
            chunk.append(ch)
        elif <int>ch == quote_ord:
            if doublequote:
                chunk.append(quote_ord)
                chunk.append(quote_ord)
            else:
                if not has_escape:
                    raise ValueError(
                        "sep_mode='literal': doublequote=False needs an "
                        "escapechar to encode a quotechar inside a field"
                    )
                chunk.append(escape_ord)
                chunk.append(ch)
        else:
            chunk.append(ch)
    chunk.append(quote_ord)
    return bytes(chunk)


cdef bytes _literal_encode_unquoted(const char *buf, int64_t length,
                                    const char *sep, int64_t sep_len,
                                    int escape_ord, bint protect_sep,
                                    bint border_ambiguous):
    """
    Encode a field outside any quotes.

    ``protect_sep`` is set for QUOTE_NONE, where quoting cannot be used and a
    separator (or a tail that overlaps the next separator) must instead be
    neutralised with the escape character.  Escaping every byte of a separator
    occurrence makes the reader treat them as plain content; the escape
    character itself is always doubled.  A borderline tail is broken by escaping
    the final field byte, which the reader still recovers as content.
    """
    cdef:
        Py_ssize_t i
        object chunk = bytearray()
        unsigned char ch
        bint has_escape = escape_ord >= 0
        object marks

    if protect_sep and has_escape:
        marks = bytearray(_literal_mark_seps(buf, length, sep, sep_len))
        if border_ambiguous and length > 0:
            marks[length - 1] = 1
        for i in range(length):
            ch = <unsigned char>buf[i]
            if <int>ch == escape_ord:
                chunk.append(escape_ord)
                chunk.append(ch)
            else:
                if <bint>marks[i]:
                    chunk.append(escape_ord)
                chunk.append(ch)
        return bytes(chunk)

    if protect_sep:
        # No escapechar: anything that needed neutralising was rejected by the
        # caller, so the bytes are plain content.
        for i in range(length):
            chunk.append(<unsigned char>buf[i])
        return bytes(chunk)

    for i in range(length):
        ch = <unsigned char>buf[i]
        if has_escape and <int>ch == escape_ord:
            chunk.append(escape_ord)
            chunk.append(ch)
        else:
            chunk.append(ch)
    return bytes(chunk)


cdef bytes _literal_mark_seps(const char *buf, int64_t length,
                              const char *sep, int64_t sep_len):
    """
    Mark positions covered by left-to-right, non-overlapping separator
    occurrences.  Returns one byte per input byte: 1 if it must be escaped.
    """
    cdef:
        Py_ssize_t i = 0, j
        object marks = bytearray(length)

    while i + sep_len <= length:
        if memcmp(buf + i, sep, <size_t>sep_len) == 0:
            for j in range(sep_len):
                marks[i + j] = 1
            i += sep_len
        else:
            i += 1
    return bytes(marks)


@cython.boundscheck(False)
@cython.wraparound(False)
def convert_json_to_lines(arr: str) -> str:
    """
    replace comma separated json with line feeds, paying special attention
    to quotes & brackets
    """
    cdef:
        Py_ssize_t i = 0, num_open_brackets_seen = 0, length
        bint in_quotes = False, is_escaping = False
        ndarray[uint8_t, ndim=1] narr
        unsigned char val, newline, comma, left_bracket, right_bracket, quote
        unsigned char backslash

    newline = ord("\n")
    comma = ord(",")
    left_bracket = ord("{")
    right_bracket = ord("}")
    quote = ord('"')
    backslash = ord("\\")

    narr = np.frombuffer(arr.encode("utf-8"), dtype="u1").copy()
    length = narr.shape[0]
    for i in range(length):
        val = narr[i]
        if val == quote and i > 0 and not is_escaping:
            in_quotes = ~in_quotes
        if val == backslash or is_escaping:
            is_escaping = ~is_escaping
        if val == comma:  # commas that should be \n
            if num_open_brackets_seen == 0 and not in_quotes:
                narr[i] = newline
        elif val == left_bracket:
            if not in_quotes:
                num_open_brackets_seen += 1
        elif val == right_bracket:
            if not in_quotes:
                num_open_brackets_seen -= 1

    return narr.tobytes().decode("utf-8") + "\n"  # GH:36888


# stata, pytables
@cython.boundscheck(False)
@cython.wraparound(False)
def max_len_string_array(pandas_string[:] arr) -> Py_ssize_t:
    """
    Return the maximum size of elements in a 1-dim string array.
    """
    cdef:
        Py_ssize_t i, m = 0, wlen = 0, length = arr.shape[0]
        pandas_string val

    for i in range(length):
        val = arr[i]
        wlen = word_len(val)

        if wlen > m:
            m = wlen

    return m


cpdef inline Py_ssize_t word_len(object val):
    """
    Return the maximum length of a string or bytes value.
    """
    cdef:
        Py_ssize_t wlen = 0

    if isinstance(val, str):
        wlen = PyUnicode_GET_LENGTH(val)
    elif isinstance(val, bytes):
        wlen = PyBytes_GET_SIZE(val)

    return wlen

# ------------------------------------------------------------------
# PyTables Helpers


@cython.boundscheck(False)
@cython.wraparound(False)
def string_array_replace_from_nan_rep(
    ndarray[object, ndim=1] arr,
    object nan_rep,
) -> None:
    """
    Replace the values in the array with np.nan if they are nan_rep.
    """
    cdef:
        Py_ssize_t length = len(arr), i = 0

    for i in range(length):
        if arr[i] == nan_rep:
            arr[i] = np.nan
