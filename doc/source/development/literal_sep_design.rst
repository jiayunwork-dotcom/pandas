.. _literal_sep_design:

Design notes: whole-string (literal) CSV separators
===================================================

This page records the implementation of ``sep_mode="literal"`` for
:func:`read_csv` and :meth:`DataFrame.to_csv`. It is aimed at maintainers: the
user-facing description lives in :ref:`io.literal_sep`.

The reader and the writer share one boundary rule
-------------------------------------------------

Both ends include ``pandas/_libs/include/pandas/parser/literal.h``. That header
defines the only place the "is this a separator?" question is answered:

* ``pd_literal_first_sep`` — first complete occurrence in a buffer;
* ``pd_literal_kmp_step`` — a single step of the KMP automaton (see below);
* ``pd_literal_border_ambiguous`` — would ``field + sep`` let a reader see a
  separator earlier than the field boundary?

The C tokenizer (``pandas/_libs/src/parser/tokenizer.c``) runs the automaton
incrementally over the input stream. The writer
(``pandas/_libs/writers.pyx::literal_encode_field``) runs the *same* automaton
over the field it is about to emit, and uses the *same*
``pd_literal_border_ambiguous`` check to decide whether a tail needs quoting or
escaping. There is no second, Python-side definition of the rule to drift out of
sync: the reader and writer literally call the same ``static inline`` functions.

Read-side state machine and buffering
-------------------------------------

The rule is: a separator is a complete string matched left to right outside
quoted fields; a complete match is consumed whole without reusing an overlap; a
partial prefix is ordinary content. Matching this incrementally over arbitrary
buffer boundaries needs care because a read may end in the middle of a would-be
separator.

Two candidate designs were considered.

**1. Withhold the trailing bytes (a carry buffer).** Keep the last up-to
``len(sep)-1`` bytes that could be a separator prefix in a small side buffer,
hide them from the field, and re-examine them when the next read arrives.
Costs: one small extra buffer (bounded by 8 bytes); replaying the carried bytes
means re-scanning at most 8 bytes per refill. It is simple, but the carry buffer
either duplicates input bytes or needs pointer bookkeeping, and every structural
transition (quote, escape, line break) has to remember to flush it. In practice
this is the approach that is easy to get subtly wrong for ``aba``-style overlaps.

**2. KMP automaton with only a count (chosen).** Because the separator alphabet
excludes the quote/escape characters and CR/LF, a run of plain field bytes that
could still become the separator is *always exactly a prefix of the separator*.
So the whole state is one integer: how many trailing field bytes still match a
prefix. On each plain byte we advance the KMP automaton; the number of leading
old candidate bytes that stop being viable is emitted as content, and the rest
stay withheld. Nothing is duplicated: the withheld bytes are ``sep[0:cand]``,
recoverable from the separator itself. The state survives a buffer change as a
bare integer, so a separator straddling a read edge is recognised exactly once,
and a refuted prefix is written exactly once and never re-scanned.

Costs of the chosen design:

* **Ordinary text.** One automaton step per unquoted byte, with a failure-table
  fallback that is amortised O(1) (KMP), so linear overall; no allocation on the
  hot path beyond the existing stream buffer.
* **Repeated local prefixes** (``"ababab…"`` with ``sep="aba"``). The failure
  table keeps the work linear even though prefixes recur; only the *count* is
  retained, so repeated prefixes never cause a rescan of already-consumed input.
* **Large fields.** Only the current field is buffered, as today; the automaton
  adds a constant (one ``int``) per field, not per byte.

The only additional memory is ``literal_cand`` plus the fixed 8-entry failure
table, both on the ``parser_t``. ``parser_consume_rows`` and
``parser_trim_buffers`` are untouched, so chunked reads reclaim memory exactly as
before.

Quoting and escaping
--------------------

Quotes only open at the start of a field (existing ``START_FIELD`` behaviour).
Inside a quoted field everything, including the separator, is content. Outside
quotes the automaton drives field membership. When an ``escapechar`` precedes a
byte, that byte loses structural meaning: ``ESCAPED_CHAR``/``ESCAPE_IN_QUOTED_FIELD``
push it directly, so no separator can span an escaped byte, and the automaton's
candidate count is reset at that transition. ``doublequote`` keeps its meaning.

On EOF, any trailing withheld bytes are flushed as content: an incomplete
separator is never structural. This is the only place where a *partially
matched* separator can be left dangling, which is exactly the "last field has no
trailing newline" case.

Writer semantics
----------------

``LiteralWriter`` (``pandas/io/formats/literal_writer.py``) presents the tiny
``csv.writer`` surface (``writerow``/``writerows``) the rest of pandas already
drives, so ``CSVFormatter`` and ``write_csv_rows`` keep their existing chunking,
column formatting, ``na_rep``/``float_format``/``date_format`` handling, and
index/header emission. Each cell is encoded by
``libwriters.literal_encode_field``.

* ``QUOTE_MINIMAL`` quotes a field when it contains the separator, a quotechar,
  CR/LF, **or** when its tail overlaps the following separator
  (``pd_literal_border_ambiguous``). The border check is what prevents the
  ``sep="aba"``, field ``"ab"`` case from emitting ``ababa`` (which a reader
  would split as ``"", "ba"``).
* ``QUOTE_ALL`` quotes every field.
* ``QUOTE_NONE`` cannot quote, so a separator occurrence (or a borderline tail)
  is neutralised with the escape character; without an escapechar it raises
  rather than emitting an unreadable file. Escaping every byte of a separator
  occurrence makes the reader treat them as plain content, and a borderline tail
  is broken by escaping the final field byte.

Both line terminators (``"\n"`` and ``"\r\n"``) are supported, and no extra
column is introduced at end of line.

Resource results (re-run on this tree)
--------------------------------------

``scripts/verify_literal_sep_resources.py`` generates 100 000 and 1 000 000 rows
with ``sep="aba"`` and fields dense in partial prefixes, reads them in fixed
2048-row chunks discarding each batch, and writes to a discarding sink with the
input DataFrame prepared outside the measurement. Observed peak Python
allocations:

.. code-block:: text

    read:  peak@100000=3231245 B, peak@1000000=3313339 B, delta=82094 B (0.078 MiB)
    write: peak@100000=847638 B,  peak@1000000=242268 B,  delta=-605370 B (-0.577 MiB)
    short-read 1/7/4096: each byte read exactly once
    OK

Both deltas are well under the 16 MiB budget (the write delta is negative). A
byte-counting non-seekable source confirms each input byte is requested exactly
once, so there is no re-reading and no retention of the whole input.
