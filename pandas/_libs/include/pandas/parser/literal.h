/*
Copyright (c) 2024, PyData Development Team
All rights reserved.

Distributed under the terms of the BSD Simplified License.

Shared boundary rules for literal multi-character CSV separators.

The reader (pandas/_libs/src/parser/tokenizer.c) and the writer
(pandas/_libs/writers.pyx) both include this header, so the "where does the
next field start" rule exists only once.  The rules are defined on raw bytes.
Because the separator is restricted to printable ASCII, a byte-wise match can
never start or end in the middle of a UTF-8 encoded non-ASCII character: all
non-ASCII UTF-8 bytes are >= 0x80.
*/
#pragma once

#include <stdint.h>
#include <string.h>

#define PD_LITERAL_SEP_MIN 2
#define PD_LITERAL_SEP_MAX 8

/*
 * Return the byte offset of the first complete separator occurrence in
 * ``buf[0:length]``, or -1 if there is none.
 *
 * This is the left-to-right, first-complete-match rule: at every position the
 * whole separator must be present before it counts.  A partial prefix is plain
 * field content.
 */
static inline int64_t pd_literal_first_sep(const char *buf, int64_t length,
                                           const char *sep, int64_t sep_len) {
  if (sep_len <= 0 || length < sep_len) {
    return -1;
  }
  const int64_t last = length - sep_len;
  for (int64_t i = 0; i <= last; ++i) {
    if (buf[i] == sep[0] && memcmp(buf + i, sep, (size_t)sep_len) == 0) {
      return i;
    }
  }
  return -1;
}

/*
 * Is the raw representation ``field`` followed by ``sep`` ambiguous?
 *
 * The reader starts a fresh field scan at the field's first byte.  If, in the
 * byte string ``field + sep``, the first complete separator occurrence is not
 * exactly at ``len(field)`` then a reader would split the field early.  This
 * covers both a separator buried inside the field and a separator that
 * straddles the field/separator border (e.g. sep="aba", field="ab").
 *
 * ``scratch`` must hold at least ``len + sep_len`` bytes; it is only used when
 * the field is non-empty.
 */
static inline int pd_literal_border_ambiguous(const char *field, int64_t len,
                                              const char *sep, int64_t sep_len,
                                              char *scratch) {
  if (len <= 0) {
    return 0;
  }
  memcpy(scratch, field, (size_t)len);
  memcpy(scratch + len, sep, (size_t)sep_len);
  return pd_literal_first_sep(scratch, len + sep_len, sep, sep_len) != len;
}

/*
 * --- Shared KMP automaton -------------------------------------------------
 *
 * Both the reader (incrementally, over the input stream) and the writer (over a
 * field it is about to emit, to decide where escaping is needed) advance the
 * same automaton.  Keeping it here means the "first complete separator, no
 * reuse of overlap, an escaped byte has no structural meaning" rule exists once.
 *
 * State is just the number of trailing bytes that still match a prefix of the
 * separator; by construction those bytes are exactly sep[0:state].
 */
static inline void pd_literal_build_failure(const char *sep, int64_t n,
                                            int *fail) {
  fail[0] = 0;
  int64_t k = 0;
  for (int64_t i = 1; i < n; ++i) {
    while (k > 0 && sep[i] != sep[k]) {
      k = fail[k - 1];
    }
    if (sep[i] == sep[k]) {
      k++;
    }
    fail[i] = (int)k;
  }
}

/*
 * Advance ``*state`` by one ordinary (structural-meaning-bearing) byte.
 *
 * Returns 1 when the byte completes a full separator (the caller treats the
 * whole separator as structural and resets ``*state``), 0 otherwise.  On a
 * non-match the caller must emit the bytes [0, old_state - new_state) as
 * content: those leading bytes stopped being a viable prefix, while the new
 * ``*state`` trailing bytes are the withheld prefix.
 */
static inline int pd_literal_kmp_step(const char *sep, int64_t n,
                                      const int *fail, int *state,
                                      unsigned char ch) {
  int k = *state;
  while (k > 0 && (unsigned char)sep[k] != ch) {
    k = fail[k - 1];
  }
  if ((unsigned char)sep[k] == ch) {
    k++;
  }
  if (k == n) {
    *state = 0;
    return 1;
  }
  *state = k;
  return 0;
}
