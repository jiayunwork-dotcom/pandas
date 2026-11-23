"""Tests for nullable (masked) integer / boolean columns in the HDF table format."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pandas import (
    DataFrame,
    HDFStore,
    Series,
    _testing as tm,
    concat,
    read_hdf,
)
from pandas.tests.io.pytables.common import (
    ensure_clean_store,
)

pytestmark = [pytest.mark.single_cpu]

tables = pytest.importorskip("tables")


NULLABLE_DTYPES = [
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "boolean",
]

_INT_BOUNDS = {
    "Int8": (-(2**7), 2**7 - 1),
    "Int16": (-(2**15), 2**15 - 1),
    "Int32": (-(2**31), 2**31 - 1),
    "Int64": (-(2**63), 2**63 - 1),
    "UInt8": (0, 2**8 - 1),
    "UInt16": (0, 2**16 - 1),
    "UInt32": (0, 2**32 - 1),
    "UInt64": (0, 2**64 - 1),
}


def _hand_table():
    """
    A small table whose every row can be checked by hand.

    rows (logical values):
        a (Int64) : [1, <NA>, 3, -5, 2**53, 2**53 + 1]
        b (bool)  : [True, False, <NA>, True, False, True]
        u (UInt64): [2**64-1, <NA>, 0, 7, 2**63, 3]
        s         : ['x', 'y', 'z', 'w', 'v', 'u']
    """
    return DataFrame(
        {
            "a": Series([1, None, 3, -5, 2**53, 2**53 + 1], dtype="Int64"),
            "b": Series([True, False, None, True, False, True], dtype="boolean"),
            "u": Series([2**64 - 1, None, 0, 7, 2**63, 3], dtype="UInt64"),
            "s": ["x", "y", "z", "w", "v", "u"],
        }
    )


# ---------------------------------------------------------------------------
# round trip / dtype / values / missingness / order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", NULLABLE_DTYPES)
def test_nullable_roundtrip_dtype(setup_path, dtype):
    if dtype == "boolean":
        values = [True, None, False, True]
        expected = Series(values, dtype="boolean", name="c")
    else:
        lo, hi = _INT_BOUNDS[dtype]
        values = [lo, None, hi, 0 if lo <= 0 <= hi else lo]
        # for unsigned the "0" may duplicate the lower bound; that is fine
        expected = Series(values, dtype=dtype, name="c")
    df = DataFrame({"c": expected, "other": [1.0, 2.0, 3.0, 4.0]})
    with ensure_clean_store(setup_path) as store:
        store.put("df", df, format="table")
        result = store.get("df")
    tm.assert_series_equal(result["c"], expected)
    tm.assert_frame_equal(result, df.reset_index(drop=True))


def test_nullable_roundtrip_all_types_mixed(setup_path):
    df = _hand_table()
    with ensure_clean_store(setup_path) as store:
        store.put("df", df, format="table")
        result = store.get("df")
        selected = store.select("df")
    tm.assert_frame_equal(result, df)
    tm.assert_frame_equal(selected, df)
    assert result.dtypes["a"] == "Int64"
    assert result.dtypes["b"] == "boolean"
    assert result.dtypes["u"] == "UInt64"


def test_nullable_read_hdf_and_to_hdf(tmp_path):
    path = tmp_path / "nullable.h5"
    df = _hand_table()
    df.to_hdf(path, key="df", format="table")
    result = read_hdf(path, "df")
    tm.assert_frame_equal(result, df)


def test_nullable_index_and_column_order(setup_path):
    df = _hand_table().iloc[[3, 0, 3, 1]].set_axis([10, 20, 20, 30])
    # scrambled column order
    df = df[["s", "b", "u", "a"]]
    with ensure_clean_store(setup_path) as store:
        store.put("df", df, format="table")
        result = store.get("df")
    tm.assert_frame_equal(result, df)


def test_nullable_duplicate_index_preserved(setup_path):
    df = DataFrame({"a": Series([1, None, 3], dtype="Int64")}, index=[7, 7, 9])
    with ensure_clean_store(setup_path) as store:
        store.put("df", df, format="table")
        result = store.get("df")
    tm.assert_frame_equal(result, df)


# ---------------------------------------------------------------------------
# append semantics
# ---------------------------------------------------------------------------


def test_nullable_partial_missing_append(setup_path):
    first = DataFrame(
        {
            "a": Series([1, None, 3], dtype="Int64"),
            "b": Series([True, False, None], dtype="boolean"),
            "s": ["x", "y", "z"],
        }
    )
    second = DataFrame(
        {
            "a": Series([None, 5], dtype="Int64"),
            "b": Series([None, True], dtype="boolean"),
            "s": ["p", "q"],
        }
    )
    expected = first._append(second)
    with ensure_clean_store(setup_path) as store:
        store.put("df", first, format="table")
        store.append("df", second)
        result = store.get("df")
    tm.assert_frame_equal(result, expected)


def test_nullable_first_batch_all_missing_then_valid(setup_path):
    first = DataFrame(
        {
            "a": Series([None, None], dtype="Int64"),
            "b": Series([None, None], dtype="boolean"),
            "s": ["x", "y"],
        }
    )
    second = DataFrame(
        {
            "a": Series([10, None], dtype="Int64"),
            "b": Series([True, False], dtype="boolean"),
            "s": ["z", "w"],
        }
    )
    with ensure_clean_store(setup_path) as store:
        store.put("df", first, format="table", data_columns=["a", "b"])
        store.append("df", second)
        result = store.get("df")
        # the dtype must not have shifted despite the all-missing first batch
        assert result.dtypes["a"] == "Int64"
        assert result.dtypes["b"] == "boolean"
        assert result["a"].isna().tolist() == [True, True, False, True]
        assert result["a"].iloc[2] == 10
        assert result["b"].isna().tolist() == [True, True, False, False]
        assert result["b"].iloc[2:].tolist() == [True, False]
        # and the queries still see the same physical records
        coords = store.select_as_coordinates("df", where="a.isna()")
        assert coords.tolist() == [0, 1, 3]


def test_nullable_append_different_chunksizes(setup_path):
    df = DataFrame(
        {"a": Series([1, None, 3, 4, None, 6], dtype="Int64")}, index=range(6)
    )
    with ensure_clean_store(setup_path) as store:
        store.put("df", df.iloc[:2], format="table", data_columns=["a"])
        store.append("df", df.iloc[2:4], chunksize=1)
        store.append("df", df.iloc[4:], chunksize=3)
        result = store.get("df")
    tm.assert_frame_equal(result, df)


def test_nullable_append_dtype_mismatch_keeps_schema(setup_path):
    first = DataFrame({"a": Series([1, None], dtype="Int64")})
    with ensure_clean_store(setup_path) as store:
        store.put("df", first, format="table", data_columns=["a"])
        for bad in ("Int32", "UInt64", "boolean"):
            if bad == "boolean":
                other = Series([True, False], dtype="boolean")
            else:
                other = Series([5, 6], dtype=bad)
            with pytest.raises(ValueError, match="dtype"):
                store.append("df", DataFrame({"a": other}))
        # existing rows and schema untouched
        tm.assert_frame_equal(store.get("df"), first)
        # a legal append still works
        store.append("df", DataFrame({"a": Series([7], dtype="Int64")}))
        result = store.get("df")
    assert result["a"].isna().tolist() == [False, True, False]
    assert result["a"].iloc[[0, 2]].tolist() == [1, 7]
    assert result.dtypes["a"] == "Int64"


def test_nullable_no_missing_append_does_not_change_dtype(setup_path):
    first = DataFrame({"a": Series([1, 2], dtype="Int64")})
    second = DataFrame({"a": Series([3, 4], dtype="Int64")})
    with ensure_clean_store(setup_path) as store:
        store.put("df", first, format="table")
        store.append("df", second)
        result = store.get("df")
    assert result["a"].dtype == "Int64"
    assert result["a"].tolist() == [1, 2, 3, 4]


def test_nullable_does_not_modify_caller_data(setup_path):
    df = _hand_table()
    snapshot = df.copy(deep=True)
    with ensure_clean_store(setup_path) as store:
        store.put("df", df, format="table")
    tm.assert_frame_equal(df, snapshot)


# ---------------------------------------------------------------------------
# physical layout / versioning
# ---------------------------------------------------------------------------


def test_nullable_physical_layout(tmp_path):
    path = tmp_path / "layout.h5"
    df = _hand_table()
    df.to_hdf(path, key="df", format="table", data_columns=["a", "b"])
    with tables.open_file(path, mode="r") as handle:
        node = handle.get_node("/df/table")
        # payload + validity fields exist, no pickled object column
        assert "a" in node.colnames
        assert "a__valid" in node.colnames
        assert "b__valid" in node.colnames
        assert not any(
            c.type == b"object" for c in node.description._v_colobjects.values()
        )
        assert node.col("a").dtype == np.int64
        assert node.col("a__valid").dtype == np.bool_
        # versioned logical -> physical mapping recorded on the node
        layout = json.loads(node._v_attrs.nullable_layout)
        assert layout["version"] == 1
        assert layout["columns"]["a"]["payload"] == "a"
        assert layout["columns"]["a"]["valid"] == "a__valid"
        assert layout["columns"]["a"]["dtype"] == "Int64"


def test_nullable_no_sentinel_for_missing(setup_path):
    # the payload value of a missing entry must not be used as a sentinel:
    # every payload value can also occur as a legitimate present value.
    lo = np.iinfo(np.int64).min
    df = DataFrame({"a": Series([lo, None, 0], dtype="Int64")})
    with ensure_clean_store(setup_path) as store:
        store.put("df", df, format="table", data_columns=["a"])
        result = store.get("df")
        # and querying for the payload value must not select the missing row
        miss = store.select("df", where="a.isna()")
        lo_rows = store.select("df", where=f"a == {lo}")
    assert result["a"].isna().tolist() == [False, True, False]
    assert result["a"].iloc[[0, 2]].tolist() == [lo, 0]
    assert miss.index.tolist() == [1]
    assert lo_rows.index.tolist() == [0]


def test_nullable_unknown_layout_version(tmp_path):
    path = tmp_path / "corrupt.h5"
    df = DataFrame({"a": Series([1, None], dtype="Int64")})
    df.to_hdf(path, key="df", format="table", data_columns=["a"])
    with tables.open_file(path, mode="a") as handle:
        node = handle.get_node("/df/table")
        node._v_attrs.nullable_layout = (
            '{"version": 999, "columns": '
            '{"a": {"payload": "a", "valid": "a__valid", "dtype": "Int64"}}}'
        )
    with pytest.raises(ValueError, match="version"):
        read_hdf(path, "df")


def test_nullable_corrupt_mapping(tmp_path):
    path = tmp_path / "corrupt2.h5"
    df = DataFrame({"a": Series([1, None], dtype="Int64")})
    df.to_hdf(path, key="df", format="table", data_columns=["a"])
    with tables.open_file(path, mode="a") as handle:
        node = handle.get_node("/df/table")
        node._v_attrs.nullable_layout = (
            '{"version": 1, "columns": {"a": {"payload": "a", "dtype": "Int64"}}}'
        )
    with pytest.raises(ValueError, match="valid"):
        read_hdf(path, "df")


def test_nullable_user_column_looks_like_helper(setup_path):
    # a user column named like the aux field must not be confused with it
    df = DataFrame(
        {
            "a": Series([1, None, 3], dtype="Int64"),
            "a__valid": ["p", "q", "r"],
        }
    )
    with ensure_clean_store(setup_path) as store:
        store.put("df", df, format="table", data_columns=["a"])
        result = store.get("df")
        res = store.select("df", where="a == 3")
    tm.assert_frame_equal(result, df)
    assert res["a__valid"].tolist() == ["r"]


def test_nullable_helper_name_collision_resolved(setup_path):
    # a *nullable* user column named exactly like the auxiliary field of another
    # nullable column must not collide with it
    df = DataFrame(
        {
            "a": Series([1, None, 3], dtype="Int64"),
            "a__valid": Series([10, None, 30], dtype="Int64"),
        }
    )
    with ensure_clean_store(setup_path) as store:
        store.put("df", df, format="table", data_columns=["a", "a__valid"])
        result = store.get("df")
        a_missing = store.select("df", where="a.isna()").index.tolist()
        a_eq = store.select("df", where="a == 1").index.tolist()
        av_eq = store.select("df", where="a__valid == 10").index.tolist()
        av_missing = store.select("df", where="a__valid.isna()").index.tolist()
    tm.assert_frame_equal(result, df)
    assert a_missing == [1]
    assert a_eq == [0]
    assert av_eq == [0]
    assert av_missing == [1]


# ---------------------------------------------------------------------------
# three-valued where evaluation
# ---------------------------------------------------------------------------


def _expected(series, op, scalar):
    """evaluate a comparison with pandas' BooleanArray semantics"""
    func = {
        "==": "__eq__",
        "!=": "__ne__",
        "<": "__lt__",
        "<=": "__le__",
        ">": "__gt__",
        ">=": "__ge__",
    }[op]
    return getattr(series, func)(scalar)


def _truth(series, op, scalar):
    """the rows whose three-valued result is exactly True"""
    return _expected(series, op, scalar).fillna(False).to_numpy(dtype=bool)


@pytest.mark.parametrize(
    "column,scalar",
    [
        ("a", 1),
        ("a", -5),
        ("a", 2**53),
        ("a", 2**53 + 1),
        ("a", 1.5),
        ("u", 2**64 - 1),
        ("u", 0),
        ("u", 2**63),
        ("b", True),
        ("b", False),
    ],
)
@pytest.mark.parametrize("op", ["==", "!=", "<", "<=", ">", ">="])
def test_nullable_comparison_rows(setup_path, column, scalar, op):
    df = _hand_table()
    if column == "b" and op not in ("==", "!="):
        pytest.skip("boolean only supports == / !=")
    logical = df[column]
    expected = _truth(logical, op, scalar)
    with ensure_clean_store(setup_path) as store:
        store.put("df", df, format="table", data_columns=["a", "b", "u"])
        result = store.select("df", where=f"{column} {op} {scalar}")
        coords = store.select_as_coordinates("df", where=f"{column} {op} {scalar}")
    assert result.index.tolist() == list(np.flatnonzero(expected))
    assert coords.tolist() == list(np.flatnonzero(expected))


@pytest.mark.parametrize(
    "where, expected_idx",
    [
        ("a.isna()", [1]),
        ("a.notna()", [0, 2, 3, 4, 5]),
        ("~(a.isna())", [0, 2, 3, 4, 5]),
        # unknown OR True -> True, so the missing row is selected
        ("(a.isna()) | (a == 1)", [0, 1]),
        # unknown AND ... -> unknown, so the missing row is not selected
        ("(a.isna()) & (a == 1)", []),
        # NOT missing -> definitely present
        ("b.isna()", [2]),
        ("b.notna()", [0, 1, 3, 4, 5]),
        # a != 1 must not select the missing row
        ("a != 1", [2, 3, 4, 5]),
        ("(a >= 0) | b.isna()", [0, 2, 4, 5]),
    ],
)
def test_nullable_isna_combinations(setup_path, where, expected_idx):
    df = _hand_table()
    with ensure_clean_store(setup_path) as store:
        store.put("df", df, format="table", data_columns=["a", "b"])
        result = store.select("df", where=where)
    assert result.index.tolist() == expected_idx


def test_nullable_kleene_not_unknown(setup_path):
    df = _hand_table()
    with ensure_clean_store(setup_path) as store:
        store.put("df", df, format="table", data_columns=["a"])
        # ~(missing comparison) stays unknown, never True
        negated = store.select("df", where="~(a == 1)")
        # a == 1 is True at row 0, unknown at row 1 and False elsewhere
        assert negated.index.tolist() == [2, 3, 4, 5]


def test_nullable_bound_scalar(setup_path):
    df = _hand_table()
    with ensure_clean_store(setup_path) as store:
        store.put("df", df, format="table", data_columns=["a"])
        bound = 3
        assert store.select("df", where="a == bound").index.tolist() == [2]
        bound = 999  # noqa: F841
        assert store.select("df", where="a == bound").index.tolist() == []


def test_nullable_ordinary_condition_unchanged(setup_path):
    # a table with no nullable columns keeps the old semantics
    df = DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", "y", "z"]})
    with ensure_clean_store(setup_path) as store:
        store.put("df", df, format="table", data_columns=["a"])
        result = store.select("df", where="a > 1")
    assert result.index.tolist() == [2]


# ---------------------------------------------------------------------------
# coordinate / delete / projection consistency
# ---------------------------------------------------------------------------


def test_nullable_coordinates_select_remove_agree(setup_path):
    df = _hand_table()
    where = "(a.isna()) | (a >= 3)"
    with ensure_clean_store(setup_path) as store:
        store.put("df", df, format="table", data_columns=["a"])
        sel = store.select("df", where=where)
        coords = store.select_as_coordinates("df", where=where)
        assert sel.index.tolist() == coords.tolist()
        removed = store.remove("df", where=where)
        assert removed == len(coords)
        remaining = store.get("df")
    expected_remaining = df.drop(index=coords.tolist())
    tm.assert_frame_equal(
        remaining.reset_index(drop=True),
        expected_remaining.reset_index(drop=True),
    )


def test_nullable_remove_does_not_drop_duplicate_label(tmp_path):
    path = tmp_path / "dup.h5"
    idx = [5, 5, 5]
    df = DataFrame(
        {"a": Series([1, None, 1], dtype="Int64", index=idx), "s": ["x", "y", "z"]},
        index=idx,
    )
    df.to_hdf(path, key="df", format="table", data_columns=["a"])
    with HDFStore(path, mode="a") as store:
        removed = store.remove("df", where="a.isna()")
        assert removed == 1
        remaining = store.get("df")
    assert remaining["s"].tolist() == ["x", "z"]
    assert remaining.index.tolist() == [5, 5]


def test_nullable_projection_excludes_helpers(setup_path):
    df = _hand_table()
    with ensure_clean_store(setup_path) as store:
        store.put("df", df, format="table", data_columns=["a", "b"])
        projected = store.select("df", where="a.isna()", columns=["s"])
        plain = store.select("df", columns=["s"])
        null_col = store.select_column("df", "a")
    assert projected.columns.tolist() == ["s"]
    assert plain.columns.tolist() == ["s"]
    assert null_col.dtype == "Int64"
    assert null_col.isna().tolist() == [False, True, False, False, False, False]
    assert null_col.dropna().tolist() == [1, 3, -5, 2**53, 2**53 + 1]


def test_nullable_empty_result_keeps_dtype(setup_path):
    df = _hand_table()
    with ensure_clean_store(setup_path) as store:
        store.put("df", df, format="table", data_columns=["a"])
        result = store.select("df", where="a > 10**17")
    assert len(result) == 0
    assert result.dtypes["a"] == "Int64"
    assert result.dtypes["b"] == "boolean"


# ---------------------------------------------------------------------------
# start/stop, iteration, reopen
# ---------------------------------------------------------------------------


def test_nullable_start_stop(setup_path):
    df = _hand_table()
    with ensure_clean_store(setup_path) as store:
        store.put("df", df, format="table", data_columns=["a"])
        ranged = store.select("df", start=1, stop=4)
        ranged_where = store.select("df", where="a.notna()", start=1, stop=4)
    tm.assert_frame_equal(ranged, df.iloc[1:4])
    assert ranged_where.index.tolist() == [2, 3]


def test_nullable_iteration_matches_full_read(setup_path):
    df = DataFrame(
        {"a": Series([1, None, 3, 4, None, 6, 7, None], dtype="Int64")},
        index=list(range(8)),
    )
    with ensure_clean_store(setup_path) as store:
        store.put("df", df, format="table", data_columns=["a"])
        full = store.select("df", where="(a >= 3) | a.isna()")
        chunks = list(store.select("df", where="(a >= 3) | a.isna()", chunksize=3))
        iterated = list(
            store.select("df", where="(a >= 3) | a.isna()", iterator=True, chunksize=2)
        )
    tm.assert_frame_equal(concat(chunks), full)
    tm.assert_frame_equal(concat(iterated), full)


def test_nullable_reopen_and_multi_key(tmp_path):
    path = tmp_path / "multi.h5"
    df1 = _hand_table()
    df2 = DataFrame(
        {
            "a": Series([None, 20, 30], dtype="Int64"),
            "b": Series([True, None, False], dtype="boolean"),
        }
    )
    df1.to_hdf(path, key="k1", format="table", data_columns=["a", "b"])
    df2.to_hdf(path, key="k2", format="table", data_columns=["a", "b"])

    # reopen read-only in a fresh store; no state from the writer
    with HDFStore(path, mode="r") as store:
        assert store.select("k1", where="a.isna()").index.tolist() == [1]
        assert store.select("k2", where="a.isna()").index.tolist() == [0]
        thr = 3  # noqa: F841 (referenced by name inside ``where``)
        assert store.select("k1", where="a >= thr").index.tolist() == [2, 4, 5]
        assert store.select("k2", where="a >= thr").index.tolist() == [1, 2]


def test_nullable_append_after_remove_and_reopen(tmp_path):
    path = tmp_path / "cycle.h5"
    df = _hand_table()
    df.to_hdf(path, key="df", format="table", data_columns=["a"])
    with HDFStore(path, mode="a") as store:
        store.remove("df", where="a.isna()")
        store.append(
            "df",
            DataFrame(
                {
                    "a": Series([None, 100], dtype="Int64"),
                    "b": Series([None, True], dtype="boolean"),
                    "u": Series([None, 1], dtype="UInt64"),
                    "s": ["n", "m"],
                },
            ),
        )
    with HDFStore(path, mode="r") as store:
        result = store.get("df")
        miss = store.select("df", where="a.isna()")
    assert result.dtypes["a"] == "Int64"
    # the second batch's first row is missing, plus the freshly appended one
    assert miss["s"].tolist() == ["n"]
    assert result["s"].tolist() == ["x", "z", "w", "v", "u", "n", "m"]


# ---------------------------------------------------------------------------
# index utilisation (business-block reads)
# ---------------------------------------------------------------------------


def _count_record_reads(table):
    """monkeypatch the low-level record reader to count block reads"""
    original = table._read_records
    state = {"n": 0}

    def wrapper(start, nrecords, recarr):
        state["n"] += 1
        return original(start, nrecords, recarr)

    table._read_records = wrapper
    return table, state, original


def test_nullable_index_reduces_block_reads(tmp_path):
    path = tmp_path / "indexed.h5"
    n = 100_000
    rng = np.random.default_rng(11)
    df = DataFrame(
        {
            "a": Series(rng.integers(1000, 100_000, n), dtype="Int64"),
            "b": Series(rng.random(n) < 0.5, dtype="boolean"),
            "payload": ["x" * 128] * n,
        }
    )
    # put the only matching values in a few contiguous regions
    df.loc[df.index[40_000:40_010], "a"] = 7
    df.to_hdf(path, key="df", format="table", data_columns=["a", "b"])
    with HDFStore(path, mode="a") as store:
        store.create_table_index("df", columns=["a"])
        store.create_table_index("df", columns=["b"])

    with HDFStore(path, mode="r") as store:
        storer = store.get_storer("df")
        table = storer.table
        nchunks = int(np.ceil(table.nrows / table.chunkshape[0]))
        assert nchunks > 10

        _, state, original = _count_record_reads(table)
        try:
            result = store.select("df", where="a == 7")
        finally:
            table._read_records = original
        assert result["a"].tolist() == [7] * 10
        # the on-disk index must actually skip blocks: far fewer than 1/10
        assert state["n"] < nchunks / 10


def test_nullable_uint64_fallback_is_exact(tmp_path):
    path = tmp_path / "u64.h5"
    values = [2**64 - 1, 2**63, 2**63 + 1, 0, 5, None]
    df = DataFrame({"u": Series(values, dtype="UInt64")})
    df.to_hdf(path, key="df", format="table", data_columns=["u"])
    with HDFStore(path, mode="a") as store:
        # creating an index on a uint64 column must not fail; it records a
        # scan fallback instead
        store.create_table_index("df", columns=["u"])
        assert "u" in list(
            getattr(store.get_storer("df").attrs, "nullable_index_fallback", [])
        )
    with HDFStore(path, mode="r") as store:
        assert store.select("df", where=f"u == {2**64 - 1}").index.tolist() == [0]
        assert store.select("df", where=f"u == {2**63 + 1}").index.tolist() == [2]
        assert store.select("df", where="u > 9007199254740992").index.tolist() == [
            0,
            1,
            2,
        ]
        assert store.select("df", where="u.isna()").index.tolist() == [5]


def test_nullable_selective_index_bool(tmp_path):
    path = tmp_path / "idxb.h5"
    n = 50_000
    df = DataFrame(
        {
            "b": Series(np.zeros(n, dtype=bool), dtype="boolean"),
            "payload": ["y" * 64] * n,
        }
    )
    df.loc[df.index[1000:1010], "b"] = True
    df.to_hdf(path, key="df", format="table", data_columns=["b"])
    with HDFStore(path, mode="a") as store:
        store.create_table_index("df", columns=["b"])
    with HDFStore(path, mode="r") as store:
        table = store.get_storer("df").table
        nchunks = int(np.ceil(table.nrows / table.chunkshape[0]))
        _, state, original = _count_record_reads(table)
        try:
            result = store.select("df", where="b == True")
        finally:
            table._read_records = original
    assert len(result) == 10
    assert state["n"] < nchunks / 10


# ---------------------------------------------------------------------------
# mixed dtypes and legacy compatibility
# ---------------------------------------------------------------------------


def test_nullable_mixed_with_legacy_columns(setup_path):
    df = DataFrame(
        {
            "i": Series([1, None, 3], dtype="Int64"),
            "f": [1.5, 2.5, 3.5],
            "s": ["a", "b", "c"],
            "j": Series([True, False, None], dtype="boolean"),
        }
    )
    with ensure_clean_store(setup_path) as store:
        store.put("df", df, format="table", data_columns=["i", "j"])
        result = store.get("df")
        only_i = store.select("df", where="i.isna()")
        both = store.select("df", where="(i.isna()) & (j == False)")
    tm.assert_frame_equal(result, df)
    assert only_i.index.tolist() == [1]
    # row 1: i missing (isna True) AND j == False (present False) -> True
    assert both.index.tolist() == [1]


def test_legacy_table_unaffected(setup_path):
    # an ordinary old-style table (no nullable columns) still reads as before
    df = DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0], "s": ["x", "y", "z"]})
    with ensure_clean_store(setup_path) as store:
        store.put("df", df, format="table", data_columns=["a"])
        result = store.get("df")
        sel = store.select("df", where="a >= 2")
    tm.assert_frame_equal(result, df)
    assert sel.index.tolist() == [1, 2]


def test_nullable_reconstruct_from_pytables_docs(tmp_path):
    """
    Rebuild a small table by hand from what a plain PyTables reader sees.

    This avoids simply validating the new writer against the new reader: the
    expected logical values are derived from the raw physical payload &
    validity columns only.
    """
    path = tmp_path / "physical.h5"
    df = _hand_table()
    df.to_hdf(path, key="df", format="table", data_columns=["a", "b", "u"])

    with tables.open_file(path, mode="r") as handle:
        node = handle.get_node("/df/table")
        # a plain PyTables user sees the payload fields plus validity fields
        raw = node.read()
        payload_a = raw["a"]
        valid_a = raw["a__valid"]
        payload_b = raw["b"]
        valid_b = raw["b__valid"]

    # rebuild Int64 from payload + validity (bytes -> python ints)
    reconstructed_a = [
        int(v) if valid else None for v, valid in zip(payload_a, valid_a)
    ]
    reconstructed_b = [
        bool(v) if valid else None for v, valid in zip(payload_b, valid_b)
    ]
    assert reconstructed_a == [1, None, 3, -5, 2**53, 2**53 + 1]
    assert reconstructed_b == [True, False, None, True, False, True]

    # and the pandas reader agrees with the hand reconstruction
    with HDFStore(path, mode="r") as store:
        result = store.get("df")
    assert result["a"].tolist() == Series(reconstructed_a, dtype="Int64").tolist()
    assert result["b"].tolist() == Series(reconstructed_b, dtype="boolean").tolist()


def test_nullable_append_subset_of_columns_ok(setup_path):
    df = DataFrame({"a": Series([1, None, 3], dtype="Int64"), "f": [1.0, 2.0, 3.0]})
    with ensure_clean_store(setup_path) as store:
        store.put("df", df, format="table", data_columns=["a"])
        store.append(
            "df",
            DataFrame({"a": Series([4, 5], dtype="Int64"), "f": [4.0, 5.0]}),
        )
        result = store.get("df")
    assert result["a"].isna().tolist() == [False, True, False, False, False]
    assert result["a"].dropna().tolist() == [1, 3, 4, 5]
    assert result.dtypes["a"] == "Int64"
