"""
Reproducible resource verification for nullable integer / boolean HDF tables.

Run with::

    PYTHONPATH=/workspace python3.11 scripts/verify_hdf_nullable_resources.py

It is *not* a test in the CI sense (it is far too slow); the pandas regression
suite (``pandas/tests/io/pytables/test_nullable.py``) covers correctness.  This
script produces the IO / memory numbers quoted for the feature.

Measurements
------------
* business-block reads of a sparse equality query, with and without an on-disk
  index, for an ``Int64`` column and a ``boolean`` column.  A "business block
  read" is one call to the low-level ``Table._read_records`` (the routine that
  pulls a table block off disk).  The index read count is the number of
  ``Index.search``/``get_chunkmap`` invocations.  Both are measured by
  monkey-patching the PyTables instance (no pandas state is involved).
* the extra Python-allocation peak of iterating a wide, selective predicate in
  ``chunksize=2048`` batches while discarding each batch, at the 200k and 800k
  row scales; the difference between the two scales must be <= 24 MiB.

Build cost and dependency import are excluded from the query measurement.
"""

from __future__ import annotations

import gc
import json
import os
import tempfile
import tracemalloc

import numpy as np

import pandas as pd


def _make(n: int) -> str:
    path = os.path.join(tempfile.mkdtemp(), "verify.h5")
    rng = np.random.default_rng(20240101)
    # target keys (0 / True) live in only a few contiguous blocks; the rest of
    # the column is drawn from a wide range so a selective predicate hits
    # well under 1/1000 of the rows.
    values = pd.array(rng.integers(1_000_000, 2_000_000, n), dtype="Int64")
    flags = pd.array(np.zeros(n, dtype=bool), dtype="boolean")
    hits = np.arange(n // 2, n // 2 + 8)
    values[hits] = 7
    flags[hits] = True
    payload = ["x" * 128] * n
    df = pd.DataFrame({"value": values, "flag": flags, "payload": payload})
    df.to_hdf(path, key="df", format="table", data_columns=["value", "flag"])
    return path


class _ReadCounter:
    """count business-block reads (``Table._read_records``) and index reads"""

    def __init__(self, table) -> None:
        self.table = table
        columns = getattr(table, "cols", None)
        self.indexes = []
        if columns is not None:
            for name in table.colnames:
                col = getattr(columns, name, None)
                if col is not None and col.is_indexed:
                    self.indexes.append(col.index)
        self.block_reads = 0
        self.index_reads = 0
        self._orig_reads = table._read_records
        self._orig_search = [ix.search for ix in self.indexes]
        self._orig_chunkmap = [ix.get_chunkmap for ix in self.indexes]

    def __enter__(self):
        def reads(start, nrecords, recarr):
            self.block_reads += 1
            return self._orig_reads(start, nrecords, recarr)

        self.table._read_records = reads
        for i, ix in enumerate(self.indexes):
            orig = self._orig_search[i]

            def search(item, _orig=orig):
                self.index_reads += 1
                return _orig(item)

            ix.search = search
            orig_cm = self._orig_chunkmap[i]

            def chunkmap(_orig=orig_cm):
                self.index_reads += 1
                return _orig()

            ix.get_chunkmap = chunkmap
        return self

    def __exit__(self, *args):
        self.table._read_records = self._orig_reads
        for i, ix in enumerate(self.indexes):
            ix.search = self._orig_search[i]
            ix.get_chunkmap = self._orig_chunkmap[i]


def _nchunks(table) -> int:
    return int(np.ceil(table.nrows / table.chunkshape[0]))


def measure_sparse_query(path: str, column: str, where: str) -> dict:
    with pd.HDFStore(path, mode="r") as store:
        table = store.get_storer("df").table
        nchunks = _nchunks(table)
        with _ReadCounter(table) as counter:
            result = store.select("df", where=where)
        return {
            "column": column,
            "where": where,
            "rows": len(result),
            "blocks": nchunks,
            "block_reads": counter.block_reads,
            "index_reads": counter.index_reads,
        }


def measure_iteration_peak(path: str, where: str, chunksize: int = 2048) -> float:
    """peak Python allocations (MiB) while iterating and discarding batches"""
    gc.collect()
    tracemalloc.start()
    try:
        with pd.HDFStore(path, mode="r") as store:
            total = 0
            for chunk in store.select("df", where=where, chunksize=chunksize):
                total += len(chunk)
                del chunk
            del total
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / 2**20


def main() -> None:
    results = {}
    for n in (200_000, 800_000):
        path = _make(n)
        if True:
            with pd.HDFStore(path, mode="r") as _rd:
                nchunks = _nchunks(_rd.get_storer("df").table)
            # baseline: drop any index that was created implicitly, then
            # measure the plain block scan
            with pd.HDFStore(path, mode="a") as store:
                table = store.get_storer("df").table
                for column in ("value", "flag"):
                    col = getattr(table.cols, column)
                    if col.is_indexed:
                        col.remove_index()
            scan_int = measure_sparse_query(path, "value", "value == 7")
            scan_bool = measure_sparse_query(path, "flag", "flag == True")

            with pd.HDFStore(path, mode="a") as store:
                store.create_table_index("df", columns=["value"])
                store.create_table_index("df", columns=["flag"])

            idx_int = measure_sparse_query(path, "value", "value == 7")
            idx_bool = measure_sparse_query(path, "flag", "flag == True")

            # wide, mostly-selecting predicate: exercises the streaming path
            peak = measure_iteration_peak(path, "value >= 0")

            # full row identity check between select / read_hdf / coordinates
            with pd.HDFStore(path, mode="r") as rd:
                full = rd.select("df", where="value == 7")
                coords = rd.select_as_coordinates("df", where="value == 7")
            rh = pd.read_hdf(path, "df", where="value == 7")
            assert pd.testing.assert_frame_equal(full, rh) is None
            assert coords.tolist() == list(range(n // 2, n // 2 + 8))
            assert len(full) == 8

            results[n] = {
                "nchunks": nchunks,
                "scan_int": scan_int,
                "scan_bool": scan_bool,
                "idx_int": idx_int,
                "idx_bool": idx_bool,
                "iter_peak_mib": round(peak, 2),
            }

    print(json.dumps(results, indent=2))

    for n, r in results.items():
        assert r["idx_int"]["block_reads"] < r["nchunks"] / 10, r
        assert r["idx_bool"]["block_reads"] < r["nchunks"] / 10, r
        assert r["idx_int"]["rows"] < n / 1000
        assert r["idx_bool"]["rows"] < n / 1000

    delta = abs(results[800_000]["iter_peak_mib"] - results[200_000]["iter_peak_mib"])
    print(f"iteration peak delta: {delta:.2f} MiB")
    assert delta <= 24, delta
    print("OK")


if __name__ == "__main__":
    main()
