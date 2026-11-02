#!/usr/bin/env python3
"""DuckDB timing arm for the lance-graph-quack differential suite.

Runs the same cases as `oracle.py` (same fixtures, same `cases.tsv`) and
reports a DuckDB timing column next to each case. Parity comes first: a case
is timed only if DuckDB's encoded result equals the committed `expected`.

Boundaries, in plain words:
  - `t_execute_ns` includes the Python result conversion of `fetchall()`. That
    is small for the scalar cases and larger for the `rows_proj` row
    projection, which materialises Python tuples.
  - `t_prepare_ns` times `PREPARE q AS <sql>` alone; the `DEALLOCATE q` runs
    OUTSIDE the timed region (it is cleanup, not part of preparing).
  - The numbers are a reference arm: reported, not gated. No timing assertion
    exists anywhere in this file.
  - DuckDB runs single-threaded by default (`--threads 1`) to match Quack's
    single-threaded execution.

Output goes to stdout only; nothing is written to any file.

Usage (run OUTSIDE the repo so tests/duckdb/ does not shadow the package):
  cd /tmp && <venv>/bin/python -I <repo>/crates/lance-graph-quack/tests/duckdb/timing.py \
      [--threads N] [--reps R]
"""

import argparse
import csv
import importlib.util
import pathlib
import statistics
import sys
import time

import duckdb

HERE = pathlib.Path(__file__).resolve().parent

# `-I` removes the script directory from sys.path, so load oracle.py by path.
_spec = importlib.util.spec_from_file_location("lgq_oracle", HERE / "oracle.py")
oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oracle)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--reps", type=int, default=101)
    args = ap.parse_args()

    con = duckdb.connect(database=":memory:")
    con.execute(f"SET threads TO {int(args.threads)}")
    oracle.load(con)

    with open(oracle.CASES, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        assert header == ["id", "sql", "expected"], f"unexpected header: {header}"
        cases = [r for r in reader if r]

    now = time.perf_counter_ns
    failed = 0
    for case_id, sql, expected in cases:
        got = oracle.encode(case_id, con.execute(sql).fetchall())
        if got != expected:
            print(f"DUCKDB case={case_id} parity=FAIL")
            failed += 1
            continue

        prep = []
        for _ in range(args.reps):
            t0 = now()
            con.execute(f"PREPARE q AS {sql}")
            t1 = now()
            con.execute("DEALLOCATE q")
            prep.append(t1 - t0)

        con.execute(f"PREPARE q AS {sql}")
        exe = []
        last = None
        for _ in range(args.reps):
            t0 = now()
            last = con.execute("EXECUTE q").fetchall()
            exe.append(now() - t0)
        con.execute("DEALLOCATE q")

        if oracle.encode(case_id, last) != expected:
            print(f"DUCKDB case={case_id} parity=FAIL")
            failed += 1
            continue

        print(
            f"DUCKDB case={case_id} duckdb={duckdb.__version__} "
            f"threads={args.threads} reps={args.reps} "
            f"t_prepare_ns={int(statistics.median(prep))} "
            f"t_execute_ns={int(statistics.median(exe))} parity=OK"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
