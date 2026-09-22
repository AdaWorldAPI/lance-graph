#!/usr/bin/env python3
"""DuckDB oracle for the lance-graph-quack differential suite.

Loads the three committed fixture CSVs into DuckDB, runs every case's SQL
from `cases.tsv`, and REWRITES the `expected` column in place. DuckDB is the
semantic oracle here — nothing in this file computes an answer by hand.

Usage:  python oracle.py   (run from anywhere; paths are script-relative)

Encoding (must match the Rust side's `run_query`/`run_group` in
`../duckdb_differential.rs` exactly):
  - a bare scalar (COUNT/SUM/MIN/MAX)  -> its decimal string
  - EXISTS(...)                        -> "0" or "1"
  - a grouped result (key, value)      -> "k1:v1;k2:v2;..." sorted by key
  - a projection of `rid`              -> sorted comma-separated row indices
"""

import csv
import pathlib
import sys

try:
    import duckdb
except ImportError:
    sys.exit("duckdb is not importable — run with the venv's python, e.g. "
              "<scratchpad>/ddv/bin/python tests/duckdb/oracle.py")

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
CASES = HERE / "cases.tsv"

# Case ids whose query returns one row of (key, value) pairs to encode as
# "k:v;k:v;...". Every other id is either a bare scalar or `rows_proj`.
GROUPED = {
    "group_count_cc",
    "group_sum_cc",
    "join_group_sum_country",
    "group_min_cc",
    "group_max_cc",
    "group_max_cc_sparse",
    "join_group_count_country",
    "join_group_min_country",
    "group_avg_cc",
    "join_group_avg_country",
}
BOOLEAN = {"exists_neg"}
ROWS = {"rows_proj"}


def encode(case_id: str, rows: list[tuple]) -> str:
    if case_id in ROWS:
        # `rows_proj`'s query is `SELECT rid ... ORDER BY rid` — already
        # sorted; still sort defensively so the encoding's own contract
        # ("sorted") does not silently depend on DuckDB's ORDER BY.
        return ",".join(str(r[0]) for r in sorted(rows, key=lambda r: r[0]))
    if case_id in GROUPED:
        pairs = sorted(rows, key=lambda r: r[0])
        # An empty MIN/MAX group is SQL NULL (the LEFT JOIN over the key
        # series keeps the group); encode it literally, never as 0.
        return ";".join(f"{k}:{'NULL' if v is None else v}" for k, v in pairs)
    if case_id in BOOLEAN:
        (val,) = rows[0]
        return "1" if val else "0"
    # A bare scalar: exactly one row, one column.
    (val,) = rows[0]
    if val is None:
        # NULL only happens if a filter admits zero rows (e.g. an empty
        # MIN/MAX group) — surfaced as the literal string so a fixture bug
        # that makes this happen is visible in cases.tsv rather than
        # silently coerced to 0.
        return "NULL"
    return str(val)


def main() -> None:
    con = duckdb.connect(database=":memory:")
    con.execute(
        f"CREATE TABLE partner AS SELECT * FROM read_csv_auto('{DATA / 'partner.csv'}')"
    )
    con.execute(
        f"CREATE TABLE doc AS SELECT * FROM read_csv_auto('{DATA / 'doc.csv'}')"
    )
    con.execute(
        f"CREATE TABLE line AS SELECT * FROM read_csv_auto('{DATA / 'line.csv'}')"
    )

    with open(CASES, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        assert header == ["id", "sql", "expected"], f"unexpected header: {header}"
        out_rows = [header]
        for row in reader:
            if not row:
                continue
            case_id, sql = row[0], row[1]
            rows = con.execute(sql).fetchall()
            expected = encode(case_id, rows)
            out_rows.append([case_id, sql, expected])
            print(f"{case_id}: {expected}")

    with open(CASES, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerows(out_rows)


if __name__ == "__main__":
    main()
