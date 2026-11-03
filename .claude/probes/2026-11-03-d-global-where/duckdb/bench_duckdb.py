#!/usr/bin/env python3
"""D-GLOBAL-WHERE P0 probe, DuckDB side (Part 3 of SPEC.md).

LAB SCRIPT. Not production code, not part of any build or CI gate. It measures
DuckDB (python package, in-memory, normal optimizer, no disabled passes) on the
same datasets and the same SQL as the Rust `bench` binary, and writes
`$OUT/duckdb.json` for `compare.py`.

Run:  python3 -I bench_duckdb.py <OUT_DIR> [--threads 1,4] [--runs 201]
                                  [--datasets dr64_sparse,dr256_medium]

Procedure (one fresh in-memory connection per dataset x threads x workload, so
every cold figure is a real cold one and W8's DELETE cannot leak into other
workloads):
  1. `SET threads=T`; load nodes/edges/selA/selB/del from the CSVs with
     `read_csv` and explicit INTEGER types; `ANALYZE` (supported by DuckDB 2.0.0).
     load_us = CSV load + ANALYZE, reported separately from cold.
  2. cold_us = wall time of `PREPARE q AS <sql>` + first `EXECUTE q` (fetchone).
  3. warm = median / p10 / p90 of RUNS (default 201) `EXECUTE q` + fetchone().
  4. engine_us: one extra EXECUTE with JSON profiling on; the value is
     `query.total_time` (seconds) converted to microseconds. DuckDB 2.0.0 has
     NO top-level `latency` field in its JSON profile (top-level keys are io,
     operator, optimizer, physical_planner, planner, query, system), so
     `query.total_time` is the whole-statement engine time, found by inspecting
     a real profile file.
  5. floor_us = median of RUNS `EXECUTE f` for `PREPARE f AS SELECT 1`
     (python call + fetch overhead, measured on the same connection).
  6. mem_bytes = sum(memory_usage_bytes) FROM duckdb_memory() after the workload.
  7. T=1 only, datasets dr256_medium and multi_skewed, after the measurement:
     `EXPLAIN ANALYZE <sql>`
     text goes to $OUT/explain/<dataset>_<workload>.txt.
  8. W8: warm W2, time `DELETE FROM edges WHERE (src, dst) IN (SELECT src, dst
     FROM del)` (update_us), warm W2 again. Record "W8" (before delete, with
     update_us) and "W8_after".

All times are microseconds. CSV header detection is automatic (first token of
the first line non-numeric => header=true).
"""
import json
import os
import statistics
import sys
import tempfile
import time

import duckdb

# Exact SQL from SPEC.md Part 2; quoted verbatim by the report.
SQL = {
    "W1": "SELECT count(*) FROM nodes WHERE a < 50",
    "W2": "SELECT count(*) FROM nodes n WHERE n.a < 50 AND EXISTS (SELECT 1 FROM edges e JOIN nodes m ON m.id = e.dst WHERE e.src = n.id AND m.b < 300)",
    "W3": "SELECT count(*) FROM nodes n WHERE n.a < 50 AND EXISTS (SELECT 1 FROM edges e1 JOIN edges e2 ON e2.src = e1.dst JOIN nodes m ON m.id = e2.dst WHERE e1.src = n.id AND m.b < 300)",
    "W4": "SELECT count(DISTINCT e3.dst) FROM nodes n JOIN edges e1 ON e1.src = n.id JOIN edges e2 ON e2.src = e1.dst JOIN edges e3 ON e3.src = e2.dst JOIN nodes m ON m.id = e3.dst WHERE n.a < 50 AND m.b < 300",
    "W5a": "SELECT count(DISTINCT e.dst) FROM edges e JOIN nodes n ON n.id = e.src WHERE n.a < 50",
    "W6": "SELECT count(*) FROM edges e1 JOIN edges e2 ON e2.src = e1.dst JOIN nodes n ON n.id = e1.src WHERE n.a < 50",
    "W7": "SELECT count(*) FROM selA s WHERE EXISTS (SELECT 1 FROM edges e JOIN selB t ON t.id = e.dst WHERE e.src = s.id)",
    "W7i": "SELECT count(*) FROM selA JOIN selB USING (id)",
}
DELETE_SQL = "DELETE FROM edges WHERE (src, dst) IN (SELECT src, dst FROM del)"
# W8 uses the W2 text; its SQL is SQL["W2"] plus DELETE_SQL.
WORKLOADS = ["W1", "W2", "W3", "W4", "W5a", "W6", "W7", "W7i", "W8"]
EXPLAIN_DATASETS = {"dr256_medium", "multi_skewed"}


def us(ns):
    return round(ns / 1000.0, 3)


def pct(sorted_vals, p):
    i = min(len(sorted_vals) - 1, max(0, int(round(p * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


def stats(vals):
    s = sorted(vals)
    return statistics.median(s), pct(s, 0.10), pct(s, 0.90)


def has_header(path):
    with open(path) as f:
        first = f.readline().split(",")[0].strip()
    try:
        int(first)
        return False
    except ValueError:
        return True


def q(path):
    return path.replace("'", "''")


def load(con, out, ds):
    """Create the five tables; returns load time in microseconds (incl. ANALYZE)."""
    t0 = time.perf_counter_ns()
    nodes = os.path.join(out, "nodes.csv")
    spec = [
        ("nodes", nodes, "{'id':'INTEGER','a':'INTEGER','b':'INTEGER'}"),
        ("edges", os.path.join(out, f"{ds}_edges.csv"), "{'src':'INTEGER','dst':'INTEGER'}"),
        ("selA", os.path.join(out, f"{ds}_selA.csv"), "{'id':'INTEGER'}"),
        ("selB", os.path.join(out, f"{ds}_selB.csv"), "{'id':'INTEGER'}"),
        ("del", os.path.join(out, f"{ds}_delete.csv"), "{'src':'INTEGER','dst':'INTEGER'}"),
    ]
    for name, path, cols in spec:
        hdr = "true" if has_header(path) else "false"
        con.execute(
            f"CREATE TABLE {name} AS SELECT * FROM read_csv('{q(path)}', header={hdr}, columns={cols})"
        )
    con.execute("ANALYZE")
    return us(time.perf_counter_ns() - t0)


def mem_bytes(con):
    r = con.execute("SELECT sum(memory_usage_bytes) FROM duckdb_memory()").fetchone()[0]
    return int(r) if r is not None else None


def timed_exec(con, name, runs):
    """`runs` timed EXECUTEs; returns (answer, [ns])."""
    out = []
    ans = None
    stmt = f"EXECUTE {name}"
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        ans = con.execute(stmt).fetchone()[0]
        out.append(time.perf_counter_ns() - t0)
    return ans, out


def engine_us(con, name, tmpdir):
    path = os.path.join(tmpdir, "profile.json")
    if os.path.exists(path):
        os.remove(path)
    con.execute("PRAGMA enable_profiling='json'")
    con.execute(f"PRAGMA profiling_output='{q(path)}'")
    con.execute(f"EXECUTE {name}").fetchone()
    con.execute("PRAGMA disable_profiling")
    with open(path) as f:
        prof = json.load(f)
    return round(prof["query"]["total_time"] * 1e6, 3)


def floor_us(con, runs):
    con.execute("PREPARE f AS SELECT 1")
    _, ns = timed_exec(con, "f", runs)
    con.execute("DEALLOCATE f")
    return us(statistics.median(ns))


def measure(con, sql, runs, tmpdir):
    """cold + warm + engine for one SQL on an already loaded connection."""
    t0 = time.perf_counter_ns()
    con.execute(f"PREPARE q AS {sql}")
    ans = con.execute("EXECUTE q").fetchone()[0]
    cold = us(time.perf_counter_ns() - t0)
    ans2, ns = timed_exec(con, "q", runs)
    assert ans == ans2, f"answer changed between cold and warm: {ans} vs {ans2}"
    med, p10, p90 = stats(ns)
    eng = engine_us(con, "q", tmpdir)
    return ans, cold, us(med), us(p10), us(p90), eng


def warm_only(con, sql, runs, tmpdir, name):
    con.execute(f"PREPARE {name} AS {sql}")
    t0 = time.perf_counter_ns()
    first = con.execute(f"EXECUTE {name}").fetchone()[0]
    cold = us(time.perf_counter_ns() - t0)
    ans, ns = timed_exec(con, name, runs)
    assert first == ans
    med, p10, p90 = stats(ns)
    return ans, cold, us(med), us(p10), us(p90), engine_us(con, name, tmpdir)


def rec(ds, T, wl, vals, floor, upd, mem, load_us):
    ans, cold, med, p10, p90, eng = vals
    return {
        "dataset": ds, "threads": T, "workload": wl, "answer": ans,
        "cold_us": cold, "warm_med_us": med, "warm_p10_us": p10, "warm_p90_us": p90,
        "engine_us": eng, "floor_us": floor, "update_us": upd, "mem_bytes": mem,
        "load_us": load_us,
    }


def run_one(out, ds, T, wl, runs, tmpdir):
    con = duckdb.connect(":memory:")
    con.execute(f"SET threads={T}")
    load_us = load(con, out, ds)
    records = []
    if wl != "W8":
        sql = SQL[wl]
        vals = measure(con, sql, runs, tmpdir)
        fl = floor_us(con, runs)
        # Plan collection runs AFTER the cold and warm measurement, on its own
        # fresh connection: EXPLAIN ANALYZE's execution cannot pre-warm the
        # cold figure, and the engine-time profiling mode set on `con` cannot
        # turn the plan text into JSON.
        if T == 1 and ds in EXPLAIN_DATASETS:
            econ = duckdb.connect(":memory:")
            econ.execute(f"SET threads={T}")
            load(econ, out, ds)
            txt = econ.execute(f"EXPLAIN ANALYZE {sql}").fetchall()[0][1]
            econ.close()
            os.makedirs(os.path.join(out, "explain"), exist_ok=True)
            with open(os.path.join(out, "explain", f"{ds}_{wl}.txt"), "w") as f:
                f.write(f"-- {sql}\n{txt}\n")
        records.append(rec(ds, T, wl, vals, fl, None, mem_bytes(con), load_us))
    else:
        sql = SQL["W2"]
        vals = measure(con, sql, runs, tmpdir)
        t0 = time.perf_counter_ns()
        con.execute(DELETE_SQL)
        upd = us(time.perf_counter_ns() - t0)
        fl = floor_us(con, runs)
        records.append(rec(ds, T, "W8", vals, fl, upd, mem_bytes(con), load_us))
        after = warm_only(con, sql, runs, tmpdir, "q_after")
        records.append(rec(ds, T, "W8_after", after, fl, None, mem_bytes(con), load_us))
    con.close()
    return records


def discover(out):
    names = sorted(f[: -len("_edges.csv")] for f in os.listdir(out) if f.endswith("_edges.csv"))
    return names


def main(argv):
    if len(argv) < 2 or argv[1].startswith("-"):
        print(__doc__)
        return 2
    out = argv[1]
    threads = [1, 4]
    runs = 201
    only = None
    i = 2
    while i < len(argv):
        if argv[i] == "--threads":
            threads = [int(x) for x in argv[i + 1].split(",")]
        elif argv[i] == "--runs":
            runs = int(argv[i + 1])
        elif argv[i] == "--datasets":
            only = set(argv[i + 1].split(","))
        else:
            print("unknown arg", argv[i])
            return 2
        i += 2
    datasets = [d for d in discover(out) if only is None or d in only]
    if not datasets:
        print("no *_edges.csv found in", out)
        return 2
    records = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for ds in datasets:
            for T in threads:
                for wl in WORKLOADS:
                    records.extend(run_one(out, ds, T, wl, runs, tmpdir))
                print(f"{ds} threads={T} done", flush=True)
    with open(os.path.join(out, "duckdb.json"), "w") as f:
        json.dump(records, f, indent=1)
    print(f"wrote {len(records)} records to {os.path.join(out, 'duckdb.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
