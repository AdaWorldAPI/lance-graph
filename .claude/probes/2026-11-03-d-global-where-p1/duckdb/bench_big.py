#!/usr/bin/env python3
"""D-GLOBAL-WHERE P1 probe, DuckDB side (Part 4 of SPEC.md): the 10M-node corpus.

LAB SCRIPT. Not production code, not part of any build or CI gate. Measures
DuckDB (python package, in-memory, normal optimizer) on the CSVs written by the
Rust `big` binary and writes `$OUT/big/duckdb_big.json` for `compare_big.py`.

Run:  python3 -I bench_big.py <DIR> [<DIR> ...] [--threads 1,4] [--runs 51]
                              [--arms plain,indexed]

Each <DIR> is one corpus size (e.g. $OUT/big_1m, $OUT/big_10m) holding
{nodes,edges,delete}.csv; <DIR>/duckdb_big.json is written into each, and
explain texts go to <DIR>/explain/. Arms: `plain` (load + ANALYZE only) and
`indexed` (additionally CREATE INDEX idx_e ON edges(src); CREATE INDEX idx_n ON
nodes(id); build time reported as index_build_us, charged as preprocessing).
Every record carries "arm". One fresh in-memory connection
per threads x query, so every cold figure is a real cold one and the DELETE
cannot leak into another query's measurement.

Per connection:
  1. SET threads=T; load the CSVs with read_csv and explicit column types
     (id/src/dst BIGINT, a/b INTEGER); ANALYZE. load_us = load + ANALYZE.
     Indexed arm: then the two CREATE INDEX statements (index_build_us), then
     ANALYZE again (not charged to either figure).
  2. cold_us = PREPARE + first EXECUTE (t=300 for parameterized queries).
  3. answer_per_t: one EXECUTE per t in [100..500] (G1 has no parameter, its
     answer is repeated for each t so the shape matches the others).
  4. warm: RUNS EXECUTEs cycling t over [100..500]; wall median/p10/p90.
  5. engine_med_us: one profiled EXECUTE per t (JSON profiling,
     `query.total_time`), median over the five.
  6. trivial_query_wall_us: median wall of a prepared `SELECT 1`. REPORTED, NOT
     SUBTRACTED anywhere.
  7. mem_bytes: sum(memory_usage_bytes) FROM duckdb_memory() after the warm runs.
  8. threads=1 only: EXPLAIN ANALYZE (t=300 inlined) -> <DIR>/explain/G?.txt
     (plain) or <DIR>/explain/indexed_G?.txt.
  9. W8-style update: DELETE FROM edges WHERE (src,dst) IN (SELECT src,dst FROM
     del), timed as update_us; then one EXECUTE per t again ->
     after_delete_per_t.

All times are microseconds.
"""
import json
import os
import statistics
import sys
import tempfile
import time

import duckdb

TS = [100, 200, 300, 400, 500]
COLD_T = 300

SQL = {
    "G1": "SELECT count(*) FROM nodes WHERE a < 1",
    "G2": "SELECT count(*) FROM nodes n WHERE n.a < 1 AND EXISTS (SELECT 1 FROM edges e JOIN nodes m ON m.id = e.dst WHERE e.src = n.id AND m.b < $1)",
    "G3": "SELECT count(*) FROM nodes n WHERE n.a < 1 AND EXISTS (SELECT 1 FROM edges e1 JOIN edges e2 ON e2.src = e1.dst JOIN nodes m ON m.id = e2.dst WHERE e1.src = n.id AND m.b < $1)",
    "G4": "SELECT count(DISTINCT e3.dst) FROM nodes n JOIN edges e1 ON e1.src = n.id JOIN edges e2 ON e2.src = e1.dst JOIN edges e3 ON e3.src = e2.dst JOIN nodes m ON m.id = e3.dst WHERE n.a < 1 AND m.b < $1",
    "G5": "SELECT count(DISTINCT e.dst) FROM edges e JOIN nodes n ON n.id = e.src JOIN nodes m ON m.id = e.dst WHERE n.a < 1 AND m.b < $1",
    "G6": "SELECT count(*) FROM edges e1 JOIN edges e2 ON e2.src = e1.dst JOIN nodes n ON n.id = e1.src JOIN nodes m ON m.id = e2.dst WHERE n.a < 1 AND m.b < $1",
}
DELETE_SQL = "DELETE FROM edges WHERE (src, dst) IN (SELECT src, dst FROM del)"


def us(ns):
    return round(ns / 1000.0, 3)


def pct(s, p):
    return s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]


def has_header(path):
    with open(path) as f:
        first = f.readline().split(",")[0].strip()
    try:
        int(first)
        return False
    except ValueError:
        return True


def qs(path):
    return path.replace("'", "''")


def load(con, big):
    t0 = time.perf_counter_ns()
    spec = [
        ("nodes", "nodes.csv", "{'id':'BIGINT','a':'INTEGER','b':'INTEGER'}"),
        ("edges", "edges.csv", "{'src':'BIGINT','dst':'BIGINT'}"),
        ("del", "delete.csv", "{'src':'BIGINT','dst':'BIGINT'}"),
    ]
    for name, fn, cols in spec:
        p = os.path.join(big, fn)
        hdr = "true" if has_header(p) else "false"
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM read_csv('{qs(p)}', header={hdr}, columns={cols})")
    con.execute("ANALYZE")
    return us(time.perf_counter_ns() - t0)


def exe(con, name, t, param):
    stmt = f"EXECUTE {name}({t})" if param else f"EXECUTE {name}"
    return con.execute(stmt).fetchone()[0]


def engine_us(con, name, t, param, tmpdir):
    path = os.path.join(tmpdir, "profile.json")
    if os.path.exists(path):
        os.remove(path)
    con.execute("PRAGMA enable_profiling='json'")
    con.execute(f"PRAGMA profiling_output='{qs(path)}'")
    exe(con, name, t, param)
    con.execute("PRAGMA disable_profiling")
    with open(path) as f:
        return round(json.load(f)["query"]["total_time"] * 1e6, 3)


def answers(con, name, param):
    if param:
        return [exe(con, name, t, True) for t in TS]
    a = exe(con, name, 0, False)
    return [a] * len(TS)


def run_one(big, T, wl, arm, runs, tmpdir):
    sql = SQL[wl]
    param = "$1" in sql
    con = duckdb.connect(":memory:")
    con.execute(f"SET threads={T}")
    load_us = load(con, big)
    index_us = None
    if arm == "indexed":
        t0 = time.perf_counter_ns()
        con.execute("CREATE INDEX idx_e ON edges(src)")
        con.execute("CREATE INDEX idx_n ON nodes(id)")
        index_us = us(time.perf_counter_ns() - t0)
        con.execute("ANALYZE")

    t0 = time.perf_counter_ns()
    con.execute(f"PREPARE q AS {sql}")
    cold_ans = exe(con, "q", COLD_T, param)
    cold_us = us(time.perf_counter_ns() - t0)

    ans = answers(con, "q", param)
    if param:
        assert cold_ans == ans[TS.index(COLD_T)], "cold answer != warm answer"

    ns = []
    for i in range(runs):
        t = TS[i % len(TS)]
        t0 = time.perf_counter_ns()
        a = exe(con, "q", t, param)
        ns.append(time.perf_counter_ns() - t0)
        assert a == ans[TS.index(t)], f"answer drift at t={t}"
    s = sorted(ns)
    engs = [engine_us(con, "q", t, param, tmpdir) for t in TS]

    con.execute("PREPARE f AS SELECT 1")
    fl = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        con.execute("EXECUTE f").fetchone()
        fl.append(time.perf_counter_ns() - t0)
    triv = us(statistics.median(fl))

    mem = con.execute("SELECT sum(memory_usage_bytes) FROM duckdb_memory()").fetchone()[0]

    if T == 1:
        lit = sql.replace("$1", str(COLD_T))
        txt = con.execute(f"EXPLAIN ANALYZE {lit}").fetchall()[0][1]
        os.makedirs(os.path.join(big, "explain"), exist_ok=True)
        with open(os.path.join(big, "explain", f"{wl}.txt" if arm == "plain" else f"indexed_{wl}.txt"), "w") as f:
            f.write(f"-- {lit}\n{txt}\n")

    t0 = time.perf_counter_ns()
    con.execute(DELETE_SQL)
    upd = us(time.perf_counter_ns() - t0)
    after = answers(con, "q", param)
    con.close()
    return {
        "query": wl, "arm": arm, "threads": T, "index_build_us": index_us, "answer_per_t": ans, "after_delete_per_t": after,
        "cold_us": cold_us, "warm_med_us": us(statistics.median(s)),
        "warm_p10_us": us(pct(s, 0.10)), "warm_p90_us": us(pct(s, 0.90)),
        "engine_med_us": round(statistics.median(engs), 3),
        "trivial_query_wall_us": triv, "load_us": load_us, "update_us": upd,
        "mem_bytes": int(mem) if mem is not None else None,
    }


def main(argv):
    if len(argv) < 2 or argv[1].startswith("-"):
        print(__doc__)
        return 2
    dirs, threads, runs, arms = [], [1, 4], 51, ["plain", "indexed"]
    i = 1
    while i < len(argv):
        if argv[i] == "--threads":
            threads = [int(x) for x in argv[i + 1].split(",")]
            i += 2
        elif argv[i] == "--runs":
            runs = int(argv[i + 1])
            i += 2
        elif argv[i] == "--arms":
            arms = argv[i + 1].split(",")
            i += 2
        elif argv[i].startswith("-"):
            print("unknown arg", argv[i])
            return 2
        else:
            dirs.append(argv[i])
            i += 1
    for big in dirs:
        for fn in ("nodes.csv", "edges.csv", "delete.csv"):
            if not os.path.exists(os.path.join(big, fn)):
                print("missing", os.path.join(big, fn))
                return 2
    with tempfile.TemporaryDirectory() as tmpdir:
        for big in dirs:
            recs = []
            for arm in arms:
                for T in threads:
                    for wl in SQL:
                        recs.append(run_one(big, T, wl, arm, runs, tmpdir))
                        print(f"{big} arm={arm} threads={T} {wl} done", flush=True)
            dst = os.path.join(big, "duckdb_big.json")
            with open(dst, "w") as f:
                json.dump(recs, f, indent=1)
            print(f"wrote {len(recs)} records to {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
