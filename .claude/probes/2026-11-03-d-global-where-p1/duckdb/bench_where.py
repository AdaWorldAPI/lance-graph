#!/usr/bin/env python3
"""D-GLOBAL-WHERE P2 Part 5, DuckDB side: the criteria-aware WHERE.

LAB SCRIPT. Not production code, not part of any build or CI gate. Measures
DuckDB 2.0.0 (python package, in-memory, normal optimizer) on the predicate
trees the Rust `where_k` binary writes to `<DIR>/cases.json`, and writes
`<DIR>/where_duckdb.json` for `compare_where.py`.

Run:  python3 -I bench_where.py <DIR> [<DIR> ...] [--threads 1,4] [--runs 31]

Each <DIR> is `<OUT>/where/n<N>` (one corpus size). The `nodes` table is built
inside DuckDB from the same closed formula the Rust side uses, no CSV:

  f(id, j) = (((id * 2654435761 + j * 40503) % 2147483648) * 1103515245
              % 2147483648) % 1000                (BIGINT, no overflow)
  c_j = f(id, j) for j != 1;  c1 = (c0 + f(id, 1) % 20) % 1000
  c15 = NULL where f(id, 15) % 10 == 0, else f(id, 15)

and the table is ANALYZEd. The predicate SQL is built from the SAME tree, with
SQL three-valued semantics (NULL in c15 makes a comparison UNKNOWN; the WHERE
keeps TRUE rows only), which is exactly what the Rust side's
`Filter::sql_where` is required to reproduce. Answers are checked by
compare_where.py.

Per (threads, case), on one fresh connection per thread count:
  count query:  SELECT count(*) FROM nodes WHERE <pred>
  exists query: SELECT EXISTS(SELECT 1 FROM nodes WHERE <pred>)
Each is a PREPAREd statement (constants inlined: the predicate IS the case).
  cold_us      = PREPARE + first EXECUTE (the table is warm; the statement is not)
  warm         = RUNS EXECUTEs; wall median / p10 / p90
  engine_med_us= median over 5 profiled EXECUTEs of JSON `query.total_time`
  trivial_query_wall_us = median wall of a prepared `SELECT 1`, REPORTED, not
                  subtracted anywhere
threads = 1 only: EXPLAIN ANALYZE of one case per K group to <DIR>/explain/.
All times are microseconds. Load = building `nodes` + ANALYZE, reported per
thread count as build_us / analyze_us / load_us.
"""
import json
import os
import statistics
import sys
import tempfile
import time

import duckdb

SQL_OP = {"lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "=", "ne": "<>"}


def f_expr(j, idv="i"):
    return f"((((({idv} * 2654435761 + {j * 40503}) % 2147483648) * 1103515245) % 2147483648) % 1000)"


def col_expr(j):
    if j == 1:
        return f"CAST((({f_expr(0)} + ({f_expr(1)} % 20)) % 1000) AS INTEGER)"
    if j == 15:
        return f"CASE WHEN {f_expr(15)} % 10 = 0 THEN NULL ELSE CAST({f_expr(15)} AS INTEGER) END"
    return f"CAST({f_expr(j)} AS INTEGER)"


def build_sql(n):
    cols = ", ".join(f"{col_expr(j)} AS c{j}" for j in range(16))
    return f"CREATE TABLE nodes AS SELECT i AS id, {cols} FROM range({n}) t(i)"


def pred_sql(t):
    op = t["op"]
    if op == "cmp":
        return f"(c{t['col']} {SQL_OP[t['cmpop']]} {t['value']})"
    if op in ("and", "or"):
        return "(" + f" {op.upper()} ".join(pred_sql(a) for a in t["args"]) + ")"
    if op == "not":
        return f"(NOT {pred_sql(t['arg'])})"
    raise ValueError(f"bad op {op}")


def us(ns):
    return round(ns / 1000.0, 3)


def pct(s, p):
    return s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]


def qs(path):
    return path.replace("'", "''")


def engine_us(con, name, tmpdir):
    path = os.path.join(tmpdir, "profile.json")
    if os.path.exists(path):
        os.remove(path)
    con.execute("PRAGMA enable_profiling='json'")
    con.execute(f"PRAGMA profiling_output='{qs(path)}'")
    con.execute(f"EXECUTE {name}").fetchone()
    con.execute("PRAGMA disable_profiling")
    with open(path) as f:
        return round(json.load(f)["query"]["total_time"] * 1e6, 3)


def measure(con, name, sql, runs, tmpdir):
    t0 = time.perf_counter_ns()
    con.execute(f"PREPARE {name} AS {sql}")
    ans = con.execute(f"EXECUTE {name}").fetchone()[0]
    cold = us(time.perf_counter_ns() - t0)
    ns = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        a = con.execute(f"EXECUTE {name}").fetchone()[0]
        ns.append(time.perf_counter_ns() - t0)
        assert a == ans, "answer drift"
    s = sorted(ns)
    engs = [engine_us(con, name, tmpdir) for _ in range(5)]
    return {
        "answer": ans if isinstance(ans, bool) else int(ans),
        "cold_us": cold,
        "warm_med_us": us(statistics.median(s)),
        "warm_p10_us": us(pct(s, 0.10)),
        "warm_p90_us": us(pct(s, 0.90)),
        "engine_med_us": round(statistics.median(engs), 3),
    }


def explain_picks(cases):
    """One case per K group for EXPLAIN ANALYZE: prefer a regular medium AND."""
    picks = {}
    for c in cases:
        g = c["k_group"]
        pref = (not c["adversarial"]) and c["pop"] == "medium" and c["shape"] in ("and", "single")
        if g not in picks or (pref and not picks[g][1]):
            picks[g] = (c, pref)
    return [v[0] for v in picks.values()]


def run_dir(d, threads, runs, tmpdir):
    with open(os.path.join(d, "cases.json")) as f:
        spec = json.load(f)
    n = spec["n"]
    cases = spec["cases"]
    recs = []
    loads = {}
    explained = {c["id"] for c in explain_picks(cases)}
    for T in threads:
        con = duckdb.connect(":memory:")
        con.execute(f"SET threads={T}")
        t0 = time.perf_counter_ns()
        con.execute(build_sql(n))
        t1 = time.perf_counter_ns()
        con.execute("ANALYZE")
        t2 = time.perf_counter_ns()
        loads[str(T)] = {"build_us": us(t1 - t0), "analyze_us": us(t2 - t1), "load_us": us(t2 - t0)}
        con.execute("PREPARE triv AS SELECT 1")
        fl = []
        for _ in range(runs):
            t0 = time.perf_counter_ns()
            con.execute("EXECUTE triv").fetchone()
            fl.append(time.perf_counter_ns() - t0)
        triv = us(statistics.median(fl))
        print(f"{d} threads={T} built in {loads[str(T)]['build_us']:.0f} us, analyze {loads[str(T)]['analyze_us']:.0f} us", flush=True)
        for c in cases:
            pred = pred_sql(c["tree"])
            cnt = measure(con, "qc", f"SELECT count(*) FROM nodes WHERE {pred}", runs, tmpdir)
            ex = measure(con, "qe", f"SELECT EXISTS(SELECT 1 FROM nodes WHERE {pred})", runs, tmpdir)
            assert ex["answer"] == (cnt["answer"] > 0), f"exists/count disagree on {c['id']}"
            if T == 1 and c["id"] in explained:
                txt = con.execute(f"EXPLAIN ANALYZE SELECT count(*) FROM nodes WHERE {pred}").fetchall()[0][1]
                os.makedirs(os.path.join(d, "explain"), exist_ok=True)
                with open(os.path.join(d, "explain", f"{c['id']}.txt"), "w") as fh:
                    fh.write(f"-- {c['id']} (K group {c['k_group']})\n-- WHERE {pred}\n{txt}\n")
            recs.append({
                "id": c["id"], "threads": T, "count": cnt["answer"], "exists": ex["answer"],
                "count_cold_us": cnt["cold_us"], "count_warm_med_us": cnt["warm_med_us"],
                "count_warm_p10_us": cnt["warm_p10_us"], "count_warm_p90_us": cnt["warm_p90_us"],
                "count_engine_med_us": cnt["engine_med_us"],
                "exists_cold_us": ex["cold_us"], "exists_warm_med_us": ex["warm_med_us"],
                "exists_warm_p10_us": ex["warm_p10_us"], "exists_warm_p90_us": ex["warm_p90_us"],
                "exists_engine_med_us": ex["engine_med_us"],
                "trivial_query_wall_us": triv,
            })
            print(f"{d} threads={T} {c['id']} count={cnt['answer']} warm {cnt['warm_med_us']:.1f} us", flush=True)
        mem = con.execute("SELECT sum(memory_usage_bytes) FROM duckdb_memory()").fetchone()[0]
        loads[str(T)]["mem_bytes"] = int(mem) if mem is not None else None
        con.close()
    out = {"n": n, "duckdb_version": duckdb.__version__, "runs": runs, "load": loads, "records": recs}
    dst = os.path.join(d, "where_duckdb.json")
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {len(recs)} records to {dst}")


def main(argv):
    if len(argv) < 2 or argv[1].startswith("-"):
        print(__doc__)
        return 2
    dirs, threads, runs = [], [1, 4], 31
    i = 1
    while i < len(argv):
        if argv[i] == "--threads":
            threads = [int(x) for x in argv[i + 1].split(",")]
            i += 2
        elif argv[i] == "--runs":
            runs = int(argv[i + 1])
            i += 2
        elif argv[i].startswith("-"):
            print("unknown arg", argv[i])
            return 2
        else:
            dirs.append(argv[i])
            i += 1
    for d in dirs:
        if not os.path.exists(os.path.join(d, "cases.json")):
            print("missing", os.path.join(d, "cases.json"))
            return 2
    with tempfile.TemporaryDirectory() as tmpdir:
        for d in dirs:
            run_dir(d, threads, runs, tmpdir)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
