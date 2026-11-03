#!/usr/bin/env python3
"""D-GLOBAL-WHERE P2 Part 5: compare the Quack/mask-risc WHERE with DuckDB.

LAB SCRIPT. Run:  python3 -I compare_where.py <DIR> [<DIR> ...]

Each <DIR> is `<OUT>/where/n<N>` holding cases.json + where_ours.json (Rust
`where_k`) + where_duckdb.json (bench_where.py). Exit 1 if any DuckDB answer
differs from the oracle count in cases.json (count, and EXISTS == count > 0);
nothing is compared until every answer matches. Writes <DIR>/compare_where.md
and prints it.

Ours per case: lower_us (plain / fused) and, per terminal, the best of the four
orderings {plain, fused, and_by_skip, and_by_skip + fused} by warm median.
`virtual` = Count/Any (no mask survives), `transient` = Rows (a Keep mask held
for the call). DuckDB: warm wall median, engine time (profiling
query.total_time), and the EXISTS variant. Ratio = DuckDB / ours (>1 means
ours is faster). Verdict: WIN > 1.10, LOSS < 0.90, else PARITY. Count is
compared with `SELECT count(*)`, Any with `SELECT EXISTS(...)`; Rows has no
DuckDB counterpart here (DuckDB returns no mask) and is reported alone.

Summaries cover the regular grid only (K group x population x shape); the
adversarial and nullable cases are listed and verdicted but excluded from the
weighted figure, which is the mean per K group weighted 34/46/14/6 over the
groups 1, 2-4, 5-16, 17+. The unweighted figure is the plain mean over the
regular cases. Group means are of microseconds, then the ratio of those means;
the geometric mean of per-case ratios is shown beside it.
"""
import json

EXPECTED_THREADS = [1, 4]
import math
import os
import sys

GROUPS = ["1", "2-4", "5-16", "17+"]
WEIGHTS = {"1": 0.34, "2-4": 0.46, "5-16": 0.14, "17+": 0.06}
TERMS = ("count", "any", "rows")


def verdict(r):
    if r is None:
        return "n/a"
    if r > 1.10:
        return "WIN"
    if r < 0.90:
        return "LOSS"
    return "PARITY"


def ratio(duck, ours):
    if duck is None or ours is None or ours <= 0:
        return None
    return duck / ours


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def pop_class(c, n):
    a = c["achieved"]
    if not c["adversarial"]:
        return c["pop"]
    if a == 0:
        return "empty"
    if a <= 100:
        return "tiny"
    return "medium" if a <= n // 20 else "large"


def best(rec, term):
    vals = []
    for v in (rec.get("variants") or {}).values():
        if v:
            vals.append(v[f"{term}_us"])
    return min(vals) if vals else None


def best_tag(rec, term):
    bt, bv = None, None
    for tag, v in (rec.get("variants") or {}).items():
        if v and (bv is None or v[f"{term}_us"] < bv):
            bt, bv = tag, v[f"{term}_us"]
    return bt


def gmean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def load(d):
    with open(os.path.join(d, "cases.json")) as f:
        cases = json.load(f)
    with open(os.path.join(d, "where_ours.json")) as f:
        ours = json.load(f)
    with open(os.path.join(d, "where_duckdb.json")) as f:
        duck = json.load(f)
    return cases, ours, duck


def check(d, cases, duck):
    bad = []
    byid = {c["id"]: c for c in cases["cases"]}
    for r in duck["records"]:
        c = byid.get(r["id"])
        if c is None:
            bad.append(f"{d}: duckdb record for unknown case {r['id']}")
            continue
        if r["count"] != c["achieved"]:
            bad.append(f"{d} {r['id']} threads={r['threads']}: duckdb count {r['count']} != oracle {c['achieved']}")
        if r["exists"] != (c["achieved"] > 0):
            bad.append(f"{d} {r['id']} threads={r['threads']}: duckdb exists {r['exists']} != oracle {c['achieved'] > 0}")
    seen = {(r["id"], r["threads"]) for r in duck["records"]}
    threads = sorted({r["threads"] for r in duck["records"]})
    # Coverage is enforced in both directions (the corrected P0 parity rule):
    # an incomplete run must fail, never print an all-agree verdict.
    if threads != EXPECTED_THREADS:
        bad.append(f"{d}: duckdb thread counts {threads}, expected {EXPECTED_THREADS}")
    for c in cases["cases"]:
        for t in EXPECTED_THREADS:
            if (c["id"], t) not in seen:
                bad.append(f"{d}: missing duckdb record {c['id']} threads={t}")
    return bad


def check_ours(d, cases, ours):
    bad = []
    oid = {r["id"]: r for r in ours["records"]}
    for c in cases["cases"]:
        if c["id"] not in oid:
            bad.append(f"{d}: missing ours record {c['id']}")
    for i in oid:
        if i not in {c["id"] for c in cases["cases"]}:
            bad.append(f"{d}: ours record for unknown case {i}")
    return bad


def table(n, cases, ours, duck, T):
    oid = {r["id"]: r for r in ours["records"]}
    did = {r["id"]: r for r in duck["records"] if r["threads"] == T}
    lines = []
    lines.append(f"### N = {n:,}, DuckDB threads = {T}")
    lines.append("")
    lines.append(
        "| K group | case | pop | shape | achieved | lower plain/fused us | ours count us (order) | ours any us | ours rows us | "
        "duck count wall us | duck count engine us | duck exists wall us | count wall ratio | count wall | count engine ratio | count engine | any ratio | any |"
    )
    lines.append("|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---|")
    rows = []
    for c in cases["cases"]:
        o = oid[c["id"]]
        dk = did[c["id"]]
        oc, oa, orr = best(o, "count"), best(o, "any"), best(o, "rows")
        r_cw = ratio(dk["count_warm_med_us"], oc)
        r_ce = ratio(dk["count_engine_med_us"], oc)
        r_a = ratio(dk["exists_warm_med_us"], oa)
        rows.append((c, o, dk, oc, oa, orr, r_cw, r_ce, r_a))
        lines.append(
            f"| {c['k_group']} | {c['id']} | {pop_class(c, n)} | {c['shape']} | {c['achieved']:,} | "
            f"{fmt(o.get('lower_us_plain'), 2)}/{fmt(o.get('lower_us_fused'), 2)} | {fmt(oc)} ({best_tag(o, 'count')}) | "
            f"{fmt(oa)} | {fmt(orr)} | {fmt(dk['count_warm_med_us'])} | {fmt(dk['count_engine_med_us'])} | "
            f"{fmt(dk['exists_warm_med_us'])} | {fmt(r_cw, 2)} | {verdict(r_cw)} | {fmt(r_ce, 2)} | {verdict(r_ce)} | "
            f"{fmt(r_a, 2)} | {verdict(r_a)} |"
        )
    lines.append("")
    lines.append("Representation: Count and Any are `virtual` (no mask survives the call); Rows is `transient` "
                 f"(one Keep mask of {(n + 63) // 64 * 8:,} bytes held for the call); `persistent` never occurs here. "
                 "Ours rows has no DuckDB counterpart.")
    lines.append("")
    lines += summaries(n, rows, T)
    return lines


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summaries(n, rows, T):
    reg = [r for r in rows if not r[0]["adversarial"]]
    lines = [f"Summary, N = {n:,}, threads = {T} (regular grid only: {len(reg)} cases; "
             f"{len(rows) - len(reg)} adversarial/nullable cases excluded from the figures below)", ""]
    lines.append("| comparison | weighted ours us | weighted duck us | weighted ratio | weighted | unweighted ours us | unweighted duck us | unweighted ratio | unweighted | geo-mean ratio |")
    lines.append("|---|---:|---:|---:|---|---:|---:|---:|---|---:|")
    specs = [
        ("count vs duck count, wall", 3, lambda r: r[2]["count_warm_med_us"]),
        ("count vs duck count, engine", 3, lambda r: r[2]["count_engine_med_us"]),
        ("any vs duck exists, wall", 4, lambda r: r[2]["exists_warm_med_us"]),
    ]
    for name, oi, dget in specs:
        ours_g, duck_g = {}, {}
        for g in GROUPS:
            rs = [r for r in reg if r[0]["k_group"] == g]
            ours_g[g] = mean([r[oi] for r in rs])
            duck_g[g] = mean([dget(r) for r in rs])
        wo = sum(WEIGHTS[g] * ours_g[g] for g in GROUPS if ours_g[g] is not None)
        wd = sum(WEIGHTS[g] * duck_g[g] for g in GROUPS if duck_g[g] is not None)
        wsum = sum(WEIGHTS[g] for g in GROUPS if ours_g[g] is not None)
        wo, wd = (wo / wsum, wd / wsum) if wsum else (None, None)
        uo = mean([r[oi] for r in reg])
        ud = mean([dget(r) for r in reg])
        rw, ru = ratio(wd, wo), ratio(ud, uo)
        gm = gmean([ratio(dget(r), r[oi]) for r in reg])
        lines.append(
            f"| {name} | {fmt(wo)} | {fmt(wd)} | {fmt(rw, 2)} | {verdict(rw)} | {fmt(uo)} | {fmt(ud)} | {fmt(ru, 2)} | {verdict(ru)} | {fmt(gm, 2)} |"
        )
    lines.append("")
    lines.append("Per K group (count vs duck count, wall; mean us over regular cases):")
    lines.append("")
    lines.append("| K group | cases | ours us | duck us | ratio | verdict | weight |")
    lines.append("|---|---:|---:|---:|---:|---|---:|")
    for g in GROUPS:
        rs = [r for r in reg if r[0]["k_group"] == g]
        o = mean([r[3] for r in rs])
        dk = mean([r[2]["count_warm_med_us"] for r in rs])
        rr = ratio(dk, o)
        lines.append(f"| {g} | {len(rs)} | {fmt(o)} | {fmt(dk)} | {fmt(rr, 2)} | {verdict(rr)} | {WEIGHTS[g]:.2f} |")
    lines.append("")
    gate = [r for r in rows if r[3] is not None and r[3] >= 1000.0]
    lines.append(f"Ours count >= 1000 us (FAILED-GATE on the substrate path, reported): {len(gate)} of {len(rows)} cases.")
    lines.append("")
    return lines


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    dirs = argv[1:]
    loaded = {}
    problems = []
    for d in dirs:
        cases, ours, duck = load(d)
        loaded[d] = (cases, ours, duck)
        problems += check(d, cases, duck)
        problems += check_ours(d, cases, ours)
    if problems:
        print("ANSWER MISMATCH between DuckDB and the oracle:")
        for p in problems:
            print("  " + p)
        return 1
    for d in dirs:
        cases, ours, duck = loaded[d]
        n = cases["n"]
        lines = [f"# D-GLOBAL-WHERE P2 Part 5: criteria-aware WHERE, N = {n:,}", ""]
        lines.append("Every DuckDB count and EXISTS answer equals the scalar oracle in cases.json "
                     "(checked before this file was written).")
        lines.append("")
        miss = [c for c in cases["cases"] if not c["target_met"]]
        if miss:
            lines.append("Population targets NOT met by the generator (achieved is what is measured; "
                         "never assumed from K): " + ", ".join(f"{c['id']} (target {c['target']:,}, achieved {c['achieved']:,})" for c in miss))
            lines.append("")
        lines.append("DuckDB build + ANALYZE: " + "; ".join(
            f"threads {t}: {v['load_us']:,.0f} us (build {v['build_us']:,.0f}, analyze {v['analyze_us']:,.0f})"
            for t, v in sorted(duck["load"].items(), key=lambda kv: int(kv[0]))))
        lines.append("")
        for T in sorted({r["threads"] for r in duck["records"]}):
            lines += table(n, cases, ours, duck, T)
        text = "\n".join(lines) + "\n"
        with open(os.path.join(d, "compare_where.md"), "w") as f:
            f.write(text)
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
