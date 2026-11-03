#!/usr/bin/env python3
"""D-GLOBAL-WHERE P1 probe: join ours_big.json with duckdb_big.json per corpus size.

LAB SCRIPT. Not production code.

Run:  python3 -I compare_big.py <DIR> [<DIR> ...]    e.g. $OUT/big_1m $OUT/big_10m

Each <DIR> holds ours_big.json and duckdb_big.json; <DIR>/compare_big.md is
written for each, and the scaling table (all dirs together) goes to
<first DIR>/../compare_big_scaling.md. Dirs are ordered by the size in their name
(big_1m < big_10m); the label is the part after the last `big_`.

Per dir and DuckDB thread count a table with one row per ours record x DuckDB
arm (plain / indexed): query, ours mode, DuckDB arm, ours warm med, ours prep,
ours cold, DuckDB warm wall, DuckDB engine, trivial (SELECT 1 wall, not
subtracted), DuckDB cold, DuckDB load, DuckDB index build, ratio wall (DuckDB /
ours), ratio engine, verdicts (WIN > 1.10, PARITY 0.90..1.10, LOSS < 0.90) and
break-even Q = ours prep / (DuckDB warm - ours warm), "never" when ours is
slower. A second table per dir shows ours' stage fields (discover_scan,
discover_index, index_build, expand, bind, project, fold, continuation; null
shown as "-"). The scaling table gives, per query, DuckDB warm per arm/threads
at every N and ours warm plus stage totals (discover = scan+index; working-set
= expand+bind+project+fold) at every N with the growth ratio last/first.

Exit 1 if any answer_per_t of any ours record differs from DuckDB (any arm, any
thread count); after_delete_per_t is compared when ours carries it. Exit 2 on
unreadable input. Microseconds everywhere; ours warm >= 1000 us prints
FAILED-GATE, cold/prep >= 1000 us prints COLD (reported, not gate).
"""
import json
import os
import re
import sys

STAGES = ["discover_scan_us", "discover_index_us", "index_build_us", "expand_us",
          "bind_us", "project_us", "fold_us", "continuation_us"]


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        print(f"cannot read {path}: {e}", file=sys.stderr)
        sys.exit(2)


def verdict(r):
    if r is None:
        return "n/a"
    return "WIN" if r > 1.10 else ("PARITY" if r >= 0.90 else "LOSS")


def ratio(n, d):
    if n is None or d is None or d <= 0:
        return None
    return n / d


def f1(v, kind=None):
    if v is None:
        return "-"
    s = f"{v:.1f}"
    if kind == "warm" and v >= 1000:
        s += " FAILED-GATE"
    elif kind == "cold" and v >= 1000:
        s += " COLD (reported, not gate)"
    return s


def fr(r):
    return "-" if r is None else f"{r:.2f}"


def breakeven(prep, duck, ours):
    if duck is None or ours is None or duck <= ours:
        return "never"
    if not prep:
        return "0"
    return f"{prep / (duck - ours):.1f}"


def qorder(q):
    return (len(q), q)


def size_key(path):
    name = os.path.basename(os.path.normpath(path))
    m = re.search(r"(\d+)([kmb]?)$", name.lower())
    if not m:
        return (0, name)
    return (int(m.group(1)) * {"": 1, "k": 10**3, "m": 10**6, "b": 10**9}[m.group(2)], name)


def label(path):
    return os.path.basename(os.path.normpath(path)).split("big_")[-1]


def ssum(o, keys):
    vals = [o.get(k) for k in keys]
    if all(v is None for v in vals):
        return None
    return sum(v for v in vals if v is not None)


def compare_dir(d, mism, notes):
    ours = load(os.path.join(d, "ours_big.json"))
    dd = load(os.path.join(d, "duckdb_big.json"))
    lab = label(d)
    dd_by = {(r["query"], r.get("arm", "plain"), r["threads"]): r for r in dd}
    arms = sorted({r.get("arm", "plain") for r in dd}, reverse=True)
    threads = sorted({r["threads"] for r in dd})
    rows = sorted(ours, key=lambda r: (qorder(r["query"]), r["mode"]))
    lines = [f"# D-GLOBAL-WHERE P1: corpus {lab}, ours vs DuckDB", "",
             "Microseconds throughout. Ratio = DuckDB / ours (> 1: ours faster). "
             "WIN > 1.10, PARITY 0.90..1.10, LOSS < 0.90. Break-even Q = ours prep / "
             "(DuckDB warm wall - ours warm); `never` when ours is slower. "
             "`trivial` = DuckDB prepared `SELECT 1` wall (not subtracted). "
             "DuckDB `idx build` is charged preprocessing of the indexed arm.", ""]
    hdr = ("| query | ours mode | DuckDB arm | ours warm med | ours prep | ours cold | DuckDB warm wall "
           "| DuckDB engine | trivial | DuckDB cold | DuckDB load | DuckDB idx build | ratio wall "
           "| verdict wall | ratio engine | verdict engine | break-even Q |")
    sep = "|" + "---|" * 17
    for T in threads:
        lines += [f"## {lab}: DuckDB threads={T}", "", hdr, sep]
        for o in rows:
            for arm in arms:
                dr = dd_by.get((o["query"], arm, T))
                if dr is None:
                    notes.append(f"{lab}: no duckdb record for {o['query']} arm={arm} threads={T}")
                    continue
                if o.get("answer_per_t") != dr["answer_per_t"]:
                    mism.append((lab, o["query"], o["mode"], arm, T, "answer_per_t", o.get("answer_per_t"), dr["answer_per_t"]))
                if "after_delete_per_t" in o:
                    if o["after_delete_per_t"] != dr["after_delete_per_t"]:
                        mism.append((lab, o["query"], o["mode"], arm, T, "after_delete_per_t", o["after_delete_per_t"], dr["after_delete_per_t"]))
                else:
                    notes.append(f"{lab}: ours lacks after_delete_per_t for {o['query']} {o['mode']}: not compared")
                ow = o.get("warm_med_us")
                rw = ratio(dr["warm_med_us"], ow)
                re_ = ratio(dr.get("engine_med_us"), ow)
                lines.append(
                    f"| {o['query']} | {o['mode']} | {arm} | {f1(ow, 'warm')} | {f1(o.get('prep_us'), 'cold')} "
                    f"| {f1(o.get('cold_us'), 'cold')} | {f1(dr['warm_med_us'])} | {f1(dr.get('engine_med_us'))} "
                    f"| {f1(dr.get('trivial_query_wall_us'))} | {f1(dr.get('cold_us'))} | {f1(dr.get('load_us'))} "
                    f"| {f1(dr.get('index_build_us'))} | {fr(rw)} | {verdict(rw)} | {fr(re_)} | {verdict(re_)} "
                    f"| {breakeven(o.get('prep_us'), dr['warm_med_us'], ow)} |")
        lines.append("")
    lines += [f"## {lab}: ours stages (null = not applicable)", "",
              "| query | mode | " + " | ".join(s[:-3] for s in STAGES) + " | prep | warm med |",
              "|" + "---|" * (len(STAGES) + 4)]
    for o in rows:
        lines.append(f"| {o['query']} | {o['mode']} | " + " | ".join(f1(o.get(s), 'cold') for s in STAGES)
                     + f" | {f1(o.get('prep_us'), 'cold')} | {f1(o.get('warm_med_us'), 'warm')} |")
    lines.append("")
    return ours, dd, lines


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    dirs = sorted(argv[1:], key=size_key)
    mism, notes, data, out_all = [], [], {}, []
    for d in dirs:
        ours, dd, lines = compare_dir(d, mism, notes)
        data[d] = (ours, dd)
        with open(os.path.join(d, "compare_big.md"), "w") as f:
            f.write("\n".join(lines))
        out_all += lines

    sc = ["# Corpus scaling", "",
          "Per query: DuckDB warm wall per arm/threads and ours (mode `paged` stage totals, `direct` warm) at each N; "
          "`discover` = scan + index, `ws` = expand + bind + project + fold. `x` = last N / first N "
          "(corpus-size independence holds for `ws` when x is about 1).", ""]
    labs = [label(d) for d in dirs]
    queries = sorted({o["query"] for d in dirs for o in data[d][0]}, key=qorder)
    # DuckDB scaling
    sc += ["## DuckDB warm wall (us)", "",
           "| query | arm | threads | " + " | ".join(labs) + " | x |", "|" + "---|" * (len(labs) + 4)]
    keys = sorted({(r.get("arm", "plain"), r["threads"]) for d in dirs for r in data[d][1]}, key=lambda k: (k[0] != "plain", k[1]))
    for q in queries:
        for arm, T in keys:
            vals = []
            for d in dirs:
                r = next((r for r in data[d][1] if r["query"] == q and r.get("arm", "plain") == arm and r["threads"] == T), None)
                vals.append(r["warm_med_us"] if r else None)
            sc.append(f"| {q} | {arm} | {T} | " + " | ".join(f1(v) for v in vals) + f" | {fr(ratio(vals[-1], vals[0]))} |")
    sc.append("")
    sc += ["## Ours (us)", "",
           "| query | mode | metric | " + " | ".join(labs) + " | x |", "|" + "---|" * (len(labs) + 4)]
    metrics = [("warm", lambda o: o.get("warm_med_us")),
               ("prep", lambda o: o.get("prep_us")),
               ("discover", lambda o: ssum(o, ["discover_scan_us", "discover_index_us"])),
               ("index_build", lambda o: o.get("index_build_us")),
               ("ws (expand+bind+project+fold)", lambda o: ssum(o, ["expand_us", "bind_us", "project_us", "fold_us"])),
               ("continuation", lambda o: o.get("continuation_us"))]
    for q in queries:
        modes = sorted({o["mode"] for d in dirs for o in data[d][0] if o["query"] == q})
        for m in modes:
            for name, fn in metrics:
                vals = []
                for d in dirs:
                    o = next((o for o in data[d][0] if o["query"] == q and o["mode"] == m), None)
                    vals.append(fn(o) if o else None)
                if all(v is None for v in vals):
                    continue
                sc.append(f"| {q} | {m} | {name} | " + " | ".join(f1(v) for v in vals) + f" | {fr(ratio(vals[-1], vals[0]))} |")
    sc.append("")
    tail = []
    if notes:
        tail += ["## Notes", ""] + [f"- {n}" for n in sorted(set(notes))] + [""]
    if mism:
        tail += ["## ANSWER MISMATCHES", "", "| corpus | query | mode | duckdb arm | threads | field | ours | duckdb |", "|---|---|---|---|---|---|---|---|"]
        tail += ["| " + " | ".join(str(x) for x in m) + " |" for m in mism] + [""]
    else:
        tail += ["All compared answers agree.", ""]
    text = "\n".join(out_all + sc + tail)
    with open(os.path.join(os.path.dirname(os.path.normpath(dirs[0])) or ".", "compare_big_scaling.md"), "w") as f:
        f.write("\n".join(sc + tail))
    print(text)
    if mism:
        print(f"\nFAIL: {len(mism)} answer mismatch(es)", file=sys.stderr)
        for m in mism:
            print("MISMATCH corpus=%s query=%s mode=%s arm=%s threads=%s field=%s ours=%s duckdb=%s" % m, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
