#!/usr/bin/env python3
"""D-GLOBAL-WHERE P0 probe: join ours.json with duckdb.json (Part 4 of SPEC.md).

LAB SCRIPT. Not production code.

Run:  python3 -I compare.py <OUT_DIR>      (reads ours.json + duckdb.json,
                                            writes compare.md, prints it)

Exit status 1 if any answer disagrees (every ours arm vs DuckDB at both thread
counts) or any comparison is missing in either direction (including a thread
count other than 1 and 4). DuckDB's W8 (W2 before the DELETE) is checked
against ours W2. Exit 2 on unreadable input.

All times are microseconds. Ratios are DuckDB / ours, so > 1 means ours is
faster. Verdict thresholds on the ratio: WIN > 1.10, PARITY 0.90..1.10,
LOSS < 0.90.
  * wall verdict   : (duckdb warm - python floor) / ours warm median
  * engine verdict : duckdb engine latency (profile query.total_time) / ours warm median
  * cold verdict   : duckdb cold (PREPARE + first EXECUTE) / ours cold. DuckDB's
    load_us (CSV load + ANALYZE) is reported separately in its own column and
    is NOT part of the cold comparison; ours cold is prep + WHERE + fold.
Ours warm values >= 1000 us carry FAILED-GATE (workspace rule: no ms results).
Workload names are identical in both files (W1 W2 W3 W4 W5a W6 W7 W7i W8 W8_after).
"""
import json
import os
import sys
from collections import defaultdict

EXPECTED_THREADS = [1, 4]


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        print(f"cannot read {path}: {e}", file=sys.stderr)
        sys.exit(2)


def verdict(ratio):
    if ratio is None:
        return "n/a"
    if ratio > 1.10:
        return "WIN"
    if ratio >= 0.90:
        return "PARITY"
    return "LOSS"


def ratio(num, den):
    if num is None or den is None or den <= 0:
        return None
    return num / den


def fmt(v, gate=False):
    if v is None:
        return "-"
    s = f"{v:.1f}"
    if gate and v >= 1000:
        s += " FAILED-GATE"
    return s


def fr(r):
    return "-" if r is None else f"{r:.2f}"


def wl_order(w):
    order = ["W1", "W2", "W3", "W4", "W5a", "W6", "W7", "W7i", "W8", "W8_after"]
    return order.index(w) if w in order else len(order)


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    out = argv[1]
    ours = load(os.path.join(out, "ours.json"))
    dd = load(os.path.join(out, "duckdb.json"))

    ours_by = defaultdict(list)          # (dataset, workload) -> [arm records]
    for r in ours:
        ours_by[(r["dataset"], r["workload"])].append(r)
    dd_by = {(r["dataset"], r["threads"], r["workload"]): r for r in dd}
    threads = sorted({r["threads"] for r in dd})

    mismatches = []
    notes = []
    rows = defaultdict(list)             # threads -> table rows
    summary = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    # summary[(threads, kind)][workload][verdict]

    # Coverage is checked in both directions and any gap fails the gate:
    # an incomplete run must never print "All compared answers agree".
    missing = []
    if threads != EXPECTED_THREADS:
        missing.append(f"duckdb thread counts {threads}, expected {EXPECTED_THREADS}")
    ours_keys = set(ours_by)
    dd_keys = {(r["dataset"], r["workload"]) for r in dd}
    # DuckDB's W8 is W2 measured on the same connection before the DELETE; ours
    # has no separate record for it, so its answer is checked against ours W2.
    for ds, wl in sorted(ours_keys - dd_keys):
        missing.append(f"no duckdb record for {ds} {wl}")
    for ds, wl in sorted(dd_keys):
        src = (ds, "W2") if wl == "W8" else (ds, wl)
        if src not in ours_by:
            missing.append(f"no ours record for {ds} {wl}")
        for T in EXPECTED_THREADS:
            if (ds, T, wl) not in dd_by:
                missing.append(f"no duckdb record for {ds} threads={T} {wl}")
    for ds, wl in sorted(dd_keys):
        if wl != "W8" or (ds, "W2") not in ours_by:
            continue
        for T in threads:
            d = dd_by.get((ds, T, wl))
            if d is None:
                continue
            for a in ours_by[(ds, "W2")]:
                if a.get("answer") is None:
                    missing.append(f"ours answer missing: {ds} W2 arm={a.get('arm')}")
                elif a["answer"] != d["answer"]:
                    mismatches.append((ds, "W8(before delete) vs ours W2", a.get("arm"), T, a["answer"], d["answer"]))

    keys = sorted((k for k in dd_keys if k[1] != "W8"),
                  key=lambda k: (k[0], wl_order(k[1])))
    for ds, wl in keys:
        arms = ours_by.get((ds, wl))
        if not arms:
            continue
        for T in threads:
            d = dd_by.get((ds, T, wl))
            if d is None:
                continue
            for a in arms:
                if a.get("answer") is None:
                    missing.append(f"ours answer missing: {ds} {wl} arm={a.get('arm')}")
                elif a["answer"] != d["answer"]:
                    mismatches.append((ds, wl, a.get("arm"), T, a["answer"], d["answer"]))
            floor = d.get("floor_us") or 0.0
            net = max(d["warm_med_us"] - floor, 0.0)
            best = min(arms, key=lambda a: a["warm_med_us"])
            candidates = [(a.get("arm"), a) for a in arms]
            if len(arms) > 1:
                candidates.append(("best:" + str(best.get("arm")), best))
            for label, a in candidates:
                o = a["warm_med_us"]
                r_wall = ratio(net, o)
                r_eng = ratio(d.get("engine_us"), o)
                r_cold = ratio(d.get("cold_us"), a.get("cold_us"))
                rows[T].append((ds, wl, label, a, d, net, r_wall, r_eng, r_cold))
                if label == candidates[-1][0]:
                    # summary uses the best arm (the only arm if single)
                    summary[(T, "wall")][wl][verdict(r_wall)] += 1
                    summary[(T, "engine")][wl][verdict(r_eng)] += 1
                    summary[(T, "cold")][wl][verdict(r_cold)] += 1

    lines = ["# D-GLOBAL-WHERE: ours vs DuckDB", "",
             "Microseconds throughout. Ratio = DuckDB / ours (> 1 means ours faster). "
             "WIN > 1.10, PARITY 0.90..1.10, LOSS < 0.90. "
             "`ours warm` >= 1000 us is flagged FAILED-GATE. "
             "DuckDB `load` (CSV load + ANALYZE) is reported separately and is not part of the cold ratio.", ""]
    hdr = ("| dataset | workload | ours arm | ours warm med | ours cold | DuckDB warm | DuckDB warm-floor "
           "| DuckDB engine | DuckDB cold | DuckDB load | ratio wall | verdict wall "
           "| ratio engine | verdict engine | ratio cold | verdict cold |")
    sep = "|" + "---|" * 16
    for T in threads:
        lines += [f"## DuckDB threads={T}", "", hdr, sep]
        for ds, wl, label, a, d, net, rw, re_, rc in rows[T]:
            lines.append(
                f"| {ds} | {wl} | {label} | {fmt(a['warm_med_us'], True)} | {fmt(a.get('cold_us'))} "
                f"| {fmt(d['warm_med_us'])} | {fmt(net)} | {fmt(d.get('engine_us'))} "
                f"| {fmt(d.get('cold_us'))} | {fmt(d.get('load_us'))} "
                f"| {fr(rw)} | {verdict(rw)} | {fr(re_)} | {verdict(re_)} | {fr(rc)} | {verdict(rc)} |")
        lines.append("")

    lines += ["## Summary: verdict counts across datasets (best arm per workload)", ""]
    for T in threads:
        lines += [f"### threads={T}", "", "| workload | wall W/P/L | engine W/P/L | cold W/P/L |", "|---|---|---|---|"]
        wls = sorted({w for k in summary if k[0] == T for w in summary[k]}, key=wl_order)
        for w in wls:
            cells = []
            for kind in ("wall", "engine", "cold"):
                c = summary[(T, kind)][w]
                cells.append(f"{c['WIN']}/{c['PARITY']}/{c['LOSS']}")
            lines.append(f"| {w} | " + " | ".join(cells) + " |")
        lines.append("")

    if notes:
        lines += ["## Notes", ""] + [f"- {n}" for n in notes] + [""]
    if missing:
        lines += ["## MISSING COMPARISONS", ""] + [f"- {m}" for m in missing] + [""]
    if mismatches:
        lines += ["## ANSWER MISMATCHES", "", "| dataset | workload | ours arm | duckdb threads | ours answer | duckdb answer |", "|---|---|---|---|---|---|"]
        for m in mismatches:
            lines.append("| " + " | ".join(str(x) for x in m) + " |")
        lines.append("")
    elif not missing:
        lines += ["All compared answers agree; coverage complete in both directions.", ""]

    text = "\n".join(lines)
    with open(os.path.join(out, "compare.md"), "w") as f:
        f.write(text)
    print(text)
    if missing:
        print(f"\nFAIL: {len(missing)} missing comparison(s)", file=sys.stderr)
        for m in missing:
            print("MISSING " + m, file=sys.stderr)
    if mismatches:
        print(f"\nFAIL: {len(mismatches)} answer mismatch(es)", file=sys.stderr)
        for m in mismatches:
            print("MISMATCH dataset=%s workload=%s arm=%s duckdb_threads=%s ours=%s duckdb=%s" % m, file=sys.stderr)
        return 1
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
