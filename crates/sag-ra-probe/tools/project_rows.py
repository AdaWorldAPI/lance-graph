#!/usr/bin/env python3
"""Merge one rust-analyzer DUMP and one Charon LLBC into typed relation rows
(D-SAG-6-2 lab projection).

    python3 -I project_rows.py <ra_dump.txt> <llbc_calls.txt>

where <llbc_calls.txt> is the output of llbc_calls.py. One row per fact:

    relation  subject  object  witness  evidence

Evidence classes, never mixed:
  WITNESSED   a compiler frontend reports it (resolved call, write, assert)
  POSSIBLE    dispatch whose callee the frontend does not fix
  UNJOINED    a fact one frontend has and the other cannot be joined to by
              path: the two tools name some items differently (impls,
              closures), so an absent join is UNKNOWN, never "no such item"

Nothing here is an identity: subjects are each tool's own path spelling.
"""
import ast
import sys


def ra_rows(path):
    rows, monikers = [], set()
    for line in open(path):
        if not line.startswith("sym\t"):
            continue
        cols = line.rstrip("\n").split("\t")
        f = dict(c.split("=", 1) for c in cols[2:] if "=" in c)
        m = f.get("moniker", "none")
        if m in ("none", "local") or m.split("::")[0] in ("core", "alloc", "std"):
            continue
        monikers.add(m)
        rows.append(("DEFINES", m, f["kind"], "rust-analyzer", "WITNESSED"))
        if f.get("impls", "-") not in ("-", "0"):
            rows.append(("HAS_IMPLS", m, f["impls"], "rust-analyzer", "WITNESSED"))
    return rows, monikers


def charon_rows(path, monikers):
    rows = []
    for line in open(path):
        if "\tcalls=" not in line:
            continue
        name, rest = line.rstrip("\n").split("\t", 1)
        f = dict(c.split("=", 1) for c in rest.split("\t"))
        joined = name in monikers or ("::Impl::" not in name)
        for call in ast.literal_eval(f["calls"]):
            kind, _, tgt = call.partition(":")
            if kind == "static":
                rows.append(("CALLS", name, tgt, "charon", "WITNESSED"))
            elif kind == "trait" and tgt == "TraitImpl":
                rows.append(("CALLS", name, "known impl", "charon", "WITNESSED"))
            else:
                rows.append(("MAY_DISPATCH", name, call, "charon", "POSSIBLE"))
        if int(f["deref_write_arg"]):
            rows.append(("WRITES_CALLER_MEMORY", name, f["deref_write_arg"], "charon", "WITNESSED"))
        if int(f["deref_write_local"]):
            # Local-rooted: may alias caller memory through a returned
            # reference; not classified further here.
            rows.append(("WRITES_THROUGH_LOCAL_REF", name, f["deref_write_local"], "charon", "POSSIBLE"))
        if int(f["assert"]):
            rows.append(("MAY_PANIC", name, f["assert"] + " assert", "charon", "WITNESSED"))
        if not joined or name not in monikers:
            rows.append(("UNJOINED", name, "no rust-analyzer moniker with this path", "join", "UNKNOWN"))
    return rows


def main(ra_path, charon_path):
    rows, monikers = ra_rows(ra_path)
    rows += charon_rows(charon_path, monikers)
    print("relation\tsubject\tobject\twitness\tevidence")
    for r in rows:
        print("\t".join(r))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
