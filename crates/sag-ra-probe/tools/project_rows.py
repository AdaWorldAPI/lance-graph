#!/usr/bin/env python3
"""Merge one rust-analyzer DUMP and one Charon LLBC into typed relation rows
(D-SAG-6-2 lab projection).

    python3 -I project_rows.py <ra_dump.txt> <llbc_calls.txt> [<join.tsv>]

where <llbc_calls.txt> is the output of llbc_calls.py and <join.tsv> the
output of span_join.py. With a join, subjects and local callees are
rust-analyzer monikers, joined by source position; without one, they are
Charon paths, joined by spelling (the D-SAG-6-2 baseline). One row per fact:

    relation  subject  object  witness  evidence

Evidence classes, never mixed:
  WITNESSED   a compiler frontend reports it (resolved call, write, assert)
  POSSIBLE    dispatch whose callee the frontend does not fix
  UNJOINED    a fact one frontend has and the other cannot be joined to:
              UNKNOWN, never "no such item"

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


def load_join(path):
    join = {}
    for line in open(path):
        cols = line.rstrip("\n").split("\t")
        if cols[0] == "join":
            join[cols[1]] = (cols[3], cols[4], cols[5])
    return join


def subject_of(fid, name, join, monikers):
    """The subject for a Charon function, and whether it joined."""
    if join is None:
        return name, name in monikers
    cls, mon, where = join.get(fid, ("UNJOINED", "-", ""))
    if cls == "EXACT":
        return mon, True
    if cls == "ENCLOSED":
        # Keep the closure's line and Charon's method (`call_mut`, or the
        # `call_once` shim): two closures in one function, or a shim and its
        # body, must not collapse into one subject.
        line = where.split(":")[-1].split("-")[0]
        return f"{mon}::{{closure@L{line}}}::{name.rsplit('::', 1)[-1]}", True
    if cls == "DERIVED":
        return f"{mon}::<derive {where.split(' at ')[0]}>", True
    return name, False


def charon_rows(path, monikers, join=None):
    rows = []
    for line in open(path):
        if "\tcalls=" not in line:
            continue
        name, rest = line.rstrip("\n").split("\t", 1)
        f = dict(c.split("=", 1) for c in rest.split("\t"))
        subj, joined = subject_of(f.get("id"), name, join, monikers)
        for call in ast.literal_eval(f["calls"]):
            kind, _, tgt = call.partition(":")
            if kind == "static":
                tgt, _, tid = tgt.partition("@")
                if join is not None and tid in join:
                    tgt = subject_of(tid, tgt, join, monikers)[0]
                rows.append(("CALLS", subj, tgt, "charon", "WITNESSED"))
            elif kind == "trait" and tgt == "TraitImpl":
                rows.append(("CALLS", subj, "known impl", "charon", "WITNESSED"))
            else:
                rows.append(("MAY_DISPATCH", subj, call, "charon", "POSSIBLE"))
        if int(f["deref_write_arg"]):
            rows.append(("WRITES_CALLER_MEMORY", subj, f["deref_write_arg"], "charon", "WITNESSED"))
        if int(f["deref_write_local"]):
            # Local-rooted: may alias caller memory through a returned
            # reference; not classified further here.
            rows.append(("WRITES_THROUGH_LOCAL_REF", subj, f["deref_write_local"], "charon", "POSSIBLE"))
        if int(f["assert"]):
            rows.append(("MAY_PANIC", subj, f["assert"] + " assert", "charon", "WITNESSED"))
        if not joined:
            why = "no rust-analyzer symbol at this position" if join is not None else "no rust-analyzer moniker with this path"
            rows.append(("UNJOINED", name, why, "join", "UNKNOWN"))
    return rows


def main(ra_path, charon_path, join_path=None):
    rows, monikers = ra_rows(ra_path)
    rows += charon_rows(charon_path, monikers, load_join(join_path) if join_path else None)
    print("relation\tsubject\tobject\twitness\tevidence")
    for r in rows:
        print("\t".join(r))


if __name__ == "__main__":
    main(*sys.argv[1:4])
