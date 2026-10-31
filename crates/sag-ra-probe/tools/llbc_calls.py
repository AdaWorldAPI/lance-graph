#!/usr/bin/env python3
"""Per-function call and effect facts from a Charon LLBC file (D-SAG-6-1).

Run Charon with --no-dedup-serialized-ast, then:
    python3 -I llbc_calls.py <file.llbc> <crate_name>

For each function of the crate it prints the calls it makes, classified as
  static   resolved callee (Charon `Regular` / `Fun`)
  trait:K  trait method call; K is how the impl is chosen: `Clause` (a
           generic bound, fixed per instantiation), `Dyn` (a vtable: any impl,
           including other crates'), `TraitImpl` (a known impl). Only a known
           impl is a confirmed callee
  dynamic  call through a function pointer or vtable slot: callee unknown
and effect facts read from the body: writes through a dereference rooted in
an argument (reaches caller memory; for a closure, its captured state) or in
a local (may still alias),
`Assert` statements (overflow / division / bounds checks that can panic),
and `Call` count. These are facts witnessed in Charon's output, nothing more.
"""
import collections
import json
import sys


def name(path):
    return "::".join(
        str(c["Ident"][0]) if "Ident" in c else next(iter(c)) for c in path
    )


def walk(o, key):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key:
                yield v
            yield from walk(v, key)
    elif isinstance(o, list):
        for v in o:
            yield from walk(v, key)


def root_local(place):
    """The local a place is rooted in: the innermost `Local` of its projections."""
    kind = place["kind"]
    while isinstance(kind, dict) and "Projection" in kind:
        kind = kind["Projection"][0]["kind"]
    return kind["Local"] if isinstance(kind, dict) and "Local" in kind else 10**9


def statements(o):
    if isinstance(o, dict):
        if {"kind", "id", "span"} <= o.keys():
            yield o["kind"]
        for v in o.values():
            yield from statements(v)
    elif isinstance(o, list):
        for v in o:
            yield from statements(v)


def main(path, crate):
    doc = json.load(open(path))
    t = doc["translated"]
    funs = {f["def_id"]: f for f in t["fun_decls"] if f}
    print(f"has_errors\t{doc['has_errors']}")
    for f in funs.values():
        nm = name(f["item_meta"]["name"])
        if not nm.startswith(crate + "::"):
            continue
        calls = []
        for c in walk(f["body"], "Call"):
            func = c["call"]["func"]
            kind = next(iter(func))
            if kind == "Regular":
                k = func["Regular"]["kind"]
                if "Fun" in k:
                    tgt = funs.get(k["Fun"])
                    calls.append("static:" + (name(tgt["item_meta"]["name"]) if tgt else str(k["Fun"])))
                else:
                    # A trait method call. Its trait reference says how the
                    # impl is chosen: `Clause` (a generic bound, fixed at
                    # instantiation), `Dyn` (a vtable, any impl), `TraitImpl`
                    # (a known impl), among others.
                    tr = k["Trait"][0]
                    tr = tr.get("Untagged", tr)
                    kind = tr["kind"] if isinstance(tr["kind"], str) else next(iter(tr["kind"]))
                    calls.append("trait:" + kind)
            else:
                calls.append("dynamic")
        stats = collections.Counter()
        arg_count = next(walk(f["body"], "arg_count"), 0)
        for k in statements(f["body"]):
            kk = k if isinstance(k, str) else next(iter(k))
            stats[kk] += 1
            if kk == "Assign" and "Deref" in json.dumps(k["Assign"][0]):
                # Rooted in an argument (for a closure, its captured state
                # is argument 1) the write can reach the caller's memory;
                # rooted in a local it may still alias one (a reference
                # returned by a call), so it is only "local-rooted".
                if root_local(k["Assign"][0]) <= arg_count:
                    stats["deref_write_arg"] += 1
                else:
                    stats["deref_write_local"] += 1
        print(
            f"{nm}\tcalls={calls}\tderef_write_arg={stats['deref_write_arg']}"
            f"\tderef_write_local={stats['deref_write_local']}\tassert={stats['Assert']}"
        )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
