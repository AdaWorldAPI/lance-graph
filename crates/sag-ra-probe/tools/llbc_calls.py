#!/usr/bin/env python3
"""Per-function call and effect facts from a Charon LLBC file (D-SAG-6-1).

Run Charon with --no-dedup-serialized-ast, then:
    python3 -I llbc_calls.py <file.llbc> <crate_name>

For each function of the crate it prints the calls it makes, classified as
  static   resolved callee (Charon `Regular` / `Fun`)
  trait    trait method through a generic clause: callee depends on the
           instantiation, so it is a POSSIBLE callee, never a confirmed one
  dynamic  call through a vtable or function pointer: callee unknown
and three effect facts read from the body: writes through a dereference,
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
                    calls.append("trait:" + next(iter(k)))
            else:
                calls.append("dynamic")
        stats = collections.Counter()
        for k in statements(f["body"]):
            kk = k if isinstance(k, str) else next(iter(k))
            stats[kk] += 1
            if kk == "Assign" and "Deref" in json.dumps(k["Assign"][0]):
                stats["deref_write"] += 1
        print(
            f"{nm}\tcalls={calls}\tderef_write={stats['deref_write']}"
            f"\tassert={stats['Assert']}"
        )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
