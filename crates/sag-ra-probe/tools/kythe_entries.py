#!/usr/bin/env python3
"""Turn rust-analyzer `occ` rows into Kythe entries (JSON, one per line).

    python3 -I kythe_entries.py --root <workspace> [--corpus lance-graph] [occ.txt]

Input lines (all others are ignored):

    occ  <moniker>  <kind>  <file>  <start>  <end>  <def|ref>

start/end are byte offsets into <file> (relative to --root). Output is the
shape `entrystream --read_format=json` accepts: node facts
`{"source": VName, "fact_name": n, "fact_value": base64}` and edges
`{"source": VName, "edge_kind": k, "target": VName, "fact_name": "/"}`.

Only what the input witnesses is emitted: a file node, a symbol node per
moniker, an anchor per occurrence, and one defines/binding or ref edge per
anchor. No ref/call, no containment, nothing inferred. A symbol whose kind
has no entry in KIND_TABLE is skipped (with its anchors) and counted on
stderr. A range outside the file, or start > end, exits 2.
"""
import argparse
import base64
import json
import os
import sys

KIND_TABLE = {
    "Function": "function",
    "Method": "function",
    "TraitMethod": "function",
    "Struct": "record",
    "Enum": "record",
    "Union": "record",
    "Trait": "interface",
    "Module": "package",
    "Const": "variable",
    "Static": "variable",
    "Field": "variable",
    "Variable": "variable",
    "Macro": "macro",
    "TypeAlias": "talias",
}
EDGE_OF_ROLE = {
    "def": "/kythe/edge/defines/binding",
    "ref": "/kythe/edge/ref",
}


def b64(data):
    return base64.b64encode(data).decode("ascii")


def vname(corpus, signature="", path="", language=""):
    return {"signature": signature, "corpus": corpus, "root": "", "path": path, "language": language}


def parse_occ(lines):
    for line in lines:
        cols = line.rstrip("\n").split("\t")
        if cols[0] != "occ" or len(cols) != 7:
            continue
        _, moniker, kind, path, start, end, role = cols
        if role not in EDGE_OF_ROLE:
            continue
        yield moniker, kind, path, int(start), int(end), role


def fact(source, name, value):
    return {"source": source, "fact_name": name, "fact_value": b64(value)}


def build(lines, root, corpus):
    """Return (entries, skipped) where skipped maps kind -> set of monikers."""
    entries, seen = [], set()
    skipped, files = {}, {}

    def emit(e):
        key = json.dumps(e, sort_keys=True)
        if key not in seen:
            seen.add(key)
            entries.append(e)

    for moniker, kind, path, start, end, role in parse_occ(lines):
        if kind not in KIND_TABLE:
            skipped.setdefault(kind, set()).add(moniker)
            continue
        if path not in files:
            with open(os.path.join(root, path), "rb") as fh:
                files[path] = fh.read()
        size = len(files[path])
        if start > end or end > size:
            raise ValueError(f"bad range {start}..{end} in {path} ({size} bytes): {moniker}")
        fnode = vname(corpus, path=path)
        emit(fact(fnode, "/kythe/node/kind", b"file"))
        emit(fact(fnode, "/kythe/text", files[path]))
        sym = vname(corpus, signature=moniker, language="rust")
        emit(fact(sym, "/kythe/node/kind", KIND_TABLE[kind].encode()))
        anchor = vname(corpus, signature=f"@{start}:{end}", path=path, language="rust")
        emit(fact(anchor, "/kythe/node/kind", b"anchor"))
        emit(fact(anchor, "/kythe/loc/start", str(start).encode()))
        emit(fact(anchor, "/kythe/loc/end", str(end).encode()))
        emit({"source": anchor, "edge_kind": EDGE_OF_ROLE[role], "target": sym, "fact_name": "/"})
    return entries, skipped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("occ", nargs="?", help="occ rows (default stdin)")
    ap.add_argument("--root", required=True, help="workspace root the file paths are relative to")
    ap.add_argument("--corpus", default="lance-graph")
    args = ap.parse_args(argv)
    src = open(args.occ) if args.occ else sys.stdin
    try:
        entries, skipped = build(src, args.root, args.corpus)
    except ValueError as e:
        print(f"kythe_entries: {e}", file=sys.stderr)
        sys.exit(2)
    for e in entries:
        print(json.dumps(e, sort_keys=True))
    n_sk = sum(len(v) for v in skipped.values())
    print(f"kythe_entries: {len(entries)} entries; skipped {n_sk} symbols of unmapped kind", file=sys.stderr)
    for kind, mons in sorted(skipped.items()):
        print(f"  skipped kind {kind}: {len(mons)}", file=sys.stderr)


if __name__ == "__main__":
    main()
