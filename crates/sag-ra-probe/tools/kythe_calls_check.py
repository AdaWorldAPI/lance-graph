#!/usr/bin/env python3
"""Serve Charon-derived Kythe entries and check that /xrefs returns the pinned calls.

    python3 -I kythe_calls_check.py --tools DIR --entries entries.json --pin calls.pin.tsv [--port 18098]

entries.json comes from `ruff_charon kythe <file.llbc> <crate> --strip <root>/`
(one JSON object per line). Each pin line is

    callee-signature <TAB> path <TAB> line (1-based) <TAB> expression

read off the source by hand: the call's text and where it starts. A pin passes
when Kythe returns, among the callee's references, an anchor in that file
starting at that line whose byte range covers exactly `expression`. The source
is read from --root to locate the expression; Kythe is the one asked whether
the call exists. Exit 1 on any failing pin.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kythe_xrefs_check import query_xrefs, run, wait_up  # noqa: E402


def load_pins(path, root):
    pins = []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        callee, rel, lineno, expr = line.rstrip("\n").split("\t")
        text = open(os.path.join(root, rel), "rb").read()
        lines = text.split(b"\n")
        n = int(lineno)
        start_of_line = sum(len(l) + 1 for l in lines[: n - 1])
        col = lines[n - 1].find(expr.encode())
        if col < 0:
            raise SystemExit(f"pin expression not on {rel}:{n}: {expr!r}")
        start = start_of_line + col
        pins.append((callee, rel, start, start + len(expr.encode()), expr))
    return pins


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tools", required=True)
    ap.add_argument("--entries", required=True)
    ap.add_argument("--pin", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--corpus", default="lance-graph")
    ap.add_argument("--port", type=int, default=18098)
    args = ap.parse_args(argv)
    pins = load_pins(args.pin, args.root)
    bad = 0
    with tempfile.TemporaryDirectory() as work:
        sorted_ents = os.path.join(work, "entries.bin")
        with open(args.entries) as fi, open(sorted_ents, "wb") as fo:
            run([os.path.join(args.tools, "entrystream"), "--read_format=json", "--sort",
                 "--write_format=delimited"], stdin=fi, stdout=fo)
        tables = os.path.join(work, "tables")
        run([os.path.join(args.tools, "write_tables"), "--entries", sorted_ents, "--out", tables],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        base = f"http://localhost:{args.port}"
        srv = subprocess.Popen([os.path.join(args.tools, "http_server"), "--serving_table", tables,
                                "--listen", f"localhost:{args.port}"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            wait_up(base, srv)
            for callee, rel, start, end, expr in pins:
                _, refs, _ = query_xrefs(base, args.corpus, callee)
                hit = (rel, start, end) in refs
                bad += not hit
                print(f"{'PASS' if hit else 'FAIL'}\tcall {callee} at {rel} {start}..{end} {expr!r}"
                      + ("" if hit else f"\tkythe has {sorted(k for k in refs if k[0] == rel)}"))
        finally:
            srv.terminate()
            srv.wait(timeout=5)
    print(f"call pins: {len(pins) - bad} pass, {bad} fail")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
