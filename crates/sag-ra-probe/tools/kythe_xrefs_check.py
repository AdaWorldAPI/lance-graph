#!/usr/bin/env python3
"""End to end: occ rows -> Kythe entries -> serving tables -> /xrefs, compared back.

    python3 -I kythe_xrefs_check.py --tools DIR --root WS --occ occ.txt [--pin pins.tsv] [--port 18099]

--tools holds entrystream, write_tables, http_server. For every symbol with a
def occurrence the server is asked for all definitions and all references, and
the set of (path, start, end) it returns must equal the occ rows' set, for
definitions and references separately. --pin is a TSV of
`moniker  def|ref  path  line(1-based)  column(0-based byte)`: Kythe's
returned line/column of that occurrence must match. Exit 1 on any mismatch.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kythe_entries  # noqa: E402


def expected_sets(occ_path):
    """{moniker: {"def": {(path,s,e)}, "ref": {...}}} for mapped kinds."""
    want = {}
    with open(occ_path) as fh:
        for moniker, kind, path, s, e, role in kythe_entries.parse_occ(fh):
            if kind in kythe_entries.KIND_TABLE:
                want.setdefault(moniker, {"def": set(), "ref": set()})[role].add((path, s, e))
    return {m: r for m, r in want.items() if r["def"]}


def ticket(corpus, moniker):
    return f"kythe://{corpus}?lang=rust#{urllib.parse.quote(moniker, safe='')}"


def pick(d, *names):
    for n in names:
        if n in d:
            return d[n]
    return None


def anchor_path(anchor):
    # The ticket reads `kythe://corpus?lang=rust?path=P#sig`: a second `?`, so
    # urlsplit's query would swallow it; match the path attribute directly.
    m = re.search(r"[?&]path=([^#&?]*)", anchor["ticket"].split("#", 1)[0])
    return urllib.parse.unquote(m.group(1)) if m else ""


def read_spans(related):
    """Related anchors -> {(path, start, end): (line, col)}."""
    out = {}
    for item in related or []:
        a = item["anchor"]
        span = a["span"]
        s, e = span["start"], span["end"]
        out[(anchor_path(a), int(pick(s, "byte_offset", "byteOffset") or 0), int(pick(e, "byte_offset", "byteOffset") or 0))] = (
            int(pick(s, "line_number", "lineNumber") or 0),
            int(pick(s, "column_offset", "columnOffset") or 0),
        )
    return out


def query_xrefs(base, corpus, moniker):
    body = json.dumps(
        {
            "ticket": [ticket(corpus, moniker)],
            "definition_kind": "ALL_DEFINITIONS",
            "reference_kind": "ALL_REFERENCES",
            "anchor_text": False,
        }
    ).encode()
    req = urllib.request.Request(base + "/xrefs", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        resp = json.load(r)
    xr = pick(resp, "cross_references", "crossReferences") or {}
    entry = next(iter(xr.values()), {})
    return read_spans(entry.get("definition")), read_spans(entry.get("reference")), entry


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def build_tables(args, workdir):
    ents = os.path.join(workdir, "entries.json")
    with open(ents, "w") as fh:
        run([sys.executable, "-I", os.path.join(os.path.dirname(os.path.abspath(__file__)), "kythe_entries.py"),
             "--root", args.root, "--corpus", args.corpus, args.occ], stdout=fh)
    sorted_ents = os.path.join(workdir, "entries.bin")
    with open(ents) as fi, open(sorted_ents, "wb") as fo:
        run([os.path.join(args.tools, "entrystream"), "--read_format=json", "--sort", "--write_format=delimited"],
            stdin=fi, stdout=fo)
    tables = os.path.join(workdir, "tables")
    run([os.path.join(args.tools, "write_tables"), "--entries", sorted_ents, "--out", tables],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tables


def wait_up(base, proc):
    for _ in range(100):
        if proc.poll() is not None:
            raise RuntimeError("http_server exited early")
        try:
            urllib.request.urlopen(base + "/xrefs", b"{}", timeout=2)
            return
        except urllib.error.HTTPError:
            return  # it answered (with an error for the empty request)
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("http_server did not come up")


def load_pins(path):
    pins = []
    for line in open(path):
        c = line.rstrip("\n").split("\t")
        if len(c) == 5 and not line.startswith("#"):
            pins.append((c[0], c[1], c[2], int(c[3]), int(c[4])))
    return pins


def check(base, args, want):
    ok = bad = 0
    got_all = {}
    shown = False
    for moniker in sorted(want):
        defs, refs, entry = query_xrefs(base, args.corpus, moniker)
        got_all[moniker] = (defs, refs)
        if not shown and entry.get("reference"):
            print("raw Kythe span for", moniker, "->", json.dumps(entry["reference"][0]["anchor"]["span"]))
            shown = True
        good = set(defs) == want[moniker]["def"] and set(refs) == want[moniker]["ref"]
        ok, bad = (ok + 1, bad) if good else (ok, bad + 1)
        print(f"{'PASS' if good else 'FAIL'}\t{moniker}\tdef={len(defs)}/{len(want[moniker]['def'])}\tref={len(refs)}/{len(want[moniker]['ref'])}")
        if not good:
            for role, got in (("def", defs), ("ref", refs)):
                for x in sorted(set(got) ^ want[moniker][role]):
                    print(f"  {role} mismatch {'kythe-only' if x in got else 'occ-only'} {x}")
    return ok, bad, got_all


def check_pins(pins, got_all, want):
    bad = 0
    for moniker, role, path, line, col in pins:
        # Identify the pinned occurrence by position on that line/col in Kythe's own answer.
        spans = got_all.get(moniker, ({}, {}))[0 if role == "def" else 1]
        hit = [k for k, v in spans.items() if k[0] == path and v == (line, col)]
        good = bool(hit)
        bad += not good
        print(f"{'PASS' if good else 'FAIL'}\tpin {moniker} {role} {path}:{line}:{col}")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tools", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--occ", required=True)
    ap.add_argument("--pin")
    ap.add_argument("--corpus", default="lance-graph")
    ap.add_argument("--port", type=int, default=18099)
    args = ap.parse_args(argv)
    want = expected_sets(args.occ)
    with tempfile.TemporaryDirectory() as workdir:
        tables = build_tables(args, workdir)
        base = f"http://localhost:{args.port}"
        srv = subprocess.Popen([os.path.join(args.tools, "http_server"), "--serving_table", tables,
                                "--listen", f"localhost:{args.port}"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            wait_up(base, srv)
            ok, bad, got_all = check(base, args, want)
            pin_bad = check_pins(load_pins(args.pin), got_all, want) if args.pin else 0
        finally:
            srv.terminate()
            try:
                srv.wait(timeout=5)
            except subprocess.TimeoutExpired:
                srv.kill()
    print(f"symbols: {ok} pass, {bad} fail; pin failures: {pin_bad}")
    sys.exit(1 if bad or pin_bad else 0)


if __name__ == "__main__":
    main()
