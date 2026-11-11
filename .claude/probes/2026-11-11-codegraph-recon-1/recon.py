#!/usr/bin/env python3
"""D-CODEGRAPH-RECON-1: rebuild an architecture diagram from compiler facts, score it against the
hand-curated Mermaid corpus, and emit a read-only review evidence packet.

Inputs are never written. Commands:

  recover   --store DIR --src ROOT --seeds FILE              machine-recovered Mermaid (facts only)
  score     --store DIR --src ROOT --corpus TSV [--ra TSV]   recall of the corpus' call edges
  reconcile --store DIR --src ROOT --seeds FILE --corpus TSV machine diagram + labelled human rows
  packet    --store DIR --src ROOT --ra TSV --symbol SYM [--diff FILE] [--rev SHA]
  --self-test                                                the D-CODEGRAPH-RECON-1 negatives

`--store` is a ruff_charon store (`<crate>.rows.tsv`, `<crate>.files.tsv`). `--ra` is the
`GRAPH=1` output of crates/sag-ra-probe. A missing store is an error (ARTIFACT_MISSING, exit 3),
never an empty diagram. The machine diagram reads the store only: the corpus is not opened by
`recover`, and the seed list names nodes, never edges.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

CALL = "CALLS_RESOLVED"
MAY = "MAY_DISPATCH"
LOCAL = ("lance_graph_",)  # callees outside these prefixes (core, alloc, std) are not drawn


class ArtifactMissing(Exception):
    pass


def fnv64(data):
    # Same hash as ruff_charon's files.tsv and .claude/tools/codegraph_index.py.
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def read_schema_tsv(path):
    cols, rows = None, []
    for line in path.read_text().splitlines():
        if line.startswith("#schema "):
            cols = line[len("#schema "):].split("\t")
        elif line and not line.startswith("#"):
            rows.append(dict(zip(cols, line.split("\t"))))
    return rows


class Store:
    """The rows of every crate in a ruff_charon store directory, plus file freshness."""

    def __init__(self, store, src):
        store, self.src = Path(store), Path(src)
        files = sorted(store.glob("*.rows.tsv")) if store.is_dir() else []
        if not files:
            raise ArtifactMissing(f"ARTIFACT_MISSING\tno *.rows.tsv in {store}")
        self.crates = [f.name[: -len(".rows.tsv")] for f in files]
        self.rows = [r for f in files for r in read_schema_tsv(f)]
        self.fresh = {}
        for c in self.crates:
            ft = store / f"{c}.files.tsv"
            if not ft.exists():
                raise ArtifactMissing(f"ARTIFACT_MISSING\t{ft}")
            for r in read_schema_tsv(ft):
                p = self.src / r["path"]
                self.fresh[r["path"]] = (
                    "MISSING_SOURCE" if not p.exists()
                    else "FRESH" if fnv64(p.read_bytes()) == r["fnv64"] else "STALE")
        self.public = {r["subject"] for r in self.rows if r["predicate"] == "PUBLIC"}
        self.defines = {r["subject"]: r for r in self.rows if r["predicate"] == "DEFINES"}

    def line(self, path, offset):
        p = self.src / path
        if not p.exists():
            return None
        return p.read_bytes()[: int(offset)].count(b"\n") + 1

    def edges(self):
        """(caller, callee, kind, file, line, freshness) for every call row; dispatch rows keep
        their target only (`X via Clause` -> X)."""
        out = []
        for r in self.rows:
            if r["predicate"] not in (CALL, MAY):
                continue
            callee = r["object"].split(" via ")[0]
            out.append((r["subject"], callee, r["predicate"], r["file"],
                        self.line(r["file"], r["start"]), self.fresh.get(r["file"], "UNOBSERVED")))
        return out


def short(sym):
    parts = re.sub(r"\{([^}]*)\}", r"\1", sym).split("::")
    return "::".join(parts[-2:]) if len(parts) > 1 else sym


def node_id(sym):
    return "n" + hashlib.sha1(sym.encode()).hexdigest()[:10]


def recover(store, seeds):
    """Mermaid flowchart of the seeds' direct local callees, from store rows only."""
    seeds = sorted(set(seeds))
    present = [s for s in seeds if s in store.defines]
    missing = [s for s in seeds if s not in store.defines]
    lines = ["flowchart LR", "%% machine-recovered from compiler facts only; no curated edge",
             "classDef stale fill:#fdd,stroke:#c00", "classDef unknown fill:#eee,stroke:#888,stroke-dasharray:3 3"]
    nodes, edges = {}, set()
    for caller, callee, kind, f, ln, fr in store.edges():
        if caller not in present or not callee.startswith(LOCAL):
            continue
        nodes[caller] = nodes[callee] = None
        edges.add((caller, callee, kind, f, ln, fr))
    for s in present:
        nodes[s] = None
    for s in sorted(nodes):
        lines.append(f'  {node_id(s)}["{short(s)}"]')
    for caller, callee, kind, f, ln, fr in sorted(edges, key=lambda e: (e[0], e[1], e[2], e[3], e[4] or 0)):
        a, b = node_id(caller), node_id(callee)
        if fr != "FRESH":
            # STALE, MISSING_SOURCE or UNOBSERVED: never drawn as a verified edge.
            lines.append(f"  {a} -. {fr} .-> {b}  %% {f}:{ln if ln is not None else '?'}")
        elif kind == MAY:
            lines.append(f"  {a} -.->|possible| {b}  %% {f}:{ln}")
        else:
            lines.append(f"  {a} --> {b}  %% {f}:{ln}")
    for s in present:
        if s in store.public:
            u = node_id(s) + "_ext"
            lines.append(f'  {u}["callers outside {", ".join(store.crates)}: UNKNOWN"]:::unknown -.-> {node_id(s)}')
    for s in missing:
        lines.append(f'  %% seed not in store (UNOBSERVED, not absent): {s}')
    return "\n".join(lines) + "\n"


def cited_site(cell):
    """First `path:line` in a corpus `verified_line` cell, normalised to a repo-relative path."""
    m = re.search(r"((?:crates|OGAR/crates)/[\w./-]+\.rs):(\d+)", cell)
    return (m.group(1), int(m.group(2))) if m else None


def read_corpus(path):
    lines = Path(path).read_text().splitlines()
    head = lines[0].split("\t")
    return [dict(zip(head, ln.split("\t"))) for ln in lines[1:] if ln.strip()]


def ra_calls(ra_path, src):
    """(file, line) of every rust-analyzer `call` row, plus the files it saw and its macro regions."""
    sites, files, macros, cache = set(), set(), defaultdict(list), {}
    for ln in Path(ra_path).read_text().splitlines():
        p = ln.split("\t")
        if p[0] == "file":
            files.add(p[1])
        elif p[0] == "macro":
            macros[p[1]].append((int(p[2]), int(p[3])))
        elif p[0] == "call":
            f, s = p[4], int(p[5])
            if f not in cache:
                fp = Path(src) / f
                cache[f] = fp.read_bytes() if fp.exists() else b""
            sites.add((f, cache[f][:s].count(b"\n") + 1))
    return sites, files, macros


def score(store, corpus, ra=None):
    """For every MECHANICAL-CALLGRAPH corpus row, does a compiler call site sit at its cited line
    (exact, or within the cited row's following two lines for multi-line calls)?"""
    st_sites = {(f, ln) for _, _, _, f, ln, _ in store.edges()}
    st_files = set(store.fresh)
    ra_sites, ra_files = (ra[0], ra[1]) if ra else (set(), set())
    out = []
    for r in corpus:
        if r.get("recoverable") != "MECHANICAL-CALLGRAPH":
            continue
        site = cited_site(r.get("verified_line", ""))
        row = {"id": r["id"], "site": site and f"{site[0]}:{site[1]}"}
        for arm, sites, files in (("charon", st_sites, st_files), ("ra", ra_sites, ra_files)):
            if arm == "ra" and not ra:
                continue
            if site is None:
                row[arm] = "NO_SITE"
            elif site[0] not in files:
                row[arm] = "UNOBSERVED"
            else:
                row[arm] = "HIT" if any((site[0], site[1] + d) in sites for d in (0, 1, 2)) else "MISS"
        out.append(row)
    return out


def reconcile(store, seeds, corpus):
    """The machine diagram plus corpus rows that compiler facts cannot carry, each labelled as
    human interpretation with its corpus id. A GAP is never drawn as a solid edge."""
    base = recover(store, seeds).rstrip("\n").split("\n")
    base += ["%% --- human interpretation (corpus rows; not compiler facts) ---",
             "classDef gap fill:#fff3cd,stroke:#b58900,stroke-dasharray:5 5",
             "classDef lab fill:#e8f4ff,stroke:#268bd2",
             "classDef proposed fill:#f3e8ff,stroke:#6c71c4,stroke-dasharray:2 2",
             "classDef contradicted fill:#fdd,stroke:#c00"]
    for r in corpus:
        rid, status, rec = r["id"], r.get("status", ""), r.get("recoverable", "")
        label = re.sub(r'["\[\]{}|<>]', " ", f'{rid}: {r.get("subject", "")} / {r.get("object", "")}')[:90]
        if rec == "NOT-RECOVERABLE" or status == "MISSING":
            base.append(f'  {rid}["{label}"]:::gap')
        elif status in ("LAB", "PROPOSED", "OPEN-PR") and rec != "MECHANICAL-CALLGRAPH":
            cls = {"LAB": "lab", "PROPOSED": "proposed", "OPEN-PR": "proposed"}[status]
            base.append(f'  {rid}["{label} ({status})"]:::{cls}')
        if "CONTRADICTED" in r.get("verified_line", ""):
            base.append(f'  {rid}_c["{rid} CONTRADICTED by source; not drawn as an edge"]:::contradicted')
    return "\n".join(base) + "\n"


def git(src, *args):
    try:
        return subprocess.run(["git", "-C", str(src), *args], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def field_readers(store, field):
    """(function, path:line) for every DEFINES span whose body text reads `.field` (not a call)."""
    pat = re.compile(rb"\.%s\b(?!\s*\()" % re.escape(field.encode()))
    out, cache = [], {}
    for sym, d in sorted(store.defines.items()):
        f = d["file"]
        if f not in cache:
            fp = store.src / f
            cache[f] = fp.read_bytes() if fp.exists() else b""
        body = cache[f][int(d["start"]):int(d["end"])]
        for m in pat.finditer(body):
            out.append((sym, f"{f}:{cache[f][:int(d['start']) + m.start()].count(b'\n') + 1}"))
    return out


def packet(store, ra, symbol, diff_text=None, rev=None):
    """A read-only ReviewEvidencePacket (crai-rs CR-H0 §3 OUT fields, CR-H1 identity fields,
    CR-H0-REVIEW-SEMANTICS axes) for a change to `symbol`. It holds evidence, never a verdict."""
    d = store.defines.get(symbol)
    if d is None:
        # Not a function: maybe a field (`Type::field`). Compiler facts carry no field reads, so a
        # read that sits only in a guard (`if b.when.is_some()`) would vanish from a call graph.
        # Report every in-store function body that reads `.field` as SOURCE_STRUCTURE / POSSIBLE.
        readers = field_readers(store, symbol.split("::")[-1])
        return {"subject": symbol, "status": "UNKNOWN", "reason": "not a function in the store",
                "observed_crates": store.crates,
                "field_readers": [{"function": f, "site": site, "evidence_class": "SOURCE_STRUCTURE",
                                   "qualifier": "POSSIBLE", "why": "field-name match, receiver type unresolved"}
                                  for f, site in readers]}
    src = store.src
    callers, possible = defaultdict(set), defaultdict(set)
    sites, fresh_of = defaultdict(list), {}
    for caller, callee, kind, f, ln, fr in store.edges():
        (callers if kind == CALL else possible)[callee].add(caller)
        sites[(caller, callee)].append(f"{f}:{ln if ln is not None else '?'}")
        # An edge is current only if every row behind it sits in a FRESH file.
        fresh_of[(caller, callee)] = fresh_of.get((caller, callee), True) and fr == "FRESH"

    def closure(seeds):
        """Callers reachable from `seeds` through resolved calls: (all-fresh path, reached only
        through stale evidence)."""
        ok, stale_only, frontier = set(), {}, [(x, True) for x in seeds]
        while frontier:
            x, fresh = frontier.pop()
            for c in callers.get(x, ()):
                if c == symbol:
                    continue
                f = fresh and fresh_of.get((c, x), True)
                if f and c not in ok:
                    ok.add(c)
                    stale_only.pop(c, None)
                    frontier.append((c, True))
                elif not f and c not in ok and c not in stale_only:
                    stale_only[c] = sites[(c, x)]
                    frontier.append((c, False))
        return ok, stale_only

    confirmed, stale = closure([symbol])
    # Dispatch: callers of a trait method this symbol implements may reach it (the
    # IMPLEMENTS -> MAY_DISPATCH join Codegraph::callers makes), and so may their callers.
    traits = {r["object"] for r in store.rows if r["predicate"] == "IMPLEMENTS" and r["subject"] == symbol}
    may_direct = set().union(*(possible.get(t, set()) for t in traits | {symbol})) - {symbol}
    may_up, may_stale = closure(sorted(may_direct))
    possible_all = (may_direct | may_up) - confirmed
    roots = sorted(c for c in confirmed if not callers.get(c))
    contracts = sorted({callee for caller, callee, kind, *_ in store.edges() if caller == symbol and kind == MAY})
    leaf = symbol.split("::")[-1]
    ra_sites, ra_files, macros = ra
    # Test call sites: source search by method name in test code, graded SOURCE_STRUCTURE/POSSIBLE
    # unless a compiler row confirms them. A name match is not a call of this method.
    tests = []
    for p in sorted(src.glob("crates/*/tests/**/*.rs")) + sorted(src.glob("crates/*/src/**/tests.rs")):
        rel = str(p.relative_to(src))
        data = p.read_bytes()
        for m in re.finditer(rb"\.%s\(" % re.escape(leaf.encode()), data):
            ln = data[: m.start()].count(b"\n") + 1
            in_macro = any(a <= m.start() < b for a, b in macros.get(rel, ()))
            tests.append({"site": f"{rel}:{ln}", "evidence_class": "SOURCE_STRUCTURE",
                          "qualifier": "POSSIBLE", "why": "method-name match, receiver unresolved",
                          "ra": ("CONFIRMED" if (rel, ln) in ra_sites else
                                 "UNKNOWN: inside a macro region" if in_macro else
                                 "UNKNOWN: file not indexed" if rel not in ra_files else "no call row"),
                          "run_status": "NOT_RUN"})
    span_lines = (store.line(d["file"], d["start"]), store.line(d["file"], d["end"]))
    blob = git(src, "hash-object", d["file"])
    unknown = []
    if symbol in store.public:
        unknown.append({"what": "callers in crates outside the store", "observed_crates": store.crates,
                        "why": "symbol is PUBLIC; only these crates' library targets were compiled"})
    unknown.append({"what": "test targets, examples, cfg(test) modules", "why": "store holds library targets only"})
    prov = {"tool": "charon c69e231 -> ruff_charon import (rows.tsv v2)", "toolchain": "nightly-2026-09-17",
            "sysroot": "default (cargo-miri absent; std bodies opaque)"}
    freshness = store.fresh.get(d["file"], "UNOBSERVED")
    return {
        "packet": "ReviewEvidencePacket (prototype, read-only; not a vendor API)",
        "identity": {"repo_id": "AdaWorldAPI/lance-graph", "commit_sha": rev or git(src, "rev-parse", "HEAD"),
                     "cargo_lock_sha": "NOT_TRACKED (repository tracks no Cargo.lock)",
                     "crate_instance": d["subject"].split("::")[0] + " (lib target)", "features": "default",
                     "target_triple": "x86_64-unknown-linux-gnu", "rustc_version": "nightly-2026-09-17 (charon)",
                     "file_blob_sha": blob, "source_span": {"file": d["file"], "bytes": [int(d["start"]), int(d["end"])],
                                                            "lines": list(span_lines)}},
        "changed_source": {"subject": symbol, "base": rev or git(src, "rev-parse", "HEAD"),
                           "diff": diff_text if diff_text is not None else "NOT_SUPPLIED (prototype: change of the whole span)",
                           "stale": freshness != "FRESH", "freshness": freshness,
                           "dependency_evidence_stale": bool(stale or may_stale)},
        "affected_symbols": {"confirmed": sorted(confirmed),
                             "possible": sorted(possible_all),
                             "stale": [{"caller": c, "sites": st} for c, st in sorted({**may_stale, **stale}.items())],
                             "unknown": unknown},
        "witnessed_edges": [{"caller": c, "callee": symbol, "sites": sites[(c, symbol)],
                             "fresh": fresh_of.get((c, symbol), True)} for c in sorted(callers.get(symbol, ()))],
        "possible_edges": [{"dispatch_target": t} for t in contracts],
        "co_changed_only": "NOT_RUN (no history projection in this prototype)",
        "unsupported_regions": sorted({t["site"].split(":")[0] for t in tests if not t["ra"].startswith("CONFIRMED")}),
        "findings": [
            {"id": "dependents", "summary": f"{len(confirmed)} confirmed transitive in-store callers; external callers UNKNOWN",
             "severity": None, "evidence_class": "COMPILER_DERIVED", "confidence": "as graded per item",
             "provenance": prov, "lifecycle": "open", "run_status": "PARTIAL"},
            {"id": "tests", "summary": f"{len(tests)} candidate test call sites by name; none run",
             "severity": None, "evidence_class": "SOURCE_STRUCTURE", "qualifier": "POSSIBLE",
             "items": tests, "provenance": {"tool": "source search + rust-analyzer GRAPH=1 (ra_ap 0.0.358)"},
             "lifecycle": "open", "run_status": "NOT_RUN"},
            {"id": "contracts", "summary": "trait methods the symbol dispatches through (a change may interact with any impl)",
             "severity": None, "evidence_class": "COMPILER_DERIVED", "qualifier": "POSSIBLE", "items": contracts,
             "provenance": prov, "lifecycle": "open", "run_status": "COMPLETE"},
            {"id": "execution_paths", "summary": "entry points (in-store callers with no in-store caller)",
             "severity": None, "evidence_class": "COMPILER_DERIVED", "items": roots,
             "provenance": prov, "lifecycle": "open", "run_status": "PARTIAL"},
        ],
        "r2il": {"method": "NOT_A_METHOD", "body_semantics": "UNKNOWN",
                 "why": "R2IL describes the effects of a registered DT64 MethodBinding (prepare binds them to "
                        "opcodes); a Rust function found by the compiler is not one, so its body has no R2IL reading"},
        "expected_behavior": "NOT_SUPPLIED (must come from intent/tests, independently of this packet)",
        "candidate_behavior": "NOT_INFERRED",
        "missing_necessary_evidence": ["test runs (TEST_OBSERVED needs a run that can fail, CR-H0 §5)",
                                       "compiler artifacts of consumer crates and test targets",
                                       "a resolved receiver for each name-matched test site"],
        "permission_state": "NOT_DERIVED (never from confidence)",
    }


# ---------------------------------------------------------------------------------------------
# Negative tests (D-CODEGRAPH-RECON-1 §7). Each builds a tiny synthetic store.

def mk_store(td, files, rows, crate="lance_graph_x"):
    td = Path(td)
    (td / "src").mkdir(parents=True, exist_ok=True)
    st = td / "store"
    st.mkdir(exist_ok=True)
    for path, body in files.items():
        p = td / "src" / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body)
    (st / f"{crate}.files.tsv").write_text("#version 1\n#schema path\tfnv64\n" +
                                          "".join(f"{p}\t{fnv64(b)}\n" for p, b in files.items()))
    (st / f"{crate}.rows.tsv").write_text("#version 2\n#schema predicate\tsubject\tobject\tgrade\tfile\tstart\tend\n" +
                                         "".join("\t".join(map(str, r)) + "\n" for r in rows))
    return st, td / "src"


LIB = b"fn a() { b(); }\nfn b() {}\nfn c(x: &dyn T) { x.m(); }\n"
ROWS = [("DEFINES", "lance_graph_x::a", "lance_graph_x::a", "COMPILER", "lib.rs", 0, 15),
        ("DEFINES", "lance_graph_x::b", "lance_graph_x::b", "COMPILER", "lib.rs", 16, 26),
        ("DEFINES", "lance_graph_x::c", "lance_graph_x::c", "COMPILER", "lib.rs", 27, 55),
        ("PUBLIC", "lance_graph_x::b", "lance_graph_x::b", "COMPILER", "lib.rs", 16, 26),
        ("CALLS_RESOLVED", "lance_graph_x::a", "lance_graph_x::b", "COMPILER", "lib.rs", 9, 12),
        ("MAY_DISPATCH", "lance_graph_x::c", "lance_graph_x::T::m via Clause", "COMPILER", "lib.rs", 46, 49)]


class Negatives(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.st, self.src = mk_store(self.td.name, {"lib.rs": LIB}, ROWS)

    def tearDown(self):
        self.td.cleanup()

    def test_01_stale_facts_cannot_claim_current_code(self):
        (self.src / "lib.rs").write_bytes(LIB + b"// edited\n")
        out = recover(Store(self.st, self.src), ["lance_graph_x::a"])
        self.assertIn("-. STALE .->", out)
        self.assertNotIn(" --> ", out)

    def test_02_renamed_symbol_keeps_its_own_identity(self):
        rows = ROWS + [("DEFINES", "lance_graph_x::{F}::check", "lance_graph_x::{F}::check", "COMPILER", "lib.rs", 0, 1),
                       ("DEFINES", "lance_graph_x::{L}::check", "lance_graph_x::{L}::check", "COMPILER", "lib.rs", 2, 3),
                       ("CALLS_RESOLVED", "lance_graph_x::a", "lance_graph_x::{F}::check", "COMPILER", "lib.rs", 9, 12)]
        st, src = mk_store(self.td.name + "/r", {"lib.rs": LIB}, rows)
        p = packet(Store(st, src), (set(), set(), {}), "lance_graph_x::{L}::check")
        self.assertEqual(p["affected_symbols"]["confirmed"], [])  # F::check's caller is not L::check's
        self.assertEqual(packet(Store(st, src), (set(), set(), {}), "lance_graph_x::{F}::check")
                         ["affected_symbols"]["confirmed"], ["lance_graph_x::a"])

    def test_03_dynamic_call_is_never_absent_nor_confirmed(self):
        out = recover(Store(self.st, self.src), ["lance_graph_x::c"])
        self.assertIn("-.->|possible|", out)
        self.assertNotRegex(out, r"n\w+ --> n\w+")

    def test_04_macro_hidden_call_stays_unknown(self):
        tests = self.src / "crates" / "lance_graph_x" / "tests"
        tests.mkdir(parents=True)
        body = b"fn t() { assert_eq!(z.b(), 1); }\n"
        (tests / "t.rs").write_bytes(body)
        rel = "crates/lance_graph_x/tests/t.rs"
        ra = (set(), {rel}, {rel: [(9, 31)]})
        p = packet(Store(self.st, self.src), ra, "lance_graph_x::b")
        items = [i for f in p["findings"] if f["id"] == "tests" for i in f["items"]]
        self.assertEqual([i["ra"] for i in items], ["UNKNOWN: inside a macro region"])
        self.assertIn(rel, p["unsupported_regions"])

    def test_05_missing_artifact_is_not_empty(self):
        with self.assertRaises(ArtifactMissing):
            Store(Path(self.td.name) / "nope", self.src)
        (self.st / "lance_graph_x.files.tsv").unlink()
        with self.assertRaises(ArtifactMissing):
            Store(self.st, self.src)

    def test_06_guard_only_dependency_is_not_omitted(self):
        # A field read only inside a guard has no call row; it must still be reported.
        body = b"fn p(b: &B) -> u8 { if b.when.is_some() { return 1; } 0 }\nfn q(b: &B) { b.other(); }\n"
        rows = [("DEFINES", "lance_graph_x::p", "lance_graph_x::p", "COMPILER", "lib.rs", 0, 57),
                ("DEFINES", "lance_graph_x::q", "lance_graph_x::q", "COMPILER", "lib.rs", 58, 85)]
        st, src = mk_store(self.td.name + "/g", {"lib.rs": body}, rows)
        p = packet(Store(st, src), (set(), set(), {}), "lance_graph_x::{MethodBinding}::when")
        self.assertEqual([r["function"] for r in p["field_readers"]], ["lance_graph_x::p"])

    def test_06b_public_symbol_reports_external_callers_unknown(self):
        # A public symbol whose only dependents are outside the store must report UNKNOWN, not none.
        p = packet(Store(self.st, self.src), (set(), set(), {}), "lance_graph_x::b")
        self.assertTrue(any("outside the store" in u["what"] for u in p["affected_symbols"]["unknown"]))
        p = packet(Store(self.st, self.src), (set(), set(), {}), "lance_graph_x::a")
        self.assertFalse(any("outside the store" in u["what"] for u in p["affected_symbols"]["unknown"]))

    def test_07_rbac_gap_is_never_rendered_as_an_edge(self):
        corpus = [{"id": "R9", "subject": "ScopedDecision", "object": "bind_lanes", "status": "MISSING",
                   "recoverable": "NOT-RECOVERABLE", "verified_line": ""}]
        out = reconcile(Store(self.st, self.src), ["lance_graph_x::a"], corpus)
        line = [ln for ln in out.splitlines() if "R9" in ln]
        self.assertTrue(line and all(":::gap" in ln and "-->" not in ln for ln in line))

    def test_08_a_source_reference_is_not_a_call(self):
        rows = ROWS + [("REFERENCES", "lance_graph_x::c", "lance_graph_x::b", "COMPILER", "lib.rs", 30, 31)]
        st, src = mk_store(self.td.name + "/q", {"lib.rs": LIB}, rows)
        p = packet(Store(st, src), (set(), set(), {}), "lance_graph_x::b")
        self.assertEqual(p["affected_symbols"]["confirmed"], ["lance_graph_x::a"])
        self.assertEqual(p["affected_symbols"]["possible"], [])  # a reference is not a dispatch either
        out = recover(Store(st, src), ["lance_graph_x::c"])
        self.assertNotIn(node_id("lance_graph_x::b"), out)  # c only references b: no edge, no node

    def test_08b_a_rust_function_is_not_an_r2il_method(self):
        p = packet(Store(self.st, self.src), (set(), set(), {}), "lance_graph_x::b")
        self.assertEqual((p["r2il"]["method"], p["r2il"]["body_semantics"]), ("NOT_A_METHOD", "UNKNOWN"))

    def test_11_stale_caller_evidence_is_not_confirmed(self):
        # The caller's file changed after the store was built; the queried function's file did not.
        files = {"a.rs": b"fn a() { b(); }\n", "b.rs": b"fn b() {}\n"}
        rows = [("DEFINES", "lance_graph_x::a", "lance_graph_x::a", "COMPILER", "a.rs", 0, 15),
                ("DEFINES", "lance_graph_x::b", "lance_graph_x::b", "COMPILER", "b.rs", 0, 9),
                ("CALLS_RESOLVED", "lance_graph_x::a", "lance_graph_x::b", "COMPILER", "a.rs", 9, 12)]
        st, src = mk_store(self.td.name + "/s", files, rows)
        (src / "a.rs").write_bytes(b"fn a() {}\n")  # b() removed
        p = packet(Store(st, src), (set(), set(), {}), "lance_graph_x::b")
        self.assertEqual(p["affected_symbols"]["confirmed"], [])
        self.assertEqual([s["caller"] for s in p["affected_symbols"]["stale"]], ["lance_graph_x::a"])
        self.assertTrue(p["changed_source"]["dependency_evidence_stale"])

    def test_12_dispatch_reaches_the_implementation(self):
        rows = [("DEFINES", "lance_graph_x::b", "lance_graph_x::b", "COMPILER", "lib.rs", 16, 26),
                ("DEFINES", "lance_graph_x::c", "lance_graph_x::c", "COMPILER", "lib.rs", 27, 55),
                ("DEFINES", "lance_graph_x::a", "lance_graph_x::a", "COMPILER", "lib.rs", 0, 15),
                ("IMPLEMENTS", "lance_graph_x::b", "lance_graph_x::T::m", "COMPILER", "lib.rs", 16, 26),
                ("MAY_DISPATCH", "lance_graph_x::c", "lance_graph_x::T::m via Clause", "COMPILER", "lib.rs", 46, 49),
                ("CALLS_RESOLVED", "lance_graph_x::a", "lance_graph_x::c", "COMPILER", "lib.rs", 9, 12)]
        st, src = mk_store(self.td.name + "/d", {"lib.rs": LIB}, rows)
        p = packet(Store(st, src), (set(), set(), {}), "lance_graph_x::b")
        self.assertEqual(p["affected_symbols"]["possible"], ["lance_graph_x::a", "lance_graph_x::c"])
        self.assertEqual(p["affected_symbols"]["confirmed"], [])

    def test_13_missing_source_is_never_drawn_as_verified(self):
        (self.src / "lib.rs").unlink()
        out = recover(Store(self.st, self.src), ["lance_graph_x::a"])
        self.assertIn("-. MISSING_SOURCE .->", out)
        self.assertNotIn(" --> ", out)
        self.assertNotIn(":None", out)

    def test_09_lab_proposed_merged_stay_distinct(self):
        corpus = [{"id": f"R{i}", "subject": "s", "object": "o", "status": s, "recoverable": "INTERPRETATION",
                   "verified_line": ""} for i, s in enumerate(["LAB", "PROPOSED", "IMPLEMENTED"])]
        out = reconcile(Store(self.st, self.src), ["lance_graph_x::a"], corpus)
        self.assertIn(":::lab", [ln for ln in out.splitlines() if "R0" in ln][0])
        self.assertIn(":::proposed", [ln for ln in out.splitlines() if "R1" in ln][0])
        self.assertFalse([ln for ln in out.splitlines() if '"R2' in ln or " R2[" in ln])

    def test_10_reconstruction_cannot_pass_by_copying(self):
        # recover never reads the corpus: identical output whatever the corpus says, and an edge that
        # exists only in the corpus is never drawn.
        s = Store(self.st, self.src)
        out = recover(s, ["lance_graph_x::a", "lance_graph_x::c"])
        self.assertNotIn("ghost", out)
        corpus = [{"id": "R1", "subject": "lance_graph_x::a", "object": "ghost", "status": "IMPLEMENTED",
                   "recoverable": "MECHANICAL-CALLGRAPH", "verified_line": "crates/x/src/lib.rs:1"}]
        self.assertEqual(score(s, corpus)[0]["charon"], "UNOBSERVED")
        self.assertEqual(out, recover(Store(self.st, self.src), ["lance_graph_x::c", "lance_graph_x::a"]))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?")
    ap.add_argument("--store"); ap.add_argument("--src"); ap.add_argument("--seeds")
    ap.add_argument("--corpus"); ap.add_argument("--ra"); ap.add_argument("--symbol")
    ap.add_argument("--diff"); ap.add_argument("--rev"); ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        r = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Negatives))
        return 0 if r.wasSuccessful() else 1
    try:
        st = Store(a.store, a.src)
    except ArtifactMissing as e:
        print(e)
        return 3
    seeds = [s for s in Path(a.seeds).read_text().splitlines() if s and not s.startswith("#")] if a.seeds else []
    if a.cmd == "recover":
        sys.stdout.write(recover(st, seeds))
    elif a.cmd == "reconcile":
        sys.stdout.write(reconcile(st, seeds, read_corpus(a.corpus)))
    elif a.cmd == "score":
        ra = ra_calls(a.ra, a.src) if a.ra else None
        res = score(st, read_corpus(a.corpus), ra)
        for r in res:
            print("\t".join(str(r.get(k, "-")) for k in ("id", "site", "charon", "ra")))
        for arm in ("charon", "ra"):
            vals = [r[arm] for r in res if arm in r]
            if vals:
                c = {k: vals.count(k) for k in ("HIT", "MISS", "UNOBSERVED", "NO_SITE")}
                seen = c["HIT"] + c["MISS"]
                print(f"#summary\t{arm}\t" + "\t".join(f"{k}={v}" for k, v in c.items()) +
                      f"\trecall_over_observed={c['HIT']}/{seen}")
    elif a.cmd == "packet":
        diff = Path(a.diff).read_text() if a.diff else None
        json.dump(packet(st, ra_calls(a.ra, a.src), a.symbol, diff, a.rev), sys.stdout, indent=1)
        print()
    else:
        ap.print_help()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
