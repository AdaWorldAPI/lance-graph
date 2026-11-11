#!/usr/bin/env python3
"""Build the ndarray #422 admission-review input from tools, never from the PR's conclusions.

  python3 build_q8_evidence.py --ndarray DIR --base SHA --head SHA \
      --store FOUNDRY_STORE --lg-src LANCE_GRAPH_ROOT --lg-rev EXTRACTION_SHA [--lg-head SHA] > review-input.json

Question: does the per-row quotient specialisation proposed in ndarray #422 reach the executable
Foundry consumer it names (lance-graph `foundry-parity-lab`, G1 capacity and G6 `CanCook`)?

Facts come from: `git diff --name-only` and the PR head's `Cargo.toml` (what changed, in which
target kind); a Charon → ruff_charon store of `foundry-parity-lab`'s library target (calls,
dispatch, closures, and the division/overflow assertions Charon emits at each integer division).
The only human input is ENTRY_POINTS: the consumer functions the PR's spec names, labelled as such.
"""

import argparse
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

# Named by the PR's spec (D-QUOT8-ADMISSION-0 §16, commit 3ef39dd), not derived here.
ENTRY_POINTS = {
    "G1 capacity (arm_b)": ["arm_b::{ItemQuery<'s>}::max_quantities", "arm_b::{ItemQuery<'s>}::network_capacity"],
    "G6 CanCook (method)": ["method::execute_can_cook", "method::{PreparedMethod<'s>}::prepare",
                            "method::{PreparedMethod<'s>}::run"],
}
# Where G6's methods are built: registration, not the execution entry points above.
REGISTRATION = ["method::methods"]
REFERENCE_ARMS = ["oracle::{Oracle}::network_capacity", "arm_a::{Objects}::network_capacity"]
CRATE = "foundry_parity_lab"
PLAN = ".claude/plans/u64x64-llvm-probe-v1.md"
DIV = ("DivisionByZero", "Overflow:Div")
NL = b"\n"


def fnv1a64(data):
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def git(d, *a):
    return subprocess.run(["git", "-C", d, *a], capture_output=True, text=True, check=True).stdout


def rows(path):
    cols, out = None, []
    for ln in Path(path).read_text().splitlines():
        if ln.startswith("#schema "):
            cols = ln[8:].split("\t")
        elif ln and not ln.startswith("#"):
            out.append(dict(zip(cols, ln.split("\t"))))
    return out


def locate(text, needle):
    """1-based line of the single occurrence of `needle` in `text`; refuses 0 or 2+ hits."""
    hits = [i + 1 for i, ln in enumerate(text.splitlines()) if needle in ln]
    if len(hits) != 1:
        sys.exit(f"expected exactly one line containing {needle!r}, found {hits}")
    return hits[0]


def line_of(root, f, off):
    return Path(root, f).read_bytes()[: int(off)].count(b"\n") + 1


def main(argv):
    ap = argparse.ArgumentParser()
    for a in ("--ndarray", "--base", "--head", "--store", "--lg-src", "--lg-rev"):
        ap.add_argument(a, required=True)
    # --lg-rev is the revision the store was extracted at; --lg-head, when given, is a later
    # lance-graph head the review is about. The two consumer crates must be unchanged between them.
    ap.add_argument("--lg-head")
    a = ap.parse_args(argv)
    nd, lg_root = a.ndarray, Path(a.lg_src)
    lg_src = lg_root / "crates/foundry-parity-lab"
    risc = lg_root / "crates/lance-graph-mask-risc/src"
    # The witnesses carry --lg-rev, so both inputs must be at it: the checkout (committed and
    # uncommitted state of the two consumer crates), and the store, whose file list records the
    # hash of every source file it was extracted from.
    consumer = ["crates/foundry-parity-lab", "crates/lance-graph-mask-risc"]
    if subprocess.run(["git", "-C", str(lg_root), "diff", "--quiet", a.lg_rev, "--", *consumer]).returncode != 0:
        sys.exit(f"--lg-src differs from {a.lg_rev} in the consumer crates (other revision or local changes)")
    # `git diff` does not see untracked files; a new source file would be invisible to it.
    untracked = git(str(lg_root), "status", "--porcelain", "--untracked-files=all", "--", *consumer).strip()
    if untracked:
        sys.exit(f"--lg-src has uncommitted or untracked files in the consumer crates:\n{untracked}")
    listed = rows(f"{a.store}/{CRATE}.files.tsv")
    if not listed:
        sys.exit("the store has no file list; its extraction revision cannot be checked")
    for f in listed:
        src = lg_src / f["path"]
        if not src.is_file() or fnv1a64(src.read_bytes()) != f["fnv64"]:
            sys.exit(f"the store was not extracted from this source: {f['path']} differs")
    # The file list is hashed, but rows and effects are separate files and could come from another
    # extraction. Bind them to the hashed sources by content: every row or effect with a span must
    # name a listed file and fit inside it, a definition's span must contain its own name (compiler-generated items excepted), and a
    # division effect's span must contain a division.
    hashed = {f["path"]: (lg_src / f["path"]).read_bytes() for f in listed}
    for kind, name in (("rows", f"{CRATE}.rows.tsv"), ("effects", f"{CRATE}.effects.tsv")):
        for r in rows(f"{a.store}/{name}"):
            if r.get("file") in (None, "-") or r.get("start") in (None, "-"):
                continue
            src = hashed.get(r["file"])
            if src is None:
                # Build-script outputs are recorded by absolute path and are not hashed sources.
                if Path(r["file"]).is_absolute():
                    continue
                sys.exit(f"the store's {kind} name a file its file list does not: {r['file']}")
            lo, hi = int(r["start"]), int(r["end"])
            if not 0 <= lo <= hi <= len(src):
                sys.exit(f"the store's {kind} do not fit {r['file']}: span {lo}..{hi} of {len(src)} bytes")
            text = src[lo:hi]
            # Compiler-generated items (derives, drop glue, closures) have no name in their span.
            generated = "{" in r["subject"] or any(seg.startswith("impl_") for seg in r["subject"].split("::"))
            if kind == "rows" and r["predicate"] == "DEFINES" and not generated:
                leaf = r["subject"].rsplit("::", 1)[-1].encode()
                if leaf not in text:
                    sys.exit(f"the store's rows do not match the source: {r['subject']} is not defined at {r['file']}:{lo}")
            if kind == "effects" and r["kind"] == "DivisionByZero" and b"/" not in text:
                sys.exit(f"the store's effects do not match the source: no division at {r['file']}:{lo}")
    same_at_head = None
    if a.lg_head:
        same_at_head = subprocess.run(["git", "-C", str(lg_root), "diff", "--quiet", a.lg_rev, a.lg_head, "--",
                                       "crates/foundry-parity-lab", "crates/lance-graph-mask-risc"]).returncode == 0
        if not same_at_head:
            sys.exit("consumer crates changed between the extraction revision and --lg-head; re-extract")

    # 1. What #422 changed, and in which Cargo target.
    changed = sorted(git(nd, "diff", "--name-only", a.base, a.head).split())
    cargo = git(nd, "show", f"{a.head}:Cargo.toml")
    examples = {m.group(2): m.group(1) for m in re.finditer(r'\[\[example\]\]\s*name\s*=\s*"([^"]+)"\s*path\s*=\s*"([^"]+)"', cargo)}
    example_dirs = {str(Path(p).parent) + "/": n for p, n in examples.items()}
    rust = [p for p in changed if p.endswith(".rs")]
    in_example = {p: next((n for d, n in example_dirs.items() if p.startswith(d)), None) for p in rust}
    lib_rust = [p for p, n in in_example.items() if n is None and p.startswith("src/")]
    other_rust = [p for p, n in in_example.items() if n is None and not p.startswith("src/")]

    plan = git(nd, "show", f"{a.head}:{PLAN}")

    # 2. The consumer's compiler facts.
    R = rows(f"{a.store}/{CRATE}.rows.tsv")
    E = rows(f"{a.store}/{CRATE}.effects.tsv")
    residuals = [r for r in rows(f"{a.store}/{CRATE}.residuals.tsv")]
    defs = {r["subject"] for r in R if r["predicate"] == "DEFINES"}
    callees, may, impls = collections.defaultdict(set), collections.defaultdict(set), collections.defaultdict(set)
    for r in R:
        if r["predicate"] == "CALLS_RESOLVED":
            callees[r["subject"]].add(r["object"])
        elif r["predicate"] == "MAY_DISPATCH":
            may[r["subject"]].add(r["object"].split(" via ")[0])
        elif r["predicate"] == "IMPLEMENTS":
            impls[r["object"]].add(r["subject"])
    owned = collections.defaultdict(set)  # closure bodies Charon names under their parent
    for d in defs:
        m = re.match(r"^(.*?)::\{impl Fn(?:Mut|Once)?<.*\{closure", d)
        if not m:
            continue
        parent = m.group(1)
        while parent not in defs and parent.endswith("::{closure}"):  # a nested closure: its named owner
            parent = parent[: -len("::{closure}")]
        if parent in defs:
            owned[parent].add(d)
    divs = collections.defaultdict(list)
    for e in E:
        if e["kind"] in DIV:
            ln = line_of(lg_src, e["file"], e["start"])
            text = Path(lg_src, e["file"]).read_bytes()[int(e["start"]):int(e["end"])].decode()
            site = (f"crates/foundry-parity-lab/{e['file']}:{ln}", text)
            if site not in divs[e["subject"]]:  # one site carries both a zero and an overflow assertion
                divs[e["subject"]].append(site)
    src_defs = {r["subject"]: r for r in R if r["predicate"] == "DEFINES"}

    def thresholds(sym):
        """Mask-predicate threshold operands (`t: …`) inside `sym`'s source span, with whether
        each names a lane. Source structure, read mechanically from the span Charon gave."""
        d = src_defs[sym]
        body = Path(lg_src, d["file"]).read_bytes()
        out = []
        span = body[int(d["start"]):int(d["end"])]
        pats = (rb"\bt:\s*([^,\n]+),", rb"SetExpr::i32\([^,]+,\s*Cmp::\w+,\s*([^)\n]+)\)")
        for pat in pats:
            for m in re.finditer(pat, span):
                line = body[: int(d["start"]) + m.start()].count(NL) + 1
                out.append((f"crates/foundry-parity-lab/{d['file']}:{line}",
                            m.group(1).decode().strip(), b"lane" in m.group(1)))
        return sorted(out, key=lambda x: int(x[0].rsplit(":", 1)[1]))

    def reach(root):
        # Visited per (symbol, possible?) so a dispatch or closure visit cannot hide a later
        # resolved-call path. A symbol is possible-only when no definite path reaches it.
        visited, stack = {(root, False)}, [(root, False)]
        while stack:
            x, p = stack.pop()
            nxt = [(c, p) for c in callees.get(x, ())]
            nxt += [(i, True) for t in may.get(x, ()) for i in impls.get(t, ())]
            nxt += [(c, True) for c in owned.get(x, ())]
            for state in nxt:
                if state not in visited:
                    visited.add(state)
                    stack.append(state)
        definite = {x for x, p in visited if not p}
        seen = {x for x, _ in visited}
        return seen, seen - definite

    def short(s):
        return s.split("::", 1)[1].split("::{impl")[0]

    lg = "AdaWorldAPI/lance-graph"
    w_comp = lambda st, wit, q=None: {"grade": "COMPILER_DERIVED", "qualifier": q, "statement": st, "witness": wit,
                                      "repo": lg, "revision": a.lg_rev, "tool": "charon c69e231 + ruff_charon"}
    w_src = lambda st, wit: {"grade": "SOURCE_STRUCTURE", "statement": st, "witness": wit, "repo": "AdaWorldAPI/ndarray",
                             "revision": a.head, "tool": "git"}

    ev, exec_divs, confirmed, possible, missing_entry = [], set(), [], [], []
    ev.append(w_src(f"#422 changes {len(changed)} files: {len(rust)} Rust, {sum(1 for n in in_example.values() if n)} in example target(s) "
                    f"{sorted({n for n in in_example.values() if n})}; library Rust files changed: {len(lib_rust)}; "
                    f"other Rust files changed: {len(other_rust)}",
                    f"{a.base[:12]}..{a.head[:12]}"))
    for label, roots in ENTRY_POINTS.items():
        for root in roots:
            full = f"{CRATE}::{root}"
            if full not in defs:
                ev.append(w_comp(f"{label}: `{root}` is not in the store", "-"))
                missing_entry.append(f"{label}: {root}")
                continue
            seen, poss = reach(full)
            hits = sorted(x for x in seen if x in divs)
            confirmed.append(f"{label}: {short(full)}")
            ext = sorted({x.split("::")[0] for x in seen if not x.startswith(CRATE)} - {"core", "alloc", "std"})
            ev.append(w_comp(f"{label}: `{short(full)}` reaches {len(seen)} functions ({len(poss)} only via dispatch or a closure body); "
                             f"divisions reached: {[short(h) for h in hits] or 'none'}; other crates reached: {ext}", "store"))
            exec_divs.update(hits)
    for s in sorted(exec_divs):
        for wit, text in divs[s]:
            ev.append(w_comp(f"integer division on an executable path, in `{short(s)}`: `{text}`", wit))
    threshold_rows = {s: thresholds(s) for s in sorted(exec_divs)}
    lg_src_w = lambda st, wit: {"grade": "SOURCE_STRUCTURE", "statement": st, "witness": wit, "repo": lg,
                                "revision": a.lg_rev, "tool": "source span read"}
    for s, ts in threshold_rows.items():
        for wit, t, names_lane in ts:
            ev.append(lg_src_w(f"mask-predicate threshold in `{short(s)}`: `t: {t}` ({'names a lane' if names_lane else 'names no lane'})", wit))
    reg_divs = set()
    for root in REGISTRATION:
        full = f"{CRATE}::{root}"
        if full not in defs:
            sys.exit(f"registration root {root} is not in the store")
        seen, _ = reach(full)
        reg_divs.update(x for x in seen if x in divs and x not in exec_divs)
    for s in sorted(reg_divs):
        for wit, text in divs[s]:
            ev.append(w_comp(f"integer division at G6 method registration (`{short(s)}`, reached from "
                             f"{[r for r in REGISTRATION]}, not from the G6 execution entry points): `{text}`", wit))
        for wit, t, names_lane in thresholds(s):
            ev.append(lg_src_w(f"mask-predicate threshold in `{short(s)}`: `{t}` ({'names a lane' if names_lane else 'names no lane'})", wit))
    for root in REFERENCE_ARMS:
        full = f"{CRATE}::{root}"
        seen, poss = reach(full)
        for h in sorted(x for x in seen if x in divs):
            how = ("through a closure body linked to its parent by Charon's name only" if h in poss
                   else "through resolved calls")
            for wit, text in divs[h]:
                ev.append(w_comp(f"reference arm `{short(full)}` reaches `{short(h)}` {how}: `{text}`", wit,
                                 "POSSIBLE" if h in poss else None))
                possible.append(f"{short(full)} → {short(h)}")
    # How mask-risc evaluates the `GeI32` predicate the G1/G6 thresholds feed. Not in the Charon
    # store, so read as source at the extraction revision; the kernel is read at #422's head.
    ir, ex = (risc / "ir.rs").read_text(), (risc / "exec.rs").read_text()
    kern = git(nd, "show", f"{a.head}:src/simd_masking_ops.rs")
    risc_w = lambda st, f, needle: lg_src_w(st, f"crates/lance-graph-mask-risc/src/{f}:{locate(ir if f == 'ir.rs' else ex, needle)}")
    ev.append(risc_w("mask-risc `Pred::GeI32 { lane: u16, t: i32 }` is `lane[i] >= t` with `t` a scalar `i32`",
                     "ir.rs", "GeI32 { lane: u16, t: i32 }"))
    ev.append(risc_w("the executor passes `t` unchanged to `ndarray::simd::ge_i32_to_mask`",
                     "exec.rs", "(Pred::GeI32 { lane, t: v }, None) => ge_i32_to_mask"))
    ev.append(w_src("`ge_i32_to_mask` is the complement of `lt_i32_to_mask`, a compare of each lane value against "
                    "the splatted threshold; no division",
                    f"src/simd_masking_ops.rs:{locate(kern, 'pub fn ge_i32_to_mask(')}"))
    unknown = [{"what": f"entry point {m}", "why": "not in the store; its path was not examined"} for m in missing_entry]
    unknown += [{"what": "lance-graph-mask-risc beyond `GeI32`", "why": "not in the Charon store; only the `GeI32` evaluation path was read, as source"},
               {"what": "G1 per-store min/argmin (limiting resource)", "why": "not shipped: open in EXO-227 (P1b); this review covers shipped code only"},
               {"what": "production recipe divisor distribution", "why": "no production source; the only population is the synthetic fixture"},
               {"what": f"{sum(len(v) for v in owned.values())} closure bodies in the crate", "why": "linked to their parent by Charon's naming only (POSSIBLE), not by a call row"}]
    if residuals:
        unknown.append({"what": f"{len(residuals)} residual facts", "why": "ruff_charon could not represent them"})

    stock_quotient_on_exec = any("/ d" in t for s in exec_divs for _, t in divs[s])
    # Every executable division must sit in a threshold operand that names no lane.
    def in_scalar_threshold(s, text):
        """The division is a lane-free threshold operand, directly or through one `let NAME = <division>`."""
        d = src_defs[s]
        span = Path(lg_src, d["file"]).read_bytes()[int(d["start"]):int(d["end"])].decode()
        m = re.search(r"let\s+(\w+)\s*=\s*" + re.escape(text), span)
        names = {text} | ({m.group(1)} if m else set())
        return any(not lane and any(n in t for n in names) for _, t, lane in thresholds(s))
    div_in_scalar_threshold = all(in_scalar_threshold(s, text) for s in exec_divs for _, text in divs[s])
    reg_in_scalar_threshold = all(in_scalar_threshold(s, text) for s in reg_divs for _, text in divs[s])
    g6_compares_dq = any(t.replace(" ", "") == "d*q" and not lane for s in reg_divs for _, t, lane in thresholds(s))
    findings = [{
        "id": "Q1", "kind": "admission",
        "location": {"file": "examples/u64x64/route_c.rs", "symbol": "u64x64 route C / A power-of-two quotient arms"},
        "claim": (("In the consumer's shipped code, the per-row power-of-two quotient specialisation in #422 has no "
                   "call site to reach. #422's Rust changes are confined to the `u64x64` example target, which no crate "
                   "can depend on. The G1 and G6 paths in the Charon store reach no per-row stock quotient: their "
                   "mask predicates compare a stock lane against a scalar threshold"
                   + (" (G6: `d * q`, built at method registration)" if g6_compares_dq else "")
                   + ", and mask-risc evaluates `GeI32` as a plain compare. Not covered: mask-risc predicates other "
                   "than `GeI32`, and G1's unshipped per-store min/argmin step (EXO-227 P1b).")
                  if not stock_quotient_on_exec and not lib_rust and not other_rust and not missing_entry else
                  "Not established: "
                  + "; ".join(x for x in (
                      "a stock quotient is on an executable path" if stock_quotient_on_exec else "",
                      f"library Rust changed ({len(lib_rust)} files)" if lib_rust else "",
                      f"Rust changed outside the library and the example targets: {other_rust}" if other_rust else "",
                      f"entry points missing from the store, so their paths were not examined: {missing_entry}" if missing_entry else "",
                  ) if x) + ". See evidence."),
        "evidence": ev,
        "reachability": {"confirmed": confirmed, "possible": possible, "unknown": unknown},
        "coverage": {"consumer_store": f"{CRATE} library target", "consumer_residuals": len(residuals),
                     "ndarray_library_rust_changed": len(lib_rust), "ndarray_other_rust_outside_examples": len(other_rust)},
        "falsifier": {"description": ("Run Charon on lance-graph-mask-risc and re-run the reach from the G1/G6 entry points "
                                      "into it: any `DivisionByZero` assertion on a path that takes a stock lane as an "
                                      "operand refutes the claim. Smallest next verification: the predicate kinds other "
                                      "than `GeI32` that the consumer builds."),
                      "status": "NOT_RUN"},
        "suggestion": {"refusal": "an admission finding, not a defect: no code change follows from the evidence"},
    }]
    if exec_divs:
        sites = [w for s in sorted(exec_divs) for w, _ in divs[s]]
        th = [lg_src_w(f"`t: {t}` ({'names a lane' if lane else 'names no lane'})", wit)
              for s in sorted(exec_divs) for wit, t, lane in threshold_rows[s] if "/" in t]
        findings.append({
            "id": "Q2", "kind": "research",
            "location": {"file": "crates/foundry-parity-lab/src/arm_b.rs", "symbol": "arm_b::{ItemQuery}::cook_ops"},
            "claim": (("The executable path does evaluate integer division: `(q + per_staff - 1) / per_staff`. It is the "
                       "threshold operand of a mask predicate and names no lane, so it is computed once when the predicate is "
                       "built, not per row. G6 has the same ceiling division at method registration"
                       + ("" if reg_in_scalar_threshold else " (not confirmed as a threshold operand)")
                       + ". In shipped code, a per-row quotient specialisation has nothing to apply to.")
                      if div_in_scalar_threshold else
                      "The executable path evaluates integer division outside a scalar threshold; see evidence."),
            "evidence": [w_comp("division assertion (zero and overflow) at this site", w) for w in sites]
                        + [w_comp("division assertion at G6 method registration", w) for s in sorted(reg_divs) for w, _ in divs[s]]
                        + th,
            "reachability": {"confirmed": [short(s) for s in sorted(exec_divs | reg_divs)]},
            "falsifier": {"description": "Read the Pred::GeI32 threshold sites: the division's operands are query parameters, not lanes.",
                          "status": "NOT_RUN"},
            "suggestion": {"file": f"{PLAN} (ndarray)", "line": locate(plan, "quotient is evaluated on the executable path"),
                           "text": ("Narrow 'No quotient is evaluated on the executable path' to 'no per-row stock "
                                    "quotient is evaluated': a scalar ceiling division builds the staff threshold."),
                           "base_sha": a.head},
        })
    findings.append({
        "id": "Q3", "kind": "research",
        "location": {"file": "examples/u64x64/main.rs"},
        "claim": "The PR's benchmark results (power-of-two shift no-regret; best route only on AVX-512 hot) were not re-run here.",
        "evidence": [{"grade": "ASSERTED", "statement": "benchmark results and disable runs, author-reported (plan §17)",
                      "witness": f"{PLAN}:{locate(plan, '## 17. Results, D-QUOT8-ADMISSION-0')}",
                      "repo": "AdaWorldAPI/ndarray", "revision": a.head}],
        "falsifier": {"description": ("Per tier T in v2/v3/v4, build with `env -u RUSTFLAGS cargo --config .cargo/config-$T.toml "
                                      "rustc --release --example u64x64` (plan §4) and run `u64x64 bench <scenario>` for the "
                                      "§16 scenarios; compare with §17. The F7/F8 disable arms (test `d > 0` instead of "
                                      "`is_power_of_two`) are parity checks: they must turn `u64x64 check` red, and say "
                                      "nothing about speed."),
                      "status": "NOT_RUN"},
        "suggestion": {"refusal": "a performance claim outside this review's evidence; no defect follows"},
    })
    review = {"review": {"repo": "AdaWorldAPI/ndarray", "pr": 422, "base_sha": a.base, "head_sha": a.head,
                         "heads": {lg: a.lg_rev},
                         "scope": ("PR #422 commits; consumer: lance-graph foundry-parity-lab (lib), extracted at "
                                   f"{a.lg_rev}" + (f"; foundry-parity-lab and mask-risc unchanged at {a.lg_head}"
                                                    if same_at_head else "")),
                         "reviewer": {"tool": "crai_report.py + build_q8_evidence.py"}},
              "run": {"status": "PARTIAL", "reasons": ["mask-risc not in the Charon store (GeI32 read as source only)",
                                                         "no test or benchmark executed"]},
              "findings": findings}
    json.dump(review, sys.stdout, indent=1, sort_keys=True)
    print()


if __name__ == "__main__":
    main(sys.argv[1:])
