#!/usr/bin/env python3
"""Build the ndarray #422 admission-review input from tools, never from the PR's conclusions.

  python3 build_q8_evidence.py --ndarray DIR --base SHA --head SHA \
      --store FOUNDRY_STORE --lg-src LANCE_GRAPH_ROOT --lg-rev SHA > review-ndarray-422.json

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
REFERENCE_ARMS = ["oracle::{Oracle}::network_capacity", "arm_a::{Objects}::network_capacity"]
CRATE = "foundry_parity_lab"
DIV = ("DivisionByZero", "Overflow:Div")


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


def line_of(root, f, off):
    return Path(root, f).read_bytes()[: int(off)].count(b"\n") + 1


def main(argv):
    ap = argparse.ArgumentParser()
    for a in ("--ndarray", "--base", "--head", "--store", "--lg-src", "--lg-rev"):
        ap.add_argument(a, required=True)
    a = ap.parse_args(argv)
    nd, lg_src = a.ndarray, Path(a.lg_src) / "crates/foundry-parity-lab"

    # 1. What #422 changed, and in which Cargo target.
    changed = sorted(git(nd, "diff", "--name-only", a.base, a.head).split())
    cargo = git(nd, "show", f"{a.head}:Cargo.toml")
    examples = {m.group(2): m.group(1) for m in re.finditer(r'\[\[example\]\]\s*name\s*=\s*"([^"]+)"\s*path\s*=\s*"([^"]+)"', cargo)}
    example_dirs = {str(Path(p).parent) + "/": n for p, n in examples.items()}
    rust = [p for p in changed if p.endswith(".rs")]
    in_example = {p: next((n for d, n in example_dirs.items() if p.startswith(d)), None) for p in rust}
    lib_rust = [p for p, n in in_example.items() if n is None and p.startswith("src/")]
    other_rust = [p for p, n in in_example.items() if n is None and not p.startswith("src/")]

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
        if m and m.group(1) in defs:
            owned[m.group(1)].add(d)
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
        for m in re.finditer(rb"\bt:\s*([^,\n]+),", body[int(d["start"]):int(d["end"])]):
            off = int(d["start"]) + m.start()
            out.append((f"crates/foundry-parity-lab/{d['file']}:{body[:off].count(b'\n') + 1}",
                        m.group(1).decode().strip(), b"lane" in m.group(1)))
        return out

    def reach(root):
        seen, stack, poss = set(), [(root, False)], set()
        while stack:
            x, p = stack.pop()
            nxt = [(c, p) for c in callees.get(x, ())]
            nxt += [(i, True) for t in may.get(x, ()) for i in impls.get(t, ())]
            nxt += [(c, True) for c in owned.get(x, ())]
            for c, pp in nxt:
                if c not in seen:
                    seen.add(c)
                    if pp:
                        poss.add(c)
                    stack.append((c, pp))
        return seen | {root}, poss

    def short(s):
        return s.split("::", 1)[1].split("::{impl")[0]

    lg = "AdaWorldAPI/lance-graph"
    w_comp = lambda st, wit, q=None: {"grade": "COMPILER_DERIVED", "qualifier": q, "statement": st, "witness": wit,
                                      "repo": lg, "revision": a.lg_rev, "tool": "charon c69e231 + ruff_charon"}
    w_src = lambda st, wit: {"grade": "SOURCE_STRUCTURE", "statement": st, "witness": wit, "repo": "AdaWorldAPI/ndarray",
                             "revision": a.head, "tool": "git"}

    ev, exec_divs, confirmed, possible = [], set(), [], []
    ev.append(w_src(f"#422 changes {len(changed)} files: {len(rust)} Rust, all in example target(s) "
                    f"{sorted({n for n in in_example.values() if n})}; library Rust files changed: {len(lib_rust)}",
                    f"{a.base[:12]}..{a.head[:12]}"))
    for label, roots in ENTRY_POINTS.items():
        for root in roots:
            full = f"{CRATE}::{root}"
            if full not in defs:
                ev.append(w_comp(f"{label}: `{root}` is not in the store", "-"))
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
    for root in REFERENCE_ARMS:
        full = f"{CRATE}::{root}"
        seen, poss = reach(full)
        for h in sorted(x for x in seen if x in divs):
            for wit, text in divs[h]:
                ev.append(w_comp(f"reference arm `{short(full)}` may reach `{short(h)}`: `{text}`", wit,
                                 "POSSIBLE" if h in poss else None))
                possible.append(f"{short(full)} → {short(h)}")
    unknown = [{"what": "lance-graph-mask-risc execution internals", "why": "not in the store; the G1/G6 paths call into it"},
               {"what": "production recipe divisor distribution", "why": "no production source; the only population is the synthetic fixture"},
               {"what": f"{sum(len(v) for v in owned.values())} closure bodies", "why": "linked to their parent by Charon's naming only (POSSIBLE), not by a call row"}]
    if residuals:
        unknown.append({"what": f"{len(residuals)} residual facts", "why": "ruff_charon could not represent them"})

    stock_quotient_on_exec = any("/ d" in t for s in exec_divs for _, t in divs[s])
    # Every executable division must sit in a threshold operand that names no lane.
    div_in_scalar_threshold = all(any(text in t and not lane for _, t, lane in threshold_rows[s]) for s in exec_divs for _, text in divs[s])
    findings = [{
        "id": "Q1", "kind": "admission",
        "location": {"file": "examples/u64x64/route_c.rs", "symbol": "u64x64 route C / A power-of-two quotient arms"},
        "claim": ("The per-row power-of-two quotient specialisation in #422 does not reach the named Foundry consumer. "
                  "#422's Rust changes are confined to the `u64x64` example target, which no crate can depend on. "
                  "The consumer's executable G1 and G6 paths evaluate no per-row stock quotient: they compare stock "
                  "against a precomputed `d * q`." if not stock_quotient_on_exec and not lib_rust else
                  "The per-row quotient may reach the consumer; see evidence."),
        "evidence": ev,
        "reachability": {"confirmed": confirmed, "possible": possible, "unknown": unknown},
        "coverage": {"consumer_store": f"{CRATE} library target", "consumer_residuals": len(residuals),
                     "ndarray_library_rust_changed": len(lib_rust), "ndarray_other_rust_outside_examples": len(other_rust)},
        "falsifier": {"description": ("Extract lance-graph-mask-risc and re-run the reach: a per-row division reachable from "
                                      "`execute_into` with a stock-lane operand would refute the claim. Independently, "
                                      "make `Pred::GeI32` evaluation panic on any division and run foundry-parity-lab's tests."),
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
                       "built, not per row. A per-row quotient specialisation has nothing to apply to.")
                      if div_in_scalar_threshold else
                      "The executable path evaluates integer division outside a scalar threshold; see evidence."),
            "evidence": [w_comp("division assertion (zero and overflow) at this site", w) for w in sites] + th,
            "reachability": {"confirmed": [short(s) for s in sorted(exec_divs)]},
            "falsifier": {"description": "Read the Pred::GeI32 threshold sites: the division's operands are query parameters, not lanes.",
                          "status": "NOT_RUN"},
            "suggestion": {"text": ("If the spec says 'no quotient is evaluated on the executable path', narrow it to "
                                    "'no per-row stock quotient': a scalar ceiling division builds the staff threshold."),
                           "base_sha": a.head},
        })
    findings.append({
        "id": "Q3", "kind": "research",
        "location": {"file": "examples/u64x64/main.rs"},
        "claim": "The PR's benchmark results (power-of-two shift no-regret; best route only on AVX-512 hot) were not re-run here.",
        "evidence": [{"grade": "ASSERTED", "statement": "benchmark rows in the PR, author-reported", "witness": "examples/u64x64",
                      "repo": "AdaWorldAPI/ndarray", "revision": a.head}],
        "falsifier": {"description": "Re-run `cargo run --release --example u64x64` with the disable arms (F7/F8) the PR names.",
                      "status": "NOT_RUN"},
        "suggestion": {"refusal": "a performance claim outside this review's evidence; no defect follows"},
    })
    review = {"review": {"repo": "AdaWorldAPI/ndarray", "pr": 422, "base_sha": a.base, "head_sha": a.head,
                         "heads": {lg: a.lg_rev}, "scope": "PR #422 commits; consumer: lance-graph foundry-parity-lab (lib)",
                         "reviewer": {"tool": "crai_report.py + build_q8_evidence.py"}},
              "run": {"status": "PARTIAL", "reasons": ["mask-risc internals not extracted", "no test or benchmark executed"]},
              "findings": findings}
    json.dump(review, sys.stdout, indent=1, sort_keys=True)
    print()


if __name__ == "__main__":
    main(sys.argv[1:])
