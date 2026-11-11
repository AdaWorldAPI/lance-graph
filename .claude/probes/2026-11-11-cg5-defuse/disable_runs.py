#!/usr/bin/env python3
"""Disable runs for crates/cg-defuse-probe/src/lib.rs.

Each guard is patched out (the anchor must occur exactly once, or the run is reported as
ANCHOR-MISSING and counts as a failure), `cargo test --release` is run in the crate, and
the file is restored. A guard is load-bearing when its run is RED.

Run from the repository root on a clean tree:  python3 <this file>
Exit 0 only when every guard is RED. Writes the receipt next to this file.
"""
import re, subprocess, sys
from pathlib import Path

CRATE = Path(__file__).resolve().parents[3] / "crates/cg-defuse-probe"
P = CRATE / "src/lib.rs"
RECEIPT = Path(__file__).with_name("disable_runs.receipt.tsv")
G = [
 ("egglog-flow-rule", "(rule ((= f (Flow a b)) (= (Elem a) (T)))\n      ((union (Elem b) (T))) :ruleset defuse)\n", ""),
 ("egglog-site-elem-lhs", "(rule ((= s (Site f n a b)) (= (Elem a) (T)))\n      ((union (ClassOf f n) (PerElement))) :ruleset defuse)\n", ""),
 ("egglog-site-elem-rhs", "(rule ((= s (Site f n a b)) (= (Elem b) (T)))\n      ((union (ClassOf f n) (PerElement))) :ruleset defuse)\n", ""),
 ("egglog-site-scalar-both", "(rule ((= s (Site f n a b)) (= (Scalar a) (T)) (= (Scalar b) (T)))", "(rule ((= s (Site f n a b)) (= (Scalar a) (T)))"),
 ("egglog-all-scalar-head", "(rule ((= l (Cons v r)) (= (Scalar v) (T)) (= (AllScalar r) (T)))", "(rule ((= l (Cons v r)) (= (AllScalar r) (T)))"),
 ("egglog-all-scalar-tail", "(rule ((= l (Cons v r)) (= (Scalar v) (T)) (= (AllScalar r) (T)))", "(rule ((= l (Cons v r)) (= (Scalar v) (T)))"),
 ("egglog-const-scalar", "(union (Scalar (K)) (T))\n", ""),
 ("encode-sole-def", "        if defs.len() == 1\n            && !facts.args.contains(&(f.clone(), *l))", "        if !defs.is_empty()\n            && !facts.args.contains(&(f.clone(), *l))"),
 ("encode-elem-not-scalar", "            && !facts.elems.contains(&(f.clone(), *l))\n            && let Some(d)", "            && let Some(d)"),
 ("encode-global-operand", "        Operand::Global => None,\n    }\n}\n\n/// The rows as egglog facts.", "        Operand::Global => Some(\"(K)\".to_string()),\n    }\n}\n\n/// The rows as egglog facts."),
 ("derive-conflict", "    if prove(&mut egraph, \"(= (PerElement) (ScalarThreshold))\").is_some() {", "    if false {"),
 ("derive-fuel", "        if iterations == fuel {", "        if false {"),
 ("fixpoint-flow", ".any(|d| d.2.iter().any(|s| elem.contains(&(func.clone(), *s))))", ".any(|d| d.2.iter().any(|s| false && elem.contains(&(func.clone(), *s))))"),
 ("fixpoint-sole-def", "                && defs.len() == 1\n                && defs\n                    .iter()\n                    .all(", "                && defs\n                    .iter()\n                    .all("),
 ("fixpoint-global", "        Operand::Global => false,", "        Operand::Global => true,"),
 ("parse-version", "        Some((_, \"#version 1\")) => {}\n        _ => return Err(FactsRefusal::Version),", "        Some(_) => {}\n        _ => return Err(FactsRefusal::Version),"),
 ("parse-unknown-kind", "            _ => return Err(FactsRefusal::Row(n)),", "            _ => {}"),
 ("parse-short-site", "                need(9)?;\n", ""),
 ("parse-index-is-elem", "            \"ELEM\" | \"INDEX\" => {", "            \"ELEM\" => {\n                facts.elems.insert((func, num(f[2], n)?));\n            }\n            \"INDEX\" if false => {"),
 ("source-bound-names", "used.iter().any(|w| bound.contains(w))", "used.iter().any(|w| false && bound.contains(w))"),
 ("source-loop-close", "                    if open.last().is_some_and(|(d, _)| *d == depth) {\n                        open.pop();\n                    }\n", ""),
 ("source-index", "    if target.contains('[') || ", "    if "),
 ("source-comment", "    let code = |l: &str| l.split(\"//\").next().unwrap_or(\"\").to_string();", "    let code = |l: &str| l.to_string();"),
 ("source-no-operator", "    if !target.contains('/') && !target.contains('%') {\n        return None;\n    }\n", ""),
]

def main():
    if subprocess.run(["git", "diff", "--quiet", "--", str(P)]).returncode != 0:
        sys.exit(f"{P} has local changes; commit first (the restore would not be exact)")
    orig = P.read_text()
    bad = 0
    lines = ["guard\tresult\tfailing tests"]
    try:
        for name, old, new in G:
            if orig.count(old) != 1:
                lines.append(f"{name}\tANCHOR-MISSING({orig.count(old)})\t")
                bad += 1
                continue
            P.write_text(orig.replace(old, new))
            out = subprocess.run(["cargo", "test", "--release"], cwd=CRATE,
                                 capture_output=True, text=True)
            text = out.stdout + out.stderr
            failed = sorted(set(re.findall(r"^test (\S+) \.\.\. FAILED", text, re.M)))
            if out.returncode == 0:
                res = "GREEN"; bad += 1
            elif failed:
                res = "RED"
            else:
                res = "RED(build)" if "error" in text else "RED(?)"
                bad += 1
            lines.append(f"{name}\t{res}\t{','.join(failed)}")
            print(lines[-1], flush=True)
    finally:
        P.write_text(orig)
    RECEIPT.write_text("\n".join(lines) + f"\n# {len(G)} guards, {bad} not red\n")
    sys.exit(1 if bad else 0)

main()
