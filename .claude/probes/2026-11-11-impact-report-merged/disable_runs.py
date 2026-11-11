#!/usr/bin/env python3
"""Disable runs for crates/lance-graph-codegraph/src/impact_report.rs.

Each guard is patched out (the anchor must occur exactly once, or the run is reported as
ANCHOR-MISSING and counts as a failure), `cargo test --release --test impact_coverage` is run, and
the file is restored. A guard is load-bearing when its run is RED.

Run from the repository root on a clean tree:  python3 <this file>
Exit 0 only when every guard is RED. Writes nothing but the temporary patch.
"""
import re, subprocess, sys
from pathlib import Path

CRATE = Path(__file__).resolve().parents[3] / "crates/lance-graph-codegraph"
P = CRATE / "src/impact_report.rs"
TEST = "impact_coverage"
G=[
 ("has_rows", ".filter(|_| self.has_rows(symbol))", ""),
 ("undeclared-involves-all", "                    // Any stored crate may be a consumer too, so each one's coverage counts.\n                    involved.extend(stored.iter().map(|k| (*k).to_owned()));\n", ""),
 ("consumers-not-in-store", "if !absent.is_empty() {", "if false {"),
 ("declared-consumer-involved", "                        involved.insert(d.clone());", "                        let _ = d;"),
 ("targets-not-extracted", "if missing.any() {", "if false {"),
 ("targets-undeclared", "        None => gaps.push(Gap::TargetsUndeclared {\n            krate: k.to_owned(),\n        }),", "        None => {}"),
 ("regions-excluded", ".find(|(n, r)| n == k && !r.is_empty())", ".find(|(n, r)| n == k && r.is_empty() && !r.is_empty())"),
 ("trait-dispatch", "if possible > 0 {", "if false {"),
 ("dynamic-unknown", "if unknown > 0 || (store_dynamic_calls > 0 && self.dynamic_may_reach(g)) {", "if store_dynamic_calls > 0 && self.dynamic_may_reach(g) && unknown > u32::MAX - 1 {"),
 ("dynamic-may-reach", "if unknown > 0 || (store_dynamic_calls > 0 && self.dynamic_may_reach(g)) {", "if unknown > 0 {"),
 ("public-caller-frontier", ".any(|m| m.iter().zip(self.public()).any(|(a, b)| a & b != 0));", ".any(|m| m.is_empty() && !m.is_empty());"),
 ("public-symbol-frontier", "let frontier_public = has(self.public(), g)\n            ||", "let frontier_public = false\n            ||"),
 ("all-in-store", "            Consumers::AllInStore => {\n                if frontier_public {", "            Consumers::AllInStore => {\n                if false {"),
 ("harvest-residuals", "if residuals.total() > 0 || no_residuals_file || partial {", "if no_residuals_file || partial {"),
 ("harvest-no-file", "if residuals.total() > 0 || no_residuals_file || partial {", "if residuals.total() > 0 || partial {"),
 ("harvest-partial", "if residuals.total() > 0 || no_residuals_file || partial {", "if residuals.total() > 0 || no_residuals_file {"),
 ("missing-symbol-gaps", "            if owner_stored {\n                scope_gaps(&owner, scope, &mut gaps);", "            if false {\n                scope_gaps(&owner, scope, &mut gaps);"),
 ("stale", "if !f.stale.is_empty() {", "if false {"),
 ("missing-file", "if !f.missing.is_empty() {", "if false {"),
 ("unread", "if !f.unread.is_empty() {", "if false {"),
 ("invalid-row", "        let (Some(path), Some(want)) = (fields.get(path_col), fields.get(hash_col)) else {\n            out.invalid = true;", "        let (Some(path), Some(want)) = (fields.get(path_col), fields.get(hash_col)) else {\n            out.invalid = false;"),
 ("files-unrecorded", "gaps.push(Gap::FilesUnrecorded { krate: k.clone() });", "let _ = k;"),
 ("referrers-gap", "        if referrers > 0 {", "        if false {"),
 ("referrers-only-outside", "            if object_in && !subject_in {", "            if object_in {"),
 ("referrers-only-into-answer", "            if object_in && !subject_in {", "            if !subject_in {"),
 ("referrers-only-references", "            if l.pred[i] != P_REFERENCES {\n                continue;\n            }\n", ""),
 ("freshness-unchecked", "        gaps.push(Gap::FreshnessUnchecked);\n        return;", "        return;"),
]

def main():
    if subprocess.run(["git", "diff", "--quiet", "--", str(P)]).returncode != 0:
        sys.exit(f"{P} has local changes; commit first (the restore would not be exact)")
    orig = P.read_text()
    bad = 0
    try:
        for name, old, new in G:
            if orig.count(old) != 1:
                print(f"{name}\tANCHOR-MISSING({orig.count(old)})")
                bad += 1
                continue
            P.write_text(orig.replace(old, new))
            out = subprocess.run(
                ["cargo", "test", "--release", "--test", TEST],
                cwd=CRATE, capture_output=True, text=True,
            )
            text = out.stdout + out.stderr
            m = re.search(r"test result: (\w+)\. (\d+) passed; (\d+) failed", text)
            failed = sorted(set(re.findall(r"^test (\S+) \.\.\. FAILED", text, re.M)))
            if not m:
                print(f"{name}\tBUILD-ERROR")
                bad += 1
            elif m.group(1) == "FAILED":
                print(f"{name}\tRED\t{','.join(f.split('::')[-1] for f in failed)}")
            else:
                print(f"{name}\tGREEN\tno test depends on this guard")
                bad += 1
    finally:
        P.write_text(orig)
    print(f"guards\t{len(G)}\tnot-red\t{bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
