#!/usr/bin/env python3
"""Disable runs for crates/lance-graph-codegraph/src/effects.rs.

Each guard is patched out (the anchor must occur exactly once, or the run is reported as
ANCHOR-MISSING and counts as a failure), `cargo test --release --test effects` is run, and
the file is restored. A guard is load-bearing when its run is RED.

Run from the repository root on a clean tree:  python3 <this file>
Exit 0 only when every guard is RED. Writes nothing but the temporary patch.
"""
import re, subprocess, sys
from pathlib import Path

CRATE = Path(__file__).resolve().parents[3] / "crates/lance-graph-codegraph"
P = CRATE / "src/effects.rs"
TEST = "effects"
G = [
 ("version", "    if found != 1 {", "    if false {"),
 ("schema", "    if lines.next() != Some(SCHEMA) {", "    if lines.next().is_none() {"),
 ("arity", "let [subject, effect, kind, file, start, end] = f[..] else {", "let [subject, effect, kind, file, start, end, ..] = f[..] else {"),
 ("assert-empty-kind", '("ASSERT", k) if !k.is_empty() =>', '("ASSERT", k) =>'),
 ("site-order", 'if file != "-" && start <= end =>', 'if file != "-" =>'),
 ("site-dash-file", 'if file != "-" && start <= end =>', 'if start <= end =>'),
 ("site-partial-dash", '("-", _, _) if start == "-" && end == "-" => None,', '("-", _, _) => None,'),
 ("crate-not-read", "        if !self.crates.contains(krate) {\n            return EffectAnswer::CrateNotRead;\n        }\n", ""),
 ("none-witnessed", "        if hits.is_empty() {\n            EffectAnswer::NoneWitnessed\n        } else {\n            EffectAnswer::Witnessed(hits)\n        }", "        EffectAnswer::Witnessed(hits)"),
 ("subject-filter", ".filter(|e| e.subject == subject)", ".filter(|e| !e.subject.is_empty() || e.subject == subject)"),
 ("kind-filter", ".filter(move |e| e.kind == *kind)", ".filter(move |e| e.kind == *kind || true)"),
 ("class-division", '"DivisionByZero" | "Overflow:Div" => Some(EffectClass::Division),', '"DivisionByZero" => Some(EffectClass::Division),'),
 ("class-remainder", '"RemainderByZero" | "Overflow:Rem" => Some(EffectClass::Remainder),', '"RemainderByZero" => Some(EffectClass::Remainder),'),
 ("class-shift", '"Overflow:Shl" | "Overflow:Shr" => Some(EffectClass::Shift),', '"Overflow:Shl" => Some(EffectClass::Shift),'),
 ("class-write", "EffectKind::DerefWriteArg | EffectKind::DerefWriteLocal => Some(EffectClass::Write),", "EffectKind::DerefWriteArg => Some(EffectClass::Write),\n            EffectKind::DerefWriteLocal => None,"),
 ("identity-digest", "fnv64: fnv1a(text.as_bytes()),", "fnv64: fnv1a(krate.as_bytes()),"),
 ("identity-rows", "rows: (rows.len() - before) as u32,", "rows: rows.len() as u32 + 1,"),
 ("reach-own", "                        Some(p) if p == me => Some(ReachGrade::Own),\n", ""),
 ("reach-confirmed", "                            if has(&i.confirmed) {\n                                Some(ReachGrade::Confirmed)", "                            if false {\n                                Some(ReachGrade::Confirmed)"),
 ("reach-possible", "} else if has(&i.possible) {\n                                Some(ReachGrade::Possible)", "} else if false {\n                                Some(ReachGrade::Possible)"),
 ("reach-unknown", "} else if has(&i.unknown) {\n                                Some(ReachGrade::Unknown)", "} else if false {\n                                Some(ReachGrade::Unknown)"),
 ("reach-keep", ".filter(|e| keep(&e.kind)) {", ".filter(|e| keep(&e.kind) || true) {"),
 ("reach-no-position", "        let Some(me) = self.pos(symbol) else {\n            return Ok(None);\n        };", "        let me = self.pos(symbol).unwrap_or(0);"),
 ("limit-effects-not-read", "            if effects.identity(&krate).is_none() {", "            if false {"),
 ("limit-harvest", "            if status == RunStatus::Partial || !clean_harvest {", "            if status == RunStatus::Partial {"),
 ("limit-queried-symbol-outside", "        if !me_stored {\n", "        if false {\n"),
 ("limit-outside", "        if !outside.is_empty() {", "        if false {"),
 ("limit-dynamic", "        if reaches_dynamic {", "        if false {"),
 ("limit-dynamic-per-symbol", "                    reaches_dynamic = true;", "                    reaches_dynamic = true;\n                }\n                if true {\n                    reaches_dynamic = true;"),
 ("limit-unplaced", "                            unplaced.insert(subject);\n", "                            unplaced.insert(subject);\n                            unplaced.clear();\n"),
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
