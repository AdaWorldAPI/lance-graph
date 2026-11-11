#!/usr/bin/env python3
"""Disable runs for crates/cg-beh-probe: remove one guard, the named test must fail.

Run from the repository root on a clean tree (the restore is `git checkout`):
    python3 -I .claude/probes/2026-11-11-cg-beh-2/disable_runs.py
A case counts only when the patch applied exactly once, the test binary was
built and ran, and the named test is in its failure list.
"""
import os
import subprocess
import sys

CRATE = "crates/cg-beh-probe"
FACTS, QUERY, DISPATCH = (f"{CRATE}/src/{n}.rs" for n in ("facts", "query", "dispatch"))

CASES = [
    ("issuer check", FACTS,
     "let binding = store.admit(r).map_err(Missing::Admit)?;",
     "let binding = match store.admit(r) { Ok(b) => b, Err(e) => store.resolve(view, r.class, read.schema_rev).ok().and_then(|x| x.methods.iter().find(|m| m.id == r.method)).ok_or(Missing::Admit(e))? };",
     "gate_c::a_reference_from_another_issuer_yields_no_facts"),
    ("guard dependency must be a registered read", FACTS,
     ".filter(|p| binding.reads.has(*p))", "",
     "gate_a::a_guard_dependency_outside_the_registered_reads_stays_unknown"),
    ("source revision required", FACTS,
     'if rev.is_empty() || rev.eq_ignore_ascii_case("unknown") {', "if false {",
     "gate_a::a_snapshot_without_a_source_revision_is_refused"),
    ("lens revision required", FACTS,
     "read.lens_rev.ok_or(Missing::LensRevision { lens })?", "read.lens_rev.unwrap_or(SchemaRev(1))",
     "gate_a::do_steps_are_not_read_without_the_lens_they_were_bound_at"),
    ("basis read through the witnessed resolve", FACTS,
     "    let resolved = store\n        .resolve(view, class, schema_rev)\n        .map_err(Missing::Basis)?;\n    let mut h",
     "    let resolved = store\n        .resolve_algebra(view, class, schema_rev)\n        .or_else(|_| Ok::<_, Missing>(lance_graph_dt64::Resolved { class, schema: Box::leak(Box::new(lance_graph_dt64::ClassSchema { schema_rev, field_count: 0, extent: lance_graph_dt64::dr::Extent::DR64_CUBE })), fields: view.fields(lance_graph_dt64::view_class(class)), methods: &[], derived_from: None }))?;\n    let mut h",
     "gate_c::a_basis_altered_at_the_same_field_count_is_refused_not_read"),
    ("witness, not spelling, decides WHEN", QUERY,
     "if witness(b) != witness(a) {", "if spelling(b) != spelling(a) {",
     "gate_b::a_respelled_guard_keeps_its_witness_and_makes_nothing_stale"),
    ("basis component", QUERY,
     "if basis_b != basis_a {", "if false {",
     "gate_b::a_changed_field_basis_reaches_every_method_of_the_class"),
    ("DO order", QUERY,
     "    m.steps\n        .iter()\n        .map(|s| {",
     "    let mut steps: Vec<_> = m.steps.iter().collect();\n    steps.sort_by_key(|s| s.word);\n    steps\n        .into_iter()\n        .map(|s| {",
     "gate_c::reversing_independent_do_steps_is_a_do_change_and_nothing_else"),
    ("digest moved alone", QUERY,
     "if binding_changed && !covered {", "if false {",
     "gate_c::facts_that_contradict_the_binding_digest_are_refused"),
    ("covered component moved alone", QUERY,
     "if covered && !binding_changed {", "if false {",
     "gate_c::facts_that_contradict_the_binding_digest_are_refused"),
    ("dispatch witness", QUERY,
     "if current != dispatch.file_fnv {", "if false {",
     "gate_c::a_dispatch_reading_from_other_source_text_is_refused"),
    ("arm that names no word", QUERY,
     "if words.is_empty() {", "if false {",
     "gate_c::a_line_outside_the_word_arms_has_no_behaviour_edge"),
    ("no stale parity without a user", QUERY,
     "let stale = if users.is_empty() {", "let stale = if false {",
     "gate_b::an_arm_shared_by_three_opcodes_reaches_all_three_words"),
    ("word without an arm", DISPATCH,
     "self.arms.iter().find(|a| a.names.iter().any(|n| n == name))",
     "self.arms.iter().find(|a| a.names.iter().any(|n| n == name)).or(self.arms.first())",
     "gate_c::a_removed_dispatch_mapping_makes_its_word_unknown_and_no_other"),
    ("ON read from the binding", FACTS,
     'let on = positions(&binding.on)', 'let on = positions(&binding.reads)',
     "gate_a::the_lab_method_is_read_part_by_part"),
]


def run_tests():
    env = dict(os.environ, CARGO_PROFILE_DEV_DEBUG="0", CARGO_PROFILE_TEST_DEBUG="0", CARGO_INCREMENTAL="0")
    env.pop("RUSTFLAGS", None)
    p = subprocess.run(
        ["cargo", "test", "--release", "--manifest-path", f"{CRATE}/Cargo.toml", "--test", "integration"],
        capture_output=True, text=True, env=env)
    return p.returncode, p.stdout + p.stderr


def main():
    if subprocess.run(["git", "status", "--porcelain", CRATE], capture_output=True, text=True).stdout.strip():
        sys.exit("the crate has uncommitted changes; commit before disabling")
    code, out = run_tests()
    if code != 0:
        sys.exit("baseline is not green")
    bad = 0
    for name, path, old, new, test in CASES:
        src = open(path).read()
        if src.count(old) != 1:
            print(f"ANCHOR  {name}: found {src.count(old)} times")
            bad += 1
            continue
        open(path, "w").write(src.replace(old, new))
        try:
            code, out = run_tests()
        finally:
            subprocess.run(["git", "checkout", "--", path], check=True)
        ran = "test result:" in out
        failed = f"test {test} ... FAILED" in out
        verdict = "RED" if (code != 0 and ran and failed) else ("NOT-BUILT" if not ran else "GREEN")
        if verdict != "RED":
            bad += 1
            print(out[-1500:])
        print(f"{verdict:9} {name}  ->  {test}")
    print(f"{len(CASES) - bad}/{len(CASES)} disable runs red")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
