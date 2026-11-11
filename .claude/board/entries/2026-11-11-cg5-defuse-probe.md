# 2026-11-11 — CG-5: def-use rows classify division sites; loop nesting does not

**Status:** MEASURED (probe). `.claude/probes/2026-11-11-cg5-defuse/`, `crates/cg-defuse-probe`. Board: EXO-237.

Def-use rows from Charon LLBC (783 rows, 16 `Div`/`Rem` sites in `foundry-parity-lab`) were closed
three ways: egglog in proofs mode, a plain fixpoint, and a source-only scan. All three give the
answer fixed beforehand on EXO-237: `method.rs:58` and `arm_b.rs:245` are scalar thresholds,
`arm_a.rs:75` and `oracle.rs:62` are per-element quotients. egglog and the fixpoint agree on all 63
sites of the four fixtures.

**Finding:** loop nesting of the LLBC statement is wrong on `arm_b.rs:245` (Charon copies the tail
of `cook_ops` into the loop's exit paths) and on mutant m3 (scalar division moved into the loop).
Operand provenance is right on both.

Mutants: division by the loop element flips the class; removing the division removes the site.
24 disable runs, 24 red.

Cost, reported and not a gate: egglog 1.65 to 2.10 s cold per crate with proofs; plain fixpoint
about 1 ms. egglog adds the proof, not the answer.

OPEN: def-use rows as a store row kind (CG-3 format decision, EXO-235); interprocedural provenance.
