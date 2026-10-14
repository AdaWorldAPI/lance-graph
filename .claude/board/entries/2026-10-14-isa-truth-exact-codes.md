# 2026-10-14 — Exact truth codes in `causal_edge::isa::truth` (D-NALP-4 harvest)

**Status:** IN PR (branch `ccr-9f3b8075-74u47l-truth-exact`). Harvested from
the draft #1544, which stays open as a reminder and is not merged.

**DECISION:** the exact per-row truth law lives where NARS truth already
lives, in `causal_edge::isa::truth`, not in a second substrate (a lane
terminal plus `nal::Deduction` in `lance-graph-nal`).

**SCOPE:** deduction, induction, abduction. `truth::code::{deduction,
induction, abduction}` take and return `u8` codes (`x / 255`), compute in
integers and round once, half up. `Opcode::truth_code` routes the three
rules there; `execute` (so `forward` and the named register methods) and
`syllogize` pack its result. Revision and synthesis keep the `f32` path,
unchanged. The `f32` functions stay for real-valued callers.

**MEASURED** (`cargo run --release -p causal-edge --example
truth_exact_probe`, all 2³² inputs per law, 8 threads, 37.7 s):

| law | `truth::code` ≠ oracle | old `f32` path ≠ oracle |
|---|---:|---:|
| deduction | 0 | 2,413 |
| induction | 0 | 4,352 |
| abduction | 0 | 4,352 |

Every old miss is one confidence code. The oracle
(`crates/causal-edge/tests/common/truth_law.rs`) is written independently of
the crate.

**TEST-PINNED** (`tests/ce64_truth_exact.rs`, 8 tests): hand-computed codes;
two half-up ties (127.5 → 128, 110.5 → 111, the second rejects
ties-to-even); the induction/abduction mirror; zero evidence; OpenNARS nal1.1
at `4d1bdb0` (0.81 deduction, 0.45 induction); the codes equal the oracle on
the every-5th grid; `execute` and `syllogize` pack the oracle's codes on
exactly the inputs where the `f32` path was off (12 deduction rows of the
grid, 17 induction and 17 abduction rows of an exhaustive three-code scan;
the grid alone holds none for induction or abduction). Disable runs, each
red: `truth_code` back to `f32` (register test), `syllogize` deduction back
to `f32` (syllogize test), round down, ties-to-even. The legacy golden
capture (`ce64_isa_golden.rs`, 126 words) is unchanged: none of its inputs
is one the `f32` path got wrong.

**OPEN:**
- Revision and synthesis are not exact. Exact revision has to settle the
  both-operands-at-255 case first (today it packs `(0, 0)`, pinned as a known
  defect in `ce64_isa_golden.rs`).
- Whether D-NALP-4/5 still need a lane terminal at all, or whether
  populations of CE64 registers carry the law (plan §4 B/C). Not decided.
- `cargo clippy -D warnings` on `causal-edge` fails on six lints in code this
  change does not touch; CI runs only `cargo test` for the crate.
