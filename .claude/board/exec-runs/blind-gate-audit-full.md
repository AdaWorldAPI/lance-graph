## Blind-gate audit (full examples sweep) — 2026-08-04

- Task: report-only audit of every `crates/*/examples/*.rs` for main-path assertions
  (assert!/assert_eq!/assert_ne!/panic!/unreachable!/process::exit) never executed by CI.
- Method: brace-matched exclusion of `#[cfg(test)] mod {}` blocks + bare `#[test] fn {}`
  bodies (found one file using the bare shape without cfg(test): `probe_sudoku_teacher.rs`),
  then grepped assertion macros in what remains. Cross-checked against all 6
  `.github/workflows/*.yml` files for every `cargo run --example` / `cargo test --examples`
  occurrence.
- Results: 231 files matched the glob; 1 (`examples/data/babel/lanes.rs`) is not a compiled
  example target (pulled in via `include!` from `probe_babel_stances.rs`) → 230 real targets.
  85 have main-path assertions. 4 are run by CI (`probe_eyes_opened`, `probe_babel_stances` in
  lance-graph-planner via rust-test.yml; `prove_it`, `substrate_compare` in jc via
  jc-proof.yml). 3 (`bake_family_codebooks.rs`, `certify_jina_v5_7lane.rs`,
  `seven_lane_encoder.rs`, all thinking-engine) have their asserting `main()` behind a
  non-default `calibration` feature — not even compiled by default, a stricter case than an
  ordinary blind gate. **BLIND GATE total: 78.**
- `--examples`/`--example` claim: CONFIRMED. No `cargo test` invocation anywhere in CI passes
  `--examples`; the only 4 example executions are explicit `cargo run --example` lines.
- Bug caught in my own method: a naive "split at first `#[cfg(test)]`" heuristic
  mis-classified `probe_antecedent_binder.rs` (its `#[cfg(test)] mod` sits BEFORE `fn main()`,
  not after) — brace-matching the excluded span fixed it; that file's true main-path count is
  1 (a `panic!` at the end of its gate report), not 0.
- Cross-cutting finding (reported, not fixed, per brief scope): `lance-graph-planner` — the
  crate 2 of the 4 CI-run examples belong to, and the crate AGENT_LOG repeatedly cites
  "324+/348+ tests passing" for — has **zero** `cargo test` invocation anywhere in any
  workflow file. Its whole `--lib`/`--tests` suite runs only via the orchestrator locally,
  never in GitHub Actions. Also identified 10 crates with zero CI presence at all (no test,
  no example run): bgz17, helix, perturbation-sim, sigker, lance-graph-arm-discovery,
  lance-graph-ontology, highheelbgz, sigma-tier-router, lance-graph-osint, thinking-engine.
- Full ranked table + per-row cheapest-fix notes + honest "did not do" list: `/tmp/audit_blind_gates.md`.
- Touched only: my own tag-file + `/tmp/audit_blind_gates.md`. Did not edit any workflow,
  any `blw_*.rs` file (owned by other agents per the brief), or any source file.
