# A10 probe outcomes, S6 Magic Sets correction, NNUE triangle baselines (2026-11-02)

**Status:** measured. A10 is a test-local probe, not promoted to production. S6 is pushed to the xlog fork and compiles; it has not been run. NNUE is PARKED for the triangle optimizer only.

## A10 — probe-first outcomes (F1/F3/F6 run before any admission writer)

Files: `cognitive-shader-driver/tests/a10_epistemic_admission_probes.rs` (F1/F3/F6) and `lance-graph-planner/tests/a10_admission_probe.rs` (admission probe, F2/F4/F5). Each guard was disable-verified red-then-green (3 + 5 disables).

| F | outcome | what it means |
|---|---|---|
| F1 | CONFIRMED | A stamped Direct × Open (`raw5 = 0`) is bit-identical to an unwritten `CausalEdge64`. Only the alpha claim distinguishes them on the hot path: `get` returns `None` when unclaimed, and a claimed row carries an `AlphaStamp` in its value. After seal, only the row's existence remains. |
| F3 | FIRES on the CE64 path, HOLDS elsewhere | `Opcode::Revision.truth_code` and `CausalEdge64::learn` get identical inputs for overlapping and independent roots and raise c on a duplicate. Two places do enforce overlap. `deepnsm_v2::belief::BeliefArena::revise_at` (production) has a stamp gate, already pinned by `revision_overlapping_stamp_is_rejected`. `GadamerRevision` refuses a repeated root only through the caller's `BasisView`; with an empty ancestry the duplicate is admitted. The test-local P-E2 gate is not production evidence. |
| F6 | HOLDS | `with_novelty_wisdom` (0..=9, saturating) and `learn()` never move bits 59..63. |
| F2 | HOLDS under the probe | A cell derived by elimination stays at its constraints' certification (Related) and cannot mint a root. Two givens from different sources stay independent. |
| F4 | FIRES | Withdrawing one of two sources blocks a fresh measurement (`Ungrounded`), but a certification written before the withdrawal stands. No writer lowers it. Withdrawing an unrelated source changes nothing. |
| F5 | HOLDS under the probe | A derivation from Causes premises stays at Related. Under the same model, an executed randomized trial earns Causes. An observational (SO) measurement stays at or below CausalCandidate (pearl's cap). |

The admission probe types four kinds, each requiring positive evidence:
- Given: an identified source, capped by the hydration proof.
- Observed: a new independent root; earns no certification, because eligibility is not acceptance.
- Derived: a rule identity and inherited roots only; at most Related.
- Measured: `pearl::reason` / `pearl::revise`.

**Explicit gaps:**
- (1) No contract type carries a derivation rule's identity; the probe uses a `RuleIdentityGap` stand-in and refuses a derivation without one.
- (2) Production admission boundary: no production writer goes through an admission check.
- (3) Retraction does not lower a written certification (F4).

### `with_epistemic_raw5` call-site census (production src, read at each site)

| site | class |
|---|---|
| `lance-graph-planner/src/pearl.rs:404` `stamp`, called by `revise` and `hydrate` | AUTHORIZED: evidence-gated, certification only rises to what the operator earned |
| `causal-edge/src/edge_v3.rs:296` `rehydrate` | AUTHORIZED: restores the existing code on the V3 lift; mints nothing |
| `deepnsm-v2/src/wire.rs:974` and `:1036` `wire` | UNDOCUMENTED SECOND WRITER: writes certification from its own grammar evidence, and promotes Related → CausalCandidate on ≥ 2 independent implication sources (stamp popcount). Consistent with the SO cap, but outside `pearl::revise`. Needs an owner decision. |
| `nars/{warrant,scent,scent_learn,mediator}.rs`, `chain_counterfactual.rs`, `edge_v3.rs:894` | inside `#[cfg(test)]`: fixtures |
| examples (`cognitive-shader-driver`, `lance-graph-planner`) and `tests/` | documented lab exceptions |

## S6 — XLOG fork Magic Sets correction

AdaWorldAPI/xlog `claude/magic-query-relative-demand` @ `ea924b3`.
- `undemanded_consumer_reasons` accepts only two kinds of reader of a target predicate: the target rules themselves, and the generated `__xlog_query_{i}` rule of a seeded query whose body is exactly that query's atom.
- Every other reader yields a reason routed through `decline_or_error`: positive, negated or epistemic, in an ordinary rule, a constraint rule, a learnable rule, or an unseeded query.
- Modes: `auto` declines and keeps the program unchanged; `on` errors; `off` is unchanged.
- Verification follows the fork's rules: `cargo check` and `clippy -p xlog-logic` are clean (warning count unchanged, 78), and no tests were added or run. Compiling is not runtime validation.
- The reach/far counterexample remains the named external falsifier.
- Known conservative case: a seeded query whose generated rule a later normalization pass rewrote is declined.

## NNUE — triangle pairing baselines

`xlog-relcomp-probe/examples/triangle_pairing_baselines.rs`, re-run by the orchestrator. Results were identical and deterministic: 72 instances, 4 train and 2 held-out families. The probe's argmin is verbatim `optimizer.rs:3395-3404`.

| arm | mean regret | optimal choices / 72 | permutation changes |
|---|---|---|---|
| a0: estimates from cardinalities | 13.13 (worst 143.8) | 45 | 24 |
| a1: estimates plus exact distinct counts | 4.78 (worst 113) | 48 | 1 |
| exact-degree fold | 1.000 | 72 | 0 |

- **Collapse:** 24 equal-size instances collapse to a three-way tie; all fall to SharedY by evaluation order. F4 kills a0.
- **Cost:** the exact fold costs 1.3–2.7 ns per insert to maintain incrementally (96 KiB), or 31–50 µs from scratch. The existing argmin costs 113–139 ns.
- **The brute-force join** is an oracle arm only: it materializes the join to check the fold, and is not timed.
- **Verdict:** PARK NNUE for this optimizer. The fix is to keep exact degree counts, not a model. NNUE is not parked globally.

Second application (Ruff × ARM × SoC):
- **Verdict: NEEDS-GROUND-TRUTH-FIRST.**
- **Candidate target:** ranking the next residue on the MedCare furnace exam. The deterministic baseline already exists: `rekey_exam` sorts by method count.
- **Not applicable yet:**
  - Ranking Concern and Scope splits: nothing generates candidate splits.
  - ARM: it has no `FeatureSpec` over ruff facts.
  - CB2Vec and NORU: absent from the workspace.
- **Gate before any trial:** a private outcome TSV in MedCare-rs and a measured precision@15 baseline.
- **Kill condition:** the ranker ranks the gate-refused `user_right` above a confirmed mint.
