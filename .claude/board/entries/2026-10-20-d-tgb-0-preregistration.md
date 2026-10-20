# 2026-10-20 — D-TGB-0: thought-git pre-registration

**Status:** MEASURED (lab probe `crates/thought-git-probe`, own workspace, excluded; depends on `lance-graph-contract` only). Operator approved lab code 2026-10-20 ("Lab-Code freigegeben, fang mit D-TGB-0 an").

## What landed

- Fixtures (`src/fixtures.rs`): one write intention per Rubicon outcome (Commit, Plan, Prune, pre-Rubicon veto), an unremarkable stack, a contradiction against a given, a divergent pair, a regression scanpath, a cold/hot surprise pair, an unwritten lane. Twelve rows of four lanes; givens on lane 0 of rows 0..4.
- Oracle (`src/oracle.rs`): the next version from scratch. Only Commit lands; a given is never overwritten (agreement not stored, a different value recorded as a contradiction); a divergence inside one version stores nothing and is recorded; an unwritten cell reads absent.
- `PREREGISTRATION.md`: falsifier, disable and fixture for D-TGB-1..10.

## Measured

- 19 tests: 9 oracle, 10 signal. Clippy `-D warnings` and fmt clean.
- Five oracle disables, each red at the named test, restored green: land every outcome (`only_commit_lands`); store agreement (`agreement_with_a_given_is_not_stored`); overwrite the given (`a_different_value_for_a_given_is_refused_and_recorded`); last writer wins (divergence test and the squash test); read missing as 0 (absence test, plus two that read an absent cell).
- Signals on real contract code, each with a can-fire and a can-stay-silent case: `AlphaOverlay::regressions`, `AlphaMask` cold/hot xor, the kanban route per outcome, the MUL gate at `Evaluation`, and `GadamerRevision` (replayed material is `Echo` and routes to `Plan` even with a `Necessary` counterfactual; a second landing has no new root; a preserved contradiction suspends and keeps its tension).
- Naive squash is safe on the unremarkable stack and hides the divergence on the divergent pair, which is the reason an anomaly stops a squash.

## OPEN

- Which anomaly signals stop a squash and at what threshold; which revision kinds besides a contradiction count; what "satisfactory" means; kanban card granularity.
- The 59..63 falsifiers on edges (`pearl::hydrate` / `pearl::revise`) need `lance-graph-planner`; they run with D-TGB-3.
