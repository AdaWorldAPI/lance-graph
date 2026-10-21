# 2026-10-21 — D-GKAT-0: PROBE, bounded GKAT checking of structured Rubicon flows

**Status:** PROBE (lab, `crates/gkat-probe`). This is a research fixture, not a production verifier. Everything here is TEST-PINNED; nothing is proven beyond what the cited paper proves.
**Claim:** EXO-50 `session_01VncKoyG6JkpsoqhqTGTv3f/026`, ACKed in `session_01G22yT6htkcdyXsihxxXdrg/036`.
**Plan:** `.claude/plans/d-lfa-lance-graph-flow-rubicon-assimilation-v1.md` § Amendment 2026-10-21 (#1673).
**Source:** `.arxiv/1907.05920_guarded_kleene_algebra_verified_control_flow_excerpt.md` (#1670); Smolka et al., POPL 2020, arXiv:1907.05920 §§2–5.

## What landed

`crates/gkat-probe`: zero dependencies, its own `[workspace]`, excluded from the workspace, with its own CI workflow (`.github/workflows/gkat-probe.yml`).

**The admitted fragment.**
- Constructs: `0`, `1`, assert `b`, action `p`, `e·f`, `if b then e else f`, `while b do e`.
- Boolean tests: `0`, `1`, `t`, `¬`, `∧`, `∨`, over at most **8 primitive tests (at most 256 atoms)**.
- Excluded: unguarded union and Kleene star.

**The source flow language** (`flow.rs`) is a graph-flow-shaped superset. Lowering admits the structured part and refuses the rest with a typed `Refusal`, never by translation.

| construct | lowered as |
|---|---|
| `NextAction::Continue` / `ContinueAndExecute` | sequence |
| `End` | accept |
| `WaitForInput` | `while ¬ready do Suspend`, where `ready` is observed at resume |
| conditional edge | `if` |
| `GoTo` / `GoBack` | `Refusal::UnstructuredJump` |
| choice without a condition | `Refusal::UnguardedChoice` |

**K3 guards.**
- A K3 observable occupies two primitive tests, `is_t` and `is_u`. The atom set excludes `is_t ∧ is_u`, which leaves exactly three values.
- A K3 guard reaches a Boolean test only through a declared projection:
  - `Where` = `{T}`
  - `Check` = `{T,U}`
  - `Is` applies to Boolean observables only.
- Any other pairing of projection and observable is refused.

**Checks.**
- **Equivalence** (`automaton::equivalent`): the steps follow the paper:
  1. guarded Thompson automata with uniform continuations (§4.2);
  2. normalization of steps into dead states (§5.2);
  3. bisimulation with a union-find (§5.3, Algorithm 1).

  It returns a counterexample prefix. Flows under different signatures are refused (`DeclarationMismatch`); the signature covers observables, projections and action alphabet.
- **Safety** (`automaton::commit_safe`): a separate check, over every reachable step of the **unnormalized** automaton.
  - Every `RequestCommit` must come immediately after an atom satisfying WHERE(rev) ∧ WHERE(mul).
  - That authorization is fixed by the checker, never taken from the flow.
  - An authorization observed before an intervening action is stale.
- **Oracle** (`oracle.rs`): an independent operational interpreter of the expression AST that shares no code with the automaton. It explores every run exhaustively up to a stated action bound.
  Divergence is detected exactly, by a repeated continuation stack at a loop head, not by a step budget. The first version used a fixed budget, which Codex showed misclassifies a terminating run of 2^14 silent steps.

## Measured (`cargo test`, 28 tests)

- **G1.** `if b then {p;q} else {q}` ≡ `(if b then {p} else {skip}); q`. The oracle agrees at bounds 0–4. A changed action label is rejected, a reversed guard is rejected, and a different tail is rejected. `p·0 ≡ 0` (normalization).
- **G2.** Rejections:
  - unguarded choice → `UnguardedChoice`;
  - `GoTo` / `GoBack` → `UnstructuredJump`;
  - the union "do `p` or `q` whatever the atom" fails the determinacy property (§2.2), so no GKAT expression denotes it.

  So does the union of `skip` and `p`: on one atom it may either accept or act. The admitted Rubicon flows are deterministic.
- **G5.**
  - For `k ≤ 8` tests: `2^k` atoms exactly. For `n ≤ 4` K3 observables: `3^n` atoms exactly.
  - `k = 9` is refused (`AtomBound`). A DR64-shaped signature (64 Boolean lanes) is refused, as is one with 5 K3 observables (10 tests).
  - On a chain of `k` guards, the automaton has `2k` states while the atom count is `2^k`. Measured (k, atoms, states, merged pairs): (1,2,2,3) … (8,256,16,17).
  - Counts only, no timings (the workspace's no-ms rule).
- **Rubicon** (11 tests, 36 atoms).
  - The reference flow has the decision tail written out twice; the optimized flow factors it out (G1) and reorders the decision. The two are **equivalent**, and both are **safe**.
  - Disable runs on the flows:
    - reversed authorization polarity → equivalence fails;
    - `RequestCommit` moved ahead of its guard → both checks fail;
    - authorization observed before `Deliberate`, then `RequestCommit` → safety fails;
    - unguarded `Choice{RequestCommit, Hold}` → refused;
    - commit authorized with CHECK → safety fails, and the counterexample atom carries a `U`.
  - CHECK-authorized reference ≡ CHECK-authorized optimized, and both are **unsafe**: equivalence does not establish safety.
- **Differential.** 600 random expression pairs over 2 tests, half of them single-point mutations:
  - 252 equivalent, 348 different, and the oracle separates all 348 within 3 actions;
  - every automaton-equivalent pair agrees with the oracle at bounds 0–4;
  - every oracle difference is an automaton difference.
- **Checker disable runs: 11, each red then restored green.**
  - bisimulation ignoring action labels;
  - safety ignoring authorization;
  - a loop body that accepts without acting accepting the loop;
  - no normalization;
  - CHECK lowered as WHERE;
  - choice encoded as `if`;
  - `GoTo` translated to skip;
  - atom bound raised to 64;
  - uniform sequence continuation not patched;
  - signature mismatch not refused;
  - K3 exclusion dropped.

  The first normalization disable stayed green because no test compared a step into a dead state with a rejection. `a_step_into_a_dead_state_is_a_rejection` was added to close that gap.

## Proof boundary

- **The paper proves it, for the admitted fragment over a fixed atom set:** bisimilarity of normalized Thompson automata coincides with guarded-string equivalence.
- **This crate's own correctness is tested, not proven.** The oracle is bounded, so its use is one-directional: a difference it finds is real; an agreement up to `n` actions is not a proof.
- **Not modelled:**
  - versions, and snapshot or codebook staleness (a separate runtime gate);
  - SQL three-valued logic beyond a declared projection, and bag counts;
  - side effects, and the Lance write. `RequestCommit` is an intent; the write belongs to the supervisor / `cycle_driver`, which this crate does not touch.
- No near-linear complexity is claimed outside the paper's fixed-atom assumption.

## Open

- **D-LFA-1 runtime flow:** unclaimed. The probe models it but does not run graph-flow.
- **Symbolic guards:** BDDs or a SAT-backed bisimulation, to go past 8 tests. Not built, not measured.
- **G3/G4/G6/G7** of the excerpt: not attempted (NULL/bag negatives, loop productivity against a derivative form, physical parity, amortization).
