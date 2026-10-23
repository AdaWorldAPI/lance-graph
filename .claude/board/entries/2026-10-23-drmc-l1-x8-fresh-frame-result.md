# 2026-10-23 — EXP-DRMC-L1 — L1 result: a certified reassociation reused on a fresh frame (D-DRMC-5 + X8)

**Status:** MEASURED, lab only (`crates/dr-math-binding-probe`). Pre-registration:
`2026-10-23-drmc-l1-x8-fresh-frame-preregistration.md`. Code: `src/recipe.rs`, `src/relcomp.rs`
(`execute_certified`, `egglog_calls`), `tests/x8_fresh_frame.rs`.

## Result: the hypothesis held

**Positive case.**
- Five proposed association orders of one 5-chain were each certified on frame 1. The target was the
  cheapest order by `Work` on frame 1's data.
- All five certificates executed on independently issued frames 2, at densities 0/64, 4/64, 32/64 and 63/64.
- egglog delta: 0 on every call.
- Each result equals both the independent pair-enumeration oracle and `Chain::eval` of the requested order.

**Refusals.**
- Commutation `r0∘r1 → r1∘r0` was refused at certification (`DifferentLeaves`), so no certificate exists.
- `execute_certified` refused each of the following, and computed nothing in each case:
  - `WrongTarget` (runtime ABI ≠ certificate ABI);
  - `NotCertifiedQuery` (another order of the same leaves);
  - `NotThisChain` (a 4-chain).

**Silent twin.** Same shape, other data: executed every time and never refused.

## Disable runs (all committed first, every anchor asserted)

| run | patch | turned red |
|---|---|---|
| D1 | drop the ABI check | `a_different_runtime_abi_is_refused` (only) |
| D2 | drop the source-digest check | `an_order_the_certificate_does_not_answer_is_refused` (only) |
| D3 | call `prove_reassociation` inside `execute_certified` | `a_certificate_from_frame_one_executes_on_a_fresh_frame_without_egglog` + `report_costs` (both assert the egglog delta) |

The `NotThisChain` guard has no disable run. One was not pre-registered, so none is claimed.

## Measured (reported, not gates; release, 4 cores, load about 1)

- Cold `certify_reassociation`: **15 696 µs**.
- Hot `execute_certified`: **2 142 ns/call**. It runs the certified cheaper order.
- `Chain::eval` of the requested (left-deep) order: 7 216 ns/call.
- Certificate: 1 284 B (1 012 B proof text, 9 compiled steps).
- Under D3, 20 000 calls took 297 s. That is the cost of putting the prover on the reuse path.

## What this does not show

**Proof checking.** There is no independent proof checker. The proof text is egglog's own and is stored,
not re-checked at reuse (S2 is still open).

**Choice of target.** The cheapest target was picked on frame 1. On frame 2 the stored order may not be
cheapest. Specialization selection is out of scope.

**Identity.** `source_digest` and `compiled_digest` are SHA-256 over the lowered program. They identify
the query and the artifact, not the proof. The law reference stays a separate field. The passive proof
reference (option B, joint with CC1) is not designed here.

**Coverage.** Thought-Git linkage, R2IL/Quack lowering and a second law are not covered.

`L1 | STATUS: measured | OUTCOME: certificate reused on a fresh frame with 0 egglog calls, 3 refusals pinned by disable runs | OPEN: independent checker, option B reference, L2 (distributivity + false ∩ variant)`

## Correction (2026-10-23, follow-up PR)

The 1 284 B certificate size counted egglog's rendered proof (1 012 B) as the proof. The certificate now holds egglog's `ProofStore` + `ProofId`; `report_costs` prints the certificate without the proof (272 B, 9 steps) and the proof as 8 nodes, with the 1 012 B rendering for display only. The arena's heap size is not measured.
