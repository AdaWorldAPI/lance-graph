# Batch selection over OGAR's Ruler planes (shared-index law, identity witnesses only)

**Date:** 2026-11-06. **Status:** LAB (test-only, `lance-graph-dir-sim/tests/ordinal_chain_lab.rs`). Follow-up to merged #1964 / OGAR#371.

## Shared-index law as it exists today

- `Ruler::from_fold` is OGAR's aligned **identity-witness** view: one slot per AD row; Entra anchor/id/time and the EXO external id are placed only for an uncontested counterpart (`CONTESTED_AD` / `CONTESTED_ENTRA` / several mailboxes leave the slot empty). It copies identity witnesses (GUIDs, timestamps), never property values, and is not a canonical SoA64K.
- `fold_aligned` turns it into six 64-slot planes: `forward`, `backsync`, `cloud`, `aligned`, `forward_conflict`, `backsync_conflict`; `Planes::confirmed` = forward ∧ backsync ∧ aligned.
- Coordinate references for a slot `i`: AD row `i`, Entra row `Output::ad_target[i]` (else `ad_back`), EXO row `Output::entra_mailbox[j]`. These describe a tuple; none is stored.
- **Not aligned:** every Entra/EXO business-value lane stays in its own source order. A mask over the identity planes does not authorize reading those lanes at slot positions.

## What the lab proves

One lowered Quack program (`Filter::and` of plane leaves, `Agg::Rows` / `Agg::Count`) over the six planes, one `FrameStore::check` per pass:
- equals an independent scalar oracle slot by slot over multi-word seeded populations with ten fault classes and shuffled Entra/EXO rows; kept slots' `Output` references equal the oracle's rows;
- reproduces dir-sim's existing per-row `CloudMailboxes::from_fold` selection (cloud ∧ ¬fwd_conflict ∧ ¬back_conflict) as a batch pass;
- reordered populations select the same identity triples;
- foreign, re-sealed, source- and coordinate-stale frames refuse before execution.

Disable runs (committed first): dropping `cloud`, `aligned`, the `¬backsync_conflict` leaf or the per-pass frame check each turns its test red.

## Measured (release; reported, not gated)

64K slots, 35,202 kept: hot keep ~0.55–0.63 µs/pass, hot count ~0.40–0.67 µs/pass, **0 allocations**. Cold Index+fold+Ruler+Planes 43,000–52,000 µs (cold correspondence, outside the fold).

## Open

- `CloudMailboxes::from_fold` still selects per row and materializes `(Guid, Guid)` tuples; `with_exchange_guids` is a second sort/search GUID index. Not changed here.
- Reading an Entra/EXO property at kept slots needs either source-owned co-addressed lanes or an admitted gather; neither exists (Gather / ADMISSION B / NotASide / OP-14 unchanged).
- DR property routing (versioned) ↔ Ruler slot selection has no owner contract: DT64 `BindStore` route versions and `FrameStore` revisions are independent clocks with no stated relation.
