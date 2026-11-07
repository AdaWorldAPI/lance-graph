# DT64 L5b: the source owns the historical basis; DT64 reads it at the bound revision (D-DT64-7-L5b)

**Date:** 2026-11-07. **Status:** in PR (stacked on #1975). Plan: `.claude/plans/d-dt64-7-behaviour-addressing-registration-v1.md` §1f, item 5.

## What changed

- `ClassView` gains `fields_at(class, rev)`, keyed by concept half, and `lens_fields_at(lens, rev)`, keyed by full classid (O9). Both default to `None`.
- DT64 `basis_at` reads a class's basis at its bound revision. It takes the source's historical answer when the source has one, and otherwise the current basis, accepted only under the existing proof (count now; exact once L5a lands).
- A lens never falls back to its left half, for its history or for its current basis.
- `WithPolyfill` (`lance-graph-r2il`) forwards both accessors.

## Measured

`a_history_keeping_source_replays_each_revision_against_its_own_basis` resolves ORDER at revision 1 (4 fields) and ORDER_V2 at revision 2 (5 fields) from one source. `a_source_without_history_refuses_replay_after_evolution` keeps its assertion.

Disable runs, each red then green:

| guard removed | test that turns red |
|---|---|
| ignore class history | `a_history_keeping_source_replays_each_revision_against_its_own_basis` |
| key lens history by the left half | `two_lenses_sharing_a_concept_half_keep_separate_histories` |
| drop `WithPolyfill` forwarding | `the_adapter_forwards_the_consumers_history` |

## Refinement recorded

The plan's earlier wording, "a `None` never falls back to the current basis", would have broken all 27 current-only views (H4, re-measured). The current basis is used only when it is proved equal to the bound basis, never in place of that proof.

OPEN: until L5a lands, that proof is the field count only.
