# Decision J: the DT64 class basis is exact within a SchemaRev; findings pinned (D-DT64-7-L5)

**Date:** 2026-11-07. **Status:** in PR (tests and plan only; no source change).
Follows the read-only DT64 convergence audit (/5plus3: 5 savants, 3 reviewers, ratified v3). Plan: `.claude/plans/d-dt64-7-behaviour-addressing-registration-v1.md` §1f.

## Decision (operator, 2026-11-07)

The ordered `predicate_iri` basis is immutable within one class `SchemaRev`. Reordering, replacing, removing or appending a field takes a new revision.

- ClassView's append-only wording describes compatible evolution across revisions, never mutation within one.
- A label-only rename that keeps every `predicate_iri` keeps resolving.
- `MethodRef` and `plan_id` gain the class revision and basis identity. The lens revision is not duplicated, and value typing stays with the source.
- `LaneBinding` stays off R2IL execution until the meaning-side witness is enforced.
- D5, O9, EXO-140 and EXO-46 stand. A digest, a mask or an admission is never permission to execute.

## Finding (pinned, not fixed)

`check_count` compares the schema revision and the field count only (`lance-graph-dt64/src/lib.rs:876-897`), and `admit` never reads the view (`:834-860`). So an equal-length reorder, or an equal-length `predicate_iri` change, passes `resolve` and `admit` while the method keeps reading the old positions.

The lane side is already protected: sources map by `predicate_iri` (D-LANESOCK-3). The method side has no equivalent.

Pinned in `crates/lance-graph-dt64/tests/bind_store.rs`:

- `finding_an_equal_length_reorder_resolves_and_admits`: flips when L5a lands.
- `finding_an_equal_length_predicate_change_resolves`: flips when L5a lands.
- `a_label_only_rename_resolves_and_must_keep_resolving`: the silent twin; stays green.
- `finding_a_historical_basis_is_unreachable_through_an_unversioned_view`: flips when L5b lands.
- `schema_revision_and_field_basis_are_checked`: re-annotated. The same-revision append refusal is now the ruled behaviour.

## Historical-replay consequence (reported separately)

`ClassView::fields(class)` takes no revision. Once a source's basis evolves, an old classid can no longer be resolved at the revision it was bound at, even though its behaviour is immutable (D5). A derived classid for the new revision does not help.

This already holds today for a count change, and the pinned finding shows it. Once the exact witness lands it will hold for every evolution.

Replay therefore needs a source-owned historical basis. The plan proposes the O9 pattern: a defaulted, revision-keyed `ClassView` accessor that answers `None` and refuses, never falling back to the current basis. It is a contract change and gets its own PR. It should land before, or together with, the exact witness.

## H4 census

- **Scope:** all 27 `ClassView` impls in `crates/`.
- **None grows or reorders its basis in place**, and none is revision-aware.
- **6 are in non-test `src/`.**
- **Conclusion:** no shipped consumer relies on same-revision growth, and none can answer a historical basis yet.

**Fruit:** L5a exact witness (reuse O10, DR `extends` for cross-revision prefix) · L5b historical basis accessor · L5c `MethodRef` / `plan_id` identity · L5d test-only `LaneBinding` → `FieldLanes` with coordinate, scope, mapping and permit refusals.
