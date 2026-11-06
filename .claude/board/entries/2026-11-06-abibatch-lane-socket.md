# AbiBatch lanes through the DT64 lane socket (D-LANESOCK-2)

**Date:** 2026-11-06. **Status:** read adapter, not production-admitted (no authoritative reader-scope witness). Follows D-LANESOCK-1 (#1967).

## What it is

`lance_graph_report::AbiLanes` implements `lance_graph_dt64::LaneSource` over a resident `AbiBatch`:
- field position → `ClassView::fields(view_class(class))[pos].predicate_iri` → `Catalog` → `FieldId` → `AbiBatch::column` lane index;
- `borrow` hands out the batch's own `Arc` lane as `LaneRef` (pointer equality tested);
- one `FrameStore` per source, revisions `(SourceId, 0, generation)`; `publish` takes a later generation of the same source and advances it;
- `I32` / `U32` only; a `U64` reference lane is unbound.

DT64 binds the class and admits the method; Quack / mask-RISC folds. No store, field registry, codebook or BindStore was added.

## Admission scope

- Reader scope: DR's `Scope` is a plain `{ id }` with no issuer. `AbiLanes` has no production constructor; `AbiLanes::unwitnessed` exists only under the test-only feature `unwitnessed-scope`.
- No writer: a read adapter over resident lanes.
- Open, unchanged: value ingress, durable writer, codebook, exact decimal, D4/S-INGRESS.

## Falsifiers (`crates/lance-graph-report/tests/lane_socket.rs`)

Borrowing (pointer equality), scalar parity on two folds with one binding, reuse with zero hot allocations, published replacement refuses the old binding, old row order refused after a publish, foreign frame, another source's frame + store, wrong route, incompatible coordinate space, unknown / absent / `U64` field, field outside the basis, foreign scope. Disable runs (committed first): admitting `U64`, copying the lane, skipping the frame advance, dropping the generation from the coordinate space, an allocation in `lane_of`, and dropping the adapter's own frame check each turn a test red.

## RegistryClassView

`class_resolver::tests::concept_half_keys_the_basis_but_not_the_registry_category`: keyed by `classid_concept` (= DT64 `view_class`), the basis resolves. Defect pinned, not fixed: `dolce_category_id` / `is_known` read the same key as the registry's append-order `entity_type_id`, so a registered class's DOLCE category is unreachable through the concept key.
