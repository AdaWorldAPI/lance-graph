# Corrected harvest — bound computation

Supersedes the five-PR sequence in `CLAUDE_LANE_FOLD_CAPSTONE.md` for this thread. The harvest stood. The proposed form did not. The pipeline rule is `IMMATERIAL_HANDOVER.md`.

## The germ

```text
BOUND COMPUTATION
  = Destination { space, ordinal }
  + AlgebraLaw          (planner-facing, metadata)
  + Contribution
```

Execution keeps its own state. `FoldState::merge` stays in report. `Hom` stays in lane-fold. Neither moves into the contract as a function of `i64`.

```text
BIND
  → Destination
  → AlgebraLaw
       ├ DATA State   FoldState::merge / identity
       └ DO State     MutationState, later, not now
```

Three proofs of the same pattern already exist:

- Quack: names and literals bind once, the `Query` holds none of them (`Draft::bind`).
- Report: text stays in CAM, the fold holds `FieldId` plus an ordinal. A rename does not touch the ordinal, the mask, the fold, or the cache (`boundary.rs`).
- Lane-fold: repeated walks collapse to one terminal.

## Do not extract

- An executing `merge(i64, i64)` trait in `lance-graph-contract`. Later state is a bitmap, a vector, or a mutation, and the trait would have frozen the representation.
- A universal `Refuse`. The lane variants are precise because they are local. A shared enum grows an `Other`. Rule only: every lowering returns an explicit allocation-free refusal, and does not silently pick a generic execution.
- `RouteRef`, until two binders produce the same destination contract.
- `BindingProjection`.

## Generation

`SourceRef { id, generation }` in `lance-graph-report/src/plan.rs` is the authority pattern. Do not invent a second one. Ask whether a destination space needs the same pair. Do not stamp generation onto every ref.

## PR sequence

1. `AlgebraLaw` in the contract. Flags only: associative, commutative, idempotent, ordered, invertible, identity-kind. Report and lane-fold each implement a descriptor. No `merge` body.
2. `Destination { space, ordinal }` harvested from CAM. Built only from an existing ordinal. Renaming a label must not change it.
3. Two counters on `boundary.rs`: `internal_serialization_bytes`, `materialized_population_bytes`. One test asserts both are 0 across a fold.
4. A conformance probe, not a shared binder. A quack bound query and a report plan are asserted to name the same conceptual pair: destination plus algebra. They do not share a struct yet.
5. Lane-fold `collapse` reads `AlgebraLaw` and does not learn executor internals. Quack is not modified to execute.

Between 4 and 5, watch whether a second planner arm repeats a refusal class. Only then consider `RefusalClass` in the contract. `RouteRef` is a sixth PR, and only if PR 4 showed two paths producing one destination.
