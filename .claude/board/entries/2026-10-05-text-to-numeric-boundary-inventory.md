# 2026-10-05 — text → numeric id → Quack → mask-risc: one pattern, four id semantics

## MEASURED — who runs the pattern (production callers traced)

| consumer | text in | binder | numeric id | Quack in prod? | final primitive | text during execution |
|---|---|---|---|---|---|---|
| lance-graph-report | field name, categorical label | `Catalog` → `FieldId`; `CamLabels::ordinal` (CAM over `ContentId`) | `FieldId`, per-field CAM ordinal | yes (`exec.rs` calls `lance_graph_quack::lower`) | `Pred::EqU32` via `Cmp::EqU32` | none (`string_fence.rs`; counters in `reference_workload.rs`) |
| lance-graph-sap | CATS columns | `bind.rs` first-occurrence dict per batch (forward map discarded) | batch-local `u32` code, 0 = NULL; employee via `numc` (parsed, not dict) | yes (`CatsQuery::prepare`) | `EqU32` / `GeI32` / `LeI32` + `GroupSumI32` | none; reverse labels kept for egress only |
| lance-graph-java (lgj-abi) | none (static Java field constants) | compile-time `LaneId` | lane index + i64 operand | **no** — `plan_lower.rs` (quack is a dev-dependency) | same `Pred` vocabulary (`EqU32`, `GtI32`, `MatchU32`, …) | none |
| lance-graph-dir-sim | UPN / SMTP, query literal | `Dicts::intern` / `key_lookup` | store-lifetime `ValueId` / `KeyId` | yes | `Pred::EqU32`, `MatchFacet16Strided`, `Gather` | none (`string_fence.rs`, `where_eq.rs`) |

Java's `lowering_convergence` compares ANSWERS (row counts / group sums) over 28 combine vectors and 9 opcodes, not Programs. Moving Java onto Quack would be convergence, not invention.

## FINDING — ids are not interchangeable

- **`ContentId`** (contract): content identity, fnv1a64 of exact bytes. Process-independent, survives reopen, legal anywhere. It is u64, and it has no normalization.
- **report CAM ordinal**: a per-field category. `rename` re-points the label and keeps the ordinal. That is presentation identity, safe because a coordinate means the category, not the text.
- **report `FieldId`**: schema/catalog identity.
- **SAP code**: one bound batch only (`CatsQuery` borrows the batch). It cannot be persisted, and a text literal cannot be resolved to it after binding.
- **dir-sim `ValueId`**: the exact text, shared across attributes, append-only for the store's lifetime. A different text is a different value, and changing it is a `SetAttribute`. It is persisted in `Change` / `ExecutionPlan` / `Precondition`.
- **dir-sim `KeyId`**: the normalized comparison form, with the same lifetime. It is used only for comparison: uniqueness and rename ordering.
- **OGAR `ValuePool` `StrRef`**: a per-batch byte offset, not deduplicated. It is not an identity.
- **Quack `Col`**: a lane position inside one `Planes`.

Consequence: replacing `ValueId` with the CAM ordinal would be wrong. A CAM rename keeps the id and changes the text, so a plan's compare-and-set would silently target a different string.

## OPEN

- Should `ValueId` converge onto `ContentId` (content identity, reopen-stable)? That would need u64 lanes and a collision policy. Decision pending.
- SAP has no query-time literal → code path, because its forward map is dropped at bind.
- Java production still lowers outside Quack.
