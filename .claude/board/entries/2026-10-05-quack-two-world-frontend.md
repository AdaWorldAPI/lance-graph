# 2026-10-05 — Quack two-world frontend: describe once, resolve once, execute numeric

## DECISION — the two worlds and the one membrane

- **Developer world:** table and field names, textual literals, typed descriptors and (later) SQL text. This side may allocate, normalize, hash, consult catalogs, KV and CAM, and fail.
- **Execution world:** fixed-width only — `TableId`, `Col`, `Mask`, numeric literals, ids (`ValueId`, `KeyId`, CAM ordinal, SAP code, `Guid128`, `Dn128`), `ResolvedReading`.
- **One crossing:** `lance_graph_quack::bind::Draft::bind(&dyn Binder) -> Query`.
- **No `ResolvedQuery` type.** `quack::Query` is already that form: `Filter` / `Cmp` / `Col` / `Agg` are all numeric, and `tests/string_fence.rs` keeps it so.
- **SCOPE:** the frontend lives in `quack::bind`, the crate's only text-bearing module. The executable IR and the mask-risc / ndarray layers stay text-free.
- **BASIS:** report already ran this pattern in production (`Catalog`, `CamLabels`, `Selection` → `Cmp::EqU32`). `bind` generalises the *lifecycle*, not report's types.

## WORKING-MODEL — one lifecycle, three domains, distinct types

| domain | description (developer world) | resolution | stable numeric contract | executed by |
|---|---|---|---|---|
| storage (#1326) | `SlabDeclaration` + SPOG concept | `Activation::resolve_for_context` | `ResolvedReading` | population reader |
| query | `table(..).where_eq(..)` | `Draft::bind(&dyn Binder)` | `quack::Query` | `lower` → mask-risc |
| schema | `create_table(..).width(..)` | `TableDeclaration::register(&mut dyn Registrar)` | `ResolvedTable { id, width }` | **no owner yet** |

The shared invariant is the lifecycle, not a unified type. Each resolved type keeps its domain's semantics.

## MEASURED — the proofs (tests)

- `quack/src/bind.rs` tests:
  - A text literal binds with exactly 4 binder calls (table, live plane, field, code).
  - The bound `Query` `==` the hand-written expert `Query`, and their lowered `Program`s are equal.
  - Executing twice makes 0 more binder calls; the live plane excludes the dead row.
  - A typed descriptor (`FieldRef`) binds to the same query.
  - Bind errors are reported in the developer's vocabulary and mint nothing.
- `quack/tests/string_fence.rs`: no text tokens in `lib.rs` outside tests (more than 1,000 lines scanned); `bind.rs` is the only other module.
- `report/tests/quack_bind.rs`:
  - Report's `Catalog` + `CamLabels` acting as a `Binder` cost one CAM lookup at bind.
  - After a label rename, the new label binds to the *identical* query and the old label is `UnknownValue`. The ordinal identity survives.
- `dir-sim/tests/quack_bind.rs`:
  - `smtp` (`KeyId`) binds two spellings to one query; `smtp_exact` (`ValueId`) binds them to two.
  - The key-bound query lowers to exactly `key_eq_program(key)`, the program dir-sim executes.

## OPEN — seams, not built here

- **DDL actuator:** nothing owns a table catalog or physical allocation, so `Registrar` has only a test implementation. `CREATE TABLE … WIDTH …` stops at `ResolvedTable` until an owner exists.
- **Java:** today `View.where` → `plan_lower` → mask-risc; answer parity with Quack is proven by `lowering_convergence`. Convergence target: Java DSL → `Binder` → `quack::Query` → `lower`, which retires `plan_lower`.
- **SAP:** its forward text → code map is dropped at bind, so it has no `Binder::code`. Keeping that map, cold, per batch would let `activity = 'Consulting'` bind after ingest; codes stay batch-local.
- **IAM:** `users_with_key` can become `table("users").where_eq("smtp", …)` over an IAM `Binder` with no identity change (the test above is that binder).
- **SQL frontend:** `SELECT COUNT(*) FROM users WHERE smtp = '…'` parses to the same `Draft`. No parser is in scope.
