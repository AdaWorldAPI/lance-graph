# 2026-10-05 — Quack two-world frontend: describe once, resolve once, canonicalize once, execute numeric

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

The membrane does two things once: it resolves meaning (names, labels, source types) and it fixes the physical representation. The common law:

```
external ambiguity
    ↓ once
resolved semantic contract  +  canonical physical contract
    ↓
no reinterpretation in the population hot path
```

| domain | semantic contract | canonical physical contract |
|---|---|---|
| storage | `ResolvedReading` (which reading applies) | the stored row's integers are read little-endian (`NodeGuid`, `FacetCascade::from_bytes`, `Register128::words`) |
| query | `quack::Query` | none of its own: `Cmp::EqU32(v)` is a number. Byte order applies only where a strided lane reads raw bytes (`EqU32Strided`, `NeU32Strided`, `MaskedStridedGroupSum`) |
| schema / DTO | `ResolvedTable` | the fixed-width row ABI is little-endian; `CREATE TABLE … WIDTH` names no endianness |

The lifecycle is shared; the types are not. Nothing is routed through `ResolvedReading`.

## DECISION — the canonical byte rule (substrate policy, never application vocabulary)

- `u8` / `i8` and byte arrays: order-neutral, the declared sequence is the value. This covers `Dn128 = [u8; 16]` and the `MatchFacet(16)Strided` pattern/care bytes, so no endianness conversion is added to them.
- `u16` / `u32` / `u64` / `u128` and their signed twins, when represented as bytes: little-endian.
- A byte array's sub-field read as an integer: little-endian.
- `Register128`: four little-endian `u32` words (unchanged).
- No `Endian` field on `Cmp`, `Query` or `Program`, and no `ResolvedQuery`. Logical `LaneRef::U32` / `I32` / `U64` lanes stay numeric.

## MEASURED — byte-boundary audit (#1328 scope)

| site | representation | width | order | explicit? | canonical? | Quack sees |
|---|---|---|---|---|---|---|
| `ndarray::simd::eq_u32_strided_to_mask` (`EqU32Strided` / `NeU32Strided` executor) | strided bytes → `u32` | 4 | LE | `from_le_bytes` | yes | number (`Cmp`), kernel reads bytes |
| `ndarray` `masked_strided_group_sum` | strided bytes → `u64` group | 1–8 | LE | `b << 8k` assembly | yes | terminal over bytes |
| `mask-risc` `reference.rs` `strided_u32_at` | strided bytes → `u32` | 4 | LE | `from_le_bytes` | yes | number |
| `mask-risc` `reference.rs` strided group sum | bytes → `u64` | 1–8 | LE | `b << 8k` | yes | — |
| `ternary_match_strided(16)_to_mask` | bytes | 12 / 16 | n/a | byte compare | neutral | byte arrays |
| `quack` `aperture_facet_strided` | facet bytes → classid `u32` | 4 | LE | `from_le_bytes` | yes | number |
| `contract` `Register128::{words, from_words}` | 16 bytes ↔ 4×`u32` | 4 | LE | `from/to_le_bytes` | yes | — |
| `contract` `NodeGuid` accessors / constructors | 16 bytes ↔ classid, tiers | 2 / 3 / 4 | LE | `from/to_le_bytes` | yes | — |
| `contract` `NodeRow` ↔ `&[u8]` (`as_le_bytes`, `node_rows_from_le_bytes`) | pointer cast | 512 | n/a | all fields `[u8; N]` | neutral | — |
| `contract` `FacetCascade::{as_bytes, ref_from_bytes}` | pointer reinterpret of a native `u32` field | 16 | host | reinterpret | **LE only by compile-time guard** | — |
| `contract` `mul.rs` `transmute` | `u8` discriminant → enum | 1 | n/a | — | neutral | — |
| `quack` `LaneRef::U32` / `I32` / `U64`, dir-sim, report, sap | numeric `Vec` lanes | — | — | no byte decode | numeric | number |

No production ABI is intentionally native-endian. The one reinterpret, `FacetCascade`'s compute lens, is not a stored representation (`NodeRow::edges` is `[u8; 16]`). It is fenced by `const _: () = assert!(cfg!(target_endian = "little"))`, so a big-endian build fails to compile instead of reading a different value. It is reported here, not changed.

`quack/tests/canonical_le.rs` pins the rule:
- `EqU32Strided(0x12345678)` matches exactly the records holding `[0x78, 0x56, 0x34, 0x12]`, never the byte-swapped spelling, for strides 4 and 16 across the group path and the tail.
- The executor and the reference interpreter must agree on it.
- `Register128::from_words` writes the bytes the kernel reads back.

The fixtures are literal byte vectors, so the test does not pass merely on a little-endian host. Disable runs, each red then restored green:
- the ndarray kernel switched to `from_be_bytes` fails both tests;
- the reference reader switched to `from_be_bytes` fails both, with an executor/reference count mismatch.

The second disable first came back GREEN: equal LE and BE row counts let the count comparison tie. The fixture is now asymmetric.

## MEASURED — the proofs (tests)

- `quack/src/bind.rs` tests:
  - A text literal binds with exactly 4 binder calls (table, live plane, field, code).
  - The bound `Query` `==` the hand-written expert `Query`, and their lowered `Program`s are equal.
  - Executing twice makes 0 more binder calls; the live plane excludes the dead row.
  - A typed descriptor (`FieldRef`) binds to the same query.
  - Bind errors are reported in the developer's vocabulary and mint nothing.
- `quack/tests/string_fence.rs` (the representation fence's text half): no text tokens in `lib.rs` outside tests (more than 1,000 lines scanned); `bind.rs` is the only other module.
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
