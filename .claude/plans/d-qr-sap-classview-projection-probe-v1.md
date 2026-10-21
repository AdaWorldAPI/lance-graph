# D-QR-SAP: ClassView-guided QR256 for deterministic CATS mappings

**Date:** 2026-10-21. **Status:** LAB PLAN / PREREGISTERED PROBES, unimplemented. Rust 1.99. **No SAP connectivity or writes.**

## Thesis

Reuse `lance_graph_contract::class_view::{ClassView, FieldMask, WideFieldMask}` as the structural map of stable CATS DTO field positions, then compile existing positional mappings into a **borrowed QR projection**, not a new SAP evaluator or edge store. QR256 is a typed, bit-packed **256×256 = 8 KiB Boolean relation tile**, only when the demand has two 256-slot domains; a single 23-field mask or 23×row selection should stay a direct mask when cheaper. Store/address metadata is counted separately.

**Existing truths to preserve** (read from main): `lance-graph-sap/src/schema.rs` has 23 append-only `FIELDS`; `bind.rs::CatsBatch` binds cold, keeps null validity and borrowed `LaneRef` execution; `binder.rs::CatsBinder` resolves field names once into Quack; `edge.rs` has `HASH_ORDINALS` (16), `BAPI_ORDINALS` (8), distinct hash profiles, `bapi_sink` as a legitimate explicit wire-boundary materializer.

## Mapping contract

- `ClassView::fields(ClassId)` selects an existing canonical ordered field basis; the original `ClassView` trait is not edited. A separately validated table `SchemaSeal / ClassId / FieldRef.ordinal` ensures `FIELDS[i]` corresponds to the declared ClassView position. If no truthful field identity exists, **refuse** rather than invent one.
- `FieldMask` or `WideFieldMask::full_for(23)` expresses **structural field presence only**, not column data, permission, foreign keys, BAPI order, or Boolean graph relationships. The 23-field case remains in `WideFieldMask::Small` without allocation. Append-only bit positions are immutable.
- **Authorization is independent**: `effective = present ∩ allowed(class) ∩ requested` (cf. #1558 `visible_fields`), but a field mask alone is *not* evidence of an allowed query or cross-table identity.
- One `ProjectionSpec` (working name, not an API) binds source `ClassId`, predicate/method palette identity, `&[FieldOrdinal]` (e.g. 16 HASH or 8 BAPI), schema generation and source batch generation. The format-string metaphor `"{0}{1}" -f $laneA[$i],$laneB[$i]` selects **positions**, never builds strings in the hot path.
- Keep `HASH_ORDINALS` and `BAPI_ORDINALS` as **independent output orders and semantic profiles**. A QR transpose or field permutation cannot rewrite hash normalization, Unicode handling or notes(50) rules. Only the actual hash/BAPI terminal formats strings.

## Falsifier probes (S0→S5)

| Probe | Positive oracle | Mandatory refusal / counterexample |
|---|---|---|
| S0 Census | compare `FIELDS`, `CatsSchema::resolve`, `ClassView`, `HASH_ORDINALS`, `BAPI_ORDINALS` | field renumbering or absent identity claim |
| S1 ClassView presence | 23 bits mapped in canonical order, `WideFieldMask::full_for(23)`, two nullable fields kept distinct | NULL != empty != 0; field mask never authorizes a read |
| S2 Ordered projector | 16-position hash and 8-position BAPI read original `CatsBatch` lanes by witnessed ordinals; byte-for-byte compare with `edge.rs` sinks | reorder one field: parity must break; no silent generated edge |
| S3 Profile laws | canonical hash text and HMAC-SHA512 match **each** independent existing profile | different decimal separators, Unicode uppercase, short/non-ASCII notes refuse where existing code refuses |
| S4 QR choice | projected Boolean membership for 64 and 256 local slots agrees with scalar direct masks; measure cold projection + hot reuse | 8 KiB staging when a single 64-bit field mask suffices is a **negative** cost result |
| S5 Boundary | equivalent `Count/Exists` from Quack+mask-RISC; wire `BapiCatsInsert` only at sink | foreign positional gather during PREPARE forbidden under ADMISSION B |

## Work order and gates

1. S0 inspect #1659 borrowed `Run` + `Side`, #1660 QR and `ClassView` declarations, EXO-119 position-template proposal. Claim no other session's API.
2. S1 prototype a **lab-only** field-lens descriptor over existing CATS `LaneRef`, no new `Vec<Row>` or String runtime formatter.
3. S2 run independent exact oracles against existing `edge.rs` and `query.rs` behaviors. Prove masks/fields stable for optional fields.
4. S3 test QR materialization only for repeated 2-axis questions; compare direct mask, QR64 and QR256 where their axes mean the same thing.
5. S4 measure bind/build, cold dictionary work, index bytes, hot ns/µs mask folds, bytes emitted, and parity; failed ms hot-fold gate is **failed**, not a success.
6. S5 submit measured ADOPT / KEEP-DIRECT / REJECT finding to EXO-50 and update the capstone, **without** production writeback.

**Boundary:** Do not create fake cross-system SAP identifiers, a new SAP graph model, a second evaluator, a new relationship opcode, or speculative BAPI output. No HMAC signing without its existing authorized sink.

**References:** [#1659](https://github.com/AdaWorldAPI/lance-graph/pull/1659), [#1660](https://github.com/AdaWorldAPI/lance-graph/pull/1660), [EXO-119](https://linear.app/exo-red/issue/EXO-119), [EXO-120](https://linear.app/exo-red/issue/EXO-120), [EXO-50](https://linear.app/exo-red/issue/EXO-50), [ClassView](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-contract/src/class_view.rs).
