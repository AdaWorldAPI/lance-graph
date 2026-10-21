# D-QR-CAP: cross-consumer fixture matrix, independent oracles and evidence ledger

**Status:** Preregistered research, not run, 2026-10-21. **Owner coordination:** EXO-121 and EXO-50. Rust 1.99. This document is not a benchmark report.

## Shared minimal fixtures

- **CLASS BASIS:** `ClassId=7`, append-only fields in an ordered `ClassView::fields(7)` basis. `FieldMask` bits 0/2 present, 1 absent. `WideFieldMask` positions 0/63/64/108 distinguish zero-allocation small from >64 wide; same first 64 positions remain equal. One forked mutable schema changes field position 2 to 1 and must **refuse** the previous witness.
- **BIT-PLANE:** `QR64` 64 rows × one `u64` word, total 512 B; `QR256` 256 rows × four `u64` words, total 8192 B. Axis signatures are `Method×Slot` unary or `Source×Target` adjacency, never cast between them without a typed view.
- **RELATION:** R(u,g) = {(0,3),(1,3),(1,4)} and S(g,r) = {(3,7),(4,7)}. Boolean `UserRole(1,7)=true`, while bag count supports two derivations. XOR parity of two routes is FALSE, so XOR is a required incorrect-semantics negative control.
- **IDENTITY:** two sources both expose numeric ordinal `4` but different global GUIDs; any co-ordinal join that infers equal entity from equal ordinal must refuse. Reassign the same local slot under a different generation; stale view must refuse or recompute.
- **ROTATION:** `123/456/789 -> 369/258/147` as 90° CCW `new[r,c]=old[c,N-1-r]`; four repeated quarter turns return the same original bits. A 90° image turn is not generally a graph-isomorphism relabeling.
- **DIRTY:** two different supporting groups grant the same role; deleting one support leaves the role true. `actual_dirty = before XOR after` is ZERO for the role truth despite changed support. Provenance and commit stamps remain separate.
- **EQUATION:** exact `a=2b,b=2c,z=2a -> z=8c` under declared exact arithmetic; under `u8` saturation or floating reassociation no implicit exact-proof claim.
- **E-CLASS:** 256 shared templates × 4,096 contexts = 1,048,576 *requested* applications. Compare sharing vs independent expressions; CSE must not promise information-theoretic compression of arbitrary results.
- **QUANTUM NEGATIVE:** reversible GF(2) steps can invert when schedule recorded, but lossy Boolean `OR` and existential projection require before-support witnesses; nothing here implies quantum compute or speedup.

## Cross-consumer tests

| Axis | SAP #1661 / EXO-122 | dir-sim #1662 / EXO-123 | HubSPO #47 / EXO-124 | stockfish-rs #23 / EXO-125 |
|---|---|---|---|---|
| Mapping ground truth | CATS `FIELDS[23]`, `HASH_ORDINALS[16]`, `BAPI_ORDINALS[8]` | `UserBinder`, group/security chains | `Catalog`, `Table::planes`, owner | `ButterflyHistory` from×to, `feature::make_index` |
| Scalar oracle | existing hash/BAPI and direct Quack | `View`/actors + multiset oracle | existing list filter; synthetic assoc oracle | Shakmaty move generator, independent network refresh |
| Field basis | present vs null; 23-field small | class and attribute key basis; unknown separate | ClassView skins where actually present; synthetic if missing | candidate *geometric analog*; no existing Stockfish ClassView |
| QR role | optional Boolean field/record projection | membership × group × role existence | optional contact-company and list participation | legal move occupancy 512 B QR64 |
| Strong negative | wrong hash/BAPI position, Unicode/decimal | false SID/mail inheritance, 257th focus escape | stale mailbox owner, D-STO-3 not yet shipped | promotion collision from/to, illegal chess symmetry |
| Memory/gate | no row/DTO copy until wire sink | borrowed snapshots/delta, no access bypass | borrowed lanes, no tombstone resurrection | i16 history not converted to bool or FP |
| Cold proof | map/profile/version | group witnesses + snapshot | mailbox+list label+version | feature index/net version + exact delta |
| No-op twin | same generated hash | delete one of two witnesses | unrelated property batch | quiet move whose QR occupancy stays same |

## Metric schema (MANDATORY per candidate)

```text
Source commit, source generation, schema/class/field-basis seal,
ClassView/WideFieldMask width, auth/tenant, algebra kind, axis kind,
number of logical requests, number of distinct normalized expressions,
number of borrowed source rows and source bytes,
T_bind, T_order_attest, T_index, T_QR_materialize, T_hot_fold, T_terminal, T_proof_verify, T_writeback,
RSS_peak, QR_bytes, map_bytes, proof_bytes, allocation_count, cache misses where measurable,
oracle parity/refusal, disable red/green, result: ADOPT | KEEP_DIRECT | REJECT | DEFER.
```

Separate per-operation **nanosecond/µs** hot fold gate from whole 1,048,576-op job runtime. Compare SIMD and scalar exact oracles only when they do not share the same implementation under test. A cross-table value-directed gather is not admitted even if the test produces the right answer.

## Handoff / ownership

- CAP [EXO-121](https://linear.app/exo-red/issue/EXO-121): [capstone plan](d-qr-classview-binding-isa-capstone-v1.md), shared ABI contract and decision log.
- SAP [EXO-122](https://linear.app/exo-red/issue/EXO-122): [lance-graph #1661](https://github.com/AdaWorldAPI/lance-graph/pull/1661).
- IAM [EXO-123](https://linear.app/exo-red/issue/EXO-123): [lance-graph #1662](https://github.com/AdaWorldAPI/lance-graph/pull/1662).
- HubSPO [EXO-124](https://linear.app/exo-red/issue/EXO-124): [HubSPO-rs #47](https://github.com/AdaWorldAPI/HubSPO-rs/pull/47).
- Stockfish [EXO-125](https://linear.app/exo-red/issue/EXO-125): [stockfish-rs #23](https://github.com/AdaWorldAPI/stockfish-rs/pull/23).
- Existing physical/logic work stays with [#1659](https://github.com/AdaWorldAPI/lance-graph/pull/1659) and [#1660](https://github.com/AdaWorldAPI/lance-graph/pull/1660); coordinate over [EXO-50](https://linear.app/exo-red/issue/EXO-50), NOT by overwriting other sessions' branches.

**No result is marked MEASURED merely by opening a plan or obtaining CI green.**
