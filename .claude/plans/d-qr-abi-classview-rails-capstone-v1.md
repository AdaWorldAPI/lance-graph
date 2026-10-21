# D-QR-ABI-0: ClassView/WideFieldMask-backed QR64/QR256 relational register capstone

**Date:** 2026-10-21 · **Status:** CAPSTONE PROPOSAL, four independent consumer probes, NO production code or ABI ratification · **Rust:** 1.99.  
**Motivation:** Replace handrolled domain-specific edge constructors with *derived relationships* from existing schema, mapping, identity, and algebra contracts. This is not a new graph engine.  
**Prerequisites:** [#1659 physical borrowed ordinal/prefetch lab](https://github.com/AdaWorldAPI/lance-graph/pull/1659); [#1660 algebra/replay and QR research](https://github.com/AdaWorldAPI/lance-graph/pull/1660); [#1659→#1660 PowerShell positional mapping](https://github.com/AdaWorldAPI/lance-graph/blob/research/qr256-elimination-rotation-20261021/.claude/plans/qr256-powershell-ordinal-polyfill-bridge-v1.md); [EXO-119](https://linear.app/exo-red/issue/EXO-119); [EXO-120](https://linear.app/exo-red/issue/EXO-120).

## 1. The architectural law to test

A derived edge is admitted when its predicate, roles, coordinate identities, authority, source versions and algebraic laws are **declared and independently witnessed**. For `Member(User,Group) ∘ Grant(Group,Role)`, `UserRole(u,r) := EXISTS g, Member(u,g) AND Grant(g,r)`; an emitted `UserRoleEdge` object is optional. No premise => no claim. TRUE, FALSE and UNKNOWN/unobserved must remain distinguishable in the result contract.

```text
existing ClassView + (Wide)FieldMask + ontology/OGAR predicate and codebooks
    -> Quack bind once (source class/field/ordinals, identity, scope, version)
    -> #1659 borrowed ordered lane / off+pos / optional prefetch
    -> typed local QR64/QR256 Boolean register, OR/AND via existing mask-risc/ndarray
    -> Count/Exists as folds; Rows as explicitly demanded borrowed lens
    -> #1660 cold derivation DAG + BEFORE/AFTER and actual dirty
    -> any persistent edge/write ONLY at a separately authorized materialization terminal
```

**A mask is structural geometry, not epistemic truth.** `ClassView::fields(ClassId)` identifies an append-only ordered field basis; `FieldMask` marks positions 0..63 and stays `Copy`; `WideFieldMask` adds optional higher positions, with `Small(u64)` allocation-free below 64, borrowed `words()`/`canonical_words()` and in-place folds. These masks neither prove a relation exists nor grant read/write permission. Per-consumer `ClassId` and codebook identities require explicit schema/identity witnesses, not a guessed integer match. `WideFieldMask`'s per-position setter uses `u8`, so the first experiments stay at most 256 field-position IDs; `full_for` supports wider counts but that does not make 16-bit field ordinals automatically addressable.

## 2. Reuse the existing horizontal ISA/PART_OF rails

The physical [`FacetCascade`](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-contract/src/facet.rs) is **16 bytes**: `facet_classid(4) | 6×(hi:u8, lo:u8) = 12 bytes`. Its bytes are content-blind. The class-conditioned reading may be `(part_of:is_a)`, `(group:member)`, `(column:row)`, or another declared 256:256 palette pair. `hi_chain` and `lo_chain` each offer six ordered u8 references; `FacetTier.lo` is the low byte and `FacetTier.hi` the high byte. The current `hhtl::NiblePath::from_guid_prefix_v3` explicitly routes **both** part_of/high and is_a/low bytes, not just one.

**Horizontal register candidate:** one declared `8:8` pair selects two local `u8` coordinate codes in a typed 256×256 focus (max 65,536 pair positions). Reuse `FacetCascade`'s already-carved bytes and validated class interpretation for address binding rather than minting a second `EdgeId` encoding. `6×2×8` is a six-level **address/candidate-label budget**, not a representation of arbitrary 256×256 truth; the 8-KiB Boolean QR tile still holds existence bits separately. The 4-byte class prefix also remains separate from `ClassView::ClassId = u16`; do not conflate encodings without a checked adapter.

**Vertical HHTL16 zipper candidate (operator hypothesis, NOT yet the existing ABI):**

- ISA zipper: **16 × 8-bit** logical references, a class/subtype axis, like a distinguished-name chain.
- PART_OF zipper: **16 × 8-bit** logical references, an orthogonal containment/location axis.
- This is a proposed *coordinate/view and ancestry scheme*, not a claim that two 16-byte zipper arrays are already packed into one 16-byte `FacetCascade`. The current persistent 6-tier facets and 16-nibble `NiblePath` router are different shapes; the 16-nibble router has **4-bit steps**, while the suggested HHTL16 zipper uses **8-bit steps**. A proof of an injective/resolved mapping is required before promoting a new representation.
- Do not equate `is_a` inheritance with `part_of` transitivity or automatically compose them. Each typed predicate supplies allowed chaining, scope and law; reflexivity, cycles, unknowns and external focus require explicit rules.

Proposed **view-only** adapters: `FacetCascade → (isa_rail[6], partof_rail[6])`; `(isa_rail, partof_rail) → witnessed LocalSlot8` for a bounded focus; optional `HHTL16ISA/PartOf` expansion must carry omitted/extra path segments and depth/escape evidence. Reject truncated paths and codebook collisions, never silently alias global IDs with 8-bit local slots.

## 3. Minimal candidate ABI record (illustrative, not committed Rust)

```text
RelFrame {
  class_basis: ClassViewClassId + ordered FieldRefs + WideFieldMask,
  axes: typed (SourceKind, TargetKind), relation: PredicateId,
  coords: bounded QR64 or QR256 + local/global codebook witnesses,
  rails: [(PartOf8, IsA8); 6] | witnessed zipper view (candidate),
  borrow: sourceGeneration, sourceOwner, off/pos or direct ordinal lane,
  algebra: BoolOrAnd | GF2Xor | I16Wrapping | other exact typed law,
  method8: u8, scope: AuthorityWitness,
  replay: cold Before/After/Dirty/RuleWitnessIDs
}
```

`FieldMask` is **per-instance populated fields** and `WideFieldMask` is the widening companion, not a Boolean adjacency matrix. QR64 = 512 bytes of **truth**; QR256 = 8 KiB of **truth**, plus address map, witness, scratch, dirty and cold replay storage. A single QR tile supports one **typed predicate/axis interpretation**. A method×slot unary register cannot be misread as source×target adjacency. A bank of 256 dense QR256 binary relations would consume **2 MiB** of truth bits, not 8 KiB.

## 4. Four independent consumer PRs

| Topic | Repository | Existing seam | First independent oracle | Explicit non-goal |
|---|---|---|---|---|
| SAP/CATS | lance-graph | 23-field `CatsBatch`, Quack `CatsBinder`, `HASH_ORDINALS` 16, `BAPI_ORDINALS` 8 | same exact source hash/BAPI projections, with separate hash-format profiles | No SAP network/writes, no false claim that every 23-field mask needs QR256 |
| IAM/dir-sim | lance-graph | independently ordinalized user/group SoA, `View`, Quack `UserBinder`, nested groups | current `View` membership and RBAC behavior plus scalar multiset oracle | No automatic permission from presence; no live AD/Entra changes |
| HubSPO | HubSPO-rs | `Catalog` Binder/Registrar; `Table::planes`, stable ordinals, owner/version; list masks | existing live/list-mask oracle and independent **synthetic** association oracle | D-STO-3 association edge lanes are still queued; no fake shipped implementation |
| Stockfish | stockfish-rs | 64×64 butterfly history; HalfKA exact incremental refresh; shakmaty legality | QR64 64×64 move mask vs shakmaty, numeric i16 parity vs current code | NNUE is not a 64×64 Boolean net; never replace Stockfish/shakmaty or i16 with OR/AND |

These PRs are **draft documentation with preregistered tests**, not claims of implementation. Consumer teams may reject QR entirely if existing direct masks or ordinals are cheaper.

## 5. Cross-repo proof and cost gates

**Structural:** class field positions append-only and correctly mapped, `WideFieldMask::Small` path unchanged, >64 fields test promotion, `canonical_words` width normalization. Field presence ≠ relation truth ≠ authorization.

**Identity:** two independent rows sharing offset 5 but different identity MUST refuse; borrowed `LaneRef` / `ScalarBuffer` payload unchanged; source-specific `off/pos` preserve row order, duplicates and versions.

**Rail laws:** enumerate 256 u8 values for `hi` and `lo`; verify `u16` pair pack/unpack and LE `FacetTier` parity, six-level priority, class decoding, no missing/invalid identifier alias. Test ancestry prefix/sibling, cycles, cross-focus and differing `part_of` vs `is_a` conclusions. HHTL16 view must return REFUSE/continuation when it cannot represent original evidence.

**Algebra:** two paths to one target are TRUE under Boolean OR but 0 under GF(2) XOR; bag multiplicity preserved separately; numeric NNUE uses wrapping i16, not Boolean. E-graph equalities do not grant authority.

**Geometry:** rotation `123/456/789 → 369/258/147` (CCW 90°), four turns recover identity. Matrix rotation is not a valid directed-graph relabel unless explicitly proven.

**Persistence:** `CandidateDirty`, `ActualDirty = BEFORE XOR AFTER`, `CommittedDirty` and actual store acknowledgement separated. No cross-table foreign-value-driven gather even during preparation under [#1640 ADMISSION B](https://github.com/AdaWorldAPI/lance-graph/pull/1640). Data never committed on an unverified/stale witness.

**Economics:** benchmark `T_total = T_binding + T_attest + T_indices + T_QRprojection + K*T_fold + T_terminal + T_replay (+ T_commit)`. Report source bytes, cache misses, RSS, tuple allocations, method/DAG nodes, and incremental invalidation. Include direct-mask baseline, QR64, QR256, reused index and no-sharing control. Nanosecond *word fold* and millisecond *whole batch* must not be conflated.

## 6. Stage plan, decisions and ownership

1. **CAP-0 map** exact existing `ClassView`, `FacetCascade`, HHTL, Quack binder and #1659 code; classify each consumer as stable/shipped, queued or hypothetical; no duplicate schema vocabulary.
2. **CAP-1 law declaration** formulate typed coordinate/operand agreement, known/unknown semantics and witness validity. Only **paper** ABI until cross-consumer conformance succeeds.
3. **CAP-2 four probes in parallel** consumers use existing oracles and send parity/negative findings to EXO-50. Do not claim other sessions' implementation.
4. **CAP-3 horizontal rail test** `hi:lo` → u8×u8 local coordinates; compare with indexed QR plane. No new EdgeId payload.
5. **CAP-4 vertical zippers** propose 16×8 logical ISA and PART_OF arrays only as independently checked *views*, proving lossless reachability before storage design.
6. **CAP-5 algebra/proof** compare QR Boolean, direct mask, borrowed map, changed versions; optionally attempt bounded egglog and typed GF(2) separately per #1660.
7. **CAP-6 economics/conformance** choose ADOPT/KEEP-DIRECT/REJECT with actual per-consumer cost/line-removal evidence. ClassView field masks + tiny ABI can be promoted only by a separate operator-admission ruling.

**Open decisions to EXO-50:** which first axis `Source×Target` vs `Method×Slot`; whether horizontal `6×(part_of:is_a)` should remain typed **facets** or become an ISA addressing view; whether two 16-byte HHTL16 zippers are logical views or persisted new encodings; who owns the cross-repository positional-template lens; whether attested sorted write-side indices are permitted under EXO-46/114; consumer permission to prototype independently. This plan asserts **none** as already ratified.

**Primary anchors:** [`class_view.rs`](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-contract/src/class_view.rs), [`facet.rs`](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-contract/src/facet.rs), [`hhtl.rs`](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-contract/src/hhtl.rs), [`ordered_lane.rs`](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-contract/src/ordered_lane.rs), [EXO-50](https://linear.app/exo-red/issue/EXO-50), [EXO-119](https://linear.app/exo-red/issue/EXO-119), [EXO-120](https://linear.app/exo-red/issue/EXO-120).
