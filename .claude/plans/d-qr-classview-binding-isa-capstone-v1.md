# D-QR-CAP: ClassView / WideFieldMask as a typed relational ISA ground plate

**Date:** 2026-10-21. **Status:** CAPSTONE PLANNING PR, research only, **no ABI changes or code integration authorized**. **Rust:** 1.99.  
**Coordination and ownership:** [EXO-121](https://linear.app/exo-red/issue/EXO-121) and [EXO-50](https://linear.app/exo-red/issue/EXO-50). **Physical and algebraic precursors:** [#1659](https://github.com/AdaWorldAPI/lance-graph/pull/1659), [#1660](https://github.com/AdaWorldAPI/lance-graph/pull/1660), [cross-PR PowerShell ordinal bridge](https://github.com/AdaWorldAPI/lance-graph/blob/research/qr256-elimination-rotation-20261021/.claude/plans/qr256-powershell-ordinal-polyfill-bridge-v1.md), [EXO-119](https://linear.app/exo-red/issue/EXO-119) and [EXO-120](https://linear.app/exo-red/issue/EXO-120).

## 1. Problem and smallest solution

Many consumers have meticulous field maps, predicate methods, DTO schemas and ordinal conventions, yet repeatedly handcraft relationship edges and materialized tuples. **Hypothesis:** reuse the existing `ClassView` / `WideFieldMask` bit basis as the **field-selection mapping contract** adjacent to Quack's bind-once numeric boundary; express one *typed* relationship through a borrowed ordinal address view, optionally stage a Boolean QR64/QR256 projection, and use current mask-RISC / ndarray operators. Never require graph objects for an EXISTS/count demand; never stage QR when direct masks are cheaper.

```text
ClassView + FieldRef positions + structural WideFieldMask
        + declared relation predicate / identity / authority
                         |
                    Quack Binder
                         |
               Typed QR Binding Witness
                         |
       #1659 borrowed ordered Side/Run / direct LaneRef
               |                    |
          direct fold         QR64 / QR256 bit plane
               \                    /
                mask-RISC / ndarray Boolean algebra
                         |
          demand-specific Exists / Count / Rows
                         |
      #1660 cold optional law / proof / dirty sidecar
                         |
      independent storage commit (not implied)
```

This **ground plate** is a law of *composition given witnessed premises*, not a claim that an ABI proves the sources true. `Valid(A) ∧ Valid(B) ∧ Compatible(A,B) ∧ Law(F) ⇒ Justified(F(A,B))`, relative to version and authority witnesses.

## 2. Reuse the actual ClassView geometry correctly

Source of truth: [`crates/lance-graph-contract/src/class_view.rs`](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-contract/src/class_view.rs). Existing `ClassView` **trait** resolves `fields(ClassId) -> &[FieldRef]`, ordered N3 append-only. Existing `FieldMask(u64)` covers indices 0..63; `WideFieldMask` has a Small(u64) zero-allocation case and allocates a boxed word buffer when positions 64..255 are used. Its `with/has` selectors are `u8`, so current APIs cannot encode more than 256 distinct positions by an unqualified `u8` field index. `WideFieldMask::words()` borrowed view may retain noncanonical width after an in-place fold; `canonical_words()` yields an equal-value width. Mask width must match a declared class field count/basis.

**Existing ClassView rendering helpers use the legacy `FieldMask`**, including `project`; do not claim that changing `WideFieldMask` automatically widens `ClassView::project`. The lab may read `fields(class)` and use `WideFieldMask` as an **adjacent mapping selector**, without modifying the trait, persistent rows or render surface.

**Semantic boundary:** a field bit means **populated/selected field**, not user↔group membership, a trust value, an edge, a method body, or permission. Treat structural presence, read authorization, relationship existence, numeric evidence and uncertainty as **separate typed operands**.

## 3. Candidate register-binding contract (illustrative only)

```rust
struct QrBinding<'a> {       // pseudotype, not code or committed ABI
    class_id: ClassId,
    field_basis: &'a [FieldRef],   // ClassView::fields(class_id)
    structural: &'a WideFieldMask, // 0..=255 structural field positions
    predicate_id: PredicateId,    // separately declared source→target meaning
    axes: AxisSignature,         // Method×Slot or Source×Target
    algebra: AlgebraKind,        // BOOL_OR_AND | GF2_XOR | I16_WRAPPING | ...
    method: u8,                  // selector into a VERSIONED operation palette
    lanes: BorrowedOrdinalRefs,  // preexisting SoA LaneRef/ScalarBuffer, not copied
    witness: CoordinateWitness,  // class/field/order/identity/authority/version
}
```

No new generic AST or crate is demanded at this stage. `ClassId` is a discriminator not a global entity identity; `WideFieldMask` is a presence/selection mask, not the 256×256 bit plane. Any predicate must explicitly name source class, target class, cardinality, direction, set-vs-bag semantics and codebook/source seals. **Only an admitted predicate plus compatible identity can synthesize a derived edge.**

The PowerShell template `"{0}{1}" -f $laneA[$i],$laneB[$i]` means **compile fixed positional placeholders into borrowed lanes**. No `String`, `Vec<Tuple>` or per-method Rust evaluator in the hot path. For #1659 separate relations, each source uses its **own** `Side::off/pos` and `Run::get` index; equal numeric ordinals across tables do not prove identity. `OrderedLaneWitness` comes from the sealer, never the planner. Respect the [#1640](https://github.com/AdaWorldAPI/lance-graph/pull/1640) ADMISSION B ban on foreign-value-driven cross-table positional gathers even during preparation.

## 4. Register geometries and algebra

| Register | Truth matrix bytes | Useful role | Does NOT store |
|---|---:|---|---|
| `QR64` | 64×64/8 = **512 B** | 64×64 legal moves or small bounded binary relation | Stockfish butterfly scores, NNUE weights, full provenance |
| `QR256` | 256×256/8 = **8 KiB** | typed `Method8×Slot8` unary status OR `Source8×Target8` per-predicate adjacency | original SoA payload, index map, full multiset of edges |
| 256 dense **adjacency predicates** | 256×8 KiB = **2 MiB** | upper bound for an eager bank | lazy/borrowed subframes for sparse demand |
| `WideFieldMask` | 8 B until >64 positions | **class field presence/selection**, up to 256 bits via current `u8` selector | graph adjacency |

For graph composition:
`C[i,j] = OR_k(A[i,k] AND B[k,j])` (existential Boolean semiring).
For GF(2) linear algebra `OR` is **XOR**, which cancels duplicate paths and must not be conflated with graph EXISTS. NNUE needs exact `i16` wrapping/saturating operations as separately defined; a Boolean QR may **select** numeric lanes but cannot replace values. Boolean projection may be lossy for bags, permissions, duplicate evidence, causal interventions and unknown truth values.

A QR tile is **optional**: direct `FieldMask`, Quack `Filter`, co-addressed bitmap or ordered run seek can be strictly cheaper. Every promoted tile requires a measured amortization benefit. Per-tile 8 KiB claim excludes `present/off/pos` index bytes, Arrow buffer handles, metadata and replay sidecars.

## 5. Five PRs, independent ownership

This capstone is **one planning PR**, with four **separate, non-blocking, domain-specific planning/probe PRs**:

| Topic / issue | Independent draft PR | Entry point / oracle | Acceptance boundary |
|---|---|---|---|
| SAP [EXO-122](https://linear.app/exo-red/issue/EXO-122) | [lance-graph #1661](https://github.com/AdaWorldAPI/lance-graph/pull/1661) | CATS 23-field schema, 16 HASH, 8 BAPI ordinals, existing `edge.rs` | exact mapping/profile parity; no SAP writes |
| IAM [EXO-123](https://linear.app/exo-red/issue/EXO-123) | [lance-graph #1662](https://github.com/AdaWorldAPI/lance-graph/pull/1662) | dir-sim `View`, `UserBinder`, group security/mail closure, role oracle | no false identity/privilege; out-of-focus fallback |
| HubSPO [EXO-124](https://linear.app/exo-red/issue/EXO-124) | [HubSPO-rs #47](https://github.com/AdaWorldAPI/HubSPO-rs/pull/47) | existing `Table::planes`, `Catalog`, list masks; **synthetic** assoc until D-STO-3 | no fake production edge parity; mailbox ownership/version |
| Stockfish [EXO-125](https://linear.app/exo-red/issue/EXO-125) | [stockfish-rs #23](https://github.com/AdaWorldAPI/stockfish-rs/pull/23) | 64×64 butterfly, Shakmaty, incremental HalfKA oracle | QR64 does not replace 16 KiB i16 history / 1024-d accumulator |

All four remain lab plans until their owning sessions run genuine independent tests; the capstone does not authorize code changes or merge. Reuse [EXO-107](https://linear.app/exo-red/issue/EXO-107) for the missing row lens, [EXO-114](https://linear.app/exo-red/issue/EXO-114) for order witnesses, [EXO-46](https://linear.app/exo-red/issue/EXO-46) for admission, [EXO-86](https://linear.app/exo-red/issue/EXO-86) for contract law ownership. Avoid duplicate execution infrastructure.

## 6. Milestones / explicit decisions

**C0 Mapping census:** For each consumer, show its existing `ClassView` or candidate field basis, `FieldMask`/Wide allocation and lane ownership. No speculative ClassId minting. If source lacks real ClassView, use an explicitly **synthetic fixture**.

**C1 Typed binding proof:** Stable N3 bit positions; `ClassId+schema digest` and field basis; authority scope and version; per-side witnessed source ordinals. Make false identity, incompatible predicate, stale class shape or ACL produce **typed refusal**.

**C2 Two-axis view:** Start with no QR projection: compare direct source masks. Then optional QR64, QR256, Boolean matmul/ewise/closure backed by existing mask-RISC / ndarray. Count all setup, source bytes, extra memory, cache misses, proof writes and terminal materialization. Gate any ms *hot fold* as failure, not benchmark victory (#1657).

**C3 Proof and epistemic law:** Link accepted algebraic rewrites to a versioned palette8 rule ID. A `before_root`, derived dirty XOR and `after_root` checker must reconstruct independently; authorization and store ACK are not inferred from mathematical validity. Unknown evidence must not silently become false.

**C4 Consumer adoption decision:** Receive each topic's ADOPT (with measured win), KEEP-DIRECT, REJECT or DEFER ruling. A universal mandatory QR materializer is a failure mode. If two consumers merely share geometry but not legal semantics, that is a valid narrower win.

## 7. Operator questions before production

1. Approve **ClassView + WideFieldMask** as a *candidate reusable field mapping basis* only, with `ClassView` unchanged in phase 1?
2. First QR binary axes: `Source8×Target8` per predicate, rather than `Method8×Slot8` unary scheduling? Both can exist, but must not be confused.
3. Owner for the tiny PowerShell-style ordinal selector wrapper [EXO-119](https://linear.app/exo-red/issue/EXO-119) and ordered write-path policy [EXO-114](https://linear.app/exo-red/issue/EXO-114)?
4. Authorization to implement a *separate* lab after the docs/probes are reviewed, while #1659/#1660 remain independently owned?
5. Keep the cold Thought Git replay/writeback and all external effects behind their existing approval/commit protocol (recommended yes)?

**Stop conditions:** no custom edge generator eliminated, any wrong bit, tuple/object materialization in Count/Exists, law mismatch, lost multiplicity, forged identity, stale permission, unadmitted gather, fake NNUE equivalence, wrong checksum or a commit claimed without persistent acknowledgment.

**Success is smaller architecture, not a larger framework.**
