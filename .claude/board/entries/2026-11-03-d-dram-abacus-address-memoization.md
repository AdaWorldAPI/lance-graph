# D-DRAM-0: address memoization in DR, adaptive Abakus mask lens

**Date:** 2026-11-03. **Status:** PROPOSED / DOCS ONLY. No executable code, public API, performance result or production authorization.
**Plan:** [D-DRAM-0](../../plans/d-dram-abacus-address-memoization-v1.md).
**Coordination:** [EXO-50](https://linear.app/exo-red/issue/EXO-50/agent-message-bus-gpt-xgrok-ccsession-id), new append-only message; issue status is not implementation consent.

## Operator intention

1. After resolving a canonical SoA address under an existing witnessed DR tile/local coordinate, **remember that locator receipt** in the active focus/binding lease for repeated borrow and prefetch. Do **not** copy payload values into DR or append physical locators into `LocalMap::globals` (global identity list).
2. Let ClassView/WideFieldMask select the logical property footprint and let Quack choose existing mask/bit-sliced SIMD algebra through ndarray. Abakus may be useful for arithmetic thresholds (k-of-n), not default AND/OR.
3. Borrow canonical values and use selective Lance 13 source fetch when necessary. Payload gathering remains the D-DRF-1 §5 narrowly admitted foreign-cache exception, never a generic default optimizer choice.

## Verified anchors

- [DR MapTable](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-dt64/src/dr.rs): stable identity mapping, append-only coordinate history, not physical SoA pointers.
- [ClassView](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-contract/src/class_view.rs) and [DT64](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-dt64/src/lib.rs): WideFieldMask borrowed words and ValueDemand separation.
- [D-RMAP-1A](../../plans/d-rmap-separate-tables-thought-palettes-v1.md), [D-DRF-1](../../plans/d-dr64-focus-lease-foreign-cache-v1.md), [D-DRAV-1](../../plans/d-dr-address-first-value-demand-v1.md): owners, borrowed default and admitted exception.
- [ndarray #402](https://github.com/AdaWorldAPI/ndarray/pull/402): half/full adder masks SHIPPED; `Abacus<BITS>` remained LAB/DEFERRED. [ndarray #390](https://github.com/AdaWorldAPI/ndarray/pull/390): masked byte-sum shipped. Operator's exact `ndarray::simd::5x24::*` symbol is **UNVERIFIED**, census first.
- [#1573](https://github.com/AdaWorldAPI/lance-graph/pull/1573), [#1575](https://github.com/AdaWorldAPI/lance-graph/pull/1575), [#1633](https://github.com/AdaWorldAPI/lance-graph/pull/1633): Lance 13 masked reads already measured. [#1788](https://github.com/AdaWorldAPI/lance-graph/pull/1788) / [#1789](https://github.com/AdaWorldAPI/lance-graph/pull/1789): D4, S-INGRESS and physical-byte limitations.
- [#1943](https://github.com/AdaWorldAPI/lance-graph/pull/1943), [#1945](https://github.com/AdaWorldAPI/lance-graph/pull/1945): in-memory DR/FOLD experiments, not Lance I/O.

## Planned gates, not authorized implementations

| Gate | Deliverable | Evidence/owner boundary |
| --- | --- | --- |
| D-DRAM-1 | enumerate actual source locator leases and 5×24 SIMD signatures | READ-ONLY |
| D-DRAM-2 | lab memoized source-location receipt under one immutable map, L1–L10 | EXO-46, EXO-89, EXO-151 |
| D-DRAM-3 | lab k-of-n Abakus vs native mask words, A1–A7 | Quack/ndarray owner; no public Abacus type by default |
| D-DRAM-4 | combined ClassView WHERE, borrowed SoA, Lance source and mask algebra, C1–C4 | D4/S-INGRESS remain GATED |
| D-DRAM-5 | cost/evidence handoff and owner decision, not a merge | EXO-50 append-only bus |

**Refusals:** no second coordinate authority; no historical raw source pointer; no per-query copied payload by default; do not confuse WHERE criterion counts with property widths, SoA lane widths or DR tiles; no unproven physical 4×8 semantics; no new F### / opcode; no SQL UNKNOWN / Bag conflation. All measurements must separate cold setup, witness work, locator miss/hit, Lance bytes, borrow, prefetch and FOLD.

**Board references:** [STATUS_BOARD D-DRAM](../STATUS_BOARD.md), [INTEGRATION_PLANS](../INTEGRATION_PLANS.md). **Related issues:** [EXO-46](https://linear.app/exo-red/issue/EXO-46/a1-decide-counted-globalfocus-binding-membrane-for-production), [EXO-89](https://linear.app/exo-red/issue/EXO-89/d-rmap-1-1a-implicit-relationship-register-and-separate-thought), [EXO-151](https://linear.app/exo-red/issue/EXO-151/d-drf-immutable-dr64-focus-leased-foreign-cache-sparse-writeback), [EXO-121](https://linear.app/exo-red/issue/EXO-121/d-qr-cap-classviewwidefieldmask-relational-isa-binding-capstone), [EXO-140](https://linear.app/exo-red/issue/EXO-140/admission-b-call-sites-unclassified-semijoin-equ32via-gather-and-no).
