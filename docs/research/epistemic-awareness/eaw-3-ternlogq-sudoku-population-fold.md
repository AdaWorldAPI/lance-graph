# EAW-3 — TERNLOGQ × RelAlg: three-plane Sudoku population fold

**2026-10-27. DOCS-ONLY PREREGISTRATION. NO CODE OR MEASUREMENT.** Independent main-based stage. Previous research: [EAW-0 #1820](https://github.com/AdaWorldAPI/lance-graph/pull/1820), [EAW-1 #1821](https://github.com/AdaWorldAPI/lance-graph/pull/1821), [EAW-2 #1822](https://github.com/AdaWorldAPI/lance-graph/pull/1822).

## Scope and semantics

Three *semantic axes* (Identity × Property/Edge × Predicate) are not three *aligned bitmask operands*. DR64³ names 64³ logical possible positions, 262144 bits or 32 KiB only if dense. One DR64² 64×64 mask = 512 B; one 65536-row DR256 mask = 8 KiB; each universe has distinct source and coordinate identity.

Existing [mask-risc fuser](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-mask-risc/src/fuse.rs), [ternlog dispatcher](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-mask-risc/src/ternlog_dispatch.rs), [ndarray SIMD masks](https://github.com/AdaWorldAPI/ndarray/blob/main/src/simd_masking_ops.rs), RelAlg/Quack and D-RU-2 let-bound planes own evaluation. No new ternlog dispatcher, interpreter or physical tensor ABI.

For source-witnessed, co-addressed coordinates x:
- P(x): relationship attested PRESENT, not just a possible combination.
- A(x): typed registered rule's WHEN evaluates TRUE under K3, with separate knownness; UNKNOWN not false.
- E(x): source-declared CausalEdge64 epistemic and activation eligibility, first tested per evidence instance then projected to relationship addresses.
- C_t(x): active candidate population; F_t(x): exclusions proven by source constraints/completeness, not mere non-discovery.

A terminal selection may compute Eligible = TERNLOGQ_AND3(P, A_TRUE, E_TRUE). If C_t also participates, that is **four operands** and needs an additional operation/precomposed plane. Negation on a terminal TRUE-mask is not full K3 negation; version/knownness/provenance remain visible for later reasoning.

**Sudoku-like closure:** repeatedly apply source-declared monotone relational constraints to the current candidate population, calculate Delta = old XOR next, propagate affected co-addressed positions and stop at a verified fixed point. Nonmonotone, recursive, numeric and source-effect rules need their own admitted semantics. One ternlog instruction does not fold the entire 3D cube, and a fixed point need not be source authorization. Multiple typed mappings onto the same polyfill remain distinct while converging rules may lawfully contribute within ONE world.

## Existing evidence and open seams

The [D-DR256-WHERE capstone](https://github.com/AdaWorldAPI/lance-graph/blob/main/.claude/plans/dr256-where-capstone-v1.md) measured stale detached Keep returning 1032 vs fresh 31 source rows, a failure not a universal stale-mask refusal. [EXO-73](https://linear.app/exo-red/issue/EXO-73) recursion driver open; [EXO-140](https://linear.app/exo-red/issue/EXO-140) requires per-call-site classification of Quack Semijoin/Gather under ADMISSION B; [EXO-46](https://linear.app/exo-red/issue/EXO-46) global→focus production binding open. Physical DR³ NFR-OP-2 open.

## Pre-registered falsifiers: separate approved test-only PR, NOT RUN

| ID | Independently required discrimination |
|---|---|
| E3-1 | 2×2×2 even-parity relation has 4 exact triples but FULL SP/PO/SO pairwise projections. No fabricated 8-triple answer. |
| E3-2 | One of three planes uses another source map/generation; reject rather than ternlog aligned-by-ordinal only. |
| E3-3 | K3 UNKNOWN inside AND/NOT must not become TRUE via a TRUE-only WHERE projection. |
| E3-4 | Two simultaneously applicable source mappings use one reusable polyfill; preserve separate binding/provenance and lawful in-world contribution. |
| E3-5 | Two-hop propagation reaches independently computed fixed point, preserving each intermediate, while unstratified negation/cycle refuses. |
| E3-6 | A changed predicate source with cached Keep produces a different exact set, so unwitnessed reuse must refuse. |
| E3-7 | Activation changes derived eligibility without mutating CausalEdge64 certification or relationship presence. |
| E3-8 | An Epi5 32-code u32 selector is deliberately misused as a DR64 4096-row mask; checker detects universe mismatch. |
| E3-9 | Sparse, clustered and scattered inputs and boundaries 63/64/65, 255/256/257 all give exact full-coordinate parity. |

**Four arms:** scalar independently authored triple/K3/fixed-point oracle, 64 densely sliced 512-B DR64 masks, sparse exact slices, co-addressed factorized RelAlg+TERNLOGQ, plus a deliberately wrong shortcut as red control. Report full sets, intermediates, active words/blocks, CPU features, data/compile/bind/scope-version validation, mask traffic, fold time, source population preparation, and cost of any gathers. Measure before asserting speedups, reuse and locality; at least two unlike mechanical worlds and real labelled data for distribution-dependent claims (EXO-109/133). Keep `attest_cost`, `projection_build_cost`, `query_read_cost` separate. N=1/2/4/8/16/64 shared queries are only compatible when every witness matches.

**No promotions:** no new source relationship, new law/certificate, disabled guard, production third-axis, escaped Keep permission or source action. [EXO-127](https://linear.app/exo-red/issue/EXO-127)/[EXO-135](https://linear.app/exo-red/issue/EXO-135) protect law admission. #1812 is operator-rejected and remains quarantined.
