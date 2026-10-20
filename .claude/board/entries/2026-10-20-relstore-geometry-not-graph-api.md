# D-RST reframed: relational geometry first, petgraph optional, tile GEMM a consumer (2026-10-20)

**Status:** PROPOSAL (D-RST-0..9). Plan: `.claude/plans/relstore-quack-petgraph-traversal-v1.md`. Documentation only. Supersedes the framing of `2026-10-20-relstore-quack-petgraph-plan.md`; its ownership decisions and findings stand.

## Decision (operator, 2026-10-20, recorded as provenance)

- Petgraph is an optional consumer, intermediate and usability reference, beside the Gremlin adapter. It decides no storage format, coordinate model, algebra, optimizer, execution ABI or lifecycle. D-RST-3 is OPTIONAL / DEFERRED / MEASUREMENT-GATED; petgraph parity is the optional falsifier F-PG.
- Tile GEMM accelerates arithmetic on prepared dense operands. It does not resolve identities, scatter, gather or traverse sparse data.
- Target: minimum repeated work per relational reasoning step across front ends and compatible thought versions.

## Findings (read in source)

- `lance-graph-planner/src/adjacency/` already holds a pattern-only CSR + CSC with edge ids and a borrowed `AdjacencyBatch` that is deliberately not `Clone`/`Copy` (ruling 2026-07-29). Gaps for D-RST-9: `u64` widths, edge id = arrival position (not a stable fact identity), no version pin, binary only. Core `CsrIndex` is a third owned CSR. No fourth is added.
- The D-RST-3 experiment's view (petgraph#4) is `Copy`, which that ruling forbids; petgraph's `GraphRef: Copy` is met instead by implementing the visit traits for `&View`.
- D-RST-3's measurements: a borrowed CSR walk costs about 2 ns per edge against 300–400 ns per edge of preparation; a Quack one-hop scans the whole lane (O(E)) because the sort order is not attested (admission A6).

## Open

- A1–A6, B1–B3 (plan §8). F8 (four-path amortization) is unmeasured; path D runs only through the existing D-RMAP-2 probe kernels.

## Status ruling (operator, later 2026-10-20)

Lab exploration only. The relationship store is one option among many for certain data shapes, unfit to promote until further testing; nothing from D-RST or D-RMAP-2 is admitted to production, and no production code may be rewired to it.

Checked in source: none of the measured paths touched DataFusion. `mork-pathmap-probe`, `lance-graph-quack`, `lance-graph-mask-risc` and the petgraph `relstore-visit` experiment have no DataFusion dependency. BlasGraph and DataFusion are separate `ExecutionStrategy` arms in `lance-graph/src/query.rs`, and the BlasGraph arm returns "not yet wired to input datasets". The measured slowness is the probes' own: an indirect sort plus a second sort in preparation, a full-lane scan with per-query lowering in the Quack hop, and per-element allocation in the BF16 polyfill.
