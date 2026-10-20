# Stored relations, compiled traversal, petgraph over borrowed layouts (2026-10-20)

**Status:** PROPOSAL (D-RST-0..8). Plan: `.claude/plans/relstore-quack-petgraph-traversal-v1.md`. Documentation only.

## Decision

Operator decision B: `RelationshipMapStore` owns relationship persistence, identities, layouts, coordinate-domain seals and invalidation; Quack owns compilation and numeric query semantics; mask-risc on `ndarray::simd` stays the only population executor. D-RMAP-1A §1 and the 2026-10-05 storage-portability ruling stay in force.

## Findings (read in source)

- Multi-hop over a large domain is blocked by scatter as well as gather: pushing a frontier writes at positions taken from values, pulling reads at them. The tile (frontier broadcast ∧ a pre-transposed plane) is the only admitted shape, and it is bounded by the focus. Gremlin's `NonFunctionalChain` is therefore two problems: a phase handoff (the store's) and a hop (the ruling's).
- The petgraph fork's `experiments/lance-fold-hop` consumes the quarantined `OutgoingHop` (pinned pre-quarantine); it is not a base for this work.
- petgraph `Bfs`/`Dfs` need only `GraphRef + Visitable + IntoNeighbors`; the visit map is a `FixedBitSet` keyed by `IndexType`, so a dense-slot view reuses it unchanged.
- Quack's `Registrar` has only a test implementor; three library `Binder`s exist (sap, dir-sim, glove-parity).
- A coordinate domain stored as a relation gets its own `DurableRef`, which removes domain ABA without a counter.

- BlasGraph's shipped CSR (read in source): 2 KiB `BitVec` value per entry, `usize` indices, `mxv` clones the matrix per call (even the CSC transpose clones), `mxv` reads the vector at stored column indices (a scalar gather), and ndarray is not on the CSR/`mxv` path. The store reuses BlasGraph's storage types with a pattern-only `u32` borrowed variant; dense row blocks become bitmap rows executed by ndarray mask ops through mask-risc (`StorageFormat::Bitmap` is already reserved); only sparse rows need an indexed read.
- Principle (operator): write a relationship as a relationship; layouts are built at seal time from relation facts, so RelAlg pays translation once per version.

## Open

- Admissions A1 (global→local binding), A2 (large-domain multi-hop), A3 (scalar petgraph iteration), A4 (store home/backend), A5 (Registrar shape); blockers B1 (`DurableRef`/`ThoughtCommitId` types), B2 (fork pin), B3 (cyclic joins at scale).
