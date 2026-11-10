# 2026-11-10 — DT64-shaped 64×64 and 256×256 BF16 objects (D-DT64-GEMM-SHAPE-1/2)

**Status:** MEASURED (lab, not production admission). Crate:
`crates/dr-isa-stream-lab` (`src/shape.rs`, `src/bin/shape.rs`,
`tests/integration/shape.rs`, `BENCH-SHAPE.md`).

**Question.** Can DT64 turn source-owned lanes into the exact logical shapes
AMX consumes (64×64, 256×256), paying physical layout costs only where
unavoidable?

**Answer.** Yes for region-shaped WHERE results; staging otherwise.
**Decision: MIXED.**
- Direct for a region over a block-padded slab: 0 bytes copied, 0 allocations.
- Staged for scattered masks, interior tails, and permuted or scattered lanes.
- B is always prepared once and reused under a selection/frame/revision/width
  key.
- ndarray CBLAS BF16 is 7–11× slower and is not a consumer for this shape.

**Measured (AMX VM, ±35% between runs):** at 256×256 direct takes 146–178 µs,
staged 140–155 µs (inseparable), B preparation 149–181 µs, and a single-check
resolve 14 µs. A per-field `LaneBinding::lane` resolve takes 155 µs
(quadratic). 256 demanded fields bind directly under
`Extent::DR256_SQ_X64`, with no DT64 change.

**Open:**
- `LaneBinding` has no borrow-all-after-one-check method; the lab works
  around it with `check` plus `addr` plus `LaneSource::borrow`.
- Whether the 2^k slab pitch costs the direct read: odd-pitch runs were
  10–20% faster, inside the noise.
- #1984 is stranded on `ccr-8062bcc7-l7lkz9-vcsr` (merged into an
  already-merged parent).

**Fruit:** a `LaneBinding::borrow_all` (one check, N borrows) in
lance-graph-dt64 would remove the quadratic resolve; it needs its own
DT64 PR.
