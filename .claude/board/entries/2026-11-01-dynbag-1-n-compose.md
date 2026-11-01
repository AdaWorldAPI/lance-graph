# D-DYNBAG-1: ℕ-compose for `p(X, Z)`, exact; a dense i8 GEMM is the wrong carrier on this host

**Date 2026-11-01 · ID D-DYNBAG-1 · Status IN PR · Plan `.claude/plans/d-dynbag-holographic-bags-v1.md` §7.6 · Crate `crates/xlog-relcomp-probe` (`src/ncompose.rs`, `tests/litho.rs` part 4, `examples/ncompose_cost.rs`)**

- `p(X, Z) :- r(X, Y), s(Y, Z)` is answered as `matmul_i8_to_i32` over two DR64² 0/1 tiles. It equals both oracles (scalar witnesses per (x, z); XLOG's RIR bag walk) on the fixture and on 300 random cases. Its `> 0` reading equals `relcomp::compose`.
- Disable runs red: operands swapped (caught by the differential only); B read before counting.
- MEASURED, scalar tier (no VNNI, no AMX):

  | case | `compose` (B) | `matmul_i8_to_i32` (ℕ) |
  |---|---|---|
  | sparse | 50 ns | about 21 µs per tile |
  | dense | 2.4 µs | about 21 µs per tile |

**OPEN:**
- A bit-packed i1 × i1 → i32 word (popcount of R[x] ∧ Sᵀ[z]) in `ndarray::simd`, the D-DYNBAG-3 candidate.
- The AMX and VNNI tiers, not measured here.
- Bounds and chains longer than two atoms (refused).
- Cross-tile compose (MISSING-7).
