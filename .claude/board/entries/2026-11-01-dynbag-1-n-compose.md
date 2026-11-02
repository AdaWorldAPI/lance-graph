# D-DYNBAG-1: ℕ-compose for `p(X, Z)`, exact; `matmul_i8_to_i32` is the wrong carrier on this host as called

**Date 2026-11-01 · ID D-DYNBAG-1 · Status IN PR · Plan `.claude/plans/d-dynbag-holographic-bags-v1.md` §7.6 · Crate `crates/xlog-relcomp-probe` (`src/ncompose.rs`, `tests/litho.rs` part 4, `examples/ncompose_cost.rs`)**

- `p(X, Z) :- r(X, Y), s(Y, Z)` is answered as `matmul_i8_to_i32` over two DR64² 0/1 tiles. It equals both oracles (scalar witnesses per (x, z); XLOG's RIR bag walk) on the fixture and on 300 random cases. Its `> 0` reading equals `relcomp::compose`.
- Disable runs red: operands swapped (caught by the differential only); B read before counting.
- MEASURED, AVX-512 VNNI tier (Xeon model 0x55, no AMX; the cost example prints the tier):

  | case | `compose` (B) | `matmul_i8_to_i32` (ℕ) |
  |---|---|---|
  | sparse | 50 ns | about 21 µs per tile |
  | dense | 2.4 µs | about 21 µs per tile |

HYPOTHESIS: the 21 µs is per-call pack/alloc overhead, not arithmetic.

A first version said "no VNNI, scalar tier" from a mis-grepped flag (`avx512_vnni`); this entry is corrected.

**OPEN:**
- A bit-packed i1 × i1 → i32 word (popcount of R[x] ∧ Sᵀ[z]) in `ndarray::simd`, the D-DYNBAG-3 candidate.
- The AMX tier: not on this host; never certify it from a shared VM (ndarray AMX_GOTCHAS 14).
- A non-allocating `matmul_i8_to_i32` entry point.
- Bounds and chains longer than two atoms (refused).
- Cross-tile compose (MISSING-7).
