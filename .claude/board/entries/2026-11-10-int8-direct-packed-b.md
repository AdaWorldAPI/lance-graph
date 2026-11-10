# 2026-11-10 — INT8 direct arm on prepacked INT8 B

**Status:** MEASURED (lab). Needs AdaWorldAPI/ndarray#418.

**What.** `dr-isa-stream-lab`'s INT8 arm now reads A in place and takes B
from bands packed once (`PreparedI8B`, `PackedI8B`), through
`ndarray::simd::int8_gemm_packed_strided`. The prepared-operand reuse rule is
one `shape::ReuseKey` for BF16 and INT8.

**Result** (one AMX VM, 41 warm runs, three runs). At 256 fields the GEMM
goes from 78.3–78.8 µs to 45.5–48.1 µs, end to end from 88.9–92.2 µs to
47.5–57.9 µs; at 64 fields from 3.4–3.6 µs to 0.98–1.01 µs.

**Negative result.** One call per 16×16 tile (99–108 µs) was slower than
`gemm_i8` with its pack: per-call tile configuration and lost 2×2 blocking.
The pack only pays to hoist when the whole-matrix kernel takes it.

**Governing question** (`dr-dt64-locality-execution-contract.md`): the VNNI
pack of B was recomputed per operation; it is now done once per binding and
reused under `ReuseKey` (operand identity and revision, selection, frame,
width).
