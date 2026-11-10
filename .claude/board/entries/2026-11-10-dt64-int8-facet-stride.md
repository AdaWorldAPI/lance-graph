# 2026-11-10 — INT8 over the DT64 virtual object; FACET STRIDE audit

**Status:** MEASURED (lab). `crates/dr-isa-stream-lab/src/int8.rs`,
`BENCH-SHAPE.md` § SHAPE-3.

**INT8 arm.** `u8 × i8 → i32` through `backend::gemm_i8` (AMX TDPBUSD). A is
unsigned and B signed, with no reinterpreting cast (a test pins it). The
results satisfy `C_int8 = C_bf16 + 8·colsum(B)` exactly.

**At 256 fields:**

| | INT8 | BF16 |
|---|---|---|
| Warm end-to-end | 85–101 µs | 144–160 µs |
| GEMM | 216–228 GMAC/s | 110–113 GMAC/s |

**Correction.** #1987's "B preparation" timed the synthetic hash. Read from a
table, prepared B costs 46–62 µs (not 149–181 µs).

**Audit verdicts:**

| Operation | Verdict |
|---|---|
| Positional reading (BF16) | REUSED |
| DT64 routing | REUSED |
| AMX accumulation | REUSED |
| LUT / masked fold | NOT APPLICABLE |
| VNNI packing | NOT APPLICABLE (mandatory) |

**OPTIMIZATION FOUND (lab-local):**
- region A copy: 3.5–6× faster;
- in-place B: 30 ns instead of 58 µs.

**Open:** ndarray has no strided-A INT8 entry and no prepacked INT8 B (the
twins of #414 / `PackedBf16B`), so INT8 copies A and re-packs B on every
call. Measured apart: the pack is 23–29 µs (36–38% of the INT8 GEMM) and the
A copy 5–7 µs, together about 30–45% of INT8 warm time. A separate ndarray
PR is proposed.

**Prepared-B reuse** is keyed on object identity, full selection equality,
frame issue, width and operand identity (seed or table instance, plus
revision). There is no digest.
