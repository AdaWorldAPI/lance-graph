## 2026-08-11 — E-JUDGE-THE-FIELD-NOT-THE-ELEMENT-1

**Status:** FINDING `[G]` (crate-stated: `distance.rs:8-12`; `ndarray`
`int8_tile_gemm::int8_gemm_amx_tiled(a_u8, b_i8, …) -> [i32]`).

**Corrects `E-THE-REUSE-IS-THE-PROCESS-AND-IT-EXPOSED-A-FIT-PROBLEM-1` (below):
its 10× measurement stands, its VERDICT is inverted.**

**Operator:** *"you didn't factor in that due to normalized values the field has
different ergonomics than the single value — meaning AMX matmul, tile ops etc."*

**A normalized representation must be judged by what its FIELD does, not by what
one element decodes to.** The 10× scored **angular reconstruction error** — the
operation the substrate exists to avoid, and the exact metric this workspace
already ruled out for scoring a one-way address over a retained original. Third
instance of that error in one arc; this one landed three sections after the rule
was written down.

At field scale the ergonomics run the other way:

| | nearest-`n` | direct `(polar, azimuth)` |
|---|---|---|
| direction collapses to | **ONE index**, 256-palette domain | 3 lanes, one 16-bit **circular** |
| compare | 2 × `DistanceLut` u8 lookups — **L1 metric**, CAKES/CLAM-safe | **not a metric** — `distance.rs:8-10`, the 2π wrap "must never feed CAKES bounds" |
| LUT | 128 KB, L1/L2-resident, **`U8x64`-friendly** | 65536² is not a table |
| tile shape | a `&[u8]` plane → `int8_gemm_amx_tiled` **directly** | none |
| decode to compare? | **no** | **yes** |
| per-point error | 0.972° | 0.097° |

**Resolution — split by OPERATION, not a winner.** Compare/search/correlate a
field → the single-index path (this is what *"pay the inbound tax once"* buys,
and why palette256 is the same pattern one rank down). Materialize one bearing →
the direct path, 10× finer, but *"never reconstruct per element when the
representation is normalized"* makes that the rare path, not the design centre.

**Rule:** a per-element accuracy number is the round-trip metric wearing a
different hat. A representation can win it and simultaneously destroy the
index-domain comparison, the metric guarantee, and the tile shape that made the
substrate worth building. **Ask what the field does under the ops you actually
run — LUT, SIMD lane, tile/AMX — before ranking encodes.**

