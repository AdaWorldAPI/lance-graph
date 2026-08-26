## 2026-08-11 — E-TOPOLOGY-PICKS-THE-TABLE-NOT-THE-DOMAIN-1

**Status:** FINDING `[G]` — `DistanceLut::circular()` proven a metric
EXHAUSTIVELY over all 256³ = 16 777 216 triples (`distance.rs`, 3 tests).

**Supersedes both `E-THE-REUSE-IS-THE-PROCESS-…-1` and
`E-JUDGE-THE-FIELD-NOT-THE-ELEMENT-1`.** Operator: *"distance.rs is normalized
[a,b] amortizing in LUT."*

**The LUT is the AMORTIZATION POINT, not merely a metric or a cache.**
`quantize()` normalizes `[a,b]` once per element at ingest; `from_floor()` folds
the *same* normalization into the table. Afterwards: no bounds, no division, no
per-element normalization — a pure index lookup, in **unit-free** units, which is
exactly what licenses cross-variable comparison. O(256²) once, not O(N²).

**Consequence:** if the LUT amortizes *any* bounded `[a,b]`, a **circular** range
is just another bounded range with a different formula. Built and proven:
`circular()` = `min(|a−b|, 256−|a−b|)`, the cycle-graph geodesic on `Z_256` —
**0 violations / 16 777 216 triples**, symmetric, identity, positive. Falsifier:
`d_circ(255,0) = 1` vs `d_linear(255,0) = 255`.

**So the crate's *"raw-azimuth is NOT a metric (the 2π wrap)"* is about the
FORMULA, not about angles.** `linear()` is the wrong table for a ring;
`distance_heuristic` uses no table at all. A wrapping quantity in the 256-palette
with the circular table is metric-safe and stays in the index domain.

| azimuth as | resolution | metric? | field shape |
|---|---|---|---|
| u16 raw + `linear()` | 0.0055° | **no** | not tileable |
| **u8 palette + `circular()`** | **0.352° mean** | **yes** | `&[u8]` · 128 KB L1 · `U8x64` · AMX |
| nearest-`n` | 0.972° mean | yes | single index |

The palette azimuth beats nearest-`n` **and** keeps the field ergonomics. The
previous entry was right about the ergonomics and wrong to treat them as
disqualifying: the fix was never "abandon the direct path", it was **give the
wrapping lane its own table**.

**Rule:** **a bounded quantity's TOPOLOGY selects its table formula; it never
decides whether the quantity belongs in the palette domain.** Amortization, L1
residency, `U8x64` lane and AMX plane are identical either way — that is what
makes the substrate general rather than per-quantity.

