## 2026-08-26 — E-THE-HELIX-POLE-PENALTY-IS-THE-POLAR-BYTE-NOT-THE-CODEC-AND-THE-SPRITE-DECODE-NEVER-TOUCHES-THE-GEOMETRY-1 — measured: BOTH carriers degrade toward the pole, but helix24's term is BOUNDED (∝ y) while helix48's polar byte DIVERGES (∝ 1/r); helix24's cap wins on index and loses on angle; and three of my own claims were wrong

**Status:** FINDING [MEASURED] — three probes over the shipped `helix`
public API, run in a copy OUTSIDE the repo (no crate mutated), N ∈
{240, 1024, 4096}. Includes **two retractions of my own first-pass
metrics** and a **partial confirmation of an operator recollection**
whose attribution turned out to be the part that was off.
**Confidence:** High for the polar-byte numbers and the mechanism (the
mechanism is derivable in closed form and the measurement matches it);
the recollection reconciliation is a plausible match, not a proof of
what was originally measured.

### The geometry, and why a pole penalty is structural

`u = (n+0.5)/N` · `r = √u` (equatorial radius) · `y = √(1−u)` (pole
distance). Small `n` = **pole** (r→0, y→1); large `n` = **equator**
(r→1, y→0) — equal-area, so most indices land near the equator.

`Signed360.polar` stores **|y| in 7 bits, uniformly**. But
`lat = asin(y)`, so `d(lat)/dy = 1/√(1−y²) = 1/r` — a uniform step in
`y` is a **large** step in latitude as `r → 0`. The penalty is not a
bug; it is what uniform quantisation of `y` *means*.

**Measured, N = 4096:**

| band | count | Δlat mean | Δlat max | Δy mean | Δy max |
|---|---|---|---|---|---|
| **POLAR** 90–60° | 1024 | **0.476°** | **5.031°** | 0.00197 | 0.00394 |
| MID 60–30° | 2048 | 0.165° | 0.440° | 0.00197 | 0.00394 |
| LOW 30–10° | 900 | 0.122° | 0.257° | 0.00197 | 0.00393 |
| **EQUATOR** 10–0° | 124 | **0.114°** | **0.225°** | 0.00197 | 0.00388 |

**Δy is flat** at 0.00197 = 1/508 (half of 1/254 — textbook uniform
7-bit) in every band, which is the control: the *carrier* is uniform,
the *meaning* is not. **Δlat is 4.2× worse at the pole in the mean,
22× in the max**, and sharpens into the cap: innermost 16 = **2.391°**,
innermost 4 = **1.205°**, innermost 1 = **0.633°**.

### helix24's cap wins on INDEX and loses on ANGLE — and it is not exempt

`helix24` = `ResidueEdge` (3 B); `helix48` = `Signed360` (6 B) — the
crate's own words: *"the 24-bit hemisphere `ResidueEdge` **doubled to
48 bit**"*. Measured on the cap, N=4096:

| cap | Δidx %N | Δlat | Δlat %90 |
|---|---|---|---|
| 1 | 0.00 % | 0.000° | 0.00 % |
| 4 | 0.01 % | 0.145° | 0.16 % |
| 16 | 0.02 % | 0.162° | 0.18 % |
| 256 | 0.10 % | 0.180° | 0.20 % |
| **ALL n** | **0.17 %** | 0.130° | 0.14 % |

**The cap beats the codec average on INDEX error (0.00–0.10 % vs
0.17 %) and LOSES on ANGULAR error (0.145–0.180° vs 0.130°).** Both
follow from the same fact and the units decide which you see:
`ResidueEdge` has no polar byte, so latitude reaches it only through
`z = arctanh(r)`, which **stretches** as r→0 — consecutive indices
separate widely in aligned-space (finer *index* recovery) while each
z-bucket spans more *latitude* (coarser *angular* recovery).

**⊘ CORRECTION (codex P2 on #1040, before merge).** An earlier draft of
this entry said helix24 "never had" a latitude-dependent error term.
**That is false**, and the correct derivation is short: with `u = r²`,
`z = atanh(r)` ⇒ `dr/dz = 1 − r²`, and `lat = acos(r)` ⇒
`dlat/dr = −1/√(1−r²)`, hence

> **`dlat/dz = −√(1−r²) = −y`**

so equal-width `z` quantisation gives a latitude error **∝ y** —
**largest at the pole**, vanishing at the equator. The measured clean-
regime trend is exactly that: polar **0.167°** → mid **0.122°** → low
**0.067°**. So **both carriers are worst at the pole**; what separates
them is severity class, not presence:

| carrier | latitude term | at the pole | bound |
|---|---|---|---|
| helix24 (`arctanh(r)`, 8-bit) | `dlat/dz = −y` | maximal | **bounded**, ‖·‖ ≤ 1 |
| helix48 (`polar` byte, 7-bit `y`) | `dlat/dy = 1/r` | maximal | **divergent**, → ∞ |

The equatorial 0.422° / **4.317°** in the band table is a *different*
effect — the designed floor-range saturation of the top ~1 %
(`new()` seeds `hi = aligned(0.99·N)`) — not the `z`-quantisation law,
and it must not be read as the clean-regime trend. helix24's index error is otherwise pure
bucket quantisation — 0.14–0.20 %, i.e. `1/(2·256)`, **flat in N** —
with a mild trend the *same* direction (polar band 0.18 % vs low band
0.08 %).

**So the DIVERGENT pole penalty is a property of the POLAR BYTE, and
the polar byte exists only in helix48.** Doubling 24→48 bits buys the
sign and the full sphere; it also replaces a **bounded** pole term
(`∝ y`, already present in the 24-bit carrier) with an **unbounded**
one (`∝ 1/r`). The upgrade is not free in angular fidelity near the
axis — which is the whole point of the removable-singularity note
below.

### ⊘ Two metrics from my own first pass, retracted

1. **"helix48 index error = 0.00 % in every band" was vacuous.**
   `sprite_replay`'s decode recovers `n` from the **azimuth field
   alone**, and the azimuth is injective here — measured: **4096
   distinct u16 values over 4096 indices**. So it tests a 16-bit index
   surviving a 16-bit round-trip, never consults `rim` or `polar`, and
   **cannot show latitude dependence by construction**. A future
   session reading that module will meet the same trap: perfect
   recovery there is not a geometric result.
2. **Angular error as a per-band mean was an artefact.** Consecutive
   `n` sit ~137.5° apart (golden angle) *by design*, so ANY index miss
   yields a huge angular separation; the 11–78° figures were φ-scatter,
   not reconstruction error. Index ordering is not spatial adjacency on
   a φ-spiral — a fact the codec depends on, which makes the naive
   metric worse than useless.

### Reconciliation with the operator recollection

Recalled (2026-08-26, flagged as memory): *helix24 3–6 % on the pole
cap; polar signed360 1.6; equator 0.3–0.6.*

- **Shape: CONFIRMED.** Pole markedly worse than equator. The recalled
  ratio (1.6 vs 0.3–0.6 = **2.7–5.3×**) brackets the measured **4.2×**.
- **Magnitude: matches the POLAR BYTE's cap.** Recalled 1.6 sits
  between the measured innermost-4 (1.205°) and innermost-16 (2.391°);
  recalled 3–6 % against the polar band **max 5.031° = 5.6 % of 90°**.
- **Attribution: this is where it does not hold.** The 3–6 % was
  recalled for **helix24**, and helix24's pole cap measures 0.00–0.10 %
  (index) / 0.000–0.180° (latitude) — its *best* region, 20–100× below
  the recalled figure under all four readings tried. The magnitudes
  belong to **helix48's polar byte**.
- **Unreconciled:** the recalled equator 0.3–0.6 against a measured
  0.114° mean / 0.225° max. Recorded as open rather than explained
  away.

### Consequence

If a consumer needs uniform *angular* resolution rather than uniform
`y`, the polar byte is the wrong quantisation for it — equal-area in
`y` is deliberately NOT equal-angle, and near the pole the two diverge
by 1/r without bound. That is a **design trade to state**, not a defect
to fix: equal-area is what makes the φ-spiral lift uniform in the first
place. Anyone reading a latitude off `Signed360.polar` near the pole
should know it carries ~0.5° mean and up to ~5° worst-case error there,
against ~0.11° at the equator.

Cross-ref: `helix::residue` (`ResidueEdge` / `Signed360` /
`ResidueEncoder`), `helix::placement::HemispherePoint::{lift,
signed_lift}`, `helix::fisher_z` (`hyperbolic_depth = 2·arctanh(r)` —
the spatial flavour), `E-PALETTE256-IS-A-NEEDLE-…-1` (same session: the
address/relation/meaning separation this measurement sits inside).

