# KNOWLEDGE: Cartesian pseudo-helix vs Fisher-2z normalized — when each is right

## READ BY: family-codec-smith, palette-engineer, certification-officer,
##          truth-architect, and ANY session that encodes/decodes/renders an
##          orientation, normal, or direction — in `crates/helix`, in a consumer
##          viewer (q2 `/helix`, splat/surfel renderers), or in a bake.
## Companion to `crates/helix/KNOWLEDGE.md` (the canonical place/residue spec).
## Status: FINDING. Born from the q2 `/helix` session (2026-06-28) where the
##         author repeatedly reached for the wrong one. Append corrections; do
##         not delete.

---

## The two things that get conflated (they are NOT the same codec)

| | **Cartesian pseudo-helix** | **Fisher-2z normalized (THE helix)** |
|---|---|---|
| crate | `ndarray::hpc::splat3d::helix_orient` | `lance-graph::crates/helix` (`Signed360` / `ResidueEdge`) |
| what it is | golden-spiral **cap cascade** (cap π → 0.40 → 0.03), nearest-codebook encode, `decode()` = chained Rodrigues rotations → a Cartesian `(x,y,z)` unit vector | equal-area `√u` hemisphere + Fisher-Z (`arctanh`) alignment + rolling-floor 256-palette → **normalized endpoint indices** |
| wire | 3 bytes/level (1–3), self-contained | `Signed360` = 6 bytes `[rim.start, rim.end, floor_ver, polar, az_lo, az_hi]` |
| place / HHTL | **place-blind** (global frame, no trie coupling) | **place-coupled**: `encode(place, n)` anchors `rim.start_idx` on the HHTL path via `CurveRuler::from_place` ("normal ON TOP of HHTL") |
| distance | none — you must decode to compare | **metric-safe L1** on the endpoint order via the 256×256 `DistanceLut` (triangle inequality holds) — compare WITHOUT decoding |
| the point | the value **IS** the reconstructed vector | the value is that you **never reconstruct** |

The "pseudo" matters: `helix_orient` is *helix-shaped* (a golden spiral) but it is
**not** the Fisher-Z place/residue helix. They share an aesthetic, not an algebra.

---

## The one question that decides it: **how many times will you materialize?**

> Materialize = turn the code back into a float `(x,y,z)` (or run `atanh`/`tanh`
> on the rim, or `sin`/`cos`/`√` on polar+azimuth).

- **Cartesian pseudo-helix only pays off if you materialize** — its whole job is to
  hand you the vector. Per-element cost = **N reconstructions**.
- **Fisher-2z normalized is built to never materialize** — comparison and lookup
  live in the normalized-index domain; any reconstruction is amortized to a
  **one-time table build**. Per-element cost = **0** (or 1 LUT build).

At N = 4.2 M vertices that is the difference between 4.2 M `atanh`/trig calls and a
single 256×1024 LUT bake. **Never reconstruct per element when the representation is
normalized** — that was the session's core mistake (decoding Signed360→Cartesian per
vertex). Pre-materialize a direction LUT once; per vertex is a normalized-index
gather (CPU-SIMD friendly, one GPU texture fetch, works with no GPU at all).

---

## Use the **Cartesian pseudo-helix** (`helix_orient`) when…

- You need the **actual vector materialized once** and then you're done — a bake-time
  store where the consumer reconstructs a single time, or a one-off decode.
- There is **no metric/distance** requirement (no CAKES/CLAM pruning over orientations).
- **Place-independence is fine** — a global orientation with no HHTL trie context.
- You want **zero lance-graph dependency** (it lives in ndarray; quick/standalone).
- Prototyping, or a context where the cap-cascade's per-level refinement
  (≤0.3° at 3 bytes) is convenient and the Rodrigues decode cost is irrelevant.

## Use **Fisher-2z normalized** (`helix::Signed360` / `ResidueEdge`) when…

- You need **metric-safe O(1) distance** between orientations without decoding — the
  256×256 L1 `DistanceLut`. This is the substrate's reason-for-being.
- The orientation is a **residue on an HHTL place** (hierarchical / place-coupled
  search; the trie sets where the arc starts on the ruler).
- **N is large** and per-element reconstruction would be the cost driver → stay in the
  normalized domain, pre-materialize once.
- You render at scale: **pre-materialized direction LUT + per-vertex normalized-index
  gather** (no per-vertex trig/atanh). This is the q2 `/helix` shape.
- You're **inside the lance-graph substrate** where the Fisher-Z rolling floor /
  256-palette already exists — reuse it, don't re-derive a parallel codec.
- You need **CPU-SIMD / GPU-less** cheapness — normalized indices gather; `atanh` does not.

---

## Signed360 specifics future sessions trip on (from the session log)

1. **`Signed360` turns hemisphere → full sphere via the SIGN.** The base hemisphere
   is `(rim, azimuth)`; the **`polar` byte's partition flips which hemisphere**
   (≥128 = upper `+y`, <128 = lower `−y`). So the normal-only 6-byte `Signed360` is a
   **complete** full-sphere direction — you do NOT need a second "pos" helix to
   complete it. (`encode_signed` stores `|y|` in 7 bits + sign in the partition;
   decode: `y = polar≥128 ? (polar−128)/127 : −(127−polar)/127`.)
2. **The rim is the METRIC carrier, not a render input.** `rim.start_idx` is
   `place % 17` (the HHTL anchor) and `rim.end_idx` is the Fisher-Z (`arctanh`)
   radius. Rendering needs only `(polar, azimuth)`; **never run the rim's atanh/tanh
   to recover a direction** — that's the unnecessary materialization.
3. **Direction is place-INDEPENDENT; only the metric is place-coupled.** Same
   `(polar, azimuth)` ⇒ same world direction regardless of place. "On top of HHTL"
   refers to the rim/metric anchoring, not a per-place rotation of the direction.
4. **Encoding a 3D normal → `(n, sign)` is a nearest spherical-Fibonacci search**
   (the crate has no `from_normal` helper): pole axis = chosen world axis,
   `sign = sign(n·pole)`, `n` = nearest `HemispherePoint::lift` index (match the
   equatorial projection + `|lift|`). See `q2 scratch-fma/helixbake`.
5. **Pre-materialize, then gather.** Build a `(polar × azimuth) → direction` LUT once
   (the only place trig is allowed); per vertex copy the normalized bytes and index
   the LUT. Polar 7-bit ≈ 0.45°, azimuth 10-bit LUT col ≈ 0.35° → sub-degree, ample
   for shading.

---

## One-line rule of thumb

> If the next thing you do with the code is **compare it or look it up**, keep it
> **Fisher-2z normalized** and stay in the index domain. If the next thing is **use
> the actual vector once**, a **Cartesian pseudo-helix** decode (or a one-time LUT
> build) is fine. Reconstructing a normalized representation per element is always
> the wrong move — that is what the substrate exists to avoid.

Cross-refs: `crates/helix/KNOWLEDGE.md` (place/residue spec, Curve-Ruler, Fisher-Z
identity), `crates/helix/src/{residue,placement,curve_ruler}.rs` (the source of
truth), `I-VSA-IDENTITIES` (bundle identities, not content — same "don't materialize
the register" discipline), q2 `scratch-fma/helixbake` + `cockpit/src/BodyHelix.tsx`
(a worked encode/decode pair) and q2 `crates/osint-bake/tools/BAKE_ARTIFACTS.md`.
