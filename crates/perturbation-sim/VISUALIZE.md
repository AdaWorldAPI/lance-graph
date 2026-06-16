# Visualizing perturbation-sim — figure spec & prompts

Two faithful variants of one figure. **Rule for a math/stats audience:** the
metaphor is the hook, but every symbol must be exact and every claim must carry
its measured number or its theorem. Keep the Go-board *because* it visualizes
Cheeger duality — not for decoration.

## Canonical caption (use verbatim; numbers are the measured outputs)
> **Figure 1.** Perturbation-shape of a cascading grid failure on the Iberian
> 261-bus core (illustrative topology). A line trip near the north-centre
> (round-0, red) propagates to round-1 (orange), producing a spatial field of
> phase perturbation |Δθ| that decays with distance. **(b)** Local *infight*
> (clustered trips) trades with global *Raumgewinn* (field separation);
> Cheeger's inequality links the normalized gap μ₂ to conductance h. Empirically,
> infight and field separation are **orthogonal (ρ ≈ 0.05)**. **(c)** Four-factor
> battery of DC structural screens for contingency risk.
>
> *Measured (261-bus core, synthetic injections):* seed trip → 71/348 lines
> tripped (20.4 %); λ₂-loss 21 %; 43 islands; Cronbach α = −0.83 (distinct
> facets); infight ⟂ field ρ ≈ ±0.05; time test-retest ρ = 0.90.

## Non-negotiable rigor points (a professor will check these)
1. **`λ₂` ≠ `μ₂`.** `λ₂` = combinatorial Fiedler (the cut; Weyl/Davis–Kahan
   perturb it). `μ₂` = normalized-Laplacian gap (Cheeger bounds it). Label both
   and note the distinction. (See `METHODS.md` anti-dilution table.)
2. **Factor symbols:** `Δλ₂` (Weyl bound) · `sinθ_DK` (Davis–Kahan subspace) ·
   `Δφ` (Cheeger sweep) · `infight` (trip-fraction).
3. **Honesty panel, prominent:** "DC structural screen on synthetic injections —
   NOT a reproduction of the 28 Apr 2025 event, which was voltage-collapse
   (ENTSO-E); voltage trigger needs the AC fork."
4. **Label topology "illustrative"** unless rendering the real 261-bus layout.

## Prompt — journal-figure variant (the one for the professor / a paper)
> Light/white background scientific figure, panels labelled (a)(b)(c). (a) An
> Iberian-peninsula flat outline (no terrain texture) with ~50 nodes (two sizes:
> HV substation / MV bus) and thin teal in-service edges; from a north-centre
> seed, a red (round-0) then orange (round-1) dashed cascade of tripped lines,
> with a perceptually-uniform (inferno) |Δθ| glow on affected buses fading with
> distance; a black dashed "France–Spain separation / Fiedler cut λ₂→0" isolating
> a small "Continental Europe" cluster; a "Topology: illustrative" tag; a
> |Δθ| (rad) colorbar; a km scale. (b) "Two-scale view (Go-board motif)": an
> Infight panel (dense red local trips + "local stress mean|Δθ|" gauge) above a
> Raumgewinn panel (a Go board split by a dashed cut + "normalized algebraic
> connectivity μ₂" gauge), joined by an arrow "Cheeger exchange rate
> μ₂/2 ≤ h ≤ √(2μ₂) (measured orthogonal, ρ≈0.05)"; a note "λ₂ combinatorial
> (cut) · μ₂ normalized (Cheeger gauge)". (c) Four chips: Δλ₂ (Weyl bound),
> sinθ_DK (Davis–Kahan), Δφ (Cheeger sweep), infight (local collapse); plus an
> info panel with the honesty disclaimer (above). Add the canonical caption and a
> monospace measured-numbers sidebar. Clean, legible, no photorealism. 16:9.

## Prompt — pitch/poster variant (for a talk)
> Same content, dark-mode data-explorer aesthetic (deep navy, glowing nodes, hot
> red/orange cascade). Desaturate the glow ~20 %; keep all the same labels,
> symbols, and the honesty disclaimer. Higher drama, identical rigor.

## Figure 2 — Morton-tile pyramid: spatial perturbation & where the 4 theorems map

Explains the stacked 2-bit×2-bit (4×4) Morton/z-order pyramid and maps each
theorem to the pyramid structure it lives in. Faithful to the OGAR two-algebra
rule (sign = Walsh/XOR, magnitude = EWA splat) and the geography/screen caveats.

### Theorem → pyramid-structure mapping (the honest one-liner each)
| Theorem (factor) | Lives at | Meaning |
|---|---|---|
| **Weyl** (`Δλ₂`) | **top** (coarse/global) | whole-graph field eigenvalue λ₂ (Raumgewinn) |
| **Davis–Kahan** (`sinθ_DK`) | **mid** — the partition reorienting between levels | Fiedler-vector rotation / subspace stability |
| **Cheeger** (`Δφ`) | **the inter-tile seam** | conductance of the cut = the coarse↔fine exchange rate `μ₂/2 ≤ h ≤ √(2μ₂)` |
| **infight** | **bottom** (fine tiles) | local collapse (cascade trips) |
| *(Kron reduction)* | **the 4→1 coarsen arrow** | Schur complement: 4 fine tiles → 1 coarse super-node (basin tiering) |

Two algebras (axes of the pyramid): **sign = Walsh/XOR** (`vsa_bind`, the *scale*
axis — coarse coeff = field, fine coeff = infight); **magnitude = EWA Gaussian
splat** (`vsa_bundle`, the anisotropic footprint, anti-aliases the Z-seam).
`perturbation = Σ_L sign(addr,L)·magnitude(addr,L)` (bipolar-phase pyramid =
Walsh–Hadamard on the address tree).

### Honesty (must appear on the figure)
Tiles ride the **electrical** embedding (effective-resistance / spectral coords),
**NOT geography**. Walsh basis = graph eigenbasis **exactly only on hypercubes**
→ a fast O(n log n) **screen**; the exact eigensolve certifies the flagged tiles.
SIMD WHT via `ndarray::simd::wht_f32`.

### Prompt — Morton-pyramid figure
> Clean light/white scientific figure, 16:9, titled "Stacked Morton-Tile Pyramid
> — Spatial Perturbation & the Four Theorems" (thin teal/navy linework,
> perceptually-uniform inferno accents, monospace numbers).
>
> **Center-left — the pyramid:** a vertical stack of 4 receding plates, bottom
> (fine) → top (coarse), each a 2-bit×2-bit (4×4) Morton/z-order quadtree level.
> L0 (bottom): 16×16 cells with a faint Z-order (Morton) curve threading them, a
> bright red perturbation spike in one central cell. L1, L2: coarser 8×8 then 4×4
> grids; the perturbation rises as a widening anisotropic glow cone. L3 (top): a
> single tile (whole-network summary). Upward "coarsen 4→1" arrows between levels.
>
> **Map the four theorems (callout labels + leader lines):** L3 top →
> `Weyl — Δλ₂` "algebraic connectivity λ₂ (global field / Raumgewinn)"; mid
> partition boundary → `Davis–Kahan — sinθ_DK` "Fiedler-vector rotation / subspace
> stability"; inter-tile seam → `Cheeger — Δφ` "conductance of the cut; exchange
> rate μ₂/2 ≤ h ≤ √(2μ₂)"; L0 bottom → `infight` "local collapse (cascade trips)";
> on the 4→1 arrow → `Kron reduction (Schur complement)` "4 fine tiles → 1 coarse
> super-node".
>
> **Right column — two algebras:** Sign side: a small bipolar ±1 (black/white)
> Walsh pattern, "Walsh / XOR (vsa_bind) — SCALE axis: coarse coeff = field/
> Raumgewinn, fine coeff = infight". Magnitude side: an anisotropic Gaussian
> ellipse over a 4×4 tile straddling a Z-seam, "EWA Gaussian splat (vsa_bundle) —
> anti-aliases the Morton seam; Σ-anisotropy = spread direction ≈ cut normal". A
> one-line equation between them: perturbation = Σ_L sign(addr,L)·magnitude(addr,L)
> ("bipolar-phase pyramid = Walsh–Hadamard on the address tree").
>
> **Bottom honesty strip (visible):** "Tiles ride the electrical embedding
> (effective-resistance / spectral coords) — NOT geography. Walsh basis = graph
> eigenbasis exactly only on hypercubes → a fast O(n log n) screen; the exact
> eigensolve certifies the flagged tiles. SIMD WHT via ndarray::simd::wht_f32."
>
> Legible labels, no photorealism; the rising red→orange perturbation cone and
> the four theorem-callouts are the dominant motifs.

## Future iterations
- Map the glow to the actual `node_field` values from a real `simulate_outage`
  run (export `(lon, lat, |Δθ|)` and render to scale), not an artistic gradient.
- Render the real 261-bus largest-component layout (drop "illustrative").
- Add a small AC-fork inset (bus-voltage heatmap + `collapse_margin`) once the
  voltage-collapse path is exercised on real data.

*The final PNG(s) can live under `docs/`; this file is the reproducible spec.*
