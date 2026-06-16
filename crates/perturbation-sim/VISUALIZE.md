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

## Future iterations
- Map the glow to the actual `node_field` values from a real `simulate_outage`
  run (export `(lon, lat, |Δθ|)` and render to scale), not an artistic gradient.
- Render the real 261-bus largest-component layout (drop "illustrative").
- Add a small AC-fork inset (bus-voltage heatmap + `collapse_margin`) once the
  voltage-collapse path is exercised on real data.

*The final PNG(s) can live under `docs/`; this file is the reproducible spec.*
