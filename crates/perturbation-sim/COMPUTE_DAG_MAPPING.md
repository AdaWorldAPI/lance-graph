# perturbation-sim ≡ `compute_dag` — the electricity cascade IS the topological recompute

> **Status:** FINDING (conceptual mapping; no code dependency added). `perturbation-sim`
> stays zero-dep / workspace-excluded — this is a doc-level join, not a `lance-graph-contract`
> import. The two crates share a *mechanism*, proven on opposite ends: this crate ships the
> certified physics + full recompute + the router/field/witness encoders; `compute_dag`
> (`lance-graph-contract::class_view`) ships the incremental dispatch the physics bound certifies.
>
> **Grounds:** `E-CHESS-TENSOR-PROVEN`, `E-EXCEL-SHADER-PROJECTION`, `E-OGAR-ROUTER-ENCODER`,
> `probe-excel-compute-dag-v1`. **Cross-ref iron rule:** `I-NOISE-FLOOR-JIRAK` (significance
> uses Jirak 2016, not IID Berry–Esseen), `I-VSA-IDENTITIES` (the numeric witness arc is NOT
> the contract's identity `WitnessTable` — this crate's own `witness.rs` already guards that).

## The one-line claim

A cascading grid outage and a spreadsheet recompute are **the same dependency-driven
topological recompute**. Trip a line → redistribute flow → re-trip whatever now overloads
is, *structurally*, edit a cell → dirty its dependents → recompute them in topological order.
`perturbation-sim::simulate_outage` is the physical instance; `ClassView::compute_dag` +
`compute_dag_topo_order` + `write_row` is the abstract substrate. NNUE proves the same shape
at world-champion strength (`E-CHESS-TENSOR-PROVEN`).

## The mapping (each row a mechanism, not a rhyme)

| `perturbation-sim` (physical grid) | `compute_dag` / OGAR substrate | what it is |
|---|---|---|
| `simulate_outage` round loop (trip → recompute survivors → re-trip) | `compute_dag_topo_order` recompute dispatch (edit → dirty-set → recompute dependents) | **the cascade ≡ the topological recompute** |
| `PerturbationShape::trip_round[e]` (round each line tripped; `0` = seed) | the topological *generation* of each recomputed field = its position in `compute_dag_topo_order` | **trip generation = topo generation** |
| seed `alive[seed_line]=false` → rank-1 `E` on Laplacian `L` | the dirty seed: `write_row(seed_cell, cycle)` | **the trip = the gated edit** |
| `spectral_perturbation`: Weyl `\|λᵢ(L′)−λᵢ(L)\| ≤ ‖E‖₂`, Davis–Kahan `sinθ ≤ ‖E‖₂/gap` | the **NNUE incremental ≡ full** invariant: a bounded local edit perturbs the global field by a bounded amount, so recomputing only the dirty-set provably equals a full recompute | **the bound that certifies incrementality** |
| `PerturbationShape::node_field` (per-bus magnitude, the red footprint) | the wave/field readoff over the grid (`E-OGAR-ROUTER-ENCODER` field side) | **the wave** |
| `splat::morton2` (x/y nibble-interleave) | the 2-axis router ADDRESS (`HEEL/HIP/TWIG` 256×256 tile) | **the router** (`E-OGAR-ROUTER-ENCODER` GREEN 2-axis case) |
| `sketch::fwht` + `walsh_pyramid_energy` | the deterministic FIELD ENCODER (Walsh–Hadamard pyramid) | **the encoder** |
| `witness::particle_equals_wave` (Parseval over FWHT) | particle (pointer-chase `∑field·arc`) ≡ wave (one transform, many arcs) | **the particle/wave click, proven on a real field** |

## The crucial honesty (why this is a join, not a merge)

The two ends are **complementary halves**, deliberately:

- **`perturbation-sim` does the EXACT FULL recompute each round** — `simulate_outage` recomputes
  DC flows on the *surviving* network from scratch every round ("robust where iterated single-line
  LODF would drift", per `cascade.rs`). It is NOT incremental. What it ships *alongside* is the
  **certification apparatus**: `spectral_perturbation`'s Weyl/Davis–Kahan bounds are exactly the
  inequalities an incremental scheme needs to prove it equals the full recompute.
- **`compute_dag` is the INCREMENTAL dispatch** — recompute only the dirty dependents, in
  `compute_dag_topo_order`, each gated by the cycle-aware `write_row`. The Weyl bound this crate
  certifies is *why* that incremental recompute is sound (a bounded local change → bounded global
  perturbation → the dirty-set is the complete support of the change).

So: **this crate = the proof + the full reference + the router/field/witness encoders; the
`compute_dag` harness = the incremental consumer whose equivalence this crate's Weyl bound
certifies.** Stockfish NNUE is the existence proof that the incremental side works at scale;
this crate is the existence proof that the bound holding it together is real.

## What this does NOT claim (scope guard)

- **No speed claim.** `witness.rs` already states the particle/wave win is "one field, many arcs
  whose spectra are reusable", not a measured single-arc speedup. This doc inherits that honesty.
- **No new dependency.** `perturbation-sim` remains zero-dep, workspace-excluded, standalone.
  This is a conceptual bridge; the only wiring is the optional `ndarray-simd` git feature that
  already existed for the eigensolver/reliability path.
- **The numeric witness arc ≠ the contract `WitnessTable`.** Already guarded in `witness.rs`
  (`I-VSA-IDENTITIES` register-loss / Frankenstein hazard). The mapping above pairs *mechanisms*
  (Parseval ≡ particle/wave), never the value categories (`&[f64]` field vs 6-bit W-slot identity).
- **Per-node/per-cell evaluation semantics are general compute**, dispatched via the DO arm /
  `UnifiedStep` — not the Walsh field. The *recompute structure* transfers; the *formula content*
  (`=VLOOKUP`, a per-square chess eval, a per-bus injection) does not. Same line `E-EXCEL` /
  `E-CHESS` already drew.

## Consumer payoff

The probe `probe-excel-compute-dag-v1` lands `ClassView::compute_dag` on the clean 2-axis sheet.
This doc shows the *same* `compute_dag` already has a physical, certified instance here: an OSM →
gaussian-splat → electricity-perturbation consumer (the 3D sibling cascade of `E-OGAR-ROUTER-ENCODER`)
is a `compute_dag` consumer whose recompute order is `simulate_outage`'s round structure and whose
incremental-equivalence is Weyl-bounded. Every computed-field AR consumer (odoo `@api.depends`,
medcare lab-trends, woa calc, q2 cells, **and** the electricity-cascade analytics) reduces to the
same `compute_dag_topo_order` dispatch.
