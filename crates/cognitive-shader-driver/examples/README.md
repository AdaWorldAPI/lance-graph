# cognitive-shader-driver — examples

## villager_ai.rs

Reference integration for our fork of [Pumpkin](https://github.com/Pumpkin-MC/Pumpkin),
the Rust Minecraft server. Demonstrates how to use the lance-graph state-
classification primitives for block-game NPC AI:

| Contract primitive                         | NPC AI usage                          |
|--------------------------------------------|---------------------------------------|
| `StateAnchor` (7 anchors)                  | Villager mood (idle/trading/sleeping/…) |
| `ProprioceptionAxes` (11 named fields)     | Villager behavioural state vector     |
| `DriveMode` (Explore/Exploit/Reflect)      | Pathfinding regime                    |
| `WorldMapDto` + `WorldMapRenderer`         | Per-entity state snapshot + labels    |
| `bond` glyph (relational slot 20)          | Pet bond strength                     |
| `WorldModelDto::user_state`                | Villager's inferred model of the player (opponent model) |
| `FieldState::gestalt`                      | Multi-party trading dynamics          |
| `CollapseGate` (Flow/Hold/Block)           | Trade-commitment gate                 |

Each contract type maps to an AI tick concern the server runs per game tick.
The `VillagerRenderer` demonstrates the drop-in renderer pattern: the same
`WorldMapDto` is rendered with villager-flavoured labels (mood → anchor,
sociability → contact, rapport → attunement, etc.) without touching the
contract crate.

```bash
cargo run --example villager_ai -p cognitive-shader-driver
```

Three scenes:

1. **Pet bonding** — wolf tame sequence, bond climbs on positive interaction,
   decays when left alone.
2. **Villager trade approach** — villager drifts toward the `Focused` anchor
   over several ticks; renderer labels anchor states as villager moods.
3. **Anchor gallery** — all 7 calibration anchors rendered side-by-side
   as their villager-mood equivalents.

## moore_plasticity_probe.rs

D-GSO-2 (plan `2026-10-06-global-sudoku-replayable-orchestration-v1.md` §18 P2).
One gated, synchronous Moore hop over the 16 bytes of a `Register128` read as a
4 × 4 grid. Neighbours come from a closed-form direction mask, not an edge list;
each relation byte goes straight into the existing `PalettePerturbation::hop`
law. The tests run under the crate's `cargo test` (`test = true`).

```bash
cargo run -p cognitive-shader-driver --example moore_plasticity_probe
```

## ontology_agreement_probe.rs

D-GSO-3 (plan §18 P3). Support / silence / opposition for one claim, folded in
one pass over the observations into three scales of a `NiblePath` hierarchy:
local (the class and its siblings), up (ancestors) and down (descendants),
using the existing `ontology_warrant::Quorum`. Tension is the binary entropy of
the speaking split; the probe shows a synthetic mammal family landing in each
of settled / local exception / basin conflict / hotspot / unknown.

```bash
cargo run -p cognitive-shader-driver --example ontology_agreement_probe
```

## entropy_topology_probe.rs

D-GSO-4 (plan §18 P4) = F-ECG-1 and F-ECG-2 of `entropy-closure-causal-ground-v1`. Two
basins with identical measured field entropy but different `CausalTopology`
on CausalEdge64 bits 59..60 land in different settlement cells (Crystal vs
Glass) and take different walker routes. The bits are read only through the
`band_reading` contract (declared lens, asserted provenance); an unreadable
register produces no cell. F-ECG-3 (the census discriminates on a real
corpus) stays open: no stored corpus carries topology bits on real edges yet.

```bash
cargo run -p cognitive-shader-driver --example entropy_topology_probe
```

## fieldless_local_interaction_probe.rs

D-CTX-0. Does a cognitive texture have to exist as a field? One synchronous
Moore step over a 4 × 4 tile whose lane is the `Morton8x8` code: checked
neighbour, resident palette byte, one read of the calibrated
`bgz_tensor::FisherZTable` (equal bytes answer identity by address, never by
the diagonal), folded straight into `activation × Σ R_local` on a resident
strength register. A materialized oracle (pair list, relation population,
field) must produce the same register; the fieldless step makes no heap
allocation. `bgz-tensor` is a dev-dependency only: production borrows the
calibrated bytes.

```bash
cargo run -p cognitive-shader-driver --example fieldless_local_interaction_probe
```

## recipe_quartet_probe.rs

D-GSO-5 (plan §18 P5). Four recipes, each a `match` arm that calls one
existing primitive directly: observe/fold (`Quorum::observe`), finite-product
interrogation (`Quad8::fold_product`), Moore interrogation
(`Morton8x8::checked_offset` + `PalettePerturbation::hop` over `Register128`),
and revision (`GadamerRevision::revise`). R3 claims no counterfactual: removing
an encounter's new roots always defeats eligibility, so that attack tests
nothing; a real one needs root-to-claim structure (P7).
A counting allocator shows each recipe runs with zero allocations, so no
instruction vector is built. The ordinal meanings are a scaffold, not canon,
and `ProbeRecipe` is not a shipped `recipes::Recipe` ID.

```bash
cargo run -p cognitive-shader-driver --example recipe_quartet_probe
```

## virtual_surfel_probe.rs

D-CTX-1. A surfel needs no stored identity. On a 16 × 16 tile (lane = Morton
code, every palette ordinal present), each activated pixel's surfel is read
from its eight Moore neighbours through the Fisher-Z law, as a weight `W` and
a second moment `Σ = Σ w·d dᵀ`, and handed straight to the consumer. A
materialized `Vec<SurfelReading>` oracle must agree reading for reading, and
the virtual path allocates nothing. The orientation of `Σ` comes from the law
alone; what an identity neighbour (same material) weighs is an explicit
argument, because the probe shows it decides that orientation.

```bash
cargo run -p cognitive-shader-driver --example virtual_surfel_probe
```

`support/fisher_relation.rs` holds what both D-CTX probes share (the borrowed
Fisher-Z relation, the Moore offsets, the per-thread allocation counter). It is
included with `#[path]`, not built as an example of its own.

## ewa_render_probe.rs

D-CTX-2. Isotropic Gaussian/EWA rendering with no surfel or Gaussian
population. Each virtual surfel gives an isotropic scale `s` (the isotropic part
of its normalised second moment), the contract's `ewa_sandwich(√s·I, I) = s·I`
gives the footprint, and the footprint is accumulated straight into a 2 KB
transient field. A materialized `Vec<Surfel> → Vec<Gaussian> → field` pipeline
must agree bit for bit; an independent closed-form gather must agree within
`1e-12` of the field's maximum; rendering only redistributes the surfels' mass.
The footprint evaluator stays probe-local.

```bash
cargo run -p cognitive-shader-driver --example ewa_render_probe
```

`support/virtual_surfel.rs` holds the D-CTX-1 surfel reading that the rendering
rounds read from, so they use exactly what D-CTX-1 tested.

## recipe_selector_probe.rs

D-GSO-6 (plan §18 P6). Recipe selection as a pure function of a declared
policy version and a declared epistemic state (observations pending, frontier
bounded, local disagreement, new encounter). V1 follows the §11 loop order
(fold, bound, local, revise) and rests when nothing is open; V2 swaps two steps
to show that a policy change is a new version. A recorded `Selection`
(policy + state) replays to the same recipe from the record alone. The policy
version covers selection only, not the recipes' implementations (plan §13).

```bash
cargo run -p cognitive-shader-driver --example recipe_selector_probe
```

## reasoning_band_probe.rs

D-GSO-7 (plan §18 P7). The `ReasoningBand` on a `CausalEdge64` (bits 61..63,
read only through `band_reading::project_band`) is earned one rung per executed,
passing proof: an `SO` observation a majority corroborates gives `Association`,
a passed `PO` intervention on top of it gives `Causal`, a passed `SPO`
counterfactual on top of that gives `Counterfactual`. Observations never lift
past `Association`, a test that was not run never raises, and no rung is
skipped. A pass is never supplied: events carry trial data (intervention
counts, a removal attack on premise masks) and the probe executes the trial to
derive the outcome. A failed test, contradicting independent evidence (`Quorum`) or
confounding (`CausalMask::simpsons_paradox_risk`) lowers the band, and the same
events replay to the same edge bits. The ladder and thresholds are policy pins.

```bash
cargo run -p cognitive-shader-driver --example reasoning_band_probe
```

## relational_certification_probe.rs

D-GSO-7a (P7a). Replaces the meaning of the band rungs from `reasoning_band_probe`
(#1360, which stays as shipped). Under a reading declared per class, bits 61..63
hold the strongest relational statement the sealed model may assert:
`0 Open, 1 Associated, 2 Related, 3 Contributes, 4 CausalCandidate, 5 Causes`;
6 and 7 are reserved and refuse. Each contract is an integer fold over unit
masks and a `SupportLedger`: association needs two distinct sources, the
robustness mask can only refute, `Contributes` is a stable effect in every
declared stratum, and `Causes` needs two distinct `InterventionBacked` sources
plus executed randomized arms. No observational effect reaches `Causes`, the
inference mantissa does not move the band, and the historical `Meta` /
`Transcendent` codes do not satisfy `Causes`. An exhaustive family of 2,500
small models shows the chain is monotone when association is scoped to the
declared populations, and not when it is marginal (Simpson). Names and
thresholds are policy pins.

```bash
cargo run -p cognitive-shader-driver --example relational_certification_probe
```

## affordance_measurement_probe.rs

D-GSO-AFF-0. Recipe eligibility as a measurement: `CausalEdge64 × RecipeLaw →
EligibleRecipes: u64`. Bits 59..63 are read as one EpistemicState5 code (ten
declared codes, deliberately not ordered by strength; the rest refuse), and
bits 40..42 under their Pearl reading. Seven recipes declare `requires` /
`forbids` facts and Pearl planes; the rules compile at build time into
`[u64; 32]` and `[u64; 8]`, so a measurement is two lookups and an AND with no
allocation. Fields no recipe reads (S/P/O, F/C, bits 43..45, mantissa,
plasticity, witness) are swept and change nothing; preference can only clear
bits. Two law generations show the generation is part of the reading.

```bash
cargo run -p cognitive-shader-driver --example affordance_measurement_probe
```

## spog_witness_probe.rs

D-SPOG-W-0. A SPOG coordinate as a view over resident coordinates: G from
`graph_of(classid)` (the canonical source), the Witness slot as a sub-context
inside G only for classes that declare that reading (slot 0 = no anchor), and
S/P/O passed through as blind bytes. Undeclared classes, another reading of the
slot, and v1 / unknown provenance refuse. Compared with the real
`MailboxSoA::apply_edges` routing: they agree on all 64 × 64 slot pairs within
one graph and diverge across graphs, because routing reads no classid. That
cross-graph case bypasses the Alpha split tunnel and is not a production path
(D-ALPHA-G-0 below).

```bash
cargo run -p cognitive-shader-driver --example spog_witness_probe
```

## alpha_world_provenance_probe.rs

D-ALPHA-G-0. WorldG is attention provenance carried by the Alpha route, not a
CausalEdge64 field. An event is (Alpha-attended address, edge); WorldG is
`graph_of(addr)` and the order of worlds is the `SpogTenants` route. The same
edge bits in two worlds are two events; replaying the same claims recovers the
world sequence with all-zero edges; changing edge fields never moves the world.
Partitioned by recovered world, each `MailboxSoA` receives only its own
deliveries through the unchanged `apply_edges`; an undeclared world is refused
at the tunnel.

```bash
cargo run -p cognitive-shader-driver --example alpha_world_provenance_probe
```

## witness_angle_probe.rs

D-WA-0. Witness (`w_slot`) × Angle (bits 43..45) as a declared, sparse local
source coordinate under a per-class, versioned probe law. The same raw pair
means different sources in another class and another generation; classes
reading those bits as the pathology triad or a cohort `WitnessTable` refuse, as
do undeclared classes and generations, Witness 0 and v1 / unknown provenance.
The world comes from the attended key, never from the edge.

```bash
cargo run -p cognitive-shader-driver --example witness_angle_probe
```

## ce64_cycle_survival_probe.rs

D-CE64-TIME-0. `CausalEdge64` as the register that survives a cycle. One
`MailboxSoA` `edges` row is the only cross-cycle state; each cycle folds its
observations with the shipped `CausalEdge64::learn`, counts distinct sources in
a per-cycle `SupportLedger`, settles the EpistemicState5 code (probe-declared
step 3 ↔ 7) and writes the register back. The next cycle measures eligibility
with the #1370 law (`shared/affordance_law.rs`). Eligibility over four cycles:
`OBSERVE`, `OBSERVE`, `OBSERVE|STRATIFY`, `OBSERVE`.

```bash
cargo run -p cognitive-shader-driver --example ce64_cycle_survival_probe
```

## ce64_nextstate_probe.rs

D-CE64-NEXTSTATE-0. EpistemicState5 (bits 59..63) as a sparse next-state
mutation of the `CausalEdge64` register: only those five bits move, the write
crosses the real `MailboxSoA` cycle seam and is read in cycle k+1, the word
survives the canonical little-endian round trip, and a fresh mailbox built
from the persisted word continues identically. Also pins that the mailbox
does not double-buffer: next-cycle authority is a read discipline.

```bash
cargo run -p cognitive-shader-driver --example ce64_nextstate_probe
```

## streamdto_circuit_probe.rs

D-STREAMDTO-0. A real `StreamDto` through the shipped path: `ingest_codebook_indices`,
the mailbox read shim, `dispatch` at cycle k, the persisted row's emitted edge
written through `ShaderDriver::mailbox_mut`, `tick`, and `dispatch` at k+1,
which reads it. Then the #1370 law on the committed row. Pins that the edge
steers the next cycle only through `s_idx / 4`, that the shader's code 0
leaves eligibility where an unwritten word leaves it, and that the
`StreamDto` timestamp is not the cycle.

```bash
cargo run -p cognitive-shader-driver --features with-engine,mailbox-thoughtspace --example streamdto_circuit_probe
```

## ewa_anisotropic_probe.rs

D-CTX-3. Anisotropic EWA: the footprint is each virtual surfel's normalised
second moment `Σ_c / W`, so the orientation the Fisher-Z law gives a surfel
reaches the rendered surface. Eigenvalues are floored at ½, the smallest
isotropic scale the law can produce: without the floor a thin footprint,
sampled on the pixel grid, sums to more than its mass, and the law's bottom code
gives a singular Σ. Isotropic readings render exactly as in D-CTX-2. The probe
also measures how much the i8 quantization moves a surfel's orientation and
eigenvalues against the unquantized cosines.

```bash
cargo run -p cognitive-shader-driver --example ewa_anisotropic_probe
```

`support/ewa.rs` holds the D-CTX-2 isotropic law (window, amplitude, footprint,
scatter) that both rendering probes use.

## boundary_measure_probe.rs

D-CTX-4. One read-only measurement over the rendered surface: where is it
steepest? The operator reads the D-CTX-2 field over the inner region whose
values cannot see the tile edge, and returns a 32-byte witness (location,
magnitude, gradient) instead of a gradient field. A material boundary is found
where the palette changes, the rendered boundary response changes
deterministically with the signed pair-law code of the two materials (an
observation about the surface, not a semantic strength), a uniform tile reads flat, and measuring never changes the field.

```bash
cargo run -p cognitive-shader-driver --example boundary_measure_probe
```

## morton_order_probe.rs

D-CTX-5. Does the order in which surfels are visited change the render? Four
orders over the same 16 × 16 tile (Morton, the order D-CTX-2 uses; row-major;
4 × 4 tiled; reversed). The `f64` scatter changes in its last bits with the
order, so that order is part of its replay identity; an `i64` fixed-point
accumulator gives the same bits for every order. The probe also measures each
order's trie ascent per step through `Morton8x8::nibble_climb`.

```bash
cargo run -p cognitive-shader-driver --example morton_order_probe
```

## observation_revision_probe.rs

D-CTX-6. Closes the loop: render → measurement → observation →
`GadamerRevision` → replay. The belief state is a revision horizon with one
bit per pixel. The rendered boundary witness is presented to the revision as
an inherited interpretation and never changes belief; only an observation of
the resident palette (a new independent root) is admitted. The render decides
where to look next, not what is found: two Fisher-Z laws give different fields
and the same final belief.

```bash
cargo run -p cognitive-shader-driver --example observation_revision_probe
```
