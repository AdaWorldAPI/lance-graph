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
