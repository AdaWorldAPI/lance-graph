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
