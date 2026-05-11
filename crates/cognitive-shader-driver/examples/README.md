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
