# D-COGNITIVE-POLYFILL P0: four frontends, one native algebra (2026-11-03)

**Status:** measured, lab only. Probe `crates/cognitive-polyfill-probe` (own workspace); report `.claude/probes/2026-11-03-d-cognitive-polyfill/REPORT.md`. No production change.

## Measured

- The relational subset common to MORK, MeTTa, Narsese and Atomese is a ground, bounded, conjunctive one. A scoped two-hop EXISTS/COUNT over it runs as ONE mask-risc program over borrowed rows of a DR W64 tile. Results:
  - 12/12 tests; 125 cases agree with an independent oracle;
  - 0 bytes allocated per hot run;
  - 6 source disables red, each with its anchor asserted.
- Routing a selection by global id is required: joining on raw local positions gives the wrong answer (c04).
- The Set fold cannot answer the Bag question (2 vs 3, c08). The existing Bag route is `lance_graph_xlog::bag_count` (TRACKED MISSING-3, EXO-108).
- Existence of a NAL belief is not an edge. Existence projection and truth projection disagree on the same beliefs (c09).

## Where each language leaves the subset

- MeTTa:
  - D-MTP-1 covers functional relations only and refuses the fixture with `SecondFact`.
  - `BindingsSet` is a multiset (hyperon `matcher.rs:883`).
- MORK: the space is a Set (`set_val`, `space.rs:1561`). D-MPS-8 builds its own `JoinDomain`, beside DR.
- Narsese: cannot spell a binary relation without the product compound (R10).
- Atomese: no Classic AtomSpace source within reach.

## Open (owner)

- **W-1:** one cold symbol → DR id step for MeTTa/MORK. Today xlog takes pre-localized `u32`, MORK builds its own `JoinDomain`, and NAL uses `u16` terms.
- **W-2:** `Guards::check` needs a `FrameStore`, but `MapTable` keeps its store private.
- EXO-46, EXO-108, R10, the NAL → edge projection policy.
