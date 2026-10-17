# 2026-10-16 — D-CGW-1: CE64 bits 43..45 are the orientation

**Status:** SHIPPED IN PR (branch ccr-9f3b8075-74u47l-cgw1).

## What changed

- New `causal_edge::orientation::Orientation` (`crates/causal-edge/src/orientation.rs`):
  a 3-bit code. `Core` = 0, then Direction, Recipient, Experiencer, Purpose,
  Beneficiary, PredicativePossessor, ExternalPossessor (Haspelmath 2003 dative
  map). `from_code` reads the low three bits, so every value names exactly one
  orientation.
- `CausalEdge64::direction()/set_direction()` are now
  `orientation()/set_orientation()`. `pack` and `pack_v2` take an
  `Orientation`, so the compiler lists every writer. `layout::DIR_SHIFT` is
  renamed `ORIENT_SHIFT`. `CausalEdgeV3::direction()` became `orientation()`.
- Removed, not moved: `s/p/o_pathological`, `concern_level`,
  `CausalNetwork::detect_simpsons_paradox`. Reason: no production code wrote
  the triad. The only production writer of 43..45 was `deepnsm-v2::wire`, which
  already wrote Haspelmath codes. The planner's `ConfounderCheck`
  (`lance-graph-planner/src/pearl.rs`) computes its two directions from
  population masks. `CausalMask::simpsons_paradox_risk(so, po)` stays, documented
  as taking caller-computed directions.
- `deepnsm-v2::wire` writes the contract's `Orientation`. Its own u8 table became
  `orientation_of(DativeFunction) -> Option<Orientation>` and
  `orientation_function(Orientation)`.
- Contract `moore_tenant`: `DirectionReading`, `SignTriple`,
  `require_sign_triple` and `DirectionRefusal` (a per-tenant reading binding) are
  replaced by `MooreDirection { slot, polarity }`. `MooreTenantMut::lift_ce64`
  refuses an edge whose orientation is not `Core` with the new
  `MooreRefusal::Oriented { slot, code }`. A refusal writes nothing.
- About 85 call sites migrated across causal-edge, deepnsm-v2, lance-graph,
  lance-graph-planner, cognitive-shader-driver, lance-graph-osint, p64-bridge and
  thinking-engine. Zero literals became `Orientation::Core`. Fixture bit patterns
  kept their bits through `Orientation::from_code`.
- Probes that read 43..45 per class (`witness_angle_probe`) or as the triad
  (`moore_nars16_probe`) are marked superseded. The moore_nars16 Simpson tests
  were removed with the detector they tested. `moore_nars_gomoku_probe` no longer
  writes its Moore direction into 43..45.
- `docs/architecture/ce64-semantic-upper-half.md`: 43..45 are settled. The
  "declared reading per tenant" paragraph is replaced by one meaning per field.

## Tests

- `crates/causal-edge/tests/orientation_contract.rs` (5 tests, written first and
  committed red):
  - the code table is one-to-one over 3 bits;
  - zero is `Core`;
  - accessor and setter agree;
  - both packers write the same table;
  - field isolation of 43..45 against 40..42 and 46..49.
- Contract `moore_tenant`: `h_direction_is_slot_and_polarity` and
  `h2_an_oriented_edge_is_refused`. Disable run: removing the refusal check turns
  `h2` red (11 passed, 1 failed). Restored afterwards.
- causal-edge passes on both the v2 default and `--no-default-features` (v1).

## Edge facet census (Haiku card, spot-checked)

- Production writers of the 16-byte edge facet: two.
  - `lance-graph-contract/src/aiwar.rs:117-119` (out-of-family slots, bytes
    12..15).
  - `symbiont/src/domino.rs:95-97` (bytes 0 and 12).
- Both use the old 12+4 slot reading.
- All other non-default constructions are test code. No code assigns a row's
  `.edges` field directly.

## Decided 2026-10-16 (conversation)

- WordNet: `is_a` is the identity facet's DN path (16-nibble, already wired by
  `deepnsm-v2::promote::{tiers_of, key_at}`). `part_of` and other relations go to
  the edge facet, read as 16 slots through the ClassView. The 128-bit facet
  format already exists (D-HPR-6a), so there is no layout change. The missing
  piece is the caller: AriGraph, which is D-ARG-3.
- A leapfrog trie zipper over DN keys is a lanewise span fold (each DN level is
  an ordered lane; a seek is a span). It is input to D-ASM-11 arm B and depends
  on range-as-span (lane-fold capstone point 4). Per the capstone, a real
  element-wise zip and a cyclic n-hop are refused residuals, so the triangle in §9 of `2026-10-16-synergy-map-since-1579.md`
  qualifies only as span intersections plus one terminal AND.

## OPEN

- 46..49 are still read as activation by `wire` and as Energy4 by Moore tenants.
  D-CGW-9 moves activation to a row field.
- 50..52 carries plasticity (`learn`) and the novelty → wisdom rung (`wire`). It
  needs a ruling.
- The two production edge-facet writers still use the 12+4 slot reading.
- WordNet DN gates are not measured:
  - G1: chain depth ≤ 12/16;
  - G2: fan-out per level ≤ code width;
  - G3: multi-parent synsets.
