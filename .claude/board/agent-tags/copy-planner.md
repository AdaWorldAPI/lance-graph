# copy-planner — `Copy`-derive verdicts across the 8 planner/core crates (2026-07-29)

Operator order: *"copies are forbidden, borrows are only for the same mailbox"*;
*"only cognitive achievements > tenant"*. Reads: `zero-copy-lens-law.md`,
`data-flow.md` §2 (in `/home/user/ndarray/.claude/rules/`), `borrow-strategy.md`
(in `/home/user/q2/.claude/rules/`). EDIT-ONLY; no cargo run.

## Scope + census correction

Scope: `lance-graph`, `lance-graph-planner`, `lance-graph-supervisor`,
`lance-graph-turbovec`, `lance-graph-python`, `lance-graph-arm-discovery`,
`cognitive-shader-driver`, `causal-edge` — `src` AND `examples`.

**195 `Copy`-derive sites in scope**, not the 38 the census lists for these
crates. The census grepped `derive(Clone, Copy)`; the real spelling varies
(`derive(Clone, Copy, Debug, …)`, `derive(Debug, Clone, Copy, …)`,
`derive(Clone, Copy, PartialEq, Eq, Hash)`). **The census also MISSED the one
real violation in scope** — `AdjacencyBatch<'a>` carries two borrows and was not
in Tier A.

## Changes made (1)

- `crates/lance-graph-planner/src/adjacency/batch.rs:23` — `AdjacencyBatch<'a>`:
  `#[derive(Debug, Clone, Copy)]` → `#[derive(Debug)]` + why-comment.
  Fields are `store: &'a AdjacencyStore` and `source_ids: &'a [u64]` — every
  field is a borrow, i.e. the exact `WitnessLens<'a>` shape stripped in
  `b3515ba`. Zero cascade: all 4 call sites already pass `&AdjacencyBatch<'_>`;
  no `.clone()`, no by-value use anywhere.

## Cascades refused (report, do not touch)

1. **`SpoHead`** (`lance-graph-planner/src/cache/nars_engine.rs:29`) —
   VIOLATION, but the repair is type deletion, not a derive removal.
2. **`InteractionKinematic`** (`cognitive-shader-driver/src/sigma_rosetta.rs:904`)
   — VIOLATION (stores two diagonal terms); repair is a field-shape change.
3. **`WitnessWindow`** (`lance-graph-planner/src/traits.rs:89`) — the separate
   high-priority item; blast radius below.

Both (1) and (2) plus `WitnessWindow` are **test-only-producer types** — the
`zero-copy-lens-law.md` "shadow of storage" signature, three instances in scope.

## WitnessWindow blast radius (address-only migration)

Total surface: **1 decl, 1 read of `.rows`, 1 production consumer, 25
`witness: None` sites, 3 test constructors.** The lens twin
`standing_wave_stratified_lens` ALREADY EXISTS (`witness_fabric.rs:852`).
The one real migration hazard is the `focal_idx` (index into the gather) vs
`focal_pos` (absolute stream position) semantic swap. Full detail in the
session report.
