# copy-contract — v3-envelope-auditor run (2026-07-29)

Branch `claude/x265-x266-plans-review-h9osnl`. Scope: `derive(*Copy*)` sites in
`crates/lance-graph-contract/`. Operator order: *"copies are forbidden, borrows
are only for the same mailbox"*, *"only cognitive achievements > tenant"*.

## Census correction

The blast-radius file lists ~80 contract sites; the crate actually has **316**
`derive(… Copy …)` sites (the census matched two literal orderings, not the
general form). Verdicts below cover the full 316 by class; every site ≥ 512 B or
carrying a lifetime was read individually.

## Verdicts (summary)

- **VIOLATION — 3.** `AwarenessPlane16K` (2 KB), `SplatPlaneSet` (12 KB),
  `NodeRow` (512 B).
- **ELEVATED — the cycle/basin aggregates** (`ShaderResonance`, `AlphaComposite`,
  `MetaSummary`, `MaterializeProvenance`, `RungElevator`, `SpoBase17`, `Heel`,
  `CamSplatCertificate`) — all strictly above the tenants they derive from, so
  `Copy` is the operator's own carve-out.
- **LEGITIMATE — everything else**, incl. the two the brief flagged as suspects:
  `ColumnWindow` (an index pair, not a view — its "zero-copy borrow" doc-comment
  is a misnomer) and `ShaderHit` (a row ADDRESS + computed measures, the
  I-VSA-IDENTITIES points-to-content shape). `QualiaI4_16D` is `repr(C) u64` —
  data-flow §2 verbatim.

## Changes made (edit-only; not compiled here)

`crates/lance-graph-contract/src/splat.rs`
- `AwarenessPlane16K` (was :87) — `Copy` removed, `Clone` retained, why-comment added.
- `SplatPlaneSet` (was :176) — `Copy` removed, `Clone` retained, why-comment added.

Cascade for both: **zero**. Every consumer (`crates/jc/examples/splat_*.rs`,
`crates/lance-graph-contract/src/splat.rs` tests) uses `&` / `&mut` /
`vec![… ; n]` / `::zero()` / `::default()`. `vec!`-repeat needs `Clone`, not
`Copy`.

## Refused (cascade reported, not executed)

`NodeRow` (`canonical_node.rs:724`) — ruled VIOLATION, derive LEFT IN PLACE.
Removal is one line here plus **5 call-site fixes in a crate outside this
scope**, all in `crates/lance-graph-planner/examples/probe_sudoku_teacher.rs`:
`:307`, `:355`, `:1818` (`[blank_row(); 81]` → `core::array::from_fn(|_| blank_row())`)
and `:425`, `:793` (`let mut world = *grid;` → `grid.clone()`).
`:425` carries the comment `// NodeRow is Copy — an explicit, deliberate clone`
— the author needed a clone and `Copy` handed over a 41 KB substrate duplicate
for free. That comment is the finding.

## Handed to the Tier-A agent (census gap)

The Tier-A list found only `NodeRowPacket<'a>` in this crate. Six more
lifetime-carrying `Copy` types exist: `class_view.rs:1181 RenderRow<'a>`,
`class_view.rs:1202 ValueRow<'a>`, `unicharset_adapter.rs:54 UniCharCall<'a>`,
`:77 UniCharOut<'a>`, `recoder_adapter.rs:88 RecoderCall<'a>`, `:120 RecoderOut<'a>`.
Untouched by me. (`cognition/entity.rs:55/61/68/75` are `&'static str` handles —
program-image data, not substrate; not Tier A.)

## Naming hazard recorded

`recipe_substrate.rs:45 SubstrateView` — named "View", doc'd "the substrate a
recipe reasons over", carries NO borrow (three owned facets, 40 B, built by
`::new` and immediately `.project()`ed). LEGITIMATE, but it is the one name in
the crate that trains a reader to believe a gathered copy is a borrow.
