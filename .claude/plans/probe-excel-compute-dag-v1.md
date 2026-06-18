# probe-excel-compute-dag-v1 — land `ClassView::compute_dag` on the clean 2-axis grid (the NNUE-incremental existence proof)

> **Status:** CONJECTURE / probe scope. The named first proof for the one Core gap
> (`ClassView::compute_dag`), on the case that can't hit the 3D-Hilbert crack.
> **Grounded by:** `E-CHESS-TENSOR-PROVEN` (Stockfish NNUE *is* this, at
> world-champion strength), `E-EXCEL-SHADER-PROJECTION` (the demonstrator),
> `E-OGAR-ROUTER-ENCODER` (2-axis = GREEN), `E-SOA-CYCLE-OWNERSHIP` +
> `mailbox-cycle-aware-write-contract` (`write_row`), the DO-arm `action.rs`.

## The claim being proven

A spreadsheet (≡ a chess board's incremental-eval loop) projected onto the
shader: **cells = SoA rows (2-axis `(row,col)` router address); formulas =
`ActionDef`s with `depends_on` edges; editing a cell triggers a topological
recompute of dependents, gated per-cell by the cycle-aware `write_row`.** This
is NNUE's "only small input changes between neighboring positions → incrementally
update," re-expressed as `ClassView::compute_dag` + `write_row`.

## Why this probe (not a bigger build)

- **Lands the one Core gap** (`ClassView::{compute_dag, constraints}`, core-gap-auditor)
  that EVERY computed-field AR consumer (odoo `@api.depends`, medcare lab-trends,
  woa calc, q2 cells) needs — they all reduce to a sheet.
- **2-axis → dodges the open questions:** no 3D-Hilbert axis-count crack, no
  bit-partition proof needed (those gate the *spatial-wave* `E-OGAR-ROUTER-ENCODER`
  side, not this recompute structure).
- **Proven shape:** NNUE/AlphaZero are the existence proof; this is re-derivation,
  not invention.

## Increments

0. **`ClassView::compute_dag(classid) -> &[ComputeEdge]`** (the Core extension) —
   a per-class topological recompute manifest sourced from `depends_on` +
   `emitted_by` (harvest-shaped); registry-build **rejects cycles**. Mirrors the
   existing `ClassView::{fields, value_schema}` resolvers; stores nothing on the
   row (the harvest IS the manifest). Layout-preserving (no `NodeRow`/stride/
   `ENVELOPE_LAYOUT_VERSION` change). Sibling: `ClassView::constraints` (deferred
   to a follow-up; `validation_kind`-sourced).
1. **Minimal sheet harness** — cells = `MailboxSoA` rows, addressed `(row,col)`
   (the 2-axis `axis_binding=Spatial(x/y)` from `ClassView::axis_binding`);
   each cell's value in a value tenant (`Energy` numeric); formulas as
   `ActionDef`s (`predicate`=op, `depends_on`=precedent cells).
2. **Topological recompute on edit** — edit a cell → dirty-set → recompute
   dependents in `compute_dag` topological order, each recompute a `write_row(cycle)`
   so it carries the recompute-generation; the dispatch is the `CognitiveShader`
   "can't-NOT-recompute-while-dirty" loop.
3. **Success / falsifier criteria:**
   - editing `A1` dirties + recomputes its transitive dependents in topological
     order, each cycle-stamped (`last_write_cycle`); a non-dependent cell is NOT
     recomputed.
   - a formula loop is **rejected at registry-build** (cycle in `compute_dag`).
   - `ValueSchema::Cognitive ∩ Compressed = {Fingerprint, EntityType}` disjointness
     holds (no over-collapse, ripple guard).
   - incremental-update parity: recomputing only the dirty-set yields the same
     result as a full recompute (the NNUE incremental ≡ full-eval invariant).

## Scope line (do not over-claim)

"Shader projection" = the **dependency-driven recompute dispatch** (`compute_dag`
+ cycle), NOT Walsh-Hadamard field synthesis. Per-cell formula *semantics*
(`=VLOOKUP`, `=IF`) are general compute dispatched via the DO arm / `UnifiedStep`
— exactly as a chess engine's per-square eval is general, not the field encoder.

## Deliverables carried (all layout-preserving)

`ClassView::compute_dag` (this probe) → unblocks computed-field adoption for all
AR consumers. Pairs with the already-converged `ClassView::axis_binding`
(Spatial for the sheet) + the `Field` `ValueSchema` preset. Chess-side design
reference: NNUE incremental update + AlphaZero `8×8×C` planes (`E-CHESS-TENSOR-PROVEN`).

## Blockers / sequencing

- `compute_dag` is a `lance-graph-contract::ClassView` trait extension — additive,
  default-method, mirrors `value_schema`. No stride/version impact.
- Independent of the 3D-spatial / `lite-unified` / surreal-kv-lance threads.
- Best first proof: it lands the gap for every computed-field consumer at once.
