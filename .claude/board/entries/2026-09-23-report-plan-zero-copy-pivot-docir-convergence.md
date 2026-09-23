# 2026-09-23 — ReportPlan: fold-first reporting, zero-copy pivot, and reports as OGAR projection sources

**Status:** MEASURED (tests + allocation falsifiers + benchmark) · OPEN (composite-key
fold; DocRenderer trait; Snapshot pinning; sparse high-cardinality partitions)
**D-ids:** D-RPT-1 (report plan crate), D-RPT-2 (z8run nodes), D-RPT-3 (OGAR grid
projection), D-RPT-4 (report → ObjectSlot adapter)

## What landed
- `crates/lance-graph-report` (workspace member). `ReportPlan` = Selection + coordinate
  providers with roles + mergeable fold states. It LOWERS INTO Quack (`lance()` →
  `Program`), and mask-risc's evaluator runs it, so there is no second evaluator and no
  independent aggregate semantics.
  - `PhysicalKey` excludes roles, axis order and top-k. A rotated plan and its twin share
    one fold; rotation is `Arc<CellSpace>` + a new `View`.
  - Planner choices, each with an inertness test:
    - fold key = the widest ordinal coordinate within budget;
    - partitions = member conjuncts, evaluated tile-local; a derived `Bucket` is two range
      compares and is never a lane;
    - dense accumulator vs sparse, where discovery counts per domain and the product is
      never allocated;
    - selection carrier = validity plane, execution extent, tile-local ops, or ONE reused
      mask when the programs × passes count warrants it.
  - The execution modules are string-free: `FieldId` / `MaskId` / `SourceId` / ordinals.
    A source fence test enforces it.
  - Text lives only in `boundary` (KV = the contract's `ContentStore`/`ContentSink`; the
    CAM codebook maps (FieldId, ordinal) ↔ ContentId and stores no text), `render`
    (JSON/CSV data export) and `explain`.
- `crates/lance-graph-report-ogar` (EXCLUDED; path-deps the OGAR sibling).
  - `ReportSource` wraps ANY `DocObjectSource` and answers `grid_of` for report views.
  - One `DocCompose` then arranges object A · report · object B by reference and
    resolves them in one `resolve_doc`. `emit_typst` renders through OGAR's emitters.
  - `Live` = newest result; `Revision(n)` = source generation n, held by `Arc`;
    `Snapshot` → the slot's fallback.
- OGAR #307: `SlotOutcome::Grid` / `ResolvedGrid` / `DocObjectSource::grid_of`
  (default `None`) plus typst `emit_grid`. This is the one irreducible gap: flat
  (label, value) rows cannot carry a two-coordinate object, and u8 rail positions
  cannot address report coordinates.
- z8run #1: `z8run-lance`. FlowMessages carry a ~100-byte handle. The emitted plan ==
  the native plan, and so do the lowered programs.

## Measured
- 23 report tests + 4 convergence + 3 OGAR resolver + 6 z8run node tests.
  Disable-verified: rotation no-op, dropped partitions, grid_of bypass, orientation
  ignored, html refusal.
- Execution allocates the SAME bytes at N and 64·N (F8). The reused-mask twin grows by
  exactly the mask's words. Rotation allocates < 64 B (view) / < 256 B (plan
  reinterpret), with the payload address unchanged.
- Label resolution for a 100k-row 8×12 pivot = 20 CAM resolutions (presented members).
  The fold did 0 CAM and 0 KV work.
- `report_bench` (4M rows, avx2 build):
  - range→count 91 µs, 0 mask ops;
  - 2D pivot 32×12 177 ms (25 scans: 1 reused mask + 12 partitions × 2 states);
  - rotation 49 ns / 16 B via the result view, 124 ns / 140 B via plan reinterpret.

## OPEN
- **Composite-key fold is missing.** A partition costs one pass per member tuple, so a
  high-cardinality, densely-observed partition side is refused with `PassBudget`.
  Hand-rolling it here would be wrong: the primitive is a multi-lane group fold in
  `ndarray::simd` / mask-risc. → `ISS-REPORT-NO-COMPOSITE-KEY-GROUP-FOLD`.
- **OGAR `DocRenderer` is spec-only.** Every consumer, this adapter included, writes a
  thin block walker.
- **Snapshot pinning:** there is no content-addressed report store; it resolves to the
  fallback. `FieldView::Table` (askama side) is unbuilt.
- `lance-graph-sap` (#1257) already lowers `GroupSumI32` through Quack for CATS. The
  two are not reconciled; this crate is the domain-agnostic plan layer. UNKNOWN whether
  CATS should consume it.
