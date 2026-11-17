# D-VRL-RECIPE-0: a VRL subset lowered onto the witnessed selection path (2026-11-17)

**Status:** MEASURED, lab only (`crates/vrl-recipe-probe`, own workspace).
Follows the Vector/VRL adapter contract in
`2026-11-16-g2-dr-virtual-projection-reconcile.md` §D.

## Prior art (this builds on it, it does not replace it)

- D-DR-SELECT-K-SV-1 (#2039): `execute_sel`, a mask-risc fold over borrowed
  positions.
- D-DR-SELECT-K-SV-2 (#2040): a selection witnessed with its lanes:
  `FrameStore::check`, `LaneSource`, `BindStore::bind_lanes`,
  `LaneBinding::{check, lane, selected, borrow_all}`, `fold_bound_selection`.
- D-DRDT-SPLIT-1 (#2016): version the mapping, not the content.

A first version of this probe (same PR, earlier commits) carried its own
store, version counter and folds. That duplicated the witness above; it was
replaced. What this probe adds is only the front: the VRL subset, compiled
into a recipe that the existing path executes.

## What

- **Recipe**, compiled once against a class's `ClassView` field labels: the
  fields it reads (a `WideFieldMask`, outputs and filters), the `== v` keep
  and drop terms, and the output aliases. Paths follow VRL's in-place
  mutation (a reassigned `.key` is what later reads see).
- **VRL subset**: `.out = .key`, `.out = .other`, `.out = .properties.f.value`,
  `del(.out)`, `if .p == v { abort }`, `if .p != v { abort }`. Refused by
  name: function calls other than `del`, arithmetic, string and integer
  literals as values, comparisons other than `==`/`!=`, an `if` body other
  than `abort`, assignment into `.properties`, unknown paths.
- **Witness**: every terminal borrows its lanes through one
  `LaneBinding::borrow_all`. A moved frame, a field the binding does not
  hold, or a non-`i32` lane is refused before folding.
- **Fold**: one mask-risc `Program` per terminal: `Range` (every row),
  `EqI32` per term, `And` / `AndNot`, then `Count`, `MaskedSumI32` or `Keep`
  into a caller mask. The computation is mask-risc's on `ndarray::simd`.
- **Rows**: `materialize_rows` returns positional `Rows`: names once,
  borrowed `&[i32]` lanes, a `u32` position table. No string or event per
  row; the name-per-row `Event` map exists only in the reference
  interpreter.

## Where it lives, decided

- No `lance_graph_dt64::vrl` module and no ndarray `simd_vrl.rs`. VRL is edge
  syntax; it lowers onto words and kernels that exist (`EqI32`, `And`,
  `AndNot`, `Count`, `MaskedSumI32`, `Keep`; `simd_masking_ops`). If the
  lowering is ever promoted, it is a thin VRL → `LaneBinding` + mask-risc
  front, beside the existing alias views, once a consumer (relalg, XLOG
  methods, Quack, EP64, MeTTa, mistaber, AriGraph) calls it.

## Result

- 7 programs × widths 0, 1, 63, 64, 65, 256, 512, 1000: rows (row order,
  names), count and every output sum equal the interpreter.
- Each filtered program keeps strictly fewer of 512 rows than its unfiltered
  form.
- Allocations at n = 64, 512, 4096, 65536 (counts, not timings):
  `count` / `sum` hold mask-risc's scratch, two slots of at most one tile
  (`2 · 8 · min(⌈n/64⌉, 256)` bytes, at most 4 KiB at any n), plus 392 bytes
  of bookkeeping at every n; `materialize_rows` adds the kept mask
  (`8 · ⌈n/64⌉`) and the position table (4 bytes a kept row), plus 474 bytes
  at every n. Allocation calls are the same at every n for all three.
- 12 refusal cases return the named refusal; a moved frame is refused at
  every terminal (`LaneRefusal::Frame(SourceChanged)`).

## Falsifiers

Each patch anchored (`count == 1` asserted), run, restored from the commit:

| disable | red |
|---|---|
| `count` by `materialize_rows().len()` | `terminals_allocate_tile_scratch_mask_and_positions_only` |
| a `format!` string per materialized row | same |
| lanes borrowed by address without the DT64 check | `a_moved_frame_is_refused_at_every_terminal`, `unbound_and_non_i32_fields_are_refused` |
| drop terms ignored (`AndNot` → `slot0 & slot0`) | `recipe_equals_the_interpreter_at_every_width`, `every_filter_drops_rows` |
| `.key` always reads the source field | `recipe_equals_the_interpreter_at_every_width` |

## Found on the way

- At 65,536 rows mask-risc tiles its scratch, so a `Keep` result must go to a
  caller buffer (`Out::Mask`); reading it from a slot fails with
  `TerminalNeedsOut`. Widths up to 1000 never reach the tile edge; only the
  allocation test did.
- `Vec::resize` on an empty `Vec` reserves at least 4 words; an exactly
  sized `vec![0; words]` removed a 24-byte n-dependent difference at n = 64.

## Open

- Population terms are `== v` only; the cube's `SetExpr` and the full
  mask-risc predicate set are the natural next step.
- Ingress from real Vector (JSON) events is not built.
