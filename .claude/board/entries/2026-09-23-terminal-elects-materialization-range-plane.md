# 2026-09-23 — The terminal elects materialization: fused `Range ∩ plane → Count/Any`

**Status:** MEASURED (flagship W2b-A) · OPEN (W2b-B reuse burden; plane∩plane and
compare→Count fusion; execution extent)
**D-ids:** D-WFL-W2b‴ (arm A shipped), D-WFL-T1-FUSED′ (refuted for this shape),
D-WFL-MASKOP / D-WFL-EXPR / D-WFL-SEAMB″ (corrected, still queued), D-WFL-FUSE (confirmed)

## What was actually wrong (re-audited on main 33df0710)
- **Stale:** D-WFL-MASKOP's "`exec.rs:566` forces every slot to `words_for(n_rows)`" and the
  plan's "a `Pred::Range` at 1M rows writes 125 KB". Since #1266, scratch is tiled
  (`TILE_WORDS = 8`, `tile_words_for`), so a slot is at most 8 words.
- **Still true, and stricter:** `Range ∩ resident plane → Count` wrote derived membership
  on every tile. Quack lowers to `Pred{Range, under: gate}`. The executor then ran
  `mask_set_range` and `mask_and_assign` into the tile slot, and `Count` read it back with
  `popcount_batch_u64`. The work was also not bounded by the touched span: every tile of the
  population was walked, however narrow the range.

## What changed
In `crates/lance-graph-mask-risc`:
- `Program::fused_terminal()` recognises `[Pred{Range, under: None | Plane}]` followed by
  `Count` or `Any` of that slot. `Program::requires_scratch()` is DERIVED from it, and
  `touched_words(lo, hi)` is the one spelling of the span.
- `execute_into` folds a fused program straight from the resident plane's touched words
  plus two register-masked edge words. Validation stays total, using a local bookkeeping word.
- `Scratch::for_program` and `over_for_program` carve **zero** slots for such a program.
- `Keep` is never fused: it is the explicit election of a bitmap.

No second evaluator, no new IR and no ndarray change.

## What materialization disappeared, and what remains deliberate
- **Gone:** on the fold arm, derived words written = 0 and scratch slots carved = 0. Gated
  by a poisoned caller arena that must stay all-`u64::MAX`; a twin test proves the same
  probe sees `Keep`'s carve.
- **Allocation:** `Scratch::for_program` allocates 0 bytes for a fold, counted per thread.
- **Deliberate:** `Keep` still writes the demanded `Out::Mask`.

## D-WFL-T1-FUSED′ refuted for this shape
No new primitive was needed. A range's interior mask is all ones, so the relation is the
plane's own words over the span. The existing `popcount_batch_u64` and `mask_any` are
enough, called on the borrowed slice and on two one-word register temporaries. This
confirms D-WFL-FUSE: it was a lowering rule, not a primitive. The general slice∩slice
`popcount(a & b)` still lacks a buffer-free primitive (see the classification below).

## Measurements
`cargo run --release -p lance-graph-mask-risc --example range_fused_probe`. The two arms
agree on every case (asserted). Median ns:

| N | range | plane | materialized | fused | derived words written, mat / fused | plane words read, mat / fused |
|---|---|---|---|---|---|---|
| 4,096 | tiny | dense | 653 | 54 | 128 / 0 | 64 / 1 |
| 65,536 | 25% | dense | 9,002 | 119 | 2,048 / 0 | 1,024 / 257 |
| 1,048,576 | tiny | dense | 138,620 | 56 | 32,768 / 0 | 16,384 / 1 |
| 1,048,576 | 25% | dense | 141,688 | 1,072 | 32,768 / 0 | 16,384 / 4,097 |
| 1,048,576 | whole | dense | 148,396 | 4,435 | 32,768 / 0 | 16,384 / 16,384 |

- The fused latency is flat in N for a tiny range (54 → 56 ns from 4K to 1M rows) and
  scales with the touched span, not the population.
- The word counts are derived from the executor's contract, not from instrumentation:
  - Tiled path: two slots written over every tile.
  - Fused path: reads exactly `touched_words`.
  - The fused-arm zero is the one the gate test enforces.
- Sparse-scattered rows at 1M showed run-to-run noise, up to 297 µs on the materialized arm.

## Falsifiers (`tests/fused_terminal.rs`, disable-verified)
| gate | disable | result |
|---|---|---|
| fold == Keep→popcount/any == scalar oracle, over 4 row counts × 4 plane shapes × the named edges (empty `[65,65)` / `[0,0)`, single row, aligned/unaligned, inside one word, across a word, across tiles, sub-64 tail, whole) | tail edge word not counted | red: `count one n=1024 [60,70)` |
| poisoned arena untouched + zero slots carved | always carve the declared slots | red |
| `for_program` allocates 0 bytes for a fold | same | red |
| fusion recognised at all | `fused_terminal` → `None` | 4 of 7 red |

The allocation gate first flaked (120 stray bytes): the counter was process-wide while the
harness runs tests in parallel. It is now per-thread.

## Classification of the remaining ops (WORKING-MODEL, not measured)
| op → scalar terminal | class |
|---|---|
| `Range`, bare or gated by a resident plane | **fused** (this entry) |
| `Not(a)` → Count | fusion rule: `n − popcount(a)`; no primitive |
| `And` / `AndNot(plane, plane)` → Count | needs a T1 primitive: slice-slice `and_popcount` (ndarray has `U64x8::xor_popcount` as the precedent, but no AND form) |
| `Or` → Count | fusion rule once `and_popcount` exists: `\|a\| + \|b\| − \|a∧b\|` |
| `Xor` → Count | needs a u64-slice XOR popcount (the fused `hamming_distance_raw` exists on bytes only) |
| `Ternlog` → Count | needs `ternlog_popcount`, which generalises the three above |
| lane compares (`Gt`/`Eq`/… `_to_mask_under`) → Count | needs compare-count primitives; today a tile-local write remains |
| `Gather`, `Keep`, `ScatterOr` | deliberate materialization |

The desired direction is FEWER concepts: one `ternlog_popcount` would subsume `and`, `or`,
`xor` and `andnot` counting. Not built here; the flagship needed none of it.

## Still open
- **W2b-B**: the Keep arm is tested, but its downstream-reuse burden (name and measure the
  consumer that justifies the carrier) is not met.
- **Execution extent** (PR C): `execute_into` has no ranged entry point.
- **D-WFL-MASKOP**: `MaskOp` still reads as an assignment. Only the terminal-side lowering
  moved.
