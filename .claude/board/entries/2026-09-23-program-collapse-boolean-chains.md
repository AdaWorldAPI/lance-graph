# 2026-09-23 — Multi-op Boolean chains collapse onto the ternlog Count/Any fold

**Status:** MEASURED (local branch, not yet a PR — waits on #1270's manual review and merge) · OPEN (lane predicates, >3 leaves, the tiled path's own per-tile cost)
**D-ids:** D-WFL-FUSE (multi-op case). Builds on D-WFL-T1-FUSED / D-WFL-T1-FUSED′ (#1270, ndarray #322). OQ-5, Rayon and any scheduler: untouched.

## What changed
`Program::fused_ternlog()` no longer matches one op. It interprets the whole op sequence **symbolically**:
- every resident plane is a leaf table, in first-read order (`0xF0`, `0xCC`, `0xAA`);
- every op combines tables with the op's own Boolean law (`Ternlog` applies its immediate bitwise to the three input tables);
- each op writes the table to its destination slot, so slot reuse is modelled exactly.

The terminal slot's table becomes the fold immediate. Recognition is a fixed on-stack array (`FUSED_SLOT_CAP = 32`) and never allocates.

**Refused, and so left on the tiled path:**
- a fourth distinct plane;
- a `Pred` (comparison leaf) or `Gather` anywhere in the chain;
- a scratch slot read before it is written;
- a slot at or above `FUSED_SLOT_CAP`;
- any terminal other than Count/Any. `Keep` stays the explicit election of a bitmap.

## Gates
- **End to end:** `fuse_program` output reaches the fold.
- **Differential:** raw chains are checked against the tiled Keep path across extents and against `reference_execute` over the whole population. The raw chains are a catalogue of 7 plus 400 random ones, with an anti-vacuity check of at least 20 distinct counts.
- **Zero materialization:** a poisoned-scratch gate plus a counting allocator, with its Keep twin.
- **Disable runs** (committed first, red, then restored to green):
  - the interpreter returning only single-op programs;
  - a transposed table-apply;
  - slot reuse ignored (first write wins);
  - a fourth leaf silently aliased to leaf 0.

  All four fail inside `program_collapse.rs` itself.

## Measurement
Command: `cargo run --release -p lance-graph-mask-risc --example program_collapse_probe`. Setup:
- **Tier:** `x86-64-v3`, i.e. the AVX2 polyfill, which is the lance-graph default config. The host has avx512f, but **v4 was not measured**.
- **Size:** N = 1 048 576 rows, which is 16 384 words.
- **Values:** median ns, oracle-checked.

| chain | ops | extent | fold Count | tiled Count | fold Any | tiled Any | derived words the tiled path writes |
|---|---|---|---|---|---|---|---|
| `(a&b)\|!c` | 3 | whole | 8.6 µs | 224 µs | 0.12 µs* | 218 µs | 49 152 |
| `((a^b)&!c)\|(a&c)` | 4 | whole | 16.2 µs | 352 µs | 0.14 µs* | 351 µs | 65 536 |
| `maj(a,b,c)^a` | 2 | whole | 17.4 µs | 166 µs | 0.14 µs* | 167 µs | 32 768 |
| `a&d` (d = !a; empty by data) | 1 | whole | 6.5 µs | 120 µs | 4.4 µs | 111 µs | 16 384 |
| `(a&b)\|(c&d)`, **4 planes** | 3 | whole | — (refused) | 283 µs | — | 278 µs | 49 152 |
| `(a&b)\|!c` | 3 | 1% | 0.27 µs | 2.7 µs | 0.17 µs | 2.6 µs | 495 |

`*` = the fold exits at its first non-empty block; the tiled Any does not short-circuit. The data-empty row isolates the scan itself: **25×** for Any and **18×** for Count, with no early exit on either path.

**Readings:**
- The collapsed fold writes **0** derived words. The tiled path writes `ops × touched words`.
- **Whole-population Count gain: 10–26×.** This exceeds ndarray #322's single-op 1.3–1.9×. **Not every part of that gap is write elimination**: the tiled path also pays per-tile interpreter dispatch (TILE_WORDS = 8) for every op. This probe does not separate the two costs.
- **The boundary is visible.** The 4-plane chain has the same op count as `(a&b)|!c`, and at 283 µs it pays the whole tiled price. One ternlog addresses three inputs, and the collapse stops exactly there.
- **An algebraically constant chain** (e.g. `a & b & !a` → table `0x00`) monomorphizes to a fold that never reads the planes. That is correct, but it is not a scan, which is why the empty row is empty by data rather than by algebra.

## Open
- Lane predicates (comparison leaves) are deliberately not attempted. They deserve their own pause.
- For more than three leaves: tree splitting / multi-pass is not attempted. Only its boundary is measured.
- The split between per-tile dispatch cost and write cost on the tiled path is unmeasured.
- The avx512 (v4) tier is unmeasured.
