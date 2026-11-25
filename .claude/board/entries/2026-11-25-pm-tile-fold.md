# 2026-11-25 — P×M statistics fold per sealed DR tile and combine exactly (EXO-398)

**Status:** MEASURED-LAB. `crates/pm-tile-fold-probe`, not a workspace member.

## What was asked

Can N, S, Q, A of a Codec9 field be folded per sealed DR tile and summed
across tiles, with a part refused once its tile no longer has the frame the
part was folded under?

## What the probe does

- A tile is a source that lends one Codec9 field through the existing socket:
  `&[u8]` magnitudes plus the polarity plane, keyed by one `LaneAddr`
  (`borrow_all_present_bound`, #2160). No new socket or type in dt64.
- `fold_tile` folds the borrowed lane in place and returns a `TilePart`: four
  numbers and the binding's frame. A part is an aggregate, not a copy of rows.
- `combine` checks every part against the store of the tile it is presented
  with (`FrameStore::check`), then adds with checked arithmetic.
- Padding a tile to whole blocks is neutral: an untouched row has `M = 0`, so
  only `N` reads `touched`.

## Falsifiers (6 tests, green)

- Every part equals a row-by-row oracle, on tiles of 1, 64, 77, 1000 and 4096 rows.
- Combined parts equal the whole population in every rotation of the tile order.
- A part from before its tile was relaid is refused (`SourceChanged`); after a re-fold, the parts combine again with the new values.
- A part presented with another tile is refused (`ForeignFrame`).
- A lane whose plane or selection describes other rows is refused.
- Overflow refuses; variance is the exact rational, `None` for untouched only.

Disable runs, each red:
- `disable-combine-frame-check`: 2 tests fail.
- `N` counted from `P` alone: 3 tests fail.
- `S` ignoring polarity: 3 tests fail.

## Measured (ns per 64 lanes, best of 7, 1M lanes total, two runs per tier)

| tiles × lanes | AVX-512 fold+check | AVX-512 plain | AVX2 fold+check | AVX2 plain |
|---|---|---|---|---|
| 1 × 1M | 9.1–9.4 | 9.1–9.3 | 33.3–33.6 | 33.3–33.4 |
| 16 × 64K | 9.2–9.4 | 9.2–9.3 | 33.5–33.7 | 33.6–33.9 |
| 256 × 4K | 9.8–10.0 | 9.2–9.4 | 34.1–34.4 | 33.5–33.8 |
| 4096 × 256 | 19.3–21.0 | 9.3–10.0 | 43.8–46.5 | 33.7–35.0 |

`combine` costs 20–24 ns per part on both tiers, about 60 ns for one part.

- Down to 4K lanes per tile, the socket check and witness are under 7% of the fold.
- At 256 lanes per tile the fixed cost per tile, about 40 ns, doubles the cost per lane. That is the check and part construction, not the fold.
- The ~20 ns per `FrameStore::check` is not this probe's. `check` compares `Weak` pointers by building a temporary `Arc::downgrade` each time, which costs an atomic increment and decrement per comparison. Comparing raw pointers (`Weak::as_ptr` against `Arc::as_ptr`) would answer the same question with no atomics. Not changed here; noted for dt64.
- AVX2 is about 3.6× slower than AVX-512 for the full four-statistic fold, the same gap as ndarray #439.

## OPEN

- Whether a part should hold the frame (as here) or the caller should hold a lease; this probe picks the cheapest witness.
- Folds across tiles of different classes or widths; every tile here has one class and one width.
- The `FrameStore::check` atomics, above.
