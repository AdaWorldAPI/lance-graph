# 2026-10-04 — Register128: classid-free 128-bit register + bounded power sums as its first consumer (D-LXC-29-R)

## DECISION (operator ruling, 2026-10-04)

- **DECISION:** alternative 2, carried to a live consumer. A slab may declare the physical reading `SlabReading::Register128` (tag 1). Its bytes are 128 raw bits with no classid inside.
- **SCOPE:**
  - The concept comes from the SPOG context, resolved through `Activation::resolve_for_context` and handed back by `RegisterLanes::concept()`.
  - Facet96 (tag 0) is untouched, and none of its bytes are reused.
  - `TurbovecResidue` is not used.
  - #1323's logical APIs are unchanged.
- **BASIS:** "classid-free" means not embedded in the payload; it does not mean semantically untyped.
- **REVISIT WHEN:** a consumer needs more than two rails, or the rails collide with a future tenant.

## What landed

**lance-graph contract (`58c1c72`)**
- `SlabReading::Register128 = 1`; tags 2, 3, 0x80 and 0xFF still fail closed.
- `ValueTenant::Register0` and `ValueTenant::Register1`, each 16 bytes:
  - row bytes [252,268) and [268,284), value offsets 220 and 236;
  - appended after `EpisodicBasin` and included in `ValueSchema::Full` only;
  - the `BoardAggregates` reservation re-bases from ordinal 16 to 18.
- `register128.rs`:
  - `Register128([u8;16])`, read as 4 LE `u32` words;
  - `RegisterRails::{One, Two}`;
  - `RegisterLanes`, which is `Copy`, holds only numbers, and offers `get`/`set` per rail.
- `ResolvedReading::bind_register128(rails)`:
  - checked once per population;
  - refuses a non-Register128 slab (`NotRegister128`);
  - refuses a schema without the rails (`RegisterRailAbsent`).
- **`ENVELOPE_LAYOUT_VERSION` is NOT bumped.** Appended tenants are layout-preserving (same precedent as earlier appends), and a bump would make every existing Facet96 slab fail closed.

**ndarray (`eeb911b`, plus bench)**
- `BOUNDED_TILE_ROWS = 65,536`, with a compile-time proof that `255² · 2^16 < 2^32`.
- `masked_group_bounded_power_sums_u8{,_via,_pair}`:
  - register layout `[n, Σx, Σx², reserved]`;
  - word 3 is never read or written.
- `masked_group_bounded_cross_power_sums_u8{,_via,_pair}`, on two rails:
  - rail0 = `[n, Σx, Σx², ·]`, identical to the univariate register;
  - rail1 = `[Σy, Σy², Σxy, ·]`;
  - so `n` is stored once.
- All six go through the existing `group_walk`, so lane, via and pair semantics and the drop rules are those of the `i32` kernels.
- `widen_bounded_{,cross_}power_sums` give the exact `PowerSums` / `CrossPowerSums`.
- `fold_bounded_{,cross_}power_sums_tiles` cut a population into tiles of at most 2^16 rows and `checked_merge` each tile. A tile is refused whole if any group's merge would overflow.
- A tile over the bound returns `TileTooLarge` before any write.

**jc end-to-end (`e723858`)**
- The context is resolved and both rails bound once.
- A four-tile population is folded through ndarray.
- Each tile's registers are stored per group into `NodeRow` rails, then read back, widened and merged. The result equals the wide `i32` path, univariate and bivariate.
- A Facet96 or undeclared slab never binds.

## Proofs (each disable-verified red, then restored)

| claim | test | disable |
|---|---|---|
| rails only for a Register128 slab + Full schema | `register_rails_are_granted_only_to_a_register128_slab` | slab check removed; rail-presence check removed |
| concept from context, never payload | `a_register_takes_its_concept_from_the_context_never_the_payload` | — |
| 65,536 × 255 fits exactly | `the_full_bound_at_u8_max_fits_exactly` | — |
| 65,537 rows refused, nothing written | `one_row_past_the_bound_is_refused_before_any_write` | bound guard removed → red |
| narrow → widen exact, lane/via/pair | `narrow_then_widen_…`, `bivariate_narrow_then_widen_…` | widen word swapped → red (3 tests) |
| tiles + checked_merge == whole | `partitioned_tiles_merge_to_the_whole` | merge replaced by overwrite → red |
| overflowing tile commits nothing | `an_overflowing_merge_commits_nothing_of_the_tile` | check fused into commit loop → red |

## MEASURED (AVX-512 host, `avx512f=true`, release, 16 groups, median of 31 runs, ns/row)

| case | wide i32 | bounded u8 | ratio |
|---|---|---|---|
| univariate tile=4096 | 1.448 | 1.094 | 1.32 |
| bivariate tile=4096 | 3.032 | 1.921 | 1.58 |
| univariate tile=16384 | 1.445 | 1.095 | 1.32 |
| bivariate tile=16384 | 3.010 | 1.906 | 1.58 |
| univariate tile=65536 | 1.432 | 1.095 | 1.31 |
| bivariate tile=65536 | 3.264 | 1.911 | 1.71 |
| univariate tiled n=1,048,699 | 1.514 | 1.205 | 1.26 |
| bivariate tiled n=1,048,699 | 3.477 | 2.133 | 1.63 |

- These are single-host, single-session numbers. AVX2 and NEON were not measured.
- Input is 1 byte per lane against 4 for the wide path.
- The wide path is timed on pre-widened `i32` lanes, so the conversion cost the bounded path avoids is not included.

## OPEN

- **Zero-copy strided writes:** the kernel writes a compact `&mut [[u8;16]]` working set (one register per group). These are stored to rows per group via `RegisterLanes::set`, not strided into `NodeRow` in place.
- **No production call site yet:** the mask-risc terminal for the bounded fold is not wired.
- **Declaration storage:** where `SlabDeclaration` lives in the metadata envelope, and the writer that persists it.
- **Unmeasured tiers:** AVX2, NEON and wasm timings.
