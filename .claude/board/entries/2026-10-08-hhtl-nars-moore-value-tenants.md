# 2026-10-08 — HHTL / NARS / Moore value tenants (ordinals 18–20)

**Status:** physical layout RATIFIED and TEST-PINNED; MooreNars16 semantics
CANDIDATE. Branch `ccr-b2e415d9-4jfvyk-tenants`, commit `28ec7c0`.

## What landed

Three append-only value tenants after `Register1`. Layout-preserving: `NodeRow`
stays 512 B, tenants 0..17 and every CE64 field are unchanged, no
`ENVELOPE_LAYOUT_VERSION` bump.

| ord | tenant | kind | row bytes | slab bytes |
|---|---|---|---|---|
| 18 | `Nars16x8` | U16 × 8, LE | [284,300) | [252,268) |
| 19 | `MoorePalettePairs` | U8 × 16 | [300,316) | [268,284) |
| 20 | `MooreNars16` | U16 × 8, LE | [316,332) | [284,300) |

- Lane `i` = Moore slot `i`, order `NW,N,NE,W,E,SW,S,SE` (the probes' order).
- Pair byte `2i` = first Palette256 operand, `2i+1` = second; the law address
  is `(first << 8) | second`, the #1336 `PairAddress` orientation.
- `MooreNars16` = `Pearl3 | Energy4 | Plasticity3 | Polarity1 | Epi5`
  (bits 0..2, 3..6, 7..9, 10, 11..15), the #1406 reading.
- `ValueSchema::Full` carries all three; no narrower preset does, because no
  consumer materialises them yet. Full total 252 -> 300 B.
- `BoardAggregates` reservation re-based 18 -> 21 (canonical_node comment and
  the stale `band_reading.rs` note).

`contract::moore_tenant::{MooreSlot, MooreNars16, DirectionReading,
DirectionRefusal, MooreRefusal, MooreTenantView, MooreTenantMut}`: lane access
goes through `ValueTenant::value_offset()`, no literal offset.

## Refusals

- A sign-triple consumer refuses a Moore direction:
  `DirectionReading::require_sign_triple` returns `DirectionRefusal` for
  `Moore { slot, polarity }`.
- `MooreTenantMut::lift_ce64` refuses mixed witnesses and Epi5 codes 24..31,
  checking all eight lanes before writing any.
- The witness is not stored in these bytes; the lift returns it to the owner.

## Tests (A–L)

`moore_tenant.rs` (11 tests: A, B, C, D, E, G ×2, H, I, K, L);
`cognitive-shader-driver/tests/moore_palette_pairs_lut.rs` (F);
`lance-graph-planner/tests/moore_tenant_isa_equivalence.rs` (J, 2 tests:
256 tenants × 8 lanes × 5 ops, 0 divergences on every field except the sign
triple; restoring the wrong witness diverges).

Disable runs, each red then restored:

| disable | caught by |
|---|---|
| slot order E/W swapped | E, I |
| u16 lanes big-endian | D, I |
| pair bytes read swapped | F |
| witness check removed | I |
| refused lift writes a lane first | I |
| Moore direction passes as sign triple | H |
| energy read as 3 bits | J |
| `ce64_upper` drops the witness | J |
| `Nars16x8` descriptor U8 × 16 | A |
| `Full` drops `MooreNars16` | compile-time assert |

## OPEN

- The `Nars16x8` lane reading is not declared anywhere.
- No `SlabReading` variant declares these tenants, so `ResolvedReading`
  cannot yet gate a consumer on them the way `bind_register128` does.
- No production writer or reader. The production Simpson detector still takes
  bare edges and cannot refuse a Moore direction (pinned in #1406).
