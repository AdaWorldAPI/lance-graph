# 2026-11-25 — D-AED-2: a sealed slice at its physical width

**Status:** MEASURED (storage only). Plan: `.claude/plans/adaptive-exact-decimal-binding-v1.md` D-AED-2.

`PhysicalSlice` (`crates/exact-decimal-probe`) stores a sealed slice at the
narrowest width that holds it: PM9 in Abakus `Pm` blocks, PM17 as `u16`
magnitude plus a polarity plane, wider as `i32` plus a presence plane. P×M
is the DR512 law: untouched `P=0 M=0`, `+0` is `P=1 M=0`, `-0` never written,
presence `P ∨ M ≠ 0` (Abakus `touched` for PM9). Round trip, presence and
identity order are exact (tests in `tests/probe.rs`; two disable runs red).

Command: `cargo run --release --example physical_width` (bytes per value,
descriptor 10 B per slice, polarity or presence plane included):

| slices | values | stored | as i32 |
|---|---|---|---|
| #2182 fixture, 3 × 5 (PM9, PM17, PM17) | 15 | 9.200 | 7.600 |
| half integers, half scale 1 (PM9, PM9) | 65 536 | 1.125 | 4.125 |
| PM9 limit, \|c\| ≤ 255 | 65 536 | 1.125 | 4.125 |
| PM17 limit, \|c\| ≤ 65 535 | 65 536 | 2.125 | 4.125 |
| one value past PM17 widens the slice | 65 536 | 4.125 | 4.125 |
| 64 one-value slices | 64 | 82.000 | 22.000 |
| all absent (PM9) | 65 536 | 1.125 | 4.125 |

- A PM9 block holds 64 values, so a slice pads to a multiple of 64. Below 16
  values in its last block, `i32` is smaller. Width by magnitude alone is not
  the smallest lawful representation for short slices.
- One large value moves the whole slice to `i32`: the case D-AED-4's split
  into a new slice exists for.

OPEN:
- the owner seam. `LaneRef` has no U8/U16 operand and no source lends one; a
  variant is a LAYOUT change (D-DRISA-0/1, D-LANESOCK, v3-envelope-auditor).
  Nothing here is read by mask-risc yet.
- PM17 and wider have no Abakus arithmetic (ndarray #438).
- choosing the width by bytes for slices under one block.
