# 2026-11-25 — D-AED-2: a sealed slice at its physical width

**Status:** MEASURED (storage only). Plan: `.claude/plans/adaptive-exact-decimal-binding-v1.md` D-AED-2.

`PhysicalSlice` (`crates/exact-decimal-probe`) stores a sealed slice at the
narrowest width that holds it: PM9 packed (`u8` magnitude plus a polarity
plane, lent to Abakus as a `Pm` register per 64 values), PM17 as `u16`
magnitude plus a polarity plane, wider as `i32` plus a presence plane. P×M
is the DR512 law: untouched `P=0 M=0`, `+0` is `P=1 M=0`, `-0` never written,
presence `P ∨ M ≠ 0` (Abakus `touched` for PM9). Round trip, presence and
identity order are exact (tests in `tests/probe.rs`; two disable runs red).

Command: `cargo run --release --example physical_width`. Encoded cost per
value: a notional 10 B descriptor per slice, the magnitudes and the polarity
or presence plane. `Vec` headers and struct padding are not counted, in
either column:

| slices | values | encoded | as i32 |
|---|---|---|---|
| #2182 fixture, 3 × 5 (PM9, PM17, PM17) | 15 | 5.267 | 7.600 |
| half integers, half scale 1 (PM9, PM9) | 65 536 | 1.125 | 4.125 |
| PM9 limit, \|c\| ≤ 255 | 65 536 | 1.125 | 4.125 |
| PM17 limit, \|c\| ≤ 65 535 | 65 536 | 2.125 | 4.125 |
| one value past PM17 widens the slice | 65 536 | 4.125 | 4.125 |
| 64 one-value slices | 64 | 19.000 | 22.000 |
| all absent (PM9) | 65 536 | 1.125 | 4.125 |

- A resident Abakus `Pm` is a SIMD register padded to its alignment: 128 B
  on AVX-512 (`pm9_is_stored_packed_not_as_resident_registers`), not 72. A
  first version stored `Vec<Pm>` and counted 72 B per block, which understated
  the cost (codex review on #2187). PM9 is now stored packed and lends a `Pm`
  per 64 values for arithmetic; short slices no longer pad to 64 values.
- One large value moves the whole slice to `i32`: the case D-AED-4's split
  into a new slice exists for.

OPEN:
- the owner seam. `LaneRef` has no U8/U16 operand and no source lends one; a
  variant is a LAYOUT change (D-DRISA-0/1, D-LANESOCK, v3-envelope-auditor).
  Nothing here is read by mask-risc yet.
- PM17 and wider have no Abakus arithmetic (ndarray #438).
