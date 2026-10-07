# 2026-10-07 — The subtraction law holds; its unit is the u64 word, not the u16 cell

## MEASURED

`crates/lance-graph-benches/examples/aperture16_probe.rs`, over a 65 536-self universe:
- 11 densities × 8 layouts × 8 payload widths × 10 routes, plus VIA, K propagation over CSR, extent × mask, and target-side scheduling;
- every route asserted equal before timing.

- The `[u16; 4096]` view of the 64K mask is zero-copy: it is a shift of the same 8 KiB, endianness-proof, and checked against `align_to`. As a schedule, though, it is **1.3–2.3× slower per layout** (2.7× over all ladder cases) than u64 set-bit visitation, and 7× slower on an empty mask (4096 tests against 1024).
- Adding a full-word dense path to the u64 walk (`adapt64`): geomean 0.80, 2.4–2.9× on runs, at most 17 % worse on any layout.
- Full 16-run detection (`adapt64>16`) wins on 16-aligned islands (0.40) but is 1.25–1.64× slower on scattered layouts. A branch-free form does not remove that cost.
- Extent and mask compose multiplicatively: 2.9 µs, against 29.1 for extent only and 11.1 for mask only. The extent alone suffices only when the mask is dense.
- The mask schedules VIA and K propagation with no selection vector. The ordinal list shows no reproducible win; every apparent win reversed on re-run.
- On the target side, writing the exact next-frontier mask in the same pass is up to ~20× faster than clearing and scanning `K_next`, and the fastest route in 20 of 30 cases with run order rotated. The `[u16; 4096]` histogram pre-pass is never the fastest route (0 of 30).

## CONSEQUENCE

- No new carrier and no new V4 opcode. The mask is authoritative; u64 words are the unit.
- BUILD candidates, unbuilt:
  - gated `mask_gather_u32` (the D-GATED-GATHER-0 item);
  - a full-word dense path in the fold kernels' u64 walk;
  - next-frontier-mask-in-pass for multi-hop K.
- KILL: u16 cells as the scheduling unit; popcount-threshold switching; `MultiplicityAperture`.

## OPEN

- Single machine, single thread, hand-rolled routes. Only A1's count goes through mask-risc.
- Host timing drifted by up to 2.4× on an identical pass, with no CPU steal. A4 uses a per-row minimum over 5 runs and A6 a median over 4 rotated runs; the other modes are single runs.
- Review corrections (#1377): cache-line counts use the real allocation offset; the 100 % extent case covers the whole universe; A6 rotates route order.
- Assembly not inspected; whether the dense paths auto-vectorise is unverified.
- `quack::SemanticAperture` already uses the word "aperture" for a different concept (a facet-key care mask).

Report: `.claude/research/D-APERTURE-16-0.md`, `D-APERTURE-16-MATRIX.md`, `D-APERTURE-16-tables.md`.
