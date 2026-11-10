# 2026-11-10 — The ISA lab resolves through `LaneBinding::borrow_all`

**Status:** MEASURED (lab). `crates/dr-isa-stream-lab/src/shape.rs`.

`VirtualObject::resolve` now uses DT64's `borrow_all` (#1988) in place of a
hand-rolled check followed by `LaneBinding::addr` + `LaneSource::borrow`. Lane
provenance is the identity of the resolving object plus its frame issue,
compared in O(1).

**Measured, warm resolve at 256 fields:**

| Implementation | Time |
|---|---|
| #1987 as first measured | 13.8–13.9 µs |
| `main` after #1987's provenance fix | 26–28 µs |
| `borrow_all` (this change) | 1.6–2.0 µs |
| `lane()` per field (baseline) | 119–170 µs |

**Finding:** `LaneBinding::addr` is a linear scan. Any per-field loop over it
is O(N²), and that is where the original 14 µs came from, not from the check.

**Open:** consumers that call `addr` per field elsewhere should move to
`borrow_all`.
