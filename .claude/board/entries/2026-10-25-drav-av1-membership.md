# D-DRAV-AV1 — map-only membership mask (lab, reported not gated)

**Status:** MEASURED (lab). Contract: [D-DRAV-1](../plans/d-dr-address-first-value-demand-v1.md)
(#1775, draft) AV1. Code: `crates/drf-focus-probe/src/membership.rs`, tests
`tests/integration/membership.rs`, cost probe `examples/av1_membership.rs`.

`M[i] = contains(S, DR64[i])`: a 4096-bit (512-byte) slot mask from a sealed DR64 ordinal map
and a set of canonical row ordinals in the same attested source domain and scope.

- **Value independence is structural.** `membership(&BoundMap, &RowSet)` takes no `Soa`,
  `Facet` or `Focus`, and neither argument holds a row or a facet. A `compile_fail` doctest
  passes a `Soa` and is rejected with E0308; the same snippet with a `BoundMap` compiles
  (checked, so the doctest fails for the stated reason).
- **Checks at the binding boundary.** `BoundMap::bind` refuses a seal mismatch and any ordinal
  past the attested row count; `RowSet::new` refuses rows past it; `membership` refuses a
  source (dataset/branch/version/seal) or scope mismatch. The slot loop does no checks.
- **Tests (6), against a linear-search scalar oracle:** random maps with hits and misses;
  1 000+ slots sharing one ordinal all match (no dedupe); empty and full sets; a seeded slot
  shuffle permutes the mask; same ordinals in another dataset, version or scope refuse; stale
  seal and out-of-range refuse.
- **Disable runs (red then green):** dedupe slots by ordinal (4 tests red); drop the domain
  check (collision test red); reverse bit order within a word (3 red).

Cost split, warm medians of 2001 calls, random map over 65 536 rows (native; `x86-64-v3`
agrees within noise):

| step | ns |
|---|---|
| bind (seal + 4096 range checks) | 930-1 370 |
| set prep, k = 64 / 1 024 / 16 384 rows | 140-210 / 540-850 / 8 300-11 800 |
| membership exec (any k) | about 2 000 (0.5 ns/slot) |
| bind + prep + exec, k = 1 024 | 3 450-4 270 |
| value gather from the same map, for reference | 12 600-14 000 |

The first loop (read-modify-write of the output word per slot) took about 6 000 ns; building
each 64-slot word in a register brought it to about 2 000. Both are scalar.

Not shown: that real learning steps are map-only (A3 stays NOT RUN); preparation of a row set
from canonical values (the set here is built from an ordinal list); AV2 rehydration, which is
the next PR; joins (AV7/AV11).

OPEN: AV2 (pinned historical snapshot reads allowed, rebinding to another version, index,
scope or facet refused).
