# D-DRISA-3: the shipped carry words express threshold, Kleene and Bag exactly, and lose to native words in every shape

**Status:** MEASURED-LAB. Owner: this session (EXO-153 stage 3). Probe: `crates/drisa-register-probe/src/alu.rs`, `tests/integration/alu.rs`, `examples/alu_bench.rs`. Uses only `ndarray::simd::{half_add_mask, full_add_mask, U8x64}`; no `U24x5` or `Abacus` import, no new word, no public type. Follows P3-5 (#1954).

## What was tested

Rows are DR-aligned mask planes (one bit per row, tail past `n_rows` zero, a presence plane for absent rows). Three instruction shapes, each against a per-row scalar oracle that shares nothing with the planes:

- **k-of-n**, two-valued: n ∈ {1, 2, 4, 5, 16, 24}, k ∈ {0, 1, ⌈n/2⌉, n, n+1}; zero, full, alternating, one-hot and random inputs; 100, 4,096 and 65,573 rows (450 cells). The carry arm and a `U8x64` byte-counter arm both equal the oracle on every row.
- **Kleene k-of-n**: an input may be UNKNOWN. TRUE iff the TRUE count reaches k; FALSE iff TRUE + UNKNOWN does not; UNKNOWN otherwise. More than 1,000 UNKNOWN results in the fixture, none turned into FALSE before the terminal.
- **Bag**: each input contributes a 2-bit multiplicity per row; the SUM is compared. The Bag threshold and the support count disagree on at least 4 of 8 k values over the same inputs.

Boundary twins hold exactly: k = 1 is OR, k = n is AND, k = 0 is exactly the present rows (an absent row is never TRUE), k > n is empty; the middle k is neither OR nor AND. A presence plane with stray bits past `n_rows` never reaches a result. 3-of-5 equals its 10-term Boolean expansion.

19/19 tests in the crate (6 ALU tests). **Disable runs** (commit first, asserted anchor, `git checkout`): comparator drops its greater-than term → 5 red; presence ignored → 3 red; Kleene UNKNOWN overlaps TRUE → 1 red; Bag high bit at weight 1 → 1 red; carry kept in the same column → 4 red; tail not cleared → 1 red (only after the dirty-presence test was added: before it, presence already zeroed the tail and the guard was not load-bearing).

## Cost (ns per 64-row word, 65,573 rows, median of 31, 3 processes)

| instruction | carry words | native | carry / native |
|---|---|---|---|
| 2-of-2 vs native AND | 6.93–7.00 | 1.46 | 4.8× |
| 3-of-5 vs 10-term Boolean expansion | 12.69–12.72 | 3.79–3.81 | 3.3–3.4× |
| 3-of-5 vs `U8x64` counters | 12.69–12.72 | 5.68–5.84 | 2.2× |
| 8-of-16 vs `U8x64` | 39.7–40.4 | 13.8 | 2.9× |
| 12-of-24 vs `U8x64` | 55.5–55.8 | 19.7–20.1 | 2.8× |
| Kleene 4-of-8 vs two `U8x64` counters | 39.8–40.4 | 23.0–26.3 | 1.5–1.7× |
| Kleene 12-of-24 vs two `U8x64` | 131.6–132.7 | 76.6–77.6 | 1.7× |
| Bag sum ≥ 8 of 5 × 2-bit vs `U8x64` | 24.8–25.2 | 7.9–8.0 | 3.1–3.2× |

One host with AVX-512 and no VNNI, default x86-64 target; the facade backend selected was not recorded. A branchless ripple (no early exit) was tried and was not faster.

## Reading

The carry words are a correct and complete arithmetic vocabulary for these shapes: threshold, three-valued threshold and weighted sums all lower to them, with the boundary cases exact. They are not a competitive ALU on this host: each call works on one `u64` (64 rows), and byte counters on the same facade do 64 rows per vector add. That cause is HYPOTHESIS; no wide carry word exists to test it.

**Abakus contribution, as measured:** the semantics and the oracle, not the speed. The proposed ruling (memo D-DRISA-M6) is to lower these instructions to byte counters by default and keep the carry formulation as the reference. Not measured: counts above 255, folds that keep the planes for later arithmetic, other ISA tiers.
