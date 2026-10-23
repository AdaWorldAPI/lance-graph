# 2026-10-23 — EXP-GF2-ELIM-PREREG — Gaussian elimination over GF(2) with a replayable row-operation witness — pre-registration

**Status:** PRE-REGISTERED. Written before any code. Lab only; no production crate is touched.
**Authorization:** operator, 2026-10-23 ("1+2", answering "pre-register the GF(2) Gaussian elimination probe?").
**Source:** `.arxiv/2026-10_qr256_gaussian_egglog_rotation_falsifier_paths.md`, paths P4 (Four Russians GF(2)) and P9
(reversible row-step sidecar), stage S3. That note keeps REAL_F64, GF2_XOR and BOOL_OR_AND as three separate
contracts, and so does this probe.

## Question

Can Gaussian elimination be an algebra function whose output carries its own witness? Here the witness is the list
of elementary row operations, and an independent checker can replay that list from the input. Second question: does
a blocked (Four Russians) variant give the same answer with fewer row XORs at n = 256?

## Scope

- **Field:** GF(2) only. Rows are bit rows, addition is XOR and multiplication is AND.
- **Not in scope:**
  - REAL_F64 (`ndarray::hpc::lapack` already has float LU/QR/Cholesky; a cold tool, not touched).
  - Exact integer (Bareiss).
  - Any elimination over the Boolean OR/AND semiring. That semiring has no additive inverse, so elimination is not
    defined over it.
- **Sizes:** n = 16, 64 (one `u64` per row), 256 (`[u64; 4]` per row). Square n × n, plus 64 × 256 for the
  non-square case.
- **Where:** new standalone lab crate `crates/gf2-elim-probe` with its own `[workspace]`, std only. The row kernel is
  plain `u64` XOR, with no intrinsics (same as `relcomp::compose`).
- **Licence:** M4RI is GPLv2+. The blocked variant is written clean-room from the published method. No M4RI code is
  read or copied, and M4RI is not used as an oracle in this probe.

## What gets built

1. `eliminate(a) -> Elimination { rref, rank, pivots, schedule }`, scalar.
   - The schedule is a `Vec<RowOp>` with `RowOp::{Swap(i, j), Add { src, dst }}`, where `Add` means
     `row[dst] ^= row[src]`.
   - Both operations are self-inverse.
2. `eliminate_blocked(a, k)`: Four Russians with block width k = 8.
   - For each block it builds a 2^k table of combinations of the k pivot rows, then clears each row in one table
     lookup.
   - Its schedule is expanded to the same `RowOp` vocabulary, so one checker serves both variants.
   - Reported, not gated: the row-XOR count before expansion.
3. `replay(a, schedule) -> matrix` and `check(a, e) -> Result<(), CheckRefusal>`. This is the independent checker. It
   never calls either elimination routine. It:
   - replays the schedule on `a` and compares the result with `e.rref`;
   - checks that `e.rref` is in reduced row echelon form (each pivot is the only 1 in its column, pivot columns
     strictly increase, zero rows come last);
   - checks that `e.rank` equals the number of nonzero rows;
   - replays the schedule in reverse on `e.rref` and compares the result with `a`.
4. `inverse(a) -> Option<Matrix>`. It returns `None` unless the rank is n.

The schedule is a replay witness (P9). It is not an egglog proof and is not claimed to be one. Whether it later
becomes justifications in egglog's proof format is a separate question that this probe does not answer.

## Fixtures (deterministic xorshift seeds, fixed in the test file)

- Random at densities 1/64, 8/64, 32/64 and 63/64, for each n.
- Rank-deficient by construction: a duplicated row; a row equal to the XOR of two others; a zero column; rank
  exactly n − 3.
- The identity, a permutation matrix, the all-zero matrix, and n = 1 and n = 2.
- **The two-path counterexample.** A and B each 2 × 2, where i reaches k through two middle points. OR/AND gives 1 and
  XOR/AND gives 0. This pins that GF(2) arithmetic must never stand in for Boolean composition.

## Pass criteria (each a test)

- **G1, the witness checks.** For every fixture and both variants, `check(a, e)` passes.
- **G2, the variants agree.** Scalar and blocked give identical `rref`, `rank` and `pivots`. The RREF is unique, so
  equality is the test, not row-space containment.
- **G3, an independent rank oracle at n = 16.** Enumerate all 2^16 combinations of rows and count the distinct row
  vectors. That count must equal 2^rank. The oracle shares no code with elimination.
- **G4, the inverse.**
  - For each full-rank fixture, `a · inverse(a) = I`, checked with a naive triple-loop GF(2) multiply that shares no
    code with elimination.
  - Silent twin: the identity, a permutation matrix and random full-rank fixtures must return `Some`.
- **G5, refusals.**
  - Singular, rank-deficient and all-zero inputs return `None` from `inverse`.
  - `check` refuses a schedule with one step dropped, a schedule with one step's indices changed, and an `rref`
    with one bit flipped.
  - Can-it-stay-silent: `check` passes every unmodified fixture (that is G1).
- **G6, no field substitution.** The two-path fixture gives 1 under the Boolean compose oracle and 0 under GF(2). A
  test asserts both, so the two arithmetics cannot be swapped silently.

## Disable runs (commit first, then disable, each must turn its named test red)

- **D1:** remove the RREF-shape check from `check`. The flipped-bit refusal test must go red.
- **D2:** `inverse` returns `Some` regardless of rank. The singular refusal test must go red.
- **D3:** one wrong entry in the blocked combination table. The G2 agreement test must go red.
- **D4:** drop the reverse replay from `check`. The test with changed step indices must still fail through the
  forward replay, and a separate test that corrupts only the reverse path must go red.

## Reported, not gated

The following are reported in ns per elimination (µs for a batch), at n = 16, 64 and 256, scalar against blocked:
- row XOR count;
- schedule length;
- witness bytes;
- `check` time.

A result in milliseconds is a failed gate (CLAUDE.md P0), not a number to report.

## Kill conditions

- Any fixture where replay of the schedule ≠ `rref`, or reverse replay ≠ `a`.
- Scalar and blocked disagree on any fixture.
- `inverse` returns `Some` for a matrix of rank < n, or `None` for a matrix of rank n.
- The blocked variant does fewer XORs only because it skips rows: its schedule then fails G1.

## Not done here

- No wiring into dr-math-binding-probe, egglog-r2il-probe, or any contract type.
- The `Eliminate<Field>` atom in `heritage.rs`'s Atom27 vocabulary stays a candidate.
- Exact-integer and f64 elimination are untouched.
