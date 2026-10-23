# 2026-10-23: P-S2 — does a Palette256 compose table survive multi-hop composition? (pre-registration)

**Status:** MEASURED — **UNSTABLE** in both worlds. The pre-registration
above the line was fixed before any probe code was written; results are
below it.

## Question

`bgz_tensor::attention::ComposeTable` stores, for every pair `(a, b)`, the
palette index nearest to `entry[a].xor_bind(entry[b])`. `compose_chain`
composes hops by feeding each looked-up index into the next lookup, so every
hop snaps back onto the palette. XOR bind itself is associative and its own
inverse. The question is how much of that algebra the table keeps once every
hop is quantized.

This is the pairwise-relation follow-up to P-D3 (which tested one needle
against the codebook, not a pair law). The existing
`palette256_multihop_error_probe.rs` tests scalar toy recursions, not this
table.

## Setup (fixed)

- Production code only: `WeightPalette::build(rows, 256)`,
  `ComposeTable::build(&palette)`, `ComposeTable::compose`.
- Two synthetic worlds, deterministic LCG seed:
  - **clustered:** 64 centres in Base17 space, 16 points each with
    noise ±120 (1024 rows);
  - **uniform:** 1024 rows with every dimension uniform in ±1500.
- 2000 random chains per hop count; relation operands drawn uniformly from
  the 256 palette indices.

## Measures

- **M1 agreement(k)**, k = 1..6: share of chains where the table chain equals
  the exact reference `nearest(entry[a0] ⊕ entry[a1] ⊕ … ⊕ entry[ak])`
  (XOR in Base17 space, one snap at the end).
- **M2 unbind recovery:** share of all 65 536 pairs with
  `compose(compose(a, b), b) == a`. Exact XOR always recovers `a`.
- **M3 image collapse:** for each `b`, the number of distinct values of
  `compose(x, b)` over all 256 `x` (median, min); and the size of the
  reachable set after k hops starting from all 256 states. Reference: a
  uniformly random map on 256 states keeps about 162 (256·(1−1/e)).

## Decision rule

- **Harness guard (must hold or the probe is void):** agreement(1) = 1.000,
  because one hop is exactly one bind followed by one snap.
- **Stable** if agreement(2) ≥ 0.90 **and** unbind recovery ≥ 0.90.
- **Unstable** otherwise. Numbers are reported either way.

## Disable runs (planned)

- Replace the table chain with the exact reference: agreement must read
  1.000 at every k (the measure can show stability).
- Replace `compose` with a constant: unbind recovery must fall to about
  1/256 (the measure can show failure).

## Not run, and why (P-L3, P-SH1)

Both were proposed in the 2026-10-22 thread. On synthetic data their outcome
is set by how the world is built: an evidence table gives unseen pairs zero
confidence by construction, and a planted shadow signal stands out exactly
as far as it was planted. Whether real relations are geometric (so a cosine
table carries them) or arbitrary (so only counted evidence does) is a
property of real data. Both wait for a labelled corpus.

---

## Results

Probe: `crates/bgz-tensor/tests/p_s2_compose_chain_probe.rs` (release).

| measure | clustered | uniform |
|---|---|---|
| agreement k = 1 (guard) | 1.000 | 1.000 |
| agreement k = 2 | **0.025** | **0.043** |
| agreement k = 3..6 | 0.013, 0.006, 0.005, 0.004 | 0.021, 0.011, 0.003, 0.005 |
| unbind recovery | **0.071** | **0.169** |
| image size per relation (min / median / max) | 92 / 111 / 129 | 147 / 161 / 256 |
| reachable set after 1..6 hops (of 256) | 118, 70, 53, 43, 29, 26 | 158, 116, 94, 80, 69, 59 |

Verdict: **unstable**. Agreement falls from 1.000 to 0.025 at the second hop,
which is at the level of a random guess among a few dozen states, not slow
drift. The table loses XOR's self-inverse (7–17 % recovery against 100 %
exact). In the uniform world the median image size (161) matches a random map
on 256 states (≈162); the clustered table collapses faster than a random map.

**Mechanism (post-hoc diagnostic, not pre-registered):** P-D3's radius flag
applied to the bind result before it is snapped. 100 % (clustered) and 88.9 %
(uniform) of all 65 536 bound pairs lie outside the palette's support, so the
table answers with a confident `u8` for vectors the codebook does not cover.
Caveat: radii come from 4 training rows per entry, so the flag is strict; the
size of the agreement collapse does not depend on it.

Disable runs:
- Exact chain in place of the table chain: the pinning test fails
  ("the compose table now passes; re-register P-S2"), so the pin can fire.
- Constant law: unbind recovery < 0.01 (`unbind_recovery_collapses_for_a_constant_law`).
- Exact chain against itself reads 1.000 at every k
  (`agreement_reads_one_for_the_exact_chain`).

What this does and does not show:
- It shows that a law derived by binding representatives and snapping does not
  compose. A multi-hop law must keep its output inside the codebook's support,
  which a derived XOR-bind does not do in Base17 space.
- It does not test calibrated laws (`PaletteLut` filled from observed
  transitions). Those are the remaining candidate for pairwise composition and
  need data to fill them.
- Synthetic worlds only; real weight palettes are not measured.
- `compose_chain`'s doc comment claimed equivalence to two-layer stacking; it
  now states the measurement instead.
- CI runs `bgz-tensor` with `--lib` only, so this probe runs locally, like
  P-D3.

Closeout: `STATUS: measured | OUTCOME: ComposeTable chains do not survive two
hops (agreement 0.025/0.043, unbind 0.07/0.17); bind results leave the
codebook support | OPEN: whether a calibrated PaletteLut composes; real
palettes; P-L3/P-SH1 wait for labelled data`
