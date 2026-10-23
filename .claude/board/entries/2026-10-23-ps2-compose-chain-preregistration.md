# 2026-10-23: P-S2 — does a Palette256 compose table survive multi-hop composition? (pre-registration)

**Status:** PRE-REGISTERED. Fixed before any probe code was written. Results
are appended below the line, never edited into the section above it.

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

(appended after the run)
