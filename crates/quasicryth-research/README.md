# quasicryth-research

Direct Rust transcode of the **algebraic core** of Quasicryth (Tacconelli 2026,
[arxiv 2603.14999](https://arxiv.org/abs/2603.14999), upstream
[github.com/robtacconelli/quasicryth](https://github.com/robtacconelli/quasicryth)
v5.6.0).

**Purpose:** research and testing. Validates the workspace's φ-substrate
decisions (bgz17's `17φ/11`, helix's golden-spiral hemisphere) against the
reference algebra without depending on the upstream C build.

## What's transcoded

| Reference file | Rust module | What |
|---|---|---|
| `qtc.h` (types + constants) | `src/types.rs`, `src/constants.rs` | `Tile`, `HLevel`, `ParentMap`, `Hierarchy`, `DeepPositions`, `TilingDesc` + `PHI`, `INV_PHI`, `HIER_WORD_LENS`, the 36-tiling table |
| `fib.c` (tiling generators) | `src/tiling.rs` | Cut-and-project (`qc_word_tiling[_alpha]`), Thue-Morse, Rudin-Shapiro, period-doubling, Period-5, Sanddrift |
| `fib.c` (hierarchy) | `src/hierarchy.rs` | `build_hierarchy`, `hier_context`, `detect_deep_positions`, `deep_counts` |

## What's NOT transcoded

The reference C compressor ships a full pipeline above the algebraic core:
multi-level adaptive arithmetic coding, two-tier unigram model, word-level
LZ77, codebook construction, LZMA escape stream, tokenization, case separation,
header assembly. **None of those are in this crate.** This is the algebra, not
the compressor.

## Verification

Tests cover the five core theorems of the paper:

- **Thm 2** Fibonacci hierarchy never collapses (both L and S supertiles persist).
- **Cor 4** Period-5 collapses by level 4 or 5 (vs Fibonacci's unbounded depth).
- **Thm 9** Golden Compensation: L:S ratio = φ at every level.
- **Thm 13/Cor 15** Aperiodic advantage grows with scale.
- **Sturmian** Factor complexity ≤ n+1 (the minimality property that gives
  maximal codebook efficiency, Thm 7).

Plus algebraic and structural invariants: PV-property (φ² = φ+1),
`HIER_WORD_LENS` = Fibonacci numbers `F_3..F_12`, no-adjacent-S on all 36
canonical tilings.

```
cargo test --manifest-path crates/quasicryth-research/Cargo.toml
```

## Crate policy

Standalone, zero-dependency, `exclude`d from the lance-graph workspace —
same convention as `bgz17`, `deepnsm`, `helix`, `bgz-tensor`. Verified via
`cargo test --manifest-path`.

## Relationship to workspace crates

- **bgz17** — uses 17φ/11 ≈ 5/2 (major tenth) as octave-stacking constant
  for codebook hierarchy depth; this crate verifies the **non-collapse theorem
  that justifies the choice of φ over rational approximations**.
- **helix** — uses pure φ for golden-angle azimuth and √u for equal-area
  hemisphere placement; this crate verifies the **Sturmian minimality** that
  makes φ optimal among irrational slopes.
- **jc::weyl** — proves 1-D `{k·φ⁻¹ mod 1}` star-discrepancy is minimal at
  N=144 and N=1000; this crate's `qc_word_tiling` exercises the same φ-stride
  at hierarchy scale.

## Upstream

`https://github.com/robtacconelli/quasicryth` — v5.6.0 as of the transcode
date. The upstream is the canonical reference; this crate tracks its
algebraic surface only and does not attempt byte-for-byte compatibility
with its compressed output.
