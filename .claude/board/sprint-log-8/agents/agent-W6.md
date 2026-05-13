# Agent W6 — golden-offset

**Date:** 2026-05-13
**Branch:** `claude/lance-graph-1.95-bump`
**Status:** COMPLETE (code fixes applied; clippy verification blocked by pre-existing ndarray build failure)

## Files Modified

1. `crates/bgz-tensor/examples/golden_offset_test.rs` — 4 `manual_div_ceil` fixes
2. `crates/bgz-tensor/examples/calibrate_from_jina.rs` — 1 `map_clone` fix

## Changes Applied

### golden_offset_test.rs — 4 sites, all `manual_div_ceil`

Lines 226, 253, 280, 309 (original lint_inventory.txt numbering):

  (n + BASE_DIM - 1) / BASE_DIM  ->  n.div_ceil(BASE_DIM)

Applied to all four projection functions:
- project_i16_integer_stride (line 226)
- project_i32_integer_stride (line 253)
- project_i32_phi_fractional (line 280)
- project_i32_phi_skip (line 309)

### calibrate_from_jina.rs — 1 site, map_clone

Line 64:
  texts.iter().map(|s| *s).collect()  ->  texts.iter().copied().collect()

## Verification

Attempted clippy on both examples — blocked by pre-existing ndarray build failure:

  error[E0433]: cannot find module or crate `blake3` in this scope
    --> /home/user/ndarray/src/hpc/plane.rs:206:26

Root cause: ndarray/src/hpc/{seal.rs, merkle_tree.rs, plane.rs, vsa.rs} call blake3::
directly but are NOT guarded by #[cfg(feature = "hpc-extras")] in ndarray/src/hpc/mod.rs.
The blake3 crate is optional (hpc-extras feature) but seal and merkle_tree are included
unconditionally. Pre-existing ndarray bug, outside W6 scope.

The code changes are correct per cookbook. n.div_ceil(BASE_DIM) is stable since Rust 1.73.
iter().copied() is the canonical replacement for iter().map(|x| *x).
