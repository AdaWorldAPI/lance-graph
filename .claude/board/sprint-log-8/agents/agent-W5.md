# Agent W5 — variance-audit

**Date:** 2026-05-13
**Branch:** `claude/lance-graph-1.95-bump`
**Status:** COMPLETE

## Task

Fix all `needless_range_loop` lints in `crates/bgz-tensor/examples/variance_audit.rs`.

## Lint Sites Fixed (6 total)

All in `run_synthetic_audit()`, one per role simulation block:

| Line (pre-fix) | Role | Fix Applied |
|---|---|---|
| 173 | Q | `for (d, out) in dims.iter_mut().enumerate()` |
| 182 | K | `for (d, out) in dims.iter_mut().enumerate()` |
| 191 | V | `for (d, out) in dims.iter_mut().enumerate()` |
| 200 | Gate | `for (d, out) in dims.iter_mut().enumerate()` |
| 212 | Up | `for (d, out) in dims.iter_mut().enumerate()` |
| 221 | Down | `for (d, out) in dims.iter_mut().enumerate()` |

Pattern: `for d in 0..17 { dims[d] = expr(i, d); }` -> `for (d, out) in dims.iter_mut().enumerate() { *out = expr(i, d); }`

Each loop body uses `d` only as an arithmetic input to the RHS expression, not to index a second array, so a clean `enumerate()` on `dims.iter_mut()` is correct in all 6 cases.

## Verification

`rustup run 1.95.0 cargo clippy -p bgz-tensor --example variance_audit -- -D warnings` fails due to a pre-existing ndarray/blake3 compile error on this branch (unrelated to variance_audit.rs). Confirmed the failure is identical before and after my changes via `git stash` test. `rustfmt --check` on the file exits 0 (no syntax errors).

## Files Modified

- `crates/bgz-tensor/examples/variance_audit.rs` -- 6 needless_range_loop sites fixed
