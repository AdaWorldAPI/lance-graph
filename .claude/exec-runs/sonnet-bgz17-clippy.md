# bgz17 clippy cleanup — Sonnet run

Toolchain: `+1.97.1`. Command: `cargo +1.97.1 clippy --manifest-path crates/bgz17/Cargo.toml --all-targets -- -D warnings`.
Result: **0 errors, 0 warnings** (started at ~30 errors across lib/test/example targets).

## Fixes

| File:line | Lint | Fix |
|---|---|---|
| `src/bridge.rs:270-272` (tests mod) | `unused_imports` | Removed unused `use crate::base17::Base17`, `crate::distance_matrix::SpoDistanceMatrices`, `crate::palette::Palette` (kept `crate::scope::Bgz17Scope`, which is used). |
| `src/bridge.rs:453` | `non_snake_case` | Renamed test fn `test_scent_NOT_metric_safe` → `test_scent_not_metric_safe`. Test-only symbol, no public API impact. |
| `src/container.rs:302` (`pack_spo_crystal`) | `needless_range_loop` | `for w in W_SPO_CRYSTAL_START..=W_SPO_CRYSTAL_END { container[w] = 0 }` → `for word in container.iter_mut().take(END+1).skip(START) { *word = 0 }`. Same zeroing order preserved. |
| `src/container.rs:305` (`pack_spo_crystal`) | `needless_range_loop` | `for i in 0..n { triples[i] ... }` → `for (i, triple) in triples.iter().enumerate().take(n)`; index `i` still used for `w_off` computation into `container`. |
| `src/container.rs:386` (`pack_extended_edges`) | `needless_range_loop` | Same zero-region pattern as above (`W_EXT_EDGES_START..=END`). |
| `src/container.rs:430` (`compute_wide_checksum`) | `needless_range_loop` | `for i in 128..254 { xor ^= container[i] }` → `for word in container.iter().take(254).skip(128) { xor ^= word }`. |
| `src/container.rs:787` (test) | `assertions_on_constants` | `assert!((W_BGZ17_ANNEX_END - W_BGZ17_ANNEX_START + 1) * 8 >= SpoBase17::BYTE_SIZE)` → wrapped in `const _: () = assert!(...)` (compile-time, same guarantee, all operands are const). |
| `src/palette_semiring.rs:130` (`premultiplied_over`) | `needless_range_loop` | `for d in 0..BASE_DIM { acc[d] += ... }` → `for (d, acc_d) in acc.iter_mut().enumerate().take(BASE_DIM) { *acc_d += ... }`. |
| `src/palette_semiring.rs:147,162` (tests) | `needless_range_loop` | Same `dims[d] = f(i,d)` build pattern → `for (d, dim) in dims.iter_mut().enumerate().take(BASE_DIM) { *dim = f(i,d) }`. `d` still used inside the formula via the enumerate index. |
| `src/rabitq_compat.rs:91-97` (`OrthogonalMatrix::rotate`) | `needless_range_loop` (×2, nested) | Outer `for i in 0..d` → `for (i, out_i) in out.iter_mut().enumerate().take(d)`; inner `for j in 0..d` → `for (j, &input_j) in input.iter().enumerate().take(d)`. `i`/`j` still used to index `self.data[i*d+j]`. |
| `src/rabitq_compat.rs:127` | `manual_div_ceil` | `(d + 63) / 64` → `d.div_ceil(64)`. Same value, no behaviour change. |
| `src/palette.rs:707` (test) | `unused_variables` | Inner loop var `j` genuinely unused in loop body → prefixed `_j` per lint's own suggestion. Noted below as a pre-existing weak/vacuous test, left as-is per scope (no test-logic rewrite). |
| `src/palette.rs:592` (test helper `make_patterns`) | `needless_range_loop` | Same `dims[d] = f(i,d)` pattern as above. |
| `src/base17.rs:432` (test) | `identity_op` | `assert_eq!(d_mant, 100 * 1)` → `assert_eq!(d_mant, 100); // mantissa weight is ×1` (kept the intent as a comment instead of the no-op literal). |
| `src/distance_matrix.rs:141` (test helper) | `needless_range_loop` | Same `dims[d]=f(i,d)` pattern. |
| `src/layered.rs:202` (test helper) | `needless_range_loop` | Same pattern. |
| `src/palette_matrix.rs:243` (test helper) | `needless_range_loop` | Same pattern (BASE_DIM variant). |
| `src/prefetch.rs:324` (test helper) | `needless_range_loop` | Same pattern. |
| `src/simd.rs:317` (test helper) | `needless_range_loop` | Same pattern (BASE_DIM variant). |
| `src/tripartite.rs:144` (test helper) | `needless_range_loop` | Same pattern (adds `seed*53` term; unaffected). |
| `src/typed_palette_graph.rs:141` (test helper) | `needless_range_loop` | Same pattern (BASE_DIM variant). |
| `examples/probe_foveated_descent.rs:214` | `needless_range_loop` | `for d in 0..BASE_DIM { dims[d] = f(hp.coarse[c1].dims[d], hp.coarse[c2].dims[d]) }` → `for (d, dim) in dims.iter_mut().enumerate().take(BASE_DIM) { *dim = f(hp.coarse[c1].dims[d], hp.coarse[c2].dims[d]) }`. `d` still used to index the two read-only source arrays. |
| `examples/probe_base17_fold_ceiling.rs:13-17` (doc comment) | `doc_lazy_continuation` | Not in the pre-scoped known list (found live). Two-item `//! -` bullet list ran directly into unindented prose on the next lines with no blank-line separator. Inserted a blank `//!` line after the second bullet to end the list before the paragraph. No code/behaviour change, doc-comment text unchanged otherwise. |

All fixes are index-order-preserving (no kernel arithmetic/bit-pattern/iteration-order change) and touch no public signatures — `PaletteMatrix`, `PaletteCsr`, `Base17`, `batch_palette_distance`, `TypedPaletteGraph`, container pack/unpack APIs are untouched.

## Skipped

None. Every reported lint was fixable within the "fix only what clippy reports, no API change" constraint.

## Real bug exposed?

No functional bug found. One test (`src/palette.rs::test_sigma_band_no_empty`) has an inner loop variable (`j`) that was never used in the loop body — the nested loop just repeats the same `i`-only assertion `(palette.len() - i - 1)` extra times without adding coverage. This is a pre-existing **weak/vacuous-adjacent test structure** (the assertion only depends on `i`, not `j`), not a compile-time bug — flagged here per the falsifiability-rule spirit in CLAUDE.md, but NOT rewritten since the task scope is lint-fix-only and rewriting test semantics was out of bounds for this run.

## Final test result

```
cargo +1.97.1 test --manifest-path crates/bgz17/Cargo.toml
test result: ok. 134 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 17.66s
Doc-tests bgz17: 0 passed; 0 failed
```

(134 > the ~121 baseline mentioned in the task brief — the crate has grown since that count was taken; all pass.)

## fmt

`cargo +1.97.1 fmt --manifest-path crates/bgz17/Cargo.toml` ran clean; reformatted the 15 files touched by the clippy fixes above (whitespace/wrapping only, re-verified clippy stayed 0 errors after fmt).

Disk headroom stayed at 8.5-10 GB free throughout (never approached the 4 GB guard).
