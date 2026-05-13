# Sprint-log-8 — lance-graph 1.95 bump fleet manifest

**Branch:** `claude/lance-graph-1.95-bump`
**Goal:** fix all 42 clippy lint sites surfaced after bumping rust-toolchain.toml to 1.95.0. The pre-existing janitor sweep in PR #366 cleared most of the surface; this round closes the remaining 1.95-specific lints to make `cargo clippy --workspace --all-targets -- -D warnings` exit 0 on 1.95.0.

**Full error log:** `clippy_1.95_full.log` (80 warnings, no -D) + `clippy_1.95_deny.log` (with -D warnings, 64 error lines).
**Lint inventory:** `lint_inventory.txt` (42 deduped sites).
**Lint categories:** `manual_div_ceil` (6), `needless_range_loop` (5), `manual_range_patterns` (4), `iter_cloned_collect` (2), `unnecessary_sort_by` / `unnecessary_map_or` / `ptr_arg` / `map_clone` / `manual_range_contains` / `manual_checked_ops` / `collapsible_match` (1 each), plus rustc unused_imports/variables.

## Fleet (12 fix workers + 1 meta)

| # | Agent | File(s) | Approx hits |
|---|---|---|---|
| W1 | budget-rotation | `crates/bgz-tensor/examples/budget_rotation_test.rs` | 8 |
| W2 | gguf-euler | `crates/bgz-tensor/examples/gguf_euler_fold.rs` | 7 |
| W3 | gguf-families | `crates/bgz-tensor/examples/gguf_families.rs` | 6 |
| W4 | gguf-thinking | `crates/bgz-tensor/examples/gguf_thinking_styles.rs` | 6 |
| W5 | variance-audit | `crates/bgz-tensor/examples/variance_audit.rs` | 6 |
| W6 | golden-offset | `crates/bgz-tensor/examples/golden_offset_test.rs` + `calibrate_from_jina.rs` | 5 |
| W7 | gamma-phi | `crates/bgz-tensor/examples/gamma_phi_gguf.rs` | 4 |
| W8 | full-pipeline | `crates/bgz-tensor/examples/full_pipeline.rs` + `bgz7_hydration_quality.rs` | 6 |
| W9 | folds-imports | `crates/bgz-tensor/examples/fold_jina_embeddings.rs` + remaining bgz-tensor src | 2+ |
| W10 | contract-holograph | `crates/lance-graph-contract/src/orchestration_mode.rs` + `crates/holograph/src/navigator.rs` + `crates/highheelbgz/{simd_hardened.rs, source.rs, rehydrate.rs}` | ~5 |
| W11 | fmt-sweep | `cargo fmt --all` workspace-wide | — |
| W12 | verify | run `cargo clippy --workspace --all-targets -- -D warnings` + `cargo test --workspace` post-fleet, report final state | — |
| M | meta | synthesize 12 reports, commit + push + open PR | — |

## Permissions

`.claude/settings.local.json` allows `tee -a .claude/board/sprint-log-8/agents/*` for the agents to append their entries. Project-level `.claude/settings.json` already covers most of the workspace.

## Common 1.95 lint cookbook

- `(x + n - 1) / n` → `x.div_ceil(n)` (manual_div_ceil)
- `for i in 0..N { v[i] = ... }` → `for (i, x) in v.iter_mut().enumerate().take(N) { *x = ... }` (needless_range_loop)
- `0 | 1 | 2 | 3` → `0..=3` (manual_range_patterns)
- `.iter().cloned().collect()` → `.to_vec()` (iter_cloned_collect)
- `.iter().map(|x| x.clone())` → `.iter().cloned()` (map_clone)
- `.sort_by(|a, b| b.x.cmp(&a.x))` → `.sort_by_key(|b| Reverse(b.x))` (unnecessary_sort_by)
- `.map_or(true, |x| ...)` → `.is_none_or(|x| ...)` (unnecessary_map_or)
- `&mut Vec<T>` → `&mut [T]` (ptr_arg)
- `n >= 1 && n <= 7` → `(1..=7).contains(&n)` (manual_range_contains)
- `if n > 0 { x / n }` → `x.checked_div(n)` (manual_checked_ops)
- nested `if let` inside outer `match` → match guard (collapsible_match)
