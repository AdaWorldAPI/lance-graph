# bgz-tensor clippy sweep (Rust 1.97.1)

Scope: `crates/bgz-tensor/` only, driven via `--manifest-path` (workspace-excluded crate).

## Fixes

| File:line | Lint | Fix |
|---|---|---|
| `src/adaptive_codec.rs:681-682` | `clippy::doc_lazy_continuation` (9 errors, lines 682-690) | Inserted a blank `///` line between the end of the 2-item doc list and the following prose paragraph, so the paragraph is no longer read as an unindented list-item continuation. No wording changed. |

That single blank-line insertion cleared all 9 reported errors (they were one multi-line diagnostic cluster on the same paragraph).

## Skipped / `#[expect]`-ed

None needed — no lints touched arithmetic, quantization, table indexing, or iteration order (the AttentionSemiring / u16 distance table / u8 compose table / HHTL cascade code was already clean on this toolchain).

## Post-fix `cargo fmt`

Ran `cargo +1.97.1 fmt --manifest-path crates/bgz-tensor/Cargo.toml` per instructions. It reformatted two files beyond the one I hand-edited:
- `examples/probe_l5_fisherz_amortization.rs` — pure line-wrapping of long method chains / `println!` argument lists (no logic change).
- `src/matryoshka.rs` — minor formatting (2 lines).

Re-ran clippy after fmt: still 0 errors, 0 warnings.

## Test result

```
test result: ok. 207 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 7.98s
Doc-tests bgz_tensor: 0 passed; 0 failed
```

## Real bugs exposed by lints

None — the crate was already clippy-clean except for the single doc-comment formatting issue above, which was cosmetic (rustdoc rendering), not a code-correctness issue.

## Disk

Free space stayed well above the 4 GB floor throughout (28G → 29G → 8.6G... wait, actual readings: 10G free at start, 8.6G free after fmt — still comfortably above floor). No stop triggered.
