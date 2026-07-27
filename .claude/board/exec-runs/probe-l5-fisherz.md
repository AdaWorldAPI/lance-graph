# probe_l5_fisherz_amortization — L5 γ-fold + certified L2 FisherZTable amortization

New file: `crates/bgz-tensor/examples/probe_l5_fisherz_amortization.rs`.
Run: `cargo run --manifest-path crates/bgz-tensor/Cargo.toml --release --example probe_l5_fisherz_amortization`
(no args — defaults to the scratchpad shard; only ever ran with `--manifest-path crates/bgz-tensor/Cargo.toml`, never `--all`).

Real bytes only (Rule 23): rows sourced from a real bge-m3-f16 bgz7 shard
(`/tmp/claude-0/-home-user/bcd29cfc-5bae-5b23-b86b-0de9582a87da/scratchpad/bge-m3-f16.bgz7`),
lenient-parsed (declares 389 tensors, holds 290, exact EOF — matches the
known published-asset truncation the sibling probe already documents).
223,326 usable (non-zero) 17-dim rows recovered. Deterministic sampling via
SplitMix64 seed `0x9E3779B97F4A7C15` (same generator as
`crates/lance-graph-planner/examples/probe_palette256_ndarray.rs`).

## Full printed output (release build, second run — clean, no warnings)

```
declared tensors: 389  parsed: 290
shard: /tmp/claude-0/-home-user/bcd29cfc-5bae-5b23-b86b-0de9582a87da/scratchpad/bge-m3-f16.bgz7
usable rows: 223326

═══ PART A — L5 γ-fold (euler_gamma_fold / euler_gamma_unfold) ═══

N_FOLD_MEMBERS = 6, SPD = 32, dim = 17
fold indices (into `rows`): [91026, 154261, 73762, 97825, 139890, 50603]

build_once (euler_gamma_fold): 33015 ns
correct-index recovery: mean ρ = 0.3786, min ρ = 0.1140  (per-member: ["0.4968", "0.1627", "0.5572", "0.1140", "0.3105", "0.6307"])
wrong-index  recovery: mean ρ = 0.0817  (per-member: ["0.4829", "0.0974", "0.3532", "-0.0422", "0.2672", "-0.6685"])
falsifier margin (correct − wrong) = 0.2970 → FIRES (container addresses members)
bytes: raw 408 B vs folded 2184 B → ratio_vs_raw 0.19×, FoldedFamily::compression_ratio() = 2.99×
per-read unfold cost: mean 1623 ns/member

═══ PART B — certified L2 FisherZTable (fisher_z::FisherZTable) ═══

K_REPS = 256
build_once (FisherZTable::build): 3197797 ns
byte_size() = 65544 B (64.0 KB) [expect k×k + 8 = 65544 B]

certification over 2000 deterministic pairs: Pearson r = 0.9996, Spearman ρ = 0.9998
gate (Spearman ≥ 0.9990, u8-lane): PASS

per-read cost: lookup_i8 = 1.86 ns/lookup (1000000 iters)
float alternative cost: true cosine = 35.57 ns/cosine (10000 iters)
break_even_reads = build_ns / (cosine_ns − lookup_ns) = 3197797 / (35.57 − 1.86) = 94870.8
compare to the certification pass alone (2000 reads): amortizes only AFTER the certification run (more reads needed)

═══ SUMMARY ═══

lane     |   build_once | product_bytes |  per_read_ns |  float_alt_ns |  break_even_reads |  fidelity | falsifier
----------------------------------------------------------------------------------------------------------------
L5       |     33015 ns |       2184 B  |      1623 ns |           N/A |               N/A | ρ=0.3786 | FIRES
L2       |   3197797 ns |      65544 B  |      1.86 ns |      35.57 ns |           94870.8 | ρ=0.9998 | CERT-PASS
```

Note on run-to-run variance: `euler_gamma_fold` build cost dropped from
423,719 ns (cold, first release run right after compile) to 33,015 ns
(warm, second run) — reported the warm/clean number in the summary table
above since it's post-compile steady state; both numbers are real
measurements, not cherry-picked to hide the cold cost.

## 10-line summary (file:line references)

1. `crates/bgz-tensor/examples/probe_l5_fisherz_amortization.rs` — new example, only file added; no existing file modified.
2. Part A uses `euler_gamma_fold`/`euler_gamma_unfold` (`crates/bgz-tensor/src/euler_fold.rs:185`/`:257`) on 6 real 17-dim rows, SPD=32.
3. **L5 measured fidelity is markedly below the doc's ~0.96 anchor**: mean ρ=0.3786, min ρ=0.1140 (per-member 0.11–0.63) — real bge-m3 rows are far less mutually similar than the doc's synthetic 0.90–0.95-cosine test fixtures (`euler_fold.rs:385` `make_similar_vectors`), so SNR-band expectations don't transfer to arbitrary real rows.
4. **Falsifier FIRES**: correct-index mean ρ 0.3786 vs wrong-index mean ρ 0.0817 (margin 0.297, one wrong-index case even goes negative, −0.6685) — the container demonstrably addresses members, it isn't a no-op.
5. **L5 storage is NOT a compression win at these parameters**: raw 6×17×4 = 408 B vs folded 2184 B → 0.19× (folded is *larger*). `FoldedFamily::compression_ratio()` (`euler_fold.rs:172`) reports 2.99× because its denominator uses `StackedN`-encoded member bytes (364 B/member), not raw f32 — both numbers are printed so the discrepancy is visible, not hidden.
6. L5 per-read unfold cost: 1623 ns/member (warm); build_once 33,015 ns (warm) / 423,719 ns (cold, first call after compile).
7. Part B uses `FisherZTable::build` (`crates/bgz-tensor/src/fisher_z.rs:115`) on 256 real representative rows; `byte_size()` = 65,544 B, matches the documented 64 KB + 8 B exactly.
8. **L2 certification PASSES**: Spearman ρ=0.9998, Pearson r=0.9996 over 2000 deterministic real-row cosine pairs, clears the ρ≥0.9990 u8-lane gate used by `examples/nnue_palette_cosine.rs`.
9. L2 per-read cost 1.86 ns/lookup_i8 (1M iters) vs 35.57 ns/true-cosine (10k iters, real f32 rows) → break_even_reads ≈ 94,871 — build cost (3.2 ms) amortizes only well *after* the 2000-pair certification run itself, not within it.
10. No floats stored anywhere except the sanctioned `StackedN` bf16 backing and `FamilyGamma`'s 8 bytes; everything else (pearsons, ns, ratios) is printed only. `--manifest-path crates/bgz-tensor/Cargo.toml` used exclusively; no workspace-wide cargo invocation.
