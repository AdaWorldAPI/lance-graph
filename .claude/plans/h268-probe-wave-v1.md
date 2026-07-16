# H.268 Probe Wave v1 — WH-MAG · SIG-CHECKSUM · WALK-SPECTRUM

> Date: 2026-07-16. Status: ACTIVE (this session, autoattended).
> Parent arc: the H.268 documentation arc (ndarray #242-#245, lance-graph
> #695-#701, all MERGED). This wave converts three of its cheapest
> conjectures into measured verdicts. Everything runs CPU-only against
> shipped code; the heavier gates (PROBE-GPU-LUT wgpu harness, OGAR
> PHASE-1/PERT-RHO/PYR-1, A8) are explicitly OUT of this wave.
> Execution model per operator directive: Sonnet drafters (grindwork,
> edit-only + targeted own-crate `cargo test`), Opus filigree reviewer
> (adjudication vs pass/kill), main thread = spec, central gates,
> commits, PR, autonomous resolve + merge after clean CI/reviews.

## P1 — PROBE-WH-MAG (grades E-WH-TWO-SIDES-SIG-CHECKSUM-1 leg 1, [H])

**Question:** does Hadamard-rotating a 4×4 tile's 16-cell magnitude
vector before i4+i2 quantization reduce reconstruction error vs direct
i4+i2, as bgz-tensor's row-level result suggests?

- **Home:** `crates/bgz-tensor/src/adaptive_codec.rs` — new `#[cfg(test)]
  mod probe_wh_mag` (same crate so the private `hadamard_rotate` and the
  `quantize_f32_to_i4/i2` + dequant helpers are reachable; follow the
  file's existing test style).
- **Method:** deterministic synthetic tile fields (seeded SplitMix64-style
  LCG, NO `rand` dep), 16 cells each, three classes × ≥256 tiles:
  (a) smooth gradient + rare spike (elevation-like: linear ramp with one
  cell perturbed 8-32× the ramp step); (b) heavy-tailed spiky (2-3 outlier
  cells at 10-100× the base scale); (c) uniform noise (control — WH should
  neither help nor hurt much). For each tile: path A = i4-quantize →
  dequant → residual → i2-quantize → dequant (the shipped cascade, direct);
  path B = `hadamard_rotate`(16) first, same cascade, inverse-rotate after
  (WH is self-inverse up to 1/16 scale — get the scaling right; verify
  round-trip on an unquantized tile first as a self-check assertion).
  Metric: per-class mean squared reconstruction error ratio B/A.
- **PASS:** B/A < 0.9 on class (a) or (b) (WH materially helps where
  outliers exist). **NEUTRAL:** 0.9 ≤ B/A ≤ 1.1 everywhere (record; the
  conjecture stays open, tile-transfer unproven). **KILL:** B/A > 1.1 on
  ALL classes (WH hurts at 16-cell granularity; the row-level win does
  not transfer).
- Print the three ratios in the test output (`--nocapture`-friendly
  `eprintln!`) AND assert the self-check; the verdict assertion itself is
  informational (assert only the round-trip identity, not the pass/kill —
  the Opus reviewer adjudicates the printed numbers so a NEUTRAL result
  does not red the suite).

## P2 — PROBE-SIG-CHECKSUM (grades leg 2, [S])

**Question:** does the depth-2 truncated signature work as a replayable
trajectory digest — replay-identical, tree-like edits invisible, non-tree
perturbations caught?

- **Home:** `crates/jc/src/sig_checksum.rs`, NEW module behind the
  existing `hambly-lyons` feature (mirror `hambly_lyons.rs`'s structure,
  thresholds, and `sigker::signature_truncated` usage — READ that file
  first; reuse its `signature_distance` idea, its DIM/DEPTH constants
  style, and register nothing in the pillar `prove` list — this is a
  probe module, not a 12th pillar).
- **Method:** a deterministic 2-D "trajectory" path of ~64 points (seeded
  integer walk mapped to f64). Three tests:
  (a) `replay_identity` — same path built twice → byte-identical
  signature levels (exact f64 equality is expected — same ops, same
  order; if that proves flaky use distance < 1e-12);
  (b) `tree_like_edit_invisible` — insert an A→B→A excursion mid-path →
  signature distance to the original < the module's zero-threshold
  (reuse hambly_lyons.rs's forward-leg tolerance);
  (c) `non_tree_edit_caught` — displace one interior point off the path
  (a triangle-style detour that does NOT backtrack) → distance > the
  converse-leg threshold.
- **PASS:** all three green. **KILL:** (b) or (c) fails at the module's
  own thresholds (then the digest idea needs depth > 2 or is wrong).
- Wire `pub mod sig_checksum;` into `crates/jc/src/lib.rs` under the same
  feature gate style as `hambly_lyons`.

## P3 — PROBE-WALK-SPECTRUM (grades §10(g) anti-confabulation, [H])

**Question:** is the stride-4-mod-17 walk's dependence structure actually
"known and concentrated" (period-17 harmonics only, no energy at
palette-lattice periods) vs a PRNG baseline — the measured basis the
CodeRabbit round demanded for the I-NOISE-FLOOR-JIRAK-friendliness claim?

- **Home:** `crates/helix/src/walk_spectrum.rs`, NEW module (test-only is
  fine: `#![cfg(test)]`-style or a `#[cfg(test)] mod` in a thin file),
  using `constants::{MODULUS, STRIDE}`.
- **Method:** build the ±1 sign sequence s[k] = ±1 from the walk (e.g.
  parity of the residue `(start + STRIDE·k) mod MODULUS`, N = 4096 =
  16 × 256 full periods) and a seeded LCG ±1 baseline of equal length.
  Compute plain circular autocorrelation R(τ) for τ ∈ 1..=512 (O(N·τmax)
  is fine at this size; NO fft dep). Report: (i) the walk's |R| at
  multiples of 17 (expected ≈ 1, the known structure); (ii) the walk's
  |R| at τ ∈ {256, 128, 64, 32, 16} (the palette-lattice periods; note
  gcd(17, 256) = 1 so expected ≈ noise floor); (iii) the LCG baseline's
  max |R| over the same τ set.
- **PASS:** walk lattice-period |R| ≤ 2× the LCG baseline's max |R| AND
  walk period-17 |R| ≥ 0.9 (dependence is exactly where predicted,
  nowhere else). **KILL:** walk |R| at any lattice period > 3× baseline
  (the walk aliases against the palette lattice — §10(g) dies as stated).
- Same output discipline as P1: print the table, assert only structural
  sanity (sequence length, ±1 values), reviewer adjudicates.

## Iron rules for this wave

- Sonnet drafters: edit-only + AT MOST their own targeted
  `cargo test --manifest-path crates/<crate>/Cargo.toml [--features ...]`
  (bgz-tensor / jc+`--features hambly-lyons` / helix). NO workspace
  builds, NO worktrees, NO commits, NO new dependencies (no `rand`, no
  `rustfft` — seeded integer generators and O(N·τ) loops only).
- No model identifier in file content. `// SAFETY:` n/a (no unsafe —
  unsafe is BANNED in this wave; if a drafter thinks it needs unsafe,
  STOP and report).
- Probes must be deterministic (fixed seeds) and leave the existing test
  suites green.
- Main thread runs the central gates: targeted `cargo test` for the three
  crates + `cargo fmt --check`-equivalent on touched files.
- Opus filigree reviewer adjudicates printed numbers vs PASS/NEUTRAL/KILL
  above; verdicts land in EPIPHANIES (prepend) + this plan file's results
  section + the ndarray matrix-doc §5 probe queue (companion PR).
- Autonomous merge authority (operator, 2026-07-16 this goal): after CI
  green + external review comments resolved, the PR is merged by the
  session (squash/merge per repo default), then the wave continues.

## Results (filled post-run)

- PROBE-WH-MAG: _pending_
- PROBE-SIG-CHECKSUM: _pending_
- PROBE-WALK-SPECTRUM: _pending_
