# Reader 2 — Prozentrang / bucket / Shannon doctrine (verbatim, 2026-09-05)

## (a) Prozentrang payload law + "bucket overflow → adjust thresholds"
Stated in `.claude/knowledge/observer-effect-tfpn-doctrine.md` §2 and mirrored in `.claude/plans/cycle-loop-closure-driver-v1.md:1481-1489` (§12.9a.1) and `.claude/board/exec-runs/d-blw-5-design-main-thread.md:71-75` (§c).
- The law: never inject the raw statistic (κ/φ/`BinaryAssociation`) — a scalar is trivially echoable and builds the anchoring/Goodhart fixed point into the instrument. Inject exactly two things: (1) the distribution SHAPE over the prior pool — a palette256/HDR-bucketed census (banded exposure, popcount-stacking early exit, CI thresholds, preheat + rolling-floor bucket, "the Belichtungsmesser reading"); (2) the Prozentrang — percentile rank of the observed association within that prior distribution. `shape₀ × rank₀` is ALL that is injected, frozen at V₀ (single-measurement law, §3).
- Machinery anchors cited (FINDING): `ndarray::hpc::cascade::expose(distance) -> Band` (`cascade.rs:162-175`) + `recalibrate(&mut self, alert: &ShiftAlert)` (`cascade.rs:211`); `ndarray::hpc::statistics::percentile(&self, p)` (`statistics.rs:41`).
- D-BLW-5 realization (`d-blw-5-design-main-thread.md:71-76`): 16-bucket histogram in Fisher-2z space (equal-width in 2z ≈ equal-information buckets), `rank₀` = Prozentrang within the pooled prior.
- "bucket overflow → adjust thresholds" — this exact framing does NOT appear anywhere in doctrine, plan, board, or code. Closest analog: recalibration on `ShiftAlert` and the phrase "preheating + rolling floor bucket".

## (b) Shannon/entropy primitives in code
- `crates/thinking-engine/src/qualia.rs:688` — `fn shannon_entropy(energy: &[f32]) -> f32`, unnormalized; used normalized (`/ n.max(1.0).ln()`) at `:111`, `:159`
- `crates/lance-graph-contract/src/thought_atoms.rs:21` — normalized Shannon entropy of a weight vector in `[0,1]`, `H(p)/ln(n)`; `None` for empty, `Some(0.0)` degenerate
- `crates/lance-graph-cognitive/src/search/distribution.rs:157` — `pub fn entropy(&self) -> f32` — INT4 histogram (bits)
- `crates/lance-graph-cognitive/src/spectroscopy/features.rs:89` — `fn compute_entropy(pops: &[u32; CONTAINER_WORDS]) -> f32` — word-level popcount histogram, normalized by `log2(CONTAINER_WORDS)`
- `crates/lance-graph-planner/examples/entropy_surface_census.rs:238` — `f_shannon_entropy`; forms A–G
- `crates/lance-graph-planner/src/nars/insight.rs:132,178,183,258` — `truth_entropy` (10 bins)
- `crates/lance-graph/src/graph/arigraph/orchestrator.rs:335,383`, `sensorium.rs:29,155`, `contract/src/sensorium.rs:20`, `thinking-engine/src/osint_bridge.rs:25` — truth-entropy field docs/inline
- examples `hdr_audit.rs:302`, `tts_bgz_codebook.rs:212`, `calibrate_roles.rs:468`, `probe_token_bpe_geometry.rs:104`, `learning/src/cam_ops.rs:3038` — local computations
- `ndarray/src/hpc/styles/sdd.rs:2`, `cur.rs:2` — citations only

## (c) Bucket rollover / Prozentrang scale adjust in code or doctrine
No such mechanism exists in code. All hits are negative doc-comments:
- `crates/lance-graph-contract/src/legacy_outliers.rs:27` — "lacking proper bucket rollover → a wide contiguous field with no HHTL"
- `crates/lance-graph-contract/src/identity_quad.rs:46` — "lacking proper bucket rollover — no. Each slot's capacity is the size of…"
- `crates/lance-graph-contract/examples/probe_wordnet_44_activation.rs:100,227,278` — "saturates silently" warning; line 278 discusses "Expanding to", nothing implemented
Doctrine-side adjacent mechanisms only: (i) `cascade::recalibrate`; (ii) the append-only remeasure-guard ledger keyed `(statistic-id, arm, cohort, metric, version)` that ERRORs on a second write to a sealed key.
