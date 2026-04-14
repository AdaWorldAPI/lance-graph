# Lance 2 → 4/5 Upgrade Roadmap

> **Status**: planning doc, no migration work started.
> **Current pins** (`crates/lance-graph/Cargo.toml`):
> `lance = "2"`, `lance-linalg = "2"`, `lance-namespace = "2"`, `lance-arrow = "2"`, `lance-index = "2"`, `datafusion = "51"`, `datafusion-common/expr/sql/functions-aggregate = "51"`.
> **Target**: Lance 4.0 (stable) or 5.0-rc.1 (RC).

## Why upgrade at all

Lance 4.0 and 5.0-rc.1 ship features that directly overlap with compression work we've been doing custom in `crates/thinking-engine/examples/tts_rvq_e2e.rs` and `crates/bgz-tensor/`:

- **IVF_RQ index** (first-class) — same algorithm family as our `build_rvq`
- **IVF partitions multi-split** (5.0-rc.1, PR #6423) — adaptive partitioning for skewed distributions (candidate fix for the `text_embedding` cos=0.054 failure; see PR `AdaWorldAPI/lance-graph#177` comment)
- **HNSW-accelerated partition assignment for fp16 vectors** (4.0) — ~100-500× speedup on large-N assignment
- **BF16 support from PyTorch datasets** (5.0-rc.1) — first-class ingest
- **CacheBackend trait + CacheCodec** (5.0-rc.1) — plug slot for `bgz-tensor::HhtlDTensor` as an index cache codec
- **Distributed IVF_RQ segment builds** (5.0-rc.1) — horizontal scale for Qwen3-235B-size models
- **Index segment commit API** (4.0) — atomic multi-segment commits
- **Pre-transposed PQ codebook for SIMD L2** (4.0) — same pattern as our `l2_dist_sq` F32x16 FMA
- **File format 2.3** added in 4.0, 2.1 becomes default in 5.0-rc.1
- **Hamming distance in HNSW** (5.0-rc.1) — consumer for our `bgz17` Hamming semirings

None of these are strict *requirements* for our current stack. They become attractive when we graduate from "f32 GEMM on reconstructed weights" (our current RVQ path) to "HHTL cascade lookup" (our `bgz-tensor::HhtlDTensor` path) or "IVF_RQ storage" (Lance native).

## Blockers

### Primary: DataFusion 51 → 52.1 bump

Lance 4.0 bumps its DataFusion dependency to `52.1.0`. Our `lance-graph-planner` (10,326 LOC, 16 strategies) is tied to DataFusion 51 APIs:

| Area | Files depending on DF 51 |
|---|---|
| Cypher → DataFusion SQL planner | `crates/lance-graph/src/datafusion_planner/` (~6K LOC) |
| CAM-PQ operator | `crates/lance-graph-planner/src/physical/cam_pq_scan.rs` |
| 16 planner strategies | `crates/lance-graph-planner/src/strategy/` |
| TruthPropagating semiring execution | `crates/lance-graph-planner/src/physical/truth_semiring.rs` |
| Rule optimizer / histogram cost / DP join enum | `crates/lance-graph-planner/src/strategy/{rule,histogram,dp_join}_*.rs` |
| MUL assessment + 36 thinking styles | `crates/lance-graph-planner/src/thinking/` (indirect via DF expr types) |

A direct 51 → 52.1 bump will surface breakage in:
- `datafusion::logical_plan` API changes
- `datafusion::physical_plan` operator trait signatures
- `datafusion::sql::unparser` (we use it for Cypher→SQL)
- `datafusion_expr::Expr` variants (52.x dropped several deprecated variants)
- `datafusion-functions-aggregate` signature changes

### Secondary: file format version default

Lance 5.0-rc.1 makes 2.1 the default file format. Any baked dataset we read from Releases (`v0.1.0-bgz-data`, 41 bgz7 files) needs either:
- Pin the reader to 2.0 explicitly
- Re-bake on 2.1

The 41-shard bgz7 archive is well under 1 GB total; re-bake is acceptable cost.

### Tertiary: Java / namespace API cleanup

Lance 4.0 + 5.0-rc.1 both touched namespace APIs. We don't use Java; we use `lance-namespace = "2"` sparingly. Worth auditing but likely a 2-file fix.

## Phased migration plan

### Phase 0 — No-op baseline (this session)

Finish the RVQ reality-check on Qwen3-TTS-0.6B via the passthrough fix in PR #177. Publish codec-token-match ≥ 99% number. Lance version irrelevant.

### Phase 1 — Algorithm evaluation probe (next session, ≤ 1 day)

Deliverable: `crates/thinking-engine/examples/lance_ivf_rq_probe.rs`

- Use Lance 4.0 **as a library dependency ONLY** in one example, pinning the main workspace to Lance 2 / DF 51.
- Build an IVF_RQ index on one tensor (e.g. `model.text_embedding.weight [151936, 2048]`), read it back, measure cos per row and storage size.
- Compare against the hierarchical CLAM and passthrough baselines.
- If the IVF_RQ + multi-split result is ≥ cos 0.95 at < 1:2 storage on that tensor, migration worth pursuing.
- If not, HHTL-D via `bgz-tensor` stays the forward path.

Risk: Lance 4.0 may transitively pull in DF 52.1 even through an example. Workaround: put the example in its own crate outside the workspace (`crates/lance-graph-ivf-rq-probe/` with explicit `workspace = { resolver = "2" }` override).

### Phase 2 — Peripheral crates (~1 week)

Upgrade the crates that don't touch DataFusion-51 planner APIs:

- `lance-graph-contract` — zero deps, no change needed
- `lance-graph-catalog` — catalog providers, lance-only deps → upgradeable first
- `lance-graph-benches` — benchmarks, no planner coupling
- `crates/bgz-tensor` — 0 deps of its own, only needs `lance-arrow` indirectly

At end of phase 2, contract + catalog + benches + bgz-tensor run on Lance 4.x, but core `lance-graph` + `lance-graph-planner` remain on Lance 2 / DF 51. Workspace compiles via dual-version resolution.

### Phase 3 — DataFusion 51 → 52.1 (~2-4 weeks)

This is the hard part.

1. Bump DataFusion version in `lance-graph` and `lance-graph-planner` simultaneously
2. Fix compile errors walk:
   - `datafusion_planner/`: expression unparsing, predicate pushdown, UDF registration
   - `lance-graph-planner/src/strategy/*`: Strategy trait signatures
   - `lance-graph-planner/src/physical/*`: ExecutionPlan trait signatures
   - `lance-graph-planner/src/thinking/*`: Expr type migrations (minimal, mostly pattern matches)
3. Run `cargo test -p lance-graph -p lance-graph-planner` — expect 150+ failures initially, triage into
   - "syntactic rename" (fast)
   - "semantic API change" (need understanding)
   - "truly broken" (needs redesign)
4. Gate merge on green `cargo test --workspace`

### Phase 4 — Adopt new features (~1-2 weeks)

Once everything compiles on Lance 4.0 / DF 52.1:

- Replace our custom `build_rvq` with Lance IVF_RQ index where benchmarks justify it (the Phase 1 probe decides which tensors)
- Wire `CacheBackend` in `bgz-tensor::HhtlDTensor` so HHTL-D encodings plug in as Lance cache codecs
- Enable multi-split for the `text_embedding` path
- Switch BF16 ingest from our custom `bf16_to_f32_batch` loading to Lance's first-class BF16 dataset type (5.0-rc.1 only — defer to phase 5 unless RC is stable)

### Phase 5 — Lance 5.0 stable (when released)

- Bump 4.0 → 5.0 (minor, Lance historically stable across minor bumps)
- Adopt BF16 ingest, io_uring file reader, distributed IVF_RQ segment builds
- File format default → 2.1 (re-bake the 41 bgz7 shards in v0.1.0-bgz-data release)

## Feature priority vs migration cost

| Lance feature | Our problem it solves | Portable without full migration? |
|---|---|---|
| IVF partitions multi-split (5.0) | `text_embedding` cos=0.05 failure | **Yes** — vendor PR #6423 algorithm (~200 LOC target) |
| HNSW fp16 partition assignment (4.0) | Encoder build time at scale | Yes — vendor the kernel |
| IVF_RQ index (4.0) | Replace our custom RVQ | **No** — tightly Lance-coupled |
| CacheBackend + CacheCodec (5.0) | `bgz-tensor::HhtlDTensor` integration | No — needs Lance core |
| Distributed IVF_RQ (5.0) | 235B MoE scale | No — needs Lance core |
| BF16 PyTorch ingest (5.0) | Drop custom `bf16_to_f32_batch` | No — needs Lance core |
| Pre-transposed PQ codebook SIMD (4.0) | Already done in our `l2_dist_sq` | N/A — we did it independently |
| Hamming distance in HNSW (5.0) | Consumer for `bgz17` semirings | No — needs Lance core |
| File format 2.3 (4.0) | Shipping compressed weights to Releases | No — needs Lance core |

## Recommended forward path

**Do not migrate Lance in this codebase yet.** Two cheaper paths capture 80% of the value:

1. **Vendor the algorithms we want** — port PR #6423 (IVF multi-split) and the HNSW fp16 partition assignment kernel into `crates/bgz-tensor/src/` or `crates/thinking-engine/src/`. Pure algorithm, no Lance/DF coupling.

2. **Use Lance 4.x as an out-of-tree library for specific experiments** — isolated probe crates outside the main workspace, to evaluate features at low cost before paying the full migration tax.

A full Lance 2 → 4/5 migration is a ~3-6 week project mostly gated on DataFusion 51 → 52.1. It's the right eventual move but not this session, not next session, probably not this month. Revisit when Lance 5.0 ships stable and we have a concrete feature in phase 4 that demands it.

## Open questions (for next session)

1. **PR #6423 source** — what are the exact multi-split criteria? Density threshold? Fixed fan-out? Read the PR, log the algorithm, decide portability.
2. **Lance 4.0 vs 5.0-rc.1** — if we migrate, which target? 5.0 RC may stabilize before we finish phase 3. Pin-on-signal strategy: watch `v5.0.0` stable release, only go 5.0 stable.
3. **DataFusion 52.1 breakage scope** — estimate by running `cargo check` with DF bumped (dry run on a branch, count errors). Decides phase 3 week estimate.
4. **Lance 5.0 "non-shared centroid vector index builds"** — does this conflict with our `bgz-tensor::SharedPaletteGroup` (26 groups for Qwen3-TTS-1.7B, 5.4 MB overhead)? Needs clarification of semantics.
5. **io_uring file reader (5.0)** — requires Linux ≥ 5.6. Our teleport VM is 4.4.0. Works on real Railway / CI hosts. Cost: none, behind feature flag.

## Cross-references

- `AdaWorldAPI/lance-graph#176` (merged) — AVX-512 F32x16 FMA encoder + AMX TDPBF16PS polyfill baseline
- `AdaWorldAPI/lance-graph#177` (merged) — hierarchical CLAM dispatch (REFUTED for vocab) + F32x16 rms_norm + passthrough fix
- `docs/RVQ_ENCODER_REPLICATION.md` — runnable pipeline for any BF16 safetensors model
- `docs/RVQ_K_LADDER_TUNING.md` — shape→k decision rule (Section 3 claim REFUTED for vocab tensors)
- `docs/RVQ_ALTERNATIVES.md` — codec-family comparison
- `crates/bgz-tensor/BGZ_HHTL_D.md` — 343:1 lookup-grade encoding (the forward path we're aligning to)
- `.claude/prompts/fisher-z-wiring/` — 12-step HhtlDTensor integration plan

https://claude.ai/code/session_01NYGrxVopyszZYgLBxe4hgj
