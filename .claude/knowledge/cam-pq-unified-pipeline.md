# CAM-PQ Unified Pipeline — Existing Surface Map (2026-04-20)

> **READ BY:** any agent working on codec wiring, CAM-PQ production,
> DataFusion UDF registration, runbook/orchestration, or research
> DTOs. Before proposing new storage formats, bridge modules, or
> endpoint surfaces, read this so you don't duplicate.

## The Canonical Pipeline (all pieces exist; wiring has gaps)

```
  OrchestrationBridge::route(UnifiedStep)  ← lance-graph-contract::orchestration
                │
                ▼  step_type prefix dispatch (lg./nd./lb./crew./n8n.)
    ┌───────────┴────────────┬─────────────┬─────────────┐
    ▼                        ▼             ▼             ▼
  LanceGraph              Ndarray        Ladybug        (Crew/N8n)
  (queries)               (codec)        (cycles)
    │                        │             │
    ▼                        ▼             ▼
  datafusion_planner      ndarray::       cognitive-
    ↓                     hpc::cam_pq     shader-driver
  register_cam_udfs*      (train, encode, (dispatch, ingest)
  (BINDS UDF SURFACE      decode, ADC)
  — NOT YET CALLED)
    ↓
  cam_pq/udf.rs           cam_pq/storage.rs    cam_pq/ivf.rs       cam_pq/jitson_kernel.rs
  • CamDistanceTables     • cam_vectors_schema • IvfIndex           • cam_pq_cascade_template
  • cam_distance UDF      • cam_codebook_schema• train/assign/probe • cam_pq_full_adc_template
  • cam_heel_distance UDF • build_cam_batch    • merge_partition_   • select_kernel_template
  • register_cam_udfs     • build_codebook_      results            (RUNTIME JIT — feeds
                            batch                                   Cranelift for per-query
                          • extract_cam_        [billion-scale]      scan kernels)
                            fingerprints
                          • extract_codebook
                          • CamStorageStats
                          [D3 of plan: DONE]
```

`*` = integration gap; see below.

## Integration Gaps (targeted TODOs, not rewrites)

| Gap | Where | Fix |
|---|---|---|
| `register_cam_udfs` not called | `datafusion_planner/mod.rs` | Call at DataFusion session creation — one-line addition, unblocks the whole CAM-PQ query path |
| `cam_pq_calibrate` writes CMPQ/CMFP raw | `bgz-tensor/src/bin/cam_pq_calibrate.rs` | Replace raw writer with `build_cam_batch` + `build_codebook_batch` → Lance table. Then the DataFusion UDF reads calibration output directly |
| `planner_bridge.rs` duplicates OrchestrationBridge | `cognitive-shader-driver/src/planner_bridge.rs` | Retire; shader-driver holds `Box<dyn OrchestrationBridge>` and routes via step_type |
| Per-op Wire DTOs (Plan/Calibrate/Probe/Tensors) | `cognitive-shader-driver/src/wire.rs` | Collapse to `WireStep { step_id, step_type, args }` + `BridgeSlot` for rich payload |
| `nd.*` step-type not implemented | new OrchestrationBridge impl | Implement `OrchestrationBridge` for a codec-research driver that owns `StepDomain::Ndarray` |

## What's already DONE (don't re-derive)

- **ndarray::hpc::cam_pq** — production codec (train_geometric / train_semantic / train_hybrid, CamCodebook, CamFingerprint, DistanceTables, PackedDatabase stroke cascade). 15+ tests.
- **lance-graph-contract::cam** — `CodecRoute::{CamPq,Passthrough,Skip}`, `route_tensor(name, dims)`, `CamCodecContract` trait, `DistanceTableProvider`, `IvfContract`. 10 route tests.
- **lance-graph-contract::orchestration** — `OrchestrationBridge` trait, `UnifiedStep`, `StepDomain` enum, `BridgeSlot` trait. This is THE canonical dedup trait.
- **lance-graph/src/cam_pq/storage.rs** — Arrow/Lance schema for CAM fingerprints and codebooks. 5 tests covering build + extract round-trip.
- **lance-graph/src/cam_pq/udf.rs** — DataFusion `cam_distance` and `cam_heel_distance` UDFs with AVX-512 batch paths. `register_cam_udfs` ready.
- **lance-graph/src/cam_pq/ivf.rs** — billion-scale coarse partitioning (IvfIndex, merge_partition_results).
- **lance-graph/src/cam_pq/jitson_kernel.rs** — JITSON templates for stroke-cascade and full-ADC scan kernels. Feeds ndarray's Cranelift JIT for runtime codec calibration.
- **lance-graph-planner::physical::cam_pq_scan** — DataFusion `CamPqScanOp`. Already shipped.
- **lance-graph-planner impl OrchestrationBridge for PlannerAwareness** — `lg.plan_auto`, `lg.orchestrate`, `lg.health` step-types. 5 tests. **NEW this session.**
- **cognitive-shader-driver Wire DTOs + serve.rs** — `/v1/shader/{tensors,calibrate,probe,dispatch,ingest,runbook,plan}` + gRPC `Tensors/Calibrate/Probe` RPCs. 46 lib tests. **NEW this session.**

## What was MEASURED (doesn't extrapolate)

**PR #218 bench ICC 0.9998 at 128 rows was a trivial in-training fit.**
Full-size validation on Qwen3-TTS-0.6B (234 CamPq tensors): mean ICC
0.195, zero tensors meeting the 0.99 gate. See
`.claude/board/EPIPHANIES.md` 2026-04-20 CORRECTION entry and
`crates/bgz-tensor/examples/cam_pq_row_count_probe.rs`.

**Implication:** 6×256 PQ is centroid-starved for production transformer
tensors. Fix options: wider codebook (1024+ centroids), residual PQ,
Hadamard pre-rotation, OPQ rotation. All can be driven through the
DTO surface once the calibrate bin writes to the Lance schema — no
binary-format migrations needed to test wider codebooks.

## Integration sequence the next session should follow

1. **Wire `register_cam_udfs`** into `datafusion_planner/mod.rs`
   (one line; unblocks SQL/Cypher access to `cam_distance`).
2. **Migrate `cam_pq_calibrate`** to write via `build_codebook_batch`
   → Lance table (`cam_codebook` + `vectors`). Retire CMPQ/CMFP.
3. **Implement `OrchestrationBridge` for a codec-research driver**
   owning `StepDomain::Ndarray` — `nd.calibrate`, `nd.probe`,
   `nd.tensors` step-types.
4. **Retire `planner_bridge.rs`** — shader-driver holds
   `Box<dyn OrchestrationBridge>`, routes `UnifiedStep` by step_type.
5. **Collapse per-op Wire DTOs** to `WireStep { step_type, args }`
   + BridgeSlot results.

Total effort: ~1-2 days. Each step is additive; each is verifiable in
isolation.

## Cross-refs

- `docs/INTEGRATION_PLAN_CS.md` — 5-layer stack (Planner/Gate/Shader/BindSpace/SIMD).
- `docs/CONSUMER_WIRING_INSTRUCTIONS.md` line 207 — `Box<dyn PlannerContract>` pattern.
- `docs/SESSION_HANDOFF_PRIORITIES.md` — P2 column types, P3 GGUF hydration.
- `.claude/plans/cam-pq-production-wiring-v1.md` — D1-D7 deliverables; D1/D3/D4 now mostly DONE (needed inventory to realize).
- `.claude/board/EPIPHANIES.md` 2026-04-20 — CAM-PQ 128-row artifact correction.
