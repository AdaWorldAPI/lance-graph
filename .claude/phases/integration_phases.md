# Integration Phases: bgz17 Full-Stack Wiring

## Phase 0: Prerequisites (DONE)

```
✅ bgz17 crate: 3,743 lines, 13 modules, 61 tests, zero deps
✅ lance-graph codec-research: ZeckBF17, accumulator, diamond, transform
✅ SPO module: truth.rs, merkle.rs, semiring.rs, store.rs, builder.rs
✅ Cascade (hdr.rs): 1,467 lines, self-calibrating, shift detection
✅ Neighborhood: scope.rs, search.rs, clam.rs, storage.rs, zeckf64.rs
✅ MetadataStore: Cypher → DataFusion queries (metadata.rs)
✅ Container layout: 256-word spec documented
✅ ndarray HPC types: Fingerprint, Plane, Seal, Node, Cascade, BF16Truth ported
```

## Phase 1: Session A — blasgraph Storage + Planner ✅ DONE

**Merged:** PR #29 (commit 678e355)

```
[x] CscStorage compiles and roundtrips with CsrStorage
[x] HyperCsrStorage saves >90% memory for sparse test graph
[x] TypedGraph holds per-reltype matrices + label masks
[x] TypedGraph::from_spo_store() bridges existing SPO
[x] blasgraph_planner.rs compiles LogicalOperator::Expand → grb_mxm
[x] Planner + TruthGate: STRONG gate filters weak edges in test
[x] SIMD Hamming (types.rs) AVX-512 + AVX2 + scalar fallback
```

**Prompt:** `.claude/prompts/session_A_v3_blasgraph_csc_planner.md` (COMPLETED)

## Phase 2: Session B — bgz17 Container Annex + Semiring ✅ DONE

**Phase 2 Completion Note:** Verified 2026-04-08: All 7 deliverables implemented, 126 tests passing.

**Container (PR #28, 728 lines, 15 tests):**

```
[x] container.rs: pack_annex / unpack_annex roundtrip
[x] W126 wide checksum: compute_wide_checksum + verify_wide_checksum
[x] Pack/unpack Base17 annex (W112-124)
[x] Pack/unpack palette word (W125)
[x] SPO crystal (W128-143), extended edges (W224-239)
[x] seal_wide_meta, has_bgz17_annex
```

**Deliverables 2-7 (all implemented, 126 tests total):**

```
[x] PaletteSemiring + compose_table (palette_semiring.rs)
[x] PaletteMatrix mxm (palette_matrix.rs)
[x] PaletteCsr::from_scope_with_edges (palette_csr.rs)
[x] Base17 VSA ops: xor_bind, bundle, permute (base17.rs)
[x] SIMD batch_palette_distance: AVX-512/AVX2/scalar (simd.rs)
[x] PaletteResolution::auto_select (palette.rs)
[x] TypedPaletteGraph (typed_palette_graph.rs)
[x] `cd crates/bgz17 && cargo test` — 126 tests passing
```

**Infra for Phase 3 already wired:**

```
[x] bgz17-codec feature flag in lance-graph Cargo.toml (optional dep, default-enabled)
```

**Phase 3 gate items (Session C scope):**

```
[ ] NdarrayFingerprint::plane_to_base17() — NOT STARTED
[ ] parallel_search() dual-path — NOT STARTED
```

**Phase 4 gate items (Session D scope):**

```
[ ] FalkorCompat 3-backend routing — PARTIAL (only blasgraph wired, DataFusion + palette missing)
```

**Prompt:** `.claude/prompts/session_B_v3_bgz17_container_semiring.md` (COMPLETED)
**Agents:** palette-engineer, container-architect

## Phase 3: Session C — Dual-Path Integration

**Gate criteria (all must pass before Phase 4):**

```
[x] bgz17-codec feature flag added to lance-graph Cargo.toml (DONE — optional dep, default-enabled)
[x] bgz17 stays in workspace exclude (standalone by design, path dep works)
[ ] NdarrayFingerprint::plane_to_base17() encodes from flat PLANE (not container)
[ ] build_palette_distance_fn reads W125 palette indices from containers
[ ] ClamTree::build_from_containers works with palette distance
[ ] parallel_search returns (position, distance, TruthValue)
[ ] TruthGate::STRONG filters low-confidence results correctly
[ ] Cascade stage-1 discrimination improves with Base17 at W112 (empirical)
[ ] LFD from palette produces values in 1.0-10.0 range
[ ] `cargo test --features bgz17-codec` passes
```

**Branch:** `feat/ndarray-bgz17-dualpath`
**Prompt:** `.claude/prompts/session_C_v3_ndarray_bgz17_dualpath.md`
**Agents:** palette-engineer, container-architect, ndarray:cascade-architect

## Phase 4: Session D — Reality Check

**Gate criteria (FINAL):**

```
[ ] FalkorCompat::query_datafusion matches FalkorCompat::query_blasgraph
[ ] palette 2-hop ranking agrees with BitVec (ρ > 0.9)
[ ] TruthGate::STRONG correctly filters in all three backends
[ ] Jan→Ada→Max chain traversal works through all backends
[ ] Performance benchmark: palette faster than BitVec for KNN
[ ] Architecture map document produced
[ ] Benchmark document produced with real numbers
[ ] `cargo test --features bgz17-codec` passes
```

**Branch:** `feat/falkordb-retrofit`
**Prompt:** `.claude/prompts/session_D_v3_falkordb_retrofit.md`
**Agents:** all lance-graph agents

## Cross-Repo Phase: ndarray Alignment

**Independent of Phases 1-4, can run in parallel:**

```
[ ] ndarray blackboard updated with bgz17 awareness
[ ] ndarray cascade-architect knows about palette distance
[ ] ndarray cognitive-architect knows about Base17 encoding
[ ] ndarray truth-architect knows about container W4-7 layout
[ ] Integration prompts 04/05 in ndarray updated for bgz17
```

**Repo:** AdaWorldAPI/ndarray (branch: master)
**Agents:** ndarray:cascade-architect, ndarray:cognitive-architect

## Reference Documents

```
.claude/prompts/session_MASTER_map_v3.md          — architecture overview
.claude/knowledge/bgz17_container_mapping.md       — word-by-word container analysis
.claude/agents/integration-lead.md                 — session status + outdated list

crates/bgz17/KNOWLEDGE.md                          — bgz17 architecture
crates/lance-graph-codec-research/KNOWLEDGE.md      — codec research architecture
```
