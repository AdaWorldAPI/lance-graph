# lance-graph — Progress Tracker

> Tracks progress against the master integration plan plateaus.
> See /home/user/INTEGRATION_PLAN.md for full context.

## Plateau 0: Everything Compiles

- [x] bgz17 crate compiles and tests pass (121 tests)
- [x] lance-graph core compiles (SPO: 30 tests, Cypher: 44 tests, Semirings: 10 tests)
- [ ] One DataFusion integration test fails to compile — needs investigation

## Plateau 1: Schema Migration

- [ ] 1C.4: Update FINAL_STACK.md with crewai-rust + AriGraph awareness — DONE (2026-03-22)
- [ ] 1C.5: Update integration_phases.md — Phase 2 marked DONE (2026-03-22)

## Plateau 2: ladybug-rs Migration

- N/A for lance-graph (lance-graph is consumed, not changed)
- [ ] 2B.3: ladybug-rs wires lance-graph semirings into src/graph/spo/
- [ ] 2B.4: ladybug-rs delegates Cypher execution to lance-graph
- [ ] 2B.5: ladybug-rs uses lance-graph SPO store as backend

## Plateau 3: Full Stack Integration

- [ ] 3A.1: Add bgz17-codec feature flag to Cargo.toml
- [ ] 3A.2: Move bgz17 from workspace exclude to members
- [ ] 3A.3: NdarrayFingerprint::plane_to_base17()
- [ ] 3A.4: parallel_search() dual-path (HHTL + CLAM merge)
- [ ] 3A.5: Wire TruthGate into search results
- [ ] 3D.1: FalkorCompat 3-backend routing
- [ ] 3D.2: Palette 2-hop ranking validation (ρ > 0.9)
- [ ] 3D.3: Performance benchmark

---
*Last updated: 2026-03-22*
