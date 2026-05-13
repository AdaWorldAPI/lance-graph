# agent-W12 scratchpad — sprint-4-pr-graph.md

**Date:** 2026-05-13
**Worker:** W12 — cross-repo PR sequencing graph

## Session start

Read: SPRINT_LOG.md at `.claude/board/sprint-log-4/SPRINT_LOG.md`
- 11 TD entries + FMA demo anchor (W11)
- 12 workers (W1–W11 + W12), 2 meta agents (M1, M2)
- Branch: `claude/lance-datafusion-integration-gv0BF`

## Dependency analysis

### Critical chain (must be sequential):
W10 (slot widen u16 + bridge-err audit) →
W8 (audit sink — Lance/JSONL) →
W4 (super-domain subcrates: medcare/smb/hubspot/hiro/woa) →
W2 (q2 stub dedup — re-exports) →
W11 (FMA heart-click smoke: 75K OWL → q2 3D render)

### Gate: W3 → W7
W3 (deprecation shim for API drift) must land before W7 consumer release PRs
- W7-PR-A (lance-graph follow-up PR) can land with W10 in Wave 1
- W7-PR-B/C/D (medcare/smb consumer push) gate on W3

### Independent parallel (no critical-chain dependency):
- W5 (SIMD callcenter batch) — lance-graph only, no consumer coupling
- W6 (thinking-engine wire) — lance-graph contract, no cascade
- W9 (family hydration TTL) — lance-graph contract, reverse lookup

### Wave structure:
- Wave 1 (P0 unblockers, Day 0): W10 + W3 + W7-PR-A
- Wave 2 (consumer migration, ≤Day 3): W4 + W7-PR-B/C/D + W8 + W9
- Wave 3 (convergence demo, ≤Day 10): W2 + W11 + W5 + W6

## Repos touched per wave

Wave 1: lance-graph, medcare-rs (shim import), smb-office-rs (shim import)
Wave 2: medcare-rs, smb-office-rs, hubspot-rs, hiro-rs, woa-rs, lance-graph
Wave 3: q2, lance-graph, ndarray, stalwart (FMA demo infra)

## LOC estimates (rough)
- W10: ~200 LOC (u16 widening + BridgeError audit hook)
- W3: ~300 LOC (deprecation shim + #[deprecated] attrs)
- W7-PR-A: ~150 LOC (open follow-up PR, release notes)
- W4: ~800 LOC (5 new subcrates, Cargo.toml, bridge impls)
- W7-PR-B/C/D: ~400 LOC (3 consumer repos wired)
- W8: ~400 LOC (Lance sink + JSONL fallback + audit contract)
- W9: ~200 LOC (TTL hydration, reverse lookup populate)
- W2: ~300 LOC (re-export stubs, dedup)
- W11: ~600 LOC (smoke test harness, FMA OWL load, q2 render call)
- W5: ~350 LOC (ndarray SIMD swap in callcenter batch)
- W6: ~500 LOC (thinking-engine wire to UnifiedBridge)

## Writing spec now...

## Session 2 — 2026-05-13

Read existing sprint-4-pr-graph.md — file is already complete (~6KB, 219 lines).
All 6 required sections present:
1. Dependency graph (Mermaid flowchart with critical chain + parallel paths)
2. Per-repo PR table (17 rows, all workers covered)
3. Merge waves (Wave 1/2/3 with constraints and CI gates)
4. CI matrix (10 rows across 3 waves)
5. Rollback triggers (R1-R6 with conditions and actions)
6. Open coordination questions (Q1-Q3)

No changes needed. Deliverable verified complete.
