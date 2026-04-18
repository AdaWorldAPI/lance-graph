# Session Handoff — Priorities, Quick Wins, Agent Scopes

> For next session (Opus 4.7, 1M context, deep thinking).
> All architecture docs at `.claude/knowledge/cognitive-shader-architecture.md`.

## Deep Thinking Effort — the key opportunities

With 1M context + deep thinking, the session can hold BOTH entire
codebases (ladybug-rs + lance-graph + ndarray ~1M LOC combined) AND
the architecture docs in mind simultaneously. Use that for:

1. **Cross-repo type alignment** — see all 4 Fingerprint copies at once
2. **Whole-chain refactors** — rustynum→ndarray migration without forgetting callers
3. **Architectural invariants** — verify the 5-layer stack compiles end-to-end
4. **Era detection** — recognize which decade's assumptions a module carries

Don't burn deep thinking on single-file edits. Burn it on:
- Multi-crate refactors (fingerprint unification)
- Invariant verification (CollapseGate write protocol end-to-end)
- Architectural decisions (where does the cycle_fingerprint live)

---

## Quick Wins (≤1 hour each, P0)

| # | Task | Impact | Blocker? |
|---|---|---|---|
| QW1 | Unify `Fingerprint<256>` — replace `BitpackedVector` in holograph | Kills type duplication | Yes for P1+ |
| QW2 | `impl From` between ndarray Fingerprint ↔ holograph's types | Bridge for existing callers | No |
| QW3 | Port `Container` = `Fingerprint<256>` type alias to contract | BindSpace foothold | Yes for P2+ |
| QW4 | Add `as_u8x64()` to ndarray Fingerprint<N> | Enables multi-lane SIMD path | Yes for L1 |
| QW5 | Add `MergeMode` enum to contract (Xor/Bundle/Superposition) | Completes CollapseGate protocol | Yes for L3 |
| QW6 | Wire `ndarray::simd::*` re-export surface (add Fingerprint, MultiLaneColumn) | Namespace discipline | No but clean |
| QW7 | Rustynum→ndarray sed pass on cognitive `crate::core::rustynum_accel::*` | Unblocks SPO wip modules | Medium |

Do all 7 first. They're independent, small, and unblock everything downstream.

---

## P1 — Foundation Hardening (2-4 hours)

After quick wins, harden the foundation:

**P1.1: Complete rustynum → ndarray migration**
- 124 errors in learning crate (cam_ops.rs dominates)
- Systematic sed + manual fix per file
- Enable modules one at a time behind `wip` flag
- Target: all learning modules compile without wip after migration

**P1.2: CognitiveShader → thinking-engine wire-through**
- `thinking-engine::cognitive_stack` calls `p64-bridge::CognitiveShader`
- `CognitiveShader::cascade()` uses `bgz17::palette_semiring`
- Output: `CausalEdge64` emitted per step
- End-to-end test: text → style pick → shader → cascade → edge

**P1.3: CollapseGate write protocol in contract**
- Extend existing `CollapseGate` enum with `GateDecision` struct
- `MergeMode`: Xor (single target), Bundle (majority), Superposition (keep all)
- Trait method: `fn commit(gate, delta, target) -> Generation`
- Test: overlapping writers resolve correctly

---

## P2 — BindSpace Columns (4-8 hours)

Build the AGI address substrate:

**P2.1: BindSpace column types in contract**
```rust
pub struct BindSpaceColumns {
    pub content: Arc<[Fingerprint<256>]>,
    pub topic: Arc<[Fingerprint<256>]>,
    pub angle: Arc<[Fingerprint<256>]>,
    pub causality: Arc<[CausalEdge64]>,
    pub qualia: Arc<[[f32; 18]]>,
    pub temporal: Arc<[u64]>,
    pub shader: Arc<[u8]>,
    pub cycle: Arc<[Fingerprint<256>]>,  // cycle_fingerprint per row
}
```

**P2.2: Cascade per column implementation**
- Hamming sweep on fingerprint columns (SIMD popcount)
- Range filter on scalar columns (qualia, temporal)
- Intersect bitmaps across dimensions
- Exact step on survivors (~50 records)

**P2.3: ThinkingStyleStrategy planner**
- Read grammar triangle + spectroscopy from L4 input
- Pick one of 36 ThinkingStyles
- Emit cycle_fingerprint per cycle
- Feed into CognitiveShader config

---

## P3 — Shader Stream Loop (8-16 hours)

**P3.1: 5D stream cycle loop**
- Read columns, cascade, intersect, emit edge
- cycle_fingerprint → LanceDB persistence
- Retrieval from LanceDB as RAG input to next cycle

**P3.2: GGUF hydration pipeline**
- Load weights → palette + fingerprints + holographic memory
- Emit CausalEdge64 wiring per layer
- Store in BindSpace columns

**P3.3: Cognitive shader inference loop**
- No matmul. No FP in hot path.
- Per token: 5 cascades, intersect, gate, persist.
- Target: 10ms per token on CPU with cascade.

---

## Agent Scopes (who does what)

| Agent | Primary Scope | P0 Tasks | P1+ Tasks |
|---|---|---|---|
| **container-architect** | BindSpace types | QW3 (Container port), QW4 (as_u8x64) | P2.1 column types |
| **bus-compiler** | CognitiveShader dispatch | QW5 (MergeMode) | P1.2 shader wire-through |
| **palette-engineer** | bgz17 / HHTL-D / codec | QW1 (Fingerprint unify) | P3.2 GGUF hydration |
| **family-codec-smith** | Codec migration | QW7 (rustynum→ndarray sed) | P1.1 learning migration |
| **thought-struct-scribe** | Struct-of-arrays | — | P2.1 column types |
| **perspective-weaver** | Topic/angle dimensions | — | P2.1 (topic, angle cols) |
| **resonance-cartographer** | LanceDB retrieval | — | P3.1 RAG loop |
| **trajectory-cartographer** | CausalEdge64 branching | — | P3.1 causal state cursor |
| **truth-architect** | NARS + CollapseGate | QW5 | P1.3 write protocol |
| **ripple-architect** | End-to-end sensing loop | — | P3.3 full stream |
| **savant-research** | Cross-era provenance | — | Era tagging during migration |
| **contradiction-cartographer** | Detect conflicts | Ongoing | Ongoing |
| **adk-coordinator** | Ensemble dispatch | — | Coordinate P2+ |
| **adk-behavior-monitor** | Anti-pattern detection | Ongoing | Ongoing |
| **integration-lead** | Cross-crate wiring | QW6 (simd re-exports) | P1.2, P2.1 |

**Single-agent tasks** (no coordinator needed): QW1-QW7, P1.3, P2.2
**Multi-agent tasks** (use adk-coordinator): P1.2, P2.1, P3.1, P3.3

---

## Updates Needed on Agents

Most agents already reference `CognitiveShader` (after the Blumenstrauß
rename this session). The updates needed:

### container-architect
- Add awareness: Container = `Fingerprint<256>` type alias at 16K width
- Read-only semantics via `Arc<[u64; 256]>`
- BindSpace column types (7 dimensions, struct-of-arrays)
- cycle_fingerprint is the 8th column (emitted by L4)

### bus-compiler
- CognitiveShader is in `p64-bridge` (already renamed)
- Layer 2 in the 7-layer stack
- Reads: layer_mask + combine + contra + density_target from StyleParams
- Emits: CausalEdge64 stream (one per step)

### thought-struct-scribe
- Struct-of-arrays = BindSpace address dimensions (not records)
- 7 columns: content, topic, angle, causality, qualia, temporal, shader
- Plus cycle_fingerprint emitted by Layer 4

### perspective-weaver
- Topic and Angle are two of the 7 BindSpace dimensions
- Each is `Arc<[Fingerprint<256>]>`
- Independently Hamming-sweepable

### truth-architect
- NARS InferenceType (5 variants) already in contract
- CollapseGate (Flow/Block/Hold) already in ndarray
- New: GateDecision struct with MergeMode (Xor/Bundle/Superposition)

### resonance-cartographer
- LanceDB is Layer 6 (cold persistence)
- Per-cycle thought stream: cycle_fingerprint + CausalEdge64 output
- Retrieval via Hamming sweep on cycle_fingerprint column
- Feeds back as RAG into Layer 4 planner input

---

## Opus 4.7 Context Budget Strategy

With 1M context:

**Always in context (~100K tokens):**
- `.claude/knowledge/cognitive-shader-architecture.md` — the canonical doc
- `docs/INTEGRATION_PLAN_CS.md` — the integration plan
- `docs/HISTORICAL_CONTEXT.md` — era tags for era-aware refactoring
- Current session scratchpad

**Load per task (~50-100K tokens):**
- Agent card(s) for the specific scope
- Relevant crate source (the ONE being modified)
- Its direct callers (1-2 crates)

**Lazy-load when needed (~50K tokens each):**
- Bench results (`docs/bench_*.md`)
- Specific knowledge files (phi-spiral, bf16-hhtl-terrain, etc.)

**Reserve (~200K tokens):**
- Exploration, agent-spawned research, deep thinking scratchpad

Total typical usage: 400-500K tokens. Keep 500K+ in reserve for the
hardest refactors where you need to see everything at once.

---

## Starting Points for Next Session

1. Read `.claude/knowledge/cognitive-shader-architecture.md`
2. Pick 3-4 quick wins from the QW table above
3. Do them in parallel (independent, each ≤1 hour)
4. Then pick P1.1, P1.2, or P1.3 based on what's most blocked
5. Use `adk-coordinator` only for P2+ (multi-agent tasks)
6. Commit + push after each quick win (momentum)
7. PR after P1 (substantive milestone)
