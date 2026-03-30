# Backend Savant Infrastructure for HHTL Routing

## Overview

Three backend lookup modules implemented as pre-computed HHTL caches with
domain-specific RouteAction decisions extracted from the Qwen weight diffs.
These are internal Rust modules called by other crates in the workspace.
They never face the user. They are analogous to database indexes or
pre-computed lookup structures: not trained, not prompted, just looked up.

## Relationship to ThinkingStyle (lance-graph-contract)

The savant infrastructure is the **backend plumbing** behind the user-facing
`ThinkingStyle` enum defined in `lance-graph-contract/src/thinking.rs`.

| Layer | What it is | Analogy |
|-------|-----------|---------|
| `ThinkingStyle` (contract) | User-facing control knob ("think analytically") | SELECT query |
| `CascadeConfig` (planner) | Parameterization derived from the style | Query plan |
| Savant module (bgz-tensor) | Backend infrastructure ("which cache to query for this attention pair") | Index scan |

**How they connect:**

1. The caller selects one of the **36 ThinkingStyles** (e.g., `Analytical`, `Creative`, `Adversarial`).
2. The planner maps that style to a **CascadeConfig** (tactic weights, escalation thresholds, compose depth).
3. The CascadeConfig **parameterizes the savant's route table** — same cache, different decision boundaries.
4. All 36 ThinkingStyles reduce to **3 backend savant modules** with different CascadeConfig parameters.

```text
36 ThinkingStyles ──► 6 clusters ──► 3 savant backends
                                       │
                  CascadeConfig parameterizes each:
                  - escalation_threshold (when to leave Core)
                  - compose_depth (how many hops in specialist)
                  - tactic_weights (which tactics are active)
```

The savant modules know nothing about "thinking styles" or user intent. They
receive a CascadeConfig and an (a, b) attention pair, and return a RouteAction.
All user-facing semantics live in the contract crate and the planner.

## Architecture

```text
Token input
  │
  ▼
Core Savant (10 KB, L1 cache controller, always hot)
  route(a, b) → Skip (60%) | Attend (25%) | Escalate (15%)
  │                                           │
  │ ◄─── done, no specialist needed           ▼
  │                                     Context classifier
  │                                     (scent byte SPO planes)
  │                                           │
  │                              ┌────────────┴────────────┐
  │                              ▼                         ▼
  │                   Psychology Savant           Linguistics Savant
  │                   (behavioral pattern DB)     (grammar parser index)
  │                   route(a, b) → action        route(a, b) → action
  │                              │                         │
  └──────────────────────────────┴─────────────────────────┘
                                 ▼
                          Final attention decision
```

All three modules expose the same `route(a: u16, b: u16) -> RouteAction` interface.
Callers never interact with savants directly — they go through the HHTL cascade
dispatcher, which selects the appropriate backend based on the Core module's
escalation signal and the scent byte classifier.

## Three Savant Backend Modules

### 1. Core Savant (`core_savant.hhtl.bgz`) — L1 Cache Controller

**Role**: Always-on gatekeeper. Every attention pair hits this module first,
analogous to an L1 cache controller that handles the fast path and only escalates
to slower backends on a miss.

**Source**: 9B ∩ 27B GROUNDS layer — heads that shifted at BOTH scales.
**Size**: k=64 HIP cache, ~14 KB
**Always loaded**: resident in memory, first responder for every token.
**Tactics served**: #5 TCP (pruning), #8 CAS (abstraction scaling)

**Extraction**:
```rust
// In ndarray causal_diff.rs:
let grounds_edges: Vec<WeightEdge> = edges_v1.iter()
    .filter(|e| {
        let block = e.block.unwrap_or(u32::MAX);
        scale_invariant_blocks.contains(&block)
    })
    .cloned()
    .collect();
let core_rows: Vec<Base17> = extract_base17_from_edges(&grounds_edges, &bgz7_shards);
let core_cache = HhtlCache::build_hip(&core_rows);  // k=64
core_cache.serialize("palettes/core_savant.hhtl.bgz");
```

**Route semantics**:
- Skip: pair is universally uninteresting (neither scale cares)
- Attend: universal attention (both scales agree this matters)
- Escalate: needs specialist backend (only one scale has signal)

### 2. Psychology Savant (`psychology_savant.hhtl.bgz`) — Behavioral Pattern Recognition Backend

**Role**: Pre-computed lookup table for behavioral attention patterns, analogous
to a personality trait database. Stores which attention pairs correlate with
behavioral signals (tone, structure, self-reflection) so that the cascade can
route them without runtime inference.

**Source**: v1 \ v2 heads — Opus 4.5 behavioral traits that v2 reverted.
These are the heads that encode HOW to think (tone, structure, self-reflection),
not WHAT to compute.
**Size**: k=256 HHTL cache, ~206 KB
**Loaded on escalation**: when Core Savant returns Escalate + context classifier indicates behavioral domain.
**Tactics served**: #7 ASC (adversarial critique), #9 IRS (roleplay), #10 MCP (metacognition), #11 CR (contradiction)

**Extraction**:
```rust
// Heads that v1 changed but v2 reverted = Opus 4.5 behavioral signature
let behavior_edges: Vec<WeightEdge> = edges_v1.iter()
    .filter(|e| {
        let key = (e.block.unwrap_or(0), format!("{:?}", e.projection));
        quality_map.heads.get(&key).map_or(false, |(q, _)| *q == HeadQuality::Reverted)
    })
    .cloned()
    .collect();
let psych_rows = extract_base17_from_edges(&behavior_edges, &bgz7_shards);
let psych_cache = HhtlCache::from_base17_rows(&psych_rows, 256);
psych_cache.serialize("palettes/psychology_savant.hhtl.bgz");
```

**Route semantics**:
- Skip: this attention pair has no behavioral significance
- Attend: behavioral pattern matched (persona trait, emotional tone)
- Compose: multi-step behavioral chain (cause -> emotion -> response)
- Escalate: ambiguous — need full Base17 resolution

### 3. Linguistics Savant (`linguistics_savant.hhtl.bgz`) — Structural/Syntactic Analysis Backend

**Role**: Pre-computed lookup table for structural and syntactic attention patterns,
analogous to a grammar parser index. Stores which attention pairs correlate with
format, syntax, and precision signals so that code/format routing is an O(1) lookup.

**Source**: v2 \ v1 heads — pure Opus 4.6 signal (10K additional samples).
These are the heads that encode FORMAT, SYNTAX, PRECISION.
Plus: shared v1 ∩ v2 heads that are capacity-dependent (27B only).
**Size**: k=256 HHTL cache, ~206 KB
**Loaded on escalation**: when Core Savant returns Escalate + context classifier indicates code/format domain.
**Tactics served**: #2 HTD (decomposition), #4 RCR (reverse causality), #1 RTE (recursive)

**Extraction**:
```rust
// v2-only heads = precision/format signal
// Plus v1∩v2\9B = capacity-dependent reasoning (27B only)
let precision_edges: Vec<WeightEdge> = edges_v2.iter()
    .filter(|e| {
        let key = (e.block.unwrap_or(0), format!("{:?}", e.projection));
        let q = quality_map.heads.get(&key).map(|(q, _)| *q);
        q == Some(HeadQuality::Bad) || q == Some(HeadQuality::Uncertain)
    })
    .cloned()
    .collect();
let ling_rows = extract_base17_from_edges(&precision_edges, &bgz7_shards);
let ling_cache = HhtlCache::from_base17_rows(&ling_rows, 256);
ling_cache.serialize("palettes/linguistics_savant.hhtl.bgz");
```

**Route semantics**:
- Skip: no syntactic/format significance
- Attend: structural pattern (code block, function signature, SPO grammar)
- Compose: multi-hop syntax (nested expressions, causal chains)
- Escalate: ambiguous parse — need full resolution

## Context Classifier (Backend Dispatch)

When the Core module escalates, the scent byte SPO decomposition determines
which specialist backend handles the pair:

```rust
pub fn dispatch_savant(scent: ScentByte) -> SavantKind {
    // S-plane (dims 0-5): subject features → behavioral if persona-like
    // P-plane (dims 6-11): predicate features → linguistic if structural
    // O-plane (dims 12-16): object features → context-dependent

    if scent.s_agrees() && !scent.p_agrees() {
        // Subject resonates but predicate doesn't → behavioral context
        SavantKind::Psychology
    } else if scent.p_agrees() && !scent.s_agrees() {
        // Predicate resonates but subject doesn't → structural/linguistic
        SavantKind::Linguistics
    } else if scent.all_agree() {
        // Full agreement — both backends, merge results
        SavantKind::Both
    } else {
        // O-plane only or nothing — stay with Core
        SavantKind::Core
    }
}
```

## NARS Feedback Loop

Each backend module's route table evolves via NARS truth revision:

```text
Round 0: Routes from static weight-diff extraction
Round N: NARS revision updates truth per (archetype, action)
         High confidence + good outcomes → routes solidify
         Low confidence → Escalate more (admit uncertainty)

NarsHeadBelief tracks:
  core_savant: mostly Reinforce (universal patterns are stable)
  psychology_savant: mixed (behavioral patterns are context-dependent)
  linguistics_savant: mostly Reinforce for code, Explore for natural language
```

## File Layout

```
lance-graph/crates/bgz-tensor/
  palettes/
    qwen-scaffold.pal8              <- 4 KB  (PAL8 topology, committed)
    core_savant.hhtl.bgz            <- 14 KB (k=64 HIP, committed)
    psychology_savant.hhtl.bgz      <- 206 KB (k=256, committed)
    linguistics_savant.hhtl.bgz     <- 206 KB (k=256, committed)
  data/
    *.bgz7                          <- gitignored, hydrate-on-demand
```

## Tactic -> Savant Backend Mapping

| # | Tactic | Primary Backend | Fallback |
|---|--------|----------------|----------|
| 1 | RTE Recursive Expansion | Linguistics | Core |
| 2 | HTD Hierarchical Decomposition | Linguistics | Core |
| 3 | SMAD Multi-Agent Debate | Psychology + Linguistics | — |
| 4 | RCR Reverse Causality | Linguistics | Core |
| 5 | TCP Thought Pruning | Core | — |
| 6 | TR Thought Randomization | Core (noise injection) | — |
| 7 | ASC Adversarial Critique | Psychology | Core |
| 8 | CAS Abstraction Scaling | Core | — |
| 9 | IRS Roleplay Synthesis | Psychology | — |
| 10 | MCP Meta-Cognition | Psychology | Core |
| 11 | CR Contradiction Resolution | Psychology | Linguistics |
| 12 | TCA Temporal Context | Core | — |

## Implementation Order

1. **Core Savant first** — always needed, smallest, validates the pipeline
2. **Linguistics Savant** — v2 data is cleanest (closer to base = less noise)
3. **Psychology Savant** — v1 data is richest (most shifted heads)
4. **Dispatch logic** — scent byte classifier
5. **NARS feedback** — after inference validation
