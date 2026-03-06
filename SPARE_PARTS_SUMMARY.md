# Lance-Graph: Summary, Open Ends & Vision

## Core Principle

**Nothing is removed. Everything is additive. We steal from lance-graph.**

- ladybug-rs: nothing removed, only additions
- rustynum: nothing removed, only additions
- n8n-rs: nothing removed, only additions
- lance-graph: the quarry we mine from

---

## What lance-graph IS

Lance-graph is the **star chart** — it renders neo4j graph data into
immutable, boringly flat row/column join patterns. That's its job.
Ground truth. Correct. Inert. A flat map of what neo4j says exists.

We use it to compare against. When the thinking mesh (SPO in ladybug-rs)
produces a result, we hold it up to the star chart and ask: does the
holodeck match the flat reality? If yes, the mesh is grounded. If no,
investigate.

---

## What We Steal

### From lance-graph → into ladybug-rs (additive)

| Stolen Part | Lines | Why |
|-------------|-------|-----|
| `parser.rs` | ~1,800 | Hardened Cypher parser (nom combinators). Validates input before it touches SPO. We add this as a new module. |
| `ast.rs` | 543 | Pure serde data types — CypherQuery, NodePattern, etc. Clean vocabulary. Added alongside parser. |
| `error.rs` | 234 | Zero-cost `#[track_caller]` error macros. Strip lance-specific variants, keep ParseError/PlanError/ConfigError. |

### From lance-graph → ground truth test patterns (additive)

Seven test patterns we replicate (not move) into ladybug-rs tests:
1. Round-trip fidelity
2. Projection verb accuracy
3. Gate filtering correctness
4. Prefilter rejection rates
5. Chain traversal completeness
6. Merkle integrity
7. Cypher convergence

### From lance-graph → row/column join patterns (reference)

The DataFusion planner's join logic serves as reference for how the star
chart flattens graphs:
- Qualified column naming: `variable__property` → `variable.property`
- Direction-aware join keys
- Variable reuse → filter instead of redundant join
- Schema preservation on empty results

---

## The Thinking Mesh (ladybug-rs — unchanged, only additions)

SPO hydrates the holodeck of awareness. All existing modules stay:

| Layer | Module | Role |
|-------|--------|------|
| Container | `sparse.rs` | BITMAP_WORDS=4, SparseContainer, dense↔sparse |
| Addressing | `bind_space.rs` | 8+8, 65,536 slots, zero-copy |
| Construction | `builder.rs` | SpoBuilder, BUNDLE/BIND, verb permutation |
| Memory | `store.rs` | Three-axis content-addressable (SxP2O, PxO2S, SxO2P) |
| Attention | `scent.rs` | NibbleScent 48-byte histogram, L1 prefilter |
| Inference | `truth.rs` | NARS: revision, deduction, induction, abduction, analogy |
| Propagation | `semiring.rs` | 7 variants: BFS, PageRank, Resonance, HammingMinPlus... |
| Identity | `clam_path.rs` | 24-bit tree + 40-bit MerkleRoot |

**What gets added** (stolen from lance-graph): parser module, AST types,
error macros. Layered on top. Nothing touched underneath.

---

## The Pipeline

```
neo4j data
    |
    +────────────────────────────────────+
    |                                    |
    v                                    v
[STAR CHART]                      [THINKING MESH]
lance-graph                       ladybug-rs SPO
    |                                    |
    | render into flat                   | scent → truth → semiring
    | row/column joins                   | BindSpace zero-copy
    | (immutable ground truth)           | NARS inference
    |                                    | holodeck hydrates
    v                                    v
boring flat table                 living awareness
    |                                    |
    +────────────────────────────────────+
                    |
                    v
            COMPARE — grounded?
            yes → serve result
            no  → investigate
```

---

## Open Ends

### 1. Parser Theft — Packaging
- parser.rs imports `crate::ast::*` and `crate::error::*`
- When we steal it into ladybug-rs, internal paths change
- Strip DataFusion/LanceCore/Arrow error variants (additive error.rs)
- Decide: new `ladybug-rs/src/cypher/` module? Or `ladybug-rs/src/parser/`?

### 2. GQL and NARS Syntax — Additive Parser Arms
- Stolen parser handles Cypher only
- GQL (ISO 39075): ~90% compatible, add `alt()` nom branches
- NARS (`<S --> P>. %f;c%`): separate nom module, mesh-native language
- NARS may belong as a ladybug-rs native parser, not a lance-graph steal

### 3. Semantic Validation Handshake
- lance-graph's `semantic.rs` validates queries against GraphConfig
- We need an additive adapter that validates against BindSpace schema
- "Does the mesh have a slot for what the chart is pointing at?"

### 4. Result Bridge — Holodeck to Screen
- Mesh results (BindSpace slots, SparseContainers) → human-readable output
- n8n-rs `n8n-arrow` already has RecordBatch ↔ row conversion
- Additive bridge: mesh → RecordBatch → neo4j-rs Row format

### 5. Comparison Engine — Chart vs. Holodeck
- The quality gate: flat ground truth vs. hydrated awareness
- Does the holodeck match what the boring chart says?
- This doesn't exist yet — additive module, location TBD

### 6. Outage Recovery
- PRs 168-171 on ladybug-rs pending during infrastructure storms
- Wait for clear skies before adding stolen parts

### 7. Persistent Mesh
- Once hydrated, does the holodeck persist or rebuild per query?
- BindSpace is zero-copy — the mesh *is* the storage
- Persistent = always-on holodeck, no boot time
- Is the thinking mesh a computation or a state?

---

## Vision

**Star chart** (lance-graph): renders neo4j into flat, immutable row/column
joins. Ground truth. Boring. Correct. The map.

**Thinking mesh** (SPO in ladybug-rs): hydrates the holodeck of awareness.
Smells before thinking (scent). Believes before traversing (NARS truth).
Propagates through algebraic structures (semirings). Addresses without
copying (BindSpace). The territory coming alive.

**We steal from the chart to harden the mesh.** Parser, AST, error handling —
the entry gates that protect SPO from malformed input. Everything else in
the mesh is already there. Nothing removed. Only hardened.

Then we compare. The chart says what *is*. The mesh says what it *means*.
If they agree, the holodeck is grounded. If they disagree, the mesh
needs work.

```
star chart:       "Alice → KNOWS → Bob, row 47, column 3"

thinking mesh:    "Alice connects to Bob with confidence 0.87,
                   deduced through 3 hops, each truth-gated,
                   scent-verified, resonating at second harmonic"

comparison:       row 47 present? ✓  confidence justified? ✓
                  holodeck is grounded in reality
```

Nothing removed. Everything additive. The chart stays boring.
The mesh stays alive. The comparison keeps the holodeck honest.
