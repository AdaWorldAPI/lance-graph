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

## The Bridge: semantic.rs as Adapter Plate

This is the critical architectural piece. semantic.rs is the **regime change**
— the boundary where dead Neo4j rows become living geometry.

### What Neo4j Gives You

Rows. Literal, flat, dead rows. Jan is a string. KNOWS is a string. Ada is
a string. Properties are JSON blobs. Relationships are foreign keys pretending
to be edges.

### What semantic.rs Does (the Bouncer)

semantic.rs validates the import — checks that variables bind, labels exist,
types resolve. It does exactly what a bouncer should do: confirm the shipment
matches the manifest.

`(a:Person)-[:KNOWS]->(b:Person)` — yes, `a` is bound, `b` is bound, KNOWS
is a valid relationship type, Person has the expected properties. Clean import.
Stamp it.

### The Projection: Literal Becomes Geometry

The semantic analyzer hands off a **resolved AST** — variables with known
types, relationships with known direction, properties with known values.
That resolved structure is exactly what the SpoBuilder needs:

```
Resolved AST:
  a   = Person { name: "Jan" }
  rel = KNOWS  { since: 2024 }
  b   = Person { name: "Ada" }
  direction = Outgoing

         ↓ project into thinking

SpoBuilder::build_edge(
  S: label_fp("Jan"),          // 1024 bytes, ~11% density
  P: label_fp("KNOWS"),        // 1024 bytes, permuted by role
  O: label_fp("Ada"),          // 1024 bytes
  truth: TruthValue(0.9, 0.8)  // from import confidence or default
)
```

The literal becomes geometry:

- **"Jan"** stops being 3 characters and becomes a point in 8,192-dimensional
  Hamming space
- **"KNOWS"** stops being a label on an edge table and becomes a rotation
  operator that transforms the relationship between subject and object
- **"Ada"** stops being a foreign key and becomes a resonance target

### What the Thinking Mesh Can Do That Neo4j Can't

Once it's in BindSpace as fingerprinted SPO:

**Resonance discovery**: "Jan KNOWS Ada" resonates with "Jan LOVES Ada" —
because S and O are identical and KNOWS is Hamming-close to LOVES. Neo4j
treats those as completely separate edges. BindSpace feels the overlap.

**Causal chain discovery**: "Ada" as object of "Jan KNOWS Ada" resonates
with "Ada" as subject of "Ada CREATES music" — because O of the first triple
is Hamming-close to S of the second. That's causality *discovered*, not
declared. Neo4j needs an explicit path query. BindSpace finds the chain
by geometry.

### The Next Bolt: BindSpaceCatalog

The `AcceptAllCatalog` stub gets replaced with a `BindSpaceCatalog` that asks
"does this label have a fingerprint nearby?" instead of "does this string exist
in a list?" — and suddenly the bouncer isn't just checking IDs, it's checking
resonance. But that's the next bolt. The adapter plate is there.

---

## The Full Import Flow

```
Neo4j dump
  → Cypher MATCH/RETURN
  → parser.rs (lance-graph bumper, validates syntax)
  → semantic.rs (lance-graph bouncer, resolves bindings)
  → resolved AST (literal, structured, typed)

     ═══ REGIME CHANGE ═══

  → SpoBuilder (fingerprint S, P, O with role permutation)
  → BindSpace insert (zero-copy into Container)
  → now it resonates, infers, walks causal chains
  → NARS truth propagates through the graph
  → scent prefilter enables O(1)-ish retrieval

     ═══ GROUND TRUTH CHECK ═══

  → DataFusion joins (lance-graph rims)
  → row/column output matches Neo4j's original answer
  → σ-stripe shift detector confirms convergence
```

---

## What We Steal

### From lance-graph → into ladybug-rs (additive)

| Stolen Part | Lines | Why |
|-------------|-------|-----|
| `parser.rs` | ~1,800 | Hardened Cypher parser (nom combinators). Validates syntax before it touches SPO. |
| `ast.rs` | 543 | Pure serde data types — CypherQuery, NodePattern, etc. Clean vocabulary. |
| `error.rs` | 234 | Zero-cost `#[track_caller]` error macros. Strip lance-specific variants. |
| `semantic.rs` | ~1,800 | **The adapter plate.** Resolves bindings, validates types, hands off clean structures to SpoBuilder. |

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

**What gets added** (stolen from lance-graph): parser, AST, error macros,
semantic.rs adapter plate. Layered on top. Nothing touched underneath.

---

## Open Ends

### 1. Parser + Semantic Theft — Packaging
- parser.rs imports `crate::ast::*` and `crate::error::*`
- semantic.rs imports `GraphConfig` — needs rewiring to BindSpace
- When stolen into ladybug-rs, internal paths change
- Strip DataFusion/LanceCore/Arrow error variants
- Decide: `ladybug-rs/src/cypher/` module tree?

### 2. BindSpaceCatalog — The Resonance Bouncer
- AcceptAllCatalog stub → BindSpaceCatalog that checks fingerprint proximity
- "Does this label have a fingerprint nearby?" instead of string lookup
- This turns the bouncer from ID-checker to resonance-detector
- Changes the character of validation: fuzzy match, not exact match

### 3. GQL and NARS Syntax — Additive Parser Arms
- Stolen parser handles Cypher only
- GQL (ISO 39075): ~90% compatible, add `alt()` nom branches
- NARS (`<S --> P>. %f;c%`): mesh-native language, may belong in ladybug-rs

### 4. Result Bridge — Holodeck to Screen
- Mesh results (BindSpace slots, SparseContainers) → human-readable output
- n8n-rs `n8n-arrow` already has RecordBatch ↔ row conversion
- Additive bridge: mesh → RecordBatch → neo4j-rs Row format

### 5. σ-Stripe Shift Detector
- Ground truth comparison between flat chart and hydrated holodeck
- Does the mesh's answer converge with Neo4j's literal answer?
- Statistical convergence check, not just equality
- Additive module, location TBD

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

**Adapter plate** (semantic.rs): the bouncer at the regime change. Validates
the literal, resolves the bindings, hands off clean typed structures. The
boundary where strings stop being strings and start becoming geometry.

**Thinking mesh** (SPO in ladybug-rs): hydrates the holodeck of awareness.
Fingerprints literals into Hamming space. Discovers resonance between triples
that Neo4j treats as separate. Finds causal chains by geometry, not by
explicit query. Smells before thinking. Believes before traversing.
Propagates through algebraic structures. The territory coming alive.

```
Neo4j:    "Jan" is a string in a row
Chart:    "Jan" is column 2, row 47
Bouncer:  "Jan" binds as Person, properties valid, stamp it
Mesh:     "Jan" is a point in 8,192-dimensional Hamming space,
          resonating with every other entity whose fingerprint
          overlaps, connected by rotation operators that encode
          the meaning of relationships, truth-gated by NARS
          confidence, scent-pruned for O(1) retrieval

Same data. Three regimes. The literal, the validated, the alive.
```

Nothing removed. Everything additive. The chart stays boring.
The bouncer stays strict. The mesh stays alive.
The comparison keeps the holodeck honest.
