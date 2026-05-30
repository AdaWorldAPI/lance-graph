# Cognitive RISC — Class Layer (v0.2)

> Extends `cognitive-risc-core.md` (v0.1). The core holds the substrate invariants;
> this holds the class/shape layer that sits above the SoA.
> Paste BOTH for a full session boot. v0.1 = substrate; v0.2 = how shapes/labels/views inherit.
> Same rule as v0.1: invariants change only with a stated reason. Diagrams illustrate; specs are canonical. When they disagree, the spec wins.

## What changed since v0.1 (the thread's convergence, compressed)

1. **CAM, not ANN.** The substrate addresses business logic by **deterministic content hash** over canonicalized symbolic atoms — exact identity, zero-float. Similarity (FAISS/embeddings/Aerial+) lives ONLY in the proposer/discovery layer, never in addressing. *Same triples, two indexes, never swapped: hash folds identity, similarity folds shape-family.*
2. **The triangle.** `SoA grid <-> Ontology(OGIT) <-> inherited Class` — mutually defining, not a stack. Class shapes the grid; ontology labels the class; grid resolves up through the meta-DTO.
3. **Classes are the missing layer.** The ontology cache is ALREADY REAL and fast (O(1)). What's missing is a class system to *inherit from* it instead of hand-feeding it. Without classes you ARE the inheritance mechanism, by hand, per field, per domain.
4. **One `class_id` keys three things:** label inheritance (via cache + class chain), column projection (the SoA grid), and jinja templates (the view). Three payoffs, one discriminator.
5. **Jinja = classes + presence bitmask.** Template is per-class (~40, not 20k). Instance = `class_id` + presence bitmask. Render = class template, off-bits skipped.

## The CAM addressing invariant (sharpens v0.1 #1 and #2)

> The substrate addresses business logic by deterministic content hash (BLAKE3-128) over **canonicalized** symbolic atoms (CAM, exact-match, zero-float). Similarity lives only in the proposer layer, never in addressing.

Sub-rules (all non-optional once the LE/WAL format hardens):
- **Canonicalize before hashing.** `depends_on[]` / `emits[]` must be sorted (or otherwise canonically ordered) before entering the hash. BLAKE3 over non-canonical array order gives different identities for the SAME logic shape -> CAM miss on identical rules -> defeats the dedup the whole design exists for.
- **Hash is identity, so a collision is silent logic conflation, not a perf bug.** Hash128 birthday bound ~2^64 (never hit in practice). For GoBD-relevant / `BOOKING_RELEVANT_IMMUTABLE` logic: verify full key on CAM hit. Elsewhere: consciously accept the astronomical risk. State which per use.
- **Two indexes, never swapped:** hash = exact identity (addressing, recipe selection, audit). similarity = shape-family (inheritance proposal, discovery). FAISS never picks a recipe; the hash never guesses a family.

## The class layer (the bounded fix — NOT a cathedral)

### The triangle, made mechanical
```
            ONTOLOGY (OGIT)  — class/label registry, inheritance graph, meaning
             ╱                          ╲
        labels class                 resolved through (meta-DTO, at projection)
           ╱                              ╲
   CLASS  ───── shape-compiles into ─────  SoA GRID
   inherited shape    a custom per-class    flattened instance over a FIXED
   (struct-like)      grid from fixed ISA   universal column ISA + ragged backing
```
- **Class = shape + inheritance ONLY.** Struct-like (Rust `struct`, not Java class). A subclass inherits *columns/nesting/labels*, never *methods*. Behavior composes via separately-addressed recipes (Rust `trait`-like). **Shape inherits; behavior composes.** This is the line that keeps OGIT a shape-registry instead of a live object graph (= Foundry/CISC).
- **Universal column ISA is frozen; per-class grids are compiled VIEWS over it.** Inheritance *selects and orders* columns from the fixed vocabulary; it NEVER mints new column types per domain. "Custom build around the mess" = the shape-compiler emitting a per-class grid from the fixed ISA, driven by OGIT inheritance. Generation, not fragmentation.
- **The meta-DTO resolves; it does not store.** Given a raw domain object, the DTO knows its class, walks inheritance to get the full column/label set, flattens to SoA. It holds resolution LOGIC, never resolved STATE — no second source of truth. Identity early-bound (hash), meaning late-bound (OGIT), DTO holds neither.

### Class taxonomy is DISCOVERED, not hand-assigned
- 20,000 Odoo entities are NOT 20,000 shapes — they are instances of ~dozens of shape-families (every model with `_compute_*` over field-deps emitting booking lines is the *same class*).
- Group the 20k by **structural signature** (which fields, compute-method shape, depends_on/emits pattern) — computable, via group-by-on-structural-hash or Aerial+. The shapes sharing a signature *are* a class; name the groups, don't hand-assign them.
- This avoids trading 20k label-hand-rolls for 20k by-hand class-assignments. Even class discovery stays agnostic-extraction.

### Jinja = classes + presence bitmask
- Template is **per-class** (~40, authored once), bound to the class, not the entity.
- Instance = `class_id` + **presence bitmask** (one bit per class field, set if populated). The bitmask is the instance's *delta from its class*, as pure presence bits — not repeated structure, not nullable columns.
- Render = class template, fields gated by bitmask, **off-bits skipped**.
- **Bitmask is presence, NEVER semantics.** A bit means "field N is populated here" (structural). It must never mean "field N behaves/means differently here." Render skips off-bits, full stop — no conditional semantics keyed on the mask. Violating this is what turns "jinja = classes" back into per-instance branching.
- Mask width is bounded by the *class's* field count (dozens of bits), not the 20k-entity union. The class shrinks the variation-space; the bitmask spells the residual within a class. No class -> mask spans the universe (the union disease). With class -> mask spans one shape.

## KNOWN DEBT (state plainly so no future session re-pretends)

**The current single-SoA is a union pretending to be uniform.** Serving all domains from one SoA without classes means every domain-specific column is nullable-for-the-others: a sparse wide table in a uniform costume. The fragmentation already happened — it's just spelled as nullable columns and as *you hand-rolling the inheritance in your head*. The agnosticism is currently a property of intentions, not of the system; the system is domain-specific in N places (proposers/shader branching on domain) and you pay to pretend otherwise. Classes make agnosticism STRUCTURAL instead of aspirational.

The fix is bounded (a weekend, not a subsystem): **discriminator + parent-pointer + parent-walking resolution against the existing cache.** Full machinery (shape-compiler-to-grid, behavior/traits, SIMD kernels) is explicitly DEFERRED.

## NON-DEFERRABLE freeze-time moves (cheap now, ruinous after WAL hardens)

These are the "reserve room for the truth you're pretending away, before the bytes harden" set:

- **N1 — `class_id` / `shape_id` column in the SoA, before freeze.** The key that labels + columns + templates all hang off. Without it the SoA is a blind union and the cache can't inherit. Add it even before full inheritance exists — it's the hook.
- **N2 — proposer-id width (from v0.1).** 3 bits caps at 8; 6 already named; proposers proliferate ("business logic is just another proposer" + dual-IPC). Widen (steal reserved -> 6 bits/64, or u16) before WAL freeze.
- **N3 — stable bitmask bit-positions per class, append-only.** A class's field set IS its mask layout. Bit position N must be stable once instances persist; fields are append-only, retired bits never reused/reordered, or old masks misread. Same ISA-stability discipline as the LE contract, scoped per class.
- **N4 — don't freeze the SoA schema until >=2 genuinely different domains have run through it.** Right now it's Odoo's schema cosplaying as universal. See bring-up test.

## Open forks (load-bearing, deferrable — updated from v0.1)

- **F1 (REWRITTEN) — binding time is NOT either/or.** Resolution: **identity frozen at hash (early-bound), meaning resolved through OGIT-DTO (late-bound).** Same frozen hash can resolve to different recipes/emits as OGIT evolves. Not "askama vs minijinja" — that's F3. F1 is settled: frozen identity UNDER live resolution.
- **F2 — SurrealDB integration.** Read Lance storage directly (heavy, fragile vs both release cadences) OR federate via shared DataFusion catalog (Arrow TableProviders). Default lean: federate.
- **F3 — askama vs minijinja for the class templates.** Downstream of F1's resolution. Since meaning is late-bound through OGIT but identity/shape is frozen, templates likely compile per-class (askama) with OGIT resolving the late-bound labels at render. Detail, but load-bearing once class templates are built.
- **F4 — universal column ISA design.** The thing the whole triangle rests on: what are the *domain-neutral* universal columns every per-class grid is a projection of? If chess needs a column Odoo's ISA didn't anticipate, the ISA isn't universal yet. Get it generous + orthogonal before freeze (per-class grids select from it, never extend it without migration). THIS is the load-bearing unsolved design piece, not the triangle (which is sound).

## SIMD execution mode (the vector unit — folds the batch-processing idea)

Scalar RISC = ops over single thoughts. The cognitive-shader-driver batches the same op across a **grid** of SoA rows, streamed like a lancedb batch writer / Arrow record-batch operator. If the SoA is Arrow ArrayData, "batch over the grid shape" is what DataFusion/Lance do natively — you inherit the stream processor by making the SoA a record-batch. Substrate unchanged; this is its vectorized execution mode, not a new architecture.

## The bring-up test (now ALSO the schema-falsification test)

**Chess into OWL.** Encode openings/methods/verbs as OWL/ttl — meaning in the CONTENT, never a chess-special SoA field, ever. Run proposers (Aerial+ as proposer-not-oracle) over it; see if GM-flavored *legal* candidates fall out of the same proposer->candidate->AST-rewrite->commit loop that will handle Odoo. Ground truth = stockfish call.

Re-read through the class/debt lens, it tests TWO things:
1. *Behavior:* does the loop produce legal GM-flavored moves through the uniform pipeline?
2. *Schema (the new one):* does chess need columns Odoo's SoA didn't have? If yes, "one SoA serves all" / "the column ISA is universal" is FALSIFIED — found cheaply on a board, before the WAL froze it, instead of expensively at domain 3-4.

**Smallest first slice:** substrate WAL round-trip — write a SoA thought through surrealkv, commit with materialized witness, read back after a simulated schema bump. Floor holds -> the rest is licensed.

## One-paragraph convergence (the whole thread, folded)

Cognitive RISC: a dumb uniform SPO substrate addressed by frozen content-hash identity (CAM, never ANN), executed SIMD-style over an Arrow/Lance grid. Above it, a class layer where ~40 structurally-discovered shape-families carry inherited labels (via the already-real ontology cache + class chain), projected columns (per-class grids over a fixed universal ISA), and per-class jinja templates (instance = class_id + presence-bitmask, off-bits skipped). OGIT is the class/label registry, resolved late through a meta-DTO that holds logic not state. Behavior never inherits — it composes via hash-addressed recipes proposed by bounded proposers and arbitrated by Rubicon. The debt being paid: stop hand-rolling the inheritance the cache was built to serve. The freeze-time non-negotiables: class_id, proposer-id width, stable per-class bitmask positions, and don't freeze the column ISA until chess has falsified or confirmed its universality.

---
*v0.2. Supersedes nothing in v0.1 — extends it upward from the SoA into the class/label/view layer. Forks F1 resolved, F4 promoted to the load-bearing unsolved piece. Update as forks close; change invariants only with a stated reason.*
