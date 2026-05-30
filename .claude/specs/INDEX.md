# Cognitive RISC — Boot Index (stitched)

> **One boot-paste.** Paste this whole file at the top of a new session (Claude Code or chat)
> to load the full load-bearing context in one shot. It concatenates the four canonical
> specs in dependency order. Each spec remains the canonical, separately-editable source of
> truth in this same directory; this INDEX is a generated stitch for convenience — when it
> drifts from the individual files, the individual files win.

## Reading order (why this sequence)

1. **`cognitive-risc-core.md` (v0.1)** — the substrate invariants. The floor everything stands on. *Read first; if a line here is violated, the architecture breaks.*
2. **`cognitive-risc-classes.md` (v0.2)** — the class/shape/view layer above the SoA. Extends v0.1 upward; does not supersede it.
3. **`faiss-homology-cam-pq.md` (v0.1)** — the FAISS structural homology + the one inversion (similarity→identity, ANN→CAM). Cheat-sheet, not a dependency.
4. **`wikidata-hhtl-load.md` (v0.1)** — the architecture applied: how Wikidata lands structurally (~38GB) via HHTL/CAM. The falsification dataset.

## The thread in four lines

- **Substrate:** dumb uniform SPO; nothing semantic in the register file; meaning lives above the bytes (RISC).
- **Classes:** ~40 structurally-*discovered* shape-families inherit labels/columns/templates; behavior never inherits, it composes via hash-addressed recipes.
- **Addressing:** CAM (exact content-hash identity), never ANN. Similarity is proposer/discovery-only — two indexes, never swapped.
- **Load:** don't compress Wikidata, don't *load* most of it — store skeleton + basins + CAM-deduped shapes; stream values lazy.

---


<!-- ===== BEGIN cognitive-risc-core.md ===== -->

# Cognitive RISC — Load-Bearing Core (v0.1)

> Session-boot context. Paste at the top of any new session (Claude Code or chat).
> Contains ONLY invariants. If a line here is violated, the architecture breaks.
> Everything not here is downstream and reconstructible from here.

## The one-sentence thesis

**Dumb uniform substrate (SPO), smart operations above it.** The intelligence is never *in* the triple — it's in what operates over uniform triples. SPO's refusal to be smart is the precondition for everything above it being smart. This is RISC: dumb uniform instructions, cleverness pushed up into the compiler.

## The five-layer stack (the "RISC" frame)

| Layer | Role | Cadence | RISC analogue |
|---|---|---|---|
| **Substrate** | SoA, LE byte contract, surrealkv WAL/ACID. Policy-free state. | persistent | register file |
| **Compilation** | planning-phase JIT × AST -> compiled candidate sets | plan-speed (slow) | compiler |
| **Schedule** | kanban = the precipitated plan; hands candidates to shader | per-plan | instruction issue |
| **Execution** | cognitive shader runs precompiled candidates over SoA | sub-us (fast) | execution unit |
| **Producer** | Rubicon now / agents later. Drives compilation. | — | the program |

**Only the Producer layer changes under the AGI-inversion.** Everything below is producer-agnostic. That is the whole reason the inversion is a swap, not a rewrite.

## The invariants (violating any one = re-federation / CISC slide)

1. **Nothing semantic in the register file.** Substrate stores bytes; the layer above assigns meaning. Meaning in the byte layout = welded to today's design = inversion costs a schema migration. This is the master rule; the rest are corollaries.
2. **<f,c> + discovery_origin travel as opaque payload.** The substrate never interprets a candidate's truth-value or origin; only Producer/planner reads them. Lets candidate-combination logic be arbitrary without the substrate caring.
3. **Uniform = uniform logical schema + partitioned physical ownership.** Not one god-array. Single-writer per mailbox; cross-mailbox refs are **witness pointers**, never shared writes.
4. **Load/store discipline.** Only the **commit gate** touches the cold path (Lance/Surreal). Everything else is SoA->SoA in-arena. Hot ops never reach durable store.
5. **Witness materialization at commit.** hot->hot witnesses stay pointers (same arena clock). hot->cold and cold->hot witnesses must be **copied/snapshotted** — a cold fact must never point into an arena about to epoch-reset. This is the single rule that keeps "no compaction, just epoch reset" from corrupting provenance.
6. **Epoch reset, not compaction.** SoA is 2–6KB; per-slot fragmentation is noise. Reclaim = drop the arena when the epoch retires. Tombstones are **minimal forwarding records** (generation counter + witness back-pointer), never payload.
7. **Two-clock decoupling everywhere.** Hot path at shader speed; commit/plan at cold-store speed. Coupling them backpropagates cold latency into the shader and stalls everything. The 64k–512k SoA range is the **shock-absorber buffer** between the clocks, not a thought-count target.
8. **Backpressure is correct, not a failure.** At sub-us production it's guaranteed. Bounded mailboxes (ractor defaults to unbounded -> OOM trap). Shed under pressure by **<f,c>**: prune low-c plan-state first, protect near-commit. Degrades toward the most-believed set.
9. **Candidate generation is plan-phase, before the kanban exists.** Closes the homunculus regress: deliberation is a *phase*, the kanban is its *precipitate*. Proposers are **bounded, non-recursive** (emit k candidates). Only Rubicon does EFE arbitration. Proposers dumb, arbiter smart.
10. **Two-tier free energy.** Real Friston/EFE only at the cold/commit tier (small N). Hot tier gets a scalar proxy (<f,c> x goal-alignment, one FMA), never a planning loop. Per-thought active inference at 512k x sub-us is impossible; don't pretend otherwise.
11. **WAL persists the substrate line ONLY.** No compiled candidates, no planning artifacts in the durable set. They're reconstructible from plan + AST. If JIT'd candidates leak into the WAL, the inversion becomes a WAL-format migration.

## AST is the hub (the unification)

- **One canonical AST.** Elixir surface syntax -> AST. OWL/DOLCE/OGIT/Odoo -> *same* AST (implicit-logic extraction). Both lower to SurrealQL **and** to planner candidates.
- **Build AST as hub, not translators.** Elixir = one parser in. SurrealQL = one codegen out. Ontology extractors = other parsers in. Never Elixir->SurrealQL directly.
- **Business logic is just one proposer's candidates.** A business rule, a mined association (Aerial+), an LLM conjecture, and an AST-walk step are the *same candidate object*, differing only by `discovery_origin`. There is no business-rules subsystem — there's an AstWalker proposer that reads OWL/Odoo.
- **A move/rule/inference = a guarded rewrite over SPO state.** Same AST node shape across all domains. This is the agnosticism claim.

## Odoo extraction boundary (honest coverage line)

Declarative strata lift cleanly to AST/triples: ORM **domains** (`[('field','op',val)]` — already SPO-shaped), **ir.rule** record rules, **@api.constrains / @api.depends** field-dependency graphs. Imperative Python **method bodies do NOT** — that's the "dynamic behavior not in the static definition" wall (cf. IST/BPMN paper's 5.81% failures). AstWalker harvests declarative strata as high-confidence Curated/Extracted candidates; flags method bodies as low-c, defer-to-runtime-trace. Don't try to static-AST-walk arbitrary Python into business logic — it won't generalize.

## Foundry relationship (resolves the recurring category error)

**Foundry is an interpreter over a live ontology. This stack is a compiler over a frozen one.** Every Foundry component maps by *semantics* but **inverts by binding time**: Workshop (runtime widget binding) vs A2UI (compile-time ontology->UI projection). "dynamic" / "low-code" / "LangGraph-like" all smuggle in Foundry's *runtime-interpretation* model — reject that import. LangGraph maps to the execution semantics correctly and to the binding time incorrectly: this stack is *compiled* LangGraph-over-triplets, not interpreted.

Semantic map (correct), binding inverts (the catch):
Ontology -> Ash-resource-shaped / SPO ; Action -> governed AST rewrite ; Function -> Rust semantic fn / ractor handler ; Workshop -> A2UI projection ; Automate -> ractor mailbox + PubSub. All real by meaning, all early-bound where Foundry is late-bound.

## discovery_origin (u8) — !! ISA WIDTH AT RISK

```
bits 0-1 : ProvenanceTier (4)  -- Curated/Extracted/ArmDiscovered/Ratified   [stable]
bits 2-4 : proposer id    (8)  -- AstWalker/PairStats/Aerial+/LLM/dIPC-A/dIPC-B/... [GROWS]
bits 5-7 : reserved       (3)
```
**Proposer-id at 3 bits caps at 8; already 6 named.** "Business logic is just another proposer" + dual-IPC dialectic both imply proposers proliferate. Widen proposer field (steal reserved -> 6 bits/64, or go u16) **before surrealkv WAL hardens the LE wire format.** Once it's in the byte grammar, widening = migration across every component. This is the RISC ISA-ossification trap, live, now.

## Open forks (load-bearing but deferrable — NOT yet decided)

- **F1 — UI binding time.** Does an ontology change update running UIs *without rebuild*? YES -> runtime templating (minijinja/interpreter), Workshop is a real mechanical model. NO -> compiler, askama right, Foundry = inspiration only. *Everything in the UI layer falls out of this one answer.* Default lean: NO (compile-time, to stay coherent with plan-time-everywhere).
- **F2 — SurrealDB integration.** Read Lance storage directly (heavy, fragile vs both release cadences) **or** federate via shared DataFusion catalog (Arrow TableProviders, tractable). Default lean: federate.
- **F3 — jinja+AST vs minijinja/JIT.** Downstream of F1. Detail, but load-bearing once F1 lands.
- **F4 — proposer-id width.** Decide before WAL hardens (see above). Not really optional.

## The bring-up test (the falsifiable slice)

**Chess into OWL.** Encode openings/methods/verbs as OWL/ttl (meaning in the *content*, never the substrate — no chess-special field, ever). Run proposers (Aerial+ as proposer-not-oracle) over it. See if GM-flavored *legal* candidates fall out of the same proposer->candidate->AST-rewrite->commit loop that will handle Odoo. Ground truth is a stockfish call away. Exercises every layer on a checkable board.

**Smallest possible first slice:** substrate WAL round-trip — write a SoA thought through surrealkv, commit with materialized witness, read back after a simulated schema bump. If the LE-contract + versioning survives that, the floor holds and the rest is licensed.

## Pin versions (single coupling point)
lance 6.0.1 / lancedb 0.29 / datafusion 53. SurrealDB versioning aligns here.

---
*Pin numbers and "default leans" are as-of-this-doc; update as forks resolve. Everything above the forks section is invariant — change it only with a reason you can state.*


<!-- ===== END cognitive-risc-core.md ===== -->


<!-- ===== BEGIN cognitive-risc-classes.md ===== -->

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


<!-- ===== END cognitive-risc-classes.md ===== -->


<!-- ===== BEGIN faiss-homology-cam-pq.md ===== -->

# FAISS Homology — CAM / CAM_PQ (v0.1)

> Companion to core (v0.1), classes (v0.2), wikidata-hhtl-load (v0.1).
> "Sieht entfernt wie FAISS aus" was right from turn one — it was the FAISS *architecture* (layering), never the FAISS *algorithm* (ANN). This doc pins the homology and the ONE inversion.
> ⚠ HOMOLOGY IS A CHEAT-SHEET, NOT A DEPENDENCY. Do not pull IVF/FAISS code. Implement with Lance-SoA, HHTL-nibbles, BLAKE-CAM, provenance-reasoning-store.

## The structural homology (term for term)

| FAISS | This stack |
|---|---|
| flat vector backing arrays | **SoA** (flat columnar, ID-encoded, `(start,len)` backing) |
| IVF cells / coarse quantizer | **HHTL buckets** (16^n nibble routing, arithmetic not associative) |
| PQ residual codes | **facet codes** (product of per-facet closed-vocab indices) |
| orthogonal index (the inverted file) | **Reasoning layer** (separate indexed store, Derived tier) |
| vector-id → offset | **CAM hash → shape/row** (exact-match) |

## The ONE inversion (the whole thread, in one line)

**FAISS addresses by SIMILARITY (ANN). This stack addresses by IDENTITY (CAM).**
Same architecture — flat + bucket + index. Opposite addressing — near vs. exact.
You kept FAISS's skeleton and swapped its heart: similarity out, identity in. That's why it's *entfernt* like FAISS — the shape is there, the soul is inverted.

## CAM_PQ — product quantization, made symbolic and exact

PQ's move: split a vector into m subvectors → quantize each against a small codebook → store m codebook indices (the product code) instead of the full vector.

**CAM_PQ applies the PQ LAYOUT, not the PQ LOSS:**
- split an entity descriptor into **facet subspaces** (Abstammungs-path, capability, habitat, shape, organic, ...)
- each facet's "codebook" = its **closed OWL range** (owl:oneOf / small rdfs:range) — declared, not learned
- store one **index per facet** = the product code (a tuple of small symbolic codes)
- **CAM-hash the whole product code** for exact identity

Structurally this is **IVFPQ**: HHTL path = the coarse quantizer (IVF cell), facet codes = the PQ residual codes, CAM hash = the exact ID. But:

### Critical: exact, not lossy
PQ is lossy (quantization error). **CAM demands exactness — so facet codes are LOSSLESS.** Because the vocabularies are CLOSED and SMALL, every value gets its own code with zero quantization error. It is really **dictionary encoding with product structure**, not quantization. The "PQ feel" is the product LAYOUT (split into sub-codes, code per subspace); the exactness comes from closed-vocab codebooks. No information lost → CAM identity holds → audit/GoBD safe.

| | FAISS PQ | CAM_PQ |
|---|---|---|
| codebook | learned (k-means over floats) | declared (OWL closed range) |
| code | lossy (nearest centroid) | exact (every value has its own code) |
| addressing | ANN (similarity) | CAM (exact hash) |
| layout | product of m sub-codes | product of facet codes |
| use | retrieval by nearness | identity + dedup + codegen |

### Why CAM_PQ is cheaper than float PQ
- codebooks are **declared** (OWL/DOLCE), not trained — no learning pass, no drift
- codes are **exact** — no re-ranking pass to fix quantization error
- product code is tiny: k facets × small index ≈ one u64 → AND-testable / hashable in a cycle
- the facet u64 IS the SoA facet column → SIMD batch-AND = facet filter (cognitive-shader-driver grid run)

## Where REAL (lossy) PQ is still allowed — and only there

The invariant holds: **similarity/lossy lives ONLY in the proposer/discovery layer, never in addressing.** So genuine lossy PQ (or any ANN/embedding) is legitimate for:
- the **value stream entropy wall** (the irreducible SPO-object data that doesn't fold into a deck slot) — IF you ever want lossy compression there, it's a discovery aid, not identity
- **label/text similarity** and **shape-family discovery** (Aerial+, Jina) — proposing inheritance/relations
Never for: recipe selection, identity, the CAM key, the reasoning-store keys. Those stay exact.

**Rule restated:** facet-PQ = product layout + exact closed-vocab codes (lossless, addresses identity). float-PQ/ANN = lossy, discovery only, never addresses. Same triples, two indexes, never swapped.

## The closed picture (architecture is closed, now falsifiable)

Class (Quartett mask, inherited along HHTL as delta) + SoA (flat columnar backing) + HHTL (16^n nibble router = IVF cells, OWL/DOLCE axis template, facets as orthogonal bitmasks) + CAM (exact identity = the swapped FAISS heart) + CAM_PQ (product-structured lossless facet codes) + Reasoning (orthogonal indexed Derived store = FAISS's inverted index, over inferences). All one SPO substrate, separated by provenance, governed by: dumb-uniform below, meaning above.

**Closed, not finished.** No open edge forces a new structural question; every layer docked without overturning an invariant. The claim is now falsifiable: chess + Odoo + Wikidata-anatomy all run through the same Class+SoA+HHTL+CAM+Reasoning with no special-case. The architecture is closed in the head; whether it's closed in the bytes is told by the first dataset that runs all five layers and either comes out clean or forces a layer to lie.

---
*v0.1. The homology explains the form; your building blocks are the implementation. Cheat-sheet, not dependency.*


<!-- ===== END faiss-homology-cam-pq.md ===== -->


<!-- ===== BEGIN wikidata-hhtl-load.md ===== -->

# Wikidata Load — Maximal-Efficiency HHTL/CAM Pipeline (v0.1)

> Companion to `cognitive-risc-core.md` (v0.1) + `cognitive-risc-classes.md` (v0.2).
> This = how Wikidata lands in ~38GB instead of ~120GB by being ARCHITECTURE-driven, not compression-driven.
> Measured anchor (10 real entities): 1.43MB multilingual → 10.3KB thin+shapes = **139x** before columnar/ID-encoding.

## Principle

Don't compress Wikidata. **Don't load most of it.** Load the skeleton + basins + CAM-deduped shapes + thin rows; stream values lazy-per-basin. The reduction is structural (you store classes+masks+refs, not 115M fat JSON blobs), not a gzip trick.

## The layer model (this is the whole thing)

```
ACHSEN (deklariert, roh) — frozen identity
  Abstammungs-HHTL    subClassOf(P279)-Pfad, 16^n Nibbles      ← the ONE tree axis
  Facetten-bitmasks   geschlossene ObjectProperties             ← closed small vocab → bits
  Quartett-Werte      DatatypeProperties                        ← value tuple, position-coded
ORTHOGONAL (abgeleitet, indiziert) — re-materializable
  Reasoning-Datensatz DL-Schlüsse, einmal materialisiert, CAM-indiziert
    - transitive Hüllen (subClassOf*, partOf*)
    - inferierte Klassenzugehörigkeiten
    - Disjunktheits-Verletzungen / Konsistenz-Flags
```

Everything in the SAME SPO/Lance substrate, separated by `provenance` tier (v0.1 discovery_origin: Curated/Extracted/Derived). Raw axes = declared; reasoning store = Derived. Every edge knows if it was declared or inferred — required for GoBD/audit.

## Two-pass streaming (constant memory, never materialize)

`latest-all.json.gz` = one entity per line. Stream through gzip, parse one entity, write, forget. Constant RAM regardless of dump size; never decompress to disk.

**Pass 1 — Skeleton (structural, cheap):** per entity extract ONLY P31 (instance-of), P279 (subClassOf), en/de labels, property-id SET present. Output: the P279 DAG (= the gifted parent-pointer) + P31 classification + property registry. ~2-3M classes, ~12k properties → 1-2GB, fits RAM. Here: cut basins (HHTL levels), identify capability/facet compartments from OWL/DOLCE template.

**Pass 2 — Bucket + AST + CAM:** second stream. Per entity: class via P31 → basin via P279* reachability (precomputed pass 1) → route to HHTL bucket. Claims → AST nodes referencing the basin codebook (thin, shared). Shape = (class-set, canonical property-set) → BLAKE2b-128 → CAM. Identical shapes dedup. Entity persisted as (class_id, shape_hash, presence_bitmask, value_tuple, en, de).

## The four reduction levers (these get 120→38, not gzip)

1. **Single/few languages.** en + de as TWO separate columns, one parser (language = value not structure; both are projections of the same CAM hash). The 300+ langs are most of the 120GB — dropping them is the biggest single win.
2. **Drop references/qualifiers** (the statement provenance bloat) for the structural load.
3. **ID-encoding.** QIDs/PIDs as u32, never strings ("IDs statt Strings"). Position-coding inside a dense deck eliminates per-value prop_ids entirely.
4. **CAM shape-dedup + basin sharding.** Thousands of "human" instances share one shape; basin shards are homogeneous → compress hard.

## HHTL = the cheap bucket router (16^n)

- Fixed fan-out 16 per level → bucket path = nibble sequence → routing is bit-shift, not hash lookup. O(1) arithmetic ("super billig").
- **The mask inherits along the HHTL path as DELTAS.** A leaf deck stores only its increment over the parent path. Common wide fields live HIGH (once); specific fields live LOW (leaf). This is what prevents the sparse-union disease — decks stay dense because shared columns are inherited, not repeated.
- **ONE tree axis only (Abstammung).** Multi-parent (flying-family) is NOT a second tree branch — it's an orthogonal facet-bitmask. Bat = mammal-path + flight-bit, not two paths. Keeps 16^n a clean tree (cheap nibble addressing) AND keeps multi-parent dedup (verb "fly" stored once in the capability compartment, bit points at it).
- **Open question (the one untested assumption): P279 fan-out is wildly uneven** (some classes 2 children, some 4000). Whether it re-balances onto 16^n or forces adaptive fan-out (4^n here, 16^n there) is MEASURABLE — measure fan-out distribution on a real P279 subtree before fixing the base.

## Facets: OWL/DOLCE as the template (the brutal shortcut)

Don't guess the axes — **harvest them from OWL.** OWL declares the facet-vs-path distinction you'd otherwise measure:

| OWL construct | HHTL form |
|---|---|
| rdfs:subClassOf (transitive) | Abstammungs-path (nibbles, 16^n) |
| owl:partOf / transitive props | further path axes (inherited as delta) |
| ObjectProperty, small closed range | facet-bitmask (1 bit per range individual) |
| DatatypeProperty | Quartett slot (value-tuple position) |
| owl:oneOf / enumeration | closed vocab = exact bit-budget |
| **owl:disjointWith** | **disjoint facets = collision-free, purely additive** |
| owl:Restriction (someValuesFrom) | presence-bitmask rule (which bit must be set) |

- **disjointWith auto-solves the multi-parent conflict question.** Where OWL declares disjoint → facet bits never collide, no linearization needed. Where it doesn't (penguin: fly+swim) → exactly there you need the conflict rule. The overlap set = the non-disjoint property pairs, enumerable because declared.
- **Closed range → bitmask; open/no range → path or ref.** The template IS the decision; no empirical cardinality measurement needed.
- **DOLCE as axis skeleton (clean top facets: Object/Process/Quality/Region ≈ your Object/Organic/Properties/Shape), Wikidata properties as the fill (dirty but real leaf vocab).** DOLCE defines WHICH axes; Wikidata fills WHAT occurs in each. = the DOLCE→cross-domain→industry distillation, as HHTL axis template.

## Facet bit-budget discipline (the ISA-width trap, again)

- Closed small vocab → fixed bitmask (Habitat ~dozen → 5 bits w/ growth reserve; capabilities ~40 verbs → 6 bits). Five real facets together ≈ one u64, AND-testable in one cycle → SIMD batch-AND over the SoA facet column (the cognitive-shader-driver grid run).
- Open/unbounded vocab → NOT bitmask: `Properties` = the Quartett mask (inherits as path-delta); `ElementOf` = a ref-set (unbounded → indirection, not bits).
- **Rule:** fits permanently in 16/64 bits → bitmask. Grows unbounded → path/ref. Once facet bit-allocation is in the LE/HHTL header it's frozen: append-only, never renumber.

## Reasoning as orthogonal indexed dataset (NOT thrown away, NOT runtime)

Pre-materialize DL inferences ONCE, hash them (CAM), index them. "What follows from X" = exact-match lookup, not a reasoner run. The Derived tier.

- **CAM applied to inferences:** an inference is a derived triple `(bat, subClassOf*, vertebrate)`; materialize once, hash, index. Reasoning gets its own CAM layer.
- **Orthogonal = beside, not mixed in.** Raw axes = declared (frozen). Reasoning store = derived (separate index). Card = path + facets + values + ref-into-reasoning-layer. Declared and derived never merge — provenance preserved (GoBD).
- **Re-materializable without touching raw.** Ontology changes (new axioms/disjointness) → recompute ONLY the reasoning layer, raw axes stay. = F1's frozen-identity-under-live-resolution, for reasoning: raw frozen, reasoning re-resolvable, separately indexed.
- **Index the derived triples 3 ways:** by Subject ("all superclasses of X" / transitive closure up), by Object ("all entities implied to be Y" / down), by generating-axiom (provenance / consistency). Classic SPO store over derived triples, same Lance substrate, `provenance=Derived`.

## Open / to measure (last untested numbers)

- **Mask density per deck** (fraction of deck columns set across cards): dense deck = good Quartett = max reduction; sparse = cut basin deeper. This is THE optimization metric.
- **CAM dedup rate** (how many of N share a shape) — drives the thin-row fraction; not measurable on 10, needs a homogeneous cluster pull (~200 instances of one class).
- **P279 fan-out distribution** — does 16^n hold or force adaptive fan-out.
- **Value stream size** (the irreducible SPO-object edges that don't fold into a deck slot) — the entropy wall; position-coding shrinks it but doesn't eliminate it.

## Scaling math (115M entities, structural, en+de)
- thin rows + labels: 115M × 136B ≈ 15.6GB JSON → columnar+u32+RLE → ~5-8GB
- shape table: <0.5GB (amortizes — bounded by distinct shapes, not entities)
- statement values (separate columnar store): ~8-15GB (most compressible: sort by prop_id, dict, RLE)
- reasoning store (derived): sized by inference fan-out, separately indexed
- **Total ~15-25GB → fits 38GB with headroom** for extra languages / qualifiers / eager values.
- **Default lean:** skeleton+shapes+labels EAGER (~6GB hot), values+extra-langs LAZY-per-basin from Lance cold store. CAM hash is the key joining both. = two-clock pattern applied to the load.

---
*v0.1. The architecture is closed; this is its application to Wikidata. Numbers above the scaling section are measured/structural; scaling is projected — confirm with the mask-density + dedup-rate cluster run before trusting 15-25GB.*


<!-- ===== END wikidata-hhtl-load.md ===== -->
