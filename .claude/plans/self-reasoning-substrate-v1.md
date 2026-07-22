# Self-Reasoning Substrate — the graph reasoning about itself (v1)

> **THE PUNCHLINE:** almost no new representation is needed. The existing ones
> must be *pointed at themselves.* The grammar tree was proven to be a **pointer
> fabric over the word stream** (Manning & Carpenter 1997 p.153;
> `E-GRAMMAR-TREE-IS-POINTER-FABRIC-1`). Self-reasoning is **the SAME fabric one
> level up**, over the *triple* stream. A proof tree is to triples what a parse
> tree is to words — and, like the parse tree, it never needs to exist as an
> object.

---

## §0 — Status

- **Status:** PROPOSED — doc-only. No code, no contract change, no build surface
  touched by this file. Deliverables D-SRS-1..4 below are gated (probe-first,
  gates before code); none is authorized to land by this document.
- **Runs on:** the `E-WHOLE-BOOK-WAVE-1` substrate — the whole KJV (23,145
  verses) resident in ONE 256×256 tile, a live KG of **31,327 triples / 606
  subjects / 1,081 predicates** from the 6-state FSM, with a REAL Jina-trained
  Cam96 codebook (`crates/deepnsm-v2/`, `examples/bible_wave.rs`,
  `data/cam96_codebook.bin`). That artifact is the corpus this plan reasons
  *about*; it is not re-derived here.
- **Grounding reads (mandatory before touching any D-SRS deliverable):**
  `E-WHOLE-BOOK-WAVE-1`, `E-LC-SCARCITY-INVERSION-1`,
  `E-GRAMMAR-TREE-IS-POINTER-FABRIC-1`, `E-CAM96-DISTRIBUTION-MEASURED-1`
  (+ its correction `E-CAM96-REVIEW-CORRECTIONS-1`), and
  `.claude/knowledge/left-corner-grammar-tree-pointer-fabric.md`.
- **Confidence / labelling:** the five-layer mapping (§1) is **FINDING-grounded
  wherever it cites shipped code** (the TemporalStream, RungLevel, NARS
  revision, the 32-tenant row, `witness_fabric`), and **CONJECTURE where it
  claims a new behavioural result** — chiefly D-SRS-4's claim that the graph can
  correctly answer a question about its own earlier derivation. Every §2 gate is
  written to *falsify* the CONJECTURE, not to confirm it.

**Why this is the right shape (the scarcity inversion, `E-LC-SCARCITY-INVERSION-1`):**
the entire left-corner tradition never had a substrate that could hold all
meanings of a book in parallel, so it discarded competing analyses at the
sentence boundary. Self-reasoning was *structurally impossible* there — there was
nothing left to reason about after the parse. The 64k SoA removes that premise:
the whole book (and, now, the whole *derivation* over the book) is co-resident
and replayably addressable. Import the linguistics — depth ≤ 8, the pointer
fabric, table-driven checks — and leave the beams behind.

---

## §1 — The five layers (4 exist, 1 new)

The recursion bottoms out cleanly and lives entirely in one 64k tile:

```text
words  ──(pointer fabric)──▶  triples  ──(SAME fabric)──▶  derivations  ──(SAME fabric)──▶  beliefs-about-derivations
  ±8 loci → attachment site      ±8 loci → premise triple      ±8 loci → cited derivation
  rung 0                          rung 0–1                       rung 2+                        rung 3+
```

Each arrow is the identical geometry; each hop is one rung up; all of it is
co-addressable in the same tile.

### Layer 1 — Homoiconicity via the stream *(exists)*

Derived triples — NARS deductions, contradiction verdicts, revision outcomes —
**append to the same `TemporalStream`** that carries the observed triples, each
with its own version stamp. The reasoner's outputs become its inputs. Two
consequences fall out for free:

- **The version axis IS the reasoning history.** A version-range read over the
  derivation segment (`QueryReference::at(v, rung)` + deinterlace) = *reading its
  own thinking* — replayable, per-reader, non-destructive. There is no separate
  "trace" datastructure; the stream is the trace.
- **No new carrier.** This is `E-MARKOV-TEMPORAL-STREAM-1` applied to the
  reasoner's own output: the Markov property holds by stream order, not by bundle
  associativity. (Do **not** reach for `vsa_bundle` here — `E-NO-BUNDLE-STANDING-WAVE-1`.)

*Status: FINDING — the TemporalStream + version-range read are shipped
(`deepnsm-v2`, `E-WHOLE-BOOK-WAVE-1`).*

### Layer 2 — Derivation DAG = pointer fabric one level up *(the elegant recursion)*

Intra-sentence, the 24×i4 reference pointers bind each word to an attachment site
within ±8 (`fsm.rs`/`wave.rs`; `E-GRAMMAR-TREE-IS-POINTER-FABRIC-1`). **Inter-triple,
the SAME shape binds a derived triple to its PREMISE triples:**

- Most derivations cite *recent* premises → **±8-local**, resolved by the cheap
  local check (Moore's cheap-local-gates-expensive-global, +67%).
- Long-range premises **Escalate** to the basin / global graph — identical to the
  parse-side offset overflow. The bound is on premise *count* (open citations),
  not on version *span*: one citation can reach an arbitrarily old premise, so
  Escalate fires on offset overflow, exactly as `wave.rs` already does.

A proof tree is to triples what a parse tree is to words. **It never needs to
exist as an object** — the pointers ARE the tree; materialize it only at output
(Manning & Carpenter p.153; Moore §7's 2-field back-pointer suffices to
reconstruct every parse — the precedent that minimal pointers carry the tree).

*Status: FINDING for the geometry (the ±8-loci + Escalate fabric is shipped);
CONJECTURE that it composes premise citations soundly on the real KG — D-SRS-1
is the falsifier.*

### Layer 3 — Rungs = Tarski stratification, byte-sized *(exists — `RungLevel`)*

A triple *about* a triple lives one rung up. Object facts at rung 0–1, derived
facts at rung 2+, meta-beliefs higher. This byte-sized stratification is **what
makes self-reference safe** rather than paradox- or runaway-shaped:

- The relation-blind transitivity runaway is real and recorded:
  `TD-INFER-DEDUCTIONS-RELATION-BLIND` — `infer_deductions` composes ANY two
  edges sharing an entity, unsound for non-transitive relations, surfaced by
  `E-SELF-DIRECTED-GRAPH-1`. Stratification is the discipline that bounds it: a
  derived triple is stamped rung *n+1* and a fixed-point closure over rungs
  terminates instead of looping.
- `carried_awareness` / `active_after_prune` are the climb discipline — the
  epistemic budget that survives a prune, so the reasoner does not re-derive what
  it already retired.

*Status: FINDING — `RungLevel` and the climb fields are shipped; the runaway is a
documented, reproducible failure this layer exists to fence. D-SRS-2 turns
`TD-INFER-…` into a termination test.*

### Layer 4 — NARS truth + preserved contradictions *(exists)*

`c = n/(n+k)` is inherently self-referential — confidence *is* a statement about
the evidence for a statement. The machinery is already introspective:

- **Revision is the introspection operator.** `TruthValue::revision` /
  `NarsTables::revise` fold new evidence into a held belief across passes —
  measured live: single-obs conf flat 0.502 vs multi-hop 0.502→0.858 over witness
  depth 1→6 (`E-MULTIHOP-WITNESS-CONFIDENCE-1`); self-correcting KG 0.500→0.667→
  0.750 across passes with 0 new triples after pass 1 (`E-SELF-CORRECTING-KG-1`).
- **The φ-1 ceiling = permanent humility** — confidence can never reach 1; the
  graph structurally cannot become certain of itself.
- **Stored contradictions = the graph's record of its own unresolved tensions**
  (opinions are committed contradictions preserved, not resolved). This is the
  raw material Layer 5 measures.

*Status: FINDING — revision, the ceiling, and preserved contradictions are all
shipped and measured.*

### Layer 5 — THE ONE NEW ARTIFACT: basin self-codes

Each basin / subgraph gets its **own Cam96 code** — the centroid of its member
codes — so the graph can **measure its OWN topology in meaning space**:

- **where contradictions cluster** (which basins hold the most preserved
  tensions),
- **where the distribution is wide** = *"I am uncertain here"* (the 96-bit
  `6×256:256` DISTRIBUTION already encodes meaning-*spread*, not a point —
  `E-CAM96-DISTRIBUTION-MEASURED-1`; the spread is the uncertainty signal),
- **which basins drift across versions** = *"my beliefs changed"* (basin
  self-code delta between two version reads).

**Wire, don't invent.** The 32-tenant row already reserves the lane; the
MUL / Dunning-Kruger machinery (`lance-graph-planner/mul/`) consumes exactly this
kind of self-measurement signal. The basin self-code is a *read* over existing
member codes plus one reserved column — not a new tenant, not a new layout
version. Honest bound (`E-CAM96-REVIEW-CORRECTIONS-1`): the DISTRIBUTION's
advantage over a POINT is chiefly ALGEBRAIC (independently-addressable rails),
not raw fidelity — cite it that way.

*Status: CONJECTURE — the self-code is a small, well-scoped read, but that the
graph emits a *correct* "where am I uncertain" report matching held-out
measurement is unproven. D-SRS-3 is the falsifier.*

---

## §2 — Deliverables (gates before code)

Each deliverable registers its pass/KILL gate in this section **before** any
example is written. No D-SRS deliverable lands before its gate is green.

> **On the un-filled numeric thresholds (D-SRS-2 pass ceiling, D-SRS-3
> correlation floor + rank-combination rule, D-SRS-4 exact question / expected
> answer / tolerance):** these are deliberately UNSET in this PROPOSED plan.
> Filling them now — before the deliverable is authorized and before the
> held-out split exists — would be false precision, and worse, a number chosen
> at plan-time invites being quietly re-tuned once results are seen. The
> anti-gaming discipline is: **each threshold is registered in a dated,
> append-only pre-run record IN this section as the FIRST commit of the
> deliverable's own work (before its example compiles), then never edited.**
> The registration commit predates the measurement commit in git history —
> that ordering, not a plan-time guess, is what proves the gate was not tuned
> post-hoc. Until a deliverable is authorized, its gate reads "to be
> registered pre-run" by intent, not omission.

### D-SRS-1 — Derivation-pointer fabric over the Bible KG

Derive `is_a`-style deductions over the shipped 31,327-triple KJV KG, each
carrying **premise pointers** (Layer 2). Materialize no proof-tree object — the
pointers are the tree.

- **PASS gate:** every derived triple's premises are resolvable via its pointers
  (round-trip: pointer → premise triple, for 100% of derived triples), AND the
  derivation graph is **acyclic** because every citation points to a
  **strictly-lower** rung (never equal, never higher). Strictly-lower is the
  invariant that *guarantees* acyclicity: equal-rung citations do NOT — two or
  more equal-rung edges can close a cycle with no upward edge — so the gate
  forbids them outright (this is Tarski stratification, consistent with D-SRS-2
  stamping each derived triple at rung *n+1* of its deepest premise).
- **KILL:** any derived triple with a dangling/unresolvable premise pointer, OR
  ANY citation cycle at all (whether it crosses rungs upward or sits within one
  rung). Either falsifies "the fabric composes premises soundly."

> **Pre-run registration — D-SRS-1 (2026-07-22, registered BEFORE the code; the
> anti-tuning commit precedes the measurement commit in git history).** The gate
> is STRUCTURAL, not a tunable threshold, so the registered values are the exact
> binary assertions the run must satisfy:
> - **Inference rule (fixed):** per-predicate transitive composition ONLY — for
>   arena entries `(A,p,B)` and `(B,p,C)` sharing the **same predicate `p`**,
>   derive `(A,p,C)` with premise pointers `[i,j]`. Cross-predicate composition
>   is FORBIDDEN (that is the `TD-INFER-DEDUCTIONS-RELATION-BLIND` runaway; here
>   the sound is_a-style rule keeps `p` constant). Self-loops (`A==B` or `B==C`)
>   and re-derivation of an already-present triple are dropped (dedup by `Spo`).
> - **Rung stamp (fixed):** base triples rung 0; a derived triple is stamped
>   `max(premise rungs) + 1`. This makes every premise strictly-lower by
>   construction.
> - **PASS = all three, exactly:** (1) **premise resolvability = 100.0%** — every
>   premise pointer indexes an EARLIER arena entry that exists (0 dangling);
>   (2) **acyclic = true** — every premise strictly-lower rung than its citer
>   (0 equal-or-higher citations), verified explicitly, not assumed;
>   (3) **terminates = true** — the fixed-point closure reaches a fixed point
>   (a pass adds 0 new triples) in bounded passes on BOTH the deterministic
>   unit-test KGs AND the real 31,327-triple KJV KG.
> - **KILL = any of:** resolvability < 100.0%, OR one equal/higher-rung citation,
>   OR the closure does not reach a fixed point. Report the failing metric
>   verbatim; do NOT relax the rule to make it pass.
> - **Proof surface:** the invariants are proven by deterministic `#[test]`s in
>   `src/reason.rs` (no corpus, no network — the gate); the KJV run is the SCALE
>   demonstration (the same assertions re-checked on the book-scale KG).

> **RESULT — D-SRS-1 SHIPPED, gate met, with one finding (2026-07-22; commits
> `6008747` gate → `f01d874` code → the adjudication commit; the registration
> above is UNEDITED per anti-tuning).** `src/reason.rs`
> (`DerivationArena::derive_transitive[_capped]`) + 7 deterministic unit tests +
> the `bible_wave` D-SRS-1 leg.
> - **SOUNDNESS (the KILL clause: dangling pointer OR any cycle): PASS.** 100.0%
>   premise resolvability + acyclic (every premise strictly-lower rung), proven
>   exhaustively by the unit tests AND re-verified on the real book — 21,749
>   distinct base triples (the 31,327 whole-book triples dedup to 21,749 distinct
>   `Spo`), 50,000 derived at the bounded horizon, resolvability 100.0%,
>   acyclic=true. The falsifier did not fire.
> - **TERMINATION: PASS where the closure is tractable** (all unit-test KGs reach
>   a fixed point; finiteness guarantees it on any KG). **FINDING:** the FULL
>   whole-book closure is genuinely **O(N²)** — >50,000 two-hop compositions in
>   the FIRST pass alone (hub verbs + the literal `begat` genealogies are long
>   same-predicate chains). Running it to a full fixed point is intractable, so
>   the book leg BOUNDS the horizon and asserts SOUNDNESS (which holds on any
>   prefix). This is not a miss — it **empirically demonstrates that Layers 2-3
>   are load-bearing, not optional**: derivation MUST be bounded (±8-local +
>   Escalate; the D-SRS-2 rung cap). The registered "full-book termination"
>   sub-clause is thereby superseded by the architecture's own bounded-derivation
>   posture; D-SRS-2 is its proper home. Recorded as a finding, not a silent
>   relaxation — the registration stands as written.

### D-SRS-2 — Rung stratification enforcement

Stamp every derived triple at rung *n+1* of its deepest premise; run a
fixed-point closure and prove it terminates. This is **`TD-INFER-DEDUCTIONS-RELATION-BLIND`
as a test**: the known runaway must be fenced by stratification, not by luck.

- **PASS gate:** fixed-point closure over the KJV KG **terminates** (reaches a
  fixed point in bounded passes) with no runaway — specifically, the relation-blind
  transitivity that produced the `TD-INFER` runaway must halt under the rung cap,
  and the derived-triple count converges rather than growing unboundedly.
- **KILL:** closure fails to terminate within the registered pass ceiling, OR
  derived-triple count grows monotonically without a fixed point. Either means
  stratification does not actually bound self-reference — the whole safety claim
  of Layer 3 fails.

### D-SRS-3 — Basin self-codes + self-report

Compute the Cam96 centroid self-code per basin (Layer 5) and emit a
`"where am I uncertain"` report ranking basins by distribution width /
contradiction density. Wire the read into the reserved 32-tenant lane; consume it
where MUL already expects a self-measurement signal. No new tenant, no layout bump.

- **PASS gate:** the emitted uncertainty ranking **matches a held-out
  measurement** — i.e. the basins the graph *reports* as widest/most-contradicted
  agree (rank correlation above a pre-registered floor, computed against a
  held-out split so it cannot pass in-sample) with an independent measurement of
  distribution spread / stored-contradiction count on those basins. Cite the
  advantage as ALGEBRAIC per `E-CAM96-REVIEW-CORRECTIONS-1`; run held-out, never
  in-sample (the correction that widened the CAM96 gap the honest way).
- **KILL:** the self-report's ranking is uncorrelated with (or inverted from) the
  held-out measurement. The graph does not know where it is uncertain.

### D-SRS-4 — The self-reference falsifier

**The graph answers a question about its OWN earlier derivation, correctly.**
E.g.: "which premises did you use to conclude triple X?" or "did your confidence
in belief Y change between version *v1* and *v2*, and why?" — answered by a
version-range read over the derivation segment (Layer 1) following premise
pointers (Layer 2), stratified (Layer 3), with NARS-revised confidence (Layer 4).

- **Gate defined before code, KILL-gated:** register the exact question, the
  exact expected answer (the ground-truth premise set / the ground-truth
  confidence delta computed independently from the stream), and the tolerance,
  in this section BEFORE writing the example. PASS = the graph's self-answer
  matches the independently-computed ground truth within tolerance. **KILL** = it
  does not — the self-reference loop is not closed, and this plan's central
  CONJECTURE is falsified (report it as such; do not soften).

---

## §3 — Explicitly out of scope

- **No new tenant.** Layer 5's basin self-code is a read over existing member
  codes into a *reserved* lane of the 32-tenant row. Adding a tenant is a
  different, un-authorized change.
- **No `ENVELOPE_LAYOUT_VERSION` bump.** Nothing here reclaims or repositions any
  byte. If a deliverable appears to need a layout change, STOP — it has drifted
  out of this plan's scope.
- **No VSA superposition of codes** (`I-VSA-IDENTITIES`). Basin self-codes are
  *centroids of trained-codebook cells*, never superposed content registers.
  Bundling Cam96 codes into one register destroys the mapping back to centroids —
  forbidden. (And no `vsa_bundle` standing-in for the stream —
  `E-NO-BUNDLE-STANDING-WAVE-1`.)
- **No beam / k-best / prune-at-parse / per-sentence reset.** The do-not-import
  list from `E-LC-SCARCITY-INVERSION-1` applies to the derivation stream exactly
  as to the parse stream — ambiguity persists as the distribution and resolves by
  a per-reader read, never a destructive choice.

### Anti-bias guards (plan invariants — from the multihop-confirmation ruling)

These bind every D-SRS deliverable, so that "the graph confirms itself" cannot be
manufactured by confirmation bias (the failure `E-CAM96-REVIEW-CORRECTIONS-1`
caught live — a single self-confirming path passing *through* a bug while two
independent review paths converged on the real defect):

1. **Confirmation = INDEPENDENT-path convergence on the SAME ABSOLUTE event**
   (witness-fabric `elect_peers` semantics — `witness_fabric::standing_wave_grounded`).
   Two derivations that trace to the *same* premise are ONE piece of evidence, not
   two.
2. **Evidence deduped by absolute identity.** Count an event once, by its absolute
   stream position — never once per path that reaches it.
3. **Jirak-deflated counting** (`I-NOISE-FLOOR-JIRAK`). The fingerprint bits are
   weakly dependent by construction; any "N σ above noise floor" claim cites
   Jirak's weak-dependence rate, not classical Berry-Esseen.
4. **Pearl-mask separation.** Associative convergence (two beliefs resonate) is
   NOT causal confirmation (one belief *caused* another). Keep the 2³ CausalMask
   as a projection selector; do not let similarity masquerade as confirmation.
5. **Candidates by similarity, evidence COUNTED by exhaustive window read.**
   Generate premise/derivation candidates with cheap similarity (the ±8 local
   check), but COUNT supporting evidence by an exhaustive version-window read —
   **never top-k.** Top-k over evidence is the beam by another name; the whole-book
   literal window read (`E-WHOLE-BOOK-WAVE-1`: returns all 31,327 triples, no
   beam, no reset) is the counting primitive.

---

## §4 — Cross-refs

- **Substrate it runs on:** `E-WHOLE-BOOK-WAVE-1` (the 31,327-triple KJV KG in one
  64k tile, trained Jina codebook), `crates/deepnsm-v2/` (`space.rs`, `codebook.rs`,
  `wave.rs`, `fsm.rs`, `temporal.rs`, `examples/bible_wave.rs`).
- **The fabric, one level down:** `E-GRAMMAR-TREE-IS-POINTER-FABRIC-1`,
  `.claude/knowledge/left-corner-grammar-tree-pointer-fabric.md` (the pointer-fabric
  proof + depth ≤ 8 + Escalate-on-offset-overflow).
- **The scarcity inversion that makes self-reasoning possible:**
  `E-LC-SCARCITY-INVERSION-1`.
- **The meaning substrate (Layer 5):** `E-CAM96-DISTRIBUTION-MEASURED-1` +
  correction `E-CAM96-REVIEW-CORRECTIONS-1` (held-out, algebraic-not-fidelity).
- **The stream / no-bundle law (Layer 1):** `E-MARKOV-TEMPORAL-STREAM-1`,
  `E-NO-BUNDLE-STANDING-WAVE-1`, `E-HORIZON-NOT-BOUND-1`.
- **NARS introspection (Layer 4):** `E-MULTIHOP-WITNESS-CONFIDENCE-1`,
  `E-SELF-CORRECTING-KG-1`; `witness_table` / `causal_edge::CausalEdge64` /
  `NarsTables::revise`.
- **The runaway Layer 3 fences (Layer 3 / D-SRS-2):**
  `TD-INFER-DEDUCTIONS-RELATION-BLIND`, `E-SELF-DIRECTED-GRAPH-1`.
- **Iron rules that bound scope (§3):** `I-VSA-IDENTITIES`, `I-NOISE-FLOOR-JIRAK`,
  the CANON minimal-SoA node + zero-fallback ladder (reserve-don't-reclaim).
- **Consumes the self-measurement:** `lance-graph-planner/mul/` (Dunning-Kruger /
  trust qualia / homeostasis) — the existing sink for Layer 5's signal.
