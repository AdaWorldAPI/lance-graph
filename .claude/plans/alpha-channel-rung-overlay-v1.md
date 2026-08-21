# Alpha-channel rung overlay — v1

> **Status:** PROPOSED. No code. Register-before-code, per the dialectic-engine
> build order. Every "exists" claim below was verified by reading the file
> named, this session; every "absent" claim by a grep that returned nothing.
>
> **What this plan is:** the design for the ONE empty row of the thinking
> table — `hhtl-thinking-tables-le-contract-v1.md` §2.3, row **Rung ladder**,
> *"(unassigned) · see §3 — two axes · unminted, undesigned"*. It mints no new
> type. It is the scraping of an operator brainstorm (2026-08-21) onto homes
> that already exist, plus a short list of what genuinely does not.

## §0 — The operator's picture, in his own frame

A Photoshop **alpha channel**: an ephemeral layer laid 1:1 over the ontology,
carrying the residue of a search — *"du weißt, dieses residue ist bei der
Suche von Patient X zum Graphen übergesprungen"*. Consequences he named, each
of which turns out to be a constraint rather than a metaphor:

1. **"Alpha channel damit der Graph nicht von Patienten kontaminiert wird"** —
   the overlay is written; the graph is not.
2. **"Eye tracking der Ontologie-Gedanken"** — record WHERE the eye looked,
   not what it saw. *"Müssten wir alles im Patienten reinschreiben, würde es
   nicht mehr mit vertretbarem Aufwand gehen."*
3. **"Gedanken 2. Ordnung als thinking about thinking als graph overlay mit
   der gleichen Adresse wie der Graph aber separate thinking tables."**
4. **"Rung levels 2–10 als Alpha layer projizieren."**
5. **"Eine Maske über die Aktivitäten."**
6. **"Der Rung über dem Rung hat ein Bewusstsein über focus of attention — wo
   die Gedanken davor waren."**
7. **"Eye tracking ist kein Beweis, aber damit löst du das needle-in-a-haystack
   Problem"** — the overlay is a *pruner*, never a proof. (Same grade
   `ONTOLOGY_BAKE_STATE.md` already gives HHTL: *"a sound pruner but not a
   proof"*.)

## §1 — The scraping: nine pieces, six already have homes

| # | Brainstorm piece | Home | State (verified) |
|---|---|---|---|
| A | separate thinking tables at the graph's own addresses | HTT §2.3, row **Rung ladder** | the empty slot — *this plan* |
| B | rungs 2–10 as alpha layers | HTT §3 rung carve (two axes) + dialectic **V3** field elevation (S11) | designed, unbuilt |
| C | **mask over activities** | `PhaseCensus`, `lance-graph-supervisor/src/kanban_actor.rs` | **SHIPPED** — read-only, one `&self` pass |
| D | **focus of attention** | `RowFocusMask` (STATUS_BOARD S3.1b) | **named on the board, ABSENT from code** — grep hits only `STATUS_BOARD.md` + one handover |
| E | eye-tracking residue carrier | `cognitive-shader-driver/src/attention_mask.rs` + `attention_mask_actor.rs` | shipped; **unaudited for this use** |
| F | sudoku elimination as the search | dialectic §3 five tactics + `ReasoningGap`/`GapKind`, `lance-graph-planner/src/nars/tactics.rs` | **SHIPPED** (V1, `E-DIALECTIC-V1-TACTICS-IN-PLANNER-1`) |
| G | contamination boundary (one-way) | — | **design gap**, §2 |
| H | second-order overlay at the same address | HTT §2.3 + G | **design gap**, §3 |
| I | 64k parallel rungs 1–10 | dialectic **V4** (64k SIMT lowering) | explicitly gated: *"only after V0–V3 green at small scale"* |

**Six of nine already exist or are already planned.** The plan's real content
is D, G and H, and D is the only one that is a missing *primitive*.

## §2 — The contamination boundary is the ownership rule, not a new guard

*"Damit der Graph nicht von Patienten kontaminiert wird."* The substrate
already enforces exactly this, and it is worth stating so nobody builds a
second mechanism:

- **One writer per mailbox** (`E-AGENT-LOG-SHARED-SINK-ANTIPATTERN-1`, and the
  runtime original `SoaEnvelope::mailbox_owner` + the write-on-behalf iron
  rule).

> **⊘ CORRECTED (operator, 2026-08-21): TWO writers, and separate tables for
> the ontology.** A first draft of this section said "the overlay is a tenant
> whose owner is the session mailbox", singular — over-restrictive, and it
> mistook *one writer per mailbox* for *one writer overall*. The design is
> **two tables at the same addresses, each with its own owner**:
>
> | table | writer | lifetime |
> |---|---|---|
> | **ontology thinking table** | the ontology mailbox | durable, shared, cacheable |
> | **session overlay** | the session mailbox | ephemeral, per-search, discardable |
>
> This is not a weakening of the rule — it is the rule applied twice. Each
> table has exactly one owner; neither can write the other's rows. The
> contamination guarantee is unchanged and is now *structural in the table
> split* rather than in a singular-owner assumption.

**The invariant, stated once:** *the overlay reads the graph; the graph never
reads the overlay.* One direction, compile-checkable at the owner, and it is
what makes rung-n+1 safe to compute from rung-n residue without the residue
becoming evidence.

> **CORRECTED (CodeRabbit review, lance-graph #978, 2026-08-21).** Two
> owners writing two tables establishes only that the session mailbox cannot
> DIRECTLY mutate an ontology row. It does not, by itself, stop a
> session-derived VALUE from being handed to the ontology-mailbox owner, who
> then writes it as its own act. Mailbox ownership is a write-authorization
> boundary; the invariant this section claims is an INFORMATION-FLOW
> boundary, and those are not the same property -- a correct write-
> authorization check can sit downstream of a completed contamination. Real
> shape: an overlay computation f(patient_residue) = confidence_delta,
> threaded through a shared parameter or return value into a call the
> ontology owner makes on its own initiative. No direct write occurred; the
> graph moved anyway.
>
> Section 4's D-ACR-3 is corrected to match: the test must show no
> ontology-owned WRITE traces to a patient-tagged READ through any call
> path -- not merely that the session mailbox cannot author the write. The
> prohibited path must be named, not inferred from "the owners differ."

> **Consequence that must not be lost:** this is also why the overlay may be
> **discarded whole**. It is not a cache of derived truth (which would need
> invalidation); it is a *record of where attention went*. Dropping it costs a
> re-search, never a correctness question.

## §3 — Second-order = same address, different table

*"Gleiche Adresse wie der Graph, aber separate thinking tables."* In the
contract's own vocabulary this is not a new addressing system — HTT §2.2
(**⊘ WITHDRAWN**) already ruled that out: *one fabric, several
ClassView-resolved readings, never alternative addressing systems.*

So a second-order thought at node `g` is:

- the **same** `NodeGuid` (the address is shared, that is the point);
- a **different thinking-table row** — a different `(classid, rail)` pairing
  resolved by the ClassView, per §2.3;
- occupancy **sparse** — the eye-tracking argument is exactly that only the
  visited addresses are materialised. "1:1" names the *addressing*, never the
  occupancy.

## §3a — Thin rows, fat concepts, and where each is allowed

Operator, 2026-08-21: *"We want to avoid fat concepts in every soa. Yet fat
concepts in HHTL higher-order thinking need to allow stacking concepts."*

Two regimes, and the boundary between them is the whole rule:

| | atomic SoA row | higher-order HHTL node |
|---|---|---|
| concept carried | **by reference** — the row names a witness | **inline, and STACKABLE** |
| why | a fat concept in every row multiplies the fabric by the concept's size | the higher-order node is the ONE place the concept is materialised |
| mechanism | `WitnessEntry` / `WitnessTable<N=64>` / `WitnessLens<'a>` | the concept's own row |
| alternative | `WideFieldMask` — *the field mask for thoughts*: which thought-fields participate, without naming a concept at all | — |

**⊘ CORRECTED (operator, 2026-08-21): these are THREE unrelated witness
surfaces, not one family, and the plan's own table papered over that.**

| surface | files | maturity | is it AriGraph? |
|---|---:|---|---|
| `WitnessLens<'a>` / `WitnessTable<N=64>` / `WitnessEntry` (`witness_table.rs`, `witness_fabric.rs`) | 11 / 9 / 8 | wired, generic register-slab machinery | no — domain-agnostic |
| `CausalWitnessFacet` (`causal_witness.rs:201`) | **18** | **wired and heavily consumed** — `lance-graph-planner::nars::meta_basin`, `style_strategy`, `dispatch_guard`, and `deepnsm-v2::wave.rs`'s versioned event window (`push`/`window_at`/`window_range`) | no — a 12-byte register of loci offsets, addressed by `(Locus, i8)`, never episodic memory |
| **`EpisodicEdges64`** (`episodic_edges.rs`) | **10** | **THE real, current answer.** `(family, local)` **nibble**-structured edge refs (4 × u16, each `[nibble: family][12 bits: local]`) — grounded by locality probe #444, 98.6%/1.4% real cost model. Added 2026-07-23. **NOT mounted as a `ValueTenant`** — zero hits in `canonical_node.rs`'s list | the value shape exists; the row-slot that would make it a per-row-queryable field does not |
| **`EpisodicBasins`** (`arigraph/episodic.rs:79`) | **2** | X3 confirmed: definition + `mod.rs` only | yes — the cold-path AriGraph type |

**My previous entry mis-cited the CV3 rebase report's "write-empty" finding
against `CausalWitnessFacet` — wrong target.** `CausalWitnessFacet` is not
write-empty; it is the opposite, the most-consumed witness type in this
sweep (18 files). The write-empty / EXISTS-UNCALLED finding belongs to
`EpisodicWitness64`, which is not merely unwritten — it does not exist as
Rust at all yet.

**The AriGraph relationship, as the contract's own comment states it**
(`soa_view.rs:262-270`): *"EpisodicWitness64 IS AriGraph living in the
mailbox SoA view."* AriGraph today lives only in the cold path
(`lance-graph::graph::arigraph`: `episodic` / `witness_corpus` /
`triplet_graph`); EW64 would be that same graph promoted to a hot-path SoA
column — the `CausalEdge64` W-slot to witness arc. `E-ARIGRAPH-IS-AN-ISLAND`
names the gap directly: *"EW64/SpoWitness64 = 0 code symbols; the
Lance→surreal→kanban subscription unbuilt; `HotWitness` = `todo!()`."*

**⊘ CORRECTED, operator, 2026-08-21: "Episodic edges = nibbles. Episodic
Witness [war] vorbelastet, weil es früher den CausalEdge64 vom Witness fett
kopiert hat. Jetzt können wir die Vorbelastung ignorieren."** No further
hedging needed: `soa_view.rs`'s `"EpisodicWitness64"` named an EARLIER,
abandoned design that copied `CausalEdge64`'s witness fields wholesale —
that approach stalled, which is why the comment still reads "not yet a
code symbol." `EpisodicEdges64`'s nibble-`(family, local)` encoding is the
real replacement that shipped instead, at 10 real call sites, and there is
no live identity question left to resolve. The one remaining real gap is
purely mechanical: `EpisodicEdges64` has no `ValueTenant` mount.
`D-ACR-17` (below) closes exactly that, using the type that already
exists.

**And `bible_wave.rs` touches NONE of the four** — verified: its imports are
entirely internal to `deepnsm_v2`, zero references to
`CausalWitnessFacet`/`EpisodicBasins`/`witness_corpus`/`arigraph`. §3d's
citation of it as "the whole-book falsifier" is accurate for what it
measured (HHTL cascade coverage over Markov trajectories), but it is NOT
evidence about episodic-basin prestaging, and §3d is corrected below to stop
implying that.

`WitnessLens` is a **lens**, which is the shape this needs: the row borrows the
concept, it does not own a copy. That is the zero-copy law and
`E-A-CHIP-BEARS-LOAD-OR-IT-IS-A-JOKE-1`'s *pointers-to-evidence, never cached
conclusions*, arriving at the same answer from two directions.

## §3b — What CE64 59..63 grades, and what it does NOT

Operator, 2026-08-21: the thinking styles, the potholes and the HHTL nodes must
**agree on how to read 59..63**, so that these stay distinct:

> episodic-witness basins · epistemic knowledge · causality · *supporting*
> causality · *just related to*

Read against the layout, that list is **two orthogonal axes**, not one:

**Axis 1 — strength of the claim.** Already carried, already 3 bits:
`ReasoningBand` = `Surface(0) · Association(1) · Relation(2) · Causal(3) ·
Counterfactual(4) · Perspective(5) · Meta(6) · Transcendent(7)`. *"Just related
to"* is `Association`; *"supporting causality"* sits at `Relation`; *"causality"*
is `Causal`. The ladder exists and needs no new bit.

**Axis 2 — KIND of evidence.** *Episodic-witness basin* vs *epistemic knowledge*
is not a strength at all — a weak episodic witness and a weak epistemic claim
are the same band and different things. **CE64 has nowhere to put it**: 59..63
is `TRUTH_SHIFT`(59–60) + `SPARE_SHIFT`(61–63) and the board already records the
count for the sibling case — *"`awareness_state` orthogonal to `unknown_kind`;
NOT in CE64 (0 free bits)"* (D-CV3-3).

**So the alignment is:** the **band grades**; the **witness reference
discriminates**. Which carrier a row points at — an episodic basin, a
`WitnessTable` entry, a `CausalWitnessFacet` — IS the evidence-kind axis. No
bit is minted, and §3a's "reference, don't inline" is what makes that free.

**Three fences, each because the opposite is the attractive move:**

1. **Nothing derives the band.** `layout.rs` states it is set ONLY by an
   explicit `with_reasoning_band()` call — not from `CausalMask`,
   `InferenceType`, NARS, MUL, `ReasoningGap`, potholes or `ThinkingStyle`. An
   overlay that infers a band from a pothole has become the fifth derivation
   path the layout refuses.
2. **`ReasoningBand` is never `RungLevel`.** Unrelated enums sharing four
   variant names at different ordinals.
3. **`TrustTexture` and `CausalTopology` are the same 2 bits read differently.**
   A consumer that reads one while a producer wrote the other gets a plausible
   wrong answer, silently. Whichever this overlay uses must be named per
   `(classid, rail)` in the thinking table, not assumed.

**Operator, 2026-08-21: "Und die Abhängigkeit, wann wird episodic zu
epistemic."** Already ratified, not something to design —
`EPIPHANIES.md:5194`, *"the honesty line: episodic vs epistemic
causality"*:

> Episodic = what happened, in stream order (Lance versions, `temporal.rs`,
> "A preceded B"). Epistemic = what grounds belief (the warrant chain,
> "A is why B is held"). Recency IS episodic proximity deployed as an
> epistemic proxy — and W7's divergence gates measure precisely where that
> proxy lies (3:1: the episodically-nearest candidate is not the epistemic
> ground). The witness register's offsets are episodic ADDRESSES carrying
> epistemic MEANING; conflating the two axes is the category error this
> ruling fences.

**With a live example in exactly this plan's own KJV territory**
(`E-WHOLE-BOOK-REASONING-RUN-1`'s caveat): *"`is_a(god, light)` stores
episodic adjacency wearing epistemic inheritance's label"* — a real,
measured instance of temporal/textual proximity being mistaken for
ontological grounding, on the same corpus §3d discusses.

**Where `D-ACR-12` sits relative to this fence, checked rather than
assumed:** `D-ACR-12`'s ascent walks `NiblePath` ancestry over MONDO/UBERON
— a TAXONOMIC `is_a`/`part_of` tree, where a parent class genuinely,
definitionally grounds its children (that IS what ontological subsumption
means). It is not walking episodic/temporal proximity, so it does not
commit the fenced error by construction. But the fence names the exact
trap the NEXT session must not fall into: an ascent that instead walked
`temporal.rs`'s version stream, or a `WitnessLens` reference chosen by
recency, would be doing precisely what `is_a(god, light)` did — dressing
episodic address in epistemic clothing. `D-ACR-12`'s falsifier gains a
fourth condition: the ancestor a lookup resolves to must be reachable via
`is_ancestor_of` (address containment), never via version proximity to the
querying row.

**Operator, same message: "potholes als 59..63 using deduction
extrapolation syllogism counterfactual synthesis inference induction."**
Seven named mechanisms. Checked against what is already at those bits and
nearby, this does not get a new field:

- **It collides with `RecipeInference`, already shipped at a DIFFERENT bit
  position.** §3g/§3f already established: `inference_mantissa` (CE64 bits
  46-49, signed i4) carries `RecipeInference::{Deduction, Induction,
  Abduction, Revision, Counterfactual}` — 5 of the operator's 7 words
  already map onto it directly (`Deduction`, `Induction`, `Counterfactual`
  are literal matches; `syllogism`/`inference` are the general term-logic
  mechanism `RecipeInference` already IS; `extrapolation`/`synthesis` are
  closest to `Induction`/`Revision`'s existing job). Writing a NEW 7-value
  vocabulary into the CE64 SPARE bits (61-63, 3 bits — not even enough
  room for 7 values without waste) would duplicate a field that already
  exists 13 bits away, and would be the fifth-derivation-path fence one
  finding names again: 59-63 is graded, never mechanism-tagged.
- **It is NOT the same "7" as `bgz-tensor`'s `Zipper7LevelDescriptor`, and
  routing it there would be a homonym collision, not a fit.** Verified:
  the zipper's *"7-level bipolar zipper — 7^7 = 823,543 states"* is 7
  **discrete MAGNITUDE levels** (`{-3,-2,-1,0,1,2,3}`) per **continuous
  embedding sample** — `encode(row: &[f32], k, …)` quantizes a real
  vector's WHT-rotated, phase-strided values. There is no natural mapping
  from a WORD like `"syllogism"` to a sample position; the zipper composes
  and bundles vector magnitude, it does not carry discrete category
  identity at all. The two "7"s are unrelated by count alone, the same
  trap as `NiblePath`/anaphora-nibble/TEKAMOLO/`morton()` in §3k's table —
  now a fifth confirmed instance this session, not a fourth.

**Verdict: the 7-mechanism vocabulary's home is `RecipeInference` at
CE64 46-49, already shipped, needing no new deliverable — and this closes
out `D-ACR-7`'s reading-contract acceptance condition with a concrete
answer rather than leaving it open.**

## §3c — Versioning is temporal, not a stored field

Operator: *"Versioning using temporal."* `QueryReference::at(ref_version, rung)`
exists (`temporal.rs:188`), and the ruling `E-MARKOV-TEMPORAL-STREAM-1` already
moved the trajectory off the VSA braid onto the sorted stream: a version-range
read plus deinterlace, *"replayable — still a projection, zero copies"*, with
**episodic = Lance versions**.

Consequence for the overlay: a rung-n+1 read of "where did rung-n look" is a
**read at a version**, never a stored history column. That is also the second
reason the residue is discardable — the trajectory is recoverable by replay, so
the overlay caches nothing that replay could not produce.

## §3d — The KJV gap, and why it is blocked rather than merely unbuilt

Operator: *"KJV Bible missing epistemic causality nodes that need to be
prestaged with episodic basins."*

`deepnsm-v2/examples/bible_wave.rs` is the whole-book falsifier that already
ran — for HHTL cascade coverage over Markov trajectories. **⊘ CORRECTED:**
it is NOT a falsifier for basin prestaging, and citing it as one was wrong;
verified this session that it imports nothing from `arigraph`,
`witness_corpus`, `EpisodicBasins`, or `CausalWitnessFacet`. §3a's audit of
the witness surfaces found `bible_wave.rs` touches none of them.

Prestaging missing causality nodes as episodic basins is the **Type-B basin
promotion** seam, and HTT **X3** records that it *does not exist*:
`EpisodicMemory::basins()` (`episodic.rs:243`) returns `EpisodicBasins`
(`:79`) **by value**, with no `ValueTenant` slot reserved. Measured here: the
type appears in **two** files, its own module and `arigraph/mod.rs`.

So this is not "someone should wire it" — a promoted basin is exactly a *new
thinking-table row with no minted rail*, which puts it behind the same mint
decision as D-ACR-2 (HTT §8 Q3). Recorded as `D-ACR-6`, blocked, with its
blocker named.

**Operator, 2026-08-21: "Bei KJV sind episodicwitness als fat concepts in
den SoA."** Read as the design hazard for whoever builds D-ACR-6, not as a
bug report on `bible_wave.rs` today — the file's own `Spo` struct is
already the right shape and is the model to replicate, not the anti-pattern:

```rust
pub type WordId = u16;
pub struct Spo { pub subject: WordId, pub predicate: WordId, pub object: WordId }
```

Three 16-bit indices into the shared `PaletteVocab`/Cam96 codebook — a
reference, per §3a, not an inline concept. `TemporalStream` is
`Vec<(u64, Spo)>`: thin rows, no verse text, no `Vsa16kF32` bundle stored
per-triple.

**The fat-concept failure mode arrives at the PROMOTION step, not before
it.** A basin is a *cluster of witnessed triples*; the naive promotion is to
materialise that cluster's content (full verse text, a per-basin bundle)
inline in the basin row so a reader doesn't have to chase references — which
is exactly what §3a forbids for atomic SoA rows. The correct shape is the
one `Spo` already uses one level down: a basin row holds **references** into
the triple stream (a version range via `QueryReference::at`, §3c) and into
the vocab, never copies of their content. `D-ACR-6`'s acceptance criterion is
extended to say so explicitly: a promoted basin's own row must stay
index-width, with content reached only by following its references — the
same test §3a's `WitnessLens` already passes.

**Operator, 2026-08-21: "Bei der Bibel müssen dagegen erst die episodic
arc generiert werden und die lenses Gadamer Horizontverschmelzung usw
erkennen dann logische Verknüpfungen."** — and further: *"Hermeneutik als
logische Verknüpfungen"*, *"muss ggf als causality mechanical drin
stehen."*

This draws the line D-ACR-6 was missing: **KJV is not a rail-ancestry
problem at all.** Ontology trees (MONDO/UBERON, §3k below) have parents
because the domain is taxonomic; a book has no taxonomy to ascend — its
epistemic-causality nodes have to be **generated**, not inherited, by
reading the text. Two different mechanisms, two different sections; folding
them together would have been the same conflation §3a's four-witness-surface
correction just fixed.

**And the generation mechanism is not a gap — it is SHIPPED, verified this
session:**

1. `bible_wave.rs`'s own comment (lines 96-99) already states the seam
   precisely: the reasoning layer's stance panel *"needs verse TEXT, not
   triples"* — `stance::stream()` mints rung-lifts inside a complementizer
   window and reads negation polarity from the clause, and *"3 of 4 stances
   measured UNREACHABLE"* on the flat `(s,p,o,verse)` export. Text is now
   emitted as its own artifact for exactly this reason.
2. `lance-graph-planner/src/nars/stance.rs::stream()` is the cue-driven
   clause machine — pronoun-normalization, negation/modal/causal-cue
   catalogues — whose `ReadOut` carries `pass2_admitted`/`pass2_revised`
   counts *"the hermeneutic-circle termination check uses"* and, directly
   answering the operator's causality demand:

   ```rust
   /// Causal edges observed from `because`-cued text, as (verse, cause, effect).
   pub impls: Vec<(String, u16, u16)>,
   ```

   **This is Hermeneutik AS mechanical causality, already in the type.**
   Not a metaphor needing translation — `impls` is a plain `(verse, cause,
   effect)` triple, the same shape as every other causal record in this
   substrate.
3. **Horizontverschmelzung is not a design, it is `D-BLW-3`, SHIPPED +
   MEASURED 2026-08-04** (`examples/blw_fusion.rs`, board `STATUS_BOARD.md`
   line 336): two rank projections read under `Strict` (a-priori) vs
   `Aware` (hindsight), κ movement measured (0.49/0.46 IN/IN bands), the
   8-horizon table showing the gap **closing monotonically** rather than
   staying flat. Its own KILL condition was *"flat kappa regrades the claim
   to four independent stance reads — not Gadamer"* — and the KILL did not
   fire. The fusion is real, not decorative.

**Consequence for this plan: no new deliverable for KJV hermeneutics.** The
Gadamer/hermeneutic-circle mechanism the operator asked for already exists,
already measured, already named honestly (its own KILL condition was
falsifiable and survived). What remained open was only the promotion-format
question §3d already answers above — and that is an ontology-shaped worry
that does not apply to `stance::stream()`'s output at all, since `impls` is
already triple-width, never a fat concept.

## §3e — Rubicon: focus of attention is what witnesses the crossing

Operator, 2026-08-21: *"The Rubicon model Heckhausen in kanban needs to use the
rung 1-10 alpha channels to follow the thinking via focus of attention."*

This is a **join between two existing plans**, not a new idea.
`unified-soa-rubikon-integration-v1.md` §3 already maps the Heckhausen action
phases onto the kanban columns, Libet-anchored (`kanban.rs:25-49`):

| Heckhausen phase | Kanban column | Libet anchor |
|---|---|---|
| Predecisional (weighing) | `Planning` | spawn |
| **Rubicon crossing** (Σ-commit) | `Planning → CognitiveWork` | −550 000 µs ✅ |
| Preactional + actional | `CognitiveWork → Evaluation` | 0 |
| Postactional (evaluation) | `Evaluation → {Commit \| Plan \| Prune}` | 0 |
| Libet veto ("free won't") | `Planning → Prune` (pre-Rubicon) | ☐ −200 000 µs (PROPOSED) |

…and that plan carries an **open checkbox**: *"☐ Thinking styles ↔ Rubikon"*.
The alpha channel is what closes it, because the thing Heckhausen's model
actually asserts about the crossing is a claim about **attention**:

> **Pre-Rubicon = deliberative mindset:** open, broad, impartial — many
> candidates held at once, feasibility still negotiable.
> **Post-Rubicon = implemental mindset:** narrow, partial, *shielding* — the
> intention is protected against reconsideration.

A focus mask is a direct measurement of exactly that. So the rung 1–10 channels
do not merely *accompany* the phases; **they are the instrument that can falsify
the phase labels.**

### The falsifier (`D-ACR-8`), two-sided by construction

- **Can-fire:** focus masks sampled in `Planning` must be measurably **broader
  and less persistent** than those sampled in `CognitiveWork` on the same task.
  If they are indistinguishable, then either the kanban columns are decoration
  or the alpha channel is not recording attention — and **both of those are
  findings worth having**, which is what makes this test worth its cost.
- **Can-stay-silent:** on a task with **no real deliberation** (a single
  forced candidate — nothing to weigh, so no mindset shift to observe), the two
  phases must **not** differ. Without this half the discriminator would fire on
  every task and carry exactly as much information as one that never fires.

Both halves need `RowFocusMask` (`D-ACR-1`), so this queues behind it.

**What this does NOT license.** The measurement witnesses *which side of the
Rubicon the thinking is on*; it does not move anything across. Phase movement
stays `advance_on_gate`, and the standing tombstone applies —
`E-PROGRESSION-IS-EXISTENCE-NOT-COMMAND-1`: progression is existence, not
command, and per-owner `advance(owner)` RPCs are the deleted actor shape. An
overlay that *drives* a phase transition from a focus reading has rebuilt the
scheduler this substrate removed. It reads; `PhaseCensus` already reads
read-only in one `&self` pass, and that is the whole precedent.

## §3f — ogar-loco executes the 34 recipes; revision runs in the pre-act window

Operator, 2026-08-21: *"ogar-loco kann jetzt die 34,36 nars recipes direkt
verwenden und in −220..0 über Revision die potholes und ggf reasoning with
another style."*

**Both numbers check out, measured this session:**

| | count | where |
|---|---:|---|
| NARS recipes | **34** | `contract::recipe_dispatch::dispatch_order() -> [u8; 34]` |
| ThinkingStyles | **36** | `contract::thinking::ThinkingStyle`, 36 variants |

So *"reasoning with another style"* is a switch among the 36; the 34 are what
each style *runs*. They are not two names for one list.

**`recipe_dispatch` already exposes the pothole machinery**, and the signature
is the tell:

- `rung_delta() -> i16`.
  > **⊘ MY OWN ERROR, CORRECTED SAME DAY.** A first version of this line said
  > *"signed — a negative delta is a rung demotion, precisely the pothole →
  > rung degradation chain."* **False.** The doc states it is the *"escalation
  > depth offset (the ORDER cost, not the mantissa direction): Ded +1 → Ind +2
  > → Rev +3 → Abd +4 → Counterfactual +5"* — **all five values are positive**.
  > `i16` is a type width, not a semantic. I inferred a meaning from a type and
  > wrote it into a pushed plan; it is the same error class this whole arc is
  > about — reading a field whose population does not support the claim built
  > on it. Rung *degradation* is elsewhere and is not identified here.
- `nan_disqualifier(ctx, id) -> Option<ThoughtField>` — fail-closed.
- **`ladder(ctx) -> Vec<RecipeStep>`** — this already returns a **step
  sequence**, i.e. a program. `ogar-loco` is a program surface where every call
  is `(function : value)`. **`ladder()`'s output IS a loco program**, and that
  is mechanism rather than resemblance: one produces an ordered step list, the
  other executes ordered `(fn : value)` calls over a 256-entry codebook.

**⚠ The wiring does NOT exist yet, and the sentence should not be read as if it
does.** Grep over `OGAR/crates/ogar-loco/src` for `recipe|Recipe|nars|Nars`
returns **no files**. What is true is that loco is *domain-agnostic by design* —
consumers implement `Vocabulary` and mint their own ops above `DOMAIN_FLOOR`,
exactly as `ogar-dismech` did (`SEARCH_OPS`, `0xA3..0xA9`). So a recipe
vocabulary is the natural next impl, and it is **unbuilt**. Recorded as
`D-ACR-9`.

### ⚠ OPEN — the window disagrees with the plan it would land in

The operator's window is **−220..0**. `unified-soa-rubikon-integration-v1.md`
§3 states the Libet veto window as **−550 ms .. −200 ms** and proposes stamping
−200 000 µs on the `Planning → Prune` edge. These are not the same interval,
and they are not reconcilable by rounding:

| reading | interval | what happens there |
|---|---|---|
| the Rubikon plan | −550 .. −200 | veto, *before* the −200 mark |
| the operator | −220 .. 0 | revision over potholes, style switch, *up to the act* |

In the classic Libet paradigm the readiness potential begins ≈−550 ms and
reported awareness of the intention (W) falls ≈−200 ms, which would put a
*conscious* veto **after** W — i.e. in −200..0, the operator's interval —
because before W there is nothing conscious to veto with. That favours the
operator's reading, and would make the plan's −550..−200 the *pre-awareness*
stretch rather than the veto window.

**I am not ruling on it.** I have been wrong twice today asserting structure
from memory, and this is a claim about an experimental paradigm, not about this
codebase. It is recorded as `§8`-class open question: *which interval is the
veto and which is the revision window, and does the −200 000 µs stamp belong on
`Planning → Prune` or somewhere else?* Whoever answers should cite the source,
not recall it.

## §3g — "5 oder 14": both numbers are right, for two different criteria

Operator, 2026-08-21: *"nur 5 oder 14 von 34(36) sind bereits voll verdrahtet,
die anderen haben keinen trust value oder Verlässlichkeit."*

**The number is written in the source, and it settles the ambiguity.**
`recipe_kernels.rs` (the doc heading the `delta_conf` capability method):

> *"Measured over the 34: **no** kernel declares `ThoughtField::Confidence` in
> `writes`, and only **14** can move `delta_conf` — while 31 are
> `Operational`. So `maturity().is_production()` is a far weaker statement than
> 'this tactic can move the confidence number', and a caller that needs the
> latter must ask for it."*

So the ladder, each rung measured this session:

| criterion | count | what it means |
|---|---:|---|
| have a kernel | **34 / 34** | `all_kernels() -> [&dyn Tactic; 34]`, macro-generated — every recipe executes |
| self-declare `Operational` | **31 / 34** | per-impl `maturity()`; 3 are `Demonstration` (`Are`, `Zcf`, `Hkf`) |
| **can move `delta_conf`** | **14 / 34** | the source's own measured count |
| **route through real NARS truth functions** | **5** | and **not in this crate** — see below |

**The 5 are in a different crate, and that is the structural half of the
finding.** Measured: **0 of the 34 contract kernels reference `TruthValue` at
all**. The truth functions live in `lance-graph-planner/src/nars/tactics.rs`
(33 `TruthValue::` uses — `deduction` 1, `induction` 1, `abduction` 2,
`revise` 2, `analogy` 2), which is the V1 shipping location
(`E-DIALECTIC-V1-TACTICS-IN-PLANNER-1`, the five RCR/TR/ASC/CAS/CR). **There
are two tactic surfaces**: a 34-kernel contract catalogue and a 5-tactic
planner engine over the one truth algebra. A plan that says "the 34 exist"
without saying which surface invites building on the wrong one.

**The failure mode is already named in-tree**, which is why this matters more
than a maturity label:
`E-A-WATCHER-THAT-CANNOT-DISSENT-IS-NOT-A-WATCHER-1` — a dispatch that samples
`k` tactics and asks whether any moved `ctx.confidence` spends a slot on every
sampled tactic that *cannot* move it, and gets guaranteed agreement back. The
source cites the live instance (codex, PR #971): the newly-`Operational` `Etd`
rewrites `candidates` and returns `0.0` forever. **A maturity filter does not
close this; only the capability question does.**

**Consequence for this plan, and it is a hard one:** any overlay deliverable
that samples tactics must filter on **`delta_conf` capability**, never on
`maturity().is_production()`. §3b's grading axis is worthless if the tactics
feeding it cannot move a confidence number — the band would be set by
unanimity among mutes. Added as an acceptance condition on `D-ACR-7`.

## §3h — MUL over the rung layers (long-term)

Operator: *"long-term sollte MUL (meta uncertainty layer) auch mit den rung
layers und den epistemic knowledge vs potholes verkabelt werden über denken und
kanban_actor higher order thinking."*

`mul` is shipped (`contract::mul` — `SituationInput`, `MulAssessment`,
`DkPosition`, `TrustTexture`, `FlowState`, `GateDecision`; planner `mul/` with
Dunning-Kruger, trust qualia, compass, homeostasis, gate). It is a
**meta**-layer, which is the same tier as §3's second-order overlay — so the
join is not new plumbing but a question of *which* column each writes.

Recorded as long-term, deliberately without a deliverable id, because it
depends on all three of: `RowFocusMask` (D-ACR-1), the 59..63 reading contract
(D-ACR-7), and the `delta_conf` filter above. Sequencing it before those would
wire a meta-layer onto an axis nobody has pinned yet. Note also that
`TrustTexture` appears on **both** sides — as a MUL output and as one of the
two 2-bit readings of CE64 59..60 — so the §3b fence ("the reading must be
named per `(classid, rail)`") is load-bearing exactly here.

## §3i — Two filters the operator named, graded honestly

**`temporal.rs` as a hindsight-tautology filter.** *(plausible, unbuilt)*
`QueryReference::at(version, rung)` makes "was this derivable **at** v?" a
question the substrate can actually answer, which is what turns hindsight into
something falsifiable rather than rhetorical. The board already carries the
matching pair — S3.8: *"potholes, **first_possible vs first_derived**, strict
historical replay"*. A claim whose `first_possible` equals the beginning of the
stream is derivable at every version and therefore discriminates nothing. This
is a real use of a shipped primitive and needs no new type; it needs a probe.

**NARS frequency as an eigenvalue filter.** *(CONJECTURE — analogy until
measured)* The shape: a claim whose frequency `f` is unmoved by revision across
contexts is a fixed point of the revision operator, and a fixed point carries no
information about the context — which is what a tautology *is*. Combined with
the filter above, "always derivable" and "never moved" would be the same
property seen from two sides.

**Split verdict, because the operator's follow-up cut it in two.**

> *"Eigenvalue ist für MUL meta uncertainty Layer dunning kruger
> overconfidence sicher auch ein Thema."*

That is the stronger half, and it is **not** an analogy — it is Dunning-Kruger's
own definition, operationalized. Justified confidence *moves* when disconfirming
evidence arrives; **overconfidence is a confidence value that is a fixed point
under evidence.** "Eigenvalue 1 under the revision operator" and "overconfident"
are then the same statement, and MUL already has the carrier (`DkPosition`).

**And §3g already measured a piece of that spectrum without calling it one.**
`delta_conf` capability is exactly confidence-invariance: **20 of the 34 tactics
cannot move a confidence number at all**, so their confidence output is
invariant under every input — eigenvalue exactly 1, by construction, not by
measurement error. Which makes
`E-A-WATCHER-THAT-CANNOT-DISSENT-IS-NOT-A-WATCHER-1` and Dunning-Kruger **the
same phenomenon at two levels**: a watcher that cannot dissent, and a confidence
that cannot be lowered. That is mechanism, and it is why this half is
probe-ready today: perturb the input, see whose confidence does not move.

**The FREQUENCY half stays CONJECTURE.** "Tautology = fixed point of `f`" is
still doing metaphorical work until someone shows revision is linear enough for
a spectrum to mean anything, and `I-NOISE-FLOOR-JIRAK` is the fence: under weak
dependence the naive statistical reading is wrong, and an eigenvalue argument is
a statistical reading. **Falsifier before promotion:** take claims with measured
near-constant `f` across contexts and show they are *independently* judged
uninformative — if high-`f` stable claims turn out to be the load-bearing ones,
the analogy is inverted and dies.

The asymmetry is the useful part: the confidence half is measurable with what is
already in the tree; the frequency half needs an argument nobody has made.

## §3k — HHTL nodes as a materialized meta-connection SoA, and the missing inheritance

Operator, 2026-08-21: *"Die Idee wie gesagt HHTL nodes als Meta Verbindung
in eigene SOA zu materialisieren. Das was bei Ontologien über rails bereits
implicit ist. Aber die Vererbung fehlt (unsere Aufgabe, epistemic knowledge
from parents)."*

This is the sharpest form yet of §1 piece A / the HTT §2.3 **Rung ladder**
row this whole plan exists to fill — and it names the exact primitive that
is missing, verified this session:

**The ascent primitive exists and has ZERO callers.**

```rust
// hhtl.rs:155
pub const fn parent(self) -> Option<Self> {
    if self.depth <= 1 { None } else { Some(Self { path: self.path >> 4, depth: self.depth - 1 }) }
}
```

Grepped across the tree: every `.parent()` call site found is either
`std::path::Path::parent` (unrelated) or `holograph::dn_sparse::DottedName`'s
own `parent`/`ancestor`/`ancestors` — a **different type**, real prior art
for the SHAPE, not a shared implementation:

```rust
// dn_sparse.rs:314 — "the vertical traversal operation", O(depth), no scanning
pub fn ancestors(self) -> Vec<Self> {
    let mut result = Vec::with_capacity(self.depth() as usize);
    let mut current = self;
    while let Some(p) = current.parent() { result.push(p); current = p; }
    result
}
```

`NiblePath::is_ancestor_of` (`hhtl.rs:176`) is the ONLY consumer-facing use
of the ancestry relation today, and it answers a yes/no question — *"is A
an ancestor of B"* — never *"what does the nearest ancestor with content
know."* That second question is the missing inheritance, and it is
mechanically the ascent loop above with an early-exit the moment a row has
content:

```text
fn epistemic_lookup(addr: NiblePath, read: impl Fn(NiblePath) -> Option<Witness>) -> Option<(Witness, u8 /* hops */)> {
    let mut current = Some(addr);
    let mut hops = 0;
    while let Some(a) = current {
        if let Some(w) = read(a) { return Some((w, hops)); }
        current = a.parent();
        hops += 1;
    }
    None
}
```

**Why this is exactly "materialize the meta-connection in its OWN SoA" and
not a change to the ontology rows.** The lookup above never touches an
ontology row's own content — it reads the KEY (`NiblePath`, zero value
decode, per the canon's own P0) and walks it. The **result** — which
ancestor answered, at what hop-distance — is what gets materialized as the
Rung-ladder thinking-table row (§2.3), so a repeated query does not re-walk
the chain from scratch. This is `§3a`'s reference discipline applied one
more time: the meta-connection row holds `(source_addr, resolved_addr,
hop_distance)`, never a copy of the ancestor's content.

> **⊘ CORRECTED (operator, 2026-08-21): "Schau Mal in den session
> transcripts wie wir es in lance-graph-java gelöst haben. Where über
> ABI-shaped substrate als masking method."** The ascent-loop pseudocode
> above is written as one address walking hop-by-hop — exactly the
> per-row-crossing shape `lance-graph-java`'s own docs call out as
> *"catastrophic"* (`Predicate.java`'s doc comment, on why a
> `java.util.function.Predicate<Row>` was rejected: *"64,000 objects and
> 64,000 crossings for 64,000 entities"*). That sibling repo already solved
> the SAME shape of problem — one-hop reachability over an ABI substrate —
> and the solution is BULK, not per-address:
>
> ```java
> // RowStore.java:163-168 — measured, pinned by GraphHopTest, 2026-08-18
> public Mask hop(int edgeClassid, WideFieldMask facets, Mask src) {
>     long dst = Engine.createMask(handle, false);
>     Engine.hop(handle, edgeClassid, facets.bits(), 0, src.handle(), dst);
>     return new Mask(this, dst);
> }
> ```
>
> *"Two native crossings, flat … never per-row, never per-frontier-size."*
> `src` and the returned `Mask` are both opaque native population handles the
> whole way through — the CLAUDE.md invariant this implements verbatim:
> **"hop may look like hop; it must execute as `Mask × ClassView/WideFieldMask
> → Mask`."**
>
> **The pattern to reuse for `D-ACR-12`, not the code — `NiblePath` and
> `RowStore`'s `EdgeBlock` facets are different substrates** (this session's
> own lesson: do not conflate two ancestry-shaped things across crates
> without verifying they share a wire, per §3a and §3k's own nibble table).
> What transfers is the SHAPE: resolve the WHOLE frontier of
> childless-witness rows in lockstep, one hop per round over ALL of them at
> once, composed from kernels already shipped in `kernels.rs`:
>
> ```text
> frontier = rows_without_own_witness           // a Mask, the whole population at once
> resolved = empty Mask
> for hop in 0..max_hops {
>     frontier = ascend(frontier)                // one bulk hop, not one call per row
>     found    = frontier ∩ has_witness_mask     // simd_eq_u32_to_mask-shaped
>     resolved = resolved ∪ found                // simd_mask_or_assign — SHIPPED
>     frontier = frontier \ found                 // simd_mask_andnot — SHIPPED
>     if frontier.is_empty() { break }
> }
> ```
>
> `D-ACR-12`'s materialized `(source_addr, resolved_addr, hop_distance)`
> triple (per §3a's reference discipline) is what this loop's `resolved`
> mask carries out at each round — the hop number IS the hop-distance
> column, for free. This does not commit `D-ACR-12` to using `lgj-abi`'s
> literal `hop()` (it operates on `EdgeBlock` facets, not `NiblePath`
> ancestry — an unverified wiring, not assumed here); it commits it to the
> SAME discipline: bulk over the frontier, never per-address, and if this
> ever crosses an ABI boundary the whole chain marshals into one descriptor
> per §3k's kernel-fusion note, exactly as `View.where()` does.
>
> **Governing choice for this whole plan, operator-stated, 2026-08-21:**
> *"Wenn du später in die Verlegenheit kommst VSA-Masken für cascading rung
> awareness zu erstellen, können wir darüber sprechen. Aber ich wäre für
> explizite masking ABI traversal wie bei java. Sonst verwässern wir unsere
> Architektur."* Every cascading/rung-awareness mechanism this plan touches
> — `D-ACR-1` (`RowFocusMask`), `D-ACR-12` (bulk ascent), `D-ACR-13`
> (coverage gate), `D-ACR-14b`/`D-ACR-16` (cartography, nested kanban) —
> defaults to the EXPLICIT `Mask × ClassView/WideFieldMask → Mask` pattern
> above, never `Vsa16kF32` bundling. This is a scoping choice, not a
> contradiction of `I-VSA-IDENTITIES` (VSA keeps its own narrow,
> already-bounded niche — `.claude/plans/*` and the iron rule stand
> unchanged); it says explicitly that THIS plan's awareness cascade is not
> where that niche gets reached for. If a real need for VSA superposition
> surfaces later, it is a deliberate, separately-opened conversation — not a
> default reached for because a mask felt insufficient.
>
> **This retroactively validates HTT's own X2 withdrawal, not just avoids
> repeating it.** `D-HTT-6` (withdrawn) would have derived PARENTHOOD from
> `morton()` bit-interleave arithmetic directly — a single-address
> coordinate jump. `D-ACR-12` answers the same question — *who is the
> ancestor* — correctly on BOTH axes X2's correction separated: the right
> ADDRESS mechanism (prefix containment via `is_ancestor_of`/`parent()`,
> never coordinate geometry) AND, from this correction, the right EXECUTION
> mechanism (bulk mask algebra over the whole frontier, never a per-row
> coordinate transform). Operator, 2026-08-21: *"Das verbessert vieles von
> den Morton falsch abgebogenen."*

**Scope: ontology trees only, per §3d's correction above.** `is_ancestor_of`
and `parent()` operate on `NiblePath`, which is HHTL's taxonomic/mereological
address space (MONDO, UBERON, the rails). KJV's causal edges (`stance.rs`'s
`impls`) are not addressed this way and do not inherit through it — the two
mechanisms stay separate, as §3d now states explicitly.

**"Nibble" is a unit (4 bits), not a namespace — FOUR unrelated encodings
share the word, verified this session, and `D-ACR-12` touches exactly one:**

| # | where | what it is | relation to `D-ACR-12` |
|---|---|---|---|
| 1 | `NiblePath` (`hhtl.rs`) | **16-nibble ABSOLUTE tree address**, `parent()`/`is_ancestor_of` walk it | **this is the one** |
| 2 | `edge_v3.rs` anaphora nibble (`E-NIBBLE-ANAPHORA-EDGE-1`) | ONE signed `i4` (`−8..+7`) **RELATIVE** coreference offset — a pronoun-to-referent pointer, byte `[6]` low nibble of `CausalEdgeV3` | unrelated: relative grammar offset, not a tree address |
| 3 | TEKAMOLO carving (`edge_v3.rs` header, `grammar::tekamolo`) | Temporal/Kausal/Modal/Lokal role slots, bytes `[10..12]` — **"reserved (dormant)"** | unrelated: grammar role encoding, not addressed at all yet |
| 4 | `Facet::morton()` (`facet.rs:62`) | bit-interleave of two bytes into one `u16` — HTT **X2**: *"non-canonical research... nothing in the HHTL contract depends on it"* | unrelated by explicit prior ruling — **do not let `D-ACR-12` become its second consumer** |

Operator, 2026-08-21, naming exactly this risk: *"Nibble ist in grammar
heuristics Relativpronomen anaphora pointers und tekamolo"* and *"Aber
nibble als Morton wäre ein parallel Universum."* Four homonyms, same word,
each its own contract — the same discipline §3a already had to apply to
four unrelated "witness" surfaces. `D-ACR-12` is scoped to row 1 alone; rows
2-4 are named here so a future session does not rediscover the collision or
accidentally wire `D-ACR-12`'s inheritance walk through the wrong one.

**`D-ACR-12`.** Gates on `D-ACR-1` (`RowFocusMask`, so a lookup's ascent
path is itself recordable as an overlay trace) and gets its own falsifier —
now a THIRD condition added by the mask-native correction above:
a child with no witness of its own must resolve to its nearest ancestor's,
at the correct hop count; a child that HAS its own witness must never
ascend past it (an eager-ascent bug would silently prefer a stale ancestor
over fresh local knowledge — the can-stay-silent half); and the whole
frontier must resolve in **at most `max_hops` bulk rounds**, never in a
per-address loop — a benchmark scaling linearly with population size on a
fixed tree depth is the mask-native discipline failing, not merely slow.

## §3l — Book hydration as a checked precondition: NOT BUILT, and no near-miss quite fits

Operator, 2026-08-21 (brutal-honesty request): *"in ogar-loco die
Hydrierung von Knoten bei Büchern als fest maskierte awareness mit
kanban_actor über HHTL masking. Dann kann die Massen-Hydrierung als
Voraussetzung für higher-order thinking fest geprüft werden."*

Checked three places, brutally, rather than designing on top of an
assumption:

1. **`ogar-loco` (OGAR): ZERO.** Grepped every file in
   `crates/ogar-loco/src/` (`lib.rs`, `node.rs`, `pool.rs`, `program.rs`,
   `registry.rs`, `statements.rs`, `telemetry.rs`, `vocabulary.rs`) for
   `hydrat|awareness|precondition|gate` — the only hits are `RefusalGate`
   (a conformance-refusal telemetry enum, `ConformanceDrift` /
   `PoolFull` / …), unrelated to node hydration or awareness. **`ogar-loco`
   is a program/statement/vocabulary interpreter and has never touched
   this concern.**
2. **`kanban_actor::PhaseCensus`: a bare tally, nothing more.**
   `struct PhaseCensus { counts: [usize; 6] }` — one integer per
   `KanbanColumn`. It answers *"how many entities are in `Planning`
   right now,"* never *"are this book's nodes hydrated."* §1 piece C
   graded it SHIPPED for the activity-mask reading, and that grade
   stands — but a count is not a precondition gate, and nothing here
   should be read as claiming otherwise.
3. **"Hydration" is a THIRD homonym collision this session** — following
   the four "witness" surfaces (§3a) and four "nibble" surfaces (§3k),
   verified here rather than assumed:

   | # | where | what it means | relation to the operator's ask |
   |---|---|---|---|
   | a | `bake_hydrate.rs` (MedCare-rs), `S3_BOOT_HYDRATION_WIRING.md` | boot-time fetch of pinned TSV/Lance artifacts | closest by NAME, unrelated by MECHANISM — a deploy-time fetch, not an awareness state |
   | b | `lance-graph-ontology/src/hydrators/{owl,skos,dolce,fibo,…}.rs` | format PARSERS — external ontology syntax → internal types | "hydrate a schema," not "hydrate a node's awareness" |
   | c | `helix::examples::mu_hydration_probe.rs` (`E-3DGS-MU-HYDRATION-1`) | whether an address-derived prediction reconstructs a stored 3D-gaussian-splat position (`ρ` metric) | GPU rendering; `μ` here is the 3DGS mean symbol, nothing to do with books or thinking |

   None of the three is the operator's concept. A fourth, new meaning
   would be minted if this is built — which is a reason for care, not a
   blocker, but it means **do not silently reuse the word "hydrate" for
   this and expect readers to disambiguate from context.**

**The one real near-miss, and why it still doesn't fit.**
`E-FOVEATED-AWARENESS-1` (board, cross-referenced at `recipe_loci.rs` +
`cognitive_shader::{RungLevel,RungElevator}`) is genuinely about
*awareness tied to a trajectory*: *"its own attention trajectory
(KanbanStep/saccade sequence) IS the transmitted index… free energy = the
saccade."* And `self_directed_graph.rs` (capstone, 6 KILL-gates measured)
is a genuinely self-directed reasoning loop with a **COMMIT gate**. Read
together they are the closest shipped material to *"awareness gated by a
kanban-tracked trajectory."* But neither checks **bulk population
readiness before higher-order thinking may run** — foveated-awareness
gates a REPLAY index for rendering determinism; `self_directed_graph`
gates a WRITE-BACK after reasoning concludes. Both are downstream of the
question the operator is asking; neither is upstream of it.

**Verdict was going to be "no design" — the operator supplied one before
this landed.** Operator, immediately following: *"Brutal heißt, bevor wir
über das Buch nachdenken wird ein Inhaltsverzeichnis als HHTL-Baum mit
SoA-Knoten erstellt."* "Brutal" names the SEQUENCING, not a grading of
honesty: no incremental, content-driven tree-growing — the table of
contents is minted as the FULL HHTL tree skeleton, SoA nodes per entry,
**before** any triple-level reasoning (`stance::stream()`, §3d) touches a
single verse.

**This is buildable, and it is the same primitive as `D-ACR-12`, run in the
opposite direction.** A book's structure — book → chapter → section →
verse — IS a tree; a table of contents is nothing but that tree's own
enumeration. So the mint pass is: walk the TOC, and for each entry mint a
`NiblePath` at the depth its level implies (book=1, chapter=2, … per the
canon's own tier-per-nibble rule), with an SoA row addressed at that path —
**before** any content triple exists. This gives `D-ACR-12`'s ascent
primitive something real to walk: the tree is populated top-down by
construction, so an ascent from any future verse-level witness is
guaranteed to terminate at a real ancestor, never an unminted gap.

**The precondition check falls out of this for free, and it is `D-ACR-12`'s
own bulk-frontier loop run as a COVERAGE query rather than a lookup:** run
the bulk ascent (§3k) from every leaf (verse) address up to the TOC root;
higher-order thinking is permitted iff **every leaf's frontier reaches
`resolved` before `max_hops` runs out — zero leaves left stuck in
`frontier`**. A stuck leaf is a verse whose chapter/section was never
minted into the TOC tree: exactly the "hole" a table of contents is
supposed to rule out, made mechanically checkable rather than assumed.

**`D-ACR-13`, redesigned:** two ordered steps.
1. **TOC-mint** — walk the book's table of contents, mint one `NiblePath` +
   SoA row per level, unconditionally, before content ingestion.
2. **Coverage gate** — `D-ACR-12`'s bulk ascent run from all leaves at
   once; `resolved == all leaves` is the precondition higher-order
   thinking checks. `frontier ≠ ∅` after `max_hops` is a **hard refusal**,
   not a warning — the same fail-closed posture `predicate_domains.tsv`
   and `RefusalGate` already use elsewhere in this stack.

Gates on `D-ACR-12` directly (step 2 IS `D-ACR-12`'s loop, reused) and on
`D-ACR-1` for the same reason `D-ACR-12` does — the coverage result is
itself a mask worth recording as an overlay trace, not recomputed silently
on every check.

## §3m — Cartography, rung-dependency reasoning, nested kanban: three different states of "not ready," none the same

Operator's continuation, checked piece by piece rather than assumed to be
one gap: *"Danach läuft eine feste Kartographie der Mindmap"* / *"Dann
reasoning über rung Stufen mit dependency graph"* / *"Und nested kanban
cascade für den awareness Aufbau."*

**Cartography — `holograph::mindmap.rs` — real, rich, EXISTS-UNCALLED, has
a KNOWN bug, AND lives in a substrate unrelated to this plan's own.**
Verified empirically, per operator correction below: `holograph`'s
`Cargo.toml` deps are Arrow/DataFusion/Lance; it imports NONE of
`canonical_node`/`SoaEnvelope`/`MailboxSoA`/`ClassView`; its only
consumers are `bgz-tensor` and `lance-graph-cognitive` — neither the
canonical SoA/mailbox substrate this plan builds on
(`lance-graph-contract`/`lance-graph-supervisor`). `mindmap.rs`'s tree is
`holograph::dn_sparse::TreeAddr`, a GraphBLAS-backed address space that has
NOTHING to do with `NiblePath`/`NodeGuid`/canonical SoA rows.

> **⊘ CORRECTED (operator, 2026-08-21): "Holograph hat meines Erachtens
> nichts mit SoA zu tun."** Right, and it corrects an implicit assumption
> in this section's first draft — that a "wire a caller" fix could connect
> `mindmap.rs`'s bug-fix to the `D-ACR-13`-built SoA tree. There is no such
> connection to wire; the two trees are different substrates in different
> crates, and no bridge exists. §3k's own citation of `dn_sparse.rs` stays
> correctly scoped — it was explicit that the type is prior art for the
> SHAPE only, never shared code — but this section's ORIGINAL framing of
> `D-ACR-14` implied more than that and is corrected now.

`mindmap.rs` still has a substantial API and a known bug (`dn_sparse.rs`'s
own flag: *"unlike the mindmap.rs BFS which calls `mxv` (broken
mutability)"*) — that stays true and worth fixing on ITS OWN terms, as
`holograph`-local maintenance, unconnected to this plan. **"Fixed
cartography of the [book's] mindmap"** — a mapping pass over the
`D-ACR-13`-built SoA tree — has NO existing implementation anywhere and
must be built fresh, on the SoA substrate, using this plan's own
primitives (§3k's bulk-mask ascent/descent, `ClassView`-resolved reads)
rather than `holograph`'s disconnected machinery.

**Rung-dependency reasoning — `WorkflowDAG` (planner strategy #12) — a
STUB, not a placeholder-with-a-name.** Registered in the 16-strategy
dispatch (`strategy/mod.rs`), has a real `affinity()` (0.9 when
`context.features.has_workflow`), but `plan()`'s body is literally:

```rust
// In full implementation:
// 1. Parse workflow task declarations from query
// 2. Build task dependency graph (just another graph pattern)
// 3. Apply LangGraph-style channel+reducer semantics: ...
```

Comments describing what it would do, not code doing it. This matches
CLAUDE.md's own Phase-3 status line, *"wire planner strategies to
lance-graph core (actual parser, not regex)"* — WorkflowDAG is that line,
named precisely.

**Nested kanban cascade — genuinely absent, like `D-ACR-13`'s original
verdict.** Grepped `lance-graph-supervisor/src/*.rs` for
`nested.*kanban|kanban.*cascade` — zero. No near-miss found this time,
foveated or otherwise.

**Three different repairs, not one:**

| piece | state | what it needs |
|---|---|---|
| Cartography (`holograph`) | shipped, uncalled, one known bug, **wrong substrate for this plan** | fix `mxv` on its own terms (`D-ACR-14a`) — separately, build a real cartography pass on SoA (`D-ACR-14b`, not designed) |
| Rung-dependency reasoning | registered stub | implement `plan()` for real — the comment IS the spec, already written |
| Nested kanban cascade | absent | design needed, same honesty as `D-ACR-13`'s first verdict |

`D-ACR-14`/`D-ACR-15`/`D-ACR-16` respectively. None is sequenced ahead of
`D-ACR-13`'s two steps (TOC-mint, coverage gate) — cartography and
rung-dependency reasoning both presuppose a populated, verified-complete
tree to run over, which is exactly what `D-ACR-13` produces.

## §3n — The mint, and the 64k-parallel hydration pipeline in full

Operator, 2026-08-21, closing the thread: *"Und einen eigenen Tenant. Schau
in lance-graph .claude v3, da stehen die alten Pläne."* — followed by the
execution shape: *"Du hast 64k thoughts in parallel, du musst also nur
einen Plan an 64k SoA ausliefern. Die schreiben dann ihr Ergebnis in den
Tenant. Du liest den EpisodicWitness-Tenant aus und hydrierst daraus die
witness arc facet rails für HHTL. Die nodes erstellen die Kartographie.
Das reasoning higher order ist dann nachgelagert."*

**The mint, verified against `.claude/v3/MODULE-TABLE.md` line 167 — not
invented.** `episodic_edges.rs` is ALREADY byte-ready
(`to/from_le_bytes`/`write_le`/`read_le`/`to_u64`/`from_u64`) and the
module table's own classification already tags it **"W1 envelope/ownership
(SoA edge column)"** — i.e. it was always headed for a `ValueTenant` mount,
this is not a new idea. Stronger: `markov_soa.rs`'s own doc comment
(`MODULE-TABLE.md` line 48) says it plainly — *"truly-correct home is
still inside the EW64-in-SoA seam"* — a DIFFERENT module, already staged
elsewhere, explicitly waiting for this exact mount to open.

**Slot number, checked rather than assumed:** `ValueTenant = 15` is
**already reserved** by `BoardAggregates` (`dismech-causality-v3-v1.md:497`
— *"`HoleV3` as `ValueTenant = 16` (`BoardAggregates` already reserves
15)"*), and `16` is `HoleV3` (D-CV3-3, queued). **`ValueTenant::EpisodicEdges
= 17`** — the next free discriminant, additive, reserve-don't-reclaim, same
discipline every prior mount in this enum used.

**The 64k pipeline, three stages, matching the operator's own order:**

1. **Dispatch.** One plan, delivered to the 64k-thought SoA field — the
   kanban field's own native scale (dialectic pillar 2: *"you can't run 64k
   parallel higher-order thoughts conventionally"* → SIMT over syllogisms).
   Each of the TOC-mint pass's leaf addresses (`D-ACR-13` step 1) gets one
   parallel worker.
2. **Write.** Each worker writes its result into `ValueTenant::EpisodicEdges`
   at its own row — `EpisodicEdges64`'s own shape is a **4-slot MRU
   promote/evict tier**, not a flat append, so this is not a growing list
   per row; it is a bounded hot-tier write, the same discipline that keeps
   an SoA row fixed-width. **The nodes build the cartography AS they
   write** — operator: *"die nodes erstellen die Kartographie"* — there is
   no separate mapping pass; `D-ACR-14b` (§3m) is FOLDED into this step
   rather than sequenced after it.
3. **Hydrate + gate.** A read pass over the populated `EpisodicEdges`
   tenant hydrates `ValueTenant::CausalWitness = 14`'s facet rail (§3b's
   *"witness arc"*, the `CausalEdge64` W-slot lineage `soa_view.rs`
   already names) — this is `D-ACR-12`'s bulk-ascent loop, now with a
   real source column to read FROM. Its `resolved == all leaves` output
   IS `D-ACR-13`'s coverage gate. **Higher-order (rung 2+) reasoning is
   strictly downstream** — operator: *"das reasoning higher order ist
   dann nachgelagert"* — it may only start once this gate passes, per
   `D-ACR-13`'s original fail-closed posture.

`D-ACR-17`: mint `ValueTenant::EpisodicEdges = 17`. Field-isolation matrix
test mandatory per `I-LEGACY-API-FEATURE-GATED` (every prior mount in this
enum carries one); `ENVELOPE_LAYOUT_VERSION` unchanged (additive tenant,
not a reclaim). Gates `D-ACR-13`'s write stage — nothing to write into
until the slot exists.

## §3o — Reading without the hindsight eigenvalue: a scope question, answered

Operator, 2026-08-21: *"Die Frage ist nur ob wir DeepNSM-v2 durch thinking
styles und epistemic potholes und CE64 59..63 erweitern, daß Lesen bereits
ein intellektuelles Erlebnis ohne hindsight knowledge eigenvalue wird."*

**No — extending `deepnsm-v2` itself would reverse a ruling this plan
already relies on.** `E-DEEPNSM-V2-IS-INBOUND-LEG-REASONING-LIVES-IN-LANCE-
GRAPH-1` deliberately moved the five tactics OUT of `deepnsm-v2` into
`lance-graph-planner`; §3d already cites the boundary approvingly (*"this
leg emits text, it does not reason over it"*). Pulling thinking styles,
potholes, and CE64 grading back in would blur exactly that seam, for a
result the seam does not require blurring to get.

**The actual mechanism already exists, unbuilt but fully specified — it
is `first_possible` vs `first_derived` (board S3.8), applied LIVE rather
than only as a retrospective audit.** "An intellectual experience without
hindsight" is precisely: at verse `v`, only what was derivable using
`QueryReference::at(v, rung)` — never a conclusion that in fact depended
on verse `v+4000`. §3i already named the primitive (`temporal.rs`); this
sharpens its USE: it is not merely a post-hoc filter that flags tautologies
after the fact, it is a **live horizon** the reasoning stage reads through
while processing the stream in order.

**Where this lands: `D-ACR-15`, not `deepnsm-v2`.** Once `WorkflowDAG::plan()`
is real (its current stub state), its dependency-graph walk over `D-ACR-13`'s
hydrated tree gains one constraint: each node's reasoning step is bound to
`QueryReference::at(node's own version, rung)` — never the fully-hydrated
end state. This is additive to `D-ACR-15`'s existing acceptance condition,
not a new deliverable, and it is exactly what makes S3.8 (*"potholes,
first_possible vs first_derived, strict historical replay"*) finally have a
live consumer instead of staying an audit-only board line.

## §4 — Deliverables

| D-id | Scope | Repo | Falsifier |
|---|---|---|---|
| **D-ACR-0** | Audit `attention_mask.rs` / `attention_mask_actor.rs` against piece E: is the shipped mask a residue carrier, or something else wearing the name? Report, no code. | lance-graph | the audit names a caller, or records EXISTS-UNCALLED |
| **D-ACR-1** | `RowFocusMask` — the one missing primitive (piece D). Mask over rows visited, composable with `WideFieldMask` per S3.1b. | lance-graph | can-fire **and** can-stay-silent on non-trivial input; a focus over 0 rows and over all rows must be distinguishable from a real one |
| **D-ACR-2** | Mint the **Rung ladder** rail (HTT §2.3 row) — gated on §8 Q3 mint decision, NOT on this plan | lance-graph | `rail_carving` gains its first non-default consumer |
| **D-ACR-3** | The one-way invariant as a test, not prose: no ontology-owned write traces to a patient-tagged read through ANY call path (CodeRabbit-corrected from write-authorization-only) | lance-graph | the negative case is the test; a write whose call graph includes a session-tagged read is the bug, even if the write itself is authored by the ontology owner |
| **D-ACR-4** | Second-order row (§3) over D-ACR-1 + D-ACR-2 | lance-graph | a rung-2 read reconstructs where rung-1 looked, on a fixture where the answer is known independently |
| **D-ACR-5** | 64k lowering | lance-graph | **BLOCKED** — dialectic V4's own gate: V0–V3 green at small scale first |
| **D-ACR-6** | KJV prestaging: missing epistemic-causality nodes as episodic basins, promoted rows kept index-width (§3d — never fat concepts) | lance-graph | **BLOCKED** — HTT X3, the basin-promotion seam does not exist; needs the same mint as D-ACR-2; when built, a promoted basin row must stay reference-only, same test as `WitnessLens` |
| **D-ACR-9** | A `Vocabulary` impl exposing the 34 recipes as loco ops above `DOMAIN_FLOOR` (§3f); `ladder(ctx)` lowered to a loco program | OGAR | a pothole with a negative `rung_delta` executes as a loco call sequence and lands the same `RecipeStep` list `ladder()` returns |
| **D-ACR-8** | Rubicon witness (§3e): focus-mask breadth/persistence across `Planning → CognitiveWork`; closes the Rubikon plan's open *"Thinking styles ↔ Rubikon"* item | lance-graph | broader in `Planning` than in `CognitiveWork` on a deliberated task **AND** indistinguishable on a single-forced-candidate task |
| **D-ACR-7** | The 59..63 reading contract (§3b): name, per `(classid, rail)`, which lens applies and which witness carrier discriminates evidence-kind. **Acceptance: any tactic sampling filters on `delta_conf` capability (14/34), never on `maturity().is_production()` (31/34)** — §3g | lance-graph | a producer/consumer pair disagreeing about `TrustTexture` vs `CausalTopology` must FAIL, not return a plausible value; and a sampling pass must reject a mute tactic |
| **D-ACR-12** | Epistemic inheritance (§3k): BULK frontier ascent over `NiblePath` ancestry until content is found (mask-native, not per-address — pattern from `lance-graph-java`'s `hop()`), materialized as `(source_addr, resolved_addr, hop_distance)` in the Rung-ladder row | lance-graph | resolves to the nearest ancestor at the correct hop count; a node with its OWN witness never ascends past it; resolution cost scales with tree DEPTH (bulk rounds), never with population size |
| **D-ACR-13** | Mass hydration as a checked precondition for higher-order thinking (§3l) | OGAR + lance-graph | **NOT DESIGNED** — zero shipped precedent found in `ogar-loco` or `kanban_actor`; open pending a registered design once `D-ACR-1` exists |
| **D-ACR-14a** | `holograph`-local: fix `mindmap.rs`'s `mxv` mutability bug, on ITS OWN terms — unrelated to this plan's SoA substrate | holograph | fix ships WITH a real `holograph`-side caller — no bug-fix-with-no-consumer landing |
| **D-ACR-14b** | **⊘ FOLDED into `D-ACR-13`'s write stage (§3n)** — cartography is a byproduct of the 64k parallel hydration writes, not a separate pass | lance-graph | superseded — see `D-ACR-13` |
| **D-ACR-17** | Mint `ValueTenant::EpisodicEdges = 17` (§3n) — the byte-ready `episodic_edges.rs` type, gates `D-ACR-13`'s write stage | lance-graph | field-isolation matrix (`I-LEGACY-API-FEATURE-GATED`); `ENVELOPE_LAYOUT_VERSION` unchanged |
| **D-ACR-15** | Rung-dependency reasoning: implement `WorkflowDAG::plan()` for real — the strategy's own comment is the spec. **Bound to `QueryReference::at(node version, rung)` per node (§3o) — no hindsight leakage** | lance-graph | a query with `has_workflow` actually dispatches through a built `DEPENDS_ON` graph, not the 0.9-affinity stub; a node's conclusion changes if a LATER version is fed in and does NOT change if only earlier versions are, proving the horizon actually binds |
| **D-ACR-16** | Nested kanban cascade for awareness build-up (§3m) | lance-graph | **NOT DESIGNED** — zero shipped precedent, matching `D-ACR-13`'s original honest-empty verdict |
| **D-ACR-11** | DK/eigenvalue probe (§3i): perturb inputs, measure which tactics' confidence is invariant; cross-check the 20 non-`delta_conf` tactics land at invariance by construction | lance-graph | the 20 must show invariance (else the capability flag lies) **and** at least one of the 14 must actually move (else the flag is decoration) |
| **D-ACR-10** | Hindsight probe (§3i): `first_possible` vs `first_derived` over `QueryReference::at` on a real trajectory | lance-graph | a claim derivable at every version must be reported as discriminating nothing |

**Order is not negotiable:** D-ACR-0 (audit) → D-ACR-1 (the primitive) →
D-ACR-7 (the reading contract — before anything writes a band) → D-ACR-3 (the
boundary) → D-ACR-8 (Rubicon witness) → D-ACR-9 (loco recipe vocabulary) →
D-ACR-12 (epistemic inheritance, gates on D-ACR-1) → D-ACR-2/4/6 (mint +
second order + basins) → D-ACR-5. **D-ACR-13 is explicitly NOT sequenced** —
it has no design to sequence yet; it re-enters this order only once one is
registered. **D-ACR-17 (mint) → D-ACR-13 (dispatch+write+gate, folding in former D-ACR-14b) →
D-ACR-15 (rung-dependency reasoning, downstream of the gate) (D-ACR-14a is
unrelated and unsequenced — a holograph-local fix)** (both need a
verified-complete tree to run over); **D-ACR-16 is unsequenced, same reason
as D-ACR-13's original verdict.** **D-ACR-9 additionally
waits on the window question in §3f** — a revision pass cannot be stamped into
an interval that two documents describe differently. D-ACR-2
and everything after it sit behind an operator mint decision that this plan
does not pre-empt.

## §5 — Non-goals

- **No new address type.** S3.0/PR #973 was closed exactly here — *"CLOSED —
  NOT NEEDED (use `IdentityQuad` / `ClassAddr` / V3 rail)"*, and the ladder's
  empty column was ruled to be **HYDRATION, not ADDRESS**. The overlay is
  hydration over existing addresses.
- **No CE64 bit.** Bits 59..63 are spoken for (`TRUTH_SHIFT` 59–60 /
  `SPARE_SHIFT` 61–63) and the band there is set only by an explicit
  `with_reasoning_band()` call — **nothing derives it**. The overlay must not
  become a fifth derivation path into those bits.
- **No `ENVELOPE_LAYOUT_VERSION` bump.**
- **No proof claim.** Piece 7: a pruner. Any deliverable that starts asserting
  the residue *justifies* a conclusion has left this plan.

## §6 — Deferred (missing integration)

| # | Item | Why deferred |
|---|---|---|
| **Y1** | Whether the rung ladder's **two axes** (HTT §3) are the same two the alpha layers need, or a third pairing. Unmeasured. | §3 is session-measured for the ladder, never for an overlay |
| **Y2** | `RowFocusMask × WideFieldMask` basis collision — the HTT **X4** latent-third-basis problem applies verbatim the moment a focus mask meets a field mask | X4 is audited, not solved, deliberately |
| **Y3** | The residue's **retention policy**. §2 says it may be discarded whole; nothing says when it is. | needs a measured working-set size, which needs D-ACR-1 first |

## §7 — What this plan does NOT claim

It does not claim the overlay improves recall, that eye-tracking residue finds
needles, or that 64k rungs are reachable. It claims one thing: **the empty row
in §2.3 has a design now, and six of its nine parts already exist.** Every
number that would justify the rest has to be measured after D-ACR-1.
