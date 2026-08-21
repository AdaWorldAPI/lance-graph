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

**The witness carriers are live, not aspirational** (measured this session, file
counts across crates, `target/` excluded):

| carrier | files | verdict |
|---|---:|---|
| `WitnessLens<'a>` (`witness_fabric.rs:123`) | 11 | wired — planner, contract, holograph, deepnsm |
| `WitnessTable<const N = 64>` (`witness_table.rs:112`) | 9 | wired |
| `WitnessEntry` (`witness_table.rs:81`) | 8 | wired |
| `CausalWitnessFacet` (`causal_witness.rs:201`) | — | (write-empty per the CV3 rebase report) |
| **`EpisodicBasins`** (`arigraph/episodic.rs:79`) | **2** | **X3 confirmed: definition + `mod.rs` only** |

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
ran. Prestaging its missing causality nodes as episodic basins is the
**Type-B basin promotion** seam — and HTT **X3** records that it *does not
exist*: `EpisodicMemory::basins()` (`episodic.rs:243`) returns `EpisodicBasins`
(`:79`) **by value**, with no `ValueTenant` slot reserved. Measured here: the
type appears in **two** files, its own module and `arigraph/mod.rs`.

So this is not "someone should wire it" — a promoted basin is exactly a *new
thinking-table row with no minted rail*, which puts it behind the same mint
decision as D-ACR-2 (HTT §8 Q3). Recorded as `D-ACR-6`, blocked, with its
blocker named.

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

## §4 — Deliverables

| D-id | Scope | Repo | Falsifier |
|---|---|---|---|
| **D-ACR-0** | Audit `attention_mask.rs` / `attention_mask_actor.rs` against piece E: is the shipped mask a residue carrier, or something else wearing the name? Report, no code. | lance-graph | the audit names a caller, or records EXISTS-UNCALLED |
| **D-ACR-1** | `RowFocusMask` — the one missing primitive (piece D). Mask over rows visited, composable with `WideFieldMask` per S3.1b. | lance-graph | can-fire **and** can-stay-silent on non-trivial input; a focus over 0 rows and over all rows must be distinguishable from a real one |
| **D-ACR-2** | Mint the **Rung ladder** rail (HTT §2.3 row) — gated on §8 Q3 mint decision, NOT on this plan | lance-graph | `rail_carving` gains its first non-default consumer |
| **D-ACR-3** | The one-way invariant as a test, not prose: no ontology-owned write traces to a patient-tagged read through ANY call path (CodeRabbit-corrected from write-authorization-only) | lance-graph | the negative case is the test; a write whose call graph includes a session-tagged read is the bug, even if the write itself is authored by the ontology owner |
| **D-ACR-4** | Second-order row (§3) over D-ACR-1 + D-ACR-2 | lance-graph | a rung-2 read reconstructs where rung-1 looked, on a fixture where the answer is known independently |
| **D-ACR-5** | 64k lowering | lance-graph | **BLOCKED** — dialectic V4's own gate: V0–V3 green at small scale first |
| **D-ACR-6** | KJV prestaging: missing epistemic-causality nodes as episodic basins (§3d) | lance-graph | **BLOCKED** — HTT X3, the basin-promotion seam does not exist; needs the same mint as D-ACR-2 |
| **D-ACR-9** | A `Vocabulary` impl exposing the 34 recipes as loco ops above `DOMAIN_FLOOR` (§3f); `ladder(ctx)` lowered to a loco program | OGAR | a pothole with a negative `rung_delta` executes as a loco call sequence and lands the same `RecipeStep` list `ladder()` returns |
| **D-ACR-8** | Rubicon witness (§3e): focus-mask breadth/persistence across `Planning → CognitiveWork`; closes the Rubikon plan's open *"Thinking styles ↔ Rubikon"* item | lance-graph | broader in `Planning` than in `CognitiveWork` on a deliberated task **AND** indistinguishable on a single-forced-candidate task |
| **D-ACR-7** | The 59..63 reading contract (§3b): name, per `(classid, rail)`, which lens applies and which witness carrier discriminates evidence-kind. **Acceptance: any tactic sampling filters on `delta_conf` capability (14/34), never on `maturity().is_production()` (31/34)** — §3g | lance-graph | a producer/consumer pair disagreeing about `TrustTexture` vs `CausalTopology` must FAIL, not return a plausible value; and a sampling pass must reject a mute tactic |
| **D-ACR-11** | DK/eigenvalue probe (§3i): perturb inputs, measure which tactics' confidence is invariant; cross-check the 20 non-`delta_conf` tactics land at invariance by construction | lance-graph | the 20 must show invariance (else the capability flag lies) **and** at least one of the 14 must actually move (else the flag is decoration) |
| **D-ACR-10** | Hindsight probe (§3i): `first_possible` vs `first_derived` over `QueryReference::at` on a real trajectory | lance-graph | a claim derivable at every version must be reported as discriminating nothing |

**Order is not negotiable:** D-ACR-0 (audit) → D-ACR-1 (the primitive) →
D-ACR-7 (the reading contract — before anything writes a band) → D-ACR-3 (the
boundary) → D-ACR-8 (Rubicon witness) → D-ACR-9 (loco recipe vocabulary) →
D-ACR-2/4/6 (mint + second order + basins) → D-ACR-5. **D-ACR-9 additionally
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
