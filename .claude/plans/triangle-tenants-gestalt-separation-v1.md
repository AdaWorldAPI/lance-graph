# Triangle tenants × separated awareness surfaces × chess quarantine — v1

> **Status:** DESIGN (operator-directed 2026-07-17; no bytes land with this
> doc — every tenant below is LAYOUT-GATED behind the envelope-auditor +
> the next batched classid/tenant mint, per Addendum-12a discipline).
> **Operator brief (verbatim anchors):** three tenants Frozen/Learned/
> Explore, each `6×2 = 12` family positions, 8-bit/256 each; episodic
> witness/basins SEPARATE; gestalt awareness (mirror neurons / opponent
> meta / object resonance) different from episodic witness; "if a problem
> feels like a nail, search for the hammer — not the other way round";
> land 12-step orchestration AND 12-family redundantly and measure with
> jc (ICC/Pearson/Cronbach α/Spearman) before collapsing; the 12
> positions are ClassView-codebook-reassignable (persona/business);
> chess tenant separate for SoC until validity/reliability are measured.
> **Index:** `.claude/board/INTEGRATION_PLANS.md` (prepended same commit).
> **D-ids:** D-TRI-1..4 (STATUS_BOARD rows, same commit); D-TRI-5..6
> added with the 2026-07-17 follow-up directives.
> **Follow-up operator directives (same day, folded in before merge):**
> (1) "feels like a king attack is different from emulation of
> opponent's thinking based on counterfactual goalstate" → §2a
> correction (emulation ≠ resonance); (2) "Maslow pyramid of cognition —
> observation/gestalt first, then SPO 2³ decomposition, CausalEdge64,
> NARS candidate design" + check thinking-engine rung-escalation logic
> against .claude/v3 plans → §3a (audit receipts + composition design).

---

## §1 The triangle as three value tenants

Three NEW 12-byte value tenants per (thinking-preset) row, appended after
Kanban `[144,152)` — offsets assigned by the mint-gated spec, shown here
as the intended shape only:

| Tenant | 12 B read | Position semantics | Value semantics |
|---|---|---|---|
| `FrozenStyle` | 12 × u8 | index = `StyleFamily` ordinal 0..11 (D-TSC-1 frozen order) | palette256 atom: the row's CHECKPOINT policy for that family |
| `LearnedStyle` | 12 × u8 | same | palette256 atom: the NARS-revision-updated policy |
| `ExploreStyle` | 12 × u8 | same | palette256 atom: the exploration variant (deterministic, address-derived jitter — the D-QUANTGATE coprime walk; never RNG, replay holds) |

The value codebook is the **226-atom cognitive palette256** (144 verb
atoms ∥ 34 recipe atoms ∥ 36 persona atoms ∥ 12 family atoms; 30
reserved, append-only — RESERVE-DON'T-RECLAIM). One byte per atom; the
per-family triangle read is `(frozen[f], learned[f], explore[f])` — three
atoms, one glance, zero decode.

**Why this shape carries the ancestry pipeline** (the operator's point —
the triangle gives the thinking-engine → P64 → cognitive-shader-driver
lineage its wiring targets instead of leaving the gems orphaned):

- **Dispatch (shader)** reads `FrozenStyle` — the can't-stop-thinking
  cycle runs off the checkpoint, 40 ns-class, no lookups beyond the row.
- **Perturbation (P64 ladder)** — `StreamDto → PerturbationDto` feeds the
  `ExploreStyle` variant; `spiral_segment.rs`/`prime_fingerprint.rs`
  re-scope here as explore-lane generators (VSA-niche-compliant).
- **Learning (L4 seam)** — `BusDto` outcomes (`converged`, `cycle_count`)
  drive NARS revision INTO `LearnedStyle`; `l4_bridge.rs::commit_to_l4`
  finally gets its owner-stamped write target (resolves the W4b
  ORPHAN-WRITE flag: the write is an owner `&mut` on the row's own lane).
- **Promotion** is a kanbanstep-interior act in the Evaluation phase:
  `learned[f]` replaces `frozen[f]` only after winning the held-out arm
  (the loop the chess golden probe proved). No events, no waiting — the
  owner compares its own three bytes and writes one.
- **Rubicon tie:** Planning reads wide (explore ∪ learned candidates);
  the crossing commits to ONE atom; Evaluation compares outcome against
  the committed atom and revises. The phase→(mask, rung, policy) style
  schedule from the Rubicon discussion is exactly a read-policy over
  these three lanes — no fourth representation needed.

## §2 The distinct surfaces, kept apart (the separation the operator ordered)

Four awareness surfaces (triangle = policy, episodic = memory, gestalt =
resonance, emulation = simulation) plus the quarantined chess corpus —
five rows, one per organ, never blended:

| Surface | Question it answers | Carrier | Explicitly NOT |
|---|---|---|---|
| **Triangle tenants** (§1) | "what do I do" — competence/policy | 3 × 12 B lanes, palette256 atoms | not memory, not resonance |
| **Episodic witness + basins** | "what happened" — factual, versioned | Lance versions (`temporal.rs` pinned reads) + KanbanMove witness chains + `part_of:is_a` basin rails (le-contract L1–L3) | not a policy store; never blended into triangle bytes |
| **Gestalt awareness** | "what does this resemble / feel like" — perspectival resonance | `awareness_dto.rs::ResonanceDto` (HDR subject/predicate/object resonance, dominant `Archetype`, divergence) + palette256 distance tables (object ↔ codebook) | not episodic replay; not the witness log; **not opponent emulation** (§2a) |
| **Opponent emulation (mirror-SoA)** | "what would the other think/do" — counterfactual-goalstate simulation | a sibling SoA **compartment** running the same Rubicon machinery seeded with the abduced opponent goalstate; its output re-enters the self-SoA as witness VALUES | not a carrier tenant — a process; not resonance; not replay |
| **Chess tenants** | domain corpus (DecisionEpisodeV1: episode/candidate/iteration/event) | OWN classids, own rows — quarantined | not thinking-row lanes, until D-TRI-4 measures |

### §2a Correction (operator, 2026-07-17): emulation ≠ resonance

The first draft of this section filed the mirror-neuron SoA under gestalt
("the opponent-model sibling SoA's meta-awareness … is a gestalt read").
**That was a conflation, corrected by operator ruling:** "feels like a
king attack" is DIFFERENT from emulation of the opponent's thinking based
on a counterfactual goalstate. The original E-TASKS-SOA-AT-THE-MEMBRANE-1
wording already said so — *"**Counterfactual** and concurrent thinking
span compartments — in chess, model a 'mirror neuron'-like SoA for the
opponent's thinking"* — and this correction restores it. Three organs on
the opponent axis, never blended:

1. **Episodic replay** — what they DID (the opponent played Nf3 twice).
   Lance versions, witness chains. Pearl *See* over facts.
2. **Gestalt resonance** — what it FEELS like (the position resembles a
   kingside attack). ResonanceDto + palette distance, O(1) lookup, no
   forward simulation. Resemblance, not prediction.
3. **Counterfactual emulation** — what they WOULD think/do given THEIR
   goalstate. A sibling mirror-SoA runs the SAME thinking machinery
   (own board, own budget, own rung horizon) with the goalstate swapped
   to the opponent's. Pearl *Imagine* — a simulation, not a lookup.

The emulation loop wires entirely to SHIPPED machinery (nothing new):

- **Abduce the goalstate** from observed episodes —
  `InferenceType::Abduction`; the measured shape is D-SF-OPPONENT-3's
  "opponent model = own L1 + measured residue of their choices against
  it" (and opponent-model *inference* is the measured bottleneck, per
  E-SF-TRAP-LURE-GREEN-1 — lure synthesis was the easy half).
- **No goalstate field exists to write** — grep confirms zero
  `GoalState` types anywhere in crates/; the goalstate is phase-implicit
  in the Rubicon `KanbanColumn` (Heckhausen supplies it doctrinally,
  `mailbox-kanban-model.md:37`). "Swapping the goalstate" therefore
  means seeding the sibling SoA's own Planning phase from the abduced
  goalstate — never a field write on the self's rows.
- **Fork = the shipped scenario machinery** —
  `contract::scenario::ScenarioBranch` (`fork_seed` deterministic
  replay, `interventions`, default inference mode
  `CounterfactualSynthesis`) and
  `lance-graph-cognitive/world/counterfactual.rs::intervene()` (Pearl
  Rung 3 do-calculus on fingerprints, implemented + integration-tested).
  Board note: `World::fork` hypotheticals are NOT yet running per
  cognitive cycle — the mirror-SoA is that wiring's first customer.
- **Emissions ride the counterfactual lane** — CausalEdge64 −6 mantissa
  (`causal-edge` `InferenceType::Counterfactual`), under the existing
  iron invariant (`contract::counterfactual`): *"a counterfactual stays
  in a separate lane — it is NEVER written as observed SPO truth."*
- **Close the loop in Evaluation** — the emulation's predicted move is
  compared against the opponent's ACTUAL move; the mismatch drives NARS
  `Revision` of the abduced goalstate. Kanbanstep-interior, owner `&mut`.

The mirror-SoA is a **compartment, not a carrier**: it consumes the §2
awareness surfaces internally (its own triangle read, its own gestalt,
its own episodic view) under the swapped goalstate, and the self-SoA
observes it only through cloned witness values (the rs-graph-llm probe-3
shape — observation detached from write capability by construction). On
the §3a pyramid it sits at the APEX (RungLevel ≥ 6, Pearl level 3);
gestalt resonance sits at the BASE. The pyramid PRESERVES the operator's
correction as a base-vs-apex separation, but the distinction itself is
CATEGORICAL (O(1) resemblance-lookup vs forked simulation) — position
alone does not carry it, since episodic replay (organ 1) also sits low.

## §3 The nail→hammer rule (resonance dispatch direction)

**Rule: dispatch begins at the OBJECT.** The problem's gestalt
(resonance of the object fingerprint against the atom codebook and the
global context) retrieves candidate atoms/families; the triangle lanes
then supply the frozen/learned/explore variant of the resonant family.
The hammer is fetched by the nail's shape.

The inverted direction — a resident tool scanning the world for
applications — is the lens/hammer failure mode (the ack arc was its
session-level instance). The architecture makes the correct direction
the CHEAP one: object → palette-distance lookup (O(1), 64 KB table,
bgz17 machinery unchanged) → family → triangle read. The inverted
direction has no supported read path at all: nothing iterates "all
objects matching my current style."

Wire, don't invent: the matching surface is the existing perspectival
`ResonanceDto` + the existing palette distance/compose tables. New code
is only the glue read (gestalt → family index), probed in D-TRI-3.

## §3a The Maslow pyramid of cognition — the rung-escalation ladder (operator-directed, 2026-07-17)

**The ordering: observation/gestalt FIRST, then SPO 2³ decomposition →
CausalEdge64 NARS candidates → NARS candidate design (revision), with
counterfactual emulation at the apex.** Each level engages only when the
level below fails to settle — the Maslow reading: the base is wide (the
D-TRI-6 EXPECTATION is that most stimuli settle at resonance), the apex
narrow (full counterfactual design rare), and the pyramid's width IS the
cost amortization. **The middle-tier emissions ARE NARS candidates**
(`mailbox-kanban-model.md:42` doctrine: "emitting `CausalEdge64` NARS
candidates"); the upper tier does not INTRODUCE NARS — it DESIGNS
(revises) the candidates the middle already emitted:

| Tier | Operation | Cost shape | RungLevel band (see below) | Pearl |
|---|---|---|---|---|
| **Base** | observation/gestalt — resonance vs palette codebook (`ResonanceDto`, palette distance) | O(1) lookup | 0–2 | 1 *See* |
| **Middle** | SPO 2³ decomposition → CausalEdge64 NARS candidates (resonance-gated, ≤8/cycle) | 3 table reads → all 8 projections | 3–5 | 2 *Do* |
| **Upper** | NARS candidate DESIGN — NarsTables revision of the middle's candidates (rung-content 3, the 34 recipes, per O1-annotation) | one SPO cache load over ≤8 L1-resident cycles | 5† | 2→3 |
| **Apex** | counterfactual emulation — ScenarioBranch fork / mirror-SoA (§2a) | fork + simulate forward | 6–9 | 3 *Imagine* |

The `pearl_level()` partition is crisp (0–2 / 3–5 / 6–9); Middle and
Upper both live inside Pearl-2's 3–5 band — Upper is not a fourth Pearl
level but the *design* pass over Middle's *emission*, so its RungLevel
mark is `5†` (the top of the intervene band, where revision engages
before the 6-boundary crossing into counterfactual). The four tiers are
FUNCTIONAL, not a re-partition of the certified three-band ladder.
**Shipped-vs-designed delta named:** the driver TODAY runs the NarsTables
revision *optionally per cycle* (a per-hit feed, "observed only"); the
pyramid DESIGNS it as *non-settlement-gated* (engaged only when the base
+ middle fail to settle). That is a real behavior change the wiring
introduces, D-TRI-6-gated — not a shipped property.

**Audit receipts (2026-07-17 fleet sweep) — what already EXISTS:**

- **The ladder + homeostatic policy are SHIPPED in the contract**:
  `cognitive_shader.rs` `RungLevel` 0–9 with `pearl_level()` (0–2
  observe / 3–5 intervene / 6–9 counterfactual — "Meta/Recursive/
  Transcendent are counterfactuals *about* counterfactuals") and
  `causal_mask_bits()` (L2→PO, L3→SPO probe-certified; L1→O labeled
  CONVENTION pending its own probe); `RungElevator` — BLOCK streak
  elevates one rung, FLOW streak relaxes toward the dispatched `base`
  floor (never stuck meta forever), HOLD freezes both streaks;
  System-1 felt hints (`escalation::rung_delta`) and System-2 gate
  streaks drive the ONE ladder. `DEFAULT_THRESHOLD = 2`, hand-tuned
  (recorded per I-NOISE-FLOOR-JIRAK).
- **The driver already runs base-before-causal per cycle**: cascade
  stage [3] computes per-hit resonance, which FEEDS the optional
  NarsTables revision and GATES the stage-[5] CausalEdge64 emission
  (≤8 per cycle). This implements *base-gates-the-middle* (the base is
  consulted first and gates the emission) — the settle DISTRIBUTION
  across tiers is a separate question, D-TRI-6's, not shown by the
  gating alone.
- **The amortization property is implemented + probe-tested**:
  `nars_engine.rs` `SpoDistances::all_projections` derives all 8 Pearl
  projections from 3 table reads (probes p3b, p6 green). `StyleVectors`
  are weight vectors over the 8 projections.
- **A Maslow-consistent damping is already coded**: `meaning_axes.rs`
  volition weight `1/(1+rung·0.3)` — high rungs deliberate, low rungs
  act.

**What is MISSING — the pyramid is the composition, and it is NEW:**

- **The stateful ascent loop is UNWIRED.** thinking-engine's
  `should_elevate` (`consecutive_blocks ≥ 3 || FE > 0.15 ||
  cascade_depth ≥ 4`) is implemented, tested, and DEAD — no caller;
  `Agent.current_rung` is initialized to `Surface` and never mutated;
  `LayerId` L1–L10 is an enum no dispatcher walks; the
  `contract_bridge` selector is stateless with a ceiling at
  `Analogical` (rungs 4–9 unreachable via that path).
- **No doc composes the full ordering.** The v3 sweep found three
  PARTIAL orderings never composed: the nail→hammer rule (§3),
  V3-native dispatch (gestalt resonance → compiled template AT a rung,
  `mailbox-kanban-model.md`), and the Rubicon phase schedule + SPO 2³
  ladder. persona-vs-rung-ladder O1/O2 are exactly the recorded gaps
  (rung↔content wiring absent; the rung-4→rung-3 edge missing) where
  this composition lands.
- **The kanbanstep "one SPO cache load ≤8 L1 cycles" loop is
  doc-only** as a production mechanism; the shipped adjacent shape is
  ≤8 edges per single cycle. The property is proven; the loop is not
  built.

**The wiring design (mechanism-only, no mint, probe-gated D-TRI-6):**
the kanbanstep-interior loop holds ONE `RungElevator` per active
thought. Each cycle: (1) base read — gestalt/resonance vs codebook;
FLOW → settle/commit at the base (the D-TRI-6 expectation is that most
stimuli end here, and the FLOW streak relaxes any prior elevation back
down); (2) non-settlement
→ `on_gate` drives the elevator; the current rung's
`causal_mask_bits()` selects WHICH SPO projections the 2³ decomposition
consults; (3) candidates emit resonance-gated (existing stage [5]);
(4) apex rungs (≥ 6, `pearl_level() == 3`) unlock the counterfactual
machinery — ScenarioBranch fork, and for other-minds the §2a mirror-SoA
emulation. Homeostatic by construction: the shader "can't stop
thinking," but it RESTS AT THE BASE.

**Vocabulary demarcation (MANDATORY — five distinct "rung" vocabularies
exist; the pyramid's AXES are exactly two — semantic-depth `RungLevel`
(position) and Pearl (grade). The rung-content ladder appears only as a
per-band content annotation (unwired, O1); the temporal and DTO
vocabularies never enter the pyramid at all):**

| Vocabulary | Lives | Role in the pyramid |
|---|---|---|
| Semantic-depth `RungLevel` 0–9 | contract `cognitive_shader.rs` (canonical) | THE vertical axis — the ladder position |
| Pearl causal rungs 1/2/3 | derived via `pearl_level()`; causal-edge masks | the GRADE of causal content consulted at a position |
| Rung-content ladder 0–4 | operator ruling, `persona-vs-rung-ladder.md` — NO code type (O1) | WHAT vocabulary each band may draw on (verbs/recipes/style macros) — wiring absent, this plan does not close O1 |
| Temporal epistemic rung | `temporal.rs` `QueryReference::at(v, rung)` | READ-side admission only — **not part of the pyramid, never conflate** |
| DTO-ladder "four rungs" | `v3-substrate-primer.md` | envelope layering — unrelated |

**Hazards recorded for the wiring session (found by the sweep, all
pre-existing):** (a) L1-mask divergence — contract `causal_mask_bits`
L1→O=0b001 (labeled CONVENTION) vs causal-edge `CausalMask` L1=SO; the
base tier must pick one after the L1 probe. (b) SPO bit-order
divergence — ndarray `entropy_ladder.rs` bit0=S vs planner bit2=S
(S=0b100 is the P3-certified convention); cross-repo, do not copy masks
across without translating. (c) Pearl numbering wobble —
`scenario.rs` doc calls interventions "Pearl Rung 3" where every other
site says rung 2. (d) thinking-engine's `RungLevel` is a
variant-identical DUPLICATE of the contract type with its own dead
`should_elevate` — the wiring lands on the CONTRACT type; the duplicate
is jc-measured then collapsed, never silently (the D-TSC-1 lesson).

## §4 One register, two readings: 12 families | 12 template steps (operator refinement, mid-design)

**Each 4+12 tenant has 12 slots, and the slots read EITHER as the 12
thinking families OR as 12 thinking-template steps for orchestration** —
ClassView-selected, per the content-blind register doctrine (the 12 B
holds every sanctioned reading at once). Position addressing is
nibble-scale (12 of 16 position codes used; 4 reserved — the same
12-used/4-reserved rhythm as EdgeBlock's 12+4, observed, not claimed);
each slot's VALUE is one byte — a palette256 atom.

- **Family reading** (thinking rows): slot = `StyleFamily` ordinal, value
  = the policy atom — this is what the triangle lanes of §1 instantiate.
- **Step reading** (orchestration rows): slot = ordered template step
  0..11, value = the atom that fires at that step (a runbook/verb from
  the palette). **A compiled elixir-like template with ≤12 steps IS one
  facet** — classid names the template, the 12 slots are its microcode,
  `StepMask` (W3a) selects the active subset. Longer templates chain
  across rows via EdgeBlock. This closes the codebook-microcode design
  from the templates discussion: "notebook microcode" is literally a
  facet-resident byte-string; the elixir text stays the generated view.
- **Reassignment is free by construction:** persona rows or business
  rows read the SAME positions through a persona or process-step
  codebook (`classview codebook reassign`) — position is structure,
  codebook is meaning, no layout change ever.
- **`is_a` expands 12 → 256.** Each slot's coarse position opens into
  the palette256 via the `is_a` basin rails — the 12 are the coarse tier
  of the atom hierarchy, not a rival vocabulary.
- **The redundancy is the measurement instrument, not waste.** The
  family reading and the step reading (and the planner `thinking/`
  12-surface they both descend from) stay live in parallel; D-TRI-2
  measures their agreement with the jc battery over real cycles.
  Collapse only on measured identity; keep both if they measure as
  distinct constructs. This institutionalizes the D-TSC-1 lesson (five
  style tables diverged silently because nobody measured) and the
  1a11038 lesson (never collapse without parity pins).

## §5 Chess quarantine (SoC until the numbers exist)

Chess episode data (DecisionEpisodeV1 → episode/candidate/iteration/
event lanes) lands under its OWN domain classids, in chess-domain rows —
**never in thinking rows**. The unification gate is D-TRI-4: ICC,
Pearson, Cronbach α, Spearman between chess-learned-lane trajectories
and thinking-lane trajectories on shared probes — validity AND
reliability, per the standing jc-pillar gate, before any merge of the
corpora. Until those numbers exist, separation of concerns is the rule,
and cross-domain claims stay unbankable.

**Mint discipline:** the chess classids + the three triangle tenants +
W2a `BoardAggregates` + the Tasks-SoA task-row classid all ride **ONE
batched mint** (never solo edits). W2a's Addendum-12a machinery (T1–T6
field-isolation, cross-classid reinterpretation guard, no
ReadMode-DEFAULT fall-through, ENVELOPE_LAYOUT_VERSION regression) is
the template every one of these tenants instantiates.

## §6 Probes (all pre-registered; mechanism lands only after its probe)

| D-id | Probe | Gate |
|---|---|---|
| D-TRI-1 | Triangle tenant spec through the envelope-auditor (T1–T6 per lane) behind the batched mint | LAYOUT-GATED verdict; zero ENVELOPE_LAYOUT_VERSION change (additive lanes) |
| D-TRI-2 | 12-step ↔ 12-family agreement: jc battery over real shader cycles (ICC, Pearson/Spearman per position, Cronbach α of each 12-vector as a scale) | pre-registered thresholds BEFORE the run; collapse only on measured identity; distinct-constructs is a PASS outcome too |
| D-TRI-3 | Nail→hammer dispatch: object-resonance → atom retrieval vs the inverted baseline on a labeled fixture | retrieval accuracy + the structural check that no inverted read path exists |
| D-TRI-4 | Chess↔thinking transfer: the four metrics between chess-learned and thinking-learned trajectories | unification gate; until green, quarantine stands |
| D-TRI-5 | Emulation ≠ resonance: counterfactual-goalstate emulation vs a resonance-only baseline on opponent move prediction (builds on D-SF-OPPONENT-1/3, which measured opponent-model inference as the bottleneck) | pre-registered accuracy margin; a null result demotes §2a's apex claim to [H], never silently |
| D-TRI-6 | Pyramid settlement: distribution of settle-rungs over real shader cycles (expect base-heavy); homeostatic descent verified (FLOW streak relaxes to base); elevator threshold jc-calibrated after | gates the §3a wiring; the hand-tuned threshold=2 stays labeled hand-tuned until this runs |

## §7 What this deliberately does NOT do

No new struct wrapping the SoA columns (AGI-as-glove); no kanban at the
membrane; no confirmation bookkeeping anywhere (E-ACK-ELIMINATED-1); no
VSA carrier revival (gestalt matching = ResonanceDto + palette tables,
inside a compartment); no fifth catalogue in the palette256 without an
operator ruling (30 spare atoms are the append margin, not free real
estate); no byte lands before its mint + auditor gate.
