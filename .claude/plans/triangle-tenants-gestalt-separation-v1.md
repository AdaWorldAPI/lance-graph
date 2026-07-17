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
> **D-ids:** D-TRI-1..4 (STATUS_BOARD rows, same commit).

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

## §2 Four surfaces, kept apart (the separation the operator ordered)

| Surface | Question it answers | Carrier | Explicitly NOT |
|---|---|---|---|
| **Triangle tenants** (§1) | "what do I do" — competence/policy | 3 × 12 B lanes, palette256 atoms | not memory, not resonance |
| **Episodic witness + basins** | "what happened" — factual, versioned | Lance versions (`temporal.rs` pinned reads) + KanbanMove witness chains + `part_of:is_a` basin rails (le-contract L1–L3) | not a policy store; never blended into triangle bytes |
| **Gestalt awareness** | "what does this resemble / feel like" — perspectival resonance | `awareness_dto.rs::ResonanceDto` (HDR subject/predicate/object resonance, dominant `Archetype`, divergence) + palette256 distance tables (object ↔ codebook) | not episodic replay; not the witness log |
| **Chess tenants** | domain corpus (DecisionEpisodeV1: episode/candidate/iteration/event) | OWN classids, own rows — quarantined | not thinking-row lanes, until D-TRI-4 measures |

The mirror-neuron point lands in row 3, precisely: the opponent-model
sibling SoA's meta-awareness of "what is the opponent thinking" is a
**gestalt read** (its ResonanceDto against its own codebook and context)
— NOT an episodic replay of the opponent's moves. Episodic tells you the
opponent played Nf3 twice; gestalt tells you the position *feels like* a
kingside attack brewing. Both real, different organs, different carriers
— conflating them is how "one soup" happens on the awareness side.

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

## §7 What this deliberately does NOT do

No new struct wrapping the SoA columns (AGI-as-glove); no kanban at the
membrane; no confirmation bookkeeping anywhere (E-ACK-ELIMINATED-1); no
VSA carrier revival (gestalt matching = ResonanceDto + palette tables,
inside a compartment); no fifth catalogue in the palette256 without an
operator ruling (30 spare atoms are the append margin, not free real
estate); no byte lands before its mint + auditor gate.
