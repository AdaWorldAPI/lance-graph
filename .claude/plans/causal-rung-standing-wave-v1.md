# Plan — causal rung trajectories + standing-wave awareness (p64→v3 cognition layer)

> **Status:** CONJECTURE throughout — every phase gated on a registered probe
> (§6); no phase is a FINDING until its probe is green. **Additive only.**
> **Builds on (consumes, does not re-plan):**
> `.claude/plans/soa-32-tenant-awareness-redundancy-v1.md` (M20 — A1 `SpoFacet`
> SHIPPED; A2–A7 queued behind the `v3-envelope-auditor` + batched OGAR mint),
> `lance_graph_contract::selection` (#776 — `NamedView`/`ViewRegistry`/
> `walk_rails`, metric-agnostic, dependency-inverted `RailGraph`),
> `lance_graph_contract::distance` (the palette256² cosine-replacement table;
> generalizes to larger codebooks — same N×N table),
> `temporal.rs` sorted stream (`E-MARKOV-TEMPORAL-STREAM-1` — version-range
> reads, replayable, Chapman-Kolmogorov exact by stream order),
> `causal_edge::pearl::CausalMask` (the 2³ ladder over {S,P,O}).
>
> **Operator directive (2026-07-21, verbatim):** *"reimagine the SPO 2^3 rung
> decomposition ladder for causality trajectories — remember we amortize 8
> views over the same SPO to do e.g. counterfactual"*; *"the awareness of
> itself as in causaledge64 with mantissa/signed, but now benefit from better
> substrate"*; *"expand on that in the markov chain superposition of the
> standing wave of these awarenesses of the SoA row after temporal.rs provided
> the thing 'for free' that MIT-proposed causality learning needs gigantic
> amounts of compute for — and we even do it for Text as a stream of temporal
> updates"*; *"the endgame is the graph reasoning about itself — finding
> AriGraph basins that now could also serve causality edges — use the thinking
> atoms, NARS recipes and frozen/learned/explore styles triangle to do entropy
> work over the awareness of the whole text corpus, create a knowledge graph
> with causality and rung decomposition, providing candidates to reasoning
> which interconnect through the text — and the texture using qualia provides
> the gestalt awareness for MetaCognition and the Meta-Uncertainty Layer."*

## §0 What this is (and the anti-pattern it replaces)

The p64 way packed causality into one `CausalEdge64`: SPO as 3×u8 (bits 0–23),
the 2³ `CausalMask` as 3 bits (40–42), the NARS rung as a signed 4-bit
inference mantissa (46–49). One edge = one frozen causal state; 8 rungs would
cost 8 edges; the counterfactual had no witness arm to contrast against.

The v3 way (this plan): **the 2³ ladder becomes 8 mask-views amortized over ONE
`SpoFacet` register**, causality *trajectories* are walks over those views
along the `temporal.rs` stream, and the awareness the walk produces is stored
back into the same SoA the walk reads — the graph reasoning about itself.

Hard rules carried from the #774 post-mortem (`E-MORTON-CASCADE-CONJECTURE-
DOWNGRADE-1`): readings are **ClassView-elected, never byte-sniffed**
(le-contract §2 slot purity; the #776 carving-verification finding); the metric
is **only** `contract::distance` (never a re-modeled table); no
"wiring/replaces" language until a caller exists; FINDING only after a probe.

## §1 Grounding — every surface is shipped or already-planned

| Surface | Where | Role here |
|---|---|---|
| `CausalMask` (2³, 8 rungs) | `causal-edge/src/pearl.rs` | the ladder's vocabulary — "each bit enables one SPO plane in distance computations" |
| `SpoFacet` (3×SPO 8:8 + 3×AriGraph-witness 8:8) | `contract::awareness_facet` (M20 A1, SHIPPED) | the register the 8 views amortize over |
| `NamedView` / `ViewRegistry` / `walk_rails` / `RailGraph` | `contract::selection` (#776) | the amortization + trajectory ergonomics (masks, no query doc, no materialized tree) |
| `Distance` / palette256² table / `mean_similarity_fisher` | `contract::distance` | the per-rung metric — one table read, never a float re-model |
| `temporal.rs` sorted stream | lance-graph core (`E-MARKOV-TEMPORAL-STREAM-1`) | the version-ordered stream: replayable, C-K exact by order |
| A2 `PearlRungFacet`, A3 `NarsTruthFacet`, A5 `StreamCycleFacet`, A6 `DirectionInferenceFacet` | M20 plan §2 (QUEUED, auditor+mint gated) | the *accumulated* awareness storage this plan's cognition writes into |
| Rung-content ladder: 144 verb atoms (rung 2) / **34** NARS tactic recipes (rung 3, THE runbooks) / frozen·learned·explore triangle (rung 4) | `persona-vs-rung-ladder.md` + `cognitive_palette` + P4 lanes | the entropy-work drivers (§5). NOTE: the adjective-36 in `contract::thinking` is the SEPARATE persona storyline per the operator demarcation — not used here unless separately elected |
| `MulAssessment` / `TrustTexture` / `DkPosition` + `QualiaColumn` | `contract::mul` + planner `mul/`, `QualiaI4_16D` | the gestalt / meta-cognition consumer (§5c) |

## §2 The reimagined 2³ ladder — 8 views amortized over one register

The 3-bit `CausalMask` maps bijectively onto 8 **canonical `NamedView` masks**
over the facet's rails (subject=0, predicate=1, object=2; witness=3,4,5). The
register is written once; the rungs are projections, not storage:

| Mask | Pearl level | View selects | `distance()` reads |
|---|---|---|---|
| 000 `None` | Prior | ∅ | the marginal |
| 001 `O` | Outcome marginal | {2} | object pair |
| 010 `P` | Intervention marginal | {1} | predicate pair |
| 011 `PO` | **L2 Intervention** P(Y\|do(X)) | {1,2} | P·O planes |
| 100 `S` | Entity marginal | {0} | subject pair |
| 101 `SO` | **L1 Association** P(Y\|X) | {0,2} | S·O planes |
| 110 `SP` | Confounder detection | {0,1} | S·P planes |
| 111 `SPO` | **L3 Counterfactual** P(Y_x\|X',Y') | {0,1,2} **vs {3,4,5}** | `distance(spo(), witness())` |

The counterfactual is why the witness triple exists: semantic SPO ("THIS
subject under THAT predicate → predicted object") contrasted against the
AriGraph witness (what actually happened) — the two arms of P(Y_x|X',Y'),
each pair scored by one table read.

**Computed vs accumulated (the A2 relationship).** The mask ladder COMPUTES
the 8 projections on demand from A1 — zero storage. A2 `PearlRungFacet` STORES
accumulated rung state (marginals that are statistics over many cycles, not
recomputable from the current triple). They are one system: mask-ladder =
access path; A2 = memory. Probe D-CSW-1 decides what A2 must hold beyond what
the mask computes.

**O1 (open decision, recommendation attached):** the 8 ladder masks are
**canonical contract constants** (Pearl's ladder does not vary per class), but
*which classes read their register as `SpoFacet` at all* stays a per-class
OGAR mint (Place 2, per M20 §0.5) — canonical masks, per-class election.

## §3 Self-awareness on the better substrate — consumed from M20, not re-planned

The CausalEdge64 "awareness of itself" — the signed mantissa (direction × NARS
rule, bit-49 sign) and freq·conf truth — is EXACTLY A6 `DirectionInference-
Facet` (dir 3b + mantissa 4b → full-width 6×(8:8)) and A3 `NarsTruthFacet`
(16b → per-plane basin:strength). This plan adds **no facet**: A6/A3 land
through the M20 sequencing (auditor verdict → batched mint → jc cert). What
this plan adds is their *consumer*: the trajectory walk (§4) reads A6 to know
HOW a triple came to be believed, and NARS composition (deduction/abduction/
revision over rung readings) writes revised truth back through A3's lane.
Facet says WHAT; rung-awareness says HOW-KNOWN. Together: one row's awareness
quantum, at honest width.

## §4 The standing wave — Markov superposition over `temporal.rs`

Each SoA row's awareness across the version stream is a Markov chain (C-K
exact by stream order — the `E-MARKOV-TEMPORAL-STREAM-1` strengthening). The
**standing wave** is the superposition of per-version rung readings over a
version window (`QueryReference::at(v, rung)` + deinterlace — A5
`StreamCycleFacet`'s ±5/±50/±500 windows are its accumulated form): per rung,
the component of the reading that PERSISTS across the window vs the transient.

**Why this is "for free" (stated falsifiably).** Observational causal
discovery pays its super-exponential cost mostly to ORIENT edges — searching
over orderings/DAGs. A sorted temporal stream IS an ordering: cause precedes
effect, so orientation collapses to a replayable version-range read. Text as a
stream of temporal updates (each sentence-commit = a version tick) gives the
corpus this ordering natively. The claim is NOT "causal discovery is solved";
it is: *orientation search is eliminated; identification reduces to per-rung
stability under the stream* — and D-CSW-1 must show that per-rung standing-wave
stability separates causal from coincidental pairs better than single-cycle
readings on a real text stream, or the claim dies.

## §5 Endgame — the graph reasoning about itself

**a) AriGraph basins as causality edges.** Every facet's witness half already
points into AriGraph. Witness triples that co-occupy a basin (`part_of:is_a`
rails, le-contract L1–L3) AND survive the standing wave at the interventional/
counterfactual rungs are promoted to **causal-edge candidates carrying their
full rung decomposition** — edges derived from the graph's own awareness rows,
stored back as rows the next walk reads. Basins double as causality edges.

**b) Entropy work with the rung-content ladder.** The exploration driver over
the corpus awareness field: rung-2 thinking atoms (144 verbs) name WHAT a
candidate does; rung-3 NARS tactic recipes (the 34 runbooks) pick HOW to test
it (which inference composition); rung-4 frozen/learned/explore triangle picks
WHERE to spend entropy reduction — high-entropy basins draw the explore lane,
confirmed low-entropy basins freeze (`promote_family`). Output: the knowledge
graph with causality + rung decomposition attached; its edges are candidates
to reasoning, interconnected through the text spans that witnessed them.

**c) Qualia texture → gestalt → MetaCognition/MUL.** The qualia texture over
the awareness field (per-row `QualiaColumn`) aggregates to a gestalt reading
consumed by `MulAssessment` (`TrustTexture`, `DkPosition`, homeostasis gate):
the Meta-Uncertainty Layer sees the corpus's TEXTURE, not per-edge numbers —
meta-cognition over the whole field, closing the loop.

## §6 Probes (registered pass/fail BEFORE any code; the probe is the next deliverable)

| Probe | Claim under test | Pass | Kill |
|---|---|---|---|
| **D-CSW-1** | standing-wave rung stability separates causal from coincidental pairs on a REAL text stream (labeled pairs; certified distance table; real `temporal.rs` versions) | separation (AUC or rank ρ) beats single-cycle reading AND the p64 3-bit-mask baseline, margin registered at probe write-time | no separation, or ≤ baseline |
| **D-CSW-2** | basin co-occupancy + rung survival predicts causal-edge candidates | precision vs a labeled candidate set beats basin-only and rung-only ablations | ablations equal or better |
| **D-CSW-3** | (= M20 D-AW-5 extension) jc reliability: the full-width amortized ladder carries awareness the 64-bit cram lost | jc battery (Cronbach α / ICC) shows non-redundant width vs CE64-derived readings | jc says the 64-bit cram was sufficient — then M20's conjecture stands and this plan's storage claims shrink to match |

## §7 Honest boundary — what this plan does NOT do

- **No bytes land** before the `v3-envelope-auditor` verdict + batched OGAR
  mint (M20 sequencing owns A2–A7). No layout, offset, or tenant change here.
- **No new struct wrapping the SoA columns** (AGI-as-glove); no new trait on
  `ClassView`; rail/reading knowledge enters only via `RailGraph` /
  ClassView election.
- **No CausalEdge64 deletion** — kept for reference per the M20 operator
  directive; p64→v3 is measured (D-CSW-3), not asserted.
- **No "wiring/replaces/lands" claims** — every artifact is CONJECTURE until
  its probe reports, and consumers are named only when a caller exists.
- **The 36 adjectives stay out** — rung-3 recipes are the 34 runbooks; the
  `contract::thinking` adjective-36 is the separate persona storyline unless
  the operator separately elects it.
