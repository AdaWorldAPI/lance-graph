# Octopus — measurement-first related-work + contract audit (`octopus-causal-cot-audit-v1`)

> **Status:** MEASUREMENT REPORT. **No code. No new type. No rename.** 2026-08-26.
> **Method:** every "exists" claim below names the file it was read from this
> session; every "absent" claim is a grep that returned nothing. Where a shipped
> plan already answers a question, it is **cited, not restated**.
> **Consumes, does not re-plan:** `entropy-closure-causal-ground-v1.md` (#1057 —
> §9/§10 of the brief are already ruled there), `dacr7-band-reading-contract-v1.md`
> (the 59..63 reading contract), `alpha-channel-rung-overlay-v1.md` (the alpha
> overlay design), `counterfactual-rung3-closure-v1.md` (rung-3 runtime gap),
> `mul-calibration-not-verdict-v1.md`, `grounding-descent-cognitive-maslow-v1.md`,
> `probe-revision-attention-view-1.md`, `dialectic-engine-v1.md`.

---

## 0. The headline measurement

**`grep -ri "octopus"` over the whole tree returns ZERO hits** — no `.rs`, no
`.md`, no plan, no board entry. Octopus is a **session-external name for an
architecture that exists under other names**. Nothing in the tree has to be
renamed, and nothing in the tree currently claims the thesis. That is the
cleanest possible starting condition for a naming decision, and it means
**§17's answer cannot be "keep the established name"** — there is no
established internal name to keep.

Second headline: **no thought-graph population exists.**
`grep -niE "thought_graph|ThoughtGraph|InfluenceGraph|trace_graph|ReasoningTrace"`
over `crates/` returns nothing. F-OCT-10 ("no second world") is currently
**satisfied by absence**, not by a guard. Its risk is entirely prospective.

---

## 1. Four-ontology classification of what is actually in the tree

| Structure | File (measured) | Ontology | Evidence |
|---|---|---|---|
| `CausalEdge64` 59..60 `CausalTopology` | `causal-edge/src/layout.rs` (via `band_reading.rs:101`) | **A — World DAG** | edge-level claim about world causal structure; `Direct / IndirectKnown / IndirectUnknown / Unknown` |
| `DismechEvidence::causal_link_type` | `contract/src/dismech_evidence.rs:52` | **A — World DAG** | "maps 1:1 onto the CE64 `CausalTopology` ordinal at hydration" |
| `SupportLedger` / `SupportReceipt` | `contract/src/causal_audit.rs:250` | **A — World DAG provenance** | per-relation receipts, `basis`+`source`+`at`+`strength`, explicitly *"per-receipt, never a pre-aggregated score"* |
| `AttentionMaskSoA` | `cognitive-shader-driver/src/attention_mask.rs` | **B-adjacent — flat receipt, NOT a DAG** | SoA rows `(mailbox_id, w_slot, active, last_touched_cycle, plasticity_residual)` + LRU. There are **no edges.** |
| `BeliefArena` + `Frontier` + `ReasoningGap`/`GapKind` | `planner/src/nars/tactics.rs:88,99,151` | **B — Reasoning DAG, latent** | derivation candidates + typed gaps; edges implicit in derivation, never materialised |
| `perturbation-sim` (`PerturbationShape`, Laplacian rank-1 trip, DC cascade) | `crates/perturbation-sim/src/{perturbation,flow,cascade}.rs` | **A — World DAG (power grid)** | **not** cognitive perturbation. Naming collision is a live trap. |
| `probe_revision_attention_view` `Selector`/`ViewPlan`/`ViewEdit` | `planner/examples/probe_revision_attention_view.rs` | **B — borrowed view, probe-local by charter** | the file states it forbids prescribing a production type |
| — | — | **C — Attribution graph** | **ABSENT. Correct. Do not build.** |
| — | — | **D — Scoring graph** | **ABSENT.** CausalPhys-style gold DAGs would land here, as grading fixtures, never as storage. |

**Verdict on the brief's strong expectations: 4/4 confirmed.** CE64 = World DAG.
Alpha = receipt (weaker than expected — a flat SoA, not even a projection).
No attribution graph. No scoring graph.

**One correction to the brief.** The brief says alpha is an "attention
projection". Measured, `AttentionMaskSoA` is a **flat LRU residency table**. It
records *that* a mailbox was touched and *when*. It has no topology to project
and no link to a next transition. Calling it a projection over-claims by one
level.

---

## 2. Alpha in the next-transition path — the load-bearing question

This is the audit's central negative finding.

**Measured consumer set of `attention_mask` outside its own module:** four hits,
all of them `pub mod` declarations or doc-comments (`lib.rs:93,94`,
`mailbox_soa.rs:11` — which says *"NO AttentionMask/LRU"* —, `attention_facet.rs:103`).
**No production call site consumes `AttentionMaskSoA` to decide anything.**

Meanwhile the actual next-transition decision is measured elsewhere and does
**not** read alpha:

```
contract/src/dispatch_mode.rs
    elect_mode(Domain) -> DispatchMode
        Clear       -> Saccade        (one tactic, select_tactic)
        Complicated -> Sweep
        Complex     -> Sweep
        Chaotic     -> Stabilize
        Confused    -> FieldGather
```

`Domain` is derived from **gate + surprise + contradiction**
(`dispatch_mode.rs:28`), not from the attention mask. `select_tactic`
(`contract/src/materialize.rs:77`) is the saccade.

**Conclusion (F-OCT-1 pre-verdict):** alpha is currently **observational, not
load-bearing** — and not marginally so. There is no edge from the attention
carrier into the transition function at all. An intervention on alpha today
would produce a *silent* arm in both the relevant and irrelevant conditions,
i.e. F-OCT-1 fails and F-OCT-2 passes vacuously. **A vacuously-passing silent
arm is exactly the `closed_class_guess` 150/150 defect one level up** (CLAUDE.md
falsifiability rule), so F-OCT-2 must not be reported as green until F-OCT-1 is.

This is not a defect to fix by wiring alpha into dispatch to make the falsifier
pass. That would be building the test's answer. It is the *measurement* that
tells you the thesis is currently **unearned on axis A**.

---

## 3. Graph-of-Thoughts vs Graph-CoT vs Causal-CoT vs the tree

| | Nodes | Edges | Built from | Does the tree do this? |
|---|---|---|---|---|
| **GoT** (Besta) | thoughts | generation dependency | the run | **No** — no thought vertex population exists |
| **Graph-CoT** (Jin) | external KG | pre-existing relations | the substrate | **Yes, structurally** — traversal over a resident graph is the whole spine |
| **Causal-CoT** (Tie, ICLR'26 submission, not accepted) | world variables | LLM-proposed causal edges | the query | **No, and must stay No** — Stage II invents mediators; that is a second world |
| **Causal Graphs Meet Thoughts** (Luo/Zhang/Li) | filtered KG edges (`Causality(r) ≥ θ`) | retrieved paths | an existing KG | **Closest external precedent.** Completion is *retrieval*, not invention — the resident-world doctrine, published. |

**Finding:** the nearest published relative is **Luo et al.**, not Causal-CoT.
The distinguishing feature is not "we do causal CoT" — it is **completion by
retrieval over a resident typed substrate, with a per-edge epistemic topology
(CE64 59..60) and a per-edge permission band (61..63) that no published system
carries.** Those two fields are the actual novelty claim.

Causal-CoT's contribution worth keeping is the **pipeline shape**, with Stages
I–II demoted to views:

```
EXTRACT     claimed alpha/surprise states from the existing receipt
COMPLETE    via existing views only — no invented mediators
CONSTRAIN   target = the next predicted transition
VERIFY      do(claimed alpha) moves it; do(matched irrelevant) does not
CONTROL     same topology, shuffled edges, must lose
```

---

## 4. CausalPhys metrics → measured native carriers

| CausalPhys metric | Native carrier that already exists | State |
|---|---|---|
| ACC | outcome of the cycle | exists |
| **Entity Faithfulness** | `AttentionMaskSoA.entries[].mailbox_id` — *which* mailbox the eye touched | **carrier exists, unread by anything** |
| **Relation Awareness** | `CausalEdge64::topology()` on the edges actually traversed | **exists and is typed** — strictly stronger than scoring prose |
| **Description Correctness** | `ClassView::band_reading` per `(classid, rail)` — did the active domain read the field under the right lens | **exists**, `band_reading.rs` |
| Interventional Faithfulness | — | **no carrier.** Hypothesis only. |
| Epistemic-Topology Faithfulness | CE64 59..60 vs traversal | carrier exists; no comparator |
| Reasoning-Band Faithfulness | CE64 61..63 `ReasoningBand` | carrier exists; **no permission check anywhere** (grep: every `ReasoningBand` hit is a probe, a test, or a doc-comment — `recipe_vocab.rs:63` explicitly says it *does not write* one) |
| Counterfactual Necessity | `AscOutcome` / `asc_challenge` (`tactics.rs:463,479`), `cr_synthesize` | partial — challenge exists, *removal* does not |

**The strongest honest claim available today:** the tree can score Relation
Awareness **typed** rather than textually, because the relation carries its own
epistemic topology. That is a real advantage over CausalPhys and it needs no
new type.

**The weakest link:** `ReasoningBand` is minted, readable, round-trip-tested
(`v2_layout_tests.rs:480`), asserted-untouched by the control loop
(`probe_revision_kanban_hinge.rs:1404`) — and **gates nothing**. F-OCT-7 has a
carrier and no mechanism.

---

## 5. #1057 governs §9/§10 — do not re-derive

`entropy-closure-causal-ground-v1.md` (merged as PR #1057) already ruled:

- 59..60 semantics, **two shipped lenses** (`CausalTopology` / `TrustTexture`,
  ordinal-identical), reading declared per `(classid, rail)`, projection
  fallible, `EdgeProvenance::Unknown` **refuses**, v1 temporal taint reaches V3
  through the lift. **No new bits.**
- Six uncoordinated entropy surfaces; no canonical H, and that is acceptable
  because the fields differ.
- `SettlementCell` (`contract/src/settlement.rs:79`) is the shipped
  cross-product — closure density × evidence competence → `Crystal` / `Glass` /
  `GroundedUnresolved` / `Fog`, with `is_glass()` at `:171`.

**This audit adds exactly one thing to that ruling: F-OCT-6.** Two basins with
identical entropy but `Direct`-dominant vs `Unknown`-dominant topology must
settle and route differently. `settlement.rs` computes the cell from four
signals; measured, **`SettlementCell::cell()` never reads a `CausalTopology`.**
So F-OCT-6 is, as of today, **expected to FAIL** — entropy and topology do not
yet meet in the settlement computation. That failure is the useful result.

---

## 6. Sudoku walker — measured fit is better than expected

The brief's loop maps onto shipped types with one gap:

| Loop stage | Shipped carrier |
|---|---|
| DETECT | `ReasoningGap` + `GapKind::{NoSharedMiddle, NoSibling, NoAbstraction, HubExcluded, BudgetExhausted}` (`tactics.rs:99`) |
| BOUND | CE64 59..60 + `Throttle{c_min, budget, hub_indegree}` (`tactics.rs:115`) |
| PROPOSE | `rcr_abduce` / `tr_diverge` / `cas_abstract` → `Frontier` of `Candidate` |
| FILTER | `Throttle`, `BeliefArena`, `TruthValue` |
| GATE | `ReasoningBand` — **carrier only, no gate** |
| TEST (removal) | **ABSENT** — `asc_challenge` challenges a belief; nothing removes a mediator and measures degradation |
| OUTCOME | `AscOutcome` (`tactics.rs:463`) |

**"Hole has shape before identity" is already implemented**: `GapKind` types the
hole by *why* it is a hole, before any candidate exists. `HubExcluded` and
`BudgetExhausted` are the honest "narrowed but unresolved" states — F-OCT-9's
carrier already ships.

Two genuine absences: the **band gate** and the **counterfactual removal test**.
Both are named in `counterfactual-rung3-closure-v1.md`, which measured that
rung 3 has *"a fully specified encoding and a fully absent runtime"* and that
`from_mantissa(−6)` silently returns `Synthesis` — a **decoder direction
inversion**. That defect is upstream of F-OCT-8 and should be fixed there, in
that plan, not here.

---

## 7. Exists / hypothesis split

**Exists (measured):** CE64 59..60 + two-lens reading contract; 61..63
`ReasoningBand` as a *carrier*; `SettlementCell` 2×2; `SupportLedger` receipts;
`GapKind`/`Frontier`/`Throttle` sudoku machinery; `elect_mode` → `DispatchMode`;
`AttentionMaskSoA` as a flat residency receipt; `MulAssessment` with
`DkPosition` (MountStupid → Plateau) and `GateDecision::{Flow,Hold,Block}`
(`mul.rs:50,100,158`); shuffle/null-result discipline as house practice
(`jc/examples/ontology_locality_probe.rs` carries an HONEST SCOPE CAVEAT header).

**Hypothesis only:** alpha being load-bearing; band permission gating; mediator
removal; entropy × topology meeting in settlement; domain lenses changing
traversal (see §8); anything named "Octopus".

---

## 8. Domain-awareness — measured, and weaker than the thesis needs

The brief's shape (one resident world, N domain ClassView lenses, shared causal
walk) is **structurally present**: `ClassView::band_reading` is a per-`(classid,
rail)` **total, zero-fallback** lookup, and projection failure is a hard fail,
not a plausible value. That is a real domain lens over one world.

But `elect_mode`'s `Domain` (`Clear/Complicated/Complex/Chaotic/Confused`) is
**Cynefin, not subject-matter domain** — it is a state classification, not a
semantic lens. The two words collide in the tree. **Any paper text using
"domain-aware" must disambiguate, or it will be read as Cynefin by anyone
inside the codebase.** Measured: `Domain::Confused` is even labelled *"the
defect zone"*, which no subject-matter domain would be.

---

## 9. Smallest board correction required

1. This file (the plan) + `INTEGRATION_PLANS.md` prepend.
2. `STATUS_BOARD.md`: D-OCT-1..10 rows, all `Queued`, one per falsifier.
3. `ISSUES.md`: `ISS-ALPHA-NOT-LOAD-BEARING` — the attention carrier has no
   production consumer; the thesis's axis A is unearned.
4. `ISSUES.md`: `ISS-REASONING-BAND-GATES-NOTHING` — 61..63 is minted, tested,
   and inert.
5. **No `EPIPHANIES.md` entry yet.** Two of the three headline findings are
   *absences*; an absence becomes an epiphany when a falsifier makes it a
   measurement, not before.

Regenerate `SUPERSESSION-INDEX.md` in the same commit (this PR adds a plan).

---

## 10. Falsifier table

| id | Claim | Current carrier | Intervention | Expected today | What failure means |
|---|---|---|---|---|---|
| **F-OCT-1** | alpha is load-bearing | `AttentionMaskSoA`; transition = `elect_mode` | perturb one claimed entry, read next `DispatchMode` | **FAIL — no edge exists** | alpha is instrumentation; axis A unearned |
| **F-OCT-2** | irrelevant arm silent | same | perturb a matched unclaimed entry | passes **vacuously** | must be reported RED while F-OCT-1 is red |
| **F-OCT-3** | relation faithfulness | `CausalEdge64::topology()` on traversed edges | right entities, wrong edges, right answer | untested | right-answer scoring hides wrong-relation reasoning |
| **F-OCT-4** | topology beats shuffle | none (no influence graph) | n/a until a projection exists | **N/A — nothing to shuffle** | guards the prospective F-OCT-10 risk |
| **F-OCT-5** | stage diagnostic completeness | `GapKind`, `AscOutcome`, `Frontier`, attention rows | mask each stage, localise a failure | untested | replay ≠ localisation |
| **F-OCT-6** | twin basin routes differently | `SettlementCell::cell()` + CE64 59..60 | equal entropy, `Direct`- vs `Unknown`-dominant | **FAIL — cell() never reads topology** | entropy and causal ground do not yet meet |
| **F-OCT-7** | band permission | `ReasoningBand` 61..63 | Surface-band mediator offered for a causal hole | **FAIL — no gate** | the band is decoration |
| **F-OCT-8** | counterfactual necessity | `asc_challenge` (partial) | remove accepted mediator, measure degradation | **FAIL — no removal path**; blocked on the `from_mantissa(−6)` inversion | acceptance is decorative |
| **F-OCT-9** | unknown may remain unknown | `GapKind::{HubExcluded,BudgetExhausted}` | narrow without resolving | **likely PASS** — the only green one | fabricated certainty |
| **F-OCT-10** | no second world | absence + `SupportLedger` | propose any new graph; demand the derived-view argument | passes by absence | a second canonical population |

**Six expected-fail, one N/A, two vacuous/untested, one likely-pass.** That is
the honest state, and it is the report's most useful sentence.

---

## 11. Naming recommendation

**Do not adopt "Octopus" as a public name, and do not rename anything.**

- The tree has **no** Octopus, so nothing is preserved by adopting it and
  nothing is broken by declining.
- "Domain-Aware Causal CoT" is **actively ambiguous in-tree** — `Domain` is a
  shipped Cynefin enum (§8).
- The measured architecture is **Graph-CoT-shaped** (traversal over a resident
  substrate), closest to Luo et al., **not** Causal-CoT-shaped (no construction,
  no completion-by-invention) — and that is a strength to state, not hide.

The defensible description of what is measured, today, is:

> **Retrieval-completed causal traversal over a resident typed substrate, with
> per-edge epistemic topology and per-edge reasoning-permission bands.**

The differentiator is the **two CE64 fields**, not the word "causal" and not the
word "thought". The thesis sentence in §14 of the brief —
*"…only if its internal state is interventionally load-bearing and its traversal
is faithful to the causal topology it claims to reason over"* — is **correct as
written and currently not earned on either axis.** Axis A fails at §2; axis B
has typed carriers and no comparator at §4.

**Recommendation: keep "Octopus" as the internal working name for the audit
track only, publish nothing until F-OCT-1 and F-OCT-3 are green, and revisit the
name from the measurement rather than from the metaphor.**

---

## 12. What this audit explicitly did not do

No code. No new type, DTO, enum, or bit. No rename. No new graph population. No
`glass_gap()`. No CE64 encoding change. No import of Causal-CoT's generated DAG
or AIG's per-trace object. GFCM stays outside CE64 semantics — measured, no
GFCM-shaped surface exists in the tree, so there is nothing to isolate yet.

---

## 13. Causal-Audit / TAP — the admission-vs-budget correction

The brief treated "Exact / CloseHit / Bridge / None + Top-K frontier" as one
idea. **The published ablation says it is two stacked filters, and only one of
them is load-bearing** (WIQA / Llama-3.1-8B):

| Component | Variant | Acc | Path Reach |
|---|---|---|---|
| TAP | target-aware rank | 67.92 | 96.23 |
| TAP | **random prune** (same K) | 64.60 | **97.17** |
| FGVA | 3-axis alignment | 67.92 | 96.23 |
| FGVA | **simple alignment** | **55.66** | **14.62** |
| CFA | counterfactual audit | 67.92 | 96.23 |
| CFA | no audit | 60.38 | 97.28 |

Read the Reach column, not the Acc column:

- **Random prune keeps reach (97.17 > 96.23).** TAP does not win by covering
  more — it wins by *which* K slots get filled, for **+3.3pp**.
- **Simple alignment destroys the graph** — reach 14.62%. The three-axis
  admission test is the load-bearing piece.
- **CFA is precision on already-reached paths** (+7.5pp, reach unchanged) —
  downstream of pruning, never a substitute for it.

**Two rules this imposes on anything we lift:**

1. **FGVA before TAP — admission before budget.** Top-K without the typed
   3-axis class reproduces "simple alignment" and the topology never reaches the
   target. Ranking is secondary.
2. **3.3pp is the lower bound any claimed topology must clear.** Random-prune is
   the closest control in this literature to our shuffled-topology null, and it
   is **weak** — it randomizes which LLM-proposed survivors get slots, not
   whether the edge set means anything. **F-OCT-4 stays stricter and stays
   unrun**, and it must beat a K-slot shuffle by *more than 3pp* to count.

### The three axes, and why State-Conflict matters here

`Φ_align(v, Y_b)` tests entity (Exact/Partial/None), quantity
(Exact/Subset/Agg/None), state (Match/Conflict/None); class by Eq. 3, with
**any state Conflict a hard kill even when entity matches exactly**. Disposition:
`Exact` terminal, `CloseHit` landing pad, `Bridge` the *only* expandable class,
`None` dropped before the visited-set (irrecoverable — `Avoid` is injected into
the expander).

**That hard kill is an accidental silent-irrelevant arm**: a node about the right
object with the wrong polarity is not a near-miss, it is noise. This is exactly
the discrimination F-OCT-2 demands, and it is the part worth stealing.

### Mapped onto alpha — the target is the transition, not a world variable

```
ENTITY     is this the state the next transition is ABOUT, or a neighbouring topic?
QUANTITY   same grain as the claimed surprise, or an aggregate/subset proxy?
STATE      polarity compatible with the predicted move, or in conflict?
```

| Class | Meaning for a claimed alpha state | Required test |
|---|---|---|
| Exact | is the transition's object, grain and polarity | `do(alpha)` **must** move the next saccade (F-OCT-1) |
| CloseHit | right object, wrong grain | not load-bearing until re-grained; bridge only if Ψ is itself interventional |
| Bridge | partial overlap | may appear in a **diagnostic projection**; never a resident node (F-OCT-10) |
| None (incl. polarity conflict) | irrelevant | `do(alpha)` **must remain silent** (F-OCT-2) |

**Steal:** the 3-axis admission grammar as a typed view over existing receipts;
`None` as a hard drop including state-conflict; `Exact` terminal for *this*
diagnostic hop, not a stored node; the 3pp floor.
**Leave:** `Φ_rank` as a second LLM call (prompt unpublished, and worth only
3.3pp); CloseHit→`Y_b` bridging as a new edge type; the BFS graph population and
its `D=4` materialisation; the 136s/instance call graph; WIQA 67.92 as a target.

**The honest limit, and it lands on us:** both `Φ_align` and `Φ_rank` are LLM
judges. Afolabi applies to the *filter* as well as the trace — a node can look
`Exact` and still be causally decorative. Causal-Audit's only interventional step
(CFA) runs **after** the graph is built, on paths, never on the prune decision.
So the admission grammar is a **shape to borrow, not evidence to inherit**: in
our tree it must be typed against `CausalTopology` / `ReasoningBand` — which are
stored fields, not judgements — rather than against a model's opinion. That
substitution is the whole reason the borrow is defensible here and was not there.

---

## 14. Cross-repo measurement — MedCare-rs and OGAR

Both repos were attached and cloned this session (`medcare-rs` @ `572eb87`,
`OGAR` @ `c02efa1`, read-only).

### 14.1 The domain-lens thesis is empirically supported — in the CONSUMER, not the spine

`medcare-rs/crates/medcare-first-thought/src/plateau.rs` carries a measured,
documented **domain-conditioned reading ruling** that is the strongest evidence
in the whole stack for §13 of the brief:

> *"Qualia is not wrong in lance-graph — it is one of the substrate's default SoA
> tenants… It was wrong here, for **medicine**: a statistical proof of an
> ontological relationship is **evidence with a truth value**, not a felt state.
> So for a clinical thought the evidence rides the **edge tenant** and the
> **qualia tenant stays ZERO**."*

That is precisely "one resident world, domain-specific lens changes the
admissible reading" — same SoA columns, same `CausalEdge64`, and the medical
domain **rules one tenant out**. The world identity, the storage and the edge
encoding are untouched, exactly as §13 predicted.

**Correction to §8 of this audit:** domain-awareness is therefore *not* purely
hypothetical. It is **shipped as a documented per-domain reading discipline in a
consumer**, and **absent as a mechanism in the spine** — nothing in
`lance-graph` enforces or even records that medicine zeroes qualia. It is a
convention held by prose. That gap is the real finding.

`medcare-first-thought` also carries `patient_tenant(patient_id) -> MailboxId`
(one patient = one mailbox = one kanban board) and grounds to `Disease` via a
data-as-config `DISEASES` table, *"never a hand-mapped literal"*, returning
`None` for an unmodelled ICD — described in-file as **"an honest hole"**. That is
F-OCT-9's discipline, independently arrived at, in a second repo.

### 14.2 OGAR is the ClassView producer — and it carries no band reading

`OGAR/crates/ogar-class-view/src/lib.rs` (662 lines) ships `OgarClassView` with
`known_class_ids()` and `object_view(class) -> Option<&ObjectView>`.
**Measured: `grep -rn "band_reading" OGAR/crates/**/*.rs` returns NOTHING.**

So the two halves of the domain lens live in different repos and do not meet:

| Half | Where | State |
|---|---|---|
| `ClassView::band_reading` — which lens applies per `(classid, rail)` | `lance-graph/crates/lance-graph-contract/src/band_reading.rs` | shipped, total, zero-fallback, fallible |
| `OgarClassView` — the actual minted class views | `OGAR/crates/ogar-class-view` | shipped, **never populates a band reading** |

**This is the mechanism gap behind F-OCT-7 stated precisely.** The contract can
*ask* a class which lens it declares; the producer that mints classes never
*answers*. A `ReasoningBand` permission gate cannot be built until OGAR mints
the declaration — and per the missing-capability STOP rule that lands as an
OGAR-tier change first, never as a hand-rolled default in the consumer or in the
walker.

`medcare-rs` likewise has zero hits for `CausalTopology` / `ReasoningBand` /
`SettlementCell`; its causal surface is `CausalEdge64` carrying NARS `(f,c)` on
the EdgeColumn (`plateau.rs`), plus `medcare-dismech` and
`medcare-cohorts/src/provenance.rs`. **The epistemic-topology field is written by
nobody in the two consumers measured.**

### 14.3 Consequence for the thesis

Axis B ("traversal is faithful to the causal topology it claims to reason over")
is weaker than §4 alone suggested. The topology field exists and is typed, but
across three repos **no producer stamps it and no consumer reads it as a
constraint**. The strongest true statement is:

> The substrate reserves a per-edge epistemic topology and a per-edge reasoning
> band; a domain-conditioned reading discipline exists and is honoured by
> convention in MedCare; no code path yet stamps, gates on, or is falsified by
> either field.

That is a real architecture with two unearned axes — which is a publishable
position only *after* F-OCT-1, F-OCT-3 and F-OCT-7 have run.

### 14.4 Board correction (cross-repo)

6. `ISSUES.md`: `ISS-BAND-READING-UNMINTED-IN-OGAR` — `band_reading` has a
   consumer-side contract in lance-graph and no producer in OGAR; F-OCT-7 is
   blocked on an OGAR-tier mint, per the missing-capability STOP rule.
7. `ISSUES.md`: `ISS-DOMAIN-LENS-BY-CONVENTION-ONLY` — MedCare's
   medicine-zeroes-qualia ruling is prose in one consumer, unenforced and
   unrecorded by the spine.

No changes are proposed to MedCare-rs or OGAR in this PR; both were read-only.
