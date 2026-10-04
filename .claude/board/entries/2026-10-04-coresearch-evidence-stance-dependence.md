# Co-research: evidence stance, dependence, reliability and pooling across five carriers (2026-10-04)

**Status:** EXPLORATION MAP — ratifies nothing. The operator chooses which rows go forward.
One row is MEASURED (X17). Harness: `.claude/agents/coresearch-council.md`.
Inputs:
- the INDRA harvest, `.claude/harvest/indra-reference-wiring.md`;
- the board entry `2026-10-03-indra-evidence-socket.md`;
- PRs #1317 and #1318.

## Question, after the premise gate

INDRA showed four evidence properties a knowledge-assembly system needs: polarity,
dependence, source reliability, and pooling across a hierarchy. Where should each one
live here?

The premise gate returned **PREMISE-SPLIT**.

"Polarity" named three different properties:
- **Per-evidence stance:** for or against, on one piece of evidence.
- **Derived depth:** the best support depth and the best falsifier depth, on the
  probe-local Tarski reading.
- **A9 `Contradiction`:** a pointer to a dissenting peer event in the ±8 window. A9's
  sign means time direction, not for/against, and its only production writer
  (`witness_fabric.rs` `elect_peers_lens`) reads no receipt.

"Two planes" was really five carriers:

| | carrier | code | holds |
|---|---|---|---|
| C1 | receipt ledger | `causal_audit` | per-receipt evidence on a relation |
| C2 | A9 window | | pointers into the ±8 window |
| C3 | Tarski statement row | | derived support / falsifier depth |
| C4 | horizon masks | `revision.rs` | reader-relative echo |
| C5 | `Stamp` + CE64 W-slot | | observation-source bits; edge's corpus-root handle |

The question was re-asked as Q1 stance, Q2 dependence (three kinds: source→source
quotation, shared observations, reader-relative echo), Q3 reliability, Q4 refinement
pooling, and Q5 cross-carrier identity.

## Correction this run forced (landed in #1318)

`SupportReceipt::source` was documented as "name the observation". Internal prior art
(`.claude/knowledge/parked-designs-841-856.md` §(a)) shows `EvidenceSourceId` was meant
as **attribution**: who attested — a paper, sensor or submitter.

A per-observation event id is a different type, and it is not built yet. With the old
wording, two sentences from one paper would count as two independent sources. The doc
now reads "name who attested, not who read it". Behaviour is unchanged. Outside
precedent agrees:
- Biolink: an aggregator re-serving a fact is still one source.
- scite: counts citing statements and distinct papers separately.

## Exploration map

Verdict vocabulary:
- **ADOPT-NOW:** PASS, small, offline.
- **PROBE:** measure first.
- **PARK:** not now; the reason is given.
- **SKIP:** trap, duplicate or anchor conflict.

Reviewer verdicts are given as bridge / firewall.

**X17 — observation pooled into a derived belief that already used its source. MEASURED.**
- Verdict: OPEN defect, operator decision.
- Measured in a scratch copy of `deepnsm-v2` (nothing committed), against figures
  pre-registered before the run.
  - A→C is derived from source 0 + source 1: c = 0.6561, empty stamp, premises [0, 1].
  - A source-0 observation of A→C then gives `Revised`, c = 0.9160, above the input's
    0.9. That is identical to a fresh source 7.
  - The stamp ends as `0x1`, so source 1 vanishes from provenance.
- Silent twin: source 7 behaves as expected. Fold twin: source 64 = source 0.
- Planner copy (`nars/belief.rs:193`): the same gap **by reading only**. Not run, because
  `ndarray` is absent in this container.
- Confound: a `Stamp` bit is a source, not an admission event (see X9). So this shows
  "pooled despite a shared source", not "the same observation counted twice".
- Fix shape:
  - X10 tri-state overlap: an empty stamp means `Unknown`, which resolves by CHOICE.
  - Then X8: derived beliefs carry the union of their premises' stamps.
- Constraint: the fix changes the `deepnsm-v2` NARS guard ("add no NARS" anchor), so it
  needs the operator's go-ahead and a red-first test.

**X1 — per-evidence stance on the receipt→relation link (C1). PROBE.**
- Reviewers: OPPORTUNITY / CONFLICT.
- Seam: a closed enum (Supports / Disputes / Mentions / Unclassified) written once by
  the ingest writer. `profile()` keeps counts per stance and never a signed net sum.
- Neither the Tarski depth nor A9 is ever computed from it.
- Outside precedent:
  - SEPIO `directionOfEvidenceProvided` (CC-BY 3.0)
  - CIViC evidence direction (CC0)
  - CiTO (CC-BY 4.0)
  - scite
- **Needs explicit operator confirmation:** stance-on-the-link is a different property
  from the polarity carried by the witness register.
- Probe: one receipt bearing on two relations with opposite stance. Kill any placement
  that has to duplicate the receipt.

**X2 — stance as tallies (for / against / mention) with neutral ≠ unknown. PROBE.**
- Reviewers: OPPORTUNITY / CONFLICT.
- Integer counts PASS. A Jøsang-style Dirichlet valuation is float and is allowed only
  as a late, reader-side valuation.
- EMPTY vs 0 is legal only under the Tarski ClassView, never on A9.
- Census kill: fewer than 1% of relations carry a dispute → PARK.

**X3 — claim-level negation / deprecation. PARK.**
- A negated proposition is a separate `RelationId`. A stored verdict is a TRAP
  (it mints a verdict).
- Probe first: does any ingest emit "X does not cause Y"?

**X4 — support / falsifier depth as a derived path property. PROBE.**
- PASS inside Tarski. It must not be fed automatically from receipts (four-plane rule).
- Q1 fixture: a dispute with no derivation must leave the falsifier slot at 0.
- Outside precedent: bipolar argumentation (Amgoud / Cayrol / Lagasquie-Schiex).

**X5 — declared source→source quotation registry (`SourceLineage`). PARK.**
- The design is ready. The kill fires today: nothing outside `causal_audit` consumes
  `profile()`.
- Shape:
  - lineage is declared only, never inferred;
  - one writer;
  - `distinct_primary_count` by collapse;
  - an undeclared lineage reads `Unknown`, never independent.
- Outside precedent: PROV-O `wasQuotedFrom` / `hadPrimarySource` (W3C), CiTO, Biolink
  source chain.
- Reviewers: firewall PASS; bridge first slice.

**X6 — copy detection from shared false values. PROBE, stage-1 census only.**
- Reviewers: WORTH-EXPLORING / TRAP inside the ledger.
- Outside source: Dong / Berti-Equille / Srivastava, PVLDB 2009.
- Allowed only as an offline proposer that writes HYPOTHESIS edges into X5.
- Kill: fewer than 5 source pairs sharing ≥10 conflicting keys in our corpora.

**X7 — idempotent fold for echo. SKIP.**
- Already have this: Stamp CHOICE, mask echo = 0.
- Denœux (cautious rule) and Jøsang averaging are float valuations, late only.

**X8 — lineage as ids, valuation applied late. PROBE.**
- Reviewers: WORTH-EXPLORING / PASS.
- Outside sources: provenance semirings (Green / Karvounarakis / Tannen 2007); ProvSQL
  (MIT); Soufflé (UPL).
- Fixture: two paths sharing a premise. A set-based stamp cannot tell this from fully
  disjoint paths.

**X9 — admission-event identity. PARK.**
- Heavy. Parked design §(a) gives the unblock condition.
- It needs an immutable full-width id, never a 64-bit digest (63.4% false overlap
  measured).
- It must not reuse the W-slot.

**X10 — tri-state overlap (KnownDisjoint / KnownOverlap / Unknown). PROBE → fix with X17.**
- The content PASSES. Where it lives (an adapter, or changing `Stamp`'s return type) is
  the operator's call.

**X11 — reliability per (source, channel), learned from curated labels. PARK.**
- Outside sources: Knowledge-Based Trust (arXiv 1502.03519); truth-discovery survey
  (arXiv 1505.02463).
- No labels source exists. Store integer outcome counts only; scores are reader-side
  floats.
- Do not mint a type beside `TrustTexture` / MUL.

**X12 — tiered aggregate that keeps conflict visible. PARK.**
- After X1 and X2.
- Outside source: ClinVar review status (public domain).
- Tiers are valuation and live in X11, never in the ledger.

**X13 — explicit hierarchy pooling with an indirectness discount. PARK.**
- Outside sources: Jung / Kim / Shim, EDBT 2019; GRADE indirectness.
- Needs X5 first.
- Internal probe G3 (one source through three sibling basins pools to c = 0.9444, the
  same as three independent sources) is the falsifier it must turn red.
- Reuse ogar-obo `is_a`, which is over terms today.

**X14 / X15. SKIP.** Already have:
- statement count ≠ distinct sources;
- nanopub / SEPIO claim–provenance split = `RelationId` / receipt / `at`.

**X16 — mint confidence at ingest. SKIP (TRAP; anchor conflict).**
- tesseract `triple_nars_truth` is a live instance: it mints OCR-quality truth at
  extraction. It is not stance; it records reliability.

## Smallest lawful path to `independent_strength`

The bridge proposed **X1 → X5 → X10**. `independent_strength` would become a count-only
projection: distinct primary sources among `Supports` receipts. It would read `None`
while any contributing source has `Unknown` lineage.

The firewall adds X9 before observation-level independence is claimed. The falsifier
notes X5's kill fires today because there is no consumer.

So the order is gated on a first consumer of `profile()`. The exception is X17/X10,
which fix a live path.

## Not searched (a "nothing found" here is not "nothing exists")

- **Not read:**
  - SemMedDB (archive was rate-limited)
  - Hetionet
  - DeepDive
  - Monarch `has_evidence`
  - the CIViC Assertion / rating pages
  - ClinVar same-lab dedup
  - scite self-citation handling
- **Read only secondhand or as an abstract:**
  - Jøsang's primary paper (secondhand)
  - Dawid–Skene (secondhand)
  - ECO (abstract only)
  - the Biolink licence (not confirmed on its page)
- **Internal:**
  - the planner's X17 path was read, not run;
  - MedCare `ProvenanceWitness` revision code was not read;
  - the OGAR relation-hierarchy hydrators were seen by file name only.

## OPEN (for the operator)

1. **X17:** approve a red-first test plus the X10 guard fix in `deepnsm-v2` (and the
   planner copy), or leave it as-is.
2. **X1:** confirm that stance-on-the-link is a separate property from witness-register
   polarity, before any field is designed.
3. Is there a first consumer of `SupportProfile` in sight? If not, X5, X12 and the
   `independent_strength` path stay parked.

coresearch evidence-stance | STATUS: measured | OUTCOME: map landed; X17 double count measured in deepnsm-v2; source convention corrected to attribution (#1318) | OPEN: items 1–3 above

## Addendum: operator working model for the 24×i4 register

Provenance: the operator stated this on 2026-10-04. It is a WORKING-MODEL, not yet
in code.

A **class-scoped signed property register**. The class's ClassView names the 24
slots. Each nibble is a signed, graded verdict on "entity has property P".

Example, class Mammal, with slots `terrestrial` and `placental`:

| | terrestrial | placental |
|---|---|---|
| fox | + | + |
| elephant | + | + |
| whale | − | + |
| possum | + | − |
| platypus | weak | − |

This is claim-level polarity with degree. It sits beside per-evidence stance on
receipts (X1) and the A9 pointer reading over the same bytes; it does not replace
either.

**Who writes it.** About 80% asserted from the ontology. The rest is derived by
`lance-graph-arm-discovery` at ingest, through tesseract-rs `tesseract-paperless`
`auto_match` (rows from the paperless and tantivy pipeline).

**Purpose.** Lift the children's shared properties into the parent as inheritable
defaults. The siblings that disagree with an inherited sign are the test set:
- whale (terrestrial −);
- possum and platypus (placental −).

A disagreement marks a missing intermediate. Its edge carries CE64
`CausalTopology::IndirectUnknownIntermediates` (bits 59–60, `causal-edge/src/layout.rs`).
The hypothesis is a candidate intermediate (Marsupialia, Monotremata). It resolves to
`IndirectKnownIntermediates` when the dissenters agree within the new sub-cohort.

Mined rules stay at `ReasoningBand::Association` (bits 61–63). Promotion to `Causal`
needs intervention-grade receipts, which `causal_audit::is_intervention_established`
already requires.

**Gap (OPEN).** A slot filled by inheritance must never be mined back as evidence for
the parent rule. That is circular confirmation, the same shape as X17 and internal
probe G3. The register has no per-slot provenance (asserted / inherited / derived)
today. No ClassView gives the 24 nibbles class-chosen slot names either: A9 and the
probe-local Tarski reading both use fixed slot lists.

**Probe (PROBE).**
- **Fixture:** the five animals above plus kangaroo and koala (Marsupialia: terrestrial
  +, placental −) and echidna (Monotremata: terrestrial +, placental −). Every proposed
  sub-cohort needs at least two independently asserted members, or the
  leave-one-sibling-out gate in Addendum 2 removes its only evidence.
- **Mine:** "Mammal → terrestrial" and "Mammal → placental".
- **Expect:**
  - exceptions are exactly {whale} and {possum, kangaroo, koala, platypus, echidna};
  - after inserting Marsupialia and Monotremata, confidence within each sub-cohort is
    1.0, and stays 1.0 with any single member left out.
- **Kill condition:** confidence rises when inherited slots are included in the
  mined rows.
- **Silent twin:** with only asserted slots mined, adding the same number of
  independently asserted children must raise support.

## Addendum 2: write-free rounds, promotion gate, execution layer

Provenance:
- Operator working model, 2026-10-04.
- Timing figures marked *operator* below are not yet recorded in the repo.

**Parent-node read-through.** The shared properties of a set of children live on
the parent's HHTL node, in its value slab, never copied into the children. This
is already planned in `lance-graph-contract/src/episodic_basin.rs`:

> "HHTL positions … BE SoA rows whose value slab carries a self-organizing
> summary of the position's children (upstream/downstream inheritance, basin
> agreement, disagreement, missing links)"

- A child row holds only what was stated about that child, either asserted or
  derived.
- An unwritten child slot reads through to the parent.
- The miner reads child rows only, so an inherited value can never confirm its
  own parent. This replaces the earlier "per-slot provenance" gap.
- **Precondition, still OPEN:** a nibble must be able to tell EMPTY (not
  stated, inherit) from 0 (observed neutral). That is the deferred EMPTY-vs-0
  question from the six-semantic-families ruling.
- The node-level hydrate step that `episodic_basin.rs` names (a position implied
  by the rails but not yet hydrated) does not exist. `graph/hydrate.rs` hydrates
  weight vectors and is unrelated.

**Write-free rounds.** All 64 thoughts × 64k lanes run in parallel. Every lane
reads the same a-priori snapshot (version v), so no lane depends on another.

- No write happens during the round, not even a sparse alpha write. Folds are
  recomputed rather than stored (board `D-WFL-ECON`: "If thinking again is
  cheaper than remembering the answer, think again").
- The alpha overlay is the in-flight mask state, not storage.
- Order inside one lane (for example a front-to-back EWA composite,
  `cognitive-shader-driver` `alpha_front_to_back_composite`) is allowed.
  Dependence between lanes is not.

**Promotion gate: the only write.** A state is promoted to snapshot v+1 only
when it crosses the Rubicon: it has to become history, evidence or state
(`D-WFL-ECON` semantic retention; `D-WFL-CACHE`). It is promoted only after
these conditions are tested:
- every lane agrees on the resolution (an order-free AND and popcount across
  lanes);
- the agreement holds across every leave-one-source-out and
  leave-one-sibling-out fill (the crossword rule: one consistent fill is not
  enough, every fill must agree);
- no lane hit its search cap. An overflow means unknown, not unique.

All states validated in a round go out as ONE Lance version.

**Finding dissenters with masks.** Split the 24 signed nibbles into a positive
plane P and a negative plane N.

| measure | formula |
|---|---|
| agree | `popcount(Pa&Pb \| Na&Nb)` |
| conflict | `popcount(Pa&Nb \| Na&Pb)` |
| unknown | the remainder |

Elephant vs whale: agree 1 (placental), conflict 1 (terrestrial).

- Parent value: AND, or popcount majority, across the children.
- A child's exceptions: `(N_child & P_parent) | (P_child & N_parent)`. Both
  directions count: a negative child under a positive default (whale, terrestrial)
  and a positive child under a negative default (an egg-laying child under a
  non-egg-laying parent).
- Dissenters with identical exception masks become missing-link candidates.

Masks only PROPOSE. With two slots, possum and platypus share a mask and would
suggest a single "non-placental" intermediate, which is not a real group. Adding
a slot such as egg-laying separates them, and the leave-one-out gate decides.

Execution runs on the shipped stack:
- `lance-graph-quack` lowers the cohort and group operators to a `Program`.
- `lance-graph-mask-risc` executes it: Boolean trees are fused to ternlog, and
  execution is tiled, with no population-sized intermediates.

**Costs.**

| operation | cost | source |
|---|---|---|
| fold | 1.7 ns hot / 4.3 ns cold | MEASURED: #1245 (1.7) and #1250 (1.7–4.2) |
| mask op | 4–12 ns hot / 12–40 ns cold | *operator* |
| write (old path, no masking) | 233 ms, plus 125 ms compute | *operator* |

At one op per lane-thought, a full round of 4.2M lane-thoughts on one core costs:
- folds: about 7 ms hot / 18 ms cold;
- masks: about 17–50 ms hot, and at most about 168 ms cold.

Even the worst cold case is below one write. One write buys about 33 hot fold
rounds, so leave-one-out rounds cost almost nothing next to a write. A real
lane-thought costs k ops; the ternlog fuser keeps k small. `D-WFL-ECON`'s W6
measures the real ratio. That measurement confirms the constant; the design
does not depend on it.
