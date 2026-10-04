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
