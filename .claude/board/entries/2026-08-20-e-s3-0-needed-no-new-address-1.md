## 2026-08-20 — E-S3-0-NEEDED-NO-NEW-ADDRESS-1 — the Stage-3 "S3.0 address" slot is closed as NOT-NEEDED; `IdentityQuad` already carries an exact four-component identity, at u24, inside the sanctioned V3 facet

**Status:** FINDING (operator-directed audit; measured against merged code).
**Confidence:** High — every claim below is a read of shipped source or a
merged measurement, not a derivation.

**The question that closes it** (operator's rule, stated during this session):
*"WHAT EXACT INFORMATION CANNOT BE EXPRESSED BY THE ADDRESSING THAT ALREADY
EXISTS?"* — and if there is no concrete falsifier showing existing addressing
insufficient, **no new absolute-address type is minted.**

**Answer: nothing that could be demonstrated.** `identity_quad::IdentityQuad`
(operator-RATIFIED 2026-08-17, `ISS-IDENTITY-QUAD-WIDE-CARVING-HOME`) already
materializes **four external identity spaces as `4 × u24` contiguous in ONE
96-bit V3 facet payload** behind a `classid(4)`, via
`LegacyOutlier::WideTriple`. It refuses rather than truncates
(`QuadError::OrdinalTooLarge`, `MAX_ORDINAL = 2^24 − 2`); its codebooks refuse
rather than saturate (`CodebookError::TooLarge`). Its stated purpose is to
resolve a crosswalk ONCE at bake time so a read becomes a fixed-offset register
read — no join, no crosswalk walk.

**A proposed `4 × u16` literal type was WITHDRAWN, on two independent grounds:**

1. **`u16` cannot hold a real ontology identity.** MedCare-rs
   `docs/ONTOLOGY_BAKE_STATE.md`:182 states it directly — *"real OBO ids run
   past `u16` (MONDO:0700092 = 700,092) and one V3 field cannot hold that."*
   The substrate already solved this with the V3 rail
   (`family:identity = (num >> 16, num & 0xFFFF)`, read via
   `obo_store::row_addr`). A u16 subject would have silently mis-addressed the
   largest ontology in the bake. Calling such a tuple "absolute identity" was
   an overclaim.
2. **`CausalLiteral` was the wrong universality.** Its own test asserted
   `TREATED_WITH` — proving the structure is GENERIC. `ASSOCIATED_WITH` /
   `TREATED_WITH` / `CAUSES` / `MEDIATES` / `PART_OF` are exact predicate
   identities over one generic literal substrate. **Causality is a predicate
   family / qualification, never universal identity.** Had a primitive been
   needed it would have been `ExactLiteralAddr(D,S,P,O)` — but per the rule
   above, none was.

**Sibling absolute-address surfaces already merged**, for a future session's
map: `ogar_elk::ClassAddr` (`classid: u32 + identity: u32`, explicitly a
pre-bake **join key**, *"not an ABI address, and deliberately not documented as
one"*), `canonical_node::NodeGuid` + the HHTL cascade, and the V3 rail above.

**The genuinely open addressing gap is a DIFFERENT one, and is not fixed by a
literal type:** `ClassId = u16` (`class_view.rs:54`) is near-exhausted for
RELATIONS — MedCare-rs `CLAUDE.md` commitment #10: *"cannot address a relation
— 11 prefixes, 8 of 280 ids over the ceiling"*, echoed in
`RAIL_OFFENE_POSTEN.md`. That is a **classid-mint capacity** question owned by
OGAR/lance-graph, to be raised with the operator in session.

**What the ladder says the real work is** (operator-requested matrix):

| | ADDRESS | HYDRATED SoA | TRAVERSAL |
|---|---|---|---|
| Bible / Rosetta | yes | **NO** | partial / context |
| OSM | yes | yes | overlay / junction |
| MedCare ontology | yes | yes | **YES, Stage 1** |
| DisMech oracle | source | structured | causal oracle |

**The empty column is not ADDRESS.** The missing work is *hydrate epistemic /
causal nodes → reason over them → think about the reasoning*. Next target is
the DisMech oracle experiment: hide known mechanism intermediates, hydrate the
addressed ontology neighbourhood, let NARS/recipes recover candidates, compare
against DisMech truth (`dismech-rs` `graph::build_causal_graph`, falsified at
1,995 diseases / 33,458 edges; 1,903 committed `pathographs/MONDO_*.json` as
ground truth).

**Also withdrawn with the type: `routing_prefix()`.** Concatenating D/S/P/O
nibbles into a `NiblePath` yields deterministic **lexicographic** prefixes. It
was labelled an "HHTL locality / cohort projection" with no consumer and no
measurement establishing that it preserves HHTL *semantic* locality —
`NiblePath` is built for the `subClassOf` Abstammung tree, where a prefix is an
ancestry claim; a lexicographic prefix over concatenated ordinals carries no
such guarantee. If it returns it needs a real consumer and a measurement first.

**Cross-ref:** `E-NIBLEPATH-DEPTH-IS-NOT-HHTL-DIMENSIONALITY-1` (below),
`E-WORDNET-IS-A-LOCALITY-PRIOR-NOT-AN-IDENTITY-ENCODING-1` (below), PR #973
(closed unmerged), `.claude/handovers/2026-08-20-s3-0-cold-start-recovery-audit.md`.

---

