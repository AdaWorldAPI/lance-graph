# S3.0 Cold-Start Recovery Audit — 2026-08-20

**Outcome: no new address type. S3.0 as specified is WITHDRAWN.**
This PR retracts #973's overclaim, corrects a WordNet overclaim this session
itself introduced, and records why the S3.0 address slot needs no new type.

## A. MERGED substrate truth

- **PR #970** [MERGED, `781c3b9b`]: CE64 v2 layout final. Bits 59-60 =
  `TrustTexture` (canonical) + ADDITIVE `CausalTopology` reading over the same
  bits. Bits 61-63 = spare + ADDITIVE `ReasoningBand`, explicit
  `with_reasoning_band()` only, never auto-derived. Deprecated v1 `temporal`
  (52-63) is NOT valid v2 chronology; version-gate edges of unknown
  provenance. Confirmed in `crates/causal-edge/src/layout.rs`.
- **PR #971** [MERGED]: Stage-2/2.5/2.6a frozen baseline. `InferenceType` is a
  LOSSY compat projection of the 4-bit signed mantissa (8/16 states corrupted,
  incl. `pack_v2` default `0 -> +1`) — carry the raw nibble. CE64<->V3
  bit-identical modulo dedup'd SPO + deprecated temporal. Capability !=
  reachability measured.
- **PR #973** [REJECTED, closed unmerged]: nothing on `main`.

## B. MEASURED substrate truth

1. **`identity_quad::IdentityQuad`** [MERGED, operator-RATIFIED 2026-08-17,
   `ISS-IDENTITY-QUAD-WIDE-CARVING-HOME`]. **Four external identity spaces,
   `4 × u24` contiguous, in ONE 96-bit V3 facet payload** via
   `LegacyOutlier::WideTriple`, behind a `classid(4)`. Refuses rather than
   truncates (`QuadError::OrdinalTooLarge`, `MAX_ORDINAL = 2^24 - 2`);
   codebooks refuse rather than saturate (`CodebookError::TooLarge`). Its
   stated purpose is resolving a crosswalk ONCE at bake time so a read is a
   fixed-offset register read — no join, no crosswalk walk.
2. **`ogar_elk::ClassAddr`** [MERGED]: `classid: u32 + identity: u32` — 8
   bytes for ONE node, and its own doc is explicit that it is a pre-bake
   **join key**, "not an ABI address, and deliberately not documented as one".
3. **Real OBO identities exceed `u16`** [MEASURED, MedCare-rs
   `docs/ONTOLOGY_BAKE_STATE.md`:182]: *"real OBO ids run past `u16`
   (MONDO:0700092 = 700,092) and one V3 field cannot hold that."* The
   substrate ALREADY solved this: the V3 rail splits a 24-bit CURIE numeric as
   `family:identity = (num >> 16, num & 0xFFFF)`, read via
   `obo_store::row_addr`. Corpus scale: MONDO 32,095 · HP 19,836 ·
   UBERON 14,975 · PATO 1,887, and relations cross families.
4. **`ClassId = u16` is near-exhausted for RELATIONS** [MEASURED, MedCare-rs
   `CLAUDE.md` commitment #10 + `RAIL_OFFENE_POSTEN.md`]: *"`ClassId = u16`
   cannot address a relation — 11 prefixes, 8 of 280 ids over the ceiling."*
   That is a real open gap — and it is a **classid-mint capacity** question
   owned by OGAR/lance-graph, NOT something a new literal type fixes.
5. **WordNet #875/#876** [MERGED, MEASURED] — see §C2 for the corrected
   reading. Also #876: a consumer can be structure-blind to an address's
   hierarchy without that meaning the address lacks structure.
6. **TEKAMOLO #839/#844** [MERGED, live code]: one 16-byte content-blind
   register, several simultaneous ClassView-selected readings
   (`G3D4`/`G4D3`/`G6D2`/`24×i4`). Capability lands as a new READING.
7. **24×i4 anaphora #850** [MEASURED, 7,657 real German relative clauses,
   88.01% in-window]: a local representation exhausting cleanly marks a TYPE
   boundary — switch reading, never widen the pointer.

## C. REJECTED claims

### C1. #973's `E-THE-LITERAL-CANNOT-LIVE-IN-THE-PATH-IT-ROOTS-1` — RETRACTED

#973 measured correctly that a `domain·S·P·O` at `u16` each is exactly 16
nibbles = `NiblePath::MAX_DEPTH`, zero slack. It then promoted that local fact
about ONE sequential depth-limited router path into an implied global claim
that exact identity cannot live in HHTL addressing and evidence must
"ref-escape" out of address space. The three counterexamples in §B5-B7 were
never consulted. The arithmetic survives; the inference does not.

### C2. This session's OWN WordNet overclaim — RETRACTED

The first draft of this recovery PR said #875 proved a *"full-width 4-ary HHTL
fold of real WordNet ancestry is an EXACT structural encoding"*. **That is
false, and #875's own numbers say so:** W5 reports *256/256 cells used,
occupancy min 29 / median 255 / max 1270* over **65,292 leaves**. That is
~255 leaves per cell — emphatically NOT injective, not an identity encoding.

What #875 actually measured is a **deterministic, taxonomy-informed HHTL
locality / search prior**: shared-address-levels vs LCA-depth corr +0.494 (vs
−0.036 shuffled), out-of-cell band recall 0.763 vs 0.031 random = 24.71×, and
a 2.47-hop sub-nibble distinction the 16-ary router cannot address. A
discriminating prior that deliberately does NOT cover everything — #875 itself
dropped its cover guard as inert for exactly this reason.

So the Bible/Rosetta + WordNet composition is:
`frozen verse identity × quasi-absolute taxonomy-informed semantic coordinate
× witness/qualia planes` — **not** evidence that 256 HHTL cells uniquely
identify a lexicon. Corrected in the PR body, EPIPHANIES, and this audit;
the module doc carrying it is deleted with the type.

### C3. `CausalLiteral` was the wrong universal type — WITHDRAWN

Two independent defects:

- **Wrong name / wrong universality.** Its own test asserts
  `CausalLiteral(7, 100, 43, 200)` as `TREATED_WITH` — demonstrating the
  structure is generic, not causal. `ASSOCIATED_WITH` / `TREATED_WITH` /
  `CAUSES` / `MEDIATES` / `PART_OF` are exact predicate identities over one
  generic literal substrate. Causality is a **predicate family /
  qualification**, never universal identity. If a primitive were needed it
  would be `ExactLiteralAddr(D,S,P,O)`.
- **`4 × u16` is NOT absolute, and cannot be.** A `u16` subject cannot hold
  MONDO:0700092 = 700,092 (§B3). The type would have silently mis-addressed
  the single largest ontology in the MedCare bake. Calling it "absolute
  identity" was a second global overclaim in the same PR that retracted one.

## D. Salvageable

- The nine tests' SHAPE (injectivity sweep, component-isolation matrix,
  many-sources-one-literal, unbound sentinel) is good discipline and should be
  reused by whatever type actually lands — but it belongs on
  `IdentityQuad`/`ClassAddr`-based composition, not on a new 4×u16 plane.
- The retraction of C1 and the two process findings.
- `routing_prefix()` is REMOVED, not merely deferred — see §G.

## E. The development ladder (operator-requested matrix)

|                    | ADDRESS | HYDRATED SoA | TRAVERSAL |
|--------------------|---------|--------------|-----------|
| Bible / Rosetta    | yes     | **NO**       | partial / context |
| OSM                | yes     | yes          | overlay / junction |
| MedCare ontology   | yes     | yes          | **YES, Stage 1** |
| DisMech oracle     | source  | structured   | causal oracle |

**The column that is empty is not ADDRESS.** Address is solved three times
over (`ClassAddr`, the V3 rail, `IdentityQuad`, `NodeGuid`/HHTL). The missing
work is: **hydrate epistemic / causal nodes → reason over them → think about
the reasoning.**

### The question S3.0 had to answer, and its answer

> **WHAT EXACT INFORMATION CANNOT BE EXPRESSED BY THE ADDRESSING THAT ALREADY
> EXISTS?**

**Nothing that this session could demonstrate.** `IdentityQuad` (classid + 4 ×
u24, ratified 2026-08-17) strictly dominates the withdrawn 4×u16 type on every
axis: wider (handles real OBO ids, which u16 does not), already V3-carving
conformant (rides a sanctioned reading rather than minting a parallel identity
plane), already refuse-don't-truncate, already bake-time-join-resolving.

There is therefore **no concrete falsifier showing existing addressing is
insufficient**, and per the operator's rule no new absolute-address type is
minted. The S3.0 slot in the Stage-3 plan is closed as NOT-NEEDED rather than
filled because the plan had a slot.

**The one genuinely open addressing gap found** is different and is NOT this:
`ClassId = u16` is near-exhausted for RELATIONS (§B4). That is a classid-mint
capacity question for OGAR/lance-graph, to be raised with the operator in
session — not something a new literal type addresses.

## F. Next target (per operator §6)

```
  EXISTING absolute address  (ClassAddr / V3 rail / IdentityQuad / NodeGuid)
          |
     hydrate node
          |
   CausalMeta + EpistemicMeta
          |
   EntropyWork · BasinSet · Attention
          |
   RowFocusMask × WideFieldMask
          |
      V3 / CE64  ->  NARS  ->  ReasoningEpisode  ->  Meta/Rubicon  ->  OGAR-loco
```

**DisMech as the oracle experiment:** hide known mechanism intermediates →
hydrate the addressed ontology neighbourhood → let NARS/recipes recover
candidates → compare against DisMech truth. Grounded: `dismech-rs`
`graph::build_causal_graph` is a real transcode of the upstream Python
resolver, falsified at **1,995 diseases / 33,458 edges** against the private
`medcare-dismech` measurement (33,328, within 0.4%), and 1,903 committed
`pathographs/MONDO_*.json` exist as ground truth. Not started this session.

## G. `routing_prefix()` — REMOVED

Concatenating D/S/P/O nibbles into a `NiblePath` yields deterministic
**lexicographic** prefixes. It was labelled an "HHTL locality / cohort
projection" — but no consumer exists and no measurement establishes that it
preserves HHTL *semantic* locality. `NiblePath` is built for the `subClassOf`
Abstammung tree, where a prefix is an ancestry claim; a lexicographic prefix
over concatenated ordinals carries no such guarantee. Calling it one was a
third unearned claim. Removed with the type; if it returns it needs a real
consumer and a measurement first.

## H. Falsifiers that would reopen S3.0

1. A concrete case where `IdentityQuad` (4 × u24 + classid) provably cannot
   express an exact proposition identity the substrate needs.
2. A measured need for a 4th component space beyond `IdentityQuad`'s four
   slots — noting its own doc says a fifth identifier space is a **second
   facet**, never a wider field.
3. Resolution of the `ClassId = u16` relation-capacity gap requiring a new
   addressable relation identity (a classid-mint question first).
