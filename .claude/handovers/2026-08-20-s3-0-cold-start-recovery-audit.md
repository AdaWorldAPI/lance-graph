# S3.0 Cold-Start Recovery Audit — 2026-08-20

## A. MERGED substrate truth

- **PR #970** [MERGED, `781c3b9b`]: `CausalEdge64` v2 layout final. Bits 59-60 =
  `TrustTexture` (canonical) with an ADDITIVE second reading `CausalTopology`
  over the *same* bits — no bits move, no auto-derivation. Bits 61-63 = spare,
  with an ADDITIVE `ReasoningBand` reading — explicit `with_reasoning_band()`
  only, never auto-derived from mantissa/confidence/style. Deprecated v1
  `temporal` (bits 52-63) is NOT valid v2 chronology; a v1 edge with
  `temporal >= 512` reads a nonzero band under v2 — version-gate required.
  Confirmed live in `crates/causal-edge/src/layout.rs`.
- **PR #971** [MERGED, `93dc57e`]: Stage-2/2.5/2.6a frozen baseline. Key
  correction carried: `InferenceType` is a LOSSY compat projection of the
  4-bit signed mantissa (8/16 states corrupted on round-trip, incl. the
  `pack_v2` default `0 -> +1`) — never route a conversion through it; carry
  the raw nibble. CE64<->V3 is bit-identical except (a) dedup'd 24-bit in-edge
  SPO and (b) deprecated v1 temporal (deliberately NOT lifted — TE stays an
  independent producer-set signed chain offset). "Recipe capability != NARS
  reachability" measured (`Mcp` truthfully declares `moves_confidence()` and
  is still silent 0/180).
- **PR #973** [REJECTED, CLOSED UNMERGED, no comments from reviewers —
  operator judgment call before any bot review completed]. Confirmed via
  `git log`/`grep`: nothing from this PR is on `main` — `causal_literal.rs`
  does not exist in the working tree.

## B. MEASURED substrate truth (the counterexamples #973 forgot)

All confirmed live/merged in this repo, read in full this session:

1. **WordNet #875/#876** [MERGED, MEASURED]. A full 4-ary depth-4 (16-nibble)
   HHTL fold of real ground-truth taxonomy is NOT lossy hashing — the ADDRESS
   itself encodes ancestry (corr +0.494 vs shuffled -0.036), the sub-nibble
   rung is load-bearing (2.47 hops the 16-ary router can't see), and #876
   separately proved a *consumer* (the ruler) can be structure-blind to an
   address's hierarchy without that meaning the address lacks structure —
   "the address encodes taxonomy" vs "the calculator reads the address as a
   number" are different, separable claims.
2. **TEKAMOLO #839/#844** [MERGED, live code: `facet.rs` + `tekamolo_facet.rs`].
   `FacetCascade = facet_classid(4) | 6×(8:8) = 16B`, ONE 128-bit register with
   MULTIPLE simultaneous ClassView-selected readings (`G3D4`/`G4D3`/`G6D2`,
   `24×i4`), never nested path-depth. TEKAMOLO names the `G4D3` carving as
   4 ORTHOGONAL 256:256:256 lanes (Temporal/Kausal/Modal/Lokal) over the SAME
   bytes — new semantic capability lands as a new *reading*, never a deeper
   path. This is the concrete, shipped instance of "orthogonal facets over one
   address" the mission brief's OSM/TEKAMOLO doctrine describes.
3. **24×i4 anaphora #850** [MEASURED, 88.01% real German relative clauses].
   Proves the general shape #973 needed but didn't reach for: a LOCAL fixed
   representation (i4, -8..+7) exhausts cleanly at a real linguistic boundary,
   and the correct response is NOT "widen the local pointer" — it's "the
   exhausted local representation marks a TYPE boundary; switch to a
   DIFFERENT sanctioned reading of the SAME address register (a basin edge),
   never invent a deeper/wider path." Directly antithetical to reading
   NiblePath's 16-nibble ceiling as grounds to abandon HHTL identity.
4. **`hhtl.rs` `NiblePath`** [MERGED, confirmed]: `MAX_DEPTH: u8 = 16`, one
   `u64`, single SEQUENTIAL router path for the specific `subClassOf`
   Abstammung tree. This is ONE HHTL-shaped type among several in this repo —
   NOT the only or canonical HHTL substrate. `FacetCascade` (item 2 above) is
   a materially different HHTL-adjacent type: a fixed 16-byte register with
   *simultaneous* multi-lens reads, not a depth-limited single path.

## C. REJECTED #973 claims — precisely scoped

**What #973's CODE actually did (and got right):** `CausalLiteral{domain:u16,
subject:u16, predicate:u16, object:u16}`, 8 bytes, `const _`-asserted size,
component equality, no CAM-PQ/evidence/source/version fields possible by
construction. This is *exactly* the mission brief's own §25 recommended
minimal S3.0 shape. The 9 tests (injectivity, field isolation, round-trip,
unbound sentinel) are sound and reusable verbatim.

**What #973's REASONING (the EPIPHANIES title + doc-comment framing)
overclaimed:** the finding is titled
`E-THE-LITERAL-CANNOT-LIVE-IN-THE-PATH-IT-ROOTS-1` and its argument runs:
"a `domain·S·P·O` address exactly fills `NiblePath`'s 16-nibble budget with
zero nibbles left for an evidence subtree beneath it -> therefore identity
and routing SPLIT -> the evidence subtree must *ref-escape* out of address
space entirely." The arithmetic (16 nibbles = `MAX_DEPTH`, zero slack) is
correct and TRUE ONLY OF `NiblePath` — a single sequential depth-limited path
type. The invalid generalization is treating that as proof that **HHTL
identity itself** (not just this one path type) cannot carry the literal, and
that evidence/meta state must therefore live in some conceptually separate,
disconnected mechanism rather than as an **orthogonal facet keyed by the same
8-byte address** — exactly the TEKAMOLO/anaphora/OSM pattern already proven
in this repo. Nothing in #973 acknowledges that a `FacetCascade`-style
multi-reading register (or simply: `CausalLiteral` as its own SoA
column/plane, with a `CausalMeta`/`EpistemicMeta` column keyed by the SAME
`CausalLiteral`) sidesteps the "budget" problem entirely, because it was
never modeled as depth-based descent from a single path in the first place.

**Verdict: `E-THE-LITERAL-CANNOT-LIVE-IN-THE-PATH-IT-ROOTS-1` is RETRACTED as
stated.** Superseded by a narrower, correctly-scoped finding (drafted below).
This is the precise process failure named in the mission brief §0/§9: a local
representation limit (NiblePath's depth ceiling) promoted into an implied
global architecture conclusion (identity+evidence must structurally split
away from HHTL) while forgetting three already-measured counterexamples in
the SAME repository that show the opposite pattern working.

## D. Salvageable #973 code/math

- `CausalLiteral { domain, subject, predicate, object }` as 4×`u16`, 8 bytes,
  `const _`-asserted, component equality — REUSE VERBATIM.
- `packed_identity_is_injective`, `changing_only_the_predicate_changes_the_literal`,
  `three_sources_asserting_the_same_proposition_mint_one_literal`,
  `component_isolation_matrix`, `identity_round_trips_exactly_in_both_forms`,
  `unbound_components_are_addressable_but_not_fully_bound` — REUSE VERBATIM
  (these test IDENTITY, which #973 got right).
- `routing_prefix(depth) -> NiblePath` / `full_path()` — REUSE, but RELABEL
  the doc comments: this is *one particular* NiblePath-shaped projection
  useful for cohort/locality queries against the existing `subClassOf`-style
  router, not "the" HHTL reading of the literal, and its lossiness at
  `depth < 16` is a fact about `NiblePath` specifically, not about HHTL
  identity in general.
- DROP/REWRITE: `routing_prefix_is_not_identity` and
  `the_full_literal_path_exhausts_the_nibble_budget` stay as tests (they are
  correct, falsifiable facts about `NiblePath`) but their surrounding prose
  must not imply "therefore identity lives outside HHTL."
- DROP the `E-THE-LITERAL-CANNOT-LIVE-IN-THE-PATH-IT-ROOTS-1` framing;
  replace per §E below.

## E. Remaining unknowns

- Whether `CausalLiteral` should ALSO be constructible as a `FacetCascade`
  reading (a 5th `CascadeShape`, `4×u16`-over-16B) for consumers that want it
  to live inside the existing content-blind register alongside TEKAMOLO/rails/
  SPO-triplet readings, vs. staying a free-standing 8-byte contract type
  consumed by reference. Not resolved this session — S3.0 replacement below
  ships the free-standing type only (matches the brief's own minimal-first
  guidance in §25), and defers the FacetCascade-reading question to S3.1/S3.2
  where the V3 local-proxy bridge is actually built.
- CausalRegimeAddr (S3.0b), Meta tree (S3.1), predicate resolution (S3.3) —
  all explicitly out of scope for this PR per the brief's own delivery order.
- Rubicon threshold semantics (§19) — not investigated this session; will be
  searched before any Rubicon-touching work, not asserted from memory.

## F. Proposed minimal S3.0 replacement

Reintroduce `crates/lance-graph-contract/src/causal_literal.rs` with:
1. The same `CausalLiteral` struct, accessors, packing, and all 9 original
   tests (salvaged per §D).
2. Module-doc and EPIPHANIES entry REWRITTEN to state the narrow, correct
   claim: `NiblePath::MAX_DEPTH` (16 nibbles) is exactly consumed by a
   4×`u16` literal, so a literal's *own* identity cannot ALSO be expressed as
   a strictly-shorter `NiblePath` prefix while remaining exact — a fact about
   `NiblePath` depth budget, not about HHTL addressing dimensionality in
   general. Explicitly cross-references WordNet #875/#876 (exact structural
   HHTL address is real and proven), TEKAMOLO #839 (orthogonal-facet pattern
   is the sanctioned way to attach evidence/meta without deepening a path),
   and anaphora #850 (a locally-exhausted representation marks a TYPE
   boundary, not a substrate ceiling) so a future session reads the
   NiblePath-specific finding correctly.
3. New finding name: `E-NIBLEPATH-DEPTH-IS-NOT-HHTL-DIMENSIONALITY-1`
   (supersedes/retracts `E-THE-LITERAL-CANNOT-LIVE-IN-THE-PATH-IT-ROOTS-1`,
   which is marked ⊘ RETRACTED in place per this repo's append-only
   convention, not deleted).
4. A second, process-level finding:
   `E-A-LOCAL-DERIVATION-CANNOT-OVERRULE-A-MEASURED-COUNTEREXAMPLE-1`,
   naming the general failure mode for future sessions.
5. Evidence/Meta placement is left EXPLICITLY OPEN (not "ref-escaped" as a
   forced conclusion) — noted as an S3.1 design question with the
   FacetCascade-orthogonal-plane option named as the leading candidate,
   rather than asserted-and-closed by this PR.

## G. Falsifiers that would kill this proposal

- If a future S3.1/S3.2 session finds that keying `CausalMeta`/`EpistemicMeta`
  as orthogonal facets over `CausalLiteral` (rather than ref-escaping) hits a
  real capacity/addressing wall analogous to NiblePath's, that would validate
  more of #973's original instinct — investigate before assuming either way.
- If `CausalLiteral` needs to itself be routable through the DOLCE/basin
  `NiblePath` tree (not just an opaque 8-byte key), the routing_prefix
  question reopens for real; this PR does not build that consumer.

## Central-constitution check (brief §27.5)

Does the proposed S3.0 replacement contradict OSM, WordNet #875/#876, Bible
Rosetta, TEKAMOLO, or the 24×i4 anaphora boundary? **NO.** It ships the same
minimal exact-identity primitive the brief itself specifies in §25, corrects
only the RETRACTED overclaim in the board narrative, and explicitly leaves
the evidence/meta placement question open rather than pre-deciding it against
the TEKAMOLO/anaphora precedent.
