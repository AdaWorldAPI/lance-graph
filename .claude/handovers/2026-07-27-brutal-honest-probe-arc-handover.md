# 2026-07-27 — Brutally honest handover: the probe arc, its hallucinations, and what actually survives

**From:** the session that ran PRs #855/#856 and the post-merge probe arc.
**To:** whoever picks this up. Read this BEFORE reading my board entries — two
merged entries currently overstate what they proved, and this file is the
corrective lens until the append-only regrades land (TD-1 below).

**Rule this handover obeys:** claims are labelled SURVIVES / WITHDRAWN /
HALLUCINATED, each with the artifact and the evidence. Nothing is softened.
Where I was wrong, the mechanism of being wrong is stated, because the
mechanism is the transferable part.

---

## 1. What I achieved (the defensible core)

- **PR #855 (merged):** furnace lanes L1–L7 measured end-to-end on real bytes.
  Amortization charged honestly after codex caught my vacuous assert: L2-only
  pays back in 0.11 passes; L1+L2 cold start needs 8.2 and is REPORTED, not
  asserted. `AdjacencyBatch` rewritten owned→borrowed (zero-copy view over the
  CSR store). Four doc-truthfulness downgrades
  (`TD-DOC-COMMENTS-CLAIM-UNWIRED-BEHAVIOUR`).
- **PR #856 (merged):** the Base17 measurement (see §3 for what its framing
  got wrong), 8 verified CodeRabbit fixes, 2 rejections with evidence.
- **GOLDEN_STEP relabel proof** — algebra, then independent confirmation:
  `(i·11) mod 17`, gcd(11,17)=1, is a permutation of residue classes; every
  symmetric readout cancels it (L1 bit-identical across steps {1,2,3,5,7,11,13}).
  Weyl pass (`jc/examples/probe_stride_discrepancy.rs`): 11/17 is RATIONAL,
  D* floors at 0.058900 vs predicted 1/17=0.058824 and barely moves with N,
  while irrationals improve 25–45×. gcd coverage is the Quintenzirkel
  precondition; low discrepancy is the payload; the integer step never had any.
- **K=200 permutation null** (`probe_wordnet_ancestry.rs`) — the harness that
  caught MY OWN false positive. The single-shuffle control I first used
  happened to sit at +1sd of the true null; every "margin over control" I
  reported before the null was subtracting a random draw.
- **Two zombie kills (operator-directed):**
  `class_view::ClassId = u16` was an ontology-registry ROW NUMBER wearing the
  word classid → renamed `EntityTypeId`, deprecated alias, NOT re-exported at
  crate root (commit a0979f2). `class_id_for_guid` moved off the v1 fold that
  REFUSED every V3-marked GUID (both classid halves nonzero = the NORMAL
  post-flip shape) onto `from_guid_prefix_v3`; the defect is now a test
  (5b798fb).
- **`contract::evidence` (769ea1e + import fix):** the §17 rebuild of the
  withdrawn source_registry — `evidence_overlap -> Disjoint|Overlap|Unknown`
  + `pooled_base`, a PURE READ over `EpisodicEdges64`. The case that falsified
  `Stamp` (one sensor observing twice could never raise confidence) is a
  passing test. Tri-state is read off the carrier (demote-on-overflow ⇒
  saturated side = Unknown, never Disjoint). 1099 contract tests green.
- **Doc-comment blast-radius audit (operator-directed, §5)** — the most
  valuable two hours of the session, because it audited ME.

## 2. What I thought I had before it collapsed — and why I believed it

**"HYDRATION CARRIES, z=+14.90, 0/200 permutations"** (a305217). I believed it
because it had everything the falsifiability rule demands: real bytes, a
K=200 permutation null, can-it-fire and can-it-stay-silent asserts, a
pre-registered verdict criterion. It looked like the most rigorous result of
the session. That is exactly why it is the most instructive failure.

**What made me delete it (775d793):** enabling target (b) required lemma
NAMES, which changed the sample from "3000 random vocab rows" to "3000 rows
that are real English nouns". Same code, same seed, same geometry, same null:
z +14.90 → +0.09 (91/200). MiniLM's vocab is 25.6% non-words (5828 `##`
pieces, 994 `[unusedNNN]`, 999 specials). The cascade had discovered the
junk/word partition — vocabulary hygiene — and the seam faithfully carried it.
I had measured input contamination and called it semantics.

**The lesson with teeth:** a permutation null tests the ASSIGNMENT, not the
SAMPLE. Every permutation inherits the same population, so no K catches a
population artifact. My z=+11.17 ancestry result is the same single-population
shape and should not be quoted without a second population. The one mechanism
that worked: the (a)→(b) gate was an ASSERT, not a warning — it panicked
rather than let me report a taxonomy number over a dead seam.

## 3. The hallucinated ledger (each with its mechanism)

1. **"HHTL awareness-location PROVEN" (a3eacb0, MERGED #855, on the board).**
   Same-HEEL 0.7981 vs shuffled 1.0711 is embedding-space DISTANCE locality —
   HHTL-as-geometric-CLAM, the frame the operator explicitly rejected
   ("using HHTL as a geometric clam instead of our Markov chain context
   building over standing wave ... would be a bad idea"). Merged, states
   PROVEN, needs an append-only regrade most urgently (TD-1).
   *Mechanism:* I proved a property of the wrong layer and named it after the
   right one.
2. **"The Base17 fold ceiling" framing (0578a52, MERGED #856).** ρ=0.2726 is
   real and replicates (0.2599 on jina-v3). But it scores a PHASE encoder
   (bgz17 encodes phase; direction is helix's `2·arctanh(s)`) with a cosine
   reconstruction metric, on an EXPERIMENTAL fold ("bgz17 encoding was used to
   fold n centroids into a single residue, experimental — the palette256
   however is real"). `bf16-hhtl-terrain.md` correction 6 names this exact
   category error; I had READ and QUOTED that file the same day.
   *Mechanism:* consult-then-recommit — reading a rule is not applying it.
   The relabel half of the entry SURVIVES and is strengthened by Weyl.
3. **The ancestry + hydration probes as QUESTIONS (f2864eb, a305217/775d793,
   unmerged).** Both treat a book/lexicon as bagged embeddings — no sequence,
   no trajectory, no causality — then ask whether meaning survived. Operator:
   "don't understand why you treat a book as embeddings, that's so terrible";
   wordnet-HHTL is "only to have Hierarchie for free" — USED, never verified
   by correlation. The whole verification framing was mine, not the design's.
   *Mechanism:* I inherited "wordnet IS HHTL" as a hypothesis to test when it
   was a construction to consume.
4. **`cross_family_palette` asserted as working machinery (chat, uncommitted).**
   Zero code. Two doc-comments (`episodic_edges.rs:21`,
   `causal-edge/syllogism.rs:51`). I told the operator the 1.4% cross-family
   case "resolves through class.cross_family_palette[family]" one message
   before the audit showed nothing exists.
5. **"The gap is one missing local_key column on an in-RAM MailboxSoA" (chat).**
   Built on `soa_view.rs:121`'s doc-comment. Operator: "there's no in ram
   mailboxsoa, they are all zero copy." The doc-comment names a structure that
   contradicts the architecture. Fifth instance of
   TD-DOC-COMMENTS-CLAIM-UNWIRED-BEHAVIOUR — used as evidence by the same
   session that fixed the first four that morning.
6. **Leg-2 GUID residue (chat).** I claimed NodeGuid uniqueness "is a debug
   assertion only" as a standing gap. Misreading: `debug_assert_identity_unique`
   is a MINT-PATH guard scoped to `is_bootstrap_address()` (classid AND family
   zero) — about 24-bit bootstrap exhaustion, not GUID collision. The whole
   GUID is unique; validation belongs at cast. (Real finding adjacent to it:
   the guard has ZERO production callers — written but unwired, TD-6.)

## 4. What I created to make it better

- The withdrawal commits themselves (775d793 pattern): result + refutation in
  one history, never amended away.
- `evidence.rs` with the falsifying case as a test, not prose.
- The Weyl probe reusing `jc::weyl::star_discrepancy` (made pub) instead of
  re-deriving the metric — after the operator's "if you really care about what
  is what you need to use JC crate".
- The blast-radius audit method (§5): count doc/comment-only vs code
  references per symbol; a mechanism whose only existence is its own
  doc-comment plus a vacuous test is scaffolding, whatever its prose says.

## 5. What is missing, why, and where to look

**M-1 · The two merged regrades (TD-1).** Append-only corrections to
`E-...AWARENESS-LOCATION` (a3eacb0's board entry) and the #856 ceiling entry.
Why missing: I did not touch merged findings without an operator call; the
call is now implicit in this arc. Where: `.claude/board/EPIPHANIES.md`
(prepend, dated), plus matching header notes in
`crates/lance-graph-planner/examples/probe_furnace_amortization.rs` and
`crates/bgz17/examples/probe_base17_fold_ceiling.rs`.

**M-2 · The reasoning wiring that was the ACTUAL thread.** The session was
supposed to be about the thinking; I rabbitholed four probes deep into a
footnote. The real path is already in-tree:
`crates/lance-graph-planner/examples/reason_whole_book.rs` — SPO belief stream
→ BeliefArena → copula-gated closure + tactics (F1: only is/was/are transit;
verbs must NOT). Causality trajectories are proven through language; the
corpus is the only place they are provable. Knowledge graph on top of the
linguistic resolve: `lance-graph-arm-discovery`. Episodic-witness vs fact:
stories carry LOWER NARS valence — a dial, not a second store (operator,
deferred for later discussion).

**M-3 · Feeding `evidence_overlap` from real derivations.** The guard is
built; nothing constructs edge words from actual belief derivations. Where the
migration lands: the two LIVE `Stamp` copies —
`crates/lance-graph-planner/src/nars/belief.rs:31` (24 refs) and
`crates/deepnsm-v2/src/belief.rs:33` (30 refs) — still model SOURCE
membership, the falsified object. `reason_whole_book`'s arena is where
derivations actually happen, so that is where edges have real referents. How
to find the seams: grep `Stamp::source|\.disjoint\(|\.union\(` in those two
files; every site is a migration point.

**M-4 · Slot-byte → row resolution, honestly stated.** Not "one encoding
decision away" (my evidence.rs commit overstated this — TD-5). Audited state:
`row_for_local_key` default-None with NO non-test implementor and a test that
asserts None==None (delete it, TD-4); `match_node_by_local_key` zero external
consumers; `cross_family_palette` zero code. What resolution SHOULD be in a
zero-copy substrate (no in-RAM owner) is an operator design call. Where to
look when it happens: `mailbox_scan.rs:33-38` states the open convention
(zero=unused; 1-based vs basin-table) and names `row_for_local_key` as the
analogous precedent; `soa_view.rs:119-127` must be rewritten first (TD-3).

**M-5 · Dependence tri-state.** Leg 4 of the withdrawal demanded tri-state for
membership AND dependence. Membership is done (`EvidenceOverlap`). Dependence
(`Independent/Dependent/Unknown`) does not exist — grep returns nothing — and
its absence is why `causal_audit.rs:346` `independent_strength` is permanently
None. One dependence model closes both.

**M-6 · Operator rulings I could not adjudicate.**
`GUIDS_PER_NODE`/"Tetris across the slots" (canonical_node.rs:791-805, zero
production consumers, cites an unverifiable 2026-06-29 ruling — real doctrine
awaiting a consumer, or excision candidate?). Drop-or-keep on the two invalid
probes (I lean drop: they legitimize geometric-CLAM as a live question).
`debug_assert_identity_unique` wiring into the mint path (TD-6 — the one fix
that ADDS a guarantee).

## 6. The mechanism summary (read this if nothing else)

Every hallucination this session had ONE shape: **answering from a name or a
doc-comment instead of the code, then building on the answer.** ClassId (name),
the v1 fold (name), cross_family_palette (doc-comment), in-RAM MailboxSoA
(doc-comment), FisherZTable-as-canon (doc-comment), the geometric-CLAM framing
(my own prior framing treated as ruling). The countermeasure that worked every
time it was applied: read the symbol's definition and count its real
consumers BEFORE citing it. The audit method in §5 of the session
(doc/comment-only vs code reference counts) mechanizes exactly that and cost
minutes. The operator's standing formulation: "the only problem when we fail
is when you don't read the code and handroll your own."

Session artifacts: PRs #855, #856 (merged); branch
`claude/medcare-rs-transcode-ruff-3y2olh` at 15f9b0a..(import-fix) unmerged;
probes in `crates/bgz17/examples/`, `crates/jc/examples/`,
`crates/helix/examples/`, `crates/lance-graph-planner/examples/`; audit files
in `.claude/board/exec-runs/` + `AUDIT-FIXLIST-2026-07-27.md`.
