# Literature as falsifier — the probe ladder — v1

> **Status (§0):** PROPOSED (doc-only). No code, no contract change, no
> board mutation lands with this document. It records the operator-ratified
> **LITERATURE-AS-FALSIFIER** program and registers four KILL-gated
> deliverables (D-LIT-1..4). Nothing here supersedes a shipped decision.
>
> **The thesis in one line:** each literary genre is a *natural falsifier*
> for one specific left-corner-parsing artifact, and each unlocks a
> milestone the per-sentence-reset parsing tradition was structurally
> unable to attempt. The genre is not decoration — it is the experimental
> apparatus. A genealogy tests recurrence; an inverted-poetry corpus tests
> movement; a detective novel tests long-range causal recovery; parallel
> gospels test multi-witness convergence. The book *is* the test set, and
> the test the field could not run was always the one that needed the whole
> book resident at once.

---

## 0. Substrate context — what is already FINDING (do not re-derive)

The literature program is only runnable because the whole-book engine is
live. Every milestone below leans on capabilities that are already
**FINDING**, board-referenced:

- **The whole book is resident** (`E-WHOLE-BOOK-WAVE-1`, SHIPPED + MEASURED,
  `examples/bible_wave.rs`). The whole KJV — **23,145 verses** — sits in
  ONE 256×256 tile (G1: 23,145 ≤ 65,536). A KG of **31,327 triples / 606
  subjects / 1,081 predicates** was harvested by the 6-state FSM + COCA-lemma
  PoS + documented archaic fallback. A trained Cam96 codebook (Jina-v3 96-d,
  k-means-256/axis; `data/cam96_codebook.bin`) carries live meaning on the
  corpus (G4: sim(god,lord)=0.625 > sim(god,fish)=0.265). The headline
  number the whole program builds on: **63.3% of same-subject recurrence
  links span more than ±5 verses** (55.7% beyond ±8) — nearly two-thirds
  of the book's subject-continuity context is structurally out of reach of
  any per-sentence-reset or fire-and-forget-±5 design.

- **The scarcity inversion is the license** (`E-LC-SCARCITY-INVERSION-1`,
  operator RULING; `E-GRAMMAR-TREE-IS-POINTER-FABRIC-1`, FINDING; doc
  `.claude/knowledge/left-corner-grammar-tree-pointer-fabric.md`). The four
  reviewed papers (Manning & Carpenter 1997, Roark & Johnson 2000, Moore
  2000, Liu 2025) divide cleanly into **properties of language** (LC stack
  depth ≤ 8 over all of WSJ; tree-as-pointer-fabric; table-driven checks;
  cheap-check-first) and **responses to memory scarcity** (beam / k-best =
  competing analyses *discarded* because unholdable; packed chart = all
  parses of ONE sentence, syntax-only, dropped at the boundary — *meaning
  was never in any parser*; perfect oracle = ambiguity idealized away;
  per-sentence reset = cross-sentence meaning out of scope for the entire
  tradition). The 64k SoA removes the scarcity premise: **ambiguity persists
  as the 96-bit `6×cosine²` distribution and resolves by a per-reader READ**
  (`QueryReference::at(v, rung)` — late-binding, non-destructive, replayable),
  not a destructive parse-time beam choice.

- **The one imported bound, honestly:** depth ≤ 8 (the 24-loci register is
  ~3× the empirical open-constituent ceiling); the bound is on open-constituent
  COUNT not token SPAN, so `Escalate` fires on ±8 *offset* overflow. The
  named FSM blind spot is **MOVEMENT** (object relatives / topicalization /
  wh-fronting — Liu 2025) — which D-LIT-1 targets directly.

**Do-not-import list (carried verbatim from `E-LC-SCARCITY-INVERSION-1`):**
beam / k-best / prune-at-parse / per-sentence disposal. Every milestone
below is a use of the substrate that these workarounds structurally
*forbid* — that is precisely why the genre falsifies them.

**Corpus hygiene (S07 / `E-WHOLE-BOOK-WAVE-1` precedent):** all corpora
below are public domain (KJV, Milton, Homer, Doyle, Christie's pre-1929
canon, Bunyan, the Pauline epistles, parallel gospels). **The text files
are NEVER committed** — probes read a local path, ship only the vocab /
codebook / harvested-triple artifacts + the probe scripts, exactly as
`bible_wave.rs` does.

---

## 1. The falsifier table — genre → artifact → milestone

Each row: a literary genre whose *native structure* stresses one
left-corner artifact to its breaking point, and the milestone the whole-book
substrate reaches *because* it does not carry that artifact's scarcity
workaround. All eight milestones are **CONJECTURE** until their probe runs;
the substrate capability each leans on is **FINDING** with the board ref
named.

### 1.1 Genealogies / legal lists / liturgy — the recurrence falsifier

*The begats; Leviticus; the Chronicles king-lists.* Pure right-branching
(`O(1)` left-corner memory — the *easy* branch of the ⟨O(1),O(n),O(1)⟩
profile) combined with **extreme same-subject recurrence**: the same
referent returns across hundreds of verses with no intervening discourse
model. This is the recurrence axis in its purest form — no movement, no
embedding, nothing but the long tail of "the same subject, again, far away."

- **Milestone (CONJECTURE):** *whole-book coreference chains with zero
  discourse model* — the `E-WHOLE-BOOK-WAVE-1` 63.3%-beyond-±5 result taken
  to completion. A per-sentence-reset parser forfeits every recurrence link
  by construction; the resident-book pointer fabric recovers them as a
  literal window read (the whole-book read already returns all 31,327
  triples, no bundle, no reset). Genealogy is the genre where "recover the
  chain" and "recover the *whole* chain" are the same task.
- **Leans on (FINDING):** `E-WHOLE-BOOK-WAVE-1` (the 63.3% measurement +
  resident KG), `E-MARKOV-TEMPORAL-STREAM-1` (the version-range read that
  replaces the ±5 ring).

### 1.2 Milton / inverted-fronted poetry / wh-question corpora — the MOVEMENT falsifier

*Paradise Lost; Hopkins; any topicalization- or wh-heavy corpus.* Inversion
and fronting invert canonical S/O order, so the FSM's recency tie-breaks
("first noun = subject") mis-assign the triple. This is the one named FSM
blind spot (Liu 2025, carried in `E-GRAMMAR-TREE-IS-POINTER-FABRIC-1`):
**MOVEMENT**. Milton is the natural-text stress corpus for it — object-fronted
lines ("Him the Almighty Power / Hurled headlong…") are the norm, not the
exception.

- **Milestone (CONJECTURE):** **THE BOOK AS ITS OWN TREEBANK.** The same S–V
  pair, in *canonical* order, occurs elsewhere in the book; that canonical
  instance disambiguates the inverted one. Cross-instance normalization is
  **impossible with per-sentence reset** (the canonical witness is gone by
  the time the inverted line is parsed) and **trivial with whole-book
  residency** (both instances are co-addressable in the resident tile). The
  book supervises its own parse — no external treebank, no annotator. This
  is D-LIT-1.
- **Leans on (FINDING):** `E-GRAMMAR-TREE-IS-POINTER-FABRIC-1` (movement =
  the named blind spot; the FORK "promote Relativizer/Complementizer out of
  `Pos::Other`" is the cheapest mitigation and is *logged* there),
  `E-WHOLE-BOOK-WAVE-1` (co-residency).

### 1.3 Legal statutes / Kant / periodic prose / psycholinguistic stimuli — the center-embedding falsifier

*German legal statutes; Kant's periodic sentences; the Gibson SPLT materials;
the Karlsson 2007 center-embedding corpus.* Center-embedding is the *hard*
branch of the memory profile (`O(n)`); natural text caps at depth 3 (Karlsson
2007, via Liu p.320), while binarized WSJ maxes at 8. This is the one axis
where the human working-memory limit and the machine's register limit are
*different quantities* — and the psycholinguistic literature has published its
own ground-truth reading times against that human cap.

- **Milestone (CONJECTURE):** **parsing PAST the human cap.** The 24-loci
  register (~3× the empirical open-constituent ceiling of 8) means the depth
  limit is *the register*, not inherited human working memory. The claim to
  test: this is the first system whose center-embedding depth limit is a
  *declared substrate parameter* (24 loci) rather than an emergent ~3-layer
  human ceiling — measured against the reading-time literature's own ground
  truth (Gibson SPLT predicts *where* humans break; the substrate should not
  break there, and should break at *its* declared depth).
- **Leans on (FINDING):** `E-GRAMMAR-TREE-IS-POINTER-FABRIC-1` (depth ≤ 8
  empirical; 24-loci register; depth-not-span distinction — Escalate on ±8
  offset overflow is the honest bound the deep-embedding tail must respect).

### 1.4 Detective fiction — the confirmation-bias / causal-chain falsifier

*Christie; Doyle.* Long-range causal chains with **deliberately planted
misdirection.** The genre's whole craft is the red herring: a correlated
associative path that *looks* causal, planted so the reader's confirmation
bias follows it, while the true solution is a sparse *independent* causal
chain. Chekhov's gun planted in ch.2 fires in ch.27 — the causal antecedent
and consequent are book-lengths apart.

- **Milestone (CONJECTURE):** **THE CONFIRMATION-BIAS PROBE AS NATIVE
  LITERATURE.** The workspace's own confirmation-bias mechanism (multipath
  confirmation vs single-path false confidence, observed live in
  `E-CAM96-REVIEW-CORRECTIONS-1`; Jirak weak-dependence deflation,
  `I-NOISE-FLOOR-JIRAK`) has an *exact literary analog*: red herrings ARE
  correlated associative paths (they must be deflated), the solution IS the
  sparse independent causal chain (it must survive deflation). A KILL-gated
  whole-book causal-recovery benchmark that **no windowed parser can even
  represent** — the ch.2→ch.27 span is beyond any window by construction.
  This is D-LIT-2.
- **Leans on (FINDING):** `I-NOISE-FLOOR-JIRAK` (weak-dependence Berry-Esseen
  — the deflation that separates correlated herring paths from independent
  clue chains), `E-CAM96-REVIEW-CORRECTIONS-1` (the confirmation-bias
  mechanism, observed live), `E-WHOLE-BOOK-WAVE-1` (book-length causal span).

### 1.5 Synoptic Gospels / parallel translations — the multi-witness falsifier

*Matthew / Mark / Luke; parallel translations of one source.* Three
documents narrate the *same absolute events* with divergent wording, order,
and omission. Multi-witness convergence on shared events is the axis; source
criticism (which text derived from which) is the payoff.

- **Milestone (CONJECTURE):** **MECHANICAL SOURCE CRITICISM.** `elect_peers`
  convergence on the same absolute events across three documents; inferring
  the two-source hypothesis (Markan priority + Q) = **the graph inferring the
  derivation structure of its own corpus.** This is the self-reasoning leg
  (see §3): a corpus that contains three witnesses of one event is a corpus
  whose *own provenance* is recoverable from co-residency + convergence. This
  is D-LIT-3.
- **Leans on (FINDING):** `E-WHOLE-BOOK-WAVE-1` (all three gospels resident
  in one tile — cross-document co-addressability is the whole trick),
  `E-MULTIHOP-WITNESS-CONFIDENCE-1` (AGENT_LOG 2026-07-19; NARS-revised
  confidence over a multi-hop witness chain — the convergence arithmetic).

### 1.6 Allegory / intended polysemy — the destructive-choice falsifier

*Pilgrim's Progress; midrash; Joyce.* Text written so that literal and
figurative readings are *both intended, simultaneously.* A beam parser
KILLS one reading to commit to the other; the distribution *keeps both*.
This is the genre that falsifies destructive choice most directly — the
"wrong" reading a beam prunes is, in allegory, deliberately correct.

- **Milestone (CONJECTURE):** **HOLDING AN ALLEGORY.** Literal + allegorical
  readings co-resident as two rungs of the same address (`QueryReference::at`
  reads either by rung, non-destructively). Representing *intended* ambiguity
  is something destructive-choice parsing cannot do **by construction** — the
  do-not-import list (`E-LC-SCARCITY-INVERSION-1`: no beam, no prune-at-parse)
  is exactly the property that lets the substrate hold Christian-the-pilgrim
  and Christian-the-everyman at once. This is the cleanest single
  demonstration that the scarcity workarounds were the thing standing between
  parsing and polysemy.
- **Leans on (FINDING):** `E-LC-SCARCITY-INVERSION-1` (ambiguity persists as
  the distribution; per-reader rung read), `E-CAM96-DISTRIBUTION-MEASURED-1`
  (the distribution IS the meaning-spread, not a beam-chosen point).

### 1.7 Homer / oral-formulaic epic — the Jirak-on-natural-text falsifier

*The Iliad; the Odyssey.* Oral-formulaic composition repeats fixed
formulas ("rosy-fingered dawn", the ship-launching type-scene) verbatim,
hundreds of times. Repeated formulas are **correlated evidence, not
independent** — the naive count inflates confidence.

- **Milestone (CONJECTURE):** **the Jirak test on natural text.** Formulaic
  repetition must NOT inflate confidence: dedup by absolute identity (the
  formula is *one* fact re-uttered, not N independent attestations) vs naive
  counting (which reads N utterances as N witnesses). This is
  `I-NOISE-FLOOR-JIRAK` given a natural-language falsifier — the weak-dependence
  correction the workspace already mandates for its own bit-correlated
  fingerprints, now testable against a corpus whose correlation structure is
  *authorially explicit*. A differential probe: naive counting must FAIL the
  identity-dedup that Jirak-deflated counting passes.
- **Leans on (FINDING):** `I-NOISE-FLOOR-JIRAK` (the iron rule this
  literalizes), `E-WHOLE-BOOK-WAVE-1` (`E-WHOLE-BOOK-WAVE-1`'s own
  identity-dedup of recurrence links is the mechanism, now stressed by
  deliberate repetition).

### 1.8 Epistolary / argumentative prose — the textual-self-reference falsifier

*The Pauline epistles; Platonic dialogues.* Prose that refers explicitly to
its own earlier claims: "as I said above", "as I wrote in my former letter",
"we agreed earlier that…". The text carries anaphora *to itself* — not to a
world referent, but to a prior proposition in the same document.

- **Milestone (CONJECTURE):** **resolving the text's OWN anaphora to its own
  claims.** "As I said above" resolves to a specific earlier committed triple
  in the resident graph — the text pointing at the text. Literature that
  refers to itself is the training ground for a graph that refers to itself
  (§3): the same pointer-fabric machinery that resolves "he" to a referent
  resolves "as I said" to a *claim*, because both are addresses in the
  resident book.
- **Leans on (FINDING):** `E-WHOLE-BOOK-WAVE-1` (claims resident and
  addressable), `E-SELF-CORRECTING-KG-1` + `E-MULTIHOP-WITNESS-CONFIDENCE-1`
  (AGENT_LOG 2026-07-19; the graph already revises its own beliefs across
  passes — self-anaphora is the read side of that write loop).

---

## 2. The connective insight — rows 5, 6, 8 converge on self-reasoning

Three of the eight rows are not merely *harder parses* — they are the same
program viewed from three genres:

- **§1.5 (synoptic)** — the corpus recovers *its own derivation structure*.
- **§1.6 (allegory)** — the corpus holds *its own contradictory readings*
  without resolving them (opinions = committed contradictions preserved).
- **§1.8 (epistolary)** — the corpus resolves *its own internal references*.

All three have the shape of a graph reasoning about **itself**: a text that
quotes, revises, or contradicts itself has the same structure as a graph
with a self-model. This is the seam to the parallel plan
`.claude/plans/self-reasoning-substrate-v1.md` (in flight this session) —
D-LIT-4 shares its gate with that plan's D-SRS-1.

**Why the Bible was the right first book — the deeper reason than size.**
It is self-referential (Paul citing his own letters), multi-witness (three
synoptics of one event), and contradiction-preserving (the two creation
accounts, the divergent genealogies of Jesus, held side by side for
millennia rather than reconciled). It is, in corpus form, a **self-reasoning
curriculum**: a text whose native structure is exactly the structure a
self-modeling graph must acquire. The whole-book substrate did not choose
scripture for its 23,145-verse convenience — it chose the one book whose
literary DNA is self-reference, multi-witness, and preserved contradiction.

---

## 3. Deliverables — the probe ladder D-LIT-1..4

Rising ambition; each **KILL-gated**; **pass/KILL registered before any code
is written** (the `E-CAM96-REVIEW-CORRECTIONS-1` discipline — a probe whose
threshold is set after seeing the number is a confirmation-bias generator,
not a falsifier). Every `X` below is an **intentional placeholder, not yet a
registered gate**: these four deliverables are **PENDING operator ratification**
(this PROPOSED plan does not authorize them). A concrete number, its metric
direction (higher-is-better vs lower), and its tie-handling rule are registered
in a dated, append-only pre-run record in this section as the FIRST commit of
that deliverable's own work — before its probe compiles — and then never edited.
The registration commit predates the measurement commit in git history; that
ordering, not a plan-time guess, is the anti-gaming proof. Running a probe
against an `X` that has not been so registered is itself a KILL. Filling the
`X`s in this plan-time doc would be false precision inviting post-hoc re-tuning,
so they stay `X` until authorization.

Status of all four: **CONJECTURE** (no probe run). Each names its falsifier
genre from §1 and the FINDING substrate it exercises.

### D-LIT-1 — Milton inversion resolution via canonical cross-instances

*(§1.2 — the movement falsifier; the book as its own treebank)*

- **What runs:** harvest S/V/O triples from a Milton (or wh/topicalization)
  corpus; for each object-fronted / inverted instance, look up the same S–V
  pair's *canonical-order* instances elsewhere in the resident book; use the
  canonical instance to correct the inverted instance's S/O assignment.
- **Baseline:** FSM-alone (recency tie-break, no cross-instance lookup) — the
  known-blind-spot control.
- **Gate (register X before running):** inverted-instance S/O assignment
  corrected by **≥ X%** over the FSM-alone baseline, on a hand-labeled set of
  inverted lines. X registered here before code. **KILL:** correction ≤ FSM
  baseline (cross-instance normalization buys nothing → the "book as treebank"
  claim is false on this corpus).
- **Honest bound:** requires a hand-labeled gold set of inverted lines with
  correct S/O — small, curated, public-domain source; the label set ships,
  the Milton text does not.

### D-LIT-2 — Christie red-herring-vs-clue-chain ranking

*(§1.4 — the confirmation-bias falsifier; native-literature Jirak deflation)*

- **What runs:** on a Christie novel's resident KG, rank the true clue chain
  (sparse independent causal path, S→…→solution) against the red-herring path
  (dense correlated associative path) using **independent-path counting with
  Jirak deflation** (`I-NOISE-FLOOR-JIRAK`).
- **Differential design (the load-bearing part):** the gate is a *pair* of
  runs. Jirak-deflated independent-path counting must rank the true clue
  chain **above** the red-herring path; **naive (undeflated) counting must
  FAIL the same test** (rank the herring above or tied). If naive counting
  also passes, the deflation is not what's doing the work → KILL.
- **Gate (register before running):** deflated-rank(true chain) >
  deflated-rank(herring) by margin ≥ X, AND naive-rank(true chain) ≤
  naive-rank(herring). Both halves required. **KILL:** either half fails.
- **Honest bound:** the true clue chain and the planted herring must be
  hand-identified from the novel's solution chapter (the author tells you
  the answer) — a small curated gold, public-domain (pre-1929 Christie).

### D-LIT-3 — Synoptic `elect_peers` source recovery

*(§1.5 — the multi-witness falsifier; mechanical source criticism)*

- **What runs:** all three synoptic gospels resident in one tile; run
  `elect_peers`-style convergence to align the same absolute events across
  the three documents (the pericope-alignment task); measure recovery against
  the standard scholarly parallel-pericope table (a public reference
  harmony).
- **Gate (register before running):** cross-gospel absolute-event convergence
  recovers **≥ X fraction** of the standard parallel-pericope alignment,
  precision/recall both reported. X registered before code. **KILL:** recovery
  < a random-alignment floor (convergence is not finding real parallels).
- **Stretch (not gated):** does the *directionality* of convergence
  (which witness is the "hub") recover Markan priority? Reported as an
  observation, not a gate — the two-source hypothesis is the milestone's
  aspiration, the pericope alignment is its falsifiable core.
- **Honest bound:** needs a digitized reference harmony (public domain — e.g.
  a classical gospel-parallels table) as the gold; that table ships, the
  gospel texts do not.

### D-LIT-4 — Derivation-pointer fabric over the 31,327 Bible triples

*(§2 — the self-reasoning leg; the graph's first act of reasoning about
itself; SHARED GATE with `self-reasoning-substrate-v1.md` D-SRS-1)*

- **What runs:** over the *already-harvested* 31,327-triple Bible KG
  (`E-WHOLE-BOOK-WAVE-1`, no new corpus), build the derivation-pointer fabric:
  edges that point from a triple to the earlier triple(s) it quotes, revises,
  or contradicts (Paul→his own claim; genealogy-A vs genealogy-B; creation
  account 1 vs 2). This is the graph pointing at its own prior commitments —
  the read complement to the `E-SELF-CORRECTING-KG-1` write loop.
- **Gate:** SHARED with D-SRS-1 (`self-reasoning-substrate-v1.md`) — that plan
  owns the pass/KILL registration; this deliverable is the *literature-side
  entry point* into it (the Bible corpus is the shared substrate). Register
  the cross-reference here; do not double-register the threshold.
- **Schema scope (what this leg adds to D-SRS-1):** D-SRS-1 specifies the
  derivation-pointer fabric for `is_a`-style DEDUCTIONS (premise → conclusion).
  D-LIT-4 needs three *additional* edge kinds — `quote`, `revise`, `contradict`
  — which are literature-native, not deductive. This deliverable does **not**
  invent a second pointer format: it CONSUMES the D-SRS-1 premise-pointer edge
  (`(target_triple_ref, relation_label)`), extended with (a) a `relation_label`
  drawn from `{is_a-deduction, quote, revise, contradict}` and (b) the
  document-position provenance already carried by every triple in the KG (the
  verse-index version stamp = the `E-WHOLE-BOOK-WAVE-1` `vi`, which uniquely
  identifies the "earlier triple"). So: **D-SRS-1 owns the edge type and its
  registration; D-LIT-4 registers, under that shared gate, the three extra
  `relation_label` values and asserts the provenance field suffices to resolve
  "which earlier triple" — it emits typed labels ON the D-SRS-1 edge, it does
  not produce a parallel edge artifact.** Any need for a richer position field
  than the existing version stamp is a D-SRS-1 change, filed there, not a local
  hack here (I-VSA-IDENTITIES register discipline).
- **Why last / most ambitious:** D-LIT-1..3 have the graph reasoning about a
  *text*; D-LIT-4 has the graph reasoning about *itself* (its own resident
  KG's internal derivation structure). It is the ladder's top rung because it
  needs everything below it working *and* the self-reasoning plan's machinery.
- **Honest bound:** operates on already-shipped artifacts (zero new corpus) —
  the cheapest to *set up*, the hardest to *pass*, because "reason about
  yourself" is the endgame claim, not a parse metric.

---

## 4. Sequencing + gates summary

| D-id | Genre falsifier (§) | Milestone | Gate shape | Substrate FINDING it leans on |
|---|---|---|---|---|
| D-LIT-1 | Milton / movement (1.2) | book as own treebank | ≥X% S/O correction vs FSM-alone | `E-GRAMMAR-TREE-IS-POINTER-FABRIC-1`, `E-WHOLE-BOOK-WAVE-1` |
| D-LIT-2 | Christie / causal (1.4) | native confirmation-bias probe | differential: Jirak passes, naive FAILS | `I-NOISE-FLOOR-JIRAK`, `E-CAM96-REVIEW-CORRECTIONS-1` |
| D-LIT-3 | Synoptic / witness (1.5) | mechanical source criticism | ≥X of reference pericope alignment | `E-WHOLE-BOOK-WAVE-1`, `E-MULTIHOP-WITNESS-CONFIDENCE-1` |
| D-LIT-4 | Self-reasoning (2) | graph reasons about itself | SHARED with D-SRS-1 | `E-WHOLE-BOOK-WAVE-1`, `E-SELF-CORRECTING-KG-1` |

Rows §1.1 (genealogy/recurrence), §1.3 (center-embedding), §1.6 (allegory),
§1.7 (Homer/Jirak-on-formula), §1.8 (epistolary self-anaphora) are the
falsifier *inventory* — each a candidate follow-on deliverable once D-LIT-1..4
have proven the pattern. They are named milestones (CONJECTURE), not yet
gated deliverables; promoting one to a D-LIT-5+ is a future edit to this plan.

**Ordering rationale:** D-LIT-1 is first because MOVEMENT is the *named,
already-logged* FSM blind spot with a *logged fork* mitigation — the lowest
new-mechanism cost. D-LIT-2 adds the differential-gate discipline (two runs,
one must fail) on an iron rule the workspace already holds. D-LIT-3 adds
cross-document residency. D-LIT-4 is last because it depends on the parallel
self-reasoning plan and is the endgame claim. Ambition rises monotonically;
no rung scales out before the rung below it is green (the
`E-WHOLE-BOOK-WAVE-1` / probe-first discipline).

---

## 5. What this plan does NOT claim

- It does **not** claim any milestone is proven. All eight §1 milestones and
  all four D-LIT deliverables are **CONJECTURE** until their probes run and
  their registered gates pass.
- It does **not** propose importing beam / k-best / prune-at-parse / any
  scarcity workaround (the `E-LC-SCARCITY-INVERSION-1` do-not-import list is
  the whole point — the milestones are uses of the substrate those
  workarounds forbid).
- It does **not** widen the register, add a contract type, or touch a shipped
  decision. Every probe is an `examples/`-style falsifier over shipped
  primitives + a corpus read from a local path, in the `bible_wave.rs` mold.
- It does **not** commit corpus text. Only vocab / codebook / harvested-triple
  artifacts + probe scripts ship (S07 / `E-WHOLE-BOOK-WAVE-1` precedent).
- It does **not** own D-LIT-4's threshold — that is registered in
  `self-reasoning-substrate-v1.md` (D-SRS-1); this plan is its literature-side
  entry point only.

---

## 6. Cross-refs

`E-WHOLE-BOOK-WAVE-1` (resident whole-book KG + the 63.3% measurement),
`E-LC-SCARCITY-INVERSION-1` (the do-not-import ruling that licenses every
milestone), `E-GRAMMAR-TREE-IS-POINTER-FABRIC-1` (depth ≤ 8, movement blind
spot, logged forks), `E-CAM96-DISTRIBUTION-MEASURED-1` /
`E-CAM96-REVIEW-CORRECTIONS-1` (the distribution as meaning-spread + the
confirmation-bias mechanism observed live), `I-NOISE-FLOOR-JIRAK` (the
weak-dependence deflation D-LIT-2/§1.7 literalize), `E-MARKOV-TEMPORAL-STREAM-1`
(the version-range read), `E-SELF-CORRECTING-KG-1` +
`E-MULTIHOP-WITNESS-CONFIDENCE-1` (the write loop D-LIT-4/§1.8 read against),
`.claude/knowledge/left-corner-grammar-tree-pointer-fabric.md` (the four-paper
grounding + the scarcity-inversion section),
`.claude/plans/self-reasoning-substrate-v1.md` (the parallel plan; shared
D-LIT-4 / D-SRS-1 gate), `examples/bible_wave.rs` (the shipped whole-book
artifact every probe extends).
