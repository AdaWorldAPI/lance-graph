# Brainstorm — learning on the V3 substrate

> **Status: BRAINSTORM — a discussion reference (2026-08-22).** Nothing here is
> ratified; no D-ids, no plan row owed. Grades: **[G]** shipped/cited,
> **[H]** plausible+boundable, **[S]** analogy-grade.
>
> **This document was rewritten by a 5+3 council, then corrected again by that
> council's own reviewers.** The
> first draft raised seven threads at equal weight. The council found that
> **five of the seven were already decided** — one killed by a prior council,
> one answered the same day, one deferred by the operator the day before, and
> two already banked as findings — and that its single pre-registered number
> had no artifact behind it. What follows keeps the two questions that are
> genuinely open and turns the other five into pointers, which is what a
> discussion reference is for. The first draft's errors are recorded, not
> erased: §0 is the list — including one error the *rewrite itself* introduced
> (row 9), which the reviewers caught and which is recorded in place.

---

## §0 What the first draft got wrong (recorded so it is not re-proposed a third time)

| first draft claimed | what was already true |
|---|---|
| RL credit as a "TD-error sign + **magnitude** bucket" in spare CE64 bits | **A council already killed this and wrote it down against resurfacing.** EPIPHANIES:8560, *"Pruned dead-ends the council killed (recorded so they don't resurface): … storing any magnitude (regret price, oracle eval, lindy age, proof hash) in a reserved A9 locus (pointers-not-magnitudes + RESERVE-DON'T-RECLAIM)"*. |
| bits 59..63 as an open design space | **The band has a ratified contract.** `INTEGRATION_PLANS` → `dacr7-band-reading-contract-v1.md` (council-ratified, G1..G10b pre-registered) and `known-unknown-handover-network-v1.md` (the operator's own 59..63 framing, fence measured twice: *"`↑n` is stacking, never widening"*). The live question is `BandReading::project`, not whether to pack new bits. |
| a BPE reading of `6×2×8bit` as a fresh **[S]** idea — **only the ontology/codec half (A below)** | **Asked and answered the same day.** `E-BPE-IS-RHYME-VQ-IS-THE-MECHANISM-FOR-6X2X8BIT-1` (EPIPHANIES:91, 2026-08-22, arxiv-grounded): *"RHYME, not mechanism"* — the hazards are already named verbatim (*"loss of `is_ancestor_of` (BPE guarantees no subsumption)"*, *"frequency-optimal replacing distance-optimal"*), with three pre-registered probes stronger than the one the draft proposed. |
| BNN over `6×2×8bit` as a fresh **[H]** idea | **Raised by the operator 2026-08-21 and DEFERRED with a named blocker.** `handovers/2026-08-21-2330:107-115`: not taken into v2 because `edge_v3.rs:29-36` says the register is *"a packed EDGE REGISTER, **NOT** a slot-pure §3 facet"*, and adopting it would be **a sixth homonym**. It needs its own deliverable and its own resolution of the typed-register-vs-content-blind contradiction. |
| Hebbian strengthening "via the plasticity counter" | **The exact conflation `E-BASIN-NOT-EDGE-PLASTICITY` killed** (EPIPHANIES:14182, *"the 4th-strike object conflation"*): "plasticity" names a cold-path basin cooling knob AND a hot-path per-edge Hebbian state, and they do not compose. With `Plasticity` = tenant 7 that was three distinct objects under one word in a single paragraph. |
| "BNN" as a Bayesian neural network, cited to `ndarray/src/hpc/bnn.rs` | **That file is a BINARY neural network** — *"Binary weights and activations (1-bit)… XNOR + popcount"*. Deterministic, bit-exact, no distributions. The whole Bayesian mapping rested on a name. |
| a "deterministic BNN" (coprime-walk sampling) satisfying Bayesian uncertainty *and* the never-RNG ruling | **Refuted on the mathematics.** A coprime-integer walk is a fixed full-period permutation: the same address always returns the identical value, so it has **zero draw-to-draw variance** and cannot represent a posterior. It is a low-discrepancy dither, which is a different object from weight-distribution sampling. |
| `ValueTenant::EpisodicBasin = 15` cited as **[G]** | Tenant 15 exists on an **unmerged branch**, not on `main`. `[G]` means shipped; a branch is not shipped. |
| "measured 0.727 / 0.455 / 0.273" as the anaphora baseline, on "the German relative-clause set" | **The numbers are real and banked — the CORPUS and the MECHANISM were both mislabelled.** `E-L9-REAL-TEXT-1` (EPIPHANIES:8621, 2026-07-21, FINDING, deliverable `crates/jc/examples/l9_loci_real_text.rs`, *"Gates registered before the first run; never loosened"*) records antecedent loci **0.727**, agreement-blind baseline **0.273**, shipped noun-only rule **0.455** — on **three public-domain Aesop fables in English**, not a German set. And 0.727 is **loci CHAINING**: the entry is explicit that *"agreement adds +0.18, chaining adds +0.27"*. |

**The pattern.** The table has **nine** rows (an earlier revision of this
paragraph said eight and mis-split them — corrected here rather than quietly).
They fall into three groups, not two:

- **A name taken for a mechanism** — BNN (binary vs Bayesian), plasticity
  (three distinct objects under one word), and the `6×2×8bit` homonym the
  2026-08-21 handover was already counting to six.
- **Proposing without reading the ledger** — the killed magnitude-in-a-locus,
  the ratified 59..63 contract, the same-day BPE finding, the deferred BNN
  reading, and a `[G]` citing an unmerged branch.
- **Getting the retraction wrong too** — row 9: the baseline was called absent
  when it exists; the real defect was narrower (wrong corpus, wrong mechanism
  label). A correction can overclaim in the opposite direction, and this one
  did.

The third group is the one worth remembering: it was caught by this document's
own review, not by its author, and only because the reviewer re-ran the search
the first pass had already reported as empty.

---

### ⊘ The word "BPE" spans THREE claims, and row 3 retired all three

Corrected 2026-08-22 by the fathoming report
(`.claude/brainstorms/2026-08-22-behavioral-ir-fathoming.md` §M). This is a
**fourth** instance of the pattern named above — a name taken for a mechanism —
and unlike the other three, this one is *this document's own*.

| | claim | status |
|---|---|---|
| **A** | BPE is the *mechanism* behind the `6×2×8bit` centroid/ontology codebooks | **RETIRED, correctly** — `E-BPE-IS-RHYME-VQ-IS-THE-MECHANISM-FOR-6X2X8BIT-1`. Its named hazards (*loss of `is_ancestor_of`*, *frequency-optimal replacing distance-optimal*) are **codebook** hazards. |
| **B** | BPE / Sequitur / Re-Pair induce reusable behavioural **macros over executed `(FnIndex : Value)` traces** | **NOT TESTED, NOT RETIRED — and published-positive.** FAST (arXiv:2501.09747, 2025) and Subwords as Skills (arXiv:2309.04459, NeurIPS 2024) do exactly this over action sequences; Sequitur (JAIR 7, 1997) and Re-Pair (DCC 1999) supply the exact-reversibility half. Blocked here by a missing interpreter, not by the cited finding. |
| **C** | BPE merge **rank** as a cheap surprisal statistic | separable from both; untouched by the cited finding. |

**A's evidence does not reach B.** Every falsifier in the cited finding concerns
centroid allocation and prefix containment; none of them says anything about
whether a *trace* can be merged into macros. A retraction requires evidence just
as an assertion does — the rule this document itself adds in §4 — and row 3
broke that rule one section above where it was written.

## §1 The two threads that are actually open

### T-A — a learned policy for anaphora resolution (was T4)

**Ships [G]:** context is a version-range read over the temporal sorted stream
(`E-MARKOV-TEMPORAL-STREAM-1` — any width, per-reader rung, replayable).
`Locus::Antecedent` is a signed ±8 nibble; the binder is escalate-never-clamp
(`probe_antecedent_binder.rs:171`, *"refused, not clamped"*); beyond ±8 is a
basin edge — identity discriminates, not position. German supervision exists in
the `de-bundle` Release asset (relative-pronoun, satzklammer, valency tables).

**The idea [H]:** learn the *resolver policy*, not the answers. Action space =
cue weights shaped like `ScanParams`' seven knobs; credit = downstream coherence
read from the witness lane. Policy parameters would need a home — the Learned
lane is the obvious candidate, **and §2 is exactly why this thread may not yet
assume it.**

**Frictions (all four, not one):**
- **A baseline exists — for a different corpus.** `E-L9-REAL-TEXT-1`
  (EPIPHANIES:8621) banks 0.727 / 0.455 / 0.273 with gates registered before the
  run, but on **three English Aesop fables**, and its 0.727 is loci **chaining**
  rather than agreement alone. The gate for a German-set policy is therefore not
  missing in principle — it is **un-measured on that corpus**, and the first
  deliverable is the German run against those same registered gates. Cheap,
  because the harness already exists.
- **Credit shape is constrained.** `E-CHAIN-PARITY-CREDIT-1` requires credit
  *"per proof-carrying `RecipeStep`, **never per endpoint**"* — it exists to
  block the self-referential free-energy-inflation hack. Endpoint-shaped credit
  is out.
- **Scalar credit is operator-deferred.** `E-REGRET-PRICER-1` is ⏸ DEFERRED — its own
  wording is *"Park until real-world-business-logic control is in scope; trigger
  = the stack begins driving real external systems"*.
- **D-QUANTGATE applies to the exploration itself** — any stochastic arm must be
  the address-derived coprime walk, not RNG. The first draft named this rule in
  its closing box but not on this thread.

**Gate (rewritten to be a twin):** measure the three arms on the German
relative-clause set, **record the numbers in the repo**, then ask whether a
learned mixture beats the recorded number **out-of-sample**. Can-it-fire: the
mixture wins. Can-it-stay-silent: it must NOT win on shuffled labels.

### T-B — potholes as the training set for escalation routing (was T5)

**Ships [G]:** the rung ladder is operator-ruled (`persona-vs-rung-ladder.md`:
rung 2 = 144 verb atoms, rung 3 = the 34 NARS tactic recipes = THE runbooks).
Escalations are counted today, so a baseline is free.

**The idea [H]:** every escalation / Unbound grounding / free-energy spike is a
labeled miss **at a rung**. Learn pothole → tactic-recipe **dispatch** (routing
among the fixed 34), never learned answers. This is the one thread the council
found both genuinely new and cheaply falsifiable.

**Frictions:**
- **YIELDS the ladder** (`F7`): routing among fixed recipes adds no rung-3
  content — the 34-lock holds, and *"any new rung-3 recipe #35"* is itself on the
  killed-dead-ends list.
- **The overlay is a pruner, never a proof** — the learner proposes, the witness
  disposes.
- **No lane or writer is named yet.** Single-writer ownership needs one before
  any of this is more than a sketch — see §2.
- **"BNN" must not be used for this.** Whatever this classifier is, it is not
  the binary-NN in `ndarray`, and reusing the name is how the last three errors
  happened.

**Gate:** at fixed accuracy, does learned dispatch reduce the Escalate rate
versus the static style table, **out-of-sample**? Without the out-of-sample arm a
memorising dispatcher passes trivially.

**⊘ Blocked — and the blockers are now named** (fathoming report, 2026-08-22):

- **Prerequisite: an interpreter.** This thread assumed pothole → dispatch
  labels are obtainable. They are not: **nothing executes the recipe IR.**
  `ogar-loco` contains no `execute` / `eval` / `interpret` / `step` / `run`; its
  own `telemetry.rs` says it *"only knows whether a candidate parses, casts, and
  segments"*; and `recipe_vocab`'s module doc disclaims execution outright.
  `ladder_program()` is a static ordering, not a corpus.
- **The label is lossy at the source.** `refusal_of`
  (`recipe_vocab.rs:313-326`) checks the ceiling and **returns**, so
  `nan_disqualifier` is *never called* when both gates trip — the second cause is
  not hidden, it is never computed. A learner would train on an `AboveCeiling`
  class silently merging "too deep" with "too deep AND ungrounded".
- **Rubicon is excluded** as feature or label: `overlap()` can exceed 1.0
  (`union` absorbs into one minimal antichain while `intersect` inserts per
  covering pair), breaking the bound `persistence_gain` and `verdict` rest on.

---

## §2 The one cross-thread defect worth keeping

Two savants found it independently from different angles: the first draft had
**three separate proposed writers into `ValueTenant::LearnedStyle`** — an RL
update, a NARS revision, and a loco parameter table — and never reconciled them
to one owner. Single-writer ownership is per **lane**; three writers into twelve
atoms would need arbitration **per atom**, which nothing specifies.

Any future proposal that writes the Learned lane must say which atom, under
which owner, or it is an orphan write.

## §3 Pointers, not threads (what the first draft should have been)

- **loco + learning skills** — determinism is kept by letting learning in at
  exactly two doors (Learned-lane parameters, trace-compiled templates), both
  behind the shipped held-out promotion gate, with the compile-down direction
  intact (*templates never degrade into prompts*). This is a restatement of what
  already ships, not a proposal, and the first draft gave it no gate at all.
- **tagging without a search sidecar** — the premise is banked
  (`E-CLASSID-CANON-HIGH-IS-A-CLUSTERED-INDEX`), but with **the caveat the first
  draft dropped**: `NodeGuid` stores classid little-endian, so a **raw prefix
  scan does NOT walk domain-first** — decode the u32 or use an order-preserving
  big-endian rendering (codex #629). And Tantivy is on the roadmap as an
  **in-binary** component (`old-stack-capability-parity.md:17,72`), not a sidecar
  to be eliminated — arguing to remove it means retiring a roadmap row, which is
  an operator call.
- **an LSTM / recurrence reading of the register** — **restored here after the
  council found it deleted outright rather than demoted.** The first draft
  proposed a recurrence whose hidden state IS the 12-byte register, with gates as
  palette-table lookups (the bgz-tensor attention-as-lookup lineage) and per-rail
  int8 requant between steps (the tesseract `E-OCR-LSTM-1` precedent, byte-parity
  proven). The BPE finding does **not** touch this: it rules on how centroids are
  *allocated*, not on whether the register can carry a recurrence. It is also
  distinct from the deferred `6×2×8bit`-as-BNN item, which is about a typed edge
  register. Open, ungated, and nobody's — but it should not vanish.
- **BPE merge RANK as a surprisal statistic** — likewise separable, likewise
  restored. `E-BPE-IS-RHYME-…` answers whether BPE is the *mechanism* behind the
  tile codebooks (it is not). It does not address whether a merge rank, being a
  frequency order, is a cheap `-log p` proxy that could modulate the
  `FieldModulation` knobs. That question is untouched by the finding that
  retired its parent thread.
- **the triangle** — `Frozen`/`Learned`/`Explore` already ship with a held-out
  promotion gate. Relabelling them Hebbian/Bayesian/BNN adds no mechanism, and
  the 226-atom cognitive palette256 codebook such a mixer would read *"does not
  exist in code yet"* (`TD-TRI-1-P4-OBLIGATIONS`).

## §4 The honesty box (unchanged, and it is the general case)

Two savants noted the first draft's real generalisation was this box, not its
seven threads — and that the threads split across **two** orthogonal axes
(*reward = a read over already-persisted structure*; *the address is the index*),
which a flat seven-item list hid.

- Every learning **write** names its single writer, its lane, **and its atom**.
- Every **reward** is a read (zero-copy law).
- Every stochastic element is **address-derived deterministic** (D-QUANTGATE) —
  and a deterministic generator is a dither, not a posterior.
- Every new codebook or merge table is **version-gated**; changing the basis
  under stored codes is silent corruption.
- Nothing touches CE64 bit fields without the operator re-opening M20.
- "Superposition" of the triangle is a **mixture**, never a bundle
  (I-VSA-IDENTITIES).
- **Frequency is not success — and that now has a citation, not a house rule.**
  Macro-FF (Botea et al., JAIR 24, 2005) ships a four-stage pipeline *because*
  raw frequency-derived macro candidates are unusable without a filter/rank
  stage; Newton & Levine (ECAI 2010) report a measured case where a macro used
  without control rules performs **worse than the no-macro baseline**. The
  failure mode is not "more search nodes" — it is *actively wrong choices
  compiled into fast-to-select composite operators*.
- **Read the ledger before proposing.** Five of the first draft's seven threads
  were already decided; the cost of checking was minutes and the cost of not
  checking was this rewrite.
- An idea graduates only through a probe with pre-registered gates — and its
  baseline is **recorded in the repo** before the probe runs, not remembered.
- **A retraction is a claim and needs the same evidence as an assertion.** "This
  does not exist" is falsifiable, and row 9 of §0 was falsified: a search that
  came back empty was reported as absence, and the entry was sitting in
  `EPIPHANIES.md` the whole time. Re-run the search before writing the word
  *nowhere*.
