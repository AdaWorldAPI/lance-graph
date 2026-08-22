# Brainstorm — learning on the V3 substrate

> **Status: BRAINSTORM — a discussion reference (2026-08-22).** Nothing here is
> ratified; no D-ids, no plan row owed. Grades: **[G]** shipped/cited,
> **[H]** plausible+boundable, **[S]** analogy-grade.
>
> **This document was rewritten by a 5+3 council after its first draft.** The
> first draft raised seven threads at equal weight. The council found that
> **five of the seven were already decided** — one killed by a prior council,
> one answered the same day, one deferred by the operator the day before, and
> two already banked as findings — and that its single pre-registered number
> had no artifact behind it. What follows keeps the two questions that are
> genuinely open and turns the other five into pointers, which is what a
> discussion reference is for. The first draft's errors are recorded, not
> erased: §0 is the list.

---

## §0 What the first draft got wrong (recorded so it is not re-proposed a third time)

| first draft claimed | what was already true |
|---|---|
| RL credit as a "TD-error sign + **magnitude** bucket" in spare CE64 bits | **A council already killed this and wrote it down against resurfacing.** EPIPHANIES:8560, *"Pruned dead-ends the council killed (recorded so they don't resurface): … storing any magnitude (regret price, oracle eval, lindy age, proof hash) in a reserved A9 locus (pointers-not-magnitudes + RESERVE-DON'T-RECLAIM)"*. |
| bits 59..63 as an open design space | **The band has a ratified contract.** `INTEGRATION_PLANS` → `dacr7-band-reading-contract-v1.md` (council-ratified, G1..G10b pre-registered) and `known-unknown-handover-network-v1.md` (the operator's own 59..63 framing, fence measured twice: *"`↑n` is stacking, never widening"*). The live question is `BandReading::project`, not whether to pack new bits. |
| a BPE reading of `6×2×8bit` as a fresh **[S]** idea | **Asked and answered the same day.** `E-BPE-IS-RHYME-VQ-IS-THE-MECHANISM-FOR-6X2X8BIT-1` (EPIPHANIES:91, 2026-08-22, arxiv-grounded): *"RHYME, not mechanism"* — the hazards are already named verbatim (*"loss of `is_ancestor_of` (BPE guarantees no subsumption)"*, *"frequency-optimal replacing distance-optimal"*), with three pre-registered probes stronger than the one the draft proposed. |
| BNN over `6×2×8bit` as a fresh **[H]** idea | **Raised by the operator 2026-08-21 and DEFERRED with a named blocker.** `handovers/2026-08-21-2330:107-115`: not taken into v2 because `edge_v3.rs:29-36` says the register is *"a packed EDGE REGISTER, **NOT** a slot-pure §3 facet"*, and adopting it would be **a sixth homonym**. It needs its own deliverable and its own resolution of the typed-register-vs-content-blind contradiction. |
| Hebbian strengthening "via the plasticity counter" | **The exact conflation `E-BASIN-NOT-EDGE-PLASTICITY` killed** (EPIPHANIES:14182, *"the 4th-strike object conflation"*): "plasticity" names a cold-path basin cooling knob AND a hot-path per-edge Hebbian state, and they do not compose. With `Plasticity` = tenant 7 that was three distinct objects under one word in a single paragraph. |
| "BNN" as a Bayesian neural network, cited to `ndarray/src/hpc/bnn.rs` | **That file is a BINARY neural network** — *"Binary weights and activations (1-bit)… XNOR + popcount"*. Deterministic, bit-exact, no distributions. The whole Bayesian mapping rested on a name. |
| a "deterministic BNN" (coprime-walk sampling) satisfying Bayesian uncertainty *and* the never-RNG ruling | **Refuted on the mathematics.** A coprime-integer walk is a fixed full-period permutation: the same address always returns the identical value, so it has **zero draw-to-draw variance** and cannot represent a posterior. It is a low-discrepancy dither, which is a different object from weight-distribution sampling. |
| `ValueTenant::EpisodicBasin = 15` cited as **[G]** | Tenant 15 exists on an **unmerged branch**, not on `main`. `[G]` means shipped; a branch is not shipped. |
| "measured 0.727 / 0.455 / 0.273" as T4's pre-registered baseline | **The numbers appear nowhere in the repo** — no board file, plan, probe report, test, or commit. A session-memory claim with no artifact, offered as the document's only pre-registered gate. |

**The pattern, twice over.** Three of the eight are the same failure the
workspace keeps paying for: **a name taken for a mechanism** — BNN (binary vs
Bayesian), plasticity (three objects), and the `6×2×8bit` homonym the 2026-08-21
handover was already counting to six. The other five are a different failure:
**proposing without reading the ledger.** Both are cheap to avoid and neither
was.

---

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
read from the witness lane. Policy parameters would live in the Learned lane.

**Frictions (all four, not one):**
- **The baseline does not exist in the repo.** Until the comparison is measured
  and recorded as an artifact, this thread has **no gate**. That is the first
  deliverable, and it is cheap.
- **Credit shape is constrained.** `E-CHAIN-PARITY-CREDIT-1` requires credit
  *"per proof-carrying `RecipeStep`, **never per endpoint**"* — it exists to
  block the self-referential free-energy-inflation hack. Endpoint-shaped credit
  is out.
- **Scalar credit is operator-deferred.** `E-REGRET-PRICER-1` is ⏸ DEFERRED with
  a stated trigger (*"when the stack begins driving real external systems"*).
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
- **Read the ledger before proposing.** Five of the first draft's seven threads
  were already decided; the cost of checking was minutes and the cost of not
  checking was this rewrite.
- An idea graduates only through a probe with pre-registered gates — and its
  baseline is **recorded in the repo** before the probe runs, not remembered.
