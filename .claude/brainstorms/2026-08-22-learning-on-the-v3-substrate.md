# Brainstorm — learning on the V3 substrate (BPE pairs, the triangle, RL signals)

> **Status: BRAINSTORM — a discussion reference, written at operator request
> (2026-08-22). Nothing here is ratified, nothing is a plan (no D-ids, no
> INTEGRATION_PLANS row owed), and every claim is graded:**
> **[G]** = shipped/cited, **[H]** = plausible + boundable, **[S]** = analogy-grade.
> The falsifiability rule applies to this file's own future: an idea leaves this
> document only through a probe with pre-registered gates, never by being liked.

Seven threads. Each: **what ships today** (cited) → **the idea** → **the
friction** (which ruling it touches) → **the cheapest falsifier**.

---

## T1 — The RL reward carrier: CausalEdge64 bits 59..63 vs a facet lane

**Ships [G]:** CE64 survives as exactly THREE carriers (mailbox baton edge,
perturbation baseline, p64 palette address) — the awareness-mantissa is retired
(M20), and the ruling is explicit: *"Do NOT extend CausalEdge64 bit fields to
carry new awareness semantics; new semantics land as facet layouts"*
(`le-contract.md` § "Let go of the cramped 64-bit register"). The v2 layout
already reclaimed bits 52–63 for plasticity[2] + W-slot + lens + spare, and that
reclaim produced the 5-instance `I-LEGACY-API-FEATURE-GATED` catalogue.

**The idea:** every causal edge carries its own credit trace — a TD-error sign +
magnitude bucket in the high spare bits ("CE-V3" as a read-mode of the same
u64), so reinforcement is edge-local and needs no side table.

**The friction:** this is verbatim the retired direction. Packing a new
awareness semantic into CE64 bits re-opens M20, and history says every reclaim
of those bits cost a field-isolation matrix + version gate. The ruled home for
new width is a **facet lane**.

**The reconciliation worth discussing instead [H]:** the reward may not need any
new bits at all. `CausalWitness` already carries `Quorum` (locus 14) and
`Contradiction` (locus 15); `Plasticity` (tenant 7) and `Energy` (tenant 6)
already exist. An RL update that **reads** witness statistics (`belief_runs` is
shipped in `witness_fabric.rs`) and **writes only the Learned lane** obeys the
zero-copy law (reward = a read), single-writer ownership, and M20 — with zero
new carriers.

**Falsifier:** a probe computing TD-style credit from Contradiction-locus runs
on a real corpus vs the same corpus shuffled; pre-register that the credit
discriminates (and by how much) before any policy consumes it.

## T2 — The triangle as learning "superposition": frozen / learned / explore ↔ Hebbian / Bayesian / BNN

**Ships [G]:** `ValueTenant::{FrozenStyle=10, LearnedStyle=11, ExploreStyle=12}`
— 12 palette256 atoms each. Learned is NARS-revision-written by the L4 seam and
**promotes to frozen only after winning the held-out arm**; Explore is
deterministic address-derived jitter (D-QUANTGATE coprime walk, **never RNG** —
replay holds). ndarray ships a BNN substrate (`src/hpc/bnn.rs`).

**The idea:** read the triangle as three learning regimes over one policy shape:
Hebbian (co-activation strengthening via the plasticity counter) feeding
Frozen's checkpoint; Bayesian (NARS truth revision — already what Learned *is*)
in the middle; BNN (weight-distribution sampling) as the Explore arm. The
"superposition" is the dispatch reading all three atoms and mixing.

**The friction, twice:**
1. *"Superposition" must not mean VSA bundling.* Palette atoms are content
   registers; I-VSA-IDENTITIES forbids superposing content codes. The sound
   reading is a **mixture/selector**: three stored policies, one arbiter —
   never a bundled code.
2. *A BNN samples; Explore is ruled deterministic.* Reconciliation [H]: make
   the BNN's noise source the coprime walk itself — address-derived,
   bit-replayable sampling. A **deterministic BNN** satisfies both the
   Bayesian-uncertainty idea and the replay ruling.

**Falsifier:** the promotion gate already defines the metric. Add the explore
arm and measure regret vs the static schedule on the held-out arm; the mixture
earns its complexity only if it beats both pure arms.

## T3 — A BPE reading of the `6×(8:8)` register (and an LSTM as 6 × BPE lanes)

**Ships [G]:** L1–L4 are `6×(8:8)`; L4 is palette256² CAM_PQ ("digital" style,
similarity = one table read). OGAR requires codebooks be **hierarchical 4⁴** so
a byte's nibbles are the centroid's ancestry — that is what keeps prefix routing
rigorous. Precedent for int8 recurrence: the tesseract-recognizer LSTM is
byte-parity-proven with ±127 clip requant between timesteps (`E-OCR-LSTM-1`).

**The idea [S]:** treat each `(8:8)` pair as a **BPE merge** — two symbols
promoted to one codebook slot by frequency — and the six rails as a six-deep
merge stack. "Training" = minting new centroids for frequent pairs, constrained
to respect 4⁴ ancestry so `is_ancestor_of` survives. Then an LSTM whose hidden
state IS the 12-byte register: gates as palette-table lookups (the bgz-tensor
attention-as-lookup precedent), recurrence as per-rail requant — an LSTM over
six BPE lanes, no floats in the hot path.

**A second, cheaper BPE hook [S]:** BPE merge **rank** is a frequency statistic,
i.e. a `-log p` proxy. A rare pair is high surprisal. That is a one-table-read
free-energy estimate that could drive the FieldModulation knobs (style
switching on surprise) without any model at all.

**The friction:** a merge mints a symbol — and codebooks mint **with the class
in the registry**, trained once, amortized (OGAR D-AMORT). Changing the merge
table changes the basis; D-RCC-2's own note applies verbatim: codebook-SET
identity is part of schema resolution and must be version-gated, or stored
codes silently change meaning.

**Falsifier:** do pair-frequency-minted centroids beat flat k-means-256 on
held-out reconstruction ρ (the hierarchical-4⁴-vs-flat test that is already
named in OGAR's tier-interpretation section)? If flat wins, the BPE reading is
decoration.

## T4 — Markov context building + RelativPronomen anaphora + the RL option

**Ships [G]:** context is a **version-range read over the temporal sorted
stream** (`E-MARKOV-TEMPORAL-STREAM-1` — any width, per-reader rung,
replayable; the ±5 window generalized). Anaphora: `Locus::Antecedent` is a
signed ±8 nibble; the binder is escalate-never-clamp
(`probe_antecedent_binder.rs`); beyond ±8 is a **basin edge** — identity
discriminates, not position. Measured resolution on real text: agreement-aware
0.727 vs noun-only 0.455 vs agreement-blind recency 0.273. German supervision
exists: `de-bundle/relative_pronoun.tsv` (+ satzklammer, valency).

**The idea [H]:** RL over the *resolver policy*, not the answers. Action space =
the grammar-heuristic cue weights already shaped like `ScanParams`' 7 knobs;
thinking styles = the exploration policy over heuristics; credit = downstream
coherence read from the witness lane (fewer Contradiction loci, higher Quorum
in the following window). Policy parameters live in the Learned lane.

**The friction:** (a) reward must be a **read** and the update single-writer —
same discipline as T1; (b) don't spend RL where supervision is sitting on disk:
the German rails are labeled data, and RL is for the **residual** where the
rails are silent; (c) resolution stays lane-local — the ±8 pointer never
crosses a version/lane boundary (the versification lesson, one level down).

**Falsifier:** pre-register 0.727. On the measured German relative-clause set,
does the learned mixture beat it out-of-sample? A learned policy that matches
the static one is a cost, not a result.

## T5 — Epistemic KG vs causality graph vs causality LEARNING (and the pothole-fed BNN)

**Ships [G]:** three distinct objects that must not blur:
`EpisodicBasin` (tenant 15) = **references** into the stream (the epistemic
"who said what, when"); `CausalWitness` loci = epistemic **pointers** within a
window; `CausalEdge64` = the causal **edges** themselves (three surviving
roles). The rung ladder is operator-ruled (`persona-vs-rung-ladder.md`,
MANDATORY read: rungs 0–1 observation, 2 = the 144 verb atoms, 3 = the 34 NARS
tactic recipes = THE runbooks, 4 = StyleFamily macros + the triangle).

**The idea [H]:** separate three activities the word "learning" conflates —
(1) the graph OF causality (structure that exists), (2) causality learning as
**truth-value revision on existing edges** (NARS revise — shipped), (3)
**structure learning** (minting new edges — the expensive, dangerous one). Then:
**potholes are the training set.** Every escalation / Unbound grounding /
free-energy spike is a labeled miss AT A RUNG. A BNN learns pothole → tactic-
recipe dispatch (rung 3) — higher-order thinking as *learned escalation
routing*, never learned answers.

**The friction:** reasoning lives in lance-graph, not the inbound leg
(`E-DEEPNSM-V2-IS-INBOUND-LEG-…`); and the alpha-channel doctrine holds — the
overlay is **a pruner, never a proof**. The BNN proposes; the witness disposes.

**Falsifier:** at fixed accuracy, does learned dispatch reduce the Escalate
rate vs the static style table? Escalations are counted today, so the baseline
is free.

## T6 — ogar-loco: deterministic low-code + learning skills, without losing determinism

**Ships [G]:** compiled thinking templates (elixir-template × StepMask, Rig as
the oracle), with the compile-down direction ruled: **LLM runs compile INTO
templates; templates never degrade into prompts.** Promotion is held-out-gated.

**The idea [H]:** loco stays deterministic; learning enters at exactly two
doors — (a) parameter tables in the Learned lane, (b) new templates compiled
from traces — both behind the existing held-out promotion gate. The triangle
then gives every loco step three execution modes: frozen template / learned
parametrization / explore variant. A per-step policy, fully replayable, because
every stochastic element is address-derived (T2's deterministic-BNN move).

**The friction:** a learned path must never bypass the template contract;
template-equivalence replay grading IS the gate, not a review step.

## T7 — Polars/pandas-shaped KG tagging without the sidecar stack

**Ships [G]:** the key IS the index — *"the key prerenders nodes with zero
value decode"*; canon-high classid behaves as a clustered index (range
predicates over the decoded u32); Lance/DataFusion do the scans; palette
distance is one table read.

**The idea [H]:** tagging = writing `(classid, rails)` onto rows. Faceted
search = prefix scan + rail predicates; ranking = L4 palette distance. That
replaces the pandas+elasticsearch (or polars+tantivy) sidecar a
cognet/medcog-shaped tagger would otherwise drag in: the address does the
indexing, the codebook does the ranking, and there is no second store to drift.
Clinical specifics belong in the private consumer repo, not here — this thread
stays generic by design.

**Falsifier:** one tagged corpus, side-by-side recall + latency vs a tantivy
baseline. If the sidecar wins on both, the thread dies honestly.

---

## The cross-cutting honesty box (applies to every thread above)

- Every learning **write** names its single writer and its lane; every
  **reward** is a read (zero-copy law).
- Every stochastic element is **address-derived deterministic** (D-QUANTGATE);
  replay must hold or the idea is out.
- Every new codebook / merge table is **version-gated** — changing the basis
  under stored codes is silent corruption.
- Nothing touches CE64 bit fields without the operator re-opening M20.
- "Superposition" of the triangle is a **mixture**, never a bundle
  (I-VSA-IDENTITIES).
- An idea graduates from this file only through a probe with pre-registered
  gates — and its baseline number is written down BEFORE the probe runs.
