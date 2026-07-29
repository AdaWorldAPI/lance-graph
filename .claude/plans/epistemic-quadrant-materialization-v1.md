# Plasticity materialization — grey/white as MECHANISM, over the B6 stance panel

> **Status: PROPOSED.** Operator direction 2026-07-29: *"We can slowly switch to
> the idea of materializing thoughts as a grey matter / white matter explicit
> materialization for testing of known unknowns, unknown unknowns, unknown knowns
> and known knowns."* → clarified: ***"I'm talking about brain plasticity."***
>
> **⊘ CORRECTED, same session, by the operator.** The first draft of this doc
> read "the four quadrants" as an *epistemic taxonomy* and built a storage-
> eligibility rulebook around it. Wrong axis. Grey/white matter is a
> **mechanism** — two distinct plasticity processes — and the quadrants are what
> you *test* by running it, not a filing system. The correction matters because
> the taxonomy reading produced a design about **what may be stored**, while the
> mechanism reading produces a design about **what changes when you think**.
> Those are different systems. Second operator correction, same session: *"only
> 10 h ago you wired Wittgenstein, Kant, Hegel and Nietzsche and you already
> forgot about it"* — the arena this runs over already exists (§3); the first
> draft invented an abstract one.

## 1. The two mechanisms (this is the whole idea)

Neuroscience distinguishes two plasticity processes, and they are not variants of
each other:

| | **grey matter** — synaptic | **white matter** — myelination |
|---|---|---|
| what changes | synapse strength, dendritic spines | myelin thickness on the axon |
| what that does | changes **what** is associated | changes **conduction velocity** — *how fast a path carries* |
| timescale | fast, local, reversible | slower, structural, activity-dependent |
| in this substrate | belief/content revision (NARS truth on the SPO arena) | **edge conductance** — which traversal is cheap |

The load-bearing fact: **myelin stores no content.** It does not copy the signal;
it changes how fast the existing path carries it. A path used repeatedly gets
myelinated → becomes faster → is more likely used again. That is the entire
reinforcement loop, and it is *structural*, not representational.

## 2. Why this dissolves the tension the first draft agonized over

The first draft spent its length deciding *which derived values may be stored*.
Under the mechanism reading that question mostly evaporates:

> **A plasticity update is not a materialization.** It writes a **weight onto an
> edge that already exists** — conductance, not content. There is no second
> projection, because nothing is re-represented.

This is what the operator's earlier ruling meant by *"the entropy work is stored
and can be reused for further reasoning"*: the reuse is not a cached answer, it
is a **cheaper path to re-deriving it**. Consistent with the whole zero-copy arc —
the lens stays the only read, and what changes is the cost gradient over lanes.

Two constraints inherited, non-negotiable:

- **Write-back is gated** (`.claude/rules/borrow-strategy.md`): single writer →
  gated XOR; multiple writers → BUNDLE. **Never raw `=`** onto shared state.
- **Plasticity is a MAGNITUDE, so it bundles.** Per `I-SUBSTRATE-MARKOV`, the
  magnitude side uses `vsa_bundle`, never `MergeMode::Xor` — XOR on magnitudes
  breaks the Chapman-Kolmogorov semigroup. (Sign side may XOR; magnitude may not.
  Two operators, two algebras — the OGAR two-algebra rule.)

## 3. The arena already exists — B6's four stances

`probe_babel_stances.rs:233`: *"the SAME register switches philosopher stance
(B6's Kant / Hegel / Nietzsche / Wittgenstein panel = **four ClassView
projections of one crystal**)."* Four readers, one arena. That is precisely what
plasticity acts on, and it makes each quadrant concrete rather than abstract:

| quadrant | over the stance panel | detectable how |
|---|---|---|
| **known known** | all four stances converge; path myelinated | agreement + high conductance |
| **known unknown** | stances **disagree**, and the divergence is registered | B6 already measures cross-stance divergence |
| **unknown known** | a stance that *would* resolve it exists in the panel but is **never elected** | sweep: projections with no route |
| **unknown unknown** | **no** stance in the panel has a receptor | the panel fails *collectively* — needs a fifth stance |

The third row is the payoff, and it is why the abstract first draft was worse
than useless: "unelected connection" sounded like a metaphor. Here it is a
countable thing — *four ClassViews exist over one register; if only one is ever
elected, the other three are unknown knowns by construction.*

## 4. Measured state of the mechanism (2026-07-29)

| surface | where | state |
|---|---|---|
| `PlasticityState` (3-bit S/P/O per-plane hot/frozen) | `causal-edge/src/plasticity.rs` | wired on `CausalEdge64`; **23 `ALL_FROZEN`, 11 `ALL_HOT`, 3 selective** call sites |
| `PlasticityEngine` (STDP + Hebbian + homeostatic) | `holograph/src/rl_ops.rs:1128` | **ZERO users outside holograph** |
| STDP timing markers | `holograph/src/width_16k/schema.rs:439` | present |
| NARS self-reinforcement LoRA | ndarray `hpc/causal_diff.rs` | present |

**The finding: the field and the engine are in different crates and are not
connected.** `PlasticityState` records *whether* a plane may change;
`PlasticityEngine` knows *how* something changes. Nothing routes one to the
other, so no traversal currently updates any conductance. The state is not
"frozen" (11 sites do go hot) — it is **inert**: hot and frozen currently lead to
the same behaviour, because nothing consumes the bit to modulate a path.

That inertness is the first thing to falsify, per the P0 rule: *a knob that
changes nothing is decoration.*

## 4a. VERDICT on `PlasticityEngine`: reimagine, do not port

**Age: 27 days by calendar, one full generation by substrate.** `git log` shows
exactly ONE commit touching `holograph/src/rl_ops.rs` — `28f17cd`, 2026-07-02
13:47, the squashed `v3-substrate-migration-review` merge (#629) that landed the
whole crate at once. Nothing has touched it since. 1575 LOC, zero external
consumers.

Measure it in commits rather than days and the picture inverts: **825 commits
have landed since**, and 2026-07-02 is *precisely* the V3 flip date — both
`E-V3-FACET-4-PLUS-12` (content-blind 4+12 facet) and the classid canon-high
flip were operator-locked that day. So the crate did not arrive "before V3": it
arrived **on the boundary, inside the V3 migration PR itself, and was never
adapted to what that same PR ratified.** Every invariant it violates was locked
either that day or later — the zero-copy law is from today.

The lesson generalizes past this crate: *a migration PR is the easiest place for
un-migrated code to enter*, because the diff is already enormous and one more
crate reads as part of the sweep.

The mechanism is sound neuroscience (STDP + Hebbian + homeostatic scaling is the
right triad). The **chassis is pre-V3** and violates current invariants:

| # | violation | the rule it breaks |
|---|---|---|
| 1 | `HebbianMatrix { weights: HashMap<(usize,usize), f32> }` — connectivity stored **beside** `EdgeBlock` (12 in-family + 4 out-of-family) and `CausalEdge64` | **SECOND-PROJECTION.** Connectivity already has a home; this is a second reading of it |
| 2 | HashMap lookup per pair — data-dependent addresses | **pointer CHASING**, the exact thing ruled against 2026-07-29; the substrate's edges are computed displacements |
| 3 | `fire(&mut self, cell)` mutates while computing | `.claude/rules/data-flow.md`: *"No `&mut self` during computation. Ever."* |
| 4 | `*scale *= 0.99; *scale += …` | borrow-strategy: gated XOR (single writer) or BUNDLE (multi); **never raw `=`** on shared state |
| 5 | `f32` per connection | LE contract is byte/nibble-quantized (i4 loci, u8 palette); f32-per-pair has no lane shape |
| 6 | private `timestep: u32` | a second clock beside episodic = Lance versions / `last_active_cycle` |
| 7 | no notion of `PlasticityState` | the 3-bit S/P/O hot-frozen gate is **exactly** this engine's missing input — the measured disconnect in §4 |
| 8 | O(n²) pair space | at 32k rows ≈ 10⁹ pairs; the substrate's answer is EdgeBlock's **bounded degree**, not a growing map |

**And a live defect, not merely a mismatch — #9: depression is computed and
discarded.** `StdpRule::weight_change` returns negative `dw` for LTD, but
`PlasticityEngine::fire` applies only `if dw > 0.0`, carrying
`// TODO: directional hebbian (asymmetric matrix) for depression`. So
`a_minus: 0.012, // Slightly stronger depression (homeostasis)` is an **inert
doc-comment claim**: the homeostasis it documents never happens. The engine can
only potentiate — i.e. wiring it as-is produces exactly the unbounded
reinforcement runaway that falsifier P2 exists to catch. This is the P0
falsifiability rule at substrate level: *a doc-comment claim is not a behaviour.*

**Reimagined shape (a lane, not a port):**

- **Conductance lives in the edge** — an `EdgeBlock` slot / `CausalEdge64`
  magnitude, never a side map. Bounded degree by construction.
- **Update is BUNDLE**, because conductance is a magnitude — `I-SUBSTRATE-MARKOV`
  forbids `MergeMode::Xor` on magnitudes (it breaks Chapman-Kolmogorov).
- **Gated by the existing `PlasticityState`** 3-bit S/P/O. That bit is *for*
  this; consuming it is what makes it stop being inert (§4, falsifier P1).
- **Quantized** (u8 / i4), not f32. Same reason every other lane is.
- **Clock = Lance version / `last_active_cycle`**, not a private counter.
- **No `&mut self` compute:** a pass *returns* deltas; write-back is a separate
  gated builder step (engines return results, they do not mutate while computing).
- **LTD must actually land**, or be deleted and the homeostasis claim removed.
  Half a mechanism whose disabled half is documented as active is worse than an
  honest omission.

**Doppelspalt framing (operator, 2026-07-28→29):** the four philosophers are four
**lenses over one crystal**; interference is visible in the projections while the
bytes never move (CLAUDE.md § I-SUBSTRATE-MARKOV consequence). Plasticity's job
is therefore to modulate **which slit is elected** — the conductance of the
projections — never to accumulate a private matrix beside the crystal. That is
the same statement as row 1 of the table above, arrived at from the physics side
rather than the storage side, which is why P4 requires the non-elected stances to
stay *reachable*: closing a slit destroys the interference pattern.

**Verdict: REIMAGINE.** Keep the triad's math (`weight_change`'s exponential
windows are reusable as-is); discard the chassis. Treat `rl_ops.rs` as a
reference implementation to read, not a dependency to wire.

## 5. Falsifiers — required before any wiring lands

- **P1 — the hot/frozen bit must be INERTNESS-TESTABLE.** Flipping
  `ALL_HOT`→`ALL_FROZEN` on a path under repeated traversal must change an
  observable; if both produce identical behaviour the bit is decoration. This is
  the `heel_threshold` lesson applied to the substrate's own plasticity flag.
- **P2 — reinforcement must be OBSERVABLE and BOUNDED.** Repeated traversal must
  measurably increase conductance (fire), and a path *not* traversed must not
  drift (stay silent). Both halves. Plus saturation: unbounded reinforcement is
  the runaway that homeostatic plasticity exists to prevent — assert a ceiling.
- **P3 — content must NOT change.** The zero-copy guard for this whole design:
  after a plasticity update, re-read the arena through the lens and assert the
  **bytes are identical**. Conductance changed; content did not. If content
  moved, this stopped being myelination and became a write.
- **P4 — stance election must actually shift.** Over B6's panel, reinforcing one
  stance must change which ClassView is elected on a later pass — and the other
  three must remain *reachable* (a myelinated favourite that makes the others
  unreachable is pathology, not learning). This is the unknown-known detector's
  own falsifier.

## 6. Sequencing (explicitly NOT now — behind W6 / ZC-2 / W8)

1. **P0-census** — map `PlasticityState` consumers: who reads the bit, and does
   anything modulate on it? Expected answer from §4: nothing. Confirm before
   building. No code.
2. **P1** — make the bit inert-testable (the smallest honest first step).
3. **P2** — connect one traversal to one conductance update, gated + bundled.
4. **P3/P4** — the stance-panel loop over B6's arena.

**Gate:** if P1 shows hot and frozen are indistinguishable, the next deliverable
is the probe that makes them distinguishable — not more design. The first draft
of this doc is the standing example of what happens when design runs ahead of the
mechanism.

## 7. What survives from the first draft

Only the prior-art table, which stands: `Quadrant::classify` in
`ndarray::hpc::entropy_ladder` (Staunen / Confusion / Boredom / Wisdom) is a real
2×2 and does correspond to the four quadrants — but it is a **read** of the
current state, not the mechanism that moves between them. Under the corrected
framing it is the *instrument* (how you observe which quadrant a thought is in),
while plasticity is the *process* (what moves it). Do not mint a second enum;
reuse it as the measurement surface for P2/P4.

Also still valid: `DkPosition`, `curiosity_mul` (D-SCI-4), D-SRS-3/3b basin
uncertainty + held-out gate, D-SRS-4 derivation provenance,
`Locus::Quorum`/`Contradiction`, `WorldModelDto`.
