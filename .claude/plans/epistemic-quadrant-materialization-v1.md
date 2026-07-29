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

**Age: 102 days (3.4 months). Imported from RedisGraph. Predates every substrate
primitive it would need.**

> **⊘ CORRECTED — my first answer was a shallow-clone artifact.** I reported "27
> days, one commit, landed on the V3 boundary." All three were false. **The local
> clone is SHALLOW, grafted exactly at `28f17cd` (PR #629, 2026-07-02)** —
> `git rev-list --count --before=2026-07-02 HEAD` returns **0**. Every file in
> the repo therefore "first appears" on 2026-07-02, because that is the graft
> boundary, not a creation date. **`git log` dates are unusable in this container
> for anything before #629.** Use the GitHub API
> (`/commits?path=<file>`) instead. The operator's estimate ("1-4 months",
> "200 PRs ago", "predates the standing wave and SoA") was correct on all three;
> mine was an artifact reported as a measurement.

True history from the API — **4 commits total, and only one is substantive**:

| date | commit | what |
|---|---|---|
| **2026-04-18** | `cf0b298` | *"import holograph crate from RedisGraph as local crate"* — **the origin** |
| 2026-04-26 | `e270bba`, `05e8386` | clippy cleanups |
| 2026-05-13 | `f222c6e` | rustfmt 1.95.0 workspace sweep |

Against the primitives it would have to integrate with:

| | first commit | holograph is older by |
|---|---|---|
| `holograph/rl_ops.rs` | **2026-04-18** | — |
| `soa_envelope.rs` (the LE envelope) | 2026-06-06 | **49 days** |
| `canonical_node.rs` (`NodeRow`, the SoA) | 2026-06-13 | **56 days** |
| `witness_fabric.rs` (the standing wave) | 2026-07-21 | **94 days** |

So the operator's presumption is **confirmed, not merely plausible**: this code
predates the SoA by ~8 weeks and the standing wave by ~13 weeks, and was written
for a different system entirely (RedisGraph). It is not un-migrated V3 code — it
is **pre-substrate foreign code**. 233 PRs back (#629 vs #862). That strengthens
the verdict below from "needs adapting" to "cannot be adapted": the eight
violations in the table are not oversights, they are what you get when code is
written before the things it now has to live with existed.

**The real lesson — mechanical hygiene forges a false currency signal.** Three of
the four commits are clippy and rustfmt sweeps. The crate is warning-clean,
rustfmt-current, and appears in recent-looking commits, so every automated signal
says *maintained* — while **nothing has ever reviewed its architecture.** A crate
can be lint-green and three substrate generations stale at the same time, and the
lint-green is what makes the staleness invisible. (My retracted "migration PRs
let un-migrated code in" was a worse lesson drawn from a false premise.)

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

## 4b. Does the triad map to the rung ladder / NARS recipes / frozen-learned-discover?

> Operator conjecture, 2026-07-29: *"STDP + Hebbian + homeostatic might even map
> with the rung ladder, the 34+ NARS recipes and the frozen / learned / discover
> triangle."*

> **⊘ NOT A CONJECTURE — IT IS SHIPPED. Third rediscovery this session.** I
> adjudicated the mapping as "promising" before reading `ValueTenant`. The
> triangle is **four existing lanes**:
>
> | tenant | doc-comment (verbatim) | triad member |
> |---|---|---|
> | `Plasticity = 7` | *"**Hebbian** plasticity counter + last-active stamp"* | **Hebbian**, by name |
> | `FrozenStyle = 10` | *"Autopoiesis-triangle FROZEN lane … the row's CHECKPOINT policy"* | **homeostatic** (set point) |
> | `LearnedStyle = 11` | *"the **NARS-revision-updated** policy … `learned[f]` promotes to `frozen[f]` only after winning the held-out arm"* | consolidation |
> | `ExploreStyle = 12` | *"Autopoiesis-triangle EXPLORE lane … deterministic address-derived jitter (D-QUANTGATE coprime walk — never RNG, replay holds)"* | **discover** |
>
> Three consequences that supersede the analysis below:
>
> 1. **Leg 2's best link is the shipped mechanism, not a proposal.** I argued
>    "revision ↔ homeostatic" from first principles; `LearnedStyle`'s doc already
>    says it is *NARS-revision-updated*, and the `learned → frozen` promotion
>    gate is consolidation-after-validation — the biological LTP → consolidation
>    pipeline, already carved.
> 2. **Leg 3's matrix claim is CONFIRMED by the shape.** The three lanes hold the
>    *same* 12 `StyleFamily` slots. So the triangle literally is
>    **12 content ordinals × 3 plasticity states** — a matrix, exactly as argued,
>    and shipped that way. Do not collapse it to a 3-way mapping.
> 3. **`PlasticityEngine` is not merely wrong-chassis — it is REDUNDANT.** Tenant
>    7 is a Hebbian counter. The lane already exists; holograph would add a
>    second, HashMap-shaped copy of it. That is the §4a verdict's row 1
>    (SECOND-PROJECTION) reaching its strongest form: not "connectivity has a
>    home", but *this exact mechanism has a lane*.
>
> What remains genuinely open is **not** the mapping but the *motion*: **nothing
> in the substrate's production traversal** drives `explore → learned → frozen`.
> The lanes are carved; the promotion gate is specified; the triad math is
> unwired *on any live path*. That — not a new engine — is the work.
>
> (⊘ Qualified 2026-07-29, CodeRabbit: "nothing drives" was written before
> §4d-RESULTS existed and is now falsified by this plan's own later section.
> `probe_sudoku_teacher.rs` G5 **does** drive the triangle end to end, in both
> directions — promote *and* refuse, with write-isolation asserted on each. The
> honest statement is that the driver exists only in **probe** code, never in a
> production traversal. Same for §6's sequencing.)

**Leg 1 — frozen / learned / discover: STRONG (3:3, mechanism-level).**

| mechanism | what it does | triangle vertex |
|---|---|---|
| **STDP** | A-before-B → strengthen A→B. **Directional**, creates *new* causal edges from timing | **discover** |
| **Hebbian** | co-activation, **symmetric**. Consolidates what already co-occurs | **learned** |
| **homeostatic** | scales toward a target rate; **bounds** accumulation | **frozen** |

Receipt, not analogy: `PlasticityState::ALL_FROZEN`'s own doc-comment reads
*"Established clinical pattern."* The frozen vertex is already named that way in
shipped code. And the triad's shapes are genuinely distinct (directional /
symmetric / normalizing), so this is a mechanism correspondence, not a rhyme.

**Leg 2 — NARS: promising, with one strong link.**

- **revision ↔ homeostatic.** The strongest of all the pairings. NARS revision
  merges evidence for the same statement under the **φ-1 confidence ceiling**
  (CLAUDE.md: *"permanent humility"*). A bounded-accumulation rule with a set
  point **is** homeostatic scaling — same mechanism, different vocabulary. This
  one is worth promoting on its own.
- **abduction / induction ↔ STDP.** Both are directional and *edge-creating*:
  they hypothesize a link that was not there. STDP is the only member of the
  triad that creates directed structure.
- **deduction ↔ Hebbian.** Composition over edges that already exist; adds no
  new connectivity, strengthens established paths.

**Leg 3 — the rung ladder: NOT a correspondence. It is a MATRIX.** ⚠

This is the category error the conjecture invites, and it should be refused
explicitly:

> **A rung says what content IS. A plasticity mode says how it CHANGES.**
> Different axes.

A rung-2 verb atom can be *frozen*; a rung-3 NARS recipe can be under
*discovery*; a rung-4 StyleFamily macro can be *learned*. So "STDP = rung 3"
would flatten two orthogonal dimensions into one — the exact dilution the
`dilution-collapse-sentinel` exists to catch. The coherent statement is:

> **every rung's content carries a plasticity state** → a `rung × {discover,
> learned, frozen}` matrix, not a 3-way mapping.

Structural evidence that the substrate already agrees: `PlasticityState` is
**3-bit per-plane (S / P / O)**, i.e. plasticity is already an *axis-local*
state, not a global mode. A single edge can be hot on S and frozen on P. That is
matrix behaviour, and it is shipped.

**Status: CONJECTURE.** Before any of this is built on, it needs the
mechanism-vs-rhyme test (`cross-domain-synthesizer`: does it share a MECHANISM,
[H]+, or is it decorative rhyme, [S]?). Leg 1 and the revision↔homeostatic link
look like mechanism; leg 3 is refused as stated. **The 34 NARS recipes are rung
3 (`persona-vs-rung-ladder.md`) — do not re-label them by plasticity mode; label
them by which mode *moves* them, which is a per-recipe property to measure, not
assign.**

## 4c. The reflexivity theorem + the fusion rules (operator thought experiment, 2026-07-29)

> Operator probe: the substrate as a gigantic Sudoku with counterfactuals and
> signed qualia; Gadamer's fusion prevents loop-duplication yet *creates* a
> cognitive materialization — the Adam arc (Gen 3:7 → Aufklärung → Nietzsche →
> modernism). Yield: one theorem, two rules, one doctrine repair (the repair is
> landed in `zero-copy-lens-law.md` ⊘ REFINED — the recompute falsifier was
> type-blind and would have deleted `Locus::Quorum`).

**The reflexivity theorem (structural — checkable against the shipped value
law).** Witness loci are displacements, `target = cur + off`, and the contract
fixes **`0 = unbound`**. Pointing at yourself is therefore *unrepresentable* —
offset zero already means nothing-bound (consistent with `CONTENT_LOCI`'s "no
self-reference"). Consequence: **a reflexive realization cannot land as
routing** — the lane simply has no encoding for it.

> **⊘ CORRECTED 2026-07-29 (CodeRabbit — and the correction is right).** The
> original text continued *"it is FORCED to precipitate as a new grey-matter
> row … the only legal move the LE contract leaves."* **That is an overclaim.**
> What the contract forbids is the *routing* encoding; it does not select among
> the remaining options. At least four outcomes are contract-compliant:
>
> | outcome | contract-compliant? | what it means |
> |---|---|---|
> | mint a displaced row binding backward | yes | the reflexive event is remembered |
> | **reject** the realization | yes | refuse to represent it at all |
> | **defer** it (escalate, like an out-of-±8 antecedent) | yes | W6's shipped precedent |
> | **leave it unbound** (nibble stays 0) | yes | the zero-fallback default |
>
> So minting is a **policy choice**, and the interesting fact is that the
> substrate ALREADY implements a different one: W6's binder **escalates** an
> unrepresentable displacement rather than minting anything. Reflexivity
> (`d == 0`) takes the same path. Regrade: the *unrepresentability* is [G]
> (follows from the value law); *which outcome follows* is **policy, currently
> "escalate", and unprobed as a choice.** The Gadamer framing motivates minting
> but does not derive it.
>
> **Probe required before minting is adopted anywhere: PROBE-REFLEXIVE-POLICY** —
> feed a reflexive event, assert the chosen outcome is the one configured, and
> assert the other three are reachable by configuration (otherwise "policy" is
> decoration and it was forced after all).

W6/W7 grounding survives the regrade and is unaffected: Gen 3:7 as an election
event provable by BINDING at non-zero displacement, never a content edit on the
existing row — that claim rests on unrepresentability, not on minting.

**The fork-return rule (counterfactual Sudoku).** Bifurcation (assume, propagate,
contradict, eliminate — Pearl rung 3 inside constraint propagation): **only the
contradiction returns from a counterfactual fork.** Positive assignments of the
hypothetical world never merge back — that would duplicate a counterfactual into
the actual (the Gadamer anti-duplication rule, storage edition). Precedent
already shipped: `InferenceType::Counterfactual` carries mantissa **−6** — the
negative sign for not-this-world is in the encoding.

**The cross-term rule (the mathematical carve of Horizontverschmelzung).** Of
the interference expansion |ψ₁+ψ₂|² = |ψ₁|² + |ψ₂|² + 2Re(ψ₁\*ψ₂): the diagonal
terms are the individual stances' projections, already in the lanes — storing
one is a second projection. **Only the cross-term exists in no lane** and is
eligible for materialization (judged by rung). Irony/sarcasm = the destructive
case: both slits elected, one sign-inverted; the literal slit must stay live or
the irony collapses to plain statement. Sign is already native at three layers
(i4 loci, ±1 bipolar phase, signed inference mantissa) — negative qualia is an
*election* of the existing sign bit, not new storage.

**The quadrants in Sudoku's own vocabulary** (structurally exact, not
decorative): solved cell = known known; pencil marks = known unknown; **hidden
single = unknown known** (determined by the group, invisible from the cell);
a variant-rule cage you can't see = unknown unknown, revealed only by a solve
that contradicts. Sudoku's *naked* vs *hidden* single is literally "where a
determined value is visible from" — the stance/election distinction. Gen 3:7 is
the hidden single "naked" becoming naked.

**The arc as read-path degrees of freedom** (each historical stage names a
mechanism): direct read (one hardwired ClassView) → **reflexive displacement**
(Adam — the forced mint above) → **elected reading** (Aufklärung: the mask as
the reader's own act) → **signed reading** (Nietzsche: inversion, irony) →
**cross-term materialization** (Gadamer: the mint operator for stance 5+). The
B6 panel is the sediment of this arc; the panel's collective-failure mode
(unknown unknown outside the *union* of the four horizons) is how the arc
continues — a failed probe marks the boundary, fusion across it mints the next
stance. Grade: the theorem is [G]-shaped (follows from the shipped value law);
the fork-return and cross-term rules are [H] (mechanism-consistent, probe
pending); the arc mapping is framing, not a claim.

## 4d. PROBE-SUDOKU-TEACHER — literal Sudoku as the first teacher (operator-directed, 2026-07-29)

> Operator: *"You could even use literal Sudoku and then prepare the first baby
> step to stockfish-rs as a teacher"* + *"sudoku might need lewensteyn."*
> This is the falsifier program for §§4a–4c made LITERAL: exact ground truth,
> free oracle, every quadrant countable, and the triangle's missing MOTION
> (explore → learned → frozen) driven for the first time.

### Why Sudoku is the right first teacher

An 81-cell grid with a known solution is an oracle that costs nothing and never
grades wrong. 81 rows × 512 B ≈ 41 KB — trivially resident. Every mechanism
claim in this plan becomes a checkable assertion against it.

### The two metrics (already adjudicated — do not re-derive)

Per PROBE-BABEL-STANCES slice 2 (`probe_babel_stances.rs:163-180`): *"sequence
error → edit distance / CER; fingerprint-space search and candidate pruning →
Hamming/L1."* Applied here:

- **Grid state vs solution = HAMMING.** Cells never shift position; no indel
  exists. Monotone non-increasing per pass is gate G6.
- **Solve PATH vs teacher path = LEVENSHTEIN.** The election sequence is a
  sequence; two solvers reach the same grid by different orders, and the
  *policy* divergence the triangle promotes on is alignment-based edit distance
  over `(cell, digit)` election tokens. This is the operator's "sudoku might
  need lewensteyn" — and it is the metric that carries unchanged to chess
  (student PV vs teacher PV).

### The mapping (box-major — verified arithmetic, not the pretty version)

`pos = box*9 + cell_in_box`, `box = (r/3)*3 + c/3`, `cell_in_box = (r%3)*3 + c%3`.

- **Box peers: ALL within ±8.** Backward displacements −1..−8 for every cell
  (cell_in_box k has k ≤ 8 predecessors) — every one representable i4. The
  witness lane carries **backward box-peer displacements only**.
- **Cross-band column peers: |Δpos| ∈ [21,60] — provably ALWAYS out of window.**
- **Cross-stack row peers: |Δpos| ∈ [7,20] — MIXED** (some incidentally
  in-window). Design rule therefore: **lane = box group only**; row/col groups
  resolved by lens sweep (predicate over positions, zero-copy). Honest, not
  convenient: the dichotomy "box in-window / col out-of-window" is provable,
  the row case is not, so rows ride the sweep with columns.
- **This makes the horizon claim testable (G1):** a column-forced hidden single
  MUST be unfindable from lane-resident witnesses alone (the sweep path fires);
  a box-forced single IS findable from the lane alone (the sweep stays silent).
  Both halves — proving lane and sweep are each load-bearing.

### Quadrants, literally (S-gates)

- Naked single (cell-visible) vs **hidden single = the countable unknown known**
  (group-visible only). G2: a puzzle seeded with a hidden-single-that-is-not-
  naked finds it; an all-naked puzzle reports zero.
- Per-pass census via `ndarray::hpc::entropy_ladder::Quadrant::classify`
  (entropy = normalized candidate-set size; energy = solved-peer fraction).
  G4: census migrates toward Wisdom across passes; a fork-refusing policy on a
  bifurcation-required puzzle does NOT fully migrate (the silent half).
- **Fork-return (G3):** bifurcation = clone the slab as an explicit
  counterfactual WORLD (write-divergent scenario fork — NOT a gather; a fork
  diverges by writing, a gather duplicates for reading), propagate to
  contradiction, and **only the elimination returns**. Assert the main slab is
  byte-identical outside sanctioned writes and the fork's positive assignments
  never appear in it (§4c fork-return rule, exercised literally).

### Teacher + the first triangle MOTION (G5)

Deterministic puzzle construction (base pattern `(i*3 + i/3 + j) % 9 + 1`, fixed
permutation tables, fixed blanking masks — **no RNG**, D-QUANTGATE replay).
Two policies as style atoms: A = elections-first, B = bifurcate-early. Grade =
(solved?, cost, path-Levenshtein vs teacher). Train on K puzzles → write winner
to `LearnedStyle` slot on a designated policy row → **promote to `FrozenStyle`
ONLY after winning the held-out arm** (the lane's own doc-comment contract,
driven for the first time). Both halves: a promote case AND a refuse case
(train favors B, held-out favors A → promotion refused). Write-isolation
asserted on the triangle lanes.

Content placement: digit + given-flag as an EXPERIMENTAL reading of the
`EntityType` u16 lane (a cell-class discriminator: {empty, given, derived} ×
digit), documented per the Tekamolo honest-catalogue idiom. **No new tenant, no
layout change, offsets derived only.** Candidate sets = local pure compute
(warden non-trigger), never stored.

### §7-T — the stockfish-rs baby step (prepare, not build)

Teacher ladder: **T0 Sudoku** (binary outcome, free, no adversary) → **T1
stockfish-rs** (graded centipawns, adversarial, deep counterfactuals). The baby
step is making the promotion loop **teacher-agnostic in record shape**:
`(position_key, elections[], outcome_grade, teacher_path)` — Sudoku proves the
loop; chess swaps the oracle.

- **GPL fence (iron):** stockfish-rs is GPL-3.0 and NEVER becomes a dependency
  of lance-graph. The seam is data-only — stockfish-rs emits labeled records
  (FEN, legal moves, evals, PV) as artifacts; lance-graph consumes records.
  This mirrors stockfish-rs's own iron rule 2 one level up: *Stockfish C++ is
  the oracle only, never linked* → stockfish-rs is the oracle only, never
  linked.
- What chess adds that Sudoku cannot: graded outcomes (centipawns → NARS
  frequency, not booleans); adversarial counterfactuals (opponent reply = a
  fork you don't control); NNUE incrementality (E-CHESS #539) as the
  teacher-side zero-copy rhyme; move space (from,to) = 64×64 = **4096** — the
  node bit-width anchor already pinned in stockfish-rs.
- The path metric carries verbatim: student PV vs teacher PV = Levenshtein
  over move sequences (same Babel adjudication).
- No stockfish-rs commits in this arc; its next leaf stays in its own plan.

Probe artifact: `crates/lance-graph-planner/examples/probe_sudoku_teacher.rs`
(planner has both deps: contract for NodeRow/WitnessLens, ndarray for
Quadrant; precedent: probe_babel_stances, probe_eyes_opened).

## 4d-RESULTS (2026-07-29) — G1–G6 green, and what they do NOT show

`examples/probe_sudoku_teacher.rs`, 1094 LOC. Orchestrator-verified (not
report-trusted): `cargo fmt` needed a central fix, `clippy -p
lance-graph-planner --all-targets -- -D warnings` clean, probe run
**ALL GATES GREEN** (G1–G6).

Confirmed working: the lane/sweep horizon split (box-forced → sweep silent;
column-forced → sweep fires and *changes* the answer, lane `[7,8,9]` → `[9]`);
fork-return byte-isolation (`only_target_changed=true`,
`wrong_guess_absent=true`, `exactly_one_branch_failed=true`); triangle promotion
AND refusal with write-isolation on both (`promote learned=0xbb frozen=0xbb`,
`refuse frozen=0x00`); Hamming monotone with a genuine strict decrease.

**Three honest limitations — recorded because "ALL GATES GREEN" must not be read
as "we built a Sudoku reasoner":**

1. **This is a mechanism demonstrator on engineered fixtures, not a solver.**
   The "hard" puzzles are near-empty (Hamming series start at 81 = every cell
   unsolved) and end at 80 — *one cell resolved*. Legitimate for isolating a
   mechanism; not evidence of solving ability, and no such claim is made.
2. **G4 does not test the contrast it was designed for.** The intent was
   "bifurcation enables migration where refusal does not." The measured census
   is **identical for both** (`staunen 63, wisdom 18`); only the *easy-vs-hard*
   contrast is asserted, and the bifurcate census is printed "for comparison"
   rather than asserted. So G4 currently proves migration happens on an easy
   puzzle and not on a hard one — which is weaker than its stated claim. **The
   gate passes while under-testing; fixing it needs a puzzle where bifurcation
   is genuinely load-bearing.** (Exactly the vacuity class the P0 rule targets —
   caught here by reading the numbers, not the verdict.)
3. **Hidden singles are not in the solve loop.** G2 proves the detector fires
   and stays silent, but `run_policy` never elects via hidden singles — the
   worker found that threading them in subsumed the engineered 2-candidate
   ambiguity and erased the G5 policy distinction. So the **unknown-known
   mechanism is demonstrated in isolation, not exercised by the reasoner.**
   That is the quadrant this probe most wanted to prove, so the gap is
   material.

Also note G5's margin is one edit operation (`path_lev` 1 vs 2) at equal cost —
honest (that IS the metric) but thin; a wider-margin fixture would be sturdier.

**Follow-ups, in order:** (a) G7 ambiguity gate (§4e); (b) re-shape G4 so
bifurcate-vs-refuse is the asserted contrast; (c) thread hidden singles into
`run_policy` with a policy distinction that survives it.

## 4e. Comparison baseline: `zackthoutt/sudoku-ai` — search vs REASONING

> Operator, 2026-07-29: *"for comparison — needs reimagining using logical
> reasoning."* Verified by fetch, not assumed (I had expected a CNN and was
> wrong): it is a **constraint-satisfaction solver that explicitly rejects ML**
> — *"an AI that requires zero training data"*, two Python files
> (`sudoku.py` / `solver.py`), README claims **100 % accuracy, no failure cases,
> no metrics**, no algorithm detail (backtracking vs propagation unstated).

That makes it the RIGHT baseline: it already skips the neural detour, so the
remaining delta is purely **search vs reasoning** — no ML strawman in between.

| | constraint SEARCH (baseline) | logical REASONING (this substrate) |
|---|---|---|
| candidate state | binary in/out | NARS `(frequency, confidence)` — graded |
| a failed branch | **backtrack: undone and discarded** | **fork-return: only the contradiction comes back, as a permanent elimination** (§4c) |
| output | a filled grid | grid **+ the election path + which tactic fired at what confidence** |
| self-knowledge | none | quadrant census, DK position, "where am I uncertain" (D-SRS-3) |
| under-determined input | returns the first solution | must report **ambiguity** |
| competence claim | "100 %", unfalsifiable | must state what input would make it fail |

**The load-bearing difference is #2, and it is not a refinement — it inverts the
sign of a failed branch.** Backtracking treats a contradiction as *waste* (undo,
retry elsewhere). The fork-return rule treats it as *the only thing worth
keeping* — the elimination is a permanent gain, the positive assignments are
discarded. Same operation, opposite ledger. A search solver ends a puzzle
knowing exactly as much as it started; a reasoner ends it having accumulated
eliminations it can carry.

**Consequence for the teacher ladder (§4d / §7-T): this baseline CANNOT be a
teacher.** The promotion record is
`(position_key, elections[], outcome_grade, teacher_path)`, and a backtracking
solver's trace is *search order*, not *reasoning order* — it contains "tried 4,
failed, tried 7" where the teacher path needs "cell forced by box-peer quorum,
confidence c". Path-Levenshtein against a search trace would measure branch
scheduling, not policy. **A solver that cannot explain its path cannot teach a
policy.** That is the operational meaning of "needs reimagining using logical
reasoning": not that constraint solving is wrong (it is correct and fast), but
that *correctness without a justification trace carries no training signal*.

**The README's own best observation is where the two diverge hardest.** It notes
a "critical point": too few givens → multiple valid solutions → the puzzle
becomes *"easier again"*. For search that is true — any valid completion ends the
run sooner. **For a reasoner it is strictly harder**: the uniqueness assumption
has failed, so committing to one completion is an *error*, and the correct output
is "underdetermined, N solutions" — a known unknown, not an answer. Same puzzle,
opposite verdict, and the disagreement is measurable.

### G7 — the ambiguity gate (both halves; a search solver structurally fails it)

- **Can-commit:** on a uniquely-determined puzzle, the reasoner commits and the
  grid matches the solution (Hamming 0).
- **Can-refuse:** on a deliberately under-constrained puzzle (below the critical
  point, ≥2 valid completions), it must **report ambiguity and refuse to
  commit** — asserting it did NOT write a digit into an under-determined cell.
  A baseline that returns a valid completion here scores "success" and is
  precisely wrong.

Anti-vacuity: the under-constrained fixture must have its multiple completions
enumerated and asserted ≥2, so "reported ambiguity" cannot pass by accident on a
puzzle that was actually unique.

**Sequencing note:** G7 is specified here but is NOT in the in-flight §4d build
(gates G1–G6). Adding it mid-run would be a scope change against worker rule 1;
it lands as a follow-up increment once the G1–G6 probe returns green.

## 5. Falsifiers — required before any wiring lands

- **P1 — the hot/frozen bit must be INERTNESS-TESTABLE.** Flipping
  `ALL_HOT`→`ALL_FROZEN` on a path under repeated traversal must change an
  observable; if both produce identical behaviour the bit is decoration. This is
  the `heel_threshold` lesson applied to the substrate's own plasticity flag.
- **P2 — reinforcement must be OBSERVABLE and BOUNDED.** Repeated traversal must
  measurably increase conductance (fire), and a path *not* traversed must not
  drift (stay silent). Both halves. Plus saturation: unbounded reinforcement is
  the runaway that homeostatic plasticity exists to prevent — assert a ceiling.
- **P3 — content must NOT change** *(scoped 2026-07-29, CodeRabbit)*. As first
  written this said "assert the bytes are identical", which is **self-contradictory**:
  if conductance lives in an `EdgeBlock`/`CausalEdge64` field then *those* bytes
  must change — that is the whole update. The assertion only makes sense over
  the **complement**. Two obligations follow, and both are prerequisites, not
  refinements:
  1. **Name the conductance field before any P3 test is written.** It must be a
     *dedicated* field with a stated byte range. Do **NOT** implicitly repurpose
     `PlasticityState` (a 3-bit S/P/O gate — it says *whether* a plane may
     change, not *how much* it conducts), nor NARS `frequency`/`confidence`
     (those are belief, not conductance). Reusing any of them is the
     I-LEGACY-API-FEATURE-GATED anti-pattern: one name, two semantics.
  2. **A layout change requires a field-isolation matrix + an explicit
     serialization version gate** (the same rule that caught five instances in
     Sprint-11). Until the field is named and gated, P3 is unwritable.

  Scoped statement: after a plasticity update, re-read through the lens and
  assert **every byte OUTSIDE the named conductance range is identical**.
  Conductance changed; content did not. If content
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
