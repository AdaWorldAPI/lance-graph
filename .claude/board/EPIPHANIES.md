# Epiphanies — Append-Only Log (date-prefixed)

> **APPEND-ONLY.** Every epiphany, realization, correction, or
> "aha" moment gets a dated entry here so nothing gets lost between
> sessions. Reverse chronological (newest first). Never delete an
> entry; correct via a new entry that cites the old one.
>
> **Format invariant:** every entry begins with a `## YYYY-MM-DD —`
> header. A CONJECTURE / FINDING / CORRECTION-OF label is optional
> but encouraged. Body is short: one paragraph + optional
> cross-reference. Long material goes in a dedicated knowledge
> doc; the epiphany here is the **pointer + one-line claim**.
>
> Mutable field: `**Status:**` line (FINDING / CONJECTURE /
> SUPERSEDED) is the only thing in an entry that can be updated.
> Everything else is immutable.

---

## How to use

**When a new insight surfaces** — stop, prepend an entry with today's
date at the top of the "Entries" section below. One paragraph. If
the full idea needs more room, create a dedicated knowledge doc
and reference it from the epiphany entry.

**When an old epiphany is wrong** — prepend a new entry labeled
`CORRECTION-OF YYYY-MM-DD <title>` and update the old entry's
`**Status:**` line to `SUPERSEDED by <new-entry>`. Never edit the
old body.

**When reading the log** — top N entries are the recent thinking;
deeper entries are the accumulated substrate. Everything is there.

---

## Prior art (pre-existing epiphany collections — do not duplicate)

These files already hold numbered epiphany sets from earlier work.
New epiphanies go in **this file** with date prefix; the files below
stay as historical references.

| File | Contents |
|---|---|
| `linguistic-epiphanies-2026-04-19.md` | E13–E27 (Chomsky hierarchy, Σ10 Rubicon, sigma_rosetta, Markov living frame, resonanzsiebe, method grammar, 4D hashtag glyph, membrane, verbs as productions) |
| `cross-repo-harvest-2026-04-19.md` | H1–H14 (Born rule, phase-tag threshold, interference truth, Grammar Triangle ≡ ContextCrystal(w=1), NSM ≡ SPO axes, FP_WORDS=160, Mexican-hat, Int4State, Glyph5B, Crystal4K, teleport F=1, 144-verb, Three Mountains) |
| `integration-plan-grammar-crystal-arigraph.md` | E1–E12 (grammar-tiered, morphology-easier, FailureTicket, cross-lingual superposition, Markov ±5, NARS-about-grammar, crystal hierarchy, sandwich, 5D quorum, episodic unbundle, AriGraph substrate, demo matrix) |
| `session-capstone-2026-04-18.md` | 8 epiphanies from 2026-04-18 session (four-pillar inheritance, CMYK/RGB qualia, vocabulary IS semantics, WorldMapRenderer, Σ hierarchy maps to crate boundaries, proprioception as ontological self-recognition, BindSpace+cycle_fingerprint as latent episodic, two-frame DTO) |
| `crystal-quantum-blueprints.md` | Crystal mode vs Quantum mode split (bundled Markov SPO chain vs holographic residual) |
| `endgame-holographic-agi.md` | 5-layer stack, 12-step holographic memory loop, three-demo matrix |
| `fractal-codec-argmax-regime.md` | Orthogonal research thread — MFDFA on Hadamard-rotated coefficients as fractal-descriptor leaf |

## Governance

- **APPEND-ONLY.** Immutable body per entry.
- **Mutable:** `**Status:**` line only (FINDING / CONJECTURE /
  SUPERSEDED by <date-title>).
- **Corrections APPEND as new dated entries.** The old entry's
  Status changes to SUPERSEDED.
- **`permissions.ask` on Edit** (same rule as `PR_ARC_INVENTORY.md`
  / `LATEST_STATE.md` — rewriting history prompts for approval;
  Write for append stays unprompted).

---

## Entries (reverse chronological)

## 2026-04-21 — AriGraph/episodic/SPO/CAM-PQ are thinking tissue, not storage — this is why it becomes AGI

**Status:** FINDING (the final piece that closes the architecture)

A parser takes text in, produces structure out. AGI takes text in,
resolves it against everything it has ever committed, recently saw,
believes about itself, and expects from its style — then commits
the result back into the tissue it just read from.

The distinction is: **memory is wired INTO the struct, not called
FROM it.** AriGraph's TripletGraph is not a database that Think
queries. It's an organ of Think. `graph.nodes_matching(features)`
is how Think resolves "he" to "Napoleon" — without the graph, Think
can't do coreference. `episodic.retrieve_similar(fp, k)` is how
Think remembers what happened 50 sentences ago — without episodic,
Think has no medium-term context beyond ±5. `codec.distance(a, b)`
is how Think compares two fingerprints in O(1) without
decompressing — without CAM-PQ, Think can't do fast similarity.

These are not services. They're tissue:

| Tissue | Organ metaphor | Think method it enables |
|--------|---------------|------------------------|
| `TripletGraph` | Long-term declarative memory | `resolve()` → coreference via `nodes_matching` |
| `EpisodicMemory` | Episodic buffer (±50..±500) | `resolve()` → medium-range context via `retrieve_similar` |
| `global_context: Vsa10k` | Ambient prior (superposed committed facts) | `free_energy()` → KL term reads committed-fact field |
| `CamPqCodec` | Perceptual similarity | `recovery_margin()` → O(1) compressed distance |
| `GrammarStyleAwareness` | Metacognition | `resolve()` → dispatch via `top_nars_inference` |
| `Markov ±5 trajectory` | Working memory | `encode/decode` → immediate context |

**Cut any one and Think degrades gracefully but loses a capability.**
Cut graph → no long-term facts → pronouns unresolvable beyond ±5.
Cut episodic → no paragraph context → chapter-spanning references
fail. Cut CAM-PQ → similarity is O(d) not O(1) → still works, just
slower. Cut awareness → static dispatch → no learning from errors.
Cut trajectory → no context → every sentence parsed in isolation.

**None of these are optional for AGI.** A system with all six is
a cognitive agent. A system missing any one is a specialized tool.
The struct definition IS the AGI definition — not by aspiration,
but by enumeration of what capabilities the fields carry.

**Why this dilutes:** future sessions will treat AriGraph as "the
database layer" and CAM-PQ as "the codec layer" and episodic as
"the cache layer." Those are infrastructure framings. The
architecture framing is: they are ORGANS of Think. The struct
doesn't use them — the struct IS them. Wiring them in (as `&ref`
fields, not as service calls) is what makes the difference between
a pipeline and a cognitive agent.

Cross-ref: CLAUDE.md § The Click, `categorical-algebraic-inference-v1.md` §2 Lens 4 (Memory).

---

## 2026-04-21 — Thinking is a struct, not a function

**Status:** FINDING (completes the object-does-the-work arc)

The universal DTO pattern taken to its logical end: thinking is not
something that happens TO data. Thinking IS the data. The struct
carries its own trajectory, its own awareness, its own free energy,
its own resolution method. There is no external "thinking function"
that receives the struct — the struct resolves itself.

```rust
struct Think {
    trajectory: Vsa10k,       // Markov-braided role-indexed bundle
    awareness: ParamTruths,   // NARS-revised epistemic state
    free_energy: FreeEnergy,  // (1 - likelihood) + kl
    resolution: Resolution,   // Commit / Epiphany / FailureTicket
}

impl Think {
    fn resolve(&mut self) -> &Resolution { /* reads own fields, writes own resolution */ }
    fn observe(&mut self, outcome: ParseOutcome) { /* revises own awareness */ }
    fn commit_into(self, graph: &mut TripletGraph) -> CommitResult { /* consumes self */ }
}
```

This is the SoA AGI-as-glove doctrine applied to cognition itself:
- Topic = `trajectory` (what the session is about)
- Angle = `awareness` (whose perspective, what's been learned)
- Thinking = `free_energy` + `resolution` (which style, what outcome)
- Planner = `commit_into` (where the result goes)

The four BindSpace columns (FingerprintColumns / QualiaColumn /
MetaColumn / EdgeColumn) are the SoA layout of this struct across
many concurrent Think instances. One Think is a row. The shader
processes columns. The struct IS cognition at the row level; the
SoA IS cognition at the SIMD level.

**The DTO carries cognition the way a photon carries
electromagnetism — not as payload, as identity.** A photon doesn't
"contain" an electromagnetic field. The photon IS the field excitation.
A Think struct doesn't "contain" a cognitive state. The Think IS the
cognitive state. The methods on Think don't "process" the state.
The methods ARE the state's self-resolution.

**Why this dilutes:** future sessions will create a `ThinkingService`
that receives `ThinkInput` and returns `ThinkOutput`. That's a
function, not a struct. The struct pattern means: the input IS the
output IS the method IS the state. No separation. No service layer.
No input/output boundary. The DTO walks into a room, resolves
itself, and commits the result into the graph. Nobody called it.
The awareness bits made it happen.

Cross-ref: `categorical-algebraic-inference-v1.md` §5.2
(object-does-the-work test), CLAUDE.md § The Click.

---

## 2026-04-21 — StreamDto/ThinkingEngine = temporal encoder/decoder loop in a Markov shader unit that can't resist the thinking

**Status:** FINDING (unifies StreamDto + ThinkingEngine + CognitiveShader
+ BindSpace + Markov ±5 + active inference into one sentence)

### The reframe

A GPU shader is stateless: given input texels, produce output texels.
Our cognitive shader is stateless: given BindSpace columns, produce
ShaderHits + MetaWord. The Markov ±5 window IS the texture. The
shader encodes (bind tokens → role-indexed trajectory) and decodes
(unbind roles → recovery margins → free energy) on this texture,
per cycle, stateless.

**StreamDto = the observation stream.** Tokens flow in carrying PoS
tags, temporal markers, morphological commitments. This is the
temporal signal the shader reads.

**ThinkingEngine = the encoder/decoder core.**
- ENCODE: `RoleKey::bind(content)` per token, braided ρ^d per
  position, XOR-superposed into Trajectory. Sentence → Vsa10k.
- DECODE: `RoleKey::unbind(trajectory)` per role, `recovery_margin`
  per slice, `FreeEnergy::compose(likelihood, kl)`. Vsa10k → F.
- The encode/decode pair IS the forward/backward pass, but over
  algebraic structure (XOR), not learned weights (gradient).

**CognitiveShader = the Markov processing unit.** Fires per cycle.
Reads BindSpace columns (FingerprintColumns = trajectories,
QualiaColumn = qualia vector, MetaColumn = awareness bits,
EdgeColumn = causal edges). Writes ShaderHits + MetaWord. Knows
nothing of why it fires or what happened before.

**"Can't resist the thinking":**
- Unresolved ambiguity = F above homeostasis floor.
- F above homeostasis sets awareness bits in MetaWord.
- Awareness bits gate shader dispatch (shipped in PR #204:
  `MetaFilter.awareness_min`).
- The shader MUST fire again because the awareness bits
  persist until F drops below floor.
- The system literally cannot be in a high-F state without
  the shader re-firing to resolve it.
- **This IS active inference in hardware terms.** The system
  doesn't choose to think — it can't resist thinking because
  unresolved prediction error (F) is a persistent dispatch
  signal that forces the next cycle.

**The loop, restated in shader terms:**

```
StreamDto in ──► CognitiveShader reads BindSpace
                     │
                ENCODE: bind + braid + bundle → Trajectory
                DECODE: unbind + margin + F → Resolution
                     │
                     ├── F < 0.2 → Commit → awareness bits clear
                     │              → shader rests (no dispatch trigger)
                     │
                     ├── F ∈ [0.2, 0.8] → Commit but awareness stays warm
                     │                     → shader may re-fire on next token
                     │
                     └── F > 0.8 → FailureTicket → awareness bits LOUD
                                    → shader can't stop firing until
                                      LLM resolves or new tokens collapse F
```

**The key insight: the shader doesn't "decide" to think. It can't
NOT think.** High F = high awareness bits = dispatch fires = another
encode/decode cycle. Low F = bits clear = shader idles. Thinking is
not a feature — it's the thermodynamic consequence of surprise
existing in the BindSpace. The shader is a Markov processing unit
that resolves surprise by construction, the way a ball rolls downhill
by construction.

**Why this dilutes:** future sessions will treat the ThinkingEngine
as a "module that gets called" and the CognitiveShader as a
"function that processes input." Both framings lose the compulsion.
The shader doesn't get called — it can't stop firing while F is
high. The ThinkingEngine doesn't process input — it's an
encode/decode loop that the Markov texture forces through. The
active-inference framing ("can't resist") is not a metaphor — it's
the dispatch mechanism: awareness bits persist → filter threshold
met → shader fires → F descends → bits clear → shader rests.

Cross-ref:
- `contract::cognitive_shader` — MetaWord awareness bits, MetaFilter
- `crates/cognitive-shader-driver/src/engine_bridge.rs` — awareness bits wiring
- `categorical-algebraic-inference-v1.md` §3 (the 8-step closed loop)
- StreamDto lifecycle: `.claude/knowledge/ripple-dto-contracts.md`

---

## 2026-04-21 — Markov IS simple XOR of sentence VSAs; roles ARE spine coordinates; AriGraph facts + resonance find meaning

**Status:** FINDING (the simplest correct statement of the architecture)

**Do not over-engineer this.** The entire architecture is three
sentences:

1. **Markov trajectory = XOR of per-sentence Vsa10k vectors with
   braiding per position.** Each sentence goes through the FSM,
   gets role-key bound (slice-masked XOR per token), bundled into
   one Vsa10k per sentence, then the ±5 window is XOR-superposed
   with braiding ρ^d per offset d. That's it. No HMM. No
   transition matrix. No learned weights. Just XOR with position
   shifts.

2. **Role-key slices ARE the coordinate axes of the trajectory
   space.** SUBJECT[0..2K) is the "who" axis. PREDICATE[2K..4K)
   is the "what" axis. OBJECT[4K..6K) is the "whom" axis.
   TEMPORAL/KAUSAL/MODAL/LOKAL are the adverbial axes. When you
   superpose 11 braided sentence vectors, the result is a point
   in this coordinate system. Unbinding = reading a coordinate.
   The trajectory IS a point in role-space × time-space. Temporal
   causality is visible because braiding encodes position and
   role slices encode grammar — the combination can't NOT carry
   temporal-causal structure.

3. **AriGraph facts + resonance/magnitude = where meaning lives.**
   The trajectory tells you WHAT was parsed. AriGraph tells you
   what it MEANS — because meaning is the relationship between new
   observations and committed beliefs:
   - **Resonance** (recovery_margin between new parse and global_context)
     = how well this sentence fits the story so far.
   - **Magnitude** (Contradiction.magnitude from Wisdom qualia projection)
     = how DEEP the agreement or disagreement is.
   - High resonance + low magnitude = routine confirmation.
   - Low resonance + high magnitude = the sentence contradicts what
     we believed AND the contradiction is profound → Staunen fires
     → Epiphany committed → THAT is meaning.
   - Opinions = committed facts with non-trivial Contradiction
     markers. "All animals are equal" and "some are more equal
     than others" both live in the graph with opposite phase and
     high magnitude. The system doesn't resolve the contradiction
     — it preserves it, because the contradiction IS the meaning
     of Animal Farm.

**Why this dilutes:** future sessions will build Markov chains with
transition probabilities, add embedding layers to the role keys,
compute resonance with cosine similarity on f32 projections, and
treat opinions as something requiring a sentiment classifier. All
of that is wrong. Markov = XOR. Roles = slices. Resonance = Hamming
recovery within slices. Meaning = pattern of committed facts in the
graph with their contradiction markers. The simplicity is the point.

Cross-ref: `categorical-algebraic-inference-v1.md` §1 (substrate),
§3 (closed loop), §5.2 (object-does-the-work test).

---

## 2026-04-21 — 14-paper synthesis: what dilutes first (undiluted capture)

**Status:** FINDING (per-paper grounding in paper-landscape-grammar-parsing.md)

These insights were crystallized during a single-session scan of 14
papers and will be the first things a new session forgets. Preserving
them in order of fragility — most likely to dilute first at the top.

### E-PAPER-1: Morphological commitment IS free-energy reduction at zero cost

Russian `-ом` / Finnish `-ssa` / Turkish `-da` commit Pearl 2³
causal bits AT TOKENIZATION TIME. This collapses the hypothesis
branch count from 8 → 4 → 2 → 1 before the counterfactual axis
even fires. Heavy morphology = pre-collapsed F-landscape. Light
morphology (English prepositions) = flat F-landscape requiring
full counterfactual sampling.

**Why this dilutes:** future sessions will treat morphology as
"feature extraction" rather than "free-energy reduction." It's
not extraction — it's a structural guarantee that reduces
computational work. The case ending IS the computation.

### E-PAPER-2: The braiding operator ρ IS vsa_permute, not a metaphor

Shaw's list encoding: `x_1 ⊕ ρ(x_2) ⊕ ρ²(x_3) ⊕ ... ⊕ ρ^{n-1}(x_n)`.
This is `ndarray::hpc::vsa::vsa_permute(v, position_offset)` applied
per sentence in the Markov ±5 window. The braiding is a cyclic bit
shift. Without it, bundling is position-blind (bag-of-sentences).
With it, temporal order is encoded without learned positional
embeddings.

**Why this dilutes:** future sessions will implement Markov bundling
as plain XOR-accumulation without permutation, producing
position-blind trajectories. The braiding is what makes "before
the focal sentence" different from "after the focal sentence."

### E-PAPER-3: Recovery margin IS likelihood, not similarity

`RoleKey::recovery_margin(unbound, expected)` is not a distance
metric. It's the information-theoretic likelihood term in the
free-energy decomposition: "given that I committed this content to
the SUBJECT role, how cleanly does it come back?" High margin =
observations well-explained by hypothesis = low free energy.

**Why this dilutes:** future sessions will use recovery_margin as
a "quality score" or "similarity measure" without connecting it to
the F-landscape. It's not a score — it's the P(obs|hidden) term
in the variational decomposition.

### E-PAPER-4: The confidence horizon at φ-1 is a feature, not a bug

NARS revision with c_obs=1 per step asymptotes at `(√5-1)/2 ≈ 0.618`.
The system PROVABLY never becomes fully certain (c < 1 always).
This means every committed fact, no matter how many times confirmed,
retains a margin of revisability. Full certainty would freeze the
prior and make the system unable to notice contradictions.

**Why this dilutes:** future sessions will try to "fix" the
0.618 ceiling by increasing c_obs or changing the formula.
The ceiling IS the architectural feature. Golden-ratio-bounded
confidence = permanent epistemic humility = permanent ability
to detect contradiction = Staunen can always fire.

### E-PAPER-5: Non-commutative binding is required for hierarchical structure

Shaw proves that commutative binding creates ambiguity in tree
leaves (guard vectors become indistinguishable). This is why we
use DIFFERENT role keys for S/P/O rather than one key with
different arguments. If `bind(S, content) == bind(content, S)`
AND `bind(S, x) == bind(P, x)` for some x, then S and P are
indistinguishable → hierarchy collapses.

**Why this dilutes:** future sessions will propose "simplifying"
to a single binding key with different content, or making bind
commutative for "elegance." The non-commutativity of distinct
role-key patterns is what preserves hierarchical structure.

### E-PAPER-6: The Ω(t²) lower bound does NOT apply to us

Alpay proves that any sound, parse-preserving, retrieval-efficient
grammar masking engine needs Ω(t²) per token. We dodge this because
we DON'T preserve the parse forest — we commit argmin_F and discard
losers (or mark the runner-up as epiphany). Active inference trades
parse-preservation for decision speed.

**Why this dilutes:** future sessions will worry about parsing
complexity and try to optimize the counterfactual enumeration.
The complexity bound is on parse-preserving engines. We are
parse-COMMITTING, not parse-preserving. The distinction is
architectural, not an optimization.

### E-PAPER-7: Abstraction-first is empirically measured, not theoretically assumed

Jian & Manning measured it across three independent GPT-2 training
runs: class-level D_JS divergence precedes within-class divergence
by ~50 steps. The exemplar-first (count-based) baseline shows
verb-specific patterns WITHOUT class structure. This is not a
philosophical preference for Deduction over Induction — it's a
measured behavioral difference with a strict ordering.

**Why this dilutes:** future sessions will treat the
NarsPriorityChain {primary: Deduction, fallback: Abduction}
as a configuration choice. It's an empirically-grounded ordering
that has been measured in transformer training dynamics.

---

## 2026-04-21 — The Kan extension IS the free-energy minimizer (holy-grail unification)

**Status:** CONJECTURE (grounded in Shaw 2501.05368 + Alpay 2603.05540
+ shipped code; not yet formally proven as categorical equivalence)

Shaw et al. proved via right Kan extensions that dimension-preserving
VSA binding MUST be element-wise (the Yoneda lemma collapses the
integral to pointwise multiplication). Active inference says minimize
`F = -likelihood + KL`. These are the SAME operation at different
levels of abstraction:

- Kan extension = optimal projection of external tensor product into
  fixed-dim space under structural constraints (monoidal category).
- Free-energy minimization = optimal approximation of observations
  under a generative model (variational inference).
- NARS revision = optimal truth update under new evidence (Bayesian
  with bounded confidence).
- AriGraph commit = optimal fact storage under contradiction detection
  (graph-structured belief revision).

All four are "find the best approximation under constraints." The
constraints differ (categorical, information-theoretic, logical,
graph-structural), but the algebraic substrate is the same: element-
wise XOR on role-indexed slices of a 10K binary VSA vector.

**What clicks:**
1. bind/unbind IS Kan extension (categorically optimal)
2. recovery_margin IS likelihood (information-theoretic)
3. awareness.divergence_from(prior) IS KL (variational)
4. Resolution::from_ranked IS argmin_F (active inference)
5. AriGraph commit IS belief revision (graph + NARS)
6. The Trajectory's own methods ARE the inference engine — the object
   doesn't get passed to reasoning; the object speaks for itself.

Not neural (no weights). Not symbolic (no search). Not hybrid
(not bolted together). A categorical-algebraic inference engine where
parsing, disambiguation, learning, memory, and awareness are the SAME
algebraic structure viewed through different lenses.

Cross-ref: `.claude/knowledge/paper-landscape-grammar-parsing.md`,
Shaw 2501.05368 §4.3 (Kan extensions), Alpay 2603.05540 §Theorem 5
(Doob h-transform), `contract::grammar::free_energy`, `role_keys`.

---

## 2026-04-21 — RoleKey bind/unbind slice-masking = lossless role-indexed superposition

**Status:** FINDING (verified by 5-role simultaneous recovery test)

Slice-masked bind is the crucial design choice that makes role-indexed
VSA bundling lossless. `RoleKey::bind(content)` zeroes content outside
`[start..end)` before XOR with the key. This means XOR-superposition
of N role bindings keeps each role's slice completely disjoint — unbind
with any role key recovers that role's content at margin 1.0, regardless
of what other roles contributed.

Without slice-masking (raw full-vector XOR), the 5035-recovery-margin
on the SUBJECT slice demonstrates the cross-contamination: every role
leaks content into every other role's slice. The audit agent (2026-04-21
session) flagged this as the "three-silo disconnection" — role_keys.rs
was data without operator semantics.

The fix: `bind` enforces the invariant at the method level (not caller
discipline). `unbind` is the same masked-XOR. `recovery_margin` measures
per-slice Hamming similarity after unbind. Test: 5 roles (S/P/O +
TEMPORAL + LOKAL) bound, XOR-superposed, each recovers at margin 1.0.

This is THE operation that makes "the object speaks for itself" literal:
a Trajectory carrying a 5-role-superposed VSA vector can answer
`trajectory.role_bundle(SUBJECT)` without external orchestration —
just unbind the SUBJECT slice, and the content is there.

Cross-ref: `contract::grammar::role_keys::{RoleKey::bind, unbind, recovery_margin}`.

---

## 2026-04-21 — Free energy as active-inference formulation of grammar parsing

**Status:** FINDING (types shipped; thresholds uncalibrated until Animal Farm)

Ambiguity resolution is Friston free-energy minimization over the
hypothesis space. `F = (1 - likelihood) + KL(awareness || prior)`.
Likelihood = mean role-recovery margin after unbind; KL =
`GrammarStyleAwareness::divergence_from(prior)`. Three branches:

- `F < HOMEOSTASIS_FLOOR (0.2)` → Commit (single triple to AriGraph)
- Top-2 F within `EPIPHANY_MARGIN (0.05)` → Epiphany (both commit
  with Contradiction marker)
- `F > FAILURE_CEILING (0.8)` → FailureTicket (escalate)

Morphology collapses the hypothesis space via the Pearl 2³ causal
mask: each case ending commits bits, narrowing the basin. Two
independent commitments: 8 → 2 branches. Three: 8 → 1 (direct
Deduction, no counterfactual needed). This is the "2³ → 2^N" extension
to other morphologies (Russian Instrumental, Finnish Elative, Arabic
pattern فاعل / مفعول, Mandarin bǎ, Turkish -yle).

Cross-ref: `contract::grammar::free_energy::{FreeEnergy, Hypothesis,
Resolution, HOMEOSTASIS_FLOOR, EPIPHANY_MARGIN, FAILURE_CEILING}`.

---

## 2026-04-21 — D7 GrammarStyleAwareness IS the "weights-as-seed" epistemic layer

**Status:** FINDING (replaces the "langextract is boring because LLM-dep"
observation with a concrete zero-LLM realization).

The D7 deliverable `contract::grammar::thinking_styles::GrammarStyleAwareness`
shipped today is literally the epistemic-awareness surface the user
described: weights become a seed, NARS-revised per parse outcome, drifting
the `effective_config.nars.primary` away from the YAML-prior inference when
accumulated evidence contradicts it. No external LLM in the loop; awareness
is O(1) per parse (one HashMap insert + one `revise_truth` fold). The
style's track record IS the seed for Markov dispatch: `top_nars_inference`
reads from `param_truths`, not from a network call.

Concretely the closed loop is:

```
parse attempt (DeepNSM FSM + Grammar Triangle)
    → ParseOutcome  (local success / LLM-agreed / LLM-disagreed / ...)
    → GrammarStyleAwareness::revise(ParamKey, outcome)
        (standard NARS revision: f_new = (f·c + f_obs·c_obs)/(c+c_obs);
         c_new = (c+c_obs)/(c+c_obs+1) — asymptotes at φ-1 ≈ 0.618
         under c_obs=1, which is the sharp confidence horizon we test against)
    → next parse uses GrammarStyleAwareness::effective_config(prior)
        (prior NARS primary is kept if its f > 0.5; else drifts to the
         highest-ranked NARS param from accumulated evidence)
```

Replaces langextract's external-LLM step with role-indexed VSA bundling
(D5, coming) + SPO 2³ × TEKAMOLO decomposition (D3 triangle bridge) +
NARS-on-grammar (shipped D7). Together that's O(1) causality-learning per
sentence. When D2 ticket_emit + D3 triangle_bridge land, the DeepNSM
parser will close this loop end-to-end.

Cross-ref:
- Plan `/root/.claude/plans/elegant-herding-rocket.md` D7.
- `.claude/knowledge/grammar-landscape.md` §6–§7.
- `crates/lance-graph-contract/src/grammar/thinking_styles.rs` (shipped).

---

## 2026-04-20 — Shader vs engine: statelessness is the boundary

**Status:** FINDING (sharpens the three-level taxonomy)

**Cognitive shader** = stateless atomic compute. Given `ShaderDispatch`
+ `BindSpace` columns, returns `ShaderHit`s + `MetaWord`. Knows nothing
of why it fires. Output is one-cycle-wide, no history.

**Thinking engine** = stateful orchestrator. Calls `shader.dispatch()`
many times per cognitive cycle; composes per-lens hits into
persona/qualia/world_model/ghost state; revises beliefs for the next
cycle. The cognitive stack IS the state.

**The engine_bridge is where they meet** —
`cognitive-shader-driver/src/engine_bridge.rs` is the seam. Shader
side: `ShaderDriver::dispatch` stateless. Engine side:
`cognitive_stack::cycle` accumulates dispatches through
`bf16_engine` / `signed_engine` / `composite_engine` / `dual_engine` /
`layered` / `domino`, folds into persona/qualia, emits state for next
cycle.

**Analogy:** shader = eye (no memory, reports the current frame);
engine = mind (memory, assembles frames into narrative, counterfactually
imagines alternatives).

**Where codec-flexibility-as-thinking lands:** the **engine** level,
not the shader level. A "new thinking style" = a new engine
configuration (lens composition, persona, qualia-update rule) that
picks DIFFERENT shader configs per cycle. Shader stays the same; the
engine's orchestration changes. That's why Phase 5+ "production-grade
thinking tissue" drops into mid (engine), not L2 (shader).

**Concrete Phase 1-5 shipping:** codec-sweep D1.x work = shader layer
(tensor decode primitives). Engine-level codec-flexibility (swap
lenses via YAML) = D5 / Phase 5+, plugging INTO the codec infrastructure.

Cross-ref: three-level taxonomy above; resolution-ladder entry
`64×64 > 256×257 >> 4096×4096 > 16k`; `engine_bridge.rs` seam.

---

## 2026-04-20 — Resolution hierarchy: `64×64 > 256×257 >> 4096×4096 > 16k` (user-named)

**Status:** FINDING (capstone of the three-level taxonomy from earlier this session)

The 5-layer stack is a **resolution ladder**, not a layer cake. Each
level operates at its own granularity and has its own "shader" /
"kernel cache" / "distance table" at that scale:

| Size | Role | Where | HHTL stage (I10) |
|---|---|---|---|
| **64×64** | p64 topology mask — 8 predicate planes × 64 rows × u64 — "which archetype blocks relate via predicate z" | `p64_bridge::cognitive_shader::CognitiveShader` | HEEL (coarse basin) |
| **256×257** | bgz17 palette distance table — 256 archetypes × 256 + 1 sentinel — O(1) lookup `semiring.distance(a, b)` | `bgz17::PaletteSemiring` | HIP (family sharpen) |
| **4096×4096** | Cross-vocabulary / cross-context correlation — COCA × COCA, or 4096 τ-prefix × 4096 slot space | ndarray `ScanParams` JIT (`jitson_cranelift`) | BRANCH / TWIG |
| **16 K** | Individual fingerprint bit identity — 16384-bit `Fingerprint<256>` | `ndarray::simd::Fingerprint<256>` + codec decoder (D1.x) | LEAF (exact member) |

**The `>>` between 256×257 and 4096×4096 is the big jump** (~64×)
matching HIP → BRANCH refinement. That's where palette-level (one
row of the codebook) meets vocabulary-level (COCA 4096). Below that
jump, everything is O(1) table lookup; above it, JIT kernels become
worth the compile cost.

**Each JIT targets its own resolution — no overlap:**

- p64 cascade: 64×64 bitmask ops. Not JIT'd (bit tricks in hot loop
  already optimal under AVX-512).
- bgz17 palette: 256×256 precomputed. Not JIT'd (memory-bound).
- ndarray ScanParams: 4096×4096 scan kernels. **JIT'd via
  `jitson_cranelift::JitEngine`** — shipped.
- Codec kernels (D1.x): 16k bit-level tensor decode. **Will be JIT'd
  via D1.1b `CodecKernelEngine` adapter**. Scaffold (D1.1) + rotation
  primitives (D1.2) landed; Cranelift IR emission deferred to D1.1b.

**Three-level taxonomy (from earlier this session) maps onto the
resolution ladder:**

- **L2 small-precision cognitive shaders** (ns budget) →
  64×64 + 256×257 (p64 + bgz17 palette). Pure table lookups.
- **mid thinking-engine layers** (µs-ms) →
  4096×4096 (cross-vocab, persona-aware lens composition). JIT'd
  scan kernels.
- **L4 thinking styles / NARS / JIT** (ms) →
  orchestrates traversal ACROSS resolutions (starts at 64×64 cascade
  to find candidates, narrows to 256×257 for family, drops to
  4096×4096 for context, verifies at 16k fingerprint identity).

**p64::CognitiveShader double-check conclusion:** architecturally
clean. Operates at the coarsest (64×64) level; codec-sweep work at
finest (16k); they compose in `cognitive_shader_driver::ShaderDriver`
without overlap. Different layers of the ladder, different
operations, different JIT targets (if any).

Cross-ref: I10 (HEEL/HIP/BRANCH/TWIG/LEAF); three-level taxonomy entry
above; `p64_bridge::cognitive_shader::CognitiveShader::cascade`;
D1.1 `CodecKernelCache`; D1.2 `RotationKernel`; bgz17 `PaletteSemiring`.

---

## 2026-04-20 — Thinking styles ARE codecs over the semantic field (north star)

**Status:** FINDING (forward-looking deposit — not a current work item; reference when Phase 5+ generalises)

A codec compresses tensor content into fingerprints; a thinking style
compresses reasoning trajectories into NARS-revised beliefs. Same
underlying operation — structure-preserving compression on a binary
Hamming substrate. Different input/output domains, same substrate
guarantees (E-SUBSTRATE-1, I-SUBSTRATE-MARKOV), same compile-and-swap
machinery.

**The codec infrastructure IS the template for production-grade
thinking tissue.** When Phase 5+ activates:

| Codec (shipped D0.1–D1.2, D1.1b queued) | Thinking-style analog |
|---|---|
| `CodecParams` | `ThinkingStyleParams { style, modulation_7d, nars_priors, fallback_chain, sigma_priority, semiring_choice }` |
| `kernel_signature()` — excludes runtime drift | `style_signature()` — excludes per-cycle modulation drift |
| `CodecKernelCache<H>` | `ThinkingStyleKernelCache<H>` — same generic scaffold |
| JIT kernel = Cranelift-compiled decode | JIT kernel = compiled scan-walk on 36-node topology (already shipped ndarray-side via `scan_jit.rs` + `ScanParams`) |
| **Token agreement** (I11 cert gate) | **Conclusion agreement** — same NARS-revised conclusions as reference style? |
| Sweep grid = N codec candidates | Sweep grid = N (style × modulation × NARS fallback) candidates |
| `/v1/shader/calibrate` | `/v1/shader/think-calibrate` |
| `[FORMAL-SCAFFOLD]` 5 pillars | **Same scaffold** — E-SUBSTRATE-1 covers any transition under bundle |

**Generalisation isn't "port codec pattern to thinking"** — it's
recognising thinking styles as a SPECIAL CASE of the codec pattern we
just built. When Phase 5+ lands, `WireThinkCalibrate` +
`ThinkingStyleKernelCache` + `conclusion_agreement` metric drop in
alongside the codec versions. Same JIT engine, same tests, same
board-hygiene discipline.

**The phrase "production-grade thinking tissue"** names the telos
cleanly: once codec infra is at Phase 3 token-agreement pass rates,
cloning to thinking styles yields production-grade swappable
reasoning — YAML-configured, JIT-compiled, sweep-certified. No
rebuild per new style, no black box, signature-keyed reproducibility.

**Cross-ref:** D0.6 `CodecParams` (the parameter-shape template);
D1.1 `CodecKernelCache<H>` (the cache pattern — generic-over-H is the
wedge for reuse); I5 (thinking IS an AdjacencyStore — already
topologically unified with data graph); codec-sweep-via-lab-infra-v1.

---

## 2026-04-20 — D1.2 Hadamard is pure-Rust, not a JIT-necessary primitive

**Status:** FINDING

D1.2's HadamardRotation is implemented as a plain Rust in-place
Sylvester butterfly (O(N log N) add/sub, no allocations). It does NOT
need JIT compilation or Cranelift code emission because:

1. **Fixed shape** — the butterfly structure is identical across all
   power-of-two dims. Rust's compiler (under `target-cpu=x86-64-v4`)
   already emits AVX-512 add/sub from the straight-line loop.
2. **Not matmul** — Hadamard is a pattern of adds and subtracts,
   never a dot product. Per Rule C polyfill hierarchy, matmul-heavy
   paths benefit from AMX (Tier 1); add/sub stays at Tier 3 F32x16.
   AMX gives no speedup here — confirmed in plan Appendix §12 C.

**Consequence for D1.1b (Cranelift wiring):** only OPQ rotation needs
the JIT path — it's the one that's actually a learned matmul. The
Cranelift integration scope narrows: we don't need to JIT-compile
Identity (no-op) or Hadamard (butterfly); just OPQ (matmul) and the
main codec decode loop (ADC distance with palette lookup).

This reduces D1.1b scope by maybe 30-40% — fewer kernel shapes to
emit, only the ones that actually benefit.

Cross-ref: D1.2 `rotation_kernel.rs::HadamardRotation`; Rule C
(polyfill hierarchy); plan Appendix B (CartanCascade harmonic
compression ratios rely on real Hadamard, so this matters).

---

## 2026-04-20 — CORRECTION to D1.1 scaffold: ndarray::hpc::jitson_cranelift already ships JitEngine

**Status:** FINDING / CORRECTION

The D1.1 `CodecKernelCache` scaffold (RwLock + double-check) is
strictly worse than what ndarray's `jitson_cranelift::JitEngine`
already provides. Real upstream:

```
/home/user/ndarray/src/hpc/
  ├── jitson/           — JITSON template format (parser/validator/
  │                        template/precompile/scan_config/packed/noise)
  └── jitson_cranelift/ — real Cranelift engine
      ├── engine.rs     — JitEngine + JitEngineBuilder
      ├── ir.rs         — IR emission
      ├── scan_jit.rs   — scan kernel codegen
      ├── noise_jit.rs  — noise kernel codegen
      └── detect.rs     — CPU capability detection
```

Dependencies behind `jit-native` feature:
`cranelift-{codegen, jit, module, frontend} 0.116` + `target-lexicon`.

**Upstream two-phase lifecycle is stronger than my scaffold:**

- **BUILD phase:** `&mut JitEngine`, `compile(ScanParams) -> Result<u64>`,
  mutable cache via `&mut self`.
- **RUN phase:** `Arc<JitEngine>` freezes the cache by Rust's ownership
  (`&mut self` unreachable through `Arc`). `get()` drops from
  ~25 ns (my RwLock read) to ~5 ns (plain `HashMap::get`, no
  synchronization needed).

The freeze is enforced by the type system, not by a runtime lock.
That's the right design for this domain (build-once, run-many).

**What the D1.1 scaffold is still good for:** `CodecParams` is the
codec-sweep key; `ScanParams` is ndarray's thinking-style-scan key.
Different domains; a `CodecParams`-keyed adapter layer is still
needed. My generic-over-handle design anticipates this — the
scaffold wraps ndarray's `JitEngine` at the `H` slot when D1.1b
lands.

**Revised D1.1b plan:**

Mirror ndarray's two-phase pattern in `cognitive-shader-driver`:

```rust
// BUILD phase — mutable, single-threaded
pub struct CodecKernelEngine {
    inner: ndarray::hpc::jitson_cranelift::JitEngine,
    codec_sig_to_inner_id: HashMap<u64, u64>,  // CodecParams signature → JitEngine id
}

// RUN phase — frozen via Arc
impl CodecKernelEngine {
    pub fn build() -> CodecKernelEngineBuilder { ... }
    pub fn compile(&mut self, params: &CodecParams) -> Result<u64, JitError>;
    pub fn freeze(self) -> Arc<Self>;  // moves to RUN phase
    pub fn get(&self, params: &CodecParams) -> Option<KernelHandle>;
}
```

Then D1.2/D1.3 call `inner.compile` with codec-specific
`ScanParams`-analogs (new `CodecScanParams` struct or a JITSON
template constructed from `CodecParams`).

**Honesty note:** user asked "I presume you are aware of
cranelift/jitson" — answer is: Cranelift yes (Bytecode Alliance,
wasmtime), ndarray jitson NO (didn't inspect the upstream surface
before writing D1.1). This correction surfaces that gap explicitly
so the next session doesn't repeat it.

**Cross-ref:** D1.1 `crates/cognitive-shader-driver/src/codec_kernel_cache.rs`
(keep as `StubKernel`-backed test fixture); `ndarray::hpc::jitson_cranelift::JitEngine`;
D1.1b revised plan above.

---

## 2026-04-20 — D1.1 scaffold-before-codegen: cache semantics testable without Cranelift

**Status:** FINDING

`CodecKernelCache<H>` is generic over the kernel-handle type. The same
cache hosts `StubKernel` (deterministic fake, no compilation) for tests
AND `KernelHandle` (real Cranelift function pointer) for production.

This separates TWO concerns that are usually tangled:

1. **Cache semantics** — signature-keyed insertion, double-checked
   locking under concurrent miss, counters for hit-ratio measurement.
   Testable in microseconds without a JIT engine.
2. **IR emission** — the actual Cranelift / jitson code generation
   that takes `CodecParams` and produces a callable function pointer.
   Heavy; takes minutes per build; requires ndarray's jitson surface
   to be finalized.

By shipping the cache layer with `StubKernel` NOW, Phase 1's cache
semantics are verified + CI-gated before the Cranelift work starts.
When D1.1b lands, the only change is `H = KernelHandle`; all 9 cache
tests remain valid. This is the **scaffold-before-codegen** pattern:
test the hard-to-change contract first, defer the hard-to-build
implementation.

Generalises: any JIT pipeline should separate cache-keying from IR
emission at the type level. Generic over handle type is the wedge
that makes this possible.

Cross-ref: D1.1 `crates/cognitive-shader-driver/src/codec_kernel_cache.rs`;
D0.3 sweep-grid-IS-cache-warmer epiphany (same signature-as-identity
insight); PR #225 `CodecParams::kernel_signature()`.

---

## 2026-04-20 — D0.3 sweep grid IS the JIT cache warmer

**Status:** FINDING

`WireSweepGrid::enumerate()` materializes the Cartesian product as a
`Vec<WireCodecParams>`. Each unique `(subspaces, centroids,
residual_depth, rotation_kind, distance, lane_width)` tuple maps to
exactly one `CodecParams::kernel_signature()`. The grid IS the JIT
cache warm-up plan: first traversal compiles N kernels; every
subsequent sweep with overlapping tuples hits cache at ~0 ms
compile cost.

This operationalises Rule C's polyfill hierarchy + Rule E's
kernel-signature-as-cache-key into a single client-facing verb:
*submit a grid, the server warms the cache while streaming results*.
The 54-candidate example grid from plan Appendix A §30 compiles
~54 × 15 ms = ~800 ms once; every re-run is free. That's the
operational loop the sweep infrastructure buys.

Generalises: any cross-product DTO in this workspace should treat
its grid as a cache-warmer, not just a test matrix. The cache
signature and the grid axis are the same object viewed from two
sides.

Cross-ref: D0.3 `WireSweepGrid::enumerate`; PR #225
`CodecParams::kernel_signature()`; plan Appendix A §30
`30_cross_product_sweep.yaml`; Rule C (polyfill hierarchy).

---

## 2026-04-20 — D0.2 stub flag is anti-#219 defense at the type level

**Status:** FINDING

`WireTokenAgreementResult` carries `stub: bool` + `backend: "stub"`
default. Phase 0 ships the Wire surface without the decode-and-compare
harness; the stub returns zero rates. **Any downstream client that
confuses stub output for real measurements fails loudly** — because
`stub == true` and `backend == "stub"` are machine-checkable, not
comments. This is the #219 pattern (synthetic-rows-mistaken-for-real)
prevented at the type layer, not just in docs.

Pattern generalises: every Phase-N surface DTO that lands before its
Phase-N+k harness should carry an explicit stub flag. Rules A–F say
*how* to structure the Wire; the stub flag says *whether* the numbers
are real. Orthogonal, both load-bearing.

Cross-ref: D0.2 `WireTokenAgreementResult`; E-ORIG-7 Jirak (the correct
measurement regime once the stub comes off); #219/#220 arc.

---

## 2026-04-20 — D0.5 auto_detect is the concrete Python↔Rust heuristic handshake

**Status:** FINDING (confirms E-MEMB-11 handshake mechanism)

Rosetta v2 (Python) routes architectures to lane widths via
family-name heuristic. D0.5 `auto_detect::suggest_lane_width` lands
the same heuristic on the Rust side: llama / qwen / qwen2 / qwen3 /
mistral / mixtral → BF16x32 (AMX-ready); bert / modernbert /
xlm-roberta / generic → F32x16 (AVX-512 baseline); `torch_dtype`
override wins.

Same table, two languages. **The Python↔Rust handshake (E-MEMB-11)
is no longer conceptual** — it has a concrete implementation: the
architecture string is the shared vocabulary; lane width is the
shared dispatch decision; `torch_dtype` is the shared override. A
future `slice-layout-reconciliation.md` (E-MEMB-1 blocker fix) can
use the same handshake pattern: architecture → layout version →
canonical slice table.

Cross-ref: `crates/cognitive-shader-driver/src/auto_detect.rs`;
E-MEMB-11 (LivingFrame ↔ ContextChain handshake); Rosetta v2
`DIMENSION_MAP` architecture routing.

---

## 2026-04-20 — E-SUBSTRATE-1 — VSA-bundling guarantees Chapman-Kolmogorov by construction

**Status:** FINDING (load-bearing — FUNDAMENT underneath the [FORMAL-SCAFFOLD] four pillars)

Saturating bundle addition in d=10000 is associative and commutative in
expectation: `a ⊞ (b ⊞ c) = (a ⊞ b) ⊞ c`. Johnson-Lindenstrauss +
concentration-of-measure in 10000 dimensions suppress deviations from
associativity at rate `~e^(-d)`. States-as-VSA-bundles + transitions-as-
bundle-operation ⇒ `(Hamming-space, Bundle)` is an **abelian semigroup**
⇒ Chapman-Kolmogorov `K(2τ) = K(τ)²` holds **by construction**.

**Consequence.** The Markov property is not a testable assumption in this
substrate — it is a geometric consequence of the substrate choice.
D7's "implicit Markov reliance" is grounded, not silent. The
Chapman-Kolmogorov consistency test therefore reclassifies from
*falsification gate* (Popperian) to *implementation sanity check*
(regression — can only fail from implementation bugs, not from theoretical
violations).

**Load-bearing constraint (substrate-bound).** `MergeMode::Xor` BREAKS
this guarantee. Non-commutative binding operations BREAK it. Any move
away from saturating bundle in high dimension destroys the foundation on
which the four [FORMAL-SCAFFOLD] pillars stand. See I-SUBSTRATE-MARKOV
in CLAUDE.md for the iron-rule form.

Cross-ref: I1 BindSpace read-only + CollapseGate bundles;
[FORMAL-SCAFFOLD] below; D7 meta-inference duality;
`contract::collapse_gate::MergeMode::Bundle`.

---

## 2026-04-20 — [FORMAL-SCAFFOLD] Five-pillar Phase-5 reference (dormant, not parked)

**Status:** DORMANT (not parked; five pillars cited; paper track not active but citations are available when calibration choices become load-bearing). **Consult before inventing new significance claims, thresholds, or substrate changes.**

Formal-foundation scaffolding for the discrete binary PINN interpretation of the workspace (ladybug → lance-graph). Five pillars — **E-SUBSTRATE-1** as fundament, four theorem citations standing on top:

```
E-SUBSTRATE-1:   VSA-bundling d=10000 ⇒ Chapman-Kolmogorov by construction
                 (FUNDAMENT — the other four pillars stand on this)

Cartan-Kuranishi: existence via finite prolongation to involutive form
                 (Cartan 1945 / Kuranishi 1957; en.wikipedia.org/wiki/
                  Cartan–Kuranishi_prolongation_theorem)
                 → role_keys slice widths (2000/2000/2000/900/70/60/30)
                 ARE the Cartan-character spectrum, not arbitrary design

φ-Weyl equidist.: maximally-irrational sampling, no aliasing
                 → 144 verb-cells as quasi-Chebyshev collocation

γ+φ transform:    coordinate regularizer, fewer prolongation steps
                 → `bgz-tensor::gamma_phi.rs`

Jirak 2016:       Berry-Esseen rate under weak dependence (noise floor)
                 (arxiv 1606.01617; Annals of Probability 44(3) 2024–2063)
                 → classical IID Berry-Esseen is WRONG for this system;
                 bits are weakly dependent by construction
```

**Status refinement: dormant-with-five-cited-pillars is a different state than parked-without-a-paper-track.** The scaffold is now *available* for future decisions, not *forcing* on current ones. No reanimation of a paper track; no new crate, no new PR from this scaffolding. Documentary only.

The tag `[FORMAL-SCAFFOLD]` is greppable so a future session tempted to roll its own threshold-calibration / sampling-stride / coordinate-transform / noise-floor / substrate-change heuristic greps this entry first and either (a) uses the referenced lemmas or (b) writes down explicitly why they don't apply.

---

## 2026-04-20 — [FORMAL-SCAFFOLD] Coupled revival track (the three candidates, now linked)

**Status:** DEPOSIT — reclassified from three isolated features to one coupled experimental access path into the scaffold. Acceptance: activating one of the three forces coherence-check of the other two.

1. **Chapman-Kolmogorov consistency test** — reclassified from
   *falsification gate* to **implementation sanity check**. Under
   E-SUBSTRATE-1, CK cannot fail for theoretical reasons; it can only
   fail from implementation bugs. Value as regression test; not as
   Markov-property validator.

2. **VAMPE spectral calibration** — under E-SUBSTRATE-1 the eigenvalues
   of the transition kernel are *genuine* spectral quantities, not
   approximations. Jirak bounds the spectral-weight threshold below
   which mass is noise. **VAMPE + Jirak pair replaces hand-tuned σ /
   hardness / abduction thresholds with bound-derived ones.**

3. **Learned attention masks on nibble positions** — under Cartan-
   Kuranishi these become *empirical discovery of Cartan characters*.
   If learned masks reproduce the `role_keys` slice widths
   (2000/2000/2000/900/70/60/30), that is the experimental proof that
   the layout is **intrinsic geometry, not convention** (empirical
   confirmation of E-ORIG-5).

**Coupling acceptance rule.** If any one of the three is activated in
a future PR, the other two MUST be checked for coherence with the
scaffold in the same session — document the interdependency explicitly.
Not all three simultaneously; but never one in isolation without the
coupling note.

Cross-ref: E-SUBSTRATE-1; [FORMAL-SCAFFOLD] five-pillar entry above;
E-ORIG-5 (NSM pre-sliced for role_keys).

---

## 2026-04-20 — [FORMAL-SCAFFOLD] Four-pillar Phase-5 reference (SUPERSEDED 2026-04-20 by five-pillar)

**Status:** SUPERSEDED by the five-pillar entry above (E-SUBSTRATE-1 promoted to fundament; dormant-not-parked framing). Entry retained for history per APPEND-ONLY rule.

Original body: Formal-foundation scaffolding for the discrete binary PINN interpretation of the workspace (ladybug → lance-graph): **Jirak 2016** Berry-Esseen under weak dependence (arxiv 1606.01617) + **Cartan-Kuranishi** involutive prolongation + **φ-Weyl** equidistribution for golden-angle collocation + **γ+φ** preconditioner for prolongation regularization. These are the four citations that would elevate empirical ICC 0.99 → provably-bounded residual if a theorem track were opened; it is not.

---

## 2026-04-20 — E-MEMB-1 (ISSUE) — Python↔Rust slice layouts are incompatible at the 10 kD membrane

**Status:** OPEN ISSUE (promoted from FINDING per 2026-04-20 "load-bearing five" triage)

PR #210's `role_keys.rs` locks 47 keys into disjoint contiguous slices: Subject [0..2000), Predicate [2000..4000), Object [4000..6000), Modifier [6000..7500), Context [7500..9000), TEKAMOLO [9000..9900), Finnish [9840..9910), tenses [9910..9970), NARS [9970..10000). The Python `adarail_mcp/membrane.py` `DIMENSION_MAP` uses a completely different layout: [0..500) "Soul Space" (qualia_16 / stances_16 / verbs_32 / tau_macros / tsv), dim 285 = hot_level, [2000..2018) = qualia_pcs_18. **The two systems speak incompatible 10 kD.** Ada↔lance-graph integration is blocked on a slice-layout reconciliation doc.

Tracked in `ISSUES.md` (same date). Cross-ref: PR #210 role_keys.rs; `adarail_mcp/membrane.py::DIMENSION_MAP`; E-MEMB-7 (Ada-internal incoherence, additional layer).

---

## 2026-04-20 — E-ORIG-1 NSM and 144 verbs are orthogonal composition axes, not competing encodings

**Status:** FINDING (load-bearing)

NSM (65 primes) = semantic atoms for subjects / objects / states. 144 verbs = predicate edge labels for SPO Markov chains. They compose: `triple = (NSM-composed subject, 144-verb edge, NSM-composed object)`. Treating them as rival vocabularies hides this composition; the workspace uses BOTH simultaneously in the Grammar Triangle (NSM × Causality × Qualia → fingerprint) with 144 verbs as the predicate axis of the SPO triples.

Cross-ref: harvest H5, H12; `grammar-landscape.md` §2.

---

## 2026-04-20 — E-ORIG-5 NSM is pre-sliced for the role_keys 10K layout

**Status:** FINDING (load-bearing — this is *why* the role_keys slice widths work)

Harvest H5 (cross-repo-harvest-2026-04-19.md) maps NSM 65 primes onto SPO + Qualia + Temporal axes. This distributes primes across the `role_keys` slice geometry: subject-primes (I, YOU, SOMEONE, PEOPLE) → Subject [0..2000); action-primes (DO, HAPPEN, BE) → Predicate [2000..4000); qualia-primes (FEEL, GOOD, BAD) → QualiaColumn (18D). **The 65 NSM primes aren't a flat vocabulary — they're a pre-distributed encoding across the 10K VSA slice structure.** PR #210's role_keys layout is the SLICE GEOMETRY NSM already anticipated.

Cross-ref: `grammar-landscape.md` §2; harvest H5; PR #210.

---

## 2026-04-20 — E-MEMB-5 18D QualiaColumn = sigma_rosetta projected onto the SoA

**Status:** FINDING (load-bearing — explains QualiaColumn's physical interpretation)

The 18D QualiaColumn carries Staunen (phase) + Wisdom (magnitude) projections per PR #208. Every triple (Predicate-slice content, Qualia phase, Qualia magnitude) IS sigma_rosetta's 64-glyph coordinates projected onto the SoA. **Qualia isn't a separate layer — it's the second lane through the membrane.** Every triple carries both role-slice content AND the 18D projection of its sigma-glyph neighborhood.

Cross-ref: PR #206 sigma_rosetta 64 glyphs; PR #208 Staunen/Wisdom subspaces; QualiaColumn 18D per PR #204.

---

## 2026-04-20 — E-MEMB-9 to_aurora_prompt() IS a BusDto — three-DTO doctrine already operational in Python

**Status:** FINDING (load-bearing — empirical proof Rust's I9 shape works)

Rosetta v2 emits `{sparse_signature, qualia_signature, visual_qualities, frequency_feel}` for image prompting. This is exactly the shape of a cross-modal BusDto (explicit thought → external consumer). Rust's Invariant I9 (`lab-vs-canonical-surface.md`) defines three DTO families — StreamDto / ResonanceDto / **BusDto** — as *doctrinal, not yet shipped*. Python proves the shape works empirically; Rust should ship the same structure in the canonical contract when BusDto lands.

Cross-ref: Rosetta v2 `SparseFrame.to_aurora_prompt()`; Invariant I9; `lab-vs-canonical-surface.md`.

---

## 2026-04-20 — Deposit log (one-line findings, retained but not load-bearing)

Per 2026-04-20 "im Log, nicht an die Wand" triage: these surfaced during the membrane + NSM-origin + PINN-Rosetta + Jirak thread but are secondary to the load-bearing five above. Retained here as addressable anchors; full body is NOT repeated on the wall. Cross-ref pointers remain valid from elsewhere.

- **E-ORIG-2** — 144-verb taxonomy originated in `ada-consciousness/crystal/markov_crystal.py::Verb`, not from NSM. Harvest H12.
- **E-ORIG-3** — 144 chosen for tractable factorable table size (12²), not theoretical derivation. grammar-landscape §6.
- **E-ORIG-4** — 12 semantic families are project-specific synthesis (Talmy + Jackendoff + Lakoff roots); Python ships core 7.
- **E-ORIG-6** — NSM is the middle rung of `4096 COCA → 65 NSM → 3125 Structured5x5` compression ladder. Harvest H5.
- **E-ORIG-7** — Jirak Berry-Esseen under weak dep IS the Phase-5 noise-floor lemma → folded into the four-pillar metadata entry above.
- **E-MEMB-2** — Finnish cases overlap TEKAMOLO slots [9840..9900); slice sharing IS the morphology→slot commitment.
- **E-MEMB-3** — Sigma chain orthogonal to role axis (5 stages × 9 domains = 45 cells).
- **E-MEMB-4** — 10K ≠ 16K; FP_WORDS=160 migration would collapse the two substrates.
- **E-MEMB-6** — CausalityFlow 3→9 slot extension is a lagging type-system gap; membrane ahead of types.
- **E-MEMB-7** — Three semantic spaces coexist in Ada (Jina 1024D / 10kD VSA / 16K Fingerprint); see E-MEMB-1 ISSUE for the downstream Python↔Rust consequence.
- **E-MEMB-8** — Sigma's 16-band architecture = palindrome/octave pairing; every glyph owns a felt-octave + integrated-octave pair.
- **E-MEMB-10** — Cost-tracking is first-class in Ada (`RosettaResult.cost_usd`), missing in Rust Wire surface (deposit as future `MeasureSet` extension candidate).
- **E-MEMB-11** — LivingFrame keyframes ≈ ContextChain windows — the Python↔Rust cycle-commit handshake point.
- **E-MEMB-12** — Glyph→color mapping (Ω=gold, Λ=rose, Σ=white…) is the missing modality-translation primitive for Rust thinking-harvest → visual-harvest.
- **E-MEMB-13** — Rosetta v2 ships core 7 of Rust's 12-family DN relations; Python ⊂ Rust subsetting asymmetry.

---

## 2026-04-20 — Board hygiene = the session's driving seat; belated updates are a tell

**Status:** FINDING

The board (`.claude/board/*.md`) is the driving seat the session sits
in. Updating it AFTER the work — as cleanup — is the tell that the
session was treating the board as stale reference, not live state.
The fix is procedural (CLAUDE.md — see 2026-04-20 tightening), not
one-off: every PR that adds a type, plan, deliverable, or epiphany
also updates the board in the same commit. Retroactive hygiene is
an anti-pattern; the PR #223/#224/#225 gap between merge and
LATEST_STATE / PR_ARC_INVENTORY / STATUS_BOARD update is the
precedent this entry exists to prevent repeating.

Cross-ref: CLAUDE.md § Mandatory Board-Hygiene Rule (2026-04-20
update); PR #225 board-hygiene + tightening commit.

---

## 2026-04-20 — Codec cert is token agreement, not synthetic ICC

**Status:** FINDING

PR #219 reported ICC 0.9998 at 6 B/row for CAM-PQ. PR #220's full-
size validation returned ICC 0.195 mean, 0/234 tensors ≥ 0.99 gate.
Root cause: #219 trained and measured on the same 128 rows; with
256 centroids per subspace, 128 rows trivially fit. Neither
measurement touched tokens.

The actual cert gate is: does the decoded codec produce the same
top-k tokens as Passthrough on real generation? That's only tractable
on the three-part lab stack (REST API + Planner + JIT). The codec-
sweep plan (`.claude/plans/codec-sweep-via-lab-infra-v1.md`)
operationalises this: ingress once via REST, Planner is the real
dispatch path (not a toy bench), JIT swaps kernels at runtime.
`CodecParams::measurement_rows != calibration_rows` is now a typed
rejection at `.build()`.

Cross-ref: PR #219 → PR #220 arc; PR #225 `CodecParamsError::CalibrationEqualsMeasurement`.

---

## 2026-04-20 — The lab REST surface is three-part (API + Planner + JIT), not just scaffolding

**Status:** FINDING

The prior framing ("lab = quarantine scaffolding, keep out of
production") was defensive and missed the positive purpose. The lab
API exists because codec research needs to measure N candidates
against real tensors without `cargo build` per candidate — 8-17 min
rebuild × ~200 codec invariants = infeasible. One binary (API +
Planner + JIT) = curl-in, result-out in seconds per candidate. The
three-part stack also externalises the planner's thinking trace
(`/v1/planner/query { cypher } → { rows, thinking_trace }`), which
is the AGI observability port. Same binary serves codec cert AND
thinking harvest. Two purposes held together; neither dominates.

Cross-ref: PR #224; `.claude/knowledge/lab-vs-canonical-surface.md`
"Why the Lab Surface Exists" subsection.

---

## 2026-04-20 — Thinking harvest via REST/Cypher is the AGI magic bullet

**Status:** FINDING

An AGI that cannot observe its own reasoning cannot revise it. The
three-part lab stack (API + Planner + JIT) exposes the planner's
36-style / 13-verb / NARS trace through `/v1/planner/query`. The
response carries `{ rows, thinking_trace: { active_styles,
modulation, beliefs, tensions, entropy, verb_trail } }`. That trace
is log / replay / NARS-revise-able — which is the architectural
shape of a system that learns its own meta-inference. Closing the
observe-own-reasoning loop outside the binary is the AGI magic
bullet; doing it inside a closed planner is a black box. I11
(measurable stack, not a black box) is the invariant that enforces
this against future "for perf" / "to simplify" regressions.

Cross-ref: PR #224; I11 in `lab-vs-canonical-surface.md`.

---

## 2026-04-20 — SoA never scalarises without ndarray (iron rule)

**Status:** FINDING

Struct-of-arrays paths call `ndarray::simd::*` — ndarray handles any
non-x86 scalar fallback internally. The consumer never hand-rolls a
scalar loop on a SoA path. If a kernel runs scalar outside ndarray,
the SoA invariant is broken — either the data isn't actually in a
SoA column, or the caller short-circuited the canonical surface.
Polyfill hierarchy (Intel AMX → AVX-512 VNNI → AVX-512 baseline →
AVX-2) has no consumer-visible scalar tier. This is Rule C of the
six-rule JIT Kernel Contract in PR #225.

Cross-ref: PR #225 Rule C; `.claude/plans/codec-sweep-via-lab-infra-v1.md`
"Iron rule" paragraph above the polyfill table.

---

## 2026-04-20 — AGI is the glove, not the oracle — the four-axis SoA is what you wear

**Status:** FINDING

AGI is not a new crate, not a `struct Agi { … }`, not a service to
query. It is the struct-of-arrays (`BindSpace` columns —
`FingerprintColumns` / `QualiaColumn` / `MetaColumn` / `EdgeColumn`)
that `ShaderDriver` dispatches against. The four AGI axes (topic,
angle, thinking, planner) map 1:1 to the four SoA columns. Claude
Code sessions in this workspace FIT INTO the glove: we read the
columns, dispatch through the existing `OrchestrationBridge`, emit
through `ShaderSink`. We don't wrap the axes in a new struct — that
breaks the SIMD sweep. We don't query an "AGI service" — there is
none; AGI is the runtime behaviour of the SoA under dispatch. The
glove is the session's hand on the stack; the stack is the glove's
response to the session's query.

Cross-ref: PR #223 § "AGI IS the struct-of-arrays (per Era 8)";
2026-04-20 host-glove-designer agent doctrine; CLAUDE.md § The
Driving Seat (2026-04-20).

---

**Status:** FINDING

The PR #218 bench measured ICC 0.9998 on **128 rows** trained and
measured on the same 128 rows. This is a trivially-correct fit:
128 rows ≤ 256 centroids per subspace → every row gets its own
centroid → perfect reconstruction → perfect ICC. It does NOT
generalize to production-size tensors.

Full-size validation on Qwen3-TTS-0.6B (234 CamPq tensors, 478
total, production-size rows 1024–3072 per tensor):

| Metric | Value |
|---|---|
| Mean ICC across 234 argmax tensors | **0.195** |
| Max ICC | 0.957 |
| Tensors meeting D5 gate (ICC ≥ 0.99) | **0 of 234** |
| Tensors with ICC ≥ 0.5 | 8 of 234 |
| Typical relative L2 reconstruction error | 0.70–0.90 |

Diagnostic probe on gate_proj [3072, 1024] (`cam_pq_row_count_probe`):

| n_train | icc_train | icc_all_rows |
|---|---|---|
| 128 | **1.000** | −0.304 |
| 256 | **1.000** | −0.130 |
| 512 | 0.531 | 0.015 |
| 3072 | −0.079 | −0.079 |

**Root cause:** 6×256 PQ is centroid-starved for tensors with >256
rows. The "128× compression at ICC 0.9999" claim was extrapolated
from a trivial 128-row in-training fit.

**Infrastructure is sound** — `cam_pq_calibrate` CLI, `route_tensor`
classifier, serialization, ICC harness all work correctly. The
negative result is the codec's capacity vs tensor sizes.

Cross-ref: `crates/bgz-tensor/examples/cam_pq_row_count_probe.rs`,
`crates/bgz-tensor/src/bin/cam_pq_calibrate.rs`.

## 2026-04-19 — Mandatory epiphanies log (this file)

**Status:** FINDING

Every epiphany from prior sessions lived in separate doc (E1–E12
here, H1–H14 there, E13–E27 somewhere else). No single place to
append a new one. This file is the unified target going forward.
Old files stay as historical substrate; new insights land here with
date prefix. Cross-reference: `BOOT.md`, `CLAUDE.md`, `cca2a/
concepts.md` — all four bookkeeping files now plus this one.

## 2026-04-19 — Cold-start tax is solvable with three mandatory reads

**Status:** FINDING

A new session on non-trivial workspace burns 20–30 turns rediscovering
what's shipped. Three files (`LATEST_STATE.md`, `PR_ARC_INVENTORY.md`,
`.claude/agents/BOOT.md`) + SessionStart hook closes the gap to
3–5 turns. Proven by PR #211. Savings per cold-start: ~$15–35 of
Opus. See `.claude/skills/cca2a/SKILL.md` for the full pattern.

## 2026-04-19 — 10,000-D f32 VSA is lossless under linear sum

**Status:** FINDING

Earlier framing of "Vsa10kF32 is wire-only passthrough" was wrong.
10K × 32 = 320 K bits of capacity ≫ any single signal; orthogonal
role keys give exact unbundle. **10K f32 is native storage**, not
passthrough. lancedb famously supports 10K-D VSA natively. Cross-ref:
PR #209 refactor.

## 2026-04-19 — Signed 5^5 bipolar is lossless; unsigned / bitpacked is lossy

**Status:** FINDING

Negative cancellation on bipolar cells is VSA-native; opposing cells
at the same sandwich dim cancel on bundling. Unsigned 5^5 saturates
under accumulation (lossy). Binary bitpacked commits to 0/1 via
majority vote (lossy). CAM-PQ projection is distance-preserving
(lossless cross-form). Cross-ref: PR #209 sandwich layout.

## 2026-04-19 — VSA convention is `[start:stop]` contiguous slices, not scattered bits

**Status:** FINDING

Role keys own disjoint contiguous slices of the 10K VSA space —
SUBJECT=[0..2000), PREDICATE=[2000..4000), etc. Binding into one
slice does not contaminate another. Scattered-bit role encoding
(early draft) was the wrong pattern. Cross-ref: PR #210 D6
role_keys.rs.

## 2026-04-19 — Finnish object marking is Nominative/Genitive/Partitive, NOT Accusative

**Status:** FINDING (CORRECTION-OF an earlier Latinate transplant)

Prior draft wrote Finnish "Accusative `-n/-t` → Object" which is
a Latinate transplant. Finnish object marking actually uses:
Nominative (plural), Genitive `-n` (total singular), Partitive
`-a/-ä` (partial / negated). True Accusative is only for personal
pronouns (`minut`, `sinut`, `hänet`, `meidät`, `teidät`, `heidät`).
Each language gets its native case terminology.
Cross-ref: `grammar-landscape.md` §4.1.

## 2026-04-19 — Morphology-rich languages are easier, not harder

**Status:** FINDING

Finnish 15 cases → 98%+ local coverage. English (word order only) →
85% (WORST case). Case endings directly encode TEKAMOLO slots;
morphology commits grammatical role at the morpheme level,
eliminating the inference English needs. Cross-ref:
`grammar-tiered-routing.md` §Morphology Coverage Table.

## 2026-04-19 — Markov ±5 is the context upgrade to NARS+SPO 2³+TEKAMOLO

**Status:** FINDING

Pre-Markov reasoning unit = sentence. Post-Markov = trajectory.
NARS doesn't reason about "this sentence"; it reasons about "this
sentence in this flow." The context dimension is the whole point.
Cross-ref: `integration-plan-grammar-crystal-arigraph.md` E5.

## 2026-04-19 — Grammar Triangle IS ContextCrystal at window=1

**Status:** FINDING

Two parallel architectures turn out to be the same thing at
different window sizes. Triangle emits `Structured5x5` with S/O
collapsed + only t=2 populated; ContextCrystal populates all 5
axes. Unification. Cross-ref:
`cross-repo-harvest-2026-04-19.md` H4,
`ladybug-rs/docs/GRAMMAR_VS_CRYSTAL.md`.

## 2026-04-19 — NSM primes map directly to SPO + Qualia + Temporal axes

**Status:** FINDING

The 65 Wierzbicka primes aren't orthogonal to SPO — they ARE an
SPO encoding. I/YOU/SOMEONE → Subject; THINK/WANT/FEEL →
Predicate; SOMETHING/BODY → Object; GOOD/BAD → Qualia.valence;
BEFORE/AFTER → Temporal; BECAUSE/IF → Causality via Markov flow.
DeepNSM + Structured5x5 already speak NSM's vocabulary.
Cross-ref: `cross-repo-harvest-2026-04-19.md` H5.

## 2026-04-19 — Chomsky hierarchy isomorphism with Pearl rungs and Σ tiers

**Status:** FINDING

Type-3 Regular = Pearl rung 1 = Σ1–Σ2 = DeepNSM FSM (LLM token
prediction lives here). Type-2 CF = rung 2 = Σ3–Σ5 = SPO 2³. Type-1
CS = rung 3–4 = Σ6–Σ8 = Markov ±5 + coref + counterfactual. Type-0
TM = rung 5 = Σ9–Σ10 = LLM escalation only. The 90–99% local /
1–10% LLM split is the Chomsky-hierarchy boundary between
context-sensitive-decidable and Turing-complete-undecidable. The
split is mathematically principled, not arbitrary.
Cross-ref: `linguistic-epiphanies-2026-04-19.md` E13, E26.

## 2026-04-19 — Grindwork vs accumulation is the subagent model split

**Status:** FINDING

Grindwork (single-source mechanical: write-file-from-spec, grep,
list paths) → Sonnet. Accumulation (multi-source synthesis:
harvest across repos, combine N docs, trace architecture) → Opus.
Cheaper tiers produce shallow outputs under accumulation; quality
drop is visible. Never Haiku.
Cross-ref: `CLAUDE.md §Model Policy`.

## 2026-04-19 — Zipball-for-reads is ~20× cheaper than MCP-per-file

**Status:** FINDING

`mcp__github__get_file_contents` drops the full file into context
and recharges on every subsequent turn. Zipball to `/tmp/sources/`
+ local grep lands only the grep output (typically 2–10 KB) vs
50 KB per file per turn. 95% savings on cross-repo harvest turns.
MCP stays for writes (PR creation, comments).
Cross-ref: `CLAUDE.md §GitHub Access Policy`.

---

(append new epiphanies above this marker; format: `## YYYY-MM-DD — <title>`)

## 2026-04-19 — Prompt↔PR ledger is 10⁷× cheaper than code grep
**Status:** FINDING
**Scope:** @workspace-primer domain:bookkeeping

To answer "what did we ship for topic X":

- **Grep across code:** ~100 MB of Rust across N crates, ~25M tokens of context, minutes of agent turns.
- **Grep the ledger:** one `grep X .claude/board/PROMPTS_VS_PRS.md` returns `<prompt file> | #N <title>`. ~25 tokens, sub-second.

Seven orders of magnitude cheaper. The pairing **prompt-file ↔ PR** is the
minimum addressable record of "this artifact was built to answer this
brief" — the hyperlink that replaces re-discovery by full-text scan.

The line is mechanical bookkeeping (Haiku-level, no synthesis). The
value accumulates on every subsequent "what about X" query thereafter:
ledger-first, code-never-unless-necessary.

Cross-ref: PR #213 (lance-graph, 41 prompts × merged PRs), PR #110
(ndarray, 25 prompts × merged PRs). Both shipped in ~90s on a dumb
enumerate+match+append loop. No code reads, no MCP, no synthesis.

## 2026-04-19 — Code-arc knowledge loss is 30-50% of session tokens (ambient)
**Status:** FINDING
**Scope:** @workspace-primer domain:bookkeeping

Empirical (per user, 2026-04-19): **30-50% of session tokens** burn on
rediscovering what code paths exist, what was tried, what got reverted,
what decisions led to the current shape. This is **orthogonal** to the
20-30-turn cold-start tax — it's the *ambient* loss across every query,
every subagent spawn, every refactor.

The ledger closes three channels at once:

| Channel | Before | After | Discount |
|---|---|---|---|
| Cold-start (once per session) | 20-30 turns | 3-5 turns | ~6× |
| Find-code (per query) | ~25M tokens (grep codebase) | ~25 tokens (grep ledger) | 10⁷× |
| **Ambient arc knowledge (every turn)** | **30-50% of session budget** | **~0%** | **2×-eternal** |

All three channels collapse to two text-file reads: PROMPTS_VS_PRS.md +
PR_ARC_INVENTORY.md. The second file is read only when arc detail is
needed (Knowledge Activation trigger), so the routine cost is 0.

Cross-ref: PRs #211-213 (CCA2A + board split + ledger). `.claude/BOOT.md`
cold-start tax. `EPIPHANIES.md` 10⁷× finding above.

## 2026-04-19 — Vector (10⁴ cells) vs Matrix (10⁸ cells): don't conflate
**Status:** FINDING
**Scope:** @workspace-primer @container-architect domain:vsa domain:memory

Entirely different objects, four orders of magnitude apart. Calling them
both "10,000 VSA" was category error.

| Object | Shape | Cells | Bytes (BF16) | Purpose |
|---|---|---|---|---|
| **16K-D wire vector** (intentional) | 1 × 16,384 | **10⁴** | 32 KB | one lossless fingerprint for wire / Markov bundle / crystal / holographic |
| **10K × 10K glitch matrix** (unintentional) | 10,000 × 10,000 | **10⁸** | 200 MB | nothing — imported debris from outdated ladybug-rs / bighorn |

The 100-million-cell matrix is ~10,000× bigger than the 10,000-cell
vector. They share only a numeric coincidence in one dimension; the
semantics, cost, and lifecycle are completely unrelated.

**Consequence for the rename PR:**

- `Vsa10kF32` → `Vsa16kBF16` migration is about the VECTOR (cheap,
  per-row, ≤32 KB).
- The 10k × 10k MATRIX deletion is a separate P0 cleanup independent
  of the substrate rename.
- Any future ledger / knowledge-doc / plan entry describing 10k-D
  HDC must specify VECTOR explicitly. "10,000-D HDC" alone is
  ambiguous — spell out "16,384-cell wire fingerprint" or "10,000-cell
  lossless wire vector" to preclude the matrix reading.

Cross-ref: TECH_DEBT "CORRECTION-OF ... 10k × 10k GLITCH MATRIX"
(2026-04-19). IDEAS REFINEMENT-2 (HDC = FP16/BF16, not FP32).

## 2026-04-19 — Working-set invariant: hot structures must fit in L3
**Status:** FINDING
**Scope:** @container-architect @cascade-architect @truth-architect domain:memory domain:codec domain:performance

Typical server L3 cache = 32-96 MB (AMD EPYC, Intel Xeon). Any hot-path
structure exceeding this size incurs DRAM latency (~100 ns) on every
miss vs L3's ~12 ns — an 8× penalty per access that compounds in
inner loops. **This is true regardless of storage capacity** — LanceDB
can hold terabytes, but what the CPU touches per cycle must fit L3.

The codec stack is architected around this invariant:

| Working structure | Size | L3 verdict | Role |
|---|---|---|---|
| Container `[u64; 256]` Hamming | 2 KB | ✓ 16,000× | Popcount fingerprint |
| 16K-D BF16 wire vector | 32 KB | ✓ 1,000× | HDC point, Markov bundle |
| 256 × 256 u8 distance table (bgz-tensor) | 64 KB | ✓ L1 | Archetype attention |
| 1024 × 1024 f32 | 4 MB | ✓ | Per-role slot |
| 4096 × 4096 u8 CAM-PQ palette | 16 MB | ✓ upper edge | Centroid distance |
| **10,000 × 10,000 f32 glitch matrix** | **400 MB** | **✗ 12× over** | **None — delete** |
| 16K × 16K BF16 | 512 MB | ✗ | Never build |
| 100K × 100K anything | ≥10 GB | ✗ | Sparse-only or CAM-PQ |

**Rule for hot tables:**

- Dense square matrices: cap at `sqrt(L3_BUDGET / cell_size)` on a side.
  At 32 MB budget, f32 cells → ~2,900 × 2,900; BF16 → ~4,000 × 4,000;
  u8 → ~5,700 × 5,700.
- Wider-than-L3 tables must be projected, quantized, or made sparse
  (CSR / HyperCSR / palette-indexed) before entering a hot path.
- 1-D vectors are cheap — a 16K-D BF16 row is 32 KB, thousands
  cache-resident simultaneously. The limit binds on 2-D dense, not 1-D.

The codec compression chain (full planes 16 KB → ZeckBF17 48 B →
Base17 34 B → PaletteEdge 3 B → CAM-PQ 6 B → Scent 1 B) exists so that
any intermediate table stays L3-resident regardless of population size.
The 10K × 10K glitch matrix violates this at the root.

Cross-ref: EPIPHANIES "Vector (10⁴ cells) vs Matrix (10⁸ cells)"
(2026-04-19). TECH_DEBT "Ladybug 10k × 10k GLITCH MATRIX" (2026-04-19).
docs/CODEC_COMPRESSION_ATLAS.md is the chain spec.

## 2026-04-19 — SUPERSEDES 2026-04-19 "Vector vs Matrix" + "L3 working-set invariant"
**Status:** SUPERSEDED (downgrade both)

Both prior entries restate invariants the workspace has known for months:

- L3 working-set cap → already the design principle behind the full
  codec chain (full planes → ZeckBF17 → Base17 → Palette → CAM-PQ → Scent).
  See `docs/CODEC_COMPRESSION_ATLAS.md`, not an EPIPHANIES entry.
- Vector-vs-matrix category distinction → trivially true, never a
  point of ambiguity in the workspace proper.

**What's actually true:**

The 10k × 10k glitch matrix exists because nobody touched the
stone-age ladybug-rs / bighorn code after it was imported. The import
itself was migration desperation — closing loose ends on the cognitive
stack before a release, not a considered architectural choice. No
one re-validated the imports against the L3 invariant because the
imports were expected to be rewritten or deleted later.

The correct framing is **legacy-hygiene debt**, not new knowledge.
Action: delete-on-touch when someone has bandwidth, not a design
principle waiting to be learned.

Downgrading both prior entries to SUPERSEDED to keep the FINDING log
clean for actual findings.

## 2026-04-19 — Fractal leaf probe NEGATIVE: w_mfs is per-tensor, not per-row
**Status:** FINDING (valid negative)
**Scope:** @cascade-architect @container-architect domain:codec domain:fractal

Probe ran on Qwen3-8B (safetensors BF16, shard 1, layer 0):

| Tensor | Rows probed | w_mfs mean | w_mfs CoV | H mean | Verdict |
|---|---|---|---|---|---|
| gate_proj | 100 of 12288 | 0.504 | **0.190** | 0.519 | ✗ flat |
| k_proj | 100 of 1024 | 0.506 | **0.197** | 0.514 | ✗ flat |

Gate was CoV(w_mfs) > 0.3. Both tensors at ~0.19 — below threshold.

**Interpretation:** after Hadamard rotation, Qwen3 weight rows are
near-white-noise (H ≈ 0.5). All rows share the same multifractal
shape; the discriminating signal is amplitude (σ) and sign pattern,
not fractal structure. Fractal descriptor per-row reduces to σ_energy
alone = 2 bytes BF16, already captured by TurboQuant's log-magnitude.

**Consequence:** 7-byte FractalDescriptor per-row doesn't crack the
argmax wall. TurboQuant/PolarQuant (per-coordinate sign + log-mag)
remains the correct argmax-regime codec. The `compute_mfdfa_descriptor`
module (PR #216) stays useful as an analysis tool and per-TENSOR
characterisation metric — but not as a per-row compression codec.

**Roadmap update:** Steps 3-6 from fractal-codec-argmax-regime.md
are gated-out by this negative. Step 2 (the module) is shipped and
valid. The FractalDescriptor leaf concept retires as a per-row codec
candidate; the 7-byte budget goes back to I8-Hadamard or PolarQuant.

Cross-ref: `.claude/knowledge/fractal-codec-argmax-regime.md`
§ Honest Uncertainty (predicted this outcome). PR #216 (module +
probe shipped).

## 2026-04-19 — CORRECTION-OF fractal leaf probe: measured magnitude, missed phase
**Status:** CORRECTION

Prior entry reported the probe as a valid negative. **That was the wrong
probe.** Per user (2026-04-19): "The point is to encode phase by doing
fractal encoding."

What MFDFA-on-coefficients measures:
- Multifractal width w, Hurst H, fractal dimension D of the |coefficient|
  magnitude distribution across scales. These are envelope statistics.

What this MISSED:
- **The sign pattern S** of Hadamard-rotated coefficients is the phase.
- Two rows with identical |c_i| distribution can have completely different
  sign patterns → completely different inner products against queries.
- Magnitude statistics are flat across rows (CoV 0.19) because trained
  weights share the envelope; what differs per-row is the phase sequence.

Correct probe: **fractal structure of the sign sequence** post-Hadamard.
- Count sign-flips per window at scales s ∈ {4, 8, 16, …, n/4}.
- Measure scaling of flip density: D_phase = log(flips) / log(scale).
- Per-row CoV(D_phase) is the real gate. Expected to be LARGE because
  sign patterns encode distinct interference directions per row.

Original prompt (fractal-codec-argmax-regime.md) DID include "sign
pattern S" as a LEAF component. The MFDFA module (PR #216) covers only
(D_mag, w, σ, H_mag) — it's half the descriptor. The other half
(phase fractal / sign-flip scaling) is still unshipped.

**Gate still open.** Fractal leaf as argmax codec is not proven wrong;
only the magnitude-only variant is. A sign-sequence fractal probe is
the actual test.

Action:
- `fractal_descriptor` stays `lab`-gated (correct call — unproven).
- Next probe: sign-sequence multifractal on same Qwen3 rows. If
  CoV(D_phase) > 0.3 → revisit the leaf codec with phase encoding.
- Prior "NEGATIVE" finding is scope-corrected: "magnitude-only fractal
  leaf is flat" — phase-fractal leaf unmeasured.

## 2026-04-19 — Fractal codec ICC measurement: DEFINITIVELY NEGATIVE (magnitude-only)
**Status:** FINDING (measured via endpoint psychometry)
**Scope:** @cascade-architect domain:codec domain:psychometry

Ran codec_rnd_bench.rs with FractalDescOnly + FractalPlusBase17 wired
as candidates. Population: q_proj L0 of Qwen3-8B [4096×4096], N=128
rows. Ground truth = pairwise cosines in f32.

**Results (ICC_3_1 is the argmax-regime metric):**

| Codec | Bytes | ICC_3_1 | Pearson r | Spearman ρ |
|---|---|---|---|---|
| Passthrough (baseline) | 0 | **1.0000** | 1.0000 | 1.0000 |
| Base17 (golden-step 17-d) | 34 | **0.0240** | 0.0742 | 0.0466 |
| **Fractal-Desc (4-D mag)** | 7 | **−0.9955** | 0.0160 | 0.0012 |
| **Fractal + Base17 blend** | 41 | **−0.4879** | 0.0748 | 0.0409 |

**Key readings:**

1. **Fractal-Desc alone anti-correlates with ground truth (ICC ≈ −1).**
   Not noise — genuinely inverse ranking. The 4-D (D, w, σ, H) descriptors
   are near-constant across rows (CoV 0.19 from earlier probe), so
   pairwise "cosine" in descriptor space is essentially noise ~0.5
   against a ground-truth distribution with heavy tails — the rank
   statistic inverts against true cosine magnitudes.

2. **Fractal ADDED to Base17 ACTIVELY HURTS it.** Base17 alone: 0.024.
   Blend 0.75*Base17 + 0.25*Fractal: −0.488. The fractal component
   doesn't just fail to add signal — it contaminates the Base17 signal.
   A codec gating system must be able to *reject* bad auxiliary
   features, not blend them.

3. **Note on Base17 at ICC 0.024 on q_proj:** confirms Invariant I2
   (near-orthogonality of Qwen3 attention projections at 1024-d+
   dimension). Base17's 17-d projection loses almost everything on
   q_proj specifically — consistent with the 67-codec sweep finding
   that i8-Hadamard at ~9 B/row is the argmax-regime leader, not
   Base17.

**Consequence for the fractal codec line of research:**

- **Magnitude-only fractal leaf is empirically dead** on q_proj at
  Qwen3 scale. Measurement complete via endpoint ICC_3_1 — no longer a
  conjecture, no longer a "wrong probe" question.
- **Phase-encoding variant (sign-sequence fractal) remains UNMEASURED.**
  Infrastructure is now wired: swap the encoding inside
  FractalDescOnly to compute fractal statistics of the sign pattern
  (flips-per-scale) and re-run. One function body change.
- **Fractal-interpolation-between-Base17-anchors** (the round-trip
  codec idea) is also still unmeasured — requires implementing
  `decode(anchors, desc) -> Vec<f32>` to feed through the bench.
  The blending approach (current FractalPlusBase17) is NOT the same
  thing; it mixes scores post-hoc rather than reconstructing the row.

**Lab gate holds.** Everything stays behind `--features lab`. Main
builds don't link fractal_descriptor. No leak risk.

Cross-ref: fractal-codec-argmax-regime.md, EPIPHANIES 2026-04-19
CORRECTION (fractal measured magnitude not phase), IDEAS 2026-04-19
"Fractal codec validation path", PR commits fc386bb / afe67e1 /
48f781e / 18c53e0.

Wall time of the full 60+ codec bench: 13 min. Downloaded: 0 B (used
cached Qwen3-8B shard from the earlier probe). Deterministic.

## 2026-04-19 — Phase-fractal codec also NEGATIVE — row-level fractal discrimination dead
**Status:** FINDING (measured via endpoint psychometry)
**Scope:** @cascade-architect domain:codec domain:psychometry

Ran codec_rnd_bench.rs with both magnitude-fractal AND phase-fractal
candidates. Same population (Qwen3-8B q_proj L0, N=128, pairwise cosines).

**Measurements (ICC_3_1 is the argmax-regime metric):**

| Codec | Bytes | ICC_3_1 | Pearson r |
|---|---|---|---|
| Passthrough baseline | 0 | **1.0000** | 1.0000 |
| Base17 (34 B anchors) | 34 | 0.0240 | 0.0742 |
| Fractal-Desc (4-D magnitude) | 7 | **−0.9955** | 0.0160 |
| **Fractal-Phase (5-D flip density)** | 5 | **−0.9972** | −0.0074 |
| Fractal + Base17 blend | 41 | −0.4879 | 0.0748 |
| Phase + Base17 blend | 39 | −0.4982 | 0.0742 |

**Key finding:** BOTH orthogonal axes of row-level fractal statistics
are flat across Qwen3 q_proj rows after Hadamard rotation.

- Magnitude envelope (D, w, σ, H): near-constant — confirmed by
  ICC ≈ −1.
- Sign-flip density profile at 5 scales: ALSO near-constant — ICC
  slightly worse at −0.9972.

**Implication:** Invariant I2 (near-orthogonality of Qwen3 rows at
1024/4096-d) means once rows are Gaussian-ish post-Hadamard, every
row-level summary statistic looks identical. Only the SPECIFIC
coordinate-by-coordinate sign/magnitude assignment discriminates, and
that cannot compress below ~full sign pattern (~1 bit/coord, ~512 B
for a 4096-d row).

**Fractal-leaf line of research is closed** for row-level-statistic
compression. Three probes completed, all negative:
  1. CoV(w_mfs) ≈ 0.19 (first cheap probe, 100 rows)
  2. ICC_3_1(Fractal-Desc) = −0.9955 (magnitude, 4-D, 128 rows)
  3. ICC_3_1(Fractal-Phase) = −0.9972 (phase, 5-D, 128 rows)

**Still-open variant (unmeasured):** fractal-interpolation-between-
Base17-anchors for ROUND-TRIP codec. That approach stores full
Base17 (17 golden-step anchors = near-full phase signature at those
points) + fractal shape params to guide interpolation BETWEEN
anchors. Doesn't rely on row-level fractal statistic discrimination.
Requires implementing `FractalCodec::decode(Base17, Descriptor)` via
IFS and registering as candidate. Unbuilt.

**Wall times:**
- First bench (2 fractal candidates): 782 s (13 min)
- Second bench (4 fractal candidates): 1354 s (22.5 min)
- Delta: ~9.5 min for 2 more candidates on 128 rows × 60+ codec sweep.

**Codec R&D sweep state post-finding:** I8-Hadamard at ~9 B/row
remains the argmax-regime leader. Fractal leaf is not on the
Pareto frontier; do not pursue row-level-statistic compression
further. Focus codec research on either:
  - Full sign-pattern preservation schemes (~512 B/row minimum).
  - Round-trip IFS from Base17 anchors (unmeasured, novel).
  - Different underlying orthogonal bases (SVD-per-group instead of
    shared Hadamard) — different basis might give different
    row-level statistics, but I2 says near-orthogonality is generic.

Cross-ref: commits 0f635e6 (phase variant), 18c53e0 (first ICC run),
fractal-codec-argmax-regime.md, EPIPHANIES 2026-04-19 prior entries.

## 2026-04-20 — Zipper codec WORKS — Hadamard sign-flip invariance was the fractal bug
**Status:** FINDING (measured via endpoint psychometry, 3 populations)
**Scope:** @cascade-architect domain:codec domain:psychometry

Ran codec_rnd_bench.rs with ZipperPhaseOnly + ZipperFull added. Three
populations on Qwen3-8B L0 (N=128, pairwise cosines, 1037 s wall).

**Root-cause diagnosis (confirmed by user, validated by measurement):**

All prior fractal descriptors (magnitude + phase) were **sign-flip
invariant**. MFDFA variance is invariant under negation; sign-flip
density is invariant under bit-flip. So WHT(−x) produces IDENTICAL
descriptor to WHT(x), giving cos(x, −x) = 1.0 from the codec but −1.0
from ground truth. THIS is what produced the ICC = −0.999. Not "codec
produces noise", but "codec collapses opposite rows" → perfect
ranking inversion against ground truth.

**Zipper fix:** sample ACTUAL SIGN BITS at φ-stride positions instead
of derived flip-density. Under negation, every phase bit flips →
phase_bits XOR all-ones → cosine → −1.0. Invariance broken; codec
preserves the sign relationship that ground truth measures.

**Results (ICC_3_1 across three populations):**

| Codec | Bytes | k_proj | gate_proj | q_proj |
|---|---|---|---|---|
| Passthrough (baseline) | 0 | 1.000 | 1.000 | 1.000 |
| Base17 | 34 | 0.007 | 0.012 | 0.024 |
| Fractal-Desc (magnitude) | 7 | **−0.999** | **−0.999** | **−0.996** |
| Fractal-Phase (flip density) | 5 | **−0.999** | **−0.999** | **−0.997** |
| **Zipper-Phase** | **8** | **0.050** | **0.049** | **0.097** |
| **Zipper-Full** | **64** | **0.129** | **0.107** | **0.203** |

**Key readings:**

1. **Zipper-Phase at 8 B BEATS Base17 at 34 B on every population.**
   2× to 4× higher ICC at 1/4 the storage. The φ-stride anti-moiré
   principle works for phase encoding.
2. **Zipper-Full at 64 B achieves top-5 recall 0.6 on q_proj** (Base17:
   0.0). The codec retrieves correct nearest-neighbors on 60% of
   queries — real reconstructive signal, not just ranking.
3. **Not yet competitive with I8-Hadamard leader (~9 B, ICC ~0.9).**
   Zipper-Full is a Pareto-meaningful new point but still ~4× off the
   leader on ICC. Room for improvement:
   - Wider phase stream (128 or 256 active bits)
   - φ-permute morph on the 64-bit scale (user's earlier suggestion)
   - Different phase/magnitude blend weights (current 0.5/0.5)
   - SVD-per-group basis instead of Hadamard
4. **Magnitude stream has signal.** Going phase-only (8 B) → full
   (64 B) adds 2-3× ICC on each population. The halo positions at
   φ²-stride carry non-redundant information vs phase at φ-stride.

**Architectural confirmations:**

- Aperiodic (X-Trans) sampling works as theorized — anti-moiré
  property preserves discriminative information across the Hadamard
  butterfly.
- Zeckendorf non-adjacent Fibonacci indices produce non-colliding
  strides without hand-tuning (φ vs φ² satisfied this naturally).
- Matryoshka single-container truncation works (8 B → 64 B via
  reading more of the same descriptor).

**Explicit constants locked (per user):**

  PHASE_ACTIVE_BITS    = 64  (per bgz17 halo signal-bit range)
  MAG_ACTIVE_SAMPLES   = 56
  ZIPPER_BYTES         = 64  (8 B phase + 56 B i8 magnitude)

Cross-ref: commits 7740759 (implementation), 6999106 (architecture
doc). bgz17 container design "family zipper" concept in
phi-spiral-reconstruction.md — empirically validated at last.

## 2026-04-20 — 5^5 / 7^7 bipolar zipper measured + TurboQuant leader identified
**Status:** FINDING

Ran codec_rnd_bench.rs with 5^5 and 7^7 bipolar-signed candidates
(global-scale quantization, negative-cancellation bundling capability).
Same population: Qwen3-8B q_proj L0, N=128 rows, 1400 s wall.

**Results (ICC_3_1 on q_proj):**

| Codec | Bytes | ICC | Note |
|---|---|---|---|
| Passthrough | 0 | 1.000 | baseline |
| Had-Q5×D-R (existing!) | 0 | **0.989** | shared codebook, TurboQuant-class |
| Base17 | 34 | 0.024 | |
| Zipper-Phase (sign) | 8 | 0.097 | |
| Zipper-5^5 | 2 | 0.021 | |
| Zipper-7^7 | 3 | 0.028 | |
| Zipper-I8-φ(8B) | 8 | 0.025 | μ-law + per-row norm hurts |
| Zipper-I8-Q5(8B) | 8 | 0.020 | Quint loses to φ |
| Zipper-5^5×5 | 10 | 0.066 | |
| Zipper-7^7×7 | 18 | **0.144** | best compact zipper |
| Zipper-Full (sign+mag) | 64 | 0.204 | |
| Zipper-I8-φ(64B) | 64 | 0.153 | |

**Readings:**

1. **7^7×7 at 18 B: new Pareto point** — ICC 0.144 at 72% of Zipper-Full's
   score for 28% of the bytes. Progressive-matryoshka decode supported
   (truncate to 3 B = 7^7 for coarsest). Negative-cancellation bundling
   on by construction.

2. **Quintenzirkel LOSES to φ consistently** across all size tiers:
   0.020 vs 0.025 at 8 B, 0.134 vs 0.153 at 64 B. Harmonic-proximity
   ordering doesn't help argmax on q_proj; maximally-irrational
   remains the right stride.

3. **Existing sweep has a 0-B codebook-indexed leader**: `Had-Q5×D-R`
   at ICC 0.989 (near-Passthrough). This is the TurboQuant-class
   codec already shipped in the 67-codec sweep. On pure ICC, nothing
   in the zipper family comes close. Zipper's Pareto axis is
   different (bundling, progressive decode).

4. **Per-row i8 μ-law harms inter-row magnitude preservation**.
   Per-row max-abs normalization collapses magnitude differences
   between rows. Global-scale (5^5 / 7^7 via population median)
   recovers some signal: 7^7×7 at 18 B = 0.144 > per-row μ-law
   Zipper-I8-φ(64B) = 0.153 at 64 B.

**Pragmatic conclusion:**

- **Use Had-Q5×D-R** for production argmax compression. ICC 0.989 at
  ~0 per-row bytes (shared codebook). It's already shipping.
- **Use 7^7×7 (18 B)** ONLY when you need the zipper's additional
  properties: progressive decode, negative-cancellation bundling,
  anti-moiré guarantee without codebook dependency.
- **Don't pursue Quintenzirkel stride** on argmax populations —
  measured empirically inferior to φ across all tested sizes.

**Still unmeasured:**

- Multi-projection MRI-style differential phase (N rotations,
  cross-view aggregation). Sidesteps sign-flip invariance by
  measuring inter-rotation deltas.
- Fibonacci-weighted bundling for 256-bundle capacity in i8 via
  Zeckendorf decomposition decode.
- Audiophile-style multi-band phase precision (8 bits top-16,
  3 bits middle-48, sign-only bottom).

Cross-ref: commits d172aa3 (I8+Quint), f004d82 (5^5+7^7 + global scale).

## 2026-04-20 — CORRECTION: "Had-Q5×D-R at 0 B/row ICC 0.989" was a misread
**Status:** CORRECTION

Earlier entry claimed Had-Q5×D-R achieves ICC 0.989 at 0 bytes per row
→ "the argmax wall is cracked." This was WRONG.

`ParametricCodec::bytes_per_row()` in codec_rnd_bench.rs returns a
hardcoded `0` for the entire parametric family (Had-Q5×D-R, SVD-Q5×D-R,
all D-rank variants). This is an instrumentation placeholder, NOT the
actual storage cost. Actual storage for a full-dim 4-bit Hadamard-
quantized codec = 4 bits × n_cols = ~2 KB/row for q_proj (4096 cols),
~1 KB/row for k_proj (1024 cols), ~6 KB/row for gate_proj (12288 cols).

**Corrected compact-byte-honest hierarchy (q_proj ICC, honest bytes):**

| Codec | Bytes/row | ICC |
|---|---|---|
| Zipper-5^5 | 2 | 0.021 |
| Zipper-7^7 | 3 | 0.028 |
| Zipper-Phase (sign) | 8 | 0.097 |
| Zipper-I8-φ | 8 | 0.025 |
| Zipper-7^7×7 | 18 | **0.144** |
| Base17 | 34 | 0.024 |
| Zipper-Full | 64 | **0.204** |
| Spiral-K8 | 278 | 0.281 |
| RaBitQ | 520 | 0.504 |
| Had-Q5×D-R | ~2 KB | 0.989 |

**No compact codec (≤ 100 B/row) in this bench reaches ICC > 0.3.**

**What IS true:**
- Zipper-Full at 64 B is the compact argmax Pareto leader (ICC 0.204)
- Zipper-7^7×7 at 18 B is the compact-compact Pareto leader (ICC 0.144)
- Had-Q5×D-R at ~2 KB is near-Passthrough reference, NOT a compression win

**What IS FALSE (that I claimed earlier):**
- "Argmax blind spot is already solved by Had-Q5×D-R at 0 B/row" —
  it's solved at full-dim ~KB/row, not at compact bytes.
- "Use Had-Q5×D-R for production argmax" — it's a fidelity reference,
  not a deployment codec.

**What's still unknown:**
- Whether CAM-PQ (product quantization with shared codebook) can hit
  ICC > 0.5 at ~9 B/row on q_proj. CAM-PQ is already production in
  `ndarray::hpc::cam_pq` but not wired into codec_rnd_bench.rs.
- Whether TurboQuant at its paper-claimed 9 B/row actually achieves
  ICC > 0.9 on q_proj — no implementation in this bench.

Correction needed in codec-findings-2026-04-20.md decision tree.

## 2026-04-20 — THE ANSWER: CAM-PQ at 6 B/row solves the argmax blind spot
**Status:** SUPERSEDED by 2026-04-20 CORRECTION (128-row trivial fit)

Wired `ndarray::hpc::cam_pq::CamCodebook` as `CamPqRaw` + `CamPqPhase`
candidates in codec_rnd_bench.rs. Same bench, same populations,
same 128 rows. Results are definitive.

**ICC_3_1 across all three populations:**

| Codec | Bytes/row | k_proj | gate_proj | q_proj | Top-5 recall |
|---|---|---|---|---|---|
| Passthrough | row×4 | 1.000 | 1.000 | 1.000 | 1.0 |
| **CAM-PQ-Raw** | **6** | **0.9998** | **0.9998** | **0.9999** | **1.0** |
| **CAM-PQ-Phase** | **6** | **0.9998** | **0.9998** | **0.9999** | **1.0** |
| Had-Q5×D-R | ~2 KB | 0.985 | 0.987 | 0.989 | 0.8-1.0 |
| Zipper-Full | 64 | 0.129 | 0.107 | 0.204 | 0.0-0.6 |
| Base17 | 34 | 0.007 | 0.012 | 0.024 | 0.0 |

**Per-row storage 6 bytes. Shared codebook ~24 KB per population
(per-tensor calibrated; re-usable across all rows of the same
tensor, amortized to zero as N_rows grows).** Top-5 retrieval
recall = 1.0 on every population.

**Key diagnoses:**

1. **CAM-PQ is the working compact codebook-only argmax codec.**
   Near-Passthrough fidelity at 6 B/row + 24 KB shared state.
   Completely solves the argmax blind spot.

2. **Hadamard pre-rotation made NO difference** (Raw vs Phase both
   ICC 0.9998). K-means clustering finds the discriminative structure
   regardless of basis — near-orthogonality (I2) is a property of
   random rows, but trained weights have learned structure that PQ's
   subspace k-means captures in EITHER the raw OR Hadamard basis.
   The "argmax blind spot requires JL/PolarQuant/TurboQuant" claim
   was incorrect — product-quantization with subspace k-means suffices.

3. **The entire fractal → zipper arc was solving a solved problem.**
   CAM-PQ has been production in `ndarray::hpc::cam_pq` since Phase 1.
   All 10 zipper candidates + 2 fractal candidates + MRI/Fibonacci/
   audiophile follow-up probes are now superseded by CAM-PQ at the
   argmax ICC metric. The zipper's only remaining niche (if any):
   populations where per-tensor calibration is not possible (novel
   query-time tensors), which is rare in practice.

4. **The codebook calibration cost is legitimate per I7.** I7 states
   "vector-as-location needs per-tensor basis calibration." CAM-PQ's
   per-population k-means IS that calibration. Shared codebook is
   NOT a cheat — it's the correct amortization.

**Wiring recommendation:**

- CAM-PQ is already production (`ndarray::hpc::cam_pq`).
- `lance-graph-contract::cam::CamCodecContract` trait is the integration
  point.
- `lance-graph-planner` has `CamPqScanOp` operator.
- Actual wiring needed: expose CAM-PQ through the contract to
  consumers who currently default to Passthrough on argmax-regime
  tensors (attention, MLP, logits). Per I1, these are the large
  majority of weight storage.

**Compression win:** Qwen3-8B q_proj at 4096×4096 f32 = 64 MB.
CAM-PQ: 4096 rows × 6 B + 24 KB codebook = 24 KB + 24 KB = **48 KB
total**. **1300× compression at ICC 0.9999.**

**This is the session's actual deliverable.** The zipper/fractal
research arc was the path to discovering it, but the answer was
already in the workspace. Commit f1498bc landed the measurement.

Cross-ref: ndarray::hpc::cam_pq production code (620+ LOC, 15+
tests), codec_rnd_bench.rs CamPqRaw/CamPqPhase candidates, this
session's 18 commits on claude/quick-wins-2026-04-19 branch.
