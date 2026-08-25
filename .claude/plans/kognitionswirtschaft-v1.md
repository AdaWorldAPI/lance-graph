# Kognitionswirtschaft v1 — Maslow · Heckhausen/Rubikon · Libet as one cognitive economy

> **Status:** PROPOSED (unbuilt, unprobed). Operator-initiated 2026-08-25:
> *"maslov model Heckhausen Rubikon Libet ist Kognitionswirtschaft."*
> **Scope:** `lance-graph-contract` (one typed vocabulary) + `lance-graph-planner`
> (MUL wiring). No new crate, no new layer, no new planner.
> **Consumer story:** ada-rs consumes the agnostic types and keeps the Maslow
> narrative on its side of the Chinese wall — this document uses the name for
> orientation; **no contract type is named Maslow**.

---

## 0. Grounded facts (verified 2026-08-25, file:line)

Three of the four layers of this economy already exist in this repo. The plan
adds the fourth and types one prose field. Nothing else.

| layer | economic role | exists as | where |
|---|---|---|---|
| **Libet** | the **currency** — per-cycle thinking budget, priced in µs | `CycleBudget` — *"the per-cycle net thinking budget, in µs — the Libet window"*; carrier `KanbanMove::libet_offset_us`, anchor −550 000 µs | `lance-graph-planner/src/elevation/cycle.rs:5–77`, `contract::kanban` |
| **Heckhausen / Rubikon** | the **purchase point** — where deliberation becomes commitment | `rubicon_witness` (D-ACR-8): reads deliberative vs implemental mindset **from the focus mask alone**; read-only, `E-PROGRESSION-IS-EXISTENCE-NOT-COMMAND-1` respected | `contract/src/rubicon_witness.rs:1–43` |
| **MUL gate** | the **checkout** — Flow / Hold / Block at the deliberation boundary | `MulProvider::{assess, gate_check}`, `GateDecision::{Flow, Hold{reason}, Block{reason}}` | `contract/src/mul.rs:144–150, 190–193` |
| **Maslow (deficit typing)** | the **demand signal** — *what kind* of evidence is missing | **does not exist.** `Hold { reason: String }` — the hold ground is prose | this plan |

Ancestor source, verified against the operator's description:
`ada-consciousness/atoms/cognitive_maslow.py` (364 lines; byte-near copy in
bighorn). It states the doctrine this plan ports — *"Needs are preconditions,
not preferences. A verb invoked before its need is satisfied is pathological"*
(`:16–17`) — the five levels STIMULATION → SAFETY → RELATION → COMMITMENT →
NAVIGATION (`:51–63`), the pathology map hallucination / panic-thrash /
premature_closure / paralysis / rigidity (`:95–99`), and NAVIGATION gating
`think_about_thinking` + `freewill` (`:90`).

### F-0 — the finding that shapes the port: the crown never fired

`cognitive_maslow.py:207–208`:

```python
self.need_states[CognitiveNeed.NAVIGATION].satisfied = True   # hardcoded
self.need_states[CognitiveNeed.NAVIGATION].strength = 0.5
```

`get_current_need()` returns the lowest **unsatisfied** need; NAVIGATION is
unsatisfiable by construction, so the fifth deficit class is **unreachable
code** — it has never once been the diagnosis in the file's history.

The structural reason it *could not* be written matters more than the bug: the
four lower needs are **field-snapshot predicates** (`density > 0.1`,
`coherence > 0.3`, `coherence > 0.7 ∧ novelty < 0.3` — `:183–202`). Rigidity is
not a property of a snapshot. "Stuck in one mode" is only visible over a
**trajectory**: same style, no rung ascent, the same lower deficit hydrated
repeatedly without resolving. A meta-property is not decidable from one step —
the identical lesson ada-rs's K2 wave measured for `Motion::Breathing`. The
port therefore defines the fifth deficit as a **window predicate over the
resolution loop's own history**, or it ports four classes plus a label.

*(Candidate epiphany entry `E-THE-CROWN-NEVER-FIRED-1`; not appended to
`EPIPHANIES.md` here — council-gated, and this plan is its evidence file.)*

---

## 1. The economy, stated precisely

The four layers answer four different questions, and none can substitute for
another:

```
Piaget / rung          on which meta-height am I thinking?
Heckhausen / Rubikon   in which decision phase am I?           (rubicon_witness READS it)
Libet                  how much revocable budget remains?      (CycleBudget PRICES it)
Deficit typing         WHAT IS MISSING before I may proceed?   (this plan)
```

Kognitionswirtschaft is then literal, not metaphor: **Libet supplies the
currency, the deficit type is the price tag, MUL is the checkout, the Rubicon
is the completed purchase.** Today a `Hold` spends budget on "more context"
indiscriminately, because its ground is a `String` nothing can dispatch on.
The typed deficit makes the spend targeted: the resolver never loads *more*
context — it loads the **named kind** of missing evidence, and nothing else.

> Maslow does not choose the answer. It chooses the shape of the missing
> evidence.

The separation the gate gains, as a type distinction rather than a convention:

```
"I do not yet know enough."        →  Hold(deficit)   — a demand signal
"I know enough, but have not       →  deliberation continues; rubicon_witness
 yet committed."                       still reads deliberative
```

Two states that today collapse into one prose reason.

---

## 2. Deliverables

### D-KW-1 — the typed deficit vocabulary (contract)

One `#[repr(u8)]` enum in `contract::mul` (no new module; this **is** MUL
subject matter). Agnostic names — the levels are cognitive-science vocabulary;
only the "Maslow" brand stays on ada-rs's side of the wall:

| variant | precondition question | pathology when skipped (`:95–99`) | supply (what a resolver loads) |
|---|---|---|---|
| `Signal` | is there anything here at all? | hallucination | evidence, observations, relevant nodes |
| `Stability` | is this stable enough to reason about? | panic / thrash | counter-evidence, provenance, context |
| `Relation` | how do these things relate? | premature closure | missing edges, mediators, `part_of`/`is_a`, causal paths |
| `Warrant` | can I stand behind this? | paralysis | falsifiers, counterfactuals, uncertainty resolution |
| `Mobility` | can I change how I am thinking? | rigidity | different ClassView / ThinkingStyle / re-visit lower rung |

Lower four demand **world**; `Mobility` demands a change of **thinking** —
which is why (per F-0) its predicate has a different shape.

### D-KW-2 — the Hold gains a typed ground (contract, additive)

`GateDecision::Hold { reason: String }` is not broken — it is widened
**additively**, per I-LEGACY-API-FEATURE-GATED (no silent re-semantics of an
existing variant):

- `MulAssessment` gains `deficit: Option<EpistemicDeficit>` (default `None` —
  every existing constructor site compiles unchanged);
- `gate_check` populates it whenever it returns `Hold`;
- the prose `reason` stays, demoted to display text. The **selector** is the
  typed field; the string is never read to decide anything (the ada-rs
  F-K4-NO-PROSE-1 rule, applied at its origin).

### D-KW-3 — the four snapshot predicates (planner)

Port `:183–202` into `mul/` against `SituationInput`'s existing fields — the
mapping is nearly onto (`environment_stability`, `calibration_accuracy`,
`complexity_ratio`, `interdependency_density`, `allostatic_load` already carry
the needed signals). Thresholds are **hand-tuned and say so**, per
I-NOISE-FLOOR-JIRAK — no invented significance claims.

### D-KW-4 — the `Mobility` window predicate (planner)

The corrected fifth class, over resolution-loop history, not a snapshot:
`Mobility` is deficient when, within a window of N gate cycles, (a) the same
lower deficit was returned ≥ k times, **and** (b) the resolutions applied did
not clear it, **and** (c) the style/ClassView in use did not change. Below the
window, the honest answer is the step-local one — the classifier returns what
the window supports (the K2 `Breathing` discipline, verbatim).

### D-KW-5 — the resolver is a dispatch, not a service (planner)

One function: `deficit → resolution request` (the supply column of D-KW-1).
It **emits a typed request**; it does not fetch, does not hold state, does not
become an actor (`E-PROGRESSION-IS-EXISTENCE-NOT-COMMAND-1` — a
message-per-hold resolver is the deleted architecture). Consumers with data
access execute the request.

**Naming fence:** this repo already owns the word *hydration* —
`lance-graph-hydrate` is SoA→S3→Lance **data** hydration. The epistemic act is
named **supply/resolution** throughout; any doc using "hydrate" for the
epistemic act is wrong and gets corrected on sight.

### D-KW-6 — the gate ordering (planner wiring, smallest possible)

```
deliberation ──► MUL assess ──┬── deficit present ──► Hold(deficit) ──► targeted supply ──► re-assess
                              └── none ─────────────► rubicon_witness reads the crossing
                                                       └── CycleBudget prices what remains
```

MUL answers *whether the preconditions for commitment hold*; the Rubicon
witness answers *whether commitment happened*. Neither absorbs the other — the
separation in §1 is the deliverable, and D-KW-6 is only wiring order plus the
falsifier below. **Nothing here moves a kanban phase.** Reads only.

---

## 3. Falsifiers (each with can-fire AND can-stay-silent legs, non-trivial inputs)

- **F-KW-CLASSIFIER-1** — every one of the five classes must fire on some real
  `SituationInput`, and every one must stay silent on some other real input.
  A class that fires always or never is dead vocabulary (the 150/150
  `closed_class_guess` lesson; the 114/180-vs-144/180 dissent-channel
  precedent for the discriminating middle).
- **F-KW-MOBILITY-1** — `Mobility` must be **reachable** (F-0's dead code,
  fixed and proven by a constructed non-converging window) and must **not**
  fire on a long loop that is genuinely progressing (rung ascending, deficits
  clearing). Both legs on windows > 1 cycle.
- **F-KW-TARGETED-1** — **the falsifier the whole plan stands on.** Over a
  task set where deficits are induced deliberately, resolving the *named*
  deficit must clear the Hold in measurably fewer cycles than resolving a
  *uniformly random* deficit class (shuffled control). If targeted ≈ random,
  the taxonomy is decoration over "load more context" and D-KW-1..5 do not
  deserve to survive.
- **F-KW-SEPARATION-1** — there must exist inputs where all five deficits are
  satisfied **and** `rubicon_witness` still reads deliberative. If deficit
  clearance and Rubicon crossing always co-occur, the two-gate separation is
  one gate wearing two names.
- **F-KW-INERTNESS-1** — every ported threshold: raising it must silence
  something, lowering it must admit something (the `heel_threshold 50.0`
  lesson). A knob that moves nothing is decoration.
- **F-KW-NO-PROSE-1** — no code path reads `Hold.reason` (the `String`) to
  decide anything. Structural scan, `impl_only` pattern, proven with a
  **compiling** violation — a probe that fails to build has proven nothing
  (measured in ada-rs K4c: the first probe there broke `Copy` and was
  worthless).

---

## 4. Honest risks

- **The MUL already half-does this.** `DkPosition`, `TrustTexture`,
  `Homeostasis` carry diagnosis-shaped information. If D-KW-3's predicates end
  up re-deriving what `MulAssessment` already states, the right move is to
  *read* those fields, not to compute beside them — checked at implementation
  time, per field.
- **F-KW-TARGETED-1 needs induced deficits**, and induced ≠ natural. A
  synthetic corpus proves the machinery, not the taxonomy. The result is
  stated as exactly that until a live loop supplies natural Holds (the same
  boundary ada-rs drew for F-HELIX-CYCLICITY-1: running it on synthetic frames
  would be manufactured reachability).
- **Five classes may be the wrong carve.** The ancestor's own satisfaction
  logic collapses `Relation` into a coherence×density product — if
  F-KW-CLASSIFIER-1 shows two classes never separating on real input, merging
  them is a legitimate outcome and cheaper than defending the pyramid's shape.
- **This plan was written by the author of five false absence claims today.**
  Every "exists / does not exist" statement in §0 carries a file:line for
  that reason. Anything without one is conjecture and is labelled.
