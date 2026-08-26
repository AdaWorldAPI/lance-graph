# Plan: entropy measures closure; bits 59–60 tell whether the closure has causal footing (`entropy-closure-causal-ground-v1`)

> **Status:** PROPOSAL (measured, unbuilt) — 2026-08-26. PLAN/BOARD ONLY.
> **Companion to:** `.claude/plans/mul-calibration-not-verdict-v1.md` (the axes),
> `.claude/plans/grounding-descent-cognitive-maslow-v1.md` (the descent),
> `.claude/plans/dacr7-band-reading-contract-v1.md` (the 59..63 reading contract).
> **Law under test:**
> *Entropy tells cognition how closed its current model is. CausalEdge64 59–60
> tell it whether that closure is causally earned. From their disagreement
> emerges epistemic humility — no policeman inside the system.*

Two orthogonal, complementary mechanisms; never collapsed into one signal:

- **A. Entropy** — the organization of the current cognitive field. LOW H ≠
  knowledge; HIGH H ≠ ignorance. The regulator is bidirectional (introspection
  H↓ / exploration H↑; regulate, never minimize — ratified as
  `mul-calibration-not-verdict-v1` §8).
- **B. CE64 bits 59–60** — durable causal-epistemic topology at the edge level.

---

## 1. Exact measured semantics of bits 59–60 (Q1)

The 2-bit field carries **two shipped readings**, and *which one the producer
wrote is not recoverable from the bits* — that is the whole point of the
council-ratified **D-ACR-7 band-reading contract** (`band_reading.rs`):

**Lens 1 — `causal_edge::layout::CausalTopology`** (`layout.rs:239`):

| ord | variant | epistemic meaning |
|---|---|---|
| 0 | `Direct` | direct causal edge, no intermediates — **known structure** |
| 1 | `IndirectKnownIntermediates` | indirect, intermediates named — **known structure** |
| 2 | `IndirectUnknownIntermediates` | indirect, intermediates unknown — **projected structure** |
| 3 | `Unknown` | topology not established — **unresolved causal hole** |

**Lens 2 — `causal_edge::layout::TrustTexture`** (`layout.rs:141`):
`Crystalline=0 / Solid=1 / Fuzzy=2 / Murky=3` — ordinal-identical by design
(each `CausalTopology` doc names its `TrustTexture` twin).

**Answer to "do 59–60 already encode the distinction?": YES, under lens 1** —
`Direct`/`IndirectKnown` = known, `IndirectUnknown` = projected, `Unknown` =
hole. **No new bits are needed.** Three binding guards from D-ACR-7:

1. The reading is **declared per `(classid, rail)`** (`ClassView::band_reading`,
   total lookup, zero-fallback) and **projection is fallible** — a lens
   mismatch FAILS, never returns a plausible value.
2. **Provenance doctrine**: under v1 layout these were `temporal` bits;
   `CausalEdgeV3::from_v1` raw-copies them, so the v1 taint reaches V3 through
   the lift. `EdgeProvenance::V3Register` is an *assertion*, `Unknown`
   **refuses**. A causal-ground reading off an unasserted register is exactly
   the "plausible wrong answer" the contract exists to prevent.
3. Both carriers (`CausalEdge64` 59-60 / `CausalEdgeV3` byte-8 hi-2) serve the
   same projection on raw ordinals.

So the durable representation exists; what gates its use as "ground" is the
declaration + provenance, not the bit width.

## 2. Current entropy-related surfaces (Q3) — plural, uncoordinated

| surface | what it measures | file |
|---|---|---|
| `PerturbationDto::entropy()` | Σ −e·ln e over the codebook energy field | `thinking-engine/dto.rs:113` |
| `entropy_std(hits)` | Shannon over normalized `ShaderHit.resonance` | `cognitive-shader-driver/driver.rs:922` |
| `Distribution::entropy()` | Shannon over the INT4 histogram | `lance-graph-cognitive/search/distribution.rs:158` |
| `compute_entropy(popcounts)` | container-word popcount entropy | `spectroscopy/features.rs:95` |
| `confidence_entropy(arena)` | NARS confidence spread over a belief arena | `planner/nars/insight.rs:180` |
| `FreeEnergy` / ΔF | the drive (HOMEOSTASIS_FLOOR 0.2, EPIPHANY_MARGIN 0.05) | `contract/grammar/free_energy.rs` |

**Finding:** there is no single canonical H — each layer measures its own field.
That is acceptable (the fields differ) but the *closure* question must not be
answered by whichever entropy is nearest; see §3.

## 3. The cross-product ALREADY EXISTS: `SettlementCell` (Q4, "what exists vs hypothesis")

`contract::settlement` is the operator matrix, shipped, with a sharper
decomposition than the prompt's:

| operator quadrant | settlement cell (`settlement.rs`) |
|---|---|
| ENTROPY low × ground strong → *earned closure, preserve* | **`Crystal`** — settled and deserved |
| ENTROPY low × ground weak → *suspicious closure, explore* | **`Glass`** — dense closure on thin evidence, **the dangerous cell** |
| ENTROPY high × ground strong → *genuine complexity, introspect* | **`GroundedUnresolved`** |
| ENTROPY high × ground weak → *noise, acquire means* | **`Fog`** |

With settlement's own recorded correction, which this plan adopts verbatim:
**entropy is a THIRD signal, not one of the two axes.** The axes are **closure
density** (structural completeness — how much of what could be derived has
been) × **evidence competence** (the deepnsm-v2 `1 − U` reading). Entropy and
eigenvalue concentration *refine* the cell (`Glass + low entropy + high
eigenvalue = confidently calcified monoculture`,
`is_calcified_monoculture()`), they never define it. So the prompt's
"ENTROPY low" row is, precisely, "closure high (entropy-refined)". Scalar
collapse is structurally refused: no `glass_gap()` exists on purpose, and
`SettlementScope`/`comparable_to` refuse cross-scope subtraction.

**The hypothesis half (NOT built):** the per-edge feed from 59–60 into the
competence axis. Today competence comes from basin width
(`deepnsm-v2/basin.rs:62`, `1 − width/max_width`) — a *breadth* reading. A
`CausalTopology` census over a basin's edges (share of `Direct`/`IndirectKnown`
vs `IndirectUnknown` vs `Unknown`, projected fallibly through `band_reading`
with asserted provenance) would ground competence *causally*, per edge, using
only shipped types. That census is D-ECG-2 and it is the plan's only new
mechanism.

## 4. The self-constraint ladder emerges from the cross product (Q5, Q8)

No humility Boolean, no homunculus. The five states map onto shipped machinery:

| state | expression in current types |
|---|---|
| **I KNOW** | `Crystal` (closure high + competence high; 59-60 census strong) |
| **I DO NOT KNOW** | `Fog` / `GroundedUnresolved`, or `Glass` detected |
| **I KNOW THAT I DO NOT KNOW** | a *stable* `Unknown`-topology edge under a declared reading — the hole is represented, durably, at the edge |
| **I LACK THE MEANS** | the gap persists through Sandbox = Counterfactual + Revision (`revise_if_minority_wins` returns no winner; `suggest_reopening` exhausted) |
| **I NEED TO ASK / ACQUIRE MEANS** | the grounding-descent ladder (G1 Signal → G5 Mobility, `grounding-descent-cognitive-maslow-v1`): a persistent dirty level names the missing supply |

Routing without an execution gate (Q5): `Glass` → reopen via Sandbox
(counterfactual + revision), widen via the T9 actuators (`FieldModulation`
row selection, `WeightingKernel` switch) — all adaptation surfaces, none of
them `advance_on_gate`. The kanban gate stays untouched.

## 4b. Constraint-directed epistemic completion: the Sudoku walker (operator-ratified, 2026-08-26)

Entropy does not just regulate introspection vs exploration — it becomes the
walker's **search pressure toward epistemic holes**. The decisive case is the
third one:

```text
known                     A ─────→ B
unknown                   no useful structure at all
indirectly-known unknown  A ─────→ ? ─────→ C     ← something must fit here
```

**The hole has a shape before it has an identity.** The surrounding graph
already constrains what can occupy it — type/ontology constraints, temporal
ordering, truth values, known neighbouring causes/effects, counterfactual
compatibility. Filling it is solving a sudoku square, not generating prose.

### Three-layer division of labour

```text
ENTROPY        WHERE the field is suspicious / unresolved
CE64 59..60    WHAT kind of causal-topological hole it is
CE64 61..63    HOW a candidate may be admitted — the epistemic
               PERMISSION LEVEL, never a confidence float
```

**Guard on 61–63 (binding):** `ReasoningBand` is a reasoning-level encoding
(`Surface=0 … Causal=3, Counterfactual=4, …` — `layout.rs:353`), and it stays
one. It gates *what kind of bridge may cross the hole* — exploratory /
inferred / sufficiently grounded / known — it never silently becomes
`confidence = 0.83`. Scalar confidence lives in the NARS `(f,c)` truth value,
where it already is; the band is a permission tier over the packed CE64.

### The walker's decision table (entropy × ground, locally)

```text
stable + grounded                    → continue normally
stable + causally under-grounded     → suspicious closure → seek missing mediator
high entropy + constrained causal
  neighbourhood                      → search candidate basin
high entropy + no constraints        → acquire means / evidence instead
```

(The four rows are §3's cells read as walking policy: Crystal / Glass /
GroundedUnresolved / Fog.)

### The missing-link loop

```text
DETECT HOLE      entropy / topology disagreement
BOUND HOLE       59..60 + surrounding graph constraints
PROPOSE          walker over the resident graph → candidate set {x1, x2, …}
FILTER           TruthValue + ontology + temporal constraints
GATE             61..63 reasoning band (permission level)
TEST             Counterfactual + Revision (T10: Sandbox := CF + Revision)
ACCEPT           revision lands; topology upgrades
                 (Unknown → IndirectKnown → Direct-equivalent chains)
KEEP UNKNOWN     the hole stays REPRESENTED — the durable
                 "I know that I don't know" of §4
ASK FOR MEANS    KNOWN UNKNOWN + INSUFFICIENT MEANS → grounding descent
```

The negative test is the sharp one: **counterfactual removal.** For a candidate
`A → X → C`, remove `X` in the counterfactual lane — if explanatory coherence
collapses, `X` is a genuine mediator candidate rather than a convenient
association. A fill accepted without surviving removal is Glass at edge
granularity.

**"Unknown" never forces guessing.** Sometimes the neighbouring digits solve
the square; sometimes they only narrow `?` from 60,000 possibilities to 7 —
*that narrowing is itself epistemic structure worth persisting*; sometimes the
correct outcome is `KNOWN UNKNOWN + INSUFFICIENT MEANS`, which routes to the
descent ladder instead of a fabricated edge.

### The wider law

> **Entropy finds the holes. Causal topology gives the holes shape. The
> reasoning band controls what kind of bridge may cross them. Counterfactual +
> Revision tests whether the bridge actually carries explanatory weight.**

This turns the walker from graph traversal into constraint-directed epistemic
completion. Same D-ACR-7 guards throughout: readings declared per class,
projection fallible, unasserted registers refuse — an unknown-provenance band
must not gate anything.

Deliverable **D-ECG-6** — walker steering spec per the loop above
(census-ranked frontier: prefer `IndirectKnown` > `IndirectUnknown` >
`Unknown` for tractability; fills through the counterfactual lane only;
band-gated acceptance; revision as the only write-back; candidate-set
narrowing persisted even when unresolved). Falsifiers: **F-ECG-6** — a
`Surface`/`Association`-band fill must NOT close a causal-topology hole and a
`Causal`+-band fill must (both present in corpus), else the band gate is
decorative; **F-ECG-7** — a candidate that fails counterfactual removal must
not be accepted even when it passes every static filter (the removal test must
be able to veto), and at least one corpus candidate must fail exactly there.

## 5. TruthValue and MUL/DK placement (Q6, Q7)

- **TruthValue (NARS f,c) is complementary, not duplicative** of 59–60: `(f,c)`
  measures *evidential support strength* on a statement; `CausalTopology`
  measures *shape of the causal path* (direct / mediated / mediated-blind /
  absent). A high-confidence edge can still be `IndirectUnknownIntermediates` —
  that is precisely the "projected" middle case, and collapsing the two would
  delete it. `confidence_entropy` (insight.rs) already reads the (f,c) spread
  as its own signal.
- **MUL/DK already approximates `apparent certainty − demonstrated ground`**,
  literally: contract `TrustTexture::Overconfident` = "felt >> demonstrated"
  (`mul.rs:85`), ada-side predicate `felt > demonstrated + 0.3`, and
  `is_unskilled_overconfident()` = MountStupid. The DK quadrant of §4's matrix
  (closure high + felt certainty high + ground weak = **Glass with a confident
  reporter**) is where MUL's veto belongs: *low H → inspect ground → only then
  may convergence be earned.* MUL stays a calibrator; the domain gates stay
  domain gates (`mul-calibration-not-verdict-v1` §0 unchanged).

## 6. Smallest architecture correction (deliverables)

Nothing structural is missing. The correction is a wiring doctrine plus one
census:

| D-id | Deliverable | Gate |
|---|---|---|
| D-ECG-1 | doctrine: closure questions are answered by `SettlementCell`, never by a nearest-entropy scalar; entropy/eigenvalue refine only | doc-first |
| D-ECG-2 | `CausalTopology` census per basin (share known / projected / hole), projected **fallibly** via `band_reading` with asserted provenance, feeding the competence axis beside `1 − width/max_width` | F-ECG-1..3 |
| D-ECG-3 | `Glass` routing: detected Glass → Sandbox (counterfactual + revision) + T9 actuator widening; never an execution gate | F-ECG-4 |
| D-ECG-4 | the five-state ladder documented onto its existing carriers (§4 table), incl. "stable Unknown edge = I know that I don't know" | doc-first |
| D-ECG-5 | provenance discipline: any 59-60 ground reading from an unasserted register REFUSES (already the D-ACR-7 rule; restated as binding on this plan's consumers) | F-ECG-5 |
| D-ECG-6 | Sudoku-walker steering spec (§4b loop): census-ranked frontier, counterfactual-lane fills, 61-63 permission-band acceptance, revision write-back, narrowing persisted | F-ECG-6, F-ECG-7 |

## 7. Falsifiers — earned vs information-poor low entropy

| id | falsifier | fails when |
|---|---|---|
| F-ECG-1 | construct two basins with IDENTICAL low field entropy, one with `Direct`-dominant census, one `Unknown`-dominant; they must land in different cells (Crystal vs Glass) and route differently | the system cannot tell earned from poor closure — entropy is being read as mastery |
| F-ECG-2 | the census must fail-closed: same basin, provenance `Unknown` → projection refuses; no cell is produced from unreadable ground | a plausible cell emerges from tainted bits |
| F-ECG-3 | anti-vacuity: the census discriminates on a real corpus — not 100% one class (can-fire AND can-stay-silent, per the falsifiability rule) | census fires on everything or nothing |
| F-ECG-4 | Glass routing reaches counterfactual+revision and an actuator change, and is observable NOT to touch `advance_on_gate` | humility became an execution gate after all |
| F-ECG-5 | a v1-lifted register with nonzero legacy temporal bits must NOT read as a valid topology | the v1 trap leaks through the lift into "ground" |
| F-ECG-6 | a Surface/Association-band fill must not close a causal-topology hole; a Causal+-band fill must (both cases present in corpus) | the band gate is decorative |
| F-ECG-7 | a candidate passing all static filters but failing counterfactual removal is rejected; at least one corpus candidate fails exactly there | the removal test cannot veto — it is ceremony |

## 8. Explicitly out of scope

- No new bits, no new DTO, no confidence scalar, no `glass_gap()`.
- No change to `advance_on_gate` / kanban / CollapseGate.
- The choice among the six entropy surfaces stays per-layer; D-ECG-1 only
  forbids using any of them AS the closure axis.
