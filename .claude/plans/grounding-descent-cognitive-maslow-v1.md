# grounding-descent-cognitive-maslow-v1 — when a high rung is handed a sloppy joe, descend the needs ladder

**Status:** PROPOSED (unbuilt; every mechanism cited exists, only the loop is new)
**Date:** 2026-08-26
**Confidence:** HIGH on the trigger (it is one comparison over shipped types); MED on the
descent order (the Maslow carve is the port source's, unprobed on real input); the
`Mobility` leg inherits F-0's CONJECTURE status from `kognitionswirtschaft-v1`.
**Predecessors:** `rung-ladder-grounding-v1` (the RungShift triggers this plan listens to),
`rung-mul-grounding-v1` (the SPO 2³ screening-off used at the `Relation` step),
`kognitionswirtschaft-v1` (⊘ NOT CANONICAL — read as the sketch of the deficit
vocabulary only; its citations are not trusted here, everything below is re-verified),
D-ATOM-4 (`counterfactual.rs` — the sandbox lane the `Warrant` step uses).

---

## 0. The one sentence

**An elevation that does not lower free energy is an indictment of the ground it
stands on** — so on `RungShift +1` with `ΔF ≥ 0`, descend the cognitive-needs
ladder bottom-up, checking the revision history and the observations at each
level, and either revise or certify the ground clean.

Rungs 4–9 (`Abstract..Transcendent`) consume what rungs 0–3 assembled
(0–1 = observation, 2 = the 144 verb atoms, 3 = the 34 NARS recipes — the
operator-ruled rung-content ladder, `persona-vs-rung-ladder.md`). When the high
rung's work stalls, the fault is *usually below it* — but asserting that without
a measurement is the oracle-invention `witness_fabric` explicitly refuses. The
descent makes it a measurement.

## 1. The trigger — measurable, not felt

"Rungs 4–10 feel they were handed a sloppy joe" must be a number or it recurses
into asking a model how it feels (Goldstandard: no model-mediated value gate).

The number exists: `FreeEnergy::compose(likelihood, kl)` →
`total = (1 − likelihood) + kl` (`grammar/free_energy.rs:108–116`).
Elevation exists: `RungShift` `evaluate_shift` fires `+1` on
`sustained_block ∨ predictive_failure ∨ structural_mismatch`
(`rung-ladder-grounding-v1` §2, grounded).

```
on rung r → r+1:
    F_before = F at r  (last cycle before the shift)
    F_after  = F at r+1 (first settled cycle after)
    ΔF = F_after − F_before

    ΔF < −EPIPHANY_MARGIN (0.05)  → climb paid; no descent
    |ΔF| ≤ EPIPHANY_MARGIN        → climb bought nothing → DESCEND
    ΔF > 0                        → climb made it worse  → DESCEND
```

Reusing `EPIPHANY_MARGIN` is deliberate: the same constant that recognises an
epiphany (`ΔF < 0.05` on the Click ladder) recognises its absence. One knob,
already shipped, already meaning "the smallest ΔF that counts".

**Anti-vacuity guard (P0 falsifiability rule, both legs mandatory):** the
trigger must fire on a stalled climb AND stay silent on a paying climb, on
non-trivial input. A descent that fires on every elevation carries as much
information as one that never fires — it would *become* the eigenvalue
(`E-ANTI-EIGENVALUE-...-1`).

## 2. Two homeostases — the homonym this plan must not inherit

Audit of `kognitionswirtschaft-v1` ("check the references for homeostasis"):

| concept | type / const | where | role |
|---|---|---|---|
| **Flow homeostasis** (Csikszentmihalyi) | `mul::Homeostasis { flow_state, allostatic_load }` | `mul.rs:113–118`; fed by `SituationInput.allostatic_load` (`:20`) | gates whether you may **climb** (MUL checkout) |
| **F homeostasis** (Friston) | `HOMEOSTASIS_FLOOR = 0.2`, `is_homeostatic()` | `grammar/free_energy.rs:31,121`; deliberately mirrored at `materialize.rs:44` (`rested` at `:184,211`) | decides whether the climb **worked** |

The 1033 plan's citations are *accurate but half-blind*: its §4 names only
`mul::Homeostasis` and never connects to the F-floor — yet the F-floor is the
homeostasis that drives Commit/Epiphany/FailureTicket (the Click:
`F < 0.2` commit, `ΔF < 0.05` epiphany, `F > 0.8` ticket). This plan uses
**both, in their own lanes**: flow-homeostasis as a *precondition* (do not start
a descent while `FlowState::Anxiety` + high allostatic load — that is recovery's
job, "IF depleted → RECOVER first"), F-homeostasis as the *trigger* (§1).
Conflating them re-creates the two-`GateDecision` defect one floor up.

## 3. The descent ladder — cognitive Maslow, walked bottom-up

The port source's five needs (`ada-consciousness/atoms/cognitive_maslow.py:48–66`)
are not five deficit *labels* (the 1033 framing) — walked in order they are a
**descent protocol**: at each level, one question, one grounded check, one
existing mechanism. Names stay agnostic in-contract (Chinese wall; "Maslow"
lives on ada-rs's side).

| step | need (py) | question | grounded check | existing machinery |
|---|---|---|---|---|
| G1 | Signal (`STIMULATION`) | is there anything here at all? | the focal's observations resolve to real stream positions | `WitnessLens::at` + `CausalWitnessFacet::resolves_to`; `resolve_chain_lens` walks to rungs 0–1 |
| G2 | Stability (`SAFETY`) | is this stable enough to reason about? | revision history shape: did we abandon a value the history keeps returning to? | `belief_runs` → `suggest_reopening` (`Reverted`, `LongStableThenBrief`) |
| G3 | Relation (`RELATION`) | how do these things relate? | screening-off over the SPO 2³: is the S→O the high rung leans on spurious/mediated? | `rung-mul-grounding-v1` §0 (the 8-projection decomposition) |
| G4 | Warrant (`COMMITMENT`) | can I stand behind this? | run the road-not-taken: does the minority pole produce lower F? | `counterfactual.rs`: `CounterfactualMailbox::poll` → `minority_wins` → `revise_if_minority_wins` |
| G5 | Mobility (`NAVIGATION`) | is the *thinking* the problem? | only reachable if G1–G4 all came back clean: the ground is certified, the style is the suspect | ThinkingStyle/ClassView switch; window predicate per F-0 (trajectory, not snapshot) |

Three properties, each load-bearing:

- **Bottom-up, short-circuit.** The first dirty level is the diagnosis; deeper
  checks are not run (their results would be conditioned on a known-bad floor).
  This is `get_current_need()`'s lowest-unsatisfied semantics, kept — with F-0's
  bug fixed by *construction*: G5 is reachable, but only through four clean
  gates, so it cannot be the hardcoded-true crown.
- **Clean is a finding.** G1–G4 all clean → the sloppy joe was not the input.
  Certify the ground (a `WisdomMarker`-shaped residue, not silence) and route to
  G5. An audit that can only ever blame the input is the guard that cannot stay
  silent.
- **The oracle stays consumer-side.** `witness_fabric`'s own doc prescribes the
  shape: *"a consumer-side function taking `(belief_runs(.., upto),
  outcome_at(upto))`, which keeps the oracle outside the contract and the
  read-as-of bound intact."* `ΔF` **is** `outcome_at`. G2 is exactly that
  function, finally with its second argument.

## 4. Deliverables

| id | what | where | falsifier (both legs, non-trivial input) |
|---|---|---|---|
| D-GDM-1 | `outcome_at`: ΔF-at-shift measurement, `ElevationOutcome { rung_from, rung_to, d_f }` | planner (consumer side — NOT contract, per §3) | fires on a stalled synthetic climb; silent on a paying one; inert-knob test on `EPIPHANY_MARGIN` (raising must silence, lowering must admit) |
| D-GDM-2 | the descent walker G1→G5, short-circuit, flow-homeostasis precondition | planner, over root-exported `witness_fabric` fns | each level: one seeded-dirty fixture it catches, one clean fixture it passes; G5 unreachable while any of G1–G4 dirty |
| D-GDM-3 | `GroundCertificate` — the clean-audit residue | contract (it is a rung-elevated derivation: cross-input, new kind — `ELEVATED` under the zero-copy law, name the rung) | certificate only after 4 clean legs; a dirty G2 fixture must make certification impossible |
| D-GDM-4 | G2 wiring: `suggest_reopening` + ΔF outcome → `awareness.revise` / `InferenceType::Revision` | planner | a `Reverted` history + stalled ΔF yields a revision request; same history + paying ΔF yields none |
| D-GDM-5 | distribution probe: over real (not induced) stalled climbs, which level is dirtiest? | probe example | if G5 ≥ 80% of diagnoses, the ladder is a default with extra steps — merge/re-carve (the same legitimacy `kognitionswirtschaft-v1` §4 grants its own five classes) |

## 5. Deliberately NOT in this plan

- **No CE64 band/topology writes.** The first band producer trips the
  `temporal()`-sort trap #970 documented (`evidence_trail` decomposes 52..63;
  a written band is the dominant sort term), and the truth field is inside the
  same window. Descent runs on in-memory state + the witness registers.
- **No new gate state.** `Revise` is the *outcome* of a descent
  (`revise_if_minority_wins`, `awareness.revise`), never a fourth
  `GateDecision` variant — the discriminants are asserted by deepnsm-v2 and
  mirrored by ndarray (`ISS-GATEDECISION-ORDINAL-COLLISION`).
- **No actor, no fetch.** The walker emits typed requests and reads; it never
  becomes a mailbox-per-descent (`E-PROGRESSION-IS-EXISTENCE-NOT-COMMAND-1`).
- **No model-mediated "feels sloppy".** The trigger is §1's comparison, or the
  plan is dead.

## 6. Open questions

- **OQ-1:** `F_after` needs "first *settled* cycle after the shift" — settled by
  what count? Candidate: the `pMetricWindow = 5` already in RungShift, one
  window. Hand-tuned; must say so (I-NOISE-FLOOR-JIRAK).
- **OQ-2:** does a `GroundCertificate` expire? A certified ground plus new
  observations at rungs 0–1 is stale by construction. Candidate: certificate
  carries the stream length at issue; longer stream → void.
- **OQ-3:** G3 needs `all_projections` de-grounded first (D-RUNG-MUL-1, the
  cube-metric misreading). If that ships late, G3 degrades to edge-existence
  checks on the `part_of:is_a` rails — stated, not hidden.

---

## 7. Ratified sharpenings (2026-08-26, same day — session review of the CE64 thread)

Appended after review of the full arc (#970 → the epiphany-detector thread →
this plan). Ratified direction; the base sections above stand, these bind them.

### 7a. The three-axis separation is final

```
RungLevel 0..9        LIVE cognitive process — rung_delta → apply_delta.
                      Elevation happens HERE. Has a feeder today.
CausalTopology 59..60 DURABLE causal-path fact. No in-tree producer today.
ReasoningBand 61..63  DURABLE edge reading, 8 states. NOT the rung ladder
                      (8 states cannot carry 10 rungs; shared names sit at
                      different ordinals — #970 refused any *Rung* name).
```

The old thread's error was representation-before-generator: hunting CE64 for
the carrier of "associated with → explains" before noticing the live
elevation axis already had a feeder. `Topology 2 → 1` remains a real epistemic
event (a mediator became known — a change in causal KNOWLEDGE); `Rung 5 → 6`
is a change in thinking MODE. They may meet causally; they are not one axis.

**One-line form:** CE64 records what became known. RungLevel records how far
thinking climbed. Free energy tells whether the climb earned its height.

### 7b. RungLevel is an ordinal elevation axis, NOT ten dimensions

Ten *states* on one ordinal ladder (Surface 0 … Transcendent 9), never a 10-D
vector. This is a standing trap: the 33-TSV already groups nine Rung dims as a
dimension-group, so the vectorising misread is one refactor away. Anything that
sums, averages, or dot-products rung values has left the semantics.

Corollary: **rung 9 is not truth.** Transcendent is the highest reflection
height, and can be built entirely on sand — which is exactly when §1's trigger
sends it back down to Signal and the witnesses. Higher ≠ righter is what
separates this ladder from a pyramid.

### 7c. Provenance rule for any future topology projection

Derivation lives in witness / revision / evidence, where every step keeps its
provenance and stays addressable. Bits 59–60 may carry a *projection of the
result* — only under the #970 provenance contract (source-authoritative begins
with an explicit producer; nothing derives it). A traversal-derived `1` written
into the same two bits as a corpus-asserted `1` is byte-identical and
epistemically different — the second-truth shape this whole arc is allergic to.
So the parent-topology inheritance rule (`parent + part_of:is_a rail → child`),
attractive as it is, does NOT get to write 59–60; it lives witness-side, and a
durable projection is a separate, gated act.

### 7d. Dirty ≠ diagnostic — the descent must re-measure after repair

§3's short-circuit stops at the first dirty level. The negative finding
(`Topology 2 → 1` occurred, ΔF did not fall) proves a level can be repairably
dirty and still not be the missing grounding component. Therefore the walker
loops:

```
descend → first dirty level → repair/supply → re-measure ΔF
    ΔF falls   → done; that WAS the missing ground (positive finding)
    ΔF flat    → record "repaired but not the deficit" (negative finding,
                 kept — it prunes the search for every later descent)
               → CONTINUE descending from the next level
all levels clean/repaired, ΔF still flat → GroundCertificate + route to G5
```

Without this, the walker halts at the first *plausible* cause rather than the
first *causal* one — the System-1 shortcut, mechanized. The negative findings
are first-class output: "the mediator was not what this thought was missing"
is knowledge, and D-GDM-5's distribution probe should count them.

### 7e. Scope pin: the other 0..9 stack stays out

The `L0 Substrate → L9 Crystallization` stack is not this ladder and does not
enter §3. If it is ever wired in, it is a second, separate 0..9 axis
(processing stage), and conflating the two would be the homeostasis/GateDecision
homonym failure a third time.

### 7g. Styles are a per-stratum axis, not the fourth floor (operator-ruled 2026-08-26)

The rung-4 anchoring of `StyleFamily` predates the tower: it comes from the
era when `RungLevel` was one scalar position and no alpha layer existed — the
same world-model the NOT-CANONICAL 1033 plan was written in. With ten rung
instances, no instantiated stratum may be operationally idle because the old
model had a single active position.

So the demarcation sharpens once more:

```
content ladder (0–3)   hierarchy of OBJECTS      observations / verb atoms / recipes
style                  per-stratum AXIS           frozen × learned × exploration,
                                                  chosen PER instantiated stratum
selector input         that stratum's ΔF bit      paying → frozen templates
                                                  stalled → exploration
                                                  between → learned revision
```

Style-per-stratum is the response dimension of §7f's masked ΔF vector — one
design, two faces: the vector says WHERE it stalls, the style says HOW that
stratum answers. Falsifier: two instantiated strata must hold two different
styles simultaneously, switchable independently; tower-global style state is
the scalar era rebuilt and fails.

Board: `E-STYLES-ANCHOR-AT-RUNG-4-IS-A-SCALAR-ERA-ARTIFACT-...-1`;
demarcation: `persona-vs-rung-ladder.md` O9 (content rows 0–3 untouched;
persona-36 storyline untouched).
