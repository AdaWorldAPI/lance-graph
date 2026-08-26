## 2026-08-23 — E-TWO-KEY-ELEVATION-WINDOW-IS-NARROW-AND-THE-CORPUS-STRADDLES-IT-1 — one `RungElevator` actuator path is driven end-to-end; the two shipped rules that must agree for elevation overlap in only a ~0.125-wide band, and both negative controls fall on opposite sides of it

**Status:** FINDING (measured — `PROBE-REVISION-RUNG-ACTUATOR-1`, 19/19 gates
green). **Confidence:** High for the measurements; reproducible from the
commit. **Scope:** this probe establishes one concrete `RungElevator`
actuator path and measures the overlap of two existing heuristics around
that actuator. It does **not** establish a canonical metacognitive
controller.

**The actuator path, driven.** The chain receipt → membrane →
`InnerCouncil::from_signals` → `CollapseHint` → `mul::GateDecision` →
`KanbanColumn::advance_on_gate` → `MailboxSoaOwner::try_advance_phase`, plus
the rung arm via `RungElevator::apply_delta` — **its first caller outside the
type's own unit tests** (an `examples/` caller, not a production-path one).
The same correction applies to #998's `MailboxSoA::promote_family`
(`mailbox_soa.rs:829`): its exhaustive call census is unit tests plus one
example, never `src/` — "first production-path call" was wrong wording and is
retracted here. Everything after the membrane is shipped machinery; the only
novel logic is `signals_from(&CycleEvidence)`, pure measured ratios with no
tuned constants.

**THE HEADLINE — the two keys barely overlap.** Operator-pinned rule:
`CollapseHint::RungElevate` is qualitative INTENT, `rung_delta(emergence,
coherence)` decides whether the shift is EARNED; elevation requires both.
`measure_two_key_window` sweeps the task-unresolved axis at saturated
coverage and measures where they agree:

```
  task-unresolved  <0.316  → Balanced / Flow          (settle)
           0.316 … 0.600   → Catalyst / RungElevate,  rung_delta =  0
           0.600 … 0.725   → Catalyst / RungElevate,  rung_delta = +1  ← the ONLY window
           0.725 …         → Guardian / Fanout,       rung_delta = +1
```

**Both negative controls are real fixtures on opposite sides of the window,
and the second one the run found rather than the design anticipating it:**
- **G16** (mid stall, task-u 0.564): intent key OPEN (RungElevate), earned key
  CLOSED (delta 0) → held. RungElevate is not a magic elevator button.
- **G18** (deep stall, task-u 0.768): earned key OPEN (delta +1), intent key
  CLOSED (Fanout) → held. **The council refuses to deepen from overwhelming
  ignorance** — it asks for evidence instead. Guardian dominates Catalyst
  above u ≈ 0.727 regardless of how saturated coverage is.

**F14, stated as measured:** `Contextual → Analogical` changes the recipe set
returned by the shipped rung-dependent selection path (`Recipe::admissible_at`
/ `RungLevel::admissible_recipes`, consumed by the live
`StyleStrategy::recipes_for_at`) from **11 to 24**; the 13 added
Control-bucket kernels all fire on the measured stalled fixture, and CAS's
grid changes `[0,0,1,1] → [0.25,0.5,0.5,0.75]`. That is the whole result — it
is not translated into an epistemic or metacognitive claim.

**THE SHARPEST THING THE AUDIT FOUND — the Evaluation fork exists in the
type system and nothing can drive it.** `KanbanColumn` already has the shape
the architecture needs: `Evaluation → [Commit, Plan, Prune]`, with `Plan →
Planning` documented as *"re-plan: re-enter Planning carrying the witness
(the 'act differently next time' exit)"* (`kanban.rs:56-58,101-109`). So the
re-run is ALREADY represented without a central scheduler — as a DAG edge.
But `advance_on_gate` (`kanban.rs:146-153`), the only shipped lowering, is
degenerate at `Evaluation`: `Flow` takes the first non-`Prune` ⇒ **always
`Commit`**; `Block` ⇒ `Prune`; `Hold` ⇒ stay. **`Plan` is structurally
unreachable through the gate** (grep: it is a transition target only in
`#[cfg(test)]` and two examples, never production `src/`). The three-way
deliberative fork collapses to commit-or-veto. And nothing reaches
`Evaluation` anyway — `cognitive_pass` filters `phase() != CognitiveWork`
(`cycle_driver.rs:719`), and `shade_owner`, the only production caller of
`advance_on_gate`, is reachable only from that loop: **the `Evaluation →
{Commit, Plan, Prune}` decision has no shipped production caller.**

**A focus-of-attention representation EXISTS and must not be re-invented.**
`contract::attention_facet::{AttentionFocusFacet, RowFocusMask, FocusAxis}`
— *"Where focus landed or was projected"* (`attention_facet.rs:178`), breadth
as covered POPULATION not entry count — with `contract::rubicon_witness` as
its read-only instrument. Zero callers outside its own crate. Its constraint
is already written: *"It READS. It never moves anything… An overlay that
DRIVES a transition from a focus reading has rebuilt the scheduler this
substrate removed"* (`rubicon_witness.rs:26-31`). Its falsifier `D-ACR-8` is
two-sided and **queued, not run**. Unresolved placement question:
`rubicon_witness` measures the Heckhausen crossing at `Planning →
CognitiveWork`, while the pre-collapse deliberative surface in the target
architecture is `Evaluation → Commit` — two Rubicons or one mis-placed
label, unanswered.

**Also established:** `Commit` is DECLARED, not implemented (`calcify` is a
`todo!()`, D-ATOM-5), so the Rubicon's irreversibility today is the DAG
legality table, not calcification; `RubiconPhase` (`cognitive-compiler`) is a
SEPARATE enum in a scaffold crate with zero cross-references to
`KanbanColumn`; and no `src/` file mentions both a Revision symbol and
`KanbanColumn` — Revision does not touch the phase machine in production.

**Audit hygiene note:** one lane cited this probe's own header as
corroborating evidence for the absence of a focus mechanism. That is
circular — the header is this session's own text — and the citation was
struck. The absence claims above rest on call censuses, not on the probe's
self-description.

**NOT OBSERVED / OUT OF SCOPE.** This probe has no observer capable of
establishing problem-texture discrimination, resonance behavior, MUL
grounding behavior, or Frozen/Learned/Explore superposition. Its observation
surface does not contain them; this bounds the probe, and is not a result
about the architecture either way.

**Anti-conflation note.** The probe's `RungLevel` / `RungElevator` vocabulary
is unrelated to `lance_graph_planner::temporal`'s `QueryReference` /
`EpistemicMode`; no conversion or call path exists between them. Worth
stating only because both spell it "rung" — a census of every rung-named
surface found nine distinct vocabularies and zero conversions across this
pair (every `QueryReference::at` / `EpistemicMode::for_rung` call site passes
a literal or a local `const`).

**FOLLOW-UP OBSERVATION (recorded, not addressed).** The live driver
numerically materializes `RungLevel` into `ThoughtCtx.rung`
(`driver.rs:569-577` `elevator.on_gate(gate) as u8` → `driver.rs:978`
`ctx.rung = rung`). `ThoughtCtx` documents `1..=9`
(`recipe_kernels.rs:58-59`, default `1`) while `RungLevel` contains
`Surface = 0`, so `Surface` can be materialized as `0`. No semantic failure
was demonstrated here and this PR does not alter it; whether `0` has
behavioral consequences or is stale documentation is for a later falsifier.

**Behavioral learning: no production path exists.**
`ScaffoldCompiler::synthesize` returns `Err(CompileError::NotImplemented)`
unconditionally (`cognitive-compiler/src/lib.rs:155`), and `elixir-template`
/ `template-runtime` / `template-equivalence` / `cognitive-compiler` are all
in the root `Cargo.toml` `exclude` array. `witness_fabric::ForesightSample`
(`:1704`) is a correctly-shaped, hindsight-blind prediction-vs-outcome
primitive whose every caller is in its own `#[cfg(test)]` module — test-only,
not a live learning receipt. The honest claim is therefore *future*
behavioral learning will need equivalent prediction-time provenance; nothing
today can establish that property.

**Temporal placement (narrow).** `temporal.rs` = query-level admission of
historical knowledge, tested, no production caller. `witness_fabric` = a
SEPARATE shipped grounding mechanism whose hindsight discipline is enforced
by API shape (the outcome is unreachable from the call signature) — this is
the load-bearing one. temporal → Revision = BLOCKED / absent
(`counterfactual.rs:335-336`, D-ATOM-5 / D-PERSONA-5). `temporal.rs` does not
currently participate in the cognitive loop, and nothing in this probe
touches it.

**G19, the honest null:** this corpus's reachable exhaustion depths STRADDLE
the window (0.564 < [0.600, 0.725] < 0.768), so the earned-elevation arm
(F4/F6) is driven by a clearly-labelled synthetic evidence state at the
window midpoint. Everything downstream of `CycleEvidence` is the same shipped
path either way. Whether a ~0.125-wide window is the intended design or an
accident of two independently-authored rules is an OPEN QUESTION this probe
raises and does not answer — it is measured, not judged.

**A corrected premise (the self-falsifier fired, as designed).** The first
draft used {naked-only, naked+hidden-ALL-units} and asserted a deep
both-policy stall; `derive_both_stall` panicked. A sweep showed why:
naked+hidden-over-all-units solves EVERY uniqueness-preserving reduction of
the #997 puzzle (`hidden_stall` reachable set = `[0]`; only 7 clues are
removable before uniqueness breaks), so with that pair "the admitted family
is exhausted" is UNREACHABLE and the whole escalation arm has no receipt. The
family is now {naked-only, naked+hidden-ROWS-only} — a real, sound,
oracle-checked technique with a narrower scan scope — under which exhaustion
is reachable at task-u ∈ {0.564, 0.754, 0.768}.

**The membrane is task-normalized, deliberately the harsher reading.**
coherence = what the policy explained *of its own task* (cells empty at cycle
start), never *of the board*: "24 of 81 board cells" conflates the reasoning's
competence with the puzzle's generosity in givens; "24 of the 55 I was asked
to resolve" is a statement about the reasoning. It yields SMALLER coherence
than the board reading — the stricter choice, not the flattering one.

**F14's actuator is shipped, not invented.** No kernel `gate()` reads
`ctx.rung` (verified negative). The real rung-capability gate is
`Recipe::admissible_at` / `RungLevel::admissible_recipes` (monotone,
test-pinned, consumed by the live `StyleStrategy::recipes_for_at`): elevating
Contextual → Analogical takes admissible recipes **11 → 24**, and all 13
newly-admitted Control-bucket kernels (RTE HTD MCP CR LSI PSO CDI CWS SSR ETD
IDR SPP DTMF) actually FIRE on the stalled context — work refused at the prior
rung. Second arm: `Cas`'s shipped `hdr_level` grid re-quantizes the SAME field
`[0,0,1,1]` → `[0.25,0.5,0.5,0.75]` across the same boundary.

**F5's evidence-thin vs evidence-saturated split is visible in the log:** at
IDENTICAL unresolved count, coverage 14/31 → Fanout and coverage 19/31 →
RungElevate. The distinction the operator asked for is measured, not
hand-picked; Fanout terminates honestly when coverage saturates ("cannot
gather more" is a no-op, not a loop).

**CE64 non-interference is proven ACTIVELY, not by absence-of-import** (the
#998 approach, strengthened): a live edge carrying
`CausalTopology::IndirectUnknownIntermediates` + `ReasoningBand::Causal` is
lens-verified pre-flight, passed through the entire loop, and asserted
byte-identical after (F7/F8, raw `0x7000000000000000`). F9 goes further:
Direct vs IndirectUnknownIntermediates yield IDENTICAL verdicts — epistemic
topology does not secretly act as a scheduling trigger.

**Two-`GateDecision` trap, recorded:** `mul::GateDecision` (Flow/Hold/Block,
String reasons) feeds `advance_on_gate`; the UNRELATED
`collapse_gate::GateDecision` (byte struct) feeds `RungElevator::on_gate`.
Same name, same crate, no conversion exists. This probe uses the mul one for
phase and `apply_delta` for rung — navigating the trap rather than tripping it.

**Gates green (19/19):** F1 Flow-on-progress; F2/F15 triangle success →
elevation NEVER requested ("a different way of thinking" ≠ "a more powerful
level"); F3 unwarranted TCF singleton is no lever; F5 Fanout = breadth only,
phase held; G16/G18 the two negative controls; G19 the straddle; F4/F6 earned
elevation, rung only, triangle bytes identical; F14 the capability delta;
F7/F8/F9 CE64 untouched and not a trigger; F10 owner-locality; F11 read-only
witnesses; F13 observer-insufficiency → gather, never escalate; F12 full-run
determinism; G17 membrane monotonicity + regime discrimination.

**Next slice (named, not hidden):** repeated-elevation dynamics,
Evaluation→Commit calcify, and the long-run fate of a
held-with-two-key-disagreement owner — the owner that wants depth and has not
earned it currently stays held at full coverage forever.

