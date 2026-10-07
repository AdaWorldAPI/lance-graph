# 2026-10-07 — D-PEARL-IO-0: Pearl's ladder executed in and out of CausalEdge64

**Probe:** `cognitive-shader-driver/examples/pearl_ladder_probe.rs`
(`--features with-planner`), over `examples/shared/certification_model.rs`
(the P7a model, moved unchanged) and `lance_graph_planner::dismech_counterfactual`.
**Status:** TEST-PINNED (14 tests; 12 disable runs, one per falsifier, all red).

## Question

Can an edge enter with one `EpistemicState5`, run the operator its Pearl
projection selects, and leave with a different state only because the
measurement earned it?

## Inventory (before code)

| need | existing | verdict |
|---|---|---|
| Pearl input | `CausalMask`, bits 40..42; `simpsons_paradox_risk` | used; nothing dispatched on it before |
| observational / intervention obligations | P7a `Model` folds (`relational_certification_probe`, #1369) | moved to `examples/shared/certification_model.rs`; `certify` split into `causes` + new `certify_observational` |
| counterfactual | `dismech_counterfactual::counterfactual_replay` (one replay path, −6 tag) | used as is |
| epistemic output | `EpistemicState5`, `Epi5Declarations::project_state5`, `stamp` | used |
| missing | dispatch on bits 40..42; write-back rule; topology gate | added in the probe (`pearl::{reason, revise, hydrate}`) |

## The seam

- `reason(edge, evidence, edit) -> Measured` reads bits 40..42 only. SO → P7a
  observational folds (at most `CausalCandidate`); PO → P7a `causes` on the
  executed arms; SPO → `counterfactual_replay` with one route cut; SP → SO vs
  PO direction. `Measured` has private fields: only an operator produces one.
- `revise` raises the certification to what the operator earned when that is
  strictly stronger, never lowers it, and writes bits 59..63 only. SPO and SP
  earn nothing.
- `hydrate` moves `IndirectUnknown` to `IndirectKnown` only when both bindings
  `A → B` and `B → Y` are in the sealed chain; certification never moves there.

## Measured (CUI BONO script)

| step | state |
|---|---|
| start, SO | `IndirectUnknown × Related` (SO earned `Related`, no change) |
| propose unbound candidate | unchanged (`ProposedOnly`) |
| hydrate `A → B → Y` | `IndirectKnown × Related` |
| PO, executed arms | `IndirectKnown × Causes` |
| SPO, cut `B → Y` | `LoadBearing` (frequency 185 → 120), state unchanged |
| SPO, cut `A → B` | `TruthOnly` (185 → 225), state unchanged |

- The replay's verdict is a NARS revision over the steps' frequencies, so a cut
  can raise the terminal frequency (cutting the weak step). The reaction keeps
  both frequencies. In the sweep that chose the fixture, endpoints did not
  change the frequency.
- Y = A or B: removing A where B holds shows no reaction; with B disabled it
  does; the randomized arms certify `Causes`. The chain replay is linear and
  cannot express the OR, so the contingency is a mask over the unit population.

## Falsifiers (disable-verified red)

F1 SO reads the trial (`certify` instead of `certify_observational`); F2 SPO
certifies without evidence; F3 a load-bearing cut earns `Causes`; F4 the
counterfactual terminal edge written back; F5 hydration without bindings; F6
confounding demotes `Causes`; F7 a null reaction demotes; F8 revision clears
the mantissa; F9 nondeterministic dispatch; F10 intervention receipts without
arms earn `Causes`; F11 bits 59..63 select the operator; F12 the mantissa
selects the operator.

## Open

- Production placement of `reason` / `revise` / `hydrate`.
- Whole-node removal of B (only route cuts run; in a two-step chain they are
  B's only mechanisms).
- Contingency search beyond one disabled cause; no responsibility framework.
