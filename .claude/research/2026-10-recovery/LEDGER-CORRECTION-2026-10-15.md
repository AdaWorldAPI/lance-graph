# Ledger correction, 2026-10-15

Corrects `LEDGER-REVISION-2026-10-14.md` (merged #1554, fixed in `e7d2cb5b`).
That file is not edited; this one supersedes the rows named below.
Board entry: `.claude/board/entries/2026-10-15-ledger-correction-wire-contract.md`.

## 1. Operator rulings (2026-10-15)

| # | ruling | consequence |
|---|---|---|
| R-a | O25's "refuse vs choice" was a category error. `temporal.rs` already gives every row a unique coordinate `(owner, cast_seq)`. | O25 is a defect with a known fix, not a policy decision. |
| R-b | CausalEdge64 is a global wire format and ISA register ("TCP/IP for thoughts", "like BF16 mantissa bits"). | Every header bit means the same thing everywhere. K1 (per-class readings of 40..63) is withdrawn. |
| R-c | Bits 50..52 are the Maslov pyramid as the plasticity version of 59..63: a discrete plasticity dependency ladder. | The pyramid IS the plasticity field, not a second field beside it. |
| R-d | The SPO mask (40..42) is the separation of concerns for Pearl's ladder, a fixed hydration method in the pyramid's dependencies. | Per-plane granularity belongs to 40..42, not to 50..52. |
| R-e | Pearl's ladder is later wired with Pei Wang's NAL S-P and ALL others, as views on the same substrate. | New item N15. |
| R-f | Bits 40..42 do for Pearl what bits 50..52 do for the plasticity pyramid. | Two 3-bit ladders side by side: 40..42 is how far up the causal ladder the edge has been interrogated (association → intervention → counterfactual), 50..52 is how far the edge has matured (Staunen 7 → wisdom 0). Both feed 59..63, which says what may be claimed. |

## 2. The header after R-b..R-d

| bits | field | question it answers |
|---|---|---|
| 0..23 | S, P, O | which three things |
| 24..39 | f, c | how much evidence (fixed codes, one global quantizer) |
| 40..42 | Pearl ladder (`CausalMask`) | which causal question: `SO` association, `PO` intervention, `SPO` counterfactual, `SP` confounder test, plus the marginals. Ablation has no mask of its own; it is the counterfactual cut |
| 43..45 | orientation | which way the relation is read (code: direction triad, `DIR_SHIFT = 43`, sign per S, P, O) |
| 46..49 | OPEN | opcode (code) vs activation (design); one global meaning still to choose |
| 50..52 | plasticity ladder (the pyramid) | how free the edge still is to change; freezing is earned by dependency or ablation, not by confidence |
| 53..58 | witness handle (`w_slot`, 0..=63, 0 = no anchor) | which record changed this state; the W1 → W2 → W3 history lives outside the edge |
| 59..63 | `EpistemicState5` | what the edge is licensed to claim (moved by `hydrate`) |

Three fields, three concerns: method (40..42), maturity (50..52), licence
(59..63). The first two are parallel ladders (R-f). Caveat: the pyramid is an
ordinal code, but `CausalMask` encodes a rung as a plane subset (`SO`, `PO`,
`SPO`, `SP`), and `SO` and `PO` are not ordered by inclusion. The rung order
association < intervention < counterfactual must therefore be declared, never
read off bit inclusion; confounder detection (`SP`) and the marginals sit
beside the ladder, not on it. Today's plasticity encoding (`plasticity.rs`: `S_HOT`, `P_HOT`,
`O_HOT`) mixes the first two; that is the cost of the migration.

## 3. Rows corrected

| row | was | now |
|---|---|---|
| **O25** | fix OPEN; operator picks refuse (NAL) or choice (`BeliefArena`) | **DEFECT, fix known.** Replay composes along one trajectory and never revises within it; revision runs only across distinct `temporal.rs` coordinates (the shape `nal::Revision` already has). The pinned tests in `planner/tests/chain_confidence.rs` (`replay_step_raises_confidence_along_a_deduction_chain`, `self_revision_raises_confidence_on_the_replay_path`) flip, with this entry as the recorded reason. Not done here. |
| **K1** | declare 40..63 per class through the ClassView | **WITHDRAWN** (R-b). Replaced by the remaining header decisions in §4. |
| **K2** | four provenance carriers to make one | **NARROWED:** the root is the `temporal.rs` coordinate. `nars::belief::Stamp`, `nal::Evidence.stamp`, contract `revision` `independent_roots` and the probe's roots key on it. |
| **N1** | derived edges inherit no hydration, witness or maturation | **PARTLY DISSOLVED:** a derived edge born at `0b111` (most plastic) is right under R-c. Still open: witness (53..58) and epistemic state (59..63) zeroed on derivation. |
| **O22** | nothing computes `CounterfactualVerdict` | **DONE (one step).** `Reaction::verdict` and `Measured::counterfactual_verdict` in `pearl.rs`: LoadBearing from a consistent chain → Necessary; LoadBearing from an inconsistent chain (the cut repairs it) → Dispensable (Codex P1 on #1560); Inert → Dispensable; TruthOnly → NotRun; no cut → NotRun. First end-to-end `RevisionVerdict::is_acceptable()` over a real replay; 5 disable runs red. Scope: one cut judges one step; a multi-step claim still needs the claim ↔ step map. |
| **O23** | DECISION: fourth reading or remap | unchanged, still the operator's. Caveat now concrete: at the verdict level TruthOnly and "no cut ran" both read `NotRun`; they stay distinguishable only through `Measured::reaction()`. A fourth verdict variant would make the docket itself tell them apart. |
| **O26** | the pin at `causal_bridge.rs:450` asserts the defect as correct | **DEAD CODE, downgraded.** `mod causal_bridge` is behind `feature = "wip"`, which does not compile (unresolved `crate::container`, `crate::cognitive`, `search`), and nothing outside the file refers to it. The mapping cannot reach a caller and its test never runs. The ruling (an intention is a Rubikon commitment, never `Causes`) stands for whoever revives the module. |
| **O20** | verify Stroke-1 drop against ndarray `main` | **CLOSED.** ndarray `1ecbccb`: `Cascade::query` documents that recall is not guaranteed; pinned by `stroke1_drops_a_prefix_heavy_true_hit` (blackboard 2026-10-08). |
| **N3** | no arc entries for #1494–#1504 | **DONE:** 11 entries in `PR_ARC_INVENTORY.md`, written from the merged PR bodies. |
| **N4** | no STATUS rows for D-RESIDUE-FP-0, D-MATURATION3-0 | **DONE:** new section, plus D-MATURATION3-1 (the global migration). D-HL-0 and D-NALP-0 moved from "In PR" to "Shipped (#1496)". |
| **N9 (R1)** | test that goal revision is refused in `CognitiveWork` | **DONE:** `kanban::tests::r1_goal_revision_is_refused_inside_cognitive_work`. The DAG already enforced it; disable run (a `Plan` exit from `CognitiveWork`) red. R2–R4 stay open. |
| **N10** | `causal-edge` fails `clippy -D warnings` | **CLOSED on `main`:** `cargo clippy -p causal-edge --all-targets -- -D warnings` exits 0 at `4e386f26`. |

## 4. Header decisions still open (replacing K1)

1. One global meaning for 46..49: opcode (code) or activation (design).
2. The derived-packet rule: what a syllogism writes into 53..63.
3. The pyramid's entry code: 7 (Astonishment) at birth, or a separate "new".
4. `learn`'s rule under R-c: a plane changes only if it is in the SPO mask AND the ladder stage permits change; the threshold is the operator's.
5. The migration gate: old plasticity codes 1..6 need one global version gate; `learn`, `MooreNars16` and `CausalEdgeV3` byte 3 move together.

6. **Ablation on the Pearl ladder? (operator: "probably part of it")** Two
   different operations carry the name today:
   - the counterfactual cut (`SPO`, `pearl.rs`): remove one step from one
     chain; does THIS conclusion still hold? It looks upstream (necessity).
   - the anchor ablation (`maturation_ladder.rs`, `Executed::Ablated { impact }`):
     remove a relation from an isolated copy of the population; how many
     OTHER relations reopen? It looks downstream (dependence), and it is what
     earns the pyramid's Wisdom stage (impact ≥ 1 from Integration).
   Room exists: the four single-plane codes (`None`, `S`, `P`, `O`) select no
   operator today. Undecided: whether ablation takes one of them as its own
   rung, or stays the pyramid's operation that 40..42 only feeds.
   **Operator, 2026-10-15:** ablation will be detailed through the N15 families
   (NAL, Hyperon/MeTTa, PeTTa, CE64, ONA, OpenNARS / Pei Wang's Java), not
   decided here. Existing anchors to start from:
   - OpenNARS Java runner and fixture: `.claude/knowledge/nal-recipe-audit/` (#1495);
   - removing a contributor from fused evidence (Jøsang fission, #1499 red test 2);
   - NARS retraction, a typed insufficiency today (#1504; needs a semijoin onto asserted stamps);
   - Hyperon/MeTTa over the mask polyfill (D-MTP, #1553); PeTTa is not pinned yet (#1497, #1498).
   Until then no code reserves a 40..42 code for ablation.

## 4a. Reconciled against the README diagram (2026-10-15)

The README diagram and `docs/architecture/ce64-semantic-upper-half.md` are
the reference. Against them:

- 40..42 and 59..63 match the code; 53..58 is already the witness handle in
  code. Both rows above now say so (this file first left 53..58 open).
- 43..52 are a proposed re-reading (orientation, activation, novelty) of bits
  the code writes as direction triad, inference mantissa and plasticity. The
  doc lists this as open behind a version gate; so does §4 here.
- **Tension to resolve:** the doc's "Encoding, reading, operation" section
  lets each tenant declare its own reading through a versioned binding. Ruling
  R-b says every bit means the same thing everywhere. Both hold only if the
  versioned binding is global (one wire version), not per tenant. The doc's
  wording is not changed here.

## 5. N15 — Pearl × Wang S-P × all others, as views on one wire

**Ruling:** Pearl's ladder is later wired with Pei Wang's NAL S-P and with
the other reasoning families, all as views over the same CE64 substrate.
Because the wire means the same everywhere (R-b), each family is a reading
of the same bytes, never a second encoding.

| family | what it reads from the wire | existing anchor |
|---|---|---|
| Pearl (see / do / imagine) | method from 40..42; truth from 24..39 | `pearl.rs`, `Reaction` |
| Wang NAL (S-P, copulas, NAL-1..9) | S, P from 0..23; truth codes from 24..39 | `lance-graph-nal`, `nal-statement-v1`, exact codes from #1548 |
| OpenNARS, ONA | revision and stamps | `nal_revision_oracle.rs` (#1495) |
| OpenCog PLN | strength/confidence; never read as calibrated probability | #1497 research map |
| Jøsang subjective logic | fusion and fission as an ablation control | #1499 excerpts |
| Hyperon / MeTTa | pattern matching with binding multiplicity | D-MTP (#1553), #1498 |

Home: D-HOLO H12 (semantic map of the stacked bricks).

Ablation (§4 item 6) is defined through these families, each with its own
oracle, before it gets a place on the wire.

Caveats, stated so the wiring does not overclaim:

- Pearl's rungs and NAL's figures may be orthogonal axes, not one ladder;
  wiring them as one would collapse two concerns.
- NAL copulas (inheritance, similarity, implication, equivalence) have no slot
  in the SPO mask; where they live on the wire is undecided.
- PLN strength and NARS frequency share a range, not a meaning.
- Each view needs an executable oracle before it is called equivalent
  (the D-NAL / D-MTP pattern), never mask equality alone.

## 6. Agent ledger (Haiku harvest tier)

| card | runs | items checked | wrong | missed |
|---|---|---|---|---|
| harvest-scout (ndarray cascade) | 1 | 3 (`cascade.rs:270-276`, `:813`, blackboard `:84-90`) | 0 | 0 |
| general-purpose on Haiku (PR bodies #1494–#1504) | 1 | 11 bodies used verbatim; merge SHAs taken from `git log` (not in the fetch) | 0 | 0 |
