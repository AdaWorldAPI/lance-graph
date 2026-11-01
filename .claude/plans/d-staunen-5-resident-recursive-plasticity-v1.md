# D-STAUNEN-5 | Resident recursive plasticity | v1

**Status:** PROPOSAL · **Date:** 2026-11-01 · **Prereq:** D-XLOG-NATIVE-7 (cube, #1911/#1912), D-XLOG-NATIVE-8 (#1913), xlog#2–#4 (`xlog-cube` on xlog `main`) · **Reads:** D-STAUNEN-2 (A4/A5), D-STAUNEN-3, D-STAUNEN-4 (B0/B10/B11, M1/M2/M5)

## Overview

D-STAUNEN-3 proposed an XLOG/structure/learning integration. Its first
foundation is now implemented: XLOG plans run on the resident hypercube,
recursion included. D-STAUNEN-5 is the first experiment that closes a loop
on that foundation.

**Question.** Can a novel stimulus trigger a local structural adaptation
that changes a recursively computed prediction, and can the system then
learn when that adaptation was useful?

**Scope fences (session direction).**
- No new runtime. The experiment runs on the existing cube and
  `ndarray::simd` mask words.
- No widened relation representation. One binary relation over one tile
  (at most 64 ids in v1, so a row is one `u64`).
- No Prolog meta-predicate support (`functor`/`arg`/`=..` stay refused).
- Not 64 thoughts. Two competing hypotheses, each independently verifiable.

## The setup

- **Base relation** `edge(X, Y)`: binary, over one `LocalMap`.
- **Recursive prediction** `reach = edge⁺` (transitive closure, the program
  `xlog-cube` already runs). A prediction is "is `(a, b)` in `reach`".
- **Candidate edges** `C`: a set declared before the run. Each is dormant
  (absent from `edge`) or active. This is the only structural freedom.
- **Hidden world** `G*`: a ground-truth graph over the same ids. The
  system never reads `G*`. It only sees stimuli sampled from `reach(G*)`.
- **Stimulus** `obs(a, b)`: one observed reachable pair.
  **Novel** means `(a, b)` is not in the current `reach`.

## The update, under Atlas's contract

Atlas (`AdaWorldAPI/brian2-atlas`, cloned this session, branch `dev`
`9ec52f9`) has one bounded structural update:
`NativeLIFTrainer.update_mask(masks, growth_weight)`
(`brian2-rust/python/brian2_rust/training.py`). Its contract, read from the
source:

1. The mask shape is the plan's. "topology migration is unsupported", so
   only declared connections can change.
2. Entries are 0 or 1.
3. Applied atomically between sequences, without advancing clocks.
4. A pruned cell is zeroed. A regrown cell gets its declared or learned
   initial value, and its event history and timestamps are cleared.
5. Refused before a committed runtime state exists.

The cube's update obeys the same five rules. It does not run Atlas.

| Atlas rule | Cube rule | Refusal |
|---|---|---|
| plan-shaped mask | the update toggles one member of the declared `C` | `NotACandidate` |
| entries 0/1 | an edge is active or dormant; no weights in v1 | (by type) |
| atomic, between sequences | applied between stimuli, never inside a fixpoint round | `InsideRound` (cube rule) |
| regrowth clears history | an activated edge carries no prior evidence; its record starts empty | — |
| needs committed state | refused until the base closure is sealed under a frame | `NotSealed` |

`update_mask` checks rules 1, 2 and 5 (plus an ownership layout for
dynamic plans). The refusal names in the table are the cube's; Atlas has
no `NotACandidate`, `InsideRound` or `Stale`, and no novelty gate. A novel
stimulus that no single candidate closes is refused with
`NoClosingCandidate`, changing neither generation nor masks.

Plus two rules of our own:
- **One update per novel stimulus.** A non-novel stimulus is never a
  licence to adapt.
- **Generation guard.** Every update bumps a generation; a delta, a
  prediction or a pending verdict from an older generation is refused
  (`Stale`). This is D-STAUNEN-4 M1 in its smallest form.

An activation is speculative: it is held as a THOUGHT delta over the
sealed base, never written into the base until it is consolidated (below).

## The in-cube delta

Today `xlog-cube` computes each recursive round's new rows on the host (a
set difference) and re-adds the whole relation every round. Both shortcuts
are removed for this relation:

- The relation is held as row masks over local positions:
  `row[x]` has bit `y` set when `(x, y)` holds. 64 ids, so one `u64` a row.
- Semi-naive closure: `delta' = (delta ∘ edge) andnot full`;
  `full |= delta'`. Composition is `or` over the rows of `edge` selected by
  `delta[x]`; difference is `mask_andnot`; union is `mask_or`.
- **Incremental update for one activated edge `(u, v)`.** The new pairs are
  `{x : x = u or reach(x, u)} × {y : y = v or reach(v, y)}`, minus `reach`.
  This is one column read, one row read, and an outer `or`, then
  `mask_andnot`. No fixpoint re-run.
- Deactivation (revert) is not the inverse. The closure is recomputed from
  the base plus the remaining active candidates. The plan says so rather
  than inventing a decremental algorithm.

Every mask op is an `ndarray::simd` word (`mask_and`, `mask_or`,
`mask_andnot`, `mask_any`). No tuples are built except at the one named
terminal.

## Two competing hypotheses

Each is a policy: given a novel `obs(a, b)`, which candidate to activate.
Both must make `(a, b)` predicted. They differ in what else they predict.

- **H1, parsimony.** Activate the candidate that makes `(a, b)` reachable
  with the fewest new `reach` pairs. Minimal structural surprise.
- **H2, eligibility.** Each id carries an eligibility count: how often it
  appeared in recent stimuli. Activate the closing candidate whose endpoints
  have the highest eligibility. Novelty is the third factor; eligibility is
  the trace. This is the D-STAUNEN-2 A4 rule, `Δ = M · e`, with a binary
  structural target instead of a weight.

They are independently verifiable: each can be scored against `G*` on
pairs it was not shown, and the two disagree on fixtures built for it.

## Learning when an adaptation was useful

An activated candidate is pending. Later stimuli are its evidence:
- a later stimulus that `reach` predicts **only because of** that candidate
  is a confirmation;
- a predicted pair that a held-out query from `G*` refutes is a refutation.

After `k` subsequent stimuli the pending update is **consolidated** (merged
into the sealed base, generation bump) or **reverted** (deactivated). Each
outcome is recorded against the policy that proposed the update, as
frequency/confidence counts.

**Meta-step.** A selector picks H1 or H2 for the next novel stimulus from
those records. That is the "learn when it was useful" claim at policy
level.

**Evidence rules (D-STAUNEN-4 M2/M3).**
- A stimulus counts once: the same observation repeated is one root.
- A pair derived through the closure is not independent evidence for the
  edge that produced it, so a candidate cannot confirm itself.
- Only stimuli or held-out queries arriving after the activation count.

## Phases (tests first, each gate a falsifier)

- [ ] **P0 — fixtures and oracle.** A seeded generator for `G*`, `C`
  (containing the true missing edges plus distractors) and stimulus streams.
  A scalar oracle closure. Each novel stimulus is classified by the oracle
  as one-edge-closable (the candidates that close it, listed) or not. The
  primary experiment uses closable episodes; every stream also carries at
  least one novel stimulus no single candidate closes, which P3 must refuse
  with `NoClosingCandidate` without changing the generation or the masks.
  `G*` lives only in the generator and the scorer: the episode type that
  H1, H2 and the selector receive has no field for it. Pre-register: task
  split, episode count, `k`, minimum effect, budget.
- [ ] **P1 — in-cube semi-naive closure.** Three independent checks: a
  scalar reference closure, the resident mask closure, and `xlog-cube`'s
  closure. The last is a comparator only after its own nonempty `r(V,V)`
  diagonal fixture passes; a failure there does not by itself indict the
  resident closure. Fixed fixture: `edge = {(0,1),(1,2),(2,0),(2,3)}` has
  exactly 12 closure pairs, including `(0,0)`, `(1,1)`, `(2,2)`, excluding
  `(3,3)`. Disable runs: drop the `andnot` (a cyclic fixpoint never
  empties its delta and hits the round bound); compose from `full` instead
  of `delta`. The second gives the same closure in possibly the same number
  of rounds, so it is caught on work, not rounds: in semi-naive evaluation
  each closure pair enters the delta once, so rows composed equals
  `|edge⁺|` exactly.

  **Acceptance.** For nonempty cyclic and acyclic graphs, including local
  position 63, resident `edge⁺` equals the scalar oracle bit for bit and
  agrees with a separately validated `xlog-cube` baseline. The instrumented
  run shows zero host-side delta differences and zero relations added to the
  cube. **Falsifier:** one mismatched pair, an unbounded cyclic fixpoint, or
  one host-side delta calculation. **Refusal boundary:** a P1 pass admits no
  structural adaptation, policy selection, consolidation or learning claim.
- [ ] **P2 — incremental activation.** After activating `(u, v)`, the
  incremental closure equals a full recompute, over every `(u, v)` in a
  random `C`. Disable run: drop the `x = u` term.
- [ ] **P3 — the gate.** Each refusal (`NotACandidate`, `InsideRound`,
  `NotSealed`, `Stale`) fires on its input, and each stays silent on a
  legal update. A non-novel stimulus never adapts.
- [ ] **P4 — one closed episode.** Novel stimulus → one activation → a
  prediction that was false is now true → later stimuli → consolidate or
  revert. Asserted end to end on a fixture whose answer is known.
- [ ] **P5 — H1 vs H2.** Score both on held-out pairs from `G*` over seeded
  episodes. Fixtures where they must disagree (a short distractor beside a
  recently active long path), so the comparison is not vacuous.
- [ ] **P6 — controls and meta-selection.** B0 no adaptation; B10 best
  fixed policy; B11 random candidate; B7 shuffled novelty (adapt on
  non-novel stimuli); B9 duplicated root (one observation repeated). The
  meta-selector must beat B11 and match B10 under a pre-registered paired
  criterion over seeds (not a win on every seed). Report the losing runs
  as well.
- [ ] **P7 — Atlas differential (optional, lab).** Mirror one fixture into
  an Atlas plan whose masks are `C`, apply the same activation sequence with
  `update_mask`, and compare only the contract both sides share: mask
  shape, 0/1 entries, committed state, and (for dynamic plans) the
  ownership layout. `NotACandidate`, `InsideRound`, `Stale`,
  `NoClosingCandidate` and novelty gating are the cube's own rules, tested
  in P3, never demanded from Atlas. Weight restoration differs between
  Atlas's static branch (`growth_weight`) and dynamic migration, and v1 has
  no weights, so neither is compared.
  Needs Atlas built in the session; if it does not build, record that and
  stop rather than copying its executor.

## What a pass would and would not show

A pass shows that, on one binary relation and a declared candidate set, a
novelty-gated structural update changes a recursive prediction, and that a
policy record learned from later evidence picks updates that survive
better than chance. It shows nothing about spiking dynamics, weights,
biological structural plasticity, or general learning. Atlas is the
contract source, not the executor, until P7 runs.

## Where the code goes

`crates/lance-graph-xlog` (the cube's crate), as a `plastic` lab module plus
`tests/plastic.rs`. No new crate. `xlog-cube` is not changed until P1 can
replace its host-side delta, and that is a separate PR.

## Open

- `k` and the consolidate/revert thresholds are policy pins until P5
  measures them.
- Whether H2's eligibility should decay. v1 counts over a fixed window.
- Weights (Atlas `growth_weight`) are out of v1: edges are binary.

## Relation to the xlog-cube coverage work (session review, 2026-11-01)

A review of `xlog-cube`, XLOG's RIR and the cube concluded that most of the
90 refused integration rules need new ways to compose algebra the substrate
already has, not 90 new operators. The proposed order, recorded here
because step 3 is this plan's P1:

1. Projection composition: `P_A(P_B(R)) = P_{B∘A}(R)` at preparation time
   (5 `Project(Project(..))` rules); join projections only where the whole
   output schema is kept, values borrowed by field reference.
2. Filter algebra: comparisons and `And`/`Or`/`Not` as native masks, with
   explicit width, signedness, float and UNKNOWN semantics. Column-to-column
   equality first, because it also gates `q(X) :- r(X,X)`.
3. Resident execution: no full-relation re-insert per round, set difference
   and adjacency in the cube (**= P1 here**).
4. Computed columns (`Add`/`Sub` first, checked) and grouped reductions.
5. F100 behaviour: ground-term `arg/3` etc. under the lens; unsupported
   binding modes refused.
6. N-ary relations: a separate research gate, kept blocked. More logical
   columns are more SoA lanes under one relation identity and one row
   selection, never a dense N-dimensional cube.

Two corrections to the coverage figures reported with xlog#3:
- **Admission is not correctness.** `examples/coverage.rs` gives every
  binary base relation as empty, so 140/230 means 140 shapes were accepted,
  not 140 checked answers. Each newly admitted shape needs its own
  differential test before it counts.
- **First reason only.** A rule is counted under its first refusal; fixing
  one category can expose another. The nine counts are not additive.

Consequence for P1: the `r(V,V)` diagonal in `generic_join` (implemented,
untested) is a prerequisite, because P1 compares the mask closure with
`xlog-cube`'s.
