# Research recovery: sparse-event detector × Morton-Moore × CE64 (read-only)

All findings below come from reading code and board entries on lance-graph
`origin/main` 1573eb4c and ndarray `origin/master` 8afa92d, through read-only
worktrees. No experiment was run, nothing was edited, committed or deleted.

## A. Archaeology — where the "13,000 LOC" are

**Nothing is at risk.** No repo has untracked, stashed or unpushed work
(`git status --porcelain` clean, no stashes, one worktree each besides `_ro`).
Commits carrying this session's trailer:

| repo | commits | lines added | on main |
|---|---|---|---|
| lance-graph | 64 | 14,767 (520 deleted) | every file, all merged |
| ndarray | 14 | 1,352 | all merged; the 2-D window + bit-sliced count were added and then **reverted** (4a6dd93, 547b8aa) |

Breakdown of lance-graph (lines added by the session; current size on main):

| kind | files | lines |
|---|---|---|
| probe examples | mexhat_bucket (1,329), phasor_trig (1,524), algebraic_recipe (1,865), fold_join (1,173), meta_prefilter_mask (279), parts of gomoku / crossword / moore_plasticity | ~7,800 |
| contract (production) | moore_tenant.rs (800), register128.rs (+474), hotplug.rs (+384), canonical_node.rs (+66) | ~1,750 |
| causal-edge (production + tests) | isa.rs (510), edge.rs (+157), 5 test files incl. golden vectors | ~2,100 |
| planner tests | moore_nars16_probe (800), isa_equivalence (153), chain_confidence, gadamer, d_rpf | ~1,500 |
| board / plans / docs | entries, STATUS_BOARD, plans | ~2,000 |
| lab harness | `cognitive-shader-driver/examples/shared/lab.rs` | 377 |

Not this session, but directly relevant (found during the sweep):

- `crates/perturbation-sim` (7.1k LOC, workspace-excluded): a power-grid outage
  simulator. It is NOT an event detector (no timestamps, no coincidence window,
  no ground-truth detection test). It carries an f64 `RollingFloor` that mirrors
  ndarray's `Cascade`, not ndarray's `RollingFloor`.
- `perturbation-sim/examples/MoorePertubationsfeld.rs` (#1427) and
  `moore_observable_first.rs` (D-MOORE-OBSERVABLE-FIRST-0, 2026-10-09): an
  8-lane MooreSlot stencil relaxation with **certified early decisions**. This
  is the most detector-like result in the tree (see B).

Correction to my previous recovery: I called #1427 "merged without
measurements". #1427 measured correlation per sweep budget, and its follow-up
D-MOORE-OBSERVABLE-FIRST-0 measured decision soundness with independent
oracles. UNPROVEN was wrong for the pair.

## B. Reinterpretation — what actually resembles a trigger

The KM3NeT side is from memory (SECONDHAND): its triggers are built on hit
coincidence within a DOM and a causality window |Δt| ≤ |Δx|/c + margin between
hits, then track/shower fits. No KM3NeT document was read in this session.

| piece | measured result | detector role it resembles | verdict |
|---|---|---|---|
| fold-join `popcount(A∧B)` on resident masks (D-RPF-9) | 385 ns / 64K rows, no slot, no bitmap; all arms equal the oracle | **coincidence counting** | genuinely trigger-like; shipped path |
| prefilter as mask (PROBE-PREFILTER-MASK) | `x ≥ k` on packed fields = ≤ n+1 ternary patterns; 400 filters × 4,099 rows exact | **cheap pre-trigger veto** without selection vectors | genuinely trigger-like; test-pinned, not wired |
| Mexican hat D/F/Q + exact-bound early exit (D-MHB-1/2) | exact centre-surround sum at 230–620 ns; per-query D/F choice within 0.93–1.33× of best; exact early exit −10–20 % on F, never wrong | **local-contrast trigger decision `S ≥ T`** with exact accept/reject | the cost and exactness are proven; **detection quality was never measured** — no ground truth, no ROC, no false-positive rate |
| ring-bucket Mexican hat (E) | wrong at every K (950–5,600 on \|S\| in the thousands) | — | negative result, keep |
| counterexample 5 | buckets wider than λ/8 put dark and bright fringes together | a centre-surround score is not a phase/fringe classifier | negative result, keep |
| `RollingFloor` (ndarray) | PARKED: in D-MHB-1 the deciding quantity was an exact window popcount, so there was nothing statistical to decide; its drift rule is **untested** | **adaptive background** | untested in the role the analogy needs (drifting background) |
| `Cascade` Stroke 1 | drops prefix-heavy true hits (pinned test) | statistical pre-trigger with no exact fallback | a known recall defect of a trigger-shaped primitive |
| D-MOORE-OBSERVABLE-FIRST-0 | 0 certified-wrong decisions over all fixtures and budgets; 99.9 % of 863,970 overload decisions certified while the field is still 88 % wrong; what certifies early is the **absence** of an event | **certified trigger with abstention**: decide without full reconstruction, abstain on near-ties | the strongest support for "decide before materialising the field" |
| phasor LUT / interference early exit (D-PHT-1) | LUT 4.5× vs f32 sin_cos; amplitude-bounded exit saves 8–36× only for heavy-tailed amplitudes, ~nothing for uniform | coherent-sum reconstruction | random-walk limit is a real negative for uniform sources |
| moore_plasticity_probe | synchronous one-hop sweep over all lanes; Morton and `mask_shift_morton` schedules equal | local propagation | a sweep, **not** a moving focus |
| algebraic recipe table | canonicaliser + guarded recipes, refuses collision queries | — | not detector-related |

**What does not exist anywhere** (closed search over crates/, read-confirmed by
the scout): a single focus moving cell-to-cell over Morton/Moore; any time
dimension in a spatial probe (the cross-time W(Δ) of `.arxiv/2605.01487` was
crosswalked, never run); any detection-rate or false-positive measurement for
the contrast primitives.

**Answer to the central question:** partly yes, by convergent pieces rather
than one design. The session built and measured exact, materialisation-free
versions of three trigger stages — coincidence counting, local contrast with an
exact accept/reject bound, and a mask veto — and a sibling session built the
certified-abstention decision. It never measured whether any of them DETECTS
anything, never had a time axis, and never moved a focus. The Mexican hat's
value over plain coincidence plus a floor is therefore UNKNOWN, not shown.

## C. Smallest credible event contract (design only)

Reuse, nothing new: `Morton8x8::checked_offset` (no wrap, refuses edges),
`MooreSlot::offset`, the three tenants 18–20, `CausalEdge64::{forward, learn,
syllogize}` with strict decoding, `PalettePerturbation::hop`, quack/mask-risc
for any population count.

One event at Morton address X, slot s:

1. read the row at X and the neighbour row at `X.checked_offset(s)`; refuse at
   the edge;
2. compose the two CE64 operands from resident fields;
3. run one ISA op; strict decode refuses an unexecutable opcode;
4. project the result back, writing only the fields the op declares live, by
   the owner;
5. the next X is a function of stored state (declared rule), not an argmax.

**Blocker found while checking step 2.** A full operand needs bits 0..39
(S/P/O indices, frequency, confidence). `MooreNars16` holds bits 40..52 and
59..63 only. #1406's equivalence passed because the test held a per-lane
`payload[k]` outside the slab (`moore_nars16_probe.rs:299-305`). There is no
resident home for bits 0..39 today. Candidates, all operator decisions:
`Nars16x8` as f8:c8 (undeclared), `MoorePalettePairs` as two of the three
indices, the Moore slot as the predicate. Without one, "composed from resident
fields" is false and Experiment C cannot pass.

Directional windows (N/E/S/W × corner-middle-corner): no code exists. The
shared corners mean each diagonal neighbour appears in two windows; the
contract must name ONE owning window per diagonal so evidence is not counted
twice. Strict 4-neighbour is `MortonDir` as shipped.

DTO stream: `StreamDto`/`PerturbationDto`/`BusDto` live in thinking-engine;
nothing passes a `StreamDto` into `perturb`, and `commit` rebuilds the
`PerturbationDto` from engine state. A local event fits as a producer of
`(u16, CausalEdge64)` into the existing `MailboxSoA::apply_edges`, without a new
DTO. That function does not call `learn`; it adds `mantissa/8 · confidence` to
energy. Whether that is the right sink or a second semantics is open.

## D. Falsification — pre-registered, not run

All reuse existing harnesses; none needs a new statistics library.

- **A (detection).** Arms 1–4 as specified, plus arm 0: oracle with known
  ground truth. Fixtures: sparse tracks + Poisson background + distractor
  blobs + a drifting background rate (the case RollingFloor exists for).
  Metrics: TPR at fixed FPR per arm, sign-flip CI via `lab.rs` over seeds,
  ns/event. **Kill:** Mexican hat adds no TPR over arm 2 at matched FPR →
  record negative. **Guard:** counterexample 5 says a fringe fixture is the
  wrong test for it; use blob/track fixtures.
- **B (traversal).** Moore8 / 4 windows / strict 4 / single focus over a fixed
  sparse field. Metrics: discoveries per CE64 op, row reads, coverage,
  revisit and trapping counts, byte-identical replay. Coverage must be
  reported; one focus will not cover the field.
- **C (composed register).** Composed vs materialised operands through all
  five ISA ops; corrupt one field, the witness, the address, a lane →
  each must be refused or diverge. **Blocked** by the bits 0..39 home.
- **D (learning continuity).** The previous turn's resident-cycle design; add
  same-source repeats (stamp dedup) and a removed bridge node.
- **E (runtime).** Separate rows: one LUT lookup (D-PHT-1: 3–6 ns), one
  popcount, one ISA op, one address-resolved event, one event with owner
  write-back. 7 ns per *committed* event is unmeasured; the existing numbers
  say a Mexican-hat query alone is 230–620 ns, so 7 ns can only hold for an
  event that does no window sum.

## E. Reuse decisions (no destructive action)

| component | decision | reason |
|---|---|---|
| fold-join popcount, prefilter mask | REUSE | coincidence and veto stages, exact, shipped |
| Mexican hat D/F/Q/H probe | PRESERVE as reference; ADAPT only after A | exact and fast; detection value unmeasured |
| ring-bucket arm, counterexample 5 | PRESERVE as negative results | |
| RollingFloor | ADAPT into Experiment A | untested in the drifting-background role |
| Cascade Stroke 1 | ISOLATE | recall defect pinned |
| D-MOORE-OBSERVABLE-FIRST-0 | REUSE its certificate pattern | sound decide-or-abstain without convergence |
| perturbation-sim RollingFloor | REJECT as a third floor | duplicates ndarray `Cascade` |
| phasor LUT | REUSE (shipped in ndarray) | |
| algebraic recipe | PRESERVE, unrelated | |
| tenants 18–20 | PRESERVE; hygiene per G5 | |

## F. Minimal next seam

One planner test, no production change: read a row's three Moore tenants plus
one declared home for bits 0..39, compose one CE64 operand per lane, run
`learn` against a neighbour operand, write back only `learn`'s live fields,
reload from bytes, run the next cycle. It is Experiment C and D's core, and it
cannot be written until the bits 0..39 home is decided.

## Addendum — references lance-graph #1466 and OGAR #350 (both merged 2026-10-10)

**#1466 split-tunnel persistence** (`.claude/knowledge/split-tunnel-sparse-append-resident-fold.md`):
- Supplies step 4 of the event contract. A field an op changes persists as one
  sparse `(version, owner, op, row, lane, present, value)` SET record, never a
  64-bit image. So CE64 never needs to persist as an object; it is composed
  from lanes on read. This supports D's idea. It does not remove the need for
  a lane that holds bits 0..39.
- Rule 7 decides what an event may persist: derived reads write nothing; only a
  learned outcome is stored. So NARS revision and plasticity changes persist;
  contrast scores, trigger decisions and trajectories are recomputed, never stored.
- Splits the runtime target. A fold costs about 1.7 ns (0.6 ns per resident
  record measured) and a durable write about 1.7 ms, with one Lance version per
  batch. "7 ns per event" can only describe the resident apply; durability is
  per batch. Experiment E needs both rows: event applied resident, and batch
  committed.
- Rules 4–5 are Experiment D's replay test as shipped practice: install only
  after a full replay, refuse an out-of-order log, replayed history identical.
- One writer per mailbox comes from ownership; a racing writer is only
  detected afterwards (OPEN). A moving focus must not write on behalf of
  several owners without that guarantee.

**OGAR #350 OGIT tables** (`ogar_action_handler::ogit`):
- Same contract shape the event needs: u16 row ids, u32 lanes lent zero-copy as
  mask-risc `LaneRef::U32`, refusal past 65,536 rows instead of widening, names
  kept out of lanes, nothing dropped silently (rejected / duplicates / unresolved
  reported). The last point is what `Cascade` Stroke 1 violates.
- `Allowed (src, verb, dst)` is a legality table for triples. An event's S/P/O
  could be checked against it with one mask-risc fold before the CE64 op runs:
  a semantic veto stage, distinct from the statistical trigger stages.
- Caution: OGIT ids are row ordinals of an ontology, not Palette256 indices;
  using `Allowed` would need a declared mapping between the two, not an
  assumption that they coincide.
