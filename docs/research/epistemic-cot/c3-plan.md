# D-EPI-COT C3 — plan: Epi5 bridge, scenting, numerical learning

Status: PLAN, with the first C3 slice delivered (§8). C1/C2 (`nars::support`)
is unchanged.

C3 has three parts, in this order:

- **C3a:** the Epi5 reading from ledger and certification results.
- **C3b:** scenting, meaning choosing which question to investigate next.
- **C3c:** quantized learning of investigation choices.

C3c depends on C3b, and C3b depends on an open decision about bits 50..52
(§2).

## 1. Boundaries that hold across all three parts

- Source evidence and the NARS truth laws are the only authority on
  ⟨f, c⟩. Nothing learned, scented or computed in a lane changes
  frequency or confidence, or the Epi5 bits 59..63, without admitted evidence.
- Bits 59..63 change only through the existing writer and only on admitted
  evidence. Entropy, surprise, activation, maturation and learned scores
  never write them (S2, PZ4).
- A source root counts once, however many thoughts replay it (S3, PZ6, L7).
- No new CE64 bits, no sidecar register, no new certification enum, no new
  runtime.
- Execution substrate constraints:
  - SQL legacy paths are not used.
  - z8run is not used until it is migrated to Quack.
  - rs-graph-llm (graph-flow) is used only with Lance storage, and only
    after migration to lance-graph-flow over the internal algebra and Quack.

## 2. Bits 50..52: what the code says (VERIFIED-IN-CODE)

The amendment treats Staunen as plasticity, a single capacity. The code has
**three readings** of these three bits, and they do not agree:

| reading | where | encoding |
|---|---|---|
| per-plane plasticity | `causal-edge/src/plasticity.rs`, `CausalEdge64::learn` (edge.rs ~810) | bit 0 = S hot, bit 1 = P hot, bit 2 = O hot; `0b000` = all frozen, `0b111` = all hot |
| novelty → wisdom | `CausalEdge64::novelty_wisdom` (edge.rs ~1095), written by `deepnsm-v2/src/wire.rs:972` | 0 = unaddressed, 1 = Staunen … 7 = Wisdom |
| Maslov pyramid (test-local) | `lance-graph-planner/tests/maturation_ladder.rs`, board entry 2026-10-11 | 7 = Astonishment (Staunen) … 0 = Wisdom |

Consequences:

- **The two maturation codebooks run in opposite directions.** "Preserve the
  existing stage ordering" has no single answer until one is chosen.
- **`learn` reads the same bits as plasticity.**
  - A `deepnsm-v2` edge at Wisdom (7) is `ALL_HOT` to `learn`: every plane
    adopts new observations. At Staunen (1) only S is hot.
  - Under the Maslov order, Wisdom (0) is `ALL_FROZEN`: `learn` adopts
    nothing.
  - So "Wisdom stays receptive" is false under one codebook and true by
    accident under the other.
  - Not checked: whether any `deepnsm-v2` edge reaches `learn`.
- **The migration decision is recorded as OPEN** (board entry 2026-10-11, the
  "CONTRACT DECISION" section). Its smallest viable path:
  - a declared reading, so historic rows keep plasticity;
  - a decided home for plasticity under that reading;
  - a `learn` that does not gate on maturation bits.

  `band_reading::EdgeProvenance` today refuses every existing provenance for
  the maturation reading.
- **Receptivity therefore cannot come from the gate value.** It comes from
  the reopen rule: in the Maslov probe, only a violation moves a matured
  relation backward, and forward results never do. S1 and S8 test that rule,
  not a bit value.

**Decision needed before C3b writes anything to bits 50..52:** which
maturation order is canonical, and how a class declares the maturation
reading versus the plasticity reading. Until that is decided, C3b carries
maturation transiently and does not write the bits, as D-CE64-STAUNEN-0 did.

## 3. What scenting already measured (D-CE64-STAUNEN-0, board 2026-10-07)

That probe used 32 real and 8 noise basins, one world shape and one seed.

- **Raw surprise is captured by noise:** 0.649 resolved, against 0.964 for
  round-robin. This is the noisy-TV failure.
- **Two fixes carry it:**
  - habituation (only surprise above the basin's own habitual level counts);
  - learning progress (entropy reduction that persisted one window later).
- **Fixed scheduler:** 0.952 resolved, wake latency 4.2 windows (39.9
  without surprise), noise share 0.832. Round-robin still resolves slightly
  more.
- **A 3-bit carrier loses information:** binary wakes 2.5× slower;
  thermometer almost loses the interrupt.
- **Not measured beyond epoch 1:** whether fold N is better directed because
  of folds 1..N−1.

This partly answers S4 for one world and one seed. It also shows that the
"entropy alone is not enough" rule is necessary, not just prudent.

C3b reuses this probe's scheduler and gates and adds the parts it lacks:

- **Discriminating ablation as the target.** STAUNEN-0 rewards resolution.
  S4 asks which lead yields a grounded, discriminating ablation.
- **Leads from `IndirectUnknown × Causes` (raw5 22).** Each candidate
  intermediate predicts a different ablation outcome, and the ledger's
  `sensitivity` (Δf, Δc, chains lost) is the measured information gain.
- **A second world shape and a second seed,** since the round-robin margin
  is small.
- **The `staunen` name collision:** `planner::nars::basin_resonance::staunen`
  computes `2·|f−0.5|·c`, which is not surprisal. Rename or document it
  before C3b uses the word.

## 4. C3a — Epi5 bridge (L1–L5)

Reading only. Reuse `facts_population`, `CertificationModel` and the
affordance law, moving it out of `cognitive-shader-driver/examples/shared/`
only if the move is all that is needed.

| falsifier | fixture | expected |
|---|---|---|
| L1 | association, no executed intervention arms | no `Causes`; ledger load and sensitivity still reported |
| L2 | direct RCT-backed edge | `Direct × Causes` (raw5 20) |
| L3 | intervention-backed chain with known mediator, ablation receipts | `IndirectKnown × Causes` (raw5 21) |
| L4 | intervention-backed end-to-end effect, mediator not observed | `IndirectUnknown × Causes` (raw5 22); no mediator invented |
| L5 | two causal paths | ablating one path leaves `Causes`; joint ablation removes it |

The ledger supplies derivation receipts. It never supplies the certification
itself.

## 5. C3c — numerical learning (PZ1–PZ8)

Available, verified:

- ndarray `lut9` (`src/simd_stride.rs`), at master head `2cbe131`, which
  merged #383;
- lance-graph `lane_codec` and the #1839 per-class lane layout.

**Not found:** a `lance-graph-tflite` crate on lance-graph main
(`694b243bd`). Only `lance-graph-onnx` exists. Where the TFLite kernels
live has to be settled before C3c starts.

What C3c learns:

- **Target:** which proposed ablation will discriminate between competing
  intermediates.
- **Inputs:** ⟨f, c⟩, root count and overlap, the Epi5 reading, habituated
  surprise and learning progress (C3b), and past ablation outcomes.
- **Signed residuals:** observed sensitivity minus predicted. Pass them
  through `lut9` only where the update function is odd (PZ3). Otherwise use
  another kernel, or refuse.
- **Each LUT is bound to its calibration and model generation.** A stale
  one refuses (PZ7).
- **Baselines (PZ5):** a held-out world shape, compared against round-robin,
  seeded random and the C3b scheduler. Report information gained per
  ablation and the cost in ns per decision.

The learned output is a ranking. It is never passed in as an observation.

## 6. S1–S8 mapped to parts

| falsifier | part | note |
|---|---|---|
| S1 receptivity survives Wisdom | C3b | reopen-on-violation rule (Maslov probe T9 precedent) |
| S2 no certification change from signals | C3a/C3b | bits 59..63 compared before and after |
| S3 replay is not evidence | done (L7) + C3b | ledger already pins it; extend to scheduler labels |
| S4 productive scent vs noise | C3b | STAUNEN-0 partial; add discriminating-ablation target |
| S5 revision is replayable | C3b | Thought-Git version kept; WORLD untouched |
| S6 encoding isolation | §2 decision | `EdgeProvenance` already refuses; needs the declared reading |
| S7 no forced maturation | C3b | compute, branch count and recipe count do not advance a rung |
| PZ1–PZ8 | C3c | §5 |

## 7. Open decisions

1. The canonical maturation order for bits 50..52, and the per-class
   declaration of maturation versus plasticity (§2).
2. Whether the `learn` gate's behaviour on `deepnsm-v2` edges is a live
   defect, which depends on whether those edges ever reach `learn`.
3. Where the TFLite kernels live (§5).

## 8. Delivered: first C3 slice

Source pins: lance-graph main `694b243bd`, ndarray master `2cbe131`.

### Owner and overlap map

| file | owner contract | change |
|---|---|---|
| `lance-graph-contract/src/epistemic_state5.rs` | D-EPI-CANON-0 | adds `RecipeRule` (additive) |
| `lance-graph-contract/src/certification.rs` | P7a #1369 / D-PEARL-PROD-0 | adds `CertificationModel::measured_only` (additive) |
| `lance-graph-planner/src/nars/warrant.rs` | D-EPI-COT (new) | new module |
| `lance-graph-planner/src/pearl.rs` | D-PEARL-PROD-0 | unchanged; still the only writer of bits 59..63 |
| `cognitive-shader-driver/examples/shared/affordance_law.rs` | D-GSO-AFF-0 | unchanged; still uses its own rule type (follow-up) |

EXO-50, read to its latest message (08:45Z): no open message claims these
files. Numeric admission stays with EXO-152.

### What runs

- **`RecipeRule`** (contract): `requires` / `forbids` over declared facts, a
  per-state check and the V1 population. This moves the lab law's rule into the
  contract. The lab still has its own copy; switching it over is a follow-up.
- **`CertificationModel::measured_only`** (contract): units whose outcome
  was not measured are excluded from every mask, including the trial arms,
  instead of being read as negatives. An arm left with no measured unit
  refuses with `EmptyArm`.
- **`nars::warrant`** (planner):
  - **`Marker::ground`** builds a marker from three things: the ledger's
    pooled truth and roots, the bits 59..63 projected through
    `Epi5Declarations`, and the thought slot and generation. It refuses
    with `NoEvidence`, an `Epi5` read error, or `Slot`.
  - **`tactic_rule`** gives the eligibility rule for recipes 4, 6, 7, 8
    and 11. These are policy pins, documented in the module.
  - **`fire`**:
    1. checks every premise against its rule;
    2. refuses a stale or duplicate marker;
    3. runs the planner's own RCR, TR or CAS function;
    4. admits each candidate through `DerivationLedger::admit`.

    Each judgment carries the tactic's truth, the roots of the proof through
    the new derivation, the thought, and `Unknown × Open`.

### Falsifiers

Each is disable-verified: the named disable turns its test red.

| falsifier | test | disable |
|---|---|---|
| acceptance path, roots preserved | `a_tactic_consumes_grounded_markers_and_keeps_their_roots` | — |
| eligibility follows declared facts | `eligibility_follows_the_declared_facts` | skip the rule check; drop RCR's `forbids` |
| no marker without evidence or declaration | `a_marker_needs_evidence_and_a_declared_reading` | skip the slot check |
| no promotion from premises; only trial arms promote | `a_judgment_is_never_promoted_by_its_premises` | judgment takes a premise's state |
| replayed thoughts add no evidence | `replaying_a_step_in_many_thoughts_adds_no_evidence` | — (ledger L7) |
| stale marker refused | `a_stale_marker_is_refused` | skip the staleness check |
| duplicate premise refused | `a_duplicate_premise_is_refused` | skip the duplicate check |
| UNKNOWN is not negative | `unmeasured_outcomes_are_not_read_as_negatives` | trial arms not restricted (red only after adding unmeasured control units) |
| rule population | `a_recipe_rule_population_matches_its_per_state_check` | `holds` ignores `forbids` |

L1–L5 are already pinned by `pearl.rs` tests; this slice does not duplicate
them:

- L1: `f1_observation_alone_never_promotes_to_causes`.
- L2 and L3: `cui_bono_script_moves_the_state_only_where_evidence_earned_it`,
  which ends at `IndirectKnown × Causes`.
- L4: `hydration_only_promotes_the_edge_whose_path_it_is` and
  `f5_a_proposed_beneficiary_is_not_evidence`.
- L5: `f7_one_at_a_time_removal_is_not_the_definition_of_causality`, the
  overdetermined trial.

### Clippy

`cargo clippy -p lance-graph-planner --lib --tests -D warnings` reports the
same 8 sites on main `694b243bd` as on this branch:
`nested_bands.rs` 601/978/1025, `cache/nars_engine.rs:1413` and
`tests/moore_nars16_probe.rs`. `--lib` alone is clean.

### Still open

- **Statistics:** effect size, sample-size bounds, uncertainty, and
  multiple-comparison correction across the up to five populations
  `associated` tries. Today `CertificationModel` reports none of these, and
  nothing here asserts more than it reports.
- **Missingness:** missing-at-random is assumed by `measured_only`; it is
  not tested.
- **ASC and CR (7, 11)** are not run through `fire`. They are revisions,
  and the ledger already pools them.
- **The lab affordance law** still has its own rule type.
- **C3b and C3c,** which wait on the decisions in §7.

## 9. Ledger → `pearl::hydrate` seam

**Existing:** `planner::pearl::{reason, revise, hydrate}` (#1390/#1391) already
writes bits 59..63. `hydrate` moves `IndirectUnknown` to `IndirectKnown` when
both bindings `A → B` and `B → Y` are in the sealed chain. None of the files
named `hydrate.rs` is that gate:

- `deepnsm-v2/src/hydrate.rs` addresses parsed triples to their verse;
- `lance-graph/src/graph/hydrate.rs` loads bgz7 vectors into Lance;
- `bgz-tensor/src/hydrate.rs` is a download binary.

**Added:** `nars::mediator::hydrate_from_ledger`.

1. It finds the ledger's grounded deductions of `A → Y` through `B`.
2. It hands the path to the unchanged `pearl::hydrate`.
3. It returns a receipt: derivation ids, source roots, the `Hydration`
   result, the edge after `hydrate`, and `A → Y`'s sensitivity with the
   mediated derivations removed.

What each side is allowed to do:

- **The ledger never creates a chain.** A ledger path whose links are not
  both in the sealed chain leaves the edge unchanged.
- **The adapter refuses a chain-only path.** If the ledger has no grounded
  path through `B`, there is no receipt.
- **Only subject and object are read** from each chain step. The step tag is
  not a predicate; that was the #1572 guard, which was withdrawn.

To record causal mediator paths, `Law::Deduction` now admits `Impl` in an
explicit step:

- `Copula::transits` still keeps `Impl` out of blanket closure, and `Rel`
  still never composes.
- The step earns NARS truth only. It certifies nothing.

| falsifier | test | disable |
|---|---|---|
| chain-witnessed ledger path hydrates (raw5 22 → 21) with roots | `a_ledger_path_witnessed_by_the_chain_hydrates` | — |
| ledger path alone does not hydrate | `a_ledger_path_alone_does_not_hydrate` | — (`pearl::hydrate` owns it) |
| grounded path through B required | `the_adapter_needs_a_grounded_ledger_path_through_the_mediator` | accept ungrounded derivations; accept any mediator |
| redundant mediated path not reported as necessary | `a_redundant_mediated_path_is_not_reported_as_necessary` | — |
| edge must be the target | `the_edge_must_be_the_target_relation` | skip the edge check |
| `Impl` chains, `Rel` never | `l8_an_invalid_figure_is_not_admitted` | `Rel` composes; `Impl` refused |

**One disable did not bind on its first form.** Falling back to "any proof"
changed nothing, because without evidence there are no proofs at all.
Accepting derivations without a proof does bind, and that is the form
recorded.

**Maturation (operator, 2026-10-28):** `7 = Staunen … 0 = Wisdom` (the #1494
order) is the order for new, explicitly declared maturation readings.

- Wisdom stays revisable: an independently witnessed contradiction reopens
  it.
- No existing class is reinterpreted. `deepnsm-v2` keeps its opposite
  `novelty_wisdom` order.
- Before any production write, four things must be settled: the reading,
  its provenance, a migration, and the `CausalEdge64::learn` collision.
- Until then, maturation stays transient and fail-closed.

**TFLite (operator, 2026-10-28):** the kernels are in `AdaWorldAPI/ndarray`:

- `src/simd_fixedpoint.rs`: TFLite fixed-point arithmetic;
- `src/simd_stride.rs`: `lut9`;
- `src/hpc/codec9.rs`: Palette512.

`AdaWorldAPI/tflite-micro` is the reference, and lance-graph #1800 has the
softmax oracle. No `lance-graph-tflite` crate. No training kernel exists yet.
