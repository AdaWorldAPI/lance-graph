# Why the 34-recipe schedule has depth 1 — read from source, not run

Source tree: /home/user/lance-graph at origin/main 3f8d13c. Nothing was compiled or executed.
Every claim below is from reading the cited lines.

## 1. The vocabulary of `requires` / `writes`

- Type: `ThoughtMask(pub u8)` (recipe_kernels.rs:213), a bitmask over the eight `ThoughtField`
  variants (recipe_kernels.rs:187-204): Sd=0, FreeEnergy=1, Dissonance=2, Temperature=3,
  Confidence=4, Rung=5, Candidates=6, Beliefs=7.
- So `requires`/`writes` name **struct FIELDS of one shared scratch context** (`ThoughtCtx`,
  recipe_kernels.rs:46-63), not objects, not statements, not versions. `writes` is documented
  as "POSSIBLE writes" (recipe_kernels.rs:338-352); `requires` as "what this tactic reads"
  (:354-363).
- "Known" is a groundedness flag per field: `known_from` (rung_schedule.rs:158-178) sets a bit
  iff the scalar is not NaN, or the Vec is non-empty; `Rung` is always set (:169).

## 2. The scheduler

`schedule(initial_known)` (rung_schedule.rs:188-227): a recipe is ready iff
`requires().covered_by(known)` (:196); all ready recipes form one wave; afterwards
`known |= union(writes of the wave)` (:218); repeat until no recipe is ready. A second wave can
only appear if some wave writes a field bit that was **not already known**.

## 3. The regression test

`der_plan_ist_flach_weil_kein_rezept_fuer_ein_anderes_produziert`,
rung_schedule.rs:321 (doc 304-319). It asserts `depth() == 1` and
`known_before | writes == known_before` for exactly four masks: EMPTY, 0b0010_0000 (Rung),
0b0011_0010 (FreeEnergy+Confidence+Rung), 0xFF (all) (:322-327).

## 4. Why it is depth 1 on those masks

Every kernel but one writes only fields it also requires (read-modify-write). Per kernel
(`writes` ⊆ `requires`): Rte {Rung,FE}⊆{FE,Rung}; Htd/Rcr/Tcp/Tr/Cas/Irs/Tca/Cdt/Mct/Pso/Tcf/
Etd/Hpm/Cur/Mpc/Idr write Candidates and require Candidates; Cdi {Beliefs,Dissonance} =
requires; Cws {Beliefs} ⊆ {Candidates,Confidence,Beliefs}; Amp {Rung} ⊆ {FE,Rung};
Dtmf {Temperature} ⊆ {Sd,Temperature}; the 13 others write nothing (Smad, Asc, Mcp, Cr, Ssr,
Ssam, Spp, Icr, Sdd, plus Demonstrations Are/Zcf/Hkf write nothing and require nothing).
So once a recipe is ready, every bit it writes is already in `known`; `known` never grows; a
second wave never forms. The test's own doc (rung_schedule.rs:306-311) says the same: `writes`
is a read-modify-write vocabulary, not a producer/consumer vocabulary.

## 5. The single exception — depth 2 IS reachable, but untested

`Lsi` (id 15) requires {Candidates} and writes {Sd} (recipe_kernels.rs:869-895). It is the
only kernel whose `writes` is not a subset of its `requires`. Reading the code: for
`schedule(ThoughtMask(1 << 6))` (Candidates known, Sd not), wave 0 contains Lsi, wave 0 writes
Sd, and wave 1 would contain Tcp (5, requires {Candidates,Sd}) and Ssam (28, requires {Sd}).
That mask is not among the four the test checks. Via `schedule_for` it arises when the
context has non-empty candidates and `sd = NaN`. So the honest statement is "depth 1 on the
four tested masks"; the module's own header claim ("Tiefe 1, immer", :304) is not established
for all inputs. Not executed here — this is a prediction from the masks.

## 6. Producer -> consumer pairs that exist or almost exist

| producer | field | consumer | status |
|---|---|---|---|
| LSI 15 | Sd | TCP 5, SSAM 28, DTMF 33 (declared reads) | real field edge; visible to scheduler only when Sd starts ungrounded (§5) |
| LSI 15 | Sd | gate() of Gate-bucket TR 6, CAS 8, CDT 13, TCF 20, CUR 26 | real read, UNDECLARED: `Tactic::gate` reads `ctx.sd` via `gate_state` (recipe_kernels.rs:366-371, :80-88) but those kernels' `requires` omit Sd |
| RTE 1 | FreeEnergy | AMP 23, MCP 10, SSR 21, ICR 31 | real value dependence in sequential execution (`run_recipes` doc, recipe_dispatch.rs:291-293: "Rte spends free_energy … before Amp ever reads"); invisible to the scheduler because consumers already require FE |
| RTE 1 / AMP 23 | Rung | CAS 8, RTE 1 | same: value dependence, no new bit |
| CDI 17 | Beliefs | CR 11 | semantically real: the injected (topic, 0.1, 0.6) trips CR's |df|>0.5 rule when beliefs[0].f > 0.6; invisible (both require Beliefs) |
| CWS 18 | Beliefs | CR 11, CDI 17 | CWS over-declares Beliefs in `requires` (apply only appends, recipe_kernels.rs:971-976); with an honest mask it could produce Beliefs from an empty set — the would-be edge is hidden by the over-declaration |
| DTMF 33 | Temperature | TR 6, IRS 9, CDT 13 | value dependence, no new bit (DTMF requires Temperature) |
| 17 Candidates-writers | Candidates | 21 Candidates-readers (e.g. HTD 2 reorders, then SMAD 3 / SPP 30 / SDD 32 read spread/mean) | value dependence on every collapse-to-one (MCT/MPC/TCF/HPM/IDR/CDT/CUR) — invisible |

## 7. Which of the three explanations holds

- "No recipe consumes another's output": **false as stated.** Value-level consumption exists in
  sequential execution (RTE->AMP, CDI->CR, collapse->SMAD/SPP); the run_recipes doc says so.
- "Consumption exists but the scheduler doesn't see it": **true.** The scheduler only tracks
  whether a field is grounded (one bit), not which execution last wrote it. Every
  read-after-write on an already-grounded field is invisible, so recipes that do interact are
  placed in the same wave and labelled parallel-safe. The module header's justification
  ("weil keines ein Feld liest, das ein anderes derselben Welle erst schreibt",
  rung_schedule.rs:31-33) is false for e.g. HTD (writes Candidates) and SMAD (reads Candidates)
  in one wave. In practice `wave_dispatch::dispatch_thought` (wave_dispatch.rs:61-87) hides
  this by giving each rung lane its own `seed.clone()`, so lanes never see each other's writes
  at all, and within a lane `run_recipes` runs sequentially in (rung,id) order.
- "Outputs aren't addressable, so consumption can't be expressed": **also true, and it is the
  root cause.** `Tactic::apply(&self, &mut ThoughtCtx) -> Outcome` (recipe_kernels.rs:373);
  `Outcome` carries only `fired`, a `&'static str` note, `delta_conf`, `skip` (recipe_kernels.rs:101-114)
  No recipe emits an object with an identity; `ExecutedStep` (recipe_dispatch.rs:282-289)
  exists but no recipe can receive one. `dispatch_thought` discards both the ctx copy and the
  steps (`let _steps = …`, wave_dispatch.rs:75). The nearest thing to an output object is CWS's
  belief tuple tagged with the shared sentinel topic `u32::MAX` — not an address.

## 8. What would have to be true for a second wave

1. Either the context starts with a field ungrounded that some recipe writes from other fields
   (today only Sd via LSI, §5) — trivially depth 2, but meaningless as reasoning; or
2. `writes`/`requires` would need a vocabulary of **produced objects** (e.g. "a derived
   statement of kind K", "a receipt for step S", "a revised belief on topic T") rather than
   struct fields, so that a consumer can require something no measurement provides and only a
   prior recipe produces; and
3. `apply` would need an input channel for prior objects (receipts / derived statements by
   identity) and an output channel that mints them — today the trait has neither.
The test at rung_schedule.rs:321 is designed to fail the moment (1) or (2) happens.

## 9. Things that must not be conflated (definitions and owners)

- **Rung levels 0..9 — two different ladders.**
  - `RungLevel` enum Surface=0 … Transcendent=9, cognitive_shader.rs:157-169 (owner: contract
    `cognitive_shader`). Used by `Recipe::min_rung` / `admissible_at` (recipes.rs, derived from
    Bucket+Tier; Gate->Surface, Datapath->Contextual, Control->Analogical, ExtremelyHard floor
    Counterfactual) and `RungLevel::for_pass`.
  - `recipe_dispatch::rung(id) -> u8` in 1..=9, recipe_dispatch.rs:175-184: tier base
    (CrossTier 1, Hard 3, ExtremelyHard 5) + inference delta (Ded 1 … Counterfactual 5),
    clamped. This is what `rung_schedule` sorts by and what `AlphaTunnel` lanes are indexed by.
    Actual values used are 2..9; lane 1 is never used and lane 0 is reserved (rung_schedule.rs:57-59).
  - `ThoughtCtx.rung: u8` "1..=9" (recipe_kernels.rs:57-58) is a third, mutable scalar that
    RTE/AMP raise; it is not the recipe's rung.
- **The 34 registered recipes**: `RECIPES: [Recipe; 34]` catalogue (recipes.rs:87), kernels
  registered by the `kernels!` macro, `kernel(id)` / `all_kernels()` (recipe_kernels.rs:1522, :1529),
  ids 1..=34. Owner: lance-graph-contract.
- **64-entry recipe affordance surface**: `RecipeLaw[64]` / `EligibleRecipes: u64`. In the
  contract it is ONLY a doc comment (epistemic_state5.rs:31-34). The executable law lives in an
  EXAMPLE, cognitive-shader-driver/examples/shared/affordance_law.rs:76-121 (`Rule`, `law()` ->
  `[Rule; 64]`), with only 7 slots populated (0 HYDRATE_INTERMEDIATE, 1 MECHANISM_FOLD,
  2 STRATIFY, 3 ROBUSTNESS_TEST, 4 CAUSAL_IDENTIFICATION, 5 OBSERVE_FOLD,
  6 COUNTERFACTUAL_PROBE; :104-110). These 64 slot ids are a **different vocabulary** from
  the 34 tactic ids — no mapping exists between them. Board: STATUS_BOARD D-GSO-AFF-0 (#1370),
  "transient … no scheduler/task/capability object".
- **64 strategy slots / StrategyLane**: not found. `grep -rn "StrategyLane"` over /home/user
  (all .rs) and over lance-graph .rs/.md returns nothing. Search space: every .rs under
  /home/user and every .rs/.md under /home/user/lance-graph (excluding target/). Absent under
  that name; the planner's "16 strategies" are a separate surface not examined here.
- **AlphaTunnel overlays**:
  - `AlphaOverlay<'a>` (alpha.rs:638), a thin-provisioned overlay of 512-byte `NodeRow`s at the
    same addresses as a borrowed base spine; `AlphaAllocation::over` (alpha.rs:556) allocates
    the address space at zero rows; `claim` (alpha.rs:707) materialises one row.
  - `AlphaTunnel<'a>` (alpha_tunnel.rs:73): `LEVELS = 10` (rung_schedule.rs:59) overlays over ONE
    shared allocation, one per `recipe_dispatch::rung` value; `run_wave`/`run_wave_parallel`
    hand each rung its own lane (:124-175); `merge` folds lanes in (rung, seq) order (:181-221).
  - **An unclaimed row is ABSENT, never baseline.** `AlphaOverlay::get` returns `None`
    ("not attended") and "Never falls back to the base row" (alpha.rs:763-769; module doc
    alpha.rs:24-28). Pinned by `an_unclaimed_address_reads_as_not_attended` (alpha.rs:1494).
  - The overlay records WHERE attention went (addresses + AlphaStamp), not recipe outputs; in
    `dispatch_thought` a lane claims addresses of candidates its execution moved
    (wave_dispatch.rs:73-80) and the recipes' ctx/outcomes are dropped.

## Addendum (orchestrator, 2026-10-11) — §5 MEASURED

The §5 prediction was run. A temporary test called `rung_schedule::schedule` on
every one of the 256 initial `ThoughtMask` values (the test was deleted after
reading). **64 of 256 masks schedule depth 2**: exactly those with Candidates
known and Sd unknown. The second wave is recipes `[5, 28]` (TCP, SSAM), plus 33
(DTMF) when Temperature is also known. So the doc comment on
`der_plan_ist_flach_weil_kein_rezept_fuer_ein_anderes_produziert`
(`rung_schedule.rs:321`), which says depth 1 "immer" (always), overstates what
its four sampled masks show. The only edge is LSI writing Sd, i.e. a recipe
producing a measurement-shaped field.
