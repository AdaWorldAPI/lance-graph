# Audit — the 34 NARS recipes against the CURRENT substrate

> 2026-09-14. Read `recipes.rs` (all 34) against `ndarray::simd_masking_ops`
> (the masking algebra) and the workspace's own iron rules. The question asked:
> **what needs improving.**

## ⊘ CORRECTION — `bucket` is NOT a dispatch key (read after writing this)

The audit below says the stale `bucket` column matters because it routes. **It
does not route.** Measured:

- `recipe_dispatch.rs` — **zero** uses of `bucket`. Dispatch is by
  `RecipeInference` (deduction/induction/abduction/revision/counterfactual),
  `rung` + `dispatch_order` (from `Tier` + `InferenceType::rung_delta`), and
  the `nan_disqualifier` checklist.
- `recipe_kernels.rs` — **one** use, the default `Tactic::gate`:
  `match bucket { Bucket::Gate => ctx.gate_state() != GateState::Flow, _ => true }`.

So only the **`Gate`** arm is load-bearing, and `Gate` is the one bucket this
audit found ACCURATE (TCP/CAS/TCF/CUR are the prune/cascade family the masking
algebra genuinely expresses). `Datapath` vs `Control` is **inert descriptive
metadata that nothing reads.**

Consequence: the 9 stale `Datapath` strings are **documentation debt, not live
mis-dispatch.** They still want fixing — a string naming a retired kernel reads
as a spec — but item 1 below ("re-derive bucket") is NOT urgent and NOT a
correctness risk. I claimed the opposite in session ("a routing key derived
from a retired substrate is currently deciding what executes") without reading
either module. That claim is withdrawn.

## ⊘ ALSO: the loco↔34 bridge already exists — do not build it

`lance-graph-ogar/src/recipe_vocab.rs` is the 34 as loco ops:
`RECIPE_OP_BASE = DOMAIN_FLOOR`, ids `1..=34` ↔ bytes `0x90..=0xB1`,
`op_of` / `recipe_of`, `impl Vocabulary for RecipeVocabulary`, and
`ladder_program() -> Vec<FnIndex>` — the ladder already lowered to a loco
program in `dispatch_order`. It carries two gates that answer different
questions (awareness: the kanban census; epistemic: the NaN pothole) with
`refusal_of` reporting which spoke, "because an unwilling ladder and an unable
one are different diagnoses."

Its header also settles the dependency question correctly: *"`ogar_loco` is
zero-dep by design and `lance_graph_contract` is zero-dep by charter. Neither
may import the other. A vocabulary needs both, so it lives in a consumer that
already depends on both."* An `ogar-loco/src/nars.rs` with a contract dep —
which this session started writing — is the wrong home for a reason already
recorded.

## The finding that IS serious, and is not mine

`E-RECIPE-SELECTOR-REACHABILITY-1`, measured and documented in
`recipe_dispatch.rs`'s own header: through the shipped saccade selector
(`materialize::select_tactic`) only **8 of 34 recipes are reachable**, **all 14
Infrastructure recipes never win**, and **ICR #31 is permanently shadowed by
RCR #4** on a lowest-id tie. The `ladder` mode exists to cover exactly those
three gaps, and `dispatch_mode.rs` routes between the two modes. That is a
larger fact about the 34 than anything in the substrate-string audit below.

## Headline: the `bucket` column was assigned against a substrate that is gone

**9 of 9 `Datapath` recipes name a retired or forbidden realization.** The tier
that is supposed to BE the masking ops is the tier most contaminated:

| id | code | substrate string | verdict |
|---|---|---|---|
| 19 | ARE | `ABBA unbind: A⊗B⊗B=A` | VSA — retired |
| 24 | ZCF | `VSA bind(A,B)` | VSA — retired |
| 25 | HPM | `fingerprint cosine/Hamming sweep` | Hamming — retired |
| 27 | MPC | `bundle = majority-vote-per-bit` | VSA bundle — retired |
| 28 | SSAM | `bind+similarity (Gentner)` | VSA — retired |
| 34 | HKF | `cross-domain bind(A,rel,B)` | VSA — retired |
| 14 | MCT | `… → one fingerprint` | fingerprint — retired |
| 12 | TCA | `Markov ±5` | **±5 was retired** by the whole-book finding (63.3% of same-subject links reach beyond ±5) |
| 32 | SDD | `Berry-Esseen noise floor` | **forbidden by `I-NOISE-FLOOR-JIRAK`** — classical Berry-Esseen is wrong under this system's weak dependence; Jirak 2016 is the rule |

So the `Datapath` label currently means *"was a VSA kernel"*, not *"is a masking
op"*. **I asserted earlier in this session that `bucket` is still a valid
routing column and only `substrate` was stale. That was wrong** — the bucket was
derived FROM the substrate, so it inherited the staleness.

## What the masking algebra DOES express today

Cross-checking the 34 against the real op list (`eq/ne/lt/le/gt/ge_*_to_mask`,
`ternary_match_*`, each with a gated `_under` twin; `mask_and/or/xor/andnot/not`;
`mask_ternlog<IMM>` over the full 256-table; `mask_any/all`; `masked_sum/min/max_i32`,
`masked_strided_group_sum`):

| id | code | bucket | expressible as | note |
|---|---|---|---|---|
| 5 | TCP | Gate | a gated predicate — `*_under` | prune = don't evaluate where the gate is empty |
| 8 | CAS | Gate | a cascade of `_under` tiers | the INT1/4/8/32 ladder is the gate chain |
| 20 | TCF | Gate | N masks → agreement | `ternlog` majority + `popcount` |
| 26 | CUR | Gate | coarse-to-fine `_under` chain | same shape as CAS |
| 30 | SPP | Control | N independent masks → agreement | identical algebra to TCF; the ECC/RAID framing is the same majority |

**The recipes the masking algebra can run TODAY are 4 Gate + 1 Control — and
zero Datapath.** The tier assignment is inverted relative to the substrate that
actually exists.

## Other defects found

- **22 ETD** — `"CLAM cluster geometry determines subtasks (no spec)"`. Says so
  itself. Unspecified, not merely stale.
- **18 CWS** — `"persistent BindSpace"`. The singleton BindSpace was retired
  (`E-MARKOV-TEMPORAL-STREAM-1`: *"in most cases the singleton-BindSpace VSA
  substrate is NOT used"*).
- **31 ICR** — `"CausalEdge64 −6 mantissa"` cites the **v2** layout. Truth
  (frequency/confidence) is moving to **v3**, so this reference needs re-pinning
  with that change rather than after it.
- **1 RTE** — `"Berry-Esseen stop"`. Same iron-rule violation as SDD, in a
  Control recipe.

## What needs improving, in order

1. **Re-derive `bucket` from the masking algebra**, not from the VSA substrate.
   The five above are the honest `Datapath` set today; the current nine are not.
2. **Rewrite the 9 stale `substrate` strings** to name mask-algebra
   compositions, or mark them `unrealized` — a string naming a retired kernel
   reads as a spec and is worse than an empty one.
3. **Fix the 2 Berry-Esseen citations** (1 RTE, 32 SDD) to Jirak 2016, per
   `I-NOISE-FLOOR-JIRAK`. These are iron-rule violations sitting in shipped data.
4. **Re-pin 12 TCA** off the retired ±5 window onto the version-range read
   (`QueryReference::at(v, rung)`), which is what replaced it.
5. **Re-pin 18 CWS** off the singleton BindSpace.
6. **31 ICR** rides the CausalEdge64 v3 change.
7. **22 ETD** needs a spec or an explicit `unrealized` mark.

## What this audit does NOT claim

It does not say the 34 are wrong as *tactics*. It says their recorded
realization is against a substrate that no longer exists, and that the routing
column derived from it cannot be trusted as a dispatch key until re-derived.
Nothing here was executed; this is a read of the catalogue against the op list.
