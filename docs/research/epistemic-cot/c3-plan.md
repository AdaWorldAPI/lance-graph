# D-EPI-COT C3 — plan: Epi5 bridge, scenting, numerical learning

Status: PLAN. Nothing here is implemented. C1/C2 (`nars::support`) is unchanged
by this document.

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
