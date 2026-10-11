# Scientific recovery: NARS × CE64 × Moore/Morton (read-only, 2026-10-10)

State read: lance-graph `origin/main` 1573eb4c (incl. #1464), ndarray `origin/master` 8afa92d (incl. #354), OGAR `origin/main` 864817f, as detached read-only worktrees under `/home/user/_ro/`. No edits, commits, pushes, resets or PRs. Working branches untouched (lance-graph branch is 34 behind main, ndarray 1 behind, OGAR 16 behind; none altered). #1465 is OPEN with an unverified P1 fix and is treated as not landed.

Source reports (file:line evidence, graded): `recovery/A-pr-ledger.md`, `B1-semantic-substrate.md`, `B2-execution-substrate.md`, `B3-hdr-statistical-substrate.md`, `B4-statistics-mapping.md`, `B5-learning-seam.md` (+ `sim.py` replica, `raw/` PR bodies).

Evidence grades used: VERIFIED-IN-CODE (file:line read), VERIFIED-IN-PR-BODY (reported measurement, not re-run), REPLICA (scratch python, not repo code), OPEN.

---

## A. Original intent vs actual work

### A.1 The assignment (as restated by the operator 2026-10-10)
Seven research questions (Q1 local learning via NARS edges × Moore; Q2 separate semantic / epistemic / statistical state; Q3 Moore/Morton for propagation, learning, counterfactuals; Q4 emergence without NNUE/heavy orchestration; Q5 masks/popcount/staged reductions/thresholds cut search; Q6 discovered primitives with falsifiable semantics; Q7 progressively more selective computation). Tenants 18–20 were proposed by the operator as experimental representations.

**Provenance finding (VERIFIED, A-ledger §5):** no committed artifact contains an operator-authored proposal of tenants 18–20. Earliest occurrence: `MooreNars16` in 3581b1f5 (#1406), `Nars16x8`/`MoorePalettePairs` in 28ec7c07 (#1408), both authored by this session. The only committed operator line nearby is "Operator priority (2026-10-08): CE64 + Moore + masking first". The operator's statement of 2026-10-10 is therefore the record of authorization. **DECISION recorded here:** tenants 18–20 = AUTHORIZED EXPERIMENT (operator direction, chat), NOT formally ratified (no batched OGAR mint, no envelope-auditor verdict). My earlier audit's "self-authorization" label for V2 was wrong and is withdrawn.

Category discipline used below:
| category | tenants 18–20 | strict CE64 ISA (#1407) | signed Register128 (#1410) | lab statistics |
|---|---|---|---|---|
| operator-authorized direction | YES (chat 2026-10-10) | partially: "CE64 op-set hardening … PROBE" (probe, not API change) | not recorded | yes as research harness (D-SELF-CALIBRATING-LAB) |
| hypothesis | Moore-local NARS state carries learning | illegal opcodes should be visible | registers need signed readings | — |
| experimental representation | YES | — | YES | — |
| production implementation | NO production reader/writer (B1 (b)) | YES (replay path) | dead in production: no authority declares a law (B1) | example code |
| formal canonical ratification | NO | NO | NO | NO |
| experimental finding | observational equivalence (#1406/#1408) | 55/126 legacy vectors use undefined codes | none | 13 verdicts + certificate |

### A.2 Hypothesis → apparatus → result matrix
| # | hypothesis | existing apparatus | implementation used | result (grade) | still unproven |
|---|---|---|---|---|---|
| H-1400 | persistent structural learner beats fixed/reset on Gomoku | none used from substrate | `HashMap<u64,[Ev{pos:f64,neg:f64};9]>` (gomoku probe :811-866) | +0.0238 vs best fixed, CI [+0.006,+0.0415], sign-flip p=0.0156, 16 blocks (#1440 H2) VERIFIED-IN-PR-BODY | that any canonical representation carries this learning |
| H-1406/1408 | Moore-local CE64 representation is ISA-observationally equivalent | CE64 ops (forward/learn/syllogize) as oracle | tenants 18–20, `lift_ce64`, MooreTenantView/Mut | 0 divergences over 300×8×5 and 256×8×5 (except Direction3); not bitwise (≥600/800 words differ) VERIFIED-IN-PR-BODY+code | revision state (F/C not carried: `lift_ce64` drops bits 0..39; test copies F/C from the source edge, B5) |
| H-P1 | certified minimax cuts work without changing decisions | minimax + bound | `Cfg.certified` | 0/400 changed picks; −36.2/−44.8/−61.0% folds at k=4/5/8; +28.9 folds/step saved in game, identical games 16/16 VERIFIED-IN-PR-BODY | nothing — deterministic certificate |
| H-P5 | NARS revision ≈ Beta posterior | causal-edge revision | tests only | e = Jeffreys posterior mean exactly; u8 confidence stalls at c=244 after ~20 obs (32/32); behaves as fixed window past it; 4 copies understate sd 2.04×, 8 copies 3.48× | — (these ARE the findings) |
| H-T | Thompson sampling helps | lab sampler | lab.rs gamma/beta | T1 −0.050 [−0.062,−0.038], T2 −0.0153, T4 +18.5 folds: all FALSIFIED | — |
| H-T3 | structural context helps | lab | gomoku key | +0.0247 [+0.005,+0.044], p=0.0186 SUPPORTED; global key ~32 vs ~245 folds/step | — |
| H-MHB | Mexican hat without raster | (did not use) helix LUT / mask-risc | mexhat_bucket_probe (f64 DoG, signed bit planes) | weighted lane 26–150× slower (negative, valid) | any spatial claim on Morton |
| H-1426 | NarsTables fast path earns its cost | NarsTables | probe | B1 flips ~38% of verdicts; no decision-stable table; no production reader | — |
| H-1428 | certified decisions before Moore field converges | f64 stencil + M-matrix bound | MoorePertubationsfeld | certified never wrong; LODF 17 ms ≫ PCG 170 ms, Jacobi 2.7 s | — |
| H-1432 | counterfactual crossings pay | — | probe | 0.03–0.81× baseline (do not pay); chain folding 1000/1000, breaks even after 15–36 queries | — |
| H-1427 | Moore perturbation A/B | — | probe | **no numbers reported, compile never observed, merged** | everything |

**Three findings kept separate:**
- **A. Behavioural learning (VALID RESULT):** #1400/#1440 H2/T3 — a persistent structural learner improves recipe selection. Carrier: f64 HashMap. Does NOT involve tenants 18–20 (VERIFIED-IN-CODE: zero call sites in the learning loop).
- **B. Representation (VALID RESULT):** #1406/#1408 — Moore-local CE64 lanes are observationally equivalent under forward/learn/syllogize, with correct refusals. Does NOT prove learning: MooreNars16 holds no F/C.
- **C. Statistics (VALID RESULTS):** certificate, context advantage, confidence saturation, duplicate-evidence inflation, Thompson falsified. Keep the numbers and falsifiers; relocate the methods (§C).

---
## B. Canonical substrate inventory (condensed; full tables in B1–B4)

### B.1 Semantic identity and storage
| primitive | location | contract | tests | observed limitation |
|---|---|---|---|---|
| Reading resolution (post-#1464) | contract `hotplug.rs:30-48`; `Activation::resolve_for_context`, `resolve_tenant_reading` | "How a row is read comes from the plug + the slab's own SlabDeclaration. Nothing else selects a reading." Slab declaration wins for written data; no per-concept table | hotplug tests; #1464 `_plugged` forms (ocr, aiwar, soa_graph, nan_projection) | fallback still live: unknown classid → `ReadMode::DEFAULT` (V1/Full) `canonical_node.rs:1850-1857`; `NodeGuid::read_mode()` :327 still legacy; W6 not started; only `0x1717` has `concept_override` (`lance-graph-ogar/src/lib.rs:476`) until #1465 |
| SlabReading / SlabDeclaration | `hotplug.rs:231-257` | tag-coded reading of a slab | built only in tests | **no variant for tenants 18–20**; tags 2..=7 have no callers outside hotplug.rs; never serialized |
| register_laws | `Activation::with_register_laws` | law per concept from the authority | tests only | **no production caller in lance-graph/OGAR/ndarray** → `bind_signed_register` cannot succeed in production |
| ClassView | `class_view.rs` | class-level interpretation: `value_schema`, `edge_codec_flavor`, `cascade_shape` | 16 implementors | all three still classid-keyed; none overridden; `CascadeShape` and Epi5 declaration have no `ReadMode` field → plug cannot supply them |
| FieldMask / WideFieldMask | contract | selects fields | — | one type carries 3 bit-spaces: class fields, ValueTenant presence, CausalWitness loci |
| ValueTenant / VALUE_TENANTS | `canonical_node.rs:1371-1395`; persistence rule :986-991 | contiguous 0–20, `Full.tenant_bytes()=300/480` | const asserts | 10–20 only in `Full` → Cognitive/Compressed rows cannot bind Register128 rails; offset moved 220→252→300 with no version bump (risk: `weather-poc` bakes W1 at `Full.tenant_bytes()`; none committed) |
| Moore tenants 18–20 | `moore_tenant.rs` | 8 lanes NW,N,NE,W,E,SW,S,SE; 18 U16×8, 19 U8×16 PairAddress, 20 Pearl3/Energy4/Plasticity3/Polarity1/Epi5 | equivalence tests (planner J, csd F) | raw `&[u8;480]`, no schema/reading gate; Epi5 decoded as `Epi5Gen::V1` without declaration (`:178`, Codex P2 unanswered); writes bypass `tenant_update` (Codex P2 unanswered); "Energy4" = CE64 inference mantissa; no production reader/writer |
| CE64 + EpistemicState5 | `causal-edge` layout/edge/isa.rs:208 | revision changes F,C only, keeps 53–63 | golden + contract tests | u8 C stalls at 244; c255+c255 → NaN→0 pinned |
| Palette256 / PairAddress / LUT | contract `awareness_facet.rs:32` `(u8,u8)`; `PairAddress`/`PaletteLut` in `cognitive-shader-driver/src/palette_perturbation.rs`; helix `DistanceLut` u16 256×256 `distance.rs:18` | LUT | helix metric tests | `from_floor` rounds; only linear/circular exact |
| Signed readings | B1 (d) | — | — | VALUE: Qualia 16×i4, I4x32, RegisterLaw::AxisPosition, ::Support, CoarseResidue. COORDINATE: CausalWitness loci, RegisterLaw::RelativeOffset, Signed360, MooreSlot offsets. OPCODE: CE64 mantissa. All masks unsigned. |

### B.2 Execution
| primitive | location | contract | limitation |
|---|---|---|---|
| quack Program builder | `lance-graph-quack/src/lib.rs` (`lower` :1383, `lower_group_by` :1407, `lower_group_avg`, `GroupAddr{Local,Via,Pair}` :1235, `GroupAgg` :1263) | builds, never evaluates | ordered compares i32 only; no shift/Moore op; spatial = Morton prefix → Range only |
| GroupAddr::Pair + SumI32 | quack :1542-1600, :2368 → mask-risc `GroupKey::Pair` / `GroupFold::SumSymI32` | `GROUP BY (hi,lo)` sum/count into K caller-owned i64 slots | Pair sum is NULL-preserving (empty group = seed marker, not 0) |
| mask-risc IR | `ir.rs` Pred (ordered: signed I32 only :129-135), Range :180 (index pred, order not checked), EqU32Via :158, Ternlog, one terminal :707 | fusers: Range, Ternlog ≤3, Tern2, Tern3, else Tiled (`TILE_WORDS=256`) | no shift/Morton op; any lane pred → tiled |
| zero-alloc gate | `mask-risc/tests/no_alloc.rs` | Counting allocator; 0 bytes over 1000 executes on 65,536 rows | — |
| ndarray masking | `simd_masking_ops.rs`; `mask_ternlog_popcount<IMM>` (simd.rs:673); `masked_group_count_u32_via` :1807; `*_u8_to_mask` unsigned | — | — |
| `mask_shift_morton` | :5384 | one-cell axis shift, OR-accumulates into dst, no wrap, 4^k words, slice = field | diagonal = 2 calls + scratch; no fused shift∧popcount anywhere; only Moore use = 1-word schedule in moore_plasticity_probe :387-464 |

### B.3 HDR / Belichtungsmesser (ndarray, B3)
- **Exact:** `hamming_distance_within` (bitwise.rs:221, early exit per 256 B; VPOPCNTDQ→BW→U64x8 dispatch), `packed::PackedDatabase` with both thresholds = T (lower-bound filter), `MomentsU32`/`moments_u32`, `PowerSums`/`CrossPowerSums` + `checked_merge`/`checked_affine` (None on overflow), bounded u8 folds (`BOUNDED_TILE_ROWS=65,536`), RollingFloor integer moments, nearest-rank quantiles.
- **Approximate / not an oracle:** `Cascade::query` Stroke 1 (scaled prefix + 3σ; test `stroke1_drops_a_prefix_heavy_true_hit` :813), `cascade::PackedDatabase`, 128-row preheat (storage order, not a sample), `expose` fixed fractions (doc `cascade_search.md` claims percentiles — drift).
- **Limits:** RollingFloor σ-lattice is lower-tail only; reservoir cap 1000, order-dependent, not mergeable; `for_width` is a Hamming binomial prior.
- **For Moore/Morton:** reusable unchanged = moments, PowerSums/CrossPowerSums, bounded u8 tile, quantiles, exact Hamming between two field states, `Cascade::calibrate/observe` on any u32 stream. Not reusable = cascade query/packers/preheat. HDR answers Hamming NN, not spatial autocorrelation.

### B.4 Statistics (B4)
- jc: Pearson/Spearman/α/ICC (`reliability.rs`), κ, ω, η², t-tests one-sample/paired/Welch/Student (`stats.rs:1017-1115`, `TTest{t,df,p}`), ANOVA from PowerSums, Fisher z. **No** CI, t-quantile, Holm, sign-flip/permutation, Prereg/Verdict, ECE/Brier, Gamma/Beta sampler. `jirak.rs:127` is a qualitative pillar test (rate only cited, not computed) → "use Jirak" has no callable function.
- ndarray: descriptive only (MomentsU32, PowerSums, RollingFloor, reliability returning 0.0 on degenerate input; private Box–Muller samplers).
- jc is workspace-excluded; cognitive-shader-driver has no jc dep.

---

## C. Duplication ledger (every facility this session built)

| facility | built where | existing equivalent | verdict |
|---|---|---|---|
| `lab.rs` sign_flip_p, holm, Prereg/Verdict/verdict, T975, ECE/Brier/Bin, gamma/beta samplers | csd examples/shared/lab.rs | **none equivalent** (B4 class 4); `paired` ≈ `jc::t_test_paired` but differs (exact sign-flip p + t-interval vs parametric p, no CI) | NOT duplicated substrate; WRONG IMPLEMENTATION LAYER (example instead of jc). Defects: T975 → 1.96 for df>30 (too-narrow CIs from 32 blocks); `higher_is_better` never read; FALSIFIED uses an uncorrected one-sided bound; 1e-9 absolute tolerance; doc of `reliability` misstates return. **Migration hazard:** H3 flips INCONCLUSIVE→SUPPORTED if recomputed with jc's t-test (p 0.0228 < Holm 0.025) — the sign-flip test must move with the verdicts, not be replaced. |
| Ring/centre-surround machinery (D-MHB, R-MHB-1 proposal) | mask-risc example | Counting/calibration: ndarray popcount, moments, PowerSums, RollingFloor exist. Spatial observable: **none** | Counting/thresholding = DUPLICATED SUBSTRATE (should reuse). Signed bit-sliced weight planes = SEMANTIC DEFECT (A2). The spatial observable itself is genuinely missing. R-MHB-1 was never built (only a board record). |
| `mask_row_window`, `bit_sliced_weighted_count` | ndarray (reverted, never on master) | ruling in ndarray blackboard:4: 256×256 is a palette address answered by LUTs | withdrawn; nothing to do |
| Signed Register128 readings + RegisterLaw + register_laws | contract register128.rs/hotplug.rs | post-#1464 plug path is the right home for "class declares reading" | AxisPosition/Support = legitimate value semantics in the right layer (plug/Activation) but never declared by any authority (dead). RelativeOffset = coordinate semantics → SEMANTIC DEFECT (MortonDir exists). |
| Strict `Opcode::decode` in `forward`/replay | causal-edge (#1407) | chain_admission pattern (validate once per chain) | WRONG IMPLEMENTATION LAYER (validation in replay) + compatibility break (55/126). `isa.rs` descriptors themselves = valid, consumed by mask-risc. |
| Tenants 18–20 | contract | ClassView/plug readings; no existing Moore-lane carrier | AUTHORIZED EXPERIMENT; genuinely new representation. Defects are contract-hygiene (see E). |
| f32 PhaseLut | ndarray hpc::phase | none | genuinely new; consumed only by two mask-risc examples; f32 values must stay out of decisions |
| checked_affine | ndarray | none | genuinely new, exact, VALID |
| Gomoku learner `Ev{pos,neg}` HashMap | csd example | group sums over an event table (quack Pair + SumI32) | the canonical carrier exists but was never tried — this is the missing seam (F) |

---

## D. Scientific evidence ledger (keep all of these)

Positive (with oracle):
1. Structural learner > best fixed: +0.0238 [+0.006,+0.0415], p=0.0156, 16 blocks (oracle: game score). Context T3 +0.0247 [+0.005,+0.044].
2. Certified minimax: 0/400 changed picks; −36/−45/−61% folds; 16/16 identical games (oracle: plain minimax).
3. Moore-lane ISA observational equivalence: 0 divergences (oracle: CE64 op on source edge); refusals pinned.
4. e = Jeffreys posterior mean exactly (oracle: closed-form Beta).
5. #1428: certified decisions never wrong; LODF fastest.
6. checked_affine exact composition (oracle: stepwise sums; one case where only stepwise fails pinned).

Negative (equally valuable):
7. Weighted per-query lane 26–150× slower → empirical support for A2.
8. u8 CE64 confidence stalls at c=244 after ~20 unit observations; fixed-window behaviour; REPLICA: 2000 events at P(+)=0.55 gives u8 f 0.486 vs true 0.551 (expectation flips side).
9. Duplicate evidence counted as independent: sd understated 2.04× (4 copies), 3.48× (8 copies); echo raises confidence in register path (gadamer_isa_revision.rs:265, chain_confidence.rs:89); only the Gadamer gate blocks it.
10. Thompson falsified (T1, T2, T4). Decay (H5) falsified. Delayed credit (H3) inconclusive.
11. NarsTables B1 flips ~38% verdicts; no decision-stable resolution.
12. Counterfactual crossing arms do not pay (0.03–0.81×).
13. 55/126 legacy CE64 vectors carry codes the strict ISA does not define.
14. HDR Stroke 1 drops true hits (test-pinned) → never an exact oracle.

Limitations attached: 3 seeds in #1400 (resolved by 16-block replication); timings without target-cpu everywhere; #1427 has no measurements; regret oracle reuses the training signal (only game score is independent); ECE pools serially dependent predictions (empirical, no interval).

---
## E. Corrected V1–V17 audit

Classifications: AUTHORIZED EXPERIMENT · VALID RESULT · DUPLICATED SUBSTRATE · WRONG IMPLEMENTATION LAYER · SEMANTIC DEFECT · STATISTICAL VALIDITY GAP · PROCESS VIOLATION · UNPROVEN · OBSOLETE CLAIM.

| # | item | original instruction | actual implementation | canonical alternative | genuine defect | value to retain | correction | evidence missing | class (corrected) |
|---|---|---|---|---|---|---|---|---|---|
| V1 | signed Register128 readings | none recorded | RegisterI4x32/I8x16, RegisterLaw{RelativeOffset,AxisPosition,Support}, register_laws on Activation | plug/Activation + SlabDeclaration (#1464) — the layer was roughly right | RelativeOffset is a coordinate as a signed number; no authority declares any law → dead code; register128.rs:39 doc stale | value readings AxisPosition/Support as plug-declared value channels | keep value readings only if an authority declares them; drop or gate RelativeOffset; fix doc. **Not** a blanket revert. Wait for #1465 | whether any consumer wants a signed value channel on Register128 | WRONG IMPLEMENTATION LAYER (partly), SEMANTIC DEFECT (RelativeOffset), UNPROVEN (no consumer). Earlier "per-concept table forbidden by hotplug.rs:43" was OVER-STATED: register_laws resolve through the Activation, which is the plug path |
| V2 | tenants 18–20 | operator proposal (chat) | three tenants, views, lift_ce64, equivalence tests | ClassView/plug readings; no prior Moore-lane carrier | no SlabReading variant/declaration; Epi5 inferred not declared (:178); writes bypass tenant_update; "Energy4" mislabels the inference mantissa; `ce64_upper` leaves direction 000; `Full.tenant_bytes` moved without version marker; not used by any learner; MooreNars16 holds no F/C | ISA observational equivalence; refusals; palette-pair LUT | KEEP; fix contract hygiene (declare readings through the plug, route Epi5 via `project_state5`, writes via `tenant_update`, rename/doc Energy4, version marker for Full offset); no deletion, no renumbering | whether tenant state can carry learning (F) | AUTHORIZED EXPERIMENT + VALID RESULT (representation) + SEMANTIC DEFECT (hygiene items) + UNPROVEN (learning). Earlier "self-ratification / remove" WITHDRAWN; "not formally ratified" stays true |
| V3 | strict CE64 decode in forward/replay (#1407, bad7b157) | "CE64 op-set hardening … PROBE" | `forward -> Result`, replay refuses | admission-time validation (chain_admission pattern) | replay refuses 55/126 historical vectors; same name changed semantics (I-LEGACY AP1) | isa.rs field descriptors (used by mask-risc); the 55/126 census; golden capture | compatibility decision first: total `forward` + admission check, or explicit replay-version gate. Post-#1407 consumers (d_rpf_conf_0, d_rpf_table_1 + bench, chain_counterfactual) must be inventoried | does any modern consumer rely on refusal semantics? | WRONG IMPLEMENTATION LAYER + SEMANTIC DEFECT + VALID RESULT (census) |
| V4 | mask bit as weight / signed Pos/Neg planes / f64 DoG decision model | Mexican-hat exploration authorized as research | mexhat_bucket_probe arms | value lanes + masked reductions; ndarray counting/moments | A2 violation in arm F and R-MHB-1 proposal | 26–150× negative result; f64 DoG as optional diagnostic only | storno ADOPT rows; keep probe file non-normative | none for the negative result | SEMANTIC DEFECT + DUPLICATED SUBSTRATE (counting) + VALID RESULT (negative) |
| V5 | new evaluators / R-MHB-1 terminal | — | board record only (never built) | lowering over existing words | proposal only | — | storno the proposal | — | OBSOLETE CLAIM |
| V6 | plan lines 289 / 308-309 / 362 | — | plan text | MortonDir; LUT as lookup | signed offset; LUT as bilinear form; weight-as-mask | none | append-only storno | — | OBSOLETE CLAIM |
| V7 | lab.rs | research harness authorized | 377-line stats module in example | **nothing equivalent exists** (B4) | wrong layer + 5 defects (T975 df>30, higher_is_better unread, uncorrected FALSIFIED bound, tolerance, doc) | all 13 verdicts + preregs | move to jc WITH sign-flip; tests reproduce every recorded verdict; fix defects separately and re-run | jc wiring (excluded crate) | WRONG IMPLEMENTATION LAYER + STATISTICAL VALIDITY GAP. Earlier "duplicates jc" was WRONG and is withdrawn; "IID contradicts Jirak" was OVER-STATED: block-level sign-flip over independent seeds is legitimately exchangeable; Jirak applies to sums over weakly dependent bits and has no callable function |
| V8 | "f32 ISA law exact" (D-RPF-G2) | — | board text | — | overclaim | — | storno wording | — | OBSOLETE CLAIM |
| V9 | signed-affine recommendation | — | board text; code `checked_affine` exact | — | recommendation only | checked_affine | storno recommendation | — | OBSOLETE CLAIM (code = VALID RESULT) |
| V10 | "palette is a semiring" | — | my chat + bgz17 KNOWLEDGE:178 (not mine) | LUT | terminology | — | doc supersession note | — | OBSOLETE CLAIM |
| V11 | ndarray signed window | — | reverted, never on master | — | — | overflow lesson | none | — | OBSOLETE (withdrawn) |
| V12 | tile64k_probe | — | reverted | — | — | — | none | — | OBSOLETE (withdrawn) |
| V13 | model id in trailers | harness attribution vs CLAUDE.md rule | all commits | — | rule conflict | — | operator decision for future commits; no history rewrite | — | PROCESS VIOLATION |
| V14 | duplicate commits | — | on main | — | hygiene | — | none (content identical) | — | PROCESS VIOLATION (inert) |
| V15 | ±1 LSB golden tolerance; silent fuzz skip | — | causal-edge tests | — | weakens falsifiers | — | tighten with the V3 decision | — | STATISTICAL VALIDITY GAP |
| V16 | acted before reading the substrate | operator: read apparatus first | — | this document | root cause of V4/V7 framing errors, and of my own audit's misclassifications | — | — | — | PROCESS VIOLATION |
| V17 | force-push of branch reset under "change nothing" | — | branch reset to main | — | scope | — | none further | — | PROCESS VIOLATION |
| new | #1427 merged without measurements; compile never observed | — | — | — | evidence gap | — | rerun or label CLAIMED | measurements | UNPROVEN |
| new | Codex P2s on #1408 unanswered (Epi5 inference, tenant_update bypass) | — | still in code | — | review gap | — | fix in V2 hygiene PR | — | SEMANTIC DEFECT |

Teardown items T1–T11 map onto these rows; their "REMOVE" verdicts for T1 (blanket revert) and T2 (delete tenants) are WITHDRAWN. T3 stays a compatibility decision, not an automatic revert. T7's "move to jc because duplicated" becomes "move because wrong layer; nothing to deduplicate".

---

## F. Missing-seam experiment (smallest still-unproven step)

### F.0 Premise correction (load-bearing)
The #1400 learner's state is keyed by **(board situation signature, recipe)**, not by Moore lane. Moore enters only as candidate generation (cells with a stone in their 8-neighbourhood). Tenants 18–20 are indexed by Moore slot (8 lanes), MooreNars16 carries no F/C. So "do tenants 18–20 carry the demonstrated learning?" is not well-posed as stated: the representations index a different axis. The well-posed first question is whether the learning loop runs **exactly** through canonical primitives at all; the Moore-lane question is a second, separate experiment.

### F.1 Five-part primitive ownership statement
1. **Hypothesis:** the persistent structural learner's evidence and decisions are expressible exactly through canonical primitives: a stamped event table + `GroupAddr::Pair{hi:situation_key, lo:recipe, stride:9}` grouped `SumI32`/`Count` (quack → mask-risc `GroupKey::Pair` → ndarray), with the k=1 expectation computed consumer-side, and statistical confidence kept out of CE64.
2. **Existing implementation:** quack `lower_group_by`/`lower_group_avg` with `GroupAddr::Pair` + `GroupAgg::SumI32` (lib.rs:1542-1600, :2368), mask-risc `GroupReduce`, no_alloc Counting allocator, Gadamer gate for echo, lab sign-flip/Holm for the block statistics (until moved).
3. **Missing capability (to be shown, not assumed):** (a) an incremental keyed fold (update one slot = batch result); (b) a declared reading for evidence counters on a row (if they are to live in tenants, not in caller-owned sinks); (c) decay ≠ 1 (rank-weighted fold) — not needed, H5 falsified.
4. **Independent falsifier:** Arm B as a shadow chooser on Arm A's exact trajectory must agree on 100% of decisions with decay=1.0 (integer pos7/neg7 sums vs f64 /7 sums are order-equivalent; any disagreement is a carrier bug or a tie-break difference, pre-registered). Disable run: drop one event row → agreement must fall below 100%.
5. **Canonical owner:** none new if (1) holds. If the incremental fold is needed for cost, it belongs in mask-risc as a terminal lowering over existing ndarray words; a declared counts reading belongs to the plug (SlabDeclaration), never a hand-read tenant.

### F.2 Arms (matched seeds, fresh block seed range, 16 blocks × 300 games, endpoint last 100; plus a Threat→Lookahead switch)
- **A** existing HashMap learner (BASE, decay 1.0).
- **B** event table + Pair-keyed grouped sums through quack/mask-risc; same expectation; shadow on A's trajectory, then independent run.
  - **B′** decisions from CE64 u8 F/C through `revision` (canonical epistemic carrier). REPLICA predicts divergence (C stall at 244, f drift). This tests Q2 directly: epistemic carrier ≠ statistical evidence carrier.
- **C** controls: C1 no learning; C2 shuffled credit (preserves marginals); C3 echo (each event twice) with and without stamp-dedup mask (Gadamer roots).

### F.3 Measures
Decision agreement A/B (exact); held-out frozen score; post-switch adaptation; folds per decision and rows touched; materialized bytes and heap allocations (Counting allocator; must not scale with games played beyond the event table itself); byte-identical replay; calibration (ECE all vs non-neutral, empirical only); provenance (every count traceable to stamped events); echo inflation (C3 with vs without dedup).

### F.4 Statistics
Block-level paired sign-flip with Holm (as pre-registered in the lab, to keep verdict comparability); A-vs-B agreement is deterministic, no test needed. Statistical confidence (p, CI) is never written into CE64; EpistemicState5 only carries declared certification.

### F.5 Second experiment (only after F.2 passes)
Moore-lane question: reformulate a learned quantity that is genuinely slot-local (e.g. per-direction threat evidence around a candidate cell) and test whether Nars16x8 lanes can carry it under a plug-declared reading, against the same arms. Prerequisite: a declared Nars16x8 reading (currently undeclared).

### F.6 What this does NOT need
No shift kernel, no Mexican hat, no new statistics, no new registry. The spatial ring/autocorrelation observable (fused shift∧popcount) is a separate, currently off-critical-path question; the one available route today is a pre-zeroed 8 KiB caller-owned shifted plane per axis Δ + `mask_ternlog_popcount::<AND>` — field-sized scratch, which the tiled law counts as materialization, so it is acceptable only in a probe and must not be argued from "no heap allocation".

---

## G. Minimal cleanup PR graph (proposed, not executed)

Ordered by safety, then dependency, then scientific value. Every PR: no history rewrite, rebase onto current main, concurrent Odoo/SAP/HubSpot/IAM/HIRO/HotPlug work untouched.

| # | PR | preserves | removes | moves | proving test | depends on |
|---|---|---|---|---|---|---|
| G1 | Board/doc supersession (append-only): V5/V6/V8/V9/V10 claims; mexhat ADOPT rows; "RATIFIED" → "AUTHORIZED EXPERIMENT, not ratified"; #1427 labelled CLAIMED; ndarray `cascade_search.md` percentile drift; weather-poc offset risk as ISSUE | all measurements and probe files | nothing | nothing | entries_index/supersession_index regenerate clean | none |
| G2 | **Experiment F.2** as a probe (example + test), using quack Pair grouped sums; lab statistics as-is | everything | nothing | nothing | A/B decision agreement 100% (disable run: drop one event → <100%); Counting allocator gate; byte-identical replay | none (all primitives exist) |
| G3 | lab.rs → jc (sign-flip, Holm, Prereg/Verdict, calibration, samplers to their owners), jc wired as dev-dep | all 13 verdicts | duplicate-in-example | lab.rs bodies | golden test: every recorded verdict reproduced exactly (H3 stays INCONCLUSIVE under sign-flip) | after G2 (so G2's verdicts are recorded first) |
| G4 | lab defect fixes (T975 df>30 quantile, higher_is_better, FALSIFIED multiplicity, tolerance, doc) as a separate PR that re-runs and records any verdict changes | old verdicts kept as history | defective constants | — | changed verdicts listed explicitly | G3 |
| G5 | Tenants 18–20 contract hygiene: plug-declared reading/SlabReading variant, Epi5 via `project_state5`, writes via `tenant_update`, Energy4 doc/rename shim, version marker for Full tenant offset | tenants, numbering, equivalence tests | nothing | nothing | existing equivalence tests + new refusal test for undeclared reading | after #1465 lands |
| G6 | Register128 readings: drop/gate RelativeOffset; keep value readings behind an authority declaration; fix stale doc | value readings | RelativeOffset | — | test: undeclared signed reading refused; declared value reading round-trips | after #1465 |
| G7 | CE64 replay compatibility: total replay + admission validation (or explicit replay version gate) | isa.rs, golden capture | refusal inside replay | strict decode to admission | 126 vectors replay; admission refuses exactly the 55; modern consumers adjusted | operator decision; after G2 |
| G8 | (optional, only if a spatial experiment is authorized) fused shift∧popcount as a mask-risc lowering + ndarray word | — | — | — | equals brute force; 0 bytes; symmetry | separate authorization |

---

## Final answer (the question asked)
The smallest still-unproven experiment that advances NARS × Moore learning is **F.2**: run the already-demonstrated structural learner through the canonical execution path — stamped evidence events grouped by `(situation, recipe)` via quack `GroupAddr::Pair` → mask-risc `GroupReduce` → ndarray, expectation computed consumer-side — and show exact decision agreement with the HashMap learner, zero population-scaled allocation and byte-identical replay, while the B′ arm shows whether the CE64 u8 epistemic carrier can or cannot stand in for statistical evidence, and C3 shows echo inflation with and without stamp dedup. It needs no new kernel, no new statistics and no new registry. Only if it passes is the Moore-lane question (F.5) well-posed.
