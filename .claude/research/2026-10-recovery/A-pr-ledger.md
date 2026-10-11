# A — PR scientific-results ledger (lance-graph / ndarray), built 2026-10-10

Sources: `gh api repos/AdaWorldAPI/<repo>/pulls/N` (body, files, review comments, issue comments) cached under
`scratchpad/recovery/raw/`; code read in `/home/user/_ro/lance-graph` at main `1573eb4c` (= merge sha of #1464).
Grades: VERIFIED-IN-CODE (read on main) / VERIFIED-IN-PR-BODY (quoted) / CLAIMED (asserted, no independent check here) / ABSENT.

## Index

| PR | title | merged | merge sha |
|---|---|---|---|
| lg#1400 | D-MOORE-NARS-0: recipe learning from reasoning consequences on Gomoku | 2026-10-08T00:34Z | 7524bf7e |
| lg#1406 | Probe: MooreNars16 as a Moore-local representation of ISA-visible CE64 state | 2026-10-08T08:40Z | 92c47988 |
| lg#1407 | CE64 ISA: strict decoder, register methods, field contracts | 2026-10-08T08:58Z | 16c21c4f |
| lg#1408 | contract: Nars16x8, MoorePalettePairs, MooreNars16 value tenants (ordinals 18-20) | 2026-10-08T08:59Z | a76d9ab5 |
| lg#1410 | contract: signed Register128 readings (32×i4, 16×i8) with a per-family law | 2026-10-08T15:57Z | b24375a6 |
| lg#1411 | plan: resident-projection-fold-mask-v1 (D-RPF-0..8) | 2026-10-08T16:11Z | 8fc2d08e |
| lg#1426 | D-RPF-TABLE-1: does the confidence-inert NarsTables fast path earn its semantic cost? | 2026-10-09T14:17Z | 64cbde68 |
| lg#1427 | D-MOORE-PERT-AB-0: MoorePertubationsfeld A/B probe with ICC and Spearman | 2026-10-09T17:30Z | 1d1f472b |
| lg#1428 | D-MOORE-OBSERVABLE-FIRST-0: certified decisions before the Moore field converges | 2026-10-09T20:47Z | 415bfde8 |
| lg#1432 | probe: D-SCF-CARE-PAIR-0 crossing counterfactuals and folded chain relations | 2026-10-09T23:17Z | d29d0aa4 |
| lg#1440 | D-SELF-CALIBRATING-LAB-0 phases I–IV | 2026-10-10T09:43Z | 9a430fe1 |
| lg#1456 | Lab v1 (Thompson + context arms on Gomoku); phasor_trig_probe on PhaseLut | 2026-10-10T16:22Z | 0a912218 |
| lg#1464 | V3 mandatory: the reading comes from the plug, never from the classid value (D-HPR) | 2026-10-10T18:37Z | 1573eb4c (= main) |
| lg#1465 | ogar: canon domain readings come from the plug (D-HPR-2) | **OPEN, not merged** | head c95885e4 |
| nd#344 | PowerSums / CrossPowerSums: checked_affine | 2026-10-08T20:39Z | 334dee5c |
| nd#354 | packed/cascade: pin that the two stroke packers agree at 2048 bytes | 2026-10-10T16:59Z | 8afa92da |

---

## lg#1400 — D-MOORE-NARS-0 (merged 7524bf7e)

**Hypothesis (VERIFIED-IN-PR-BODY):** "can the NARS recipe palette learn, from the consequences of its own reasoning, which recipe to run in which recurring Moore-local situation, without gradients, a central planner, or memorizing boards?" Constraint: "What is learned is situation → recipe, never position → move."

**Files:** `crates/cognitive-shader-driver/examples/moore_nars_gomoku_probe.rs` (+1779, `test = true`), Cargo.toml (+4), board entry `entries/2026-10-08-moore-nars-0-recipe-learning-gomoku.md`, STATUS_BOARD row.

**Learner data structure (VERIFIED-IN-CODE, probe on main, 2366 lines now — grew via #1440/#1456):**
- `moore_nars_gomoku_probe.rs:29` `use std::collections::HashMap;`
- `:811-815` `struct Ev { pos: f64, neg: f64 }` — "NARS evidence for one (signature, recipe) pairing."
- `:818-825` `expectation()`: `f = pos/w; c = w/(w+1); c*(f-0.5)+0.5` (0.5 when w==0).
- `:862-866` `struct Learner { stats: HashMap<u64, [Ev; NO]>, folds: [(u64, u64); NO] }`
- `:893` chooser: `value - cfg.cost_weight * (1.0 + mean).ln()` (cost term on mean folds).
- `:898-917` `credit()`: `a = activation/7.0`; positive → `e.pos += a`, negative → `e.neg -= a`; decay multiplies pos/neg; Structural memory writes both `p.key` and `p.back`.
- Persistence is `EpistemicState5` produced by `Ev::state()` and a `CausalEdge64` record (`record()`, `:944-952`) — NOT a ValueTenant.
- => The learner is a plain in-process `HashMap<u64,[Ev{f64,f64};NO]>`. **It reads/writes NO ValueTenant 18/19/20 and no MooreTenantView/Mut** (grep: zero hits in the probe for `MooreTenant|Nars16|MoorePalettePairs|ValueTenant`). VERIFIED-IN-CODE.

**Arms (VERIFIED-IN-PR-BODY; Cfg in code `:782-808`):** chooser {Fixed(op), Uniform, Learned, Thompson(later)}, signal {Retained(h), Immediate, Final, contradiction-only}, memory {Structural, Literal, Global}, `reset_per_game`, `decay`, `cost_weight` (0.02 default), `learn` (frozen eval), `certified` (later, #1440).

**Measured (VERIFIED-IN-PR-BODY; release run 427 s; 3 seeds; last of 5 windows of 400 games; score win=1/draw=0.5; target-cpu NOT stated):**
| arm vs Threat | score |
|---|---|
| best fixed (CUR/HPM) | 0.498 / 0.493 |
| persistent structural learner | 0.500 → 0.535 (per-seed last 0.509/0.539/0.559) |
| reset every game | ~0.47 |
| literal board memory | ~0.49 (hits <4%) |
| immediate credit | ~0.50 (stays HPM) |
| credit 3 turns later | 0.496 (0.495 before fix) |
| contradiction only / final only | ~0.47 / ~0.49 |
| uniform | ~0.24 |
- Transfer frozen: unseen openings 0.526 vs 0.493 untrained; 11×11 0.507 vs 0.493.
- 169 signatures; favourite spread over 8 of 9 realized recipes.
- Switch Threat→Lookahead: pretrained 0.626 > fresh 0.591 > fixed HPM 0.570 > fixed ICR 0.455.
- No CI, no p-value, no effect-size test reported. n = 3 seeds × 400 games/window (1200 games pooled in last window). The headline +0.035–0.037 over best fixed is **un-tested for significance**; seed range 0.509–0.559 spans the best-fixed 0.498. (Later #1440/#1456 lab re-measures this with Welch + bootstrap — see there.)

**Negative findings (VERIFIED-IN-PR-BODY):** "No 'less work.' The learner wins by spending about 280 folds per step against 0 for HPM, and that number rises over training." (448 cost-blind). "Unlearning is not demonstrated." Caveats: regret not independent (same one-reply signal); "ICR is privileged against Threat"; "Most games are draws."

**Tests:** 7 (~30 s debug): 6 structural (census; win before block; register never names recipe; W follows pending slots; deterministic replay; frozen eval leaves learner unchanged) + 1 learning falsifier with 4 disable runs (credit off, reset off, literal given structural key, immediate given delayed). Independent oracle: fixed deterministic opponents + game score; the census oracle is `contract::recipes::RECIPES`. CLAIMED red-then-green.

**Review that changed claims:** Codex P2×3 + CodeRabbit (fixed 9d76ff96, c10e9649): h=3 horizon counted queued steps not learner turns; frozen eval mutated fold costs (transfer numbers had to be re-measured); immediate arm leaked final outcome at game end; productive/contradiction double counting. Only h=3 moved (0.495→0.496).

**Unproven:** significance of the learner's gain; "less work"; unlearning; long-horizon attribution; stronger opponents; any tenant-resident representation of the learned state.

## lg#1406 — MooreNars16 ISA observational-equivalence probe (merged 92c47988)

**Hypothesis (VERIFIED-IN-PR-BODY):** "`MooreNars16` is **not** a lossless compression of canonical NARS24. It is a declared Moore-local representation of the NARS state the ISA actually reads: lane (u16): Pearl3 | Energy4 (raw mantissa) | Plasticity3 | Polarity1 | Epi5; tenant: Witness6 once + 8 lanes = 128 bits (+ 8 x SPOFC40)". "The bar for correctness is ISA observational equivalence. For `forward` (running and weight positions), `learn` (self and observation) and `syllogize`, a tenant operand and its canonical edge give identical results on every field except Direction3".
**Files:** probe-only — `crates/lance-graph-planner/tests/moore_nars16_probe.rs` (+760; defines its OWN `struct MooreNars16(u16)` at `:112`, VERIFIED-IN-CODE), board entry. "It does not change the CE64 layout, any CE64 operation, or any production path."
**Oracle:** the canonical `CausalEdge64` operations themselves (the edge the tenant was lifted from is run through the same ISA op; compare outputs field-by-field). Independent of the Moore representation; NOT independent of the ISA implementation.
**Measured (VERIFIED-IN-PR-BODY):** 15 tests. 300 tenants × 8 lanes × 5 ops → 0 divergences (outside Direction3). "In ≥600 of 800 weight-side `forward` results the raw words still differ, because `forward` copies the sign triple. So this is observational equivalence, not bitwise." All 16 raw mantissa nibbles reach `forward` unchanged. Five mutations each diverge (3-bit energy; dropped plasticity bit; dropped Epi5; wrong witness; polarity over plasticity). Direction/Witness/Epi5 change no other output field (can-fire: weight mantissa selects rule; plasticity gates learn). Mixed-W tenant refused; reserved Epi5 24..31 refused via `EpistemicState5::decode`.
**Disable runs (CLAIMED):** witness refusal 1 red; Epi5 refusal 1; Simpson refusal 1; bitwise comparison 3; polarity 1.
**Negative / review-changed claim:** Codex P2 — the "refusal" existed only in the probe's own `simpson_pattern`; the production `CausalNetwork::detect_simpsons_paradox` takes bare edges and "silently reports 'no pattern' for a Moore edge". Fixed by pinning this as a MEASURED GAP (`the_production_simpson_detector_cannot_refuse_a_moore_edge`), not by fixing production.
**Open (VERIFIED-IN-PR-BODY):** witness ownership scope rests on code reading not probe; Polarity1 "has no canonical CE64 field"; "Not run yet: Gomoku A/B replay, the domain runs, chess, timing."

## lg#1407 — strict CE64 ISA (merged 16c21c4f) — this is the PR that introduced `causal-edge/src/isa.rs` (507 lines on main, VERIFIED-IN-CODE)

**Claim:** "Reasoning chooses the operation. `CausalEdge64` defines what that operation means." Strict dispatch over all 16 inference codes: +1 Deduction, +2 Induction, −1 Abduction, +4 Revision step, +5 Synthesis; all others → `IsaFault::Unsupported { mantissa }`. "Before this PR, `±6` and `±7` ran the Synthesis mean and came back stamped" (silent mis-execution — a real defect fixed). Field-liveness matrix declared in `isa::contracts`.
**Oracle:** (a) 126 legacy golden vectors captured from pre-PR code: 71 bit-identical, 55 now refuse with exactly their code, "nothing else changed"; (b) 23 normative vectors vs an independent f64 reference within ±1 on F and C; (c) per-field perturbation contract tests.
**Negative finding (VERIFIED-IN-PR-BODY):** max-confidence revision defect: `revision(c=255, c=255)` → f = c = 0 (evidence_weight = f32::MAX; MAX+MAX=inf; inf/inf=NaN; NaN→u8 = 0). Pinned known-bad, excluded from normative set; fix deferred ("separate PR").
**Gadamer test:** 10 echo encounters/closed cycles leave confidence unchanged (gate never calls revision); 10 independent roots raise it; `revision` alone would inflate an echo.
**Tests (CLAIMED):** causal-edge 81 lib + 12 contract + 6 golden + 8 op-contract; 9 disable runs listed. Review: Codex P2 v1-layout fault codes (fixed bad7b157, both layouts tested), ReplayError docs (fixed).
**Open:** forward's +4 composes payload while `revision(b)` does not; forward whispers executed code into bits 46..49; `learn` mixes policy (0.9/0.7 freeze thresholds); planner truth-function copies (`TruthValue`, `nars_infer`, `NarsTables`) "still differ from the canonical ones". API break: `forward` returns `Result` — which broke main via #1408's J test (see below).

## lg#1408 — value tenants 18–20 (merged a76d9ab5)

**What (VERIFIED-IN-PR-BODY + VERIFIED-IN-CODE `canonical_node.rs:1129-1155`):** append-only tenants after Register1. "NodeRow stays 512 B, tenants 0..17 and all CE64 fields are unchanged, and there is no ENVELOPE_LAYOUT_VERSION bump."
| ord | tenant | kind | row bytes | slab bytes |
|---|---|---|---|---|
| 18 | Nars16x8 | U16×8 LE | [284,300) | [252,268) |
| 19 | MoorePalettePairs | U8×16 | [300,316) | [268,284) |
| 20 | MooreNars16 | U16×8 LE | [316,332) | [284,300) |
Lane i = Moore slot i in order NW,N,NE,W,E,SW,S,SE; pair address `(first<<8)|second` = #1336's `PairAddress`; `ValueSchema::Full` 252→300 B; narrower presets carry none; `BoardAggregates` reservation re-based to 21.
New module `contract::moore_tenant` (800 lines): `MooreSlot`, `MooreNars16`, `MooreTenantView`, `MooreTenantMut`, `DirectionReading::require_sign_triple`, `MooreTenantMut::lift_ce64` (checks all 8 lanes before writing; refuses mixed witness and Epi5 24..31).
**Tests A–L (CLAIMED counts):** 11 in moore_tenant.rs (G exhaustive over 65 536 lane values + CE64 bit map); F `cognitive-shader-driver/tests/moore_palette_pairs_lut.rs` vs `PairAddress`/`PaletteLut`/`Quad8`; J `lance-graph-planner/tests/moore_tenant_isa_equivalence.rs`: 256 tenants × 8 lanes × 5 ops → 0 divergences outside sign triple; anti-vacuity wrong-witness diverges. 10 disable runs listed.
**Oracle:** byte table/`value_offset()` for layout; `PairAddress`/`PaletteLut` for F; strict #1407 ISA for J (same oracle class as #1406, now through the shipped tenant).
**Stated limitations (VERIFIED-IN-PR-BODY, "Open"):** "The `Nars16x8` u16 reading is not declared anywhere." "No `SlabReading` variant gates these tenants yet." "There is no production reader or writer."
**Unaddressed review findings (no reply in thread; VERIFIED-IN-CODE still present on main):**
- Codex P2 "Require declared provenance before decoding Epi5": `moore_tenant.rs:178` still calls `EpistemicState5::decode(Epi5Gen::V1, self.epi5_raw())` directly, not `Epi5Declarations::project_state5`.
- Codex P2 "Record Moore writes in the tenant counters": grep `tenant_update` in moore_tenant.rs → 0 hits.
**Collateral:** main was BROKEN after #1408 merged — J test did not compile against #1407's `Result`-returning forward (`E0609: no field 0 on Result<CausalEdge64, IsaFault>`); fixed by lg#1409 (merged 2026-10-08T10:57Z, 08e1626f). Source: #1410 issue comment.
**SlabReading on main (VERIFIED-IN-CODE `hotplug.rs:231-257`):** variants are `Facet96=0, Register128=1, RegisterI4x32(law), RegisterI8x16(law)` — **no variant for tenants 18–20**; still ungated.

## lg#1410 — signed Register128 readings (merged b24375a6)

**What (VERIFIED-IN-PR-BODY):** signed readings on existing `Register0`/`Register1` rails; "no layout change". Three layers: carrier / carving (declared by slab) / law. `SlabReading` tags 2,3,4 = `RegisterI4x32(law)` (32×i4, dim 2k = low nibble of byte k, atoms::I4x32 layout); 5,6,7 = `RegisterI8x16(law)` (16×i8). `RegisterLaw { RelativeOffset, AxisPosition, Support }`. `ResolvedReading::bind_signed_register(&activation, rails)` takes NO law; refuses `RegisterLawMismatch` when slab law ≠ concept law (from `Activation::with_register_laws/register_law_for`, fail-closed `NoRegisterLawFor`). Out-of-range i4 refused not saturated; refused write leaves row unchanged. `bind_register128` (4×u32) unchanged and refuses signed carvings. "`EpistemicState5` gets no law".
**Review that changed design:** Codex P2 — law was a binder parameter; writer `Support` + reader `AxisPosition` both succeeded on `Copy` `ResolvedReading`. Fixed 96304b88: law recorded in slab tag AND declared per concept, must agree. Binding test covers 3×3 law pairs × both carvings + no-law concept; 4 disable runs.
**Measured:** none beyond tests — contract lib 1458 passed (CLAIMED); 11 disable runs listed.
**Open (VERIFIED-IN-PR-BODY):** "No authority populates `with_register_laws` yet, so in production every concept refuses to bind signed registers." "Whether a family gets its own lane or only a reading waits for a measurement: 32×i4 vs 16×i8 on KJV anaphora and the mammal fixture." "There is no production reader or writer." Only resident 24×i4 lane is `CausalWitness` (tenant 14, Facet96 as G24N4), not moved.

## lg#1411 — plan resident-projection-fold-mask-v1 (merged 8fc2d08e) — PLAN ONLY

"Proposal only, no code." Phases D-RPF-0..8. Relevant: **"D-RPF-5 — Moore tenants 18..20, gated on a `SlabReading` decision."** (VERIFIED-IN-PR-BODY). D-RPF-4 population revision blocked on four decisions incl. "the c = 255 defect" and one T1 gap. D-RPF-8: "mask-risc never executes CE64 instructions; no VSA on this path." "Not verified: ndarray was not checked out ... No real population was measured." Codex 3 findings fixed (tenant-internal windows; 24 meaningful Epi5 codes not 10; admission plane before eligibility patterns).

## lg#1426 — D-RPF-TABLE-1 (merged 64cbde68)

**Question:** does `NarsTables::build(n)` (n ∈ {1,2,4,8,16}) earn its semantic cost vs direct `CausalEdge64::revision` (A)? "This PR changes no production revision semantics and chooses neither the G1 nor the G2 policy."
**Oracle:** A's replay path and decision path checked bit-exact against `replay_step` and `counterfactual_replay` for every resolution; real implementations. 15 tests in `lance-graph-planner/tests/d_rpf_table_1.rs`; bench `examples/d_rpf_table_1_bench.rs`.
**Measured (VERIFIED-IN-PR-BODY; machine per board entry line 136: "4 vCPU Xeon @ 2.1 GHz; per core 48 KiB L1d, 2 MiB L2"; target-cpu NOT stated):**
| law | memory | replay dep/batch | micro uniform | cold | broad max dF/dC | uniform 8-step flips verdict/load-bearing/class (of 2048) |
|---|---|---|---|---|---|---|
| A | 0 | 32.7/15.7 ns | 12.1 ns | 11.5 ns | 0/0 | 0/0/0 |
| B1 | 256 KiB | 7.4/7.0 ns | 3.4 ns | 19.0 ns | 128/168 | 776/244/905 |
| B16 | 32.1 MiB | 8.9/9.8 ns | 29.2 ns | 60.3 ns | 113/14 | 96/77/426 |
- "B1 fails the pre-registered boundary": 2.2–4.4× faster hot replay; flips ~38% verdicts, ~12% load-bearing, ~44% Reaction classes. "No table resolution is decision-stable." Table wins only on hot dependent replay (review-corrected: dependent 3.7–4.4×, batch 1.6–2.2×).
- **"No production code reads any `NarsTables` revision today."** Corrects #1424 (`NarsEngine::tables` never read; shader driver discards lookup).
- The repo's own load-bearing fixture is load-bearing only under B1. `DEFAULT_FREQUENCY_BAR` rationale cites "confidence saturates at 170" = B1's constant.
- T2 (Pearl bits 40..42 = subset mask, not ladder), T4 (`learn` vs `isa::contracts::LEARN` over 8 plasticity patterns); Codex P2 → T4 gated on v2 layout.
**Open:** G2 (which law), G1 (the (255,255) cell), forward's truth discarded on every replay step.

## lg#1427 — D-MOORE-PERT-AB-0 (merged 1d1f472b)

**Scope (VERIFIED-IN-PR-BODY):** A = existing Laplacian spectral pseudoinverse per one-line outage; B = independent deterministic Jacobi relaxation on 8 canonical `MooreSlot` directions; compare phase/magnitude/flow via Pearson, Spearman, ICC(2,1), rel-L2, MAE across 42 synthetic outages × 8 iteration budgets; outage-block resampling "without an IID significance claim". 4×4 lattice only.
**Measured numbers in PR body: ABSENT.** "Rust compilation / CI has not yet been observed. Keep draft until..." — yet merged. Code prints the metrics (`MoorePertubationsfeld.rs:261`) but no numbers were recorded in PR or a board entry (no #1427 board entry in files list). Uses only `MooreSlot` (enum) from moore_tenant — not tenants 18–20.
**Kill conditions stated:** sign/magnitude/flow reconstructed from reference arm; slots sorted by conductance; external-validity/IID CI claims.

## lg#1428 — D-MOORE-OBSERVABLE-FIRST-0 (merged 415bfde8)

**Question (VERIFIED-IN-PR-BODY):** "Can the Moore relaxation from #1427 make an N-1 decision that is **provably** correct before its numerical field converges?"
**Method:** field certificate (grounded DC Laplacian is M-matrix; `A·w ≥ 1` ⇒ angle error ≤ ‖r‖∞·w_i); L1 flow certificate (flow error ≤ ‖r‖₁ by max principle for unit dipole). **Three independent oracles agree to 4.2e-13: dense Cholesky, closed-form LODF, spectral pseudoinverse.** Five fixtures + near-tie adversary (40 ratings at exact post-outage flow).
**Results:** soundness — "no certified decision was wrong and no bound was violated". Warm-start + L1: 78% of overload decisions certified at zero sweeps on heterogeneous 16×16; of 225 true overloads, 0 certified at sweep 0, 179 after 1024 Jacobi, 225 after 256 PCG. Naive (unbounded) decisions miss all at sweep 0; 11 still wrong after 1024. After 1024 warm Jacobi sweeps the angle field is "88% wrong, yet 99.9% of overload decisions are certified correct." Timing (930 outages; "median of 5, quiet machine", target-cpu NOT stated): LODF 17 ms, certified PCG ~170 ms, dense Cholesky ~2.1 s, certified Jacobi ~2.7 s.
**Falsified (negative for Moore):** "Jacobi is not the arm to use" (PCG 16–57× faster to certify); "For single-outage screening on a fixed grid, LODF beats every certified iterative method." "Correlation is not a readiness signal for decisions" (naive wrong at flow corr 0.97). Cold Jacobi never certifies field bound on 3/5 grids within 20 000 sweeps.
**Open:** sign decisions (8.6% certified); N-2; >10³ buses; "Whether Moore's locality ... ever pays off".

## lg#1432 — D-SCF-CARE-PAIR-0 (merged d29d0aa4)

**Question:** "can one certified local relation make a later derivation unnecessary?" Probe `cognitive-shader-driver/examples/crossword_crossing_care_probe.rs`, oracle = core's `count_fills`, identical answers asserted across arms.
**Negative:** "In-search counterfactuals don't pay": Single/Pair/PairZ at 0.03–0.81× baseline wall time on every workload; ~90% of Single's exclusions are plain zero-population support.
**Positive:** chain folding — two-slot chain folded to one 31×31 table, reused on two held-out instances (different board; reverse via transpose), 1000/1000 queries match `count_fills`; break-even 15–36 queries; SFCR(1000) 28–71. NYT grids have 0 unchecked cells → nothing to fold there.
**Review:** Codex P2 double-counting in `refuted` (fixed 7d82f3f, op totals −10–20%, conclusions unchanged); repeated boundary-word bug (fixed, pinned `equal_end_words_are_not_a_completion`); one P2 rejected with reason. 11 tests, 7 disable runs (CLAIMED). No Moore tenant use.

> ⊘ CORRECTION to the lg#1400 section above: I wrote "the #1440/#1456 lab re-measures this with Welch + bootstrap". Wrong — read in code (`examples/shared/lab.rs:77-160`): the lab uses an exact two-sided **paired sign-flip** p (exhaustive, n ≤ 20), a 95 % **paired t-interval** (`T975` table), **Holm** step-down, and `verdict()`. No Welch, no bootstrap.

## lg#1440 — D-SELF-CALIBRATING-LAB-0 phases I–IV (merged 9a430fe1)

"All changes are tests, probes and docs; no production default changes." Files: `causal-edge/tests/d_rpf_p5_beta.rs`, `examples/shared/lab.rs` (+311), gomoku probe (+447), crossword probe (+134), planner `chain_counterfactual.rs`, `d_rpf_table_1.rs`, 4 board entries.

**Statistics functions (VERIFIED-IN-CODE `shared/lab.rs`):** `sign_flip_p` (:77, exact, asserts n ≤ 20), `paired` (:99; d = sign·(t−b); mean, sd, `p: sign_flip_p(&d)`, CI = mean ± T975[n−2]·sd/√n), `holm` (:132), `verdict` (:149: Supported iff Holm-rejected AND mean ≥ min_effect; Falsified iff `r.hi < min_effect`; wrong block count → Inconclusive), `reliability` (:176, ECE), `brier` (:202). Gomoku wiring: `moore_nars_gomoku_probe.rs:1571-1577` (`lab::paired` ×5, `lab::holm(…, 0.05)`), `:1601` reliability.

**Phase I (G2):** "New decision-bearing wiring uses exact revision (law A). Existing defaults, including `NarsEngine::new`'s `build(1)`, stay as they are." New fixture `a_load_bearing_fixture_under_law_a`: 210 → 64 flips verdict; only B16 agrees. P8: "No production code builds a `CutContext` or calls `replay_chain`." `replay_step` emits weight's opcode with revision truth (recorded, not changed).

**Phase II-a P5 NARS↔Beta (`causal-edge/tests/d_rpf_p5_beta.rs`; board `2026-10-10-d-rpf-p5-nars-beta-correspondence.md`) — VERIFIED-IN-PR-BODY + board:**
- E1: e = c(f−½)+½ is EXACTLY the Jeffreys (k = 1, Beta(½,½)) posterior mean (max err 3.3e-16; k=2 differs ≥ 1.15e-5; f32 within 5.8e-8).
- **E2 confidence saturation:** unit Bernoulli observations stall at **c = 244 (w ≈ 22.2) after 20 observations, in 32/32 runs**; past that, revision is a fixed-window estimate; mean |Δe| vs conjugate < 0.002 at n=8, > 0.05 at n=256 (max 0.16).
- E2b: stall code depends on observation strength (c_obs=1→75, 64→236, 128→244, 240→252, 250→254).
- **E3 duplicate evidence:** "Revision cannot tell duplicated from independent evidence. Four copies ... claim a posterior sd 2.04× too small; eight copies 3.48×."
- E4: variance from (f,c) moves 0.3%/code at c=128, 29% at c=253. E5: revision never reaches c=255 ((254,254)→254).
- Disable: rounding round→ceil → E2, E2b, E5 red. Oracle: closed-form Beta posterior (independent of CE64 code).
- Consequence (board): "Any lab strategy that reads `(f, c)` as a Beta posterior is valid only below the stall; past it, intervals must come from an explicit evidence count kept elsewhere (e.g. a PowerSums fold)". NB: the Gomoku learner itself uses unbounded f64 `pos/neg`, NOT u8 codes — so the stall does not bind the learner; it binds anything storing evidence in CE64/u8 (e.g. tenants 18–20 MooreNars16's Energy4/Epi5 lanes would hold no confidence count at all).

**Phase II-b P1 certified argmax (`minimax_certified`; board `2026-10-10-d-lab-p1-certified-argmax-gomoku.md`):** certificate `L_best ≥ U_j` ⇒ exclude (root alpha-beta cut). 400 reasoning positions, oracle = plain `minimax` pick:
| k1/k2 | folds full | certified | saved | pick changed |
|---|---|---|---|---|
| 4/4 | 442,720 | 282,480 | 36.2 % | 0 |
| 5/5 | 665,076 | 367,344 | 44.8 % | 0 |
| 8/8 | 1,601,258 | 624,386 | 61.0 % | 0 |
Disable: cut at U ≤ L+1 changes a pick at 4/4. Deterministic, not statistical.

**Phase IV lab v0 (Gomoku arm; confirmatory 16 fresh blocks seeds 2001–2016; 300 games, endpoint last 100-game window; 9×9 vs Threat; 28.6 s wall; board `2026-10-10-d-self-calibrating-lab-0.md`):**
| id | hypothesis | effect | 95% CI | p (sign-flip) | Holm | verdict |
|---|---|---|---|---|---|---|
| H2 | learned beats best fixed (HPM), score | +0.0238 | [+0.0060, +0.0415] | 0.0156 | reject | SUPPORTED |
| H4 | certified minimax cuts folds/step (cost 0) | +28.9 | [+20.3, +37.6] | <0.0001 | reject | SUPPORTED (+ identical games 16/16) |
| H7 | fold-cost term cuts folds/step | +142.7 | [+118.3, +167.1] | <0.0001 | reject | SUPPORTED (at −0.026 score) |
| H3 | delayed credit beats immediate | +0.0225 | [+0.0036, +0.0414] | 0.0267 | keep | INCONCLUSIVE |
| H5 | decay 0.95 adapts faster after Threat→Aggressor | +0.0072 | [−0.0051, +0.0195] | 0.2322 | keep | FALSIFIED (CI < 0.02 min) |
- "certified minimax reduction": −29 folds/step at cost 0 in the game loop (397.6 → 368.6, ≈7.3%), vs 36–61% on the isolated 400-position corpus.
- Pareto: cost0+cert 0.5331/368.6; cost0 0.5331/397.6; decay .95 0.5159/231.9; base 0.5069/254.8; immediate 0.4844/0.0; fixed HPM 0.4831/0.0.
- **Calibration:** ECE 0.317 over all outcomes (e overstates), 0.050 over non-neutral outcomes ("e estimates P(positive | non-neutral)").
- Pilot (seeds 1001–1012) exposed 3 design errors fixed before confirmation (H4 must be cost-blind; 100 games = learning phase; calibration population). First min-effect test was vacuous; fixed.
- Crossword arm (separate Holm family, 16 puzzles): C1 Support cuts nodes +3.59 ln (≈36×) SUPPORTED; C2 Support ops −2.57 ln (≈13× more ops) FALSIFIED; C3 Single −1.07 FALSIFIED; C4 Pair −0.78 FALSIFIED.
- Limitations: one opponent family, one board size; H3 needs more blocks; record not yet SoA-shaped.
- Review: no inline review comments (codex/coderabbit limits); claims not externally reviewed.

## lg#1456 — Lab v1 (merged 0a912218)

"Two independent changes. Both are probe-only." (1) `phasor_trig_probe` uses `ndarray::simd::PhaseLut` (#343) instead of a private table; non-timing output unchanged. (2) LAB-1: `lab.rs` gains `gamma_sample`/`beta_sample` (Marsaglia–Tsang; moment test on Beta(½,½), (2.5,7.5), (30.5,10.5)); `Chooser::Thompson` samples Beta(pos+½, neg+½) (code `moore_nars_gomoku_probe.rs:885-888`); `Memory::Global` = one shared key.
**Ledger (seeds 4001–4016, own Holm family α=0.05, 16.4 s wall; board `2026-10-10-d-self-calibrating-lab-1.md`; functions `lab::paired`/`lab::holm`/`lab::report` at probe `:1690-1700`):**
| id | hypothesis | effect | 95% CI | p | Holm | verdict |
|---|---|---|---|---|---|---|
| T1 | Thompson beats expectation argmax | −0.0500 | [−0.0620, −0.0380] | <0.0001 | reject | FALSIFIED |
| T2 | Thompson adapts faster after switch | −0.0153 | [−0.0244, −0.0062] | 0.0040 | reject | FALSIFIED |
| T3 | structural context beats one global key | +0.0247 | [+0.0050, +0.0444] | 0.0186 | reject | SUPPORTED |
| T4 | Thompson reduces folds/step | −18.5 (costs more) | [−38.4, +1.4] | 0.0596 | keep | FALSIFIED (interval excludes 5-fold min) |
Pareto: argmax 0.5062/244.8 folds; Thompson 0.4562/263.3; global 0.4816/32.3 ("structural signature buys quality at about 8× the folds").
Thompson mechanism (fractional evidence keeps posteriors wide) is "not tested". Falsifiers `beta_sample_has_the_beta_moments`, `thompson_is_seeded_and_differs_from_argmax` (posterior-mean substitute → both red). 15 probe tests. No review comments (codex limit hit).

## lg#1464 — V3 mandatory: the reading comes from the plug (D-HPR) (merged 1573eb4c = current main)

**Ruling quoted in PR body (VERIFIED-IN-PR-BODY, attributed "Operator ruling, 2026-10-10"):** "Yes V3 mandatory, everything else in regards to capabilities hotplug.rs plug and play > ogar pattern making sure lockstep is deprecated" / "Any concept is immutable. That's why Ontologies exist. Different languages are a label. The concept needs to be immutable adress" / "1000 isn't hurting, just document and Mark it as deprecated. No need to force evict".
**What changed (VERIFIED-IN-PR-BODY + diff):** D-HPR-0 plan `.claude/plans/v3-mandatory-hotplug-reading-v1.md` (W1–W6) + inventory; D-HPR-1 new `_with`/`_plugged` forms taking the plug's reading instead of `BUILTIN_READ_MODES`: `ocr::LayoutBlock::to_node_row_with`, `aiwar::aiwar_node_rows_with`, `soa_graph::project_snapshot_with`/`nearest_anchor_with`, `nan_projection::project_energy_nonfinite_plugged` (resolves each classid run via `Activation::resolve_tenant_reading`; "An unplugged concept returns `NoReadingFor`, never a V1 default"). Old forms delegate. D-HPR-4: `CLASSID_*_V3` constants and `classid_read_mode` get a doc-only "Deprecated, kept" note (not `#[deprecated]`, to avoid `-D warnings` forced migration).
**Files:** aiwar.rs, canonical_node.rs (+37, doc notes only), nan_projection.rs, ocr.rs, soa_graph.rs, board + plan. **NOT touched:** hotplug.rs, register128.rs, moore_tenant.rs.
**Does it touch register128 signed readings / SlabReading tags / tenants 18–20? (VERIFIED-IN-CODE via patch grep):** grep of the #1464 patch for `register128|SlabReading|Nars16|MooreNars|MoorePalette|signed_register|RegisterLaw` hits ONLY the plan line: "How a row is read (tail, value schema, edge codec, `Facet96` vs `Register128`) comes from the plug (`HotPlug` → `CapabilityAuthority::activate` → `Activation::resolve_for_context`) plus the slab's own `SlabDeclaration`." So: no code change to signed readings, SlabReading tags or tenants 18–20. Indirect relevance only: tenants 18–20 live in `ValueSchema::Full`; under the plug path the value_schema now comes from `Activation`, and the falsifier classid `0x1718_0000` is one the legacy lookup reads as "V1 / `Full`".
**Tests:** each new form tested on classid 0x1718_0000; disable runs (form re-looks-up classid) 4/4 red (CLAIMED); contract lib `--no-default-features` 1445 tests. Codex P2 (V3 OCR test panicked without tail feature) → gated on `guid-v3-tail`.
**Open:** O1 classview for new canon mint instead of 0x1000; O2 WoA "Stundenzettel". Plan W6 (not done): "`ReadMode::DEFAULT` stops being the fallback for an unknown classid ... and `guid-v3-tail` stops being a feature gate."

## lg#1465 — ogar: canon domain readings come from the plug (D-HPR-2) — OPEN, NOT MERGED (head c95885e4)

`lance-graph-ogar::concept_override` answers five canon domains by concept (0x0701 OSINT_V3, 0x0A01 FMA_V3, 0x0E01 CPIC_V3, 0x0101 PROJECT_V3, 0x0202 ERP_V3). "**No consumer plugs these concepts today.** ... So no live reading changes." Codex **P1**: the advertised `OgarAuthority::activate → read_mode_for` path was unreachable (IDs routed to `resolve_hotplug` → `UnknownClassid`/`NoCapabilitiesFor`; test called private `plug_readings`). Author reply: confirmed, "fixed in the next push (not yet pushed)" — a reading-only activation route. **Not on main; fix not verified.** Clippy for this crate not in CI (6 pre-existing errors in bridges/mod.rs, rbac_impl.rs). No touch on register128/SlabReading/tenants 18–20 (patch grep: 0 hits).

## Other merged PRs since 2026-10-05 touching the named files (git log on main, VERIFIED-IN-CODE)

- moore_tenant.rs: only 28ec7c07 (#1408).
- canonical_node.rs: #1348 (D-MORTON-0, 2026-10-06), #1408, #1464 (doc note 9f5d3dfe).
- register128.rs / hotplug.rs: #1348, #1410 (14831543, 96304b88).
- shared/lab.rs: #1440 (df683e50, 608350e1), #1456 (730e584f).
- moore_nars_gomoku_probe.rs: #1400 (4410945c..c10e9649), #1440 (a290220c P1 certified minimax, df683e50), #1456 (730e584f).
- mexhat_bucket_probe.rs: **lg#1415** (D-MHB-1, merged 2026-10-08T19:13Z) and **lg#1418** (D-MHB-2 + D-ART-1, 19:58Z).
  - #1415: query `S(c)=Σ w(|p−c|²)` (DoG σc3 σs6 R18). Medians of 10 runs, ns/query, 1M grid: D window+LUT 246/619/1166/1516; F bit-sliced 624/592/624/604; E rings K=8 489/464/489/473 (WRONG by 950–5,600); B mask-risc 61.7k–88.9k (materialises a weight lane) at densities .01/.1/.5/.9. F exact vs direct geometry (asserted). Early exit H equals full decision, ~−20% vs F. Notes `Cascade` stroke 1 "is a statistical prune with no exact fallback"; `lsi.rs mexican_hat` "is a five-step percentile function, not a DoG"; Pillar-15 stub reports `passed: true`. Target-cpu not stated.
  - #1418: Q (per-query D/F choice) within 0.93–1.33× of the faster — ADOPT; HD/HQ early exit — REJECT. D-ART-1: 36,000 random cases equal a materialising oracle; CrossPowerSums affine summary path bitwise equal to materialise-and-fold (i128), 2.5–4× cheaper, warm transform 14 ns; static array lookup 1.2 ns vs SipHash HashMap 27 ns. Recipe table PARKED (no arithmetic IR; Quack IR Boolean-only).

## nd#344 — PowerSums/CrossPowerSums::checked_affine (merged 334dee5c)

Adds `PowerSums::checked_affine(a,b)` and `CrossPowerSums::checked_affine([a,b,c,d],[tx,ty])`: S' = A·S + n·t; M' = A·M·Aᵀ + A·S·tᵀ + t·Sᵀ·Aᵀ + n·t·tᵀ; exact i128, every step checked, `None` on overflow. Motivated by lg#1418 D-ART-1 (2.5–4× cheaper, 14 ns warm). **Oracle:** 400 random cases vs folding the transformed rows through `masked_group_cross_power_sums_i32` (1–300 rows, coeffs [−5,5], translations [−100,100], random masks, 3 groups). Precondition pinned: group key must not depend on a transformed column. 6 disable runs (dropped terms / unchecked narrowing) red; first overflow fixture was vacuous and rebuilt. `cargo test -p ndarray --lib` 2547 passed (debug 0). **Review changed claim:** Codex P2 — composition law false under intermediate overflow (translate by i64::MAX then back → None stepwise, Some composed). Now stated "whenever both return Some"; pinned. Scalar code, struct method, not a SIMD primitive.

## nd#354 — stroke-split pin (merged 8afa92da)

Final content: test `stroke_split_equals_the_cascade_split_at_2048` pinning `hpc::packed::PackedDatabase` and `hpc::cascade::PackedDatabase` split a 2048-byte row identically (128/384/1536); doc cross-links; note `cam_pq::PackedDatabase` shares only the name. **Withdrawn (negative):** `mask_row_window` (signed i64 column) and `bit_sliced_weighted_count` (signed sum over ± planes) reverted — "Signed masks do not belong on this facade. The use case ... a 256×256 tile, is the palette256 × palette256 address (`u8:u8`). That address is answered by the 256×256 distance and compose tables, not by geometric windows." CodeRabbit had flagged both as W1a-contract violations (scalar-only public free functions).

## Q5 — operator-authored text proposing tenants 18–20

**Answer: ABSENT in committed text.** Search space (closed): `.claude/` + `docs/` on main 1573eb4c for `Nars16x8|MoorePalettePairs|MooreNars16|tenant 18|ordinal 18|tenants 18` → 8 files, all agent-written: plans/2026-10-08-resident-projection-fold-mask-v1.md, board LATEST_STATE.md, STATUS_BOARD.md, entries 2026-10-08-{ce64-isa-register-contract, hhtl-nars-moore-value-tenants, moore-nars16-isa-visible-representation}.md, v3/soa_layout/{le-contract,tenants}.md. None contains an operator quote proposing the tenants (grep -i operator/ruling/verbatim in the two tenant entries: 0 hits).
- **Earliest occurrence in git (`git log -S`):** `MooreNars16` — commit **3581b1f5** (2026-10-08 07:40:55 +0000), author line `Claude <noreply@anthropic.com>`, msg "probe: MooreNars16 as a Moore-local representation of ISA-visible CE64 state", trailer `Claude-Session: …session_01EFw2WdKr1oxvaKCJC2ua2R` (lg#1406). `Nars16x8` and `MoorePalettePairs` — first at **28ec7c07** (2026-10-08 08:52:26, `Claude`, "contract: append Nars16x8, MoorePalettePairs, MooreNars16 value tenants", lg#1408).
- stockfish-rs `docs/NARS-RECIPE-LOCO-PALETTE-INVENTORY-V1.md` (the coresearch source) has no Nars16/MooreNars occurrence (grep). The coresearch entry `2026-10-08-coresearch-ce64-moore-masking-wiring.md:51` records only a priority: "Operator priority (2026-10-08): CE64 + Moore + masking first, with cognitive-shader-driver reasoning kept in the loop on the newest masking ops." — not a tenant proposal.
- GitHub org-wide issue search is blocked in this session (403, repo-scoped only), so PR bodies outside the listed PRs were not searched. Within the listed PR bodies the only operator quotes are in lg#1464 (V3/hotplug ruling), unrelated to 18–20.
- If the tenant layout came from an operator instruction, it was in chat, not in any committed artifact. Note the session id: **#1406, #1407, #1408, #1410, #1415, #1418, #1440, #1456 and nd#344/#354 all carry session_01EFw2WdKr1oxvaKCJC2ua2R — the same session id as this recovery session's attribution trailer.** #1400 = session_0177nSS…, #1411/#1426/#1428 = 01R2Eu…, #1432 = 017YPZ…, #1464/#1465 = 019fbZ…, #1427 none.

## Q1 call-site census — every reference to MooreTenantView / MooreTenantMut / Nars16x8 / MoorePalettePairs / MooreNars16 (lance-graph main; ndarray and OGAR: 0 hits)

| file | kind | what |
|---|---|---|
| lance-graph-contract/src/canonical_node.rs:1129-1155, 1346-1358, 1479-1481 | definition | enum variants 18/19/20, descriptors, Full-schema compile assert |
| lance-graph-contract/src/moore_tenant.rs (70 hits) | definition + 11 unit tests | View/Mut/lift_ce64 |
| lance-graph-contract/src/lib.rs:128-129 | `pub mod moore_tenant` | — |
| cognitive-shader-driver/tests/moore_palette_pairs_lut.rs:14,42,46,63 | TEST (F) | writes/reads pairs vs PairAddress |
| lance-graph-planner/tests/moore_tenant_isa_equivalence.rs:13,86,90 | TEST (J) | lift_ce64 + View vs strict ISA |
| lance-graph-planner/tests/moore_nars16_probe.rs | TEST, own probe-local `struct MooreNars16(u16)` at :112 | not the contract type |
| perturbation-sim/examples/{MoorePertubationsfeld,moore_observable_first}.rs | example | `MooreSlot` enum only (slot order); no tenant bytes |
- **Production (`src/`) readers/writers outside the contract: 0.** The Gomoku learning loop (`moore_nars_gomoku_probe.rs`) has 0 references; it learns in `HashMap<u64,[Ev{pos:f64,neg:f64};NO]>` and emits `CausalEdge64` records. SlabReading has no variant for 18–20 (`hotplug.rs:231-257`). This matches #1408's own "There is no production reader or writer" and plan D-RPF-5 "with no production reader; no `SlabReading` variant declares them".

