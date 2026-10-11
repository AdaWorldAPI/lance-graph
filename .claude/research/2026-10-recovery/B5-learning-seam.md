# B5 — the seam between the demonstrated behavioural learner and the canonical NARS/Moore representation

Checkout: /home/user/_ro/lance-graph @ main 1573eb4c. Read-only. Grades: VERIFIED-IN-CODE (read, file:line), CLAIMED (prose/doc only), ABSENT (searched the named place, not found).
Probe paths below are relative to crates/cognitive-shader-driver/examples/ unless noted.

## 1. The learner — moore_nars_gomoku_probe.rs (2366 lines, read in full; PR #1400 "D-MOORE-NARS-0", merged 2026-10-08; PR #1400 files = this probe + Cargo.toml + board entry 2026-10-08-moore-nars-0-recipe-learning-gomoku.md + STATUS_BOARD — so the "Moore-local recipe learning code from #1400" IS this file; later lab v0/v1 stages were added after)

### Task / world (VERIFIED-IN-CODE)
- Gomoku, n×n (9×9 five default, 11×11 transfer). `Board{n,win,cell:[u8;169],stones}` :237-243. Moore hydration = empty cells with a stone in 8-neighbourhood `candidates()` :346-366.
- Snapshot = per-candidate readings for both sides `Cell{i,s_me,c_me,s_op,c_op,dir}` :377-385, built by `snapshot()` :392-412 (2 folds per candidate). Line class ladder OPEN2..FIVE :229-235.
- Mechanical propagation (own win, else forced block) `propagate()` :442-450 — only if None does the learner choose.
- Opponents (deterministic): `Opp::{Threat,Aggressor,Lookahead}` :464-487; Lookahead = `minimax(k=5,5)`.
- Seeded opening: 2..=4 stones near centre :1112-1124. Colour alternates per episode :1334.

### Recipes (VERIFIED-IN-CODE)
- 9 realized operators `Op::{Hpm,Rcr,Tcp,Cur,Tr,Icr,Asc,Cr,Tcf}` :179-201, run by `run_op()` :622-710. Census of all 34 catalogue recipes :62-173, pinned against `contract::recipes::RECIPES` by `census_covers_the_catalogue_exactly` :2142-2161.
- ASC = 2-ply minimax top-4×top-4 (:681), or `minimax_certified` (:563-605) when `cfg.certified` (:680) — same pick by construction, fewer folds (pinned :2108-2140: k4/4 442,720→282,480 folds; k5/5 665,076→367,344; k8/8 1,601,258→624,386 on a 400-position corpus).
- NOTE (premise): the "certified minimax arm" is NOT a chooser arm; it is a `Cfg.certified` flag that swaps ASC's realization. H4 compares "cost 0" vs "cost 0 + cert" :1510-1523.

### Context key (VERIFIED-IN-CODE)
- `signature(s) -> u64` :717-730 = own best class (3b) | opp best class <<3 | min(#opp cells ≥OPEN3,2) <<6 | conflict(attack≠block) <<8 | size bucket(0..10/11..25/26+) <<9. Coordinate-free, relative.
- Backoff `coarse(s)` :732-734 = own|opp<<3|1<<40.
- `Memory::{Structural,Literal,Global}` :753-760; Literal key = FNV board hash `Board::hash` :367-373; Global key = 0 :1203.

### Learner state — the exact type (VERIFIED-IN-CODE)
```rust
// :811-815
#[derive(Clone, Copy, Default, Debug)]
struct Ev { pos: f64, neg: f64 }
// :861-866
#[derive(Default)]
struct Learner {
    stats: HashMap<u64, [Ev; NO]>,      // key = signature/coarse/board-hash/0 ; NO = 9 recipes
    folds: [(u64, u64); NO],             // global (count, total folds) per recipe, for the cost term
}
```
Expectation (NARS-style, :818-826): w = pos+neg; f = pos/w; c = w/(w+1)  [k=1]; e = c·(f−0.5)+0.5; w==0 → 0.5.
Projection to canonical 5-bit state `Ev::state(topology) -> EpistemicState5` :827-841: Certification3::{Open if w==0 or f≤0.5; Associated if c<0.5; Supports if f<0.7 or c<0.8; else CausalCandidate}, EpistemicState5::new(Epi5Gen::V1, topology, cert).

### Update function — quoted (VERIFIED-IN-CODE :900-927)
```rust
fn credit(&mut self, cfg: &Cfg, p: &Pending, activation: i32) -> EpistemicState5 {
    let mut out = Ev::default();
    let keys: &[u64] = match cfg.memory {
        Memory::Structural => &[p.key, p.back],
        Memory::Literal | Memory::Global => &[p.key],
    };
    for &k in keys {
        let row = self.stats.entry(k).or_insert([Ev::default(); NO]);
        let e = &mut row[p.op.index()];
        e.pos *= cfg.decay;
        e.neg *= cfg.decay;
        let a = f64::from(activation) / 7.0;
        if a > 0.0 { e.pos += a; } else if a < 0.0 { e.neg -= a; }
        if k == p.key { out = *e; }
    }
    let topology = if cfg.signal == Signal::Final { Topology2::IndirectKnown } else { Topology2::Direct };
    out.state(topology)
}
```
Key facts:
- Evidence is FRACTIONAL: activation ∈ [−7,7] integer, credited as a/7 ∈ {1/7..1}, so pos/neg are f64 sums of sevenths (rational with denominator 7, before decay). With decay≠1.0 they become arbitrary reals.
- Decay is per-UPDATE of a pairing (multiplicative on that cell only, applied before the add), not per time step. BASE decay = 1.0 (none) :799-808.
- Structural memory writes BOTH the fine key and the coarse backoff key (two cells per credit).
- Zero activation adds nothing (no evidence for neutral outcomes).

### Activation / outcome oracle used as training signal (VERIFIED-IN-CODE)
- `activation(before, after, contradiction)` :932-937: contradiction → −7, else clamp(2·(after−before), −7, 7). balance = best_me class − best_op class :453-455.
- `Signal::{Retained(h), Immediate, Contradiction, Final}` :740-750. Retained(1) (BASE) reads the board at the learner's next turn after the opponent replied, through the W ring (`ring:[Option<Pending>;64]`, `queue`, `wslot`) :1127-1193. End-of-game flush maps result to ±7/0 :1276-1295. Final spreads 7·result over all steps :1296-1300.
- `Pending{key,back,op,before,immediate,dir,question:CausalMask,pred}` :846-859.

### Decision rule (VERIFIED-IN-CODE `Learner::choose` :869-899)
- Fixed(op) → op; Uniform → seeded uniform over 9.
- Learned → argmax over recipes of `e − cost_weight·ln(1+mean_folds)`; row = stats[key] else (Structural) stats[back] else all-default (e=0.5 everywhere → tie → first, HPM, since `v > best.0` strict and HPM first).
- Thompson → draw Beta(pos+½, neg+½) via `lab::beta_sample` (shared/lab.rs:258) instead of e, same cost term. Draws consume the shared game rng.
- BASE: Learned, Retained(1), Structural, no reset, decay 1.0, cost 0.02, learn on, not certified :799-808.

### Arms (VERIFIED-IN-CODE)
- main(): Stage 1 fixed×9 + uniform vs Threat, 3 seeds (11,22,33) × 300 games :1746-1763; Stage 1b fixed vs Lookahead 3×100 :1765-1779; Stage 2-3 ten arms × 2000 episodes, windows 400 :1784-1868 (persistent base / reset every game / Retained(3) / contradiction-only / immediate / final-only / literal memory / cost-blind / decay 0.98 / uniform); preference dump :1870-1896; Stage 4 transfer (train on 3 seeds, freeze `learn:false`, unseen seeds 101/202/303, 9×9 and 11×11) :1898-1932; Stage 5 switch Threat→{Lookahead,Aggressor} after 2000 eps, no decay vs decay 0.98, vs fresh learner and fixed HPM/ICR :1934-1988.
- `--lab` (v0): 16 blocks seeds 2001.., 300 games, endpoint last 100 window; arms base/immediate/cost0+cert/cost0/fixed HPM/decay0.95, + switch blocks; prereg H3,H4,H7,H2,H5; paired sign-flip p + Holm :1436-1614.
- `--lab-v1`: seeds 4001.., arms learned/Thompson/global(Memory::Global), switch; prereg T1..T4 :1618-1714.
- "context" arm = Memory::Global (one key, context-free) vs Structural (T3); Literal is the memory-control arm.

### Seeds / n
- Default runs: seeds [11,22,33]; game seed = seed·1_000_003 + episode :1339. Per-step recipe rng is a fresh `Rng(step_seed)` drawn from the game rng :1219-1220 (makes TR/Thompson reproducible).
- Lab: 16 blocks × 300 games/arm; v0 seeds 2001..2016, v1 4001..4016.

### What is measured (VERIFIED-IN-CODE `Stats` :962-986)
score (win=1, draw .5), W/L, moves, mechanical moves, reasoning steps, folds and folds/step, recipe histogram (top share), productive fraction (retained>0), contradictions, memory hit fraction, oracle regret/optimality (`oracle_regret` :1052-1070 replays all 9 recipes one reply deep — same signal as training, so not independent; the board entry says so), calibration bins of the chooser's e vs observed productive (all and non-neutral) → ECE via lab::reliability.

### Independence of the oracle (premise)
- The training signal and the regret oracle are both the one-reply board consequence vs a deterministic opponent. The ONLY independent yardstick in the file is the game result (score). VERIFIED-IN-CODE (:1052-1095 vs :1159-1192) and stated in the board entry Caveats.

### CE64 trace (VERIFIED-IN-CODE)
- After each credit, `record(p, slot, a, surprise, epi)` :950-958 builds a CausalEdge64: causal mask = the step's question (PO/SO/SP), direction = move's Moore dir, inference mantissa = activation, plasticity bits = surprise (|a/7 − (2e−1)|·3.5), W slot = ring slot, epistemic raw5 = Ev::state(). Pushed to `trace`, then the trace is CLEARED after every game (`trace.clear()` :1344) and never read by the chooser.
- `the_register_never_names_the_recipe` :2188-2209: the CE64 record is identical across all 9 recipes — the record carries NO recipe identity and NO pos/neg magnitude.
- Consequence: the CE64 is a write-only PROJECTION of the learner's state; the authoritative state is the f64 HashMap. VERIFIED-IN-CODE.

### Measured results on record (CLAIMED in board entries, produced by this file; not re-run here)
- .claude/board/entries/2026-10-08-moore-nars-0-recipe-learning-gomoku.md: persistent structural learner 0.500→0.535 last window vs best fixed 0.498/0.493, reset ~0.47, literal ~0.49 (hits 0.3–3.5%), immediate ~0.50 (98–99% HPM), uniform ~0.24; ~280 folds/step vs 0 for HPM; 169 signatures.
- 2026-10-10-d-self-calibrating-lab-0.md: H2 (learned > HPM) +0.0238 SUPPORTED; H4 SUPPORTED; H7 SUPPORTED; H3 INCONCLUSIVE; H5 (decay 0.95 adapts faster) FALSIFIED; ECE 0.317 all outcomes / 0.050 non-neutral.
- 2026-10-10-d-self-calibrating-lab-1.md: T1 Thompson worse (−0.050) FALSIFIED; T2 FALSIFIED; T3 structural > global +0.0247 SUPPORTED; T4 FALSIFIED.

## 2. moore_plasticity_probe.rs (465 lines, read in full) — NOT a learner
- D-GSO-2: one synchronous gated Moore hop over a 16-byte `Register128` read as 4×4 (`moore_step` :101-109, `fold_lane` :113-123), law = `PalettePerturbation::hop` over two 64 KiB Palette256 LUTs (relation, perturb). Gate = u16 lane mask. Schedule = closed-form `moore_mask` :67-83, proven equal to `ndarray::simd::mask_shift_morton` on the Morton reading (:447-464).
- It has NO reward, NO evidence, NO NARS truth. "Plasticity" here = a lane-freeze gate on a deterministic palette update. VERIFIED-IN-CODE. It is the only canonical "Moore shift" operation in the area, and it updates BYTES by a fixed law; it cannot accumulate evidence.

## 3. The canonical representation (all read)

### Tenants 18-20 (VERIFIED-IN-CODE canonical_node.rs :1129-1155 enum doc; :1339-1363 descriptors; moore_tenant.rs full)
| tenant | ordinal | row bytes | kind | reading |
|---|---|---|---|---|
| Nars16x8 | 18 | 284..300 | 8 × LE u16 | layout RATIFIED, **reading CANDIDATE and undeclared** (canonical_node.rs:1131-1133; moore_tenant.rs:8). Nothing defines u16 = (f u8, c u8) or counts. |
| MoorePalettePairs | 19 | 300..316 | 16 × u8 = 8 (first,second) | Palette256 law address `(first<<8)|second`; pinned by tests/moore_palette_pairs_lut.rs (512 rounds, orientation load-bearing >3000 swaps differ). |
| MooreNars16 | 20 | 316..332 | 8 × LE u16 | `Pearl3|Energy4|Plasticity3|Polarity1|Epi5` (moore_tenant.rs:107-121). **No F, no C, no witness, no S/P/O sign triple.** |
- All three are in `ValueSchema::Full` only (moore_tenant.rs:428-435). Lanes are indexed by Moore SLOT (NW..SE), not by recipe or by situation.
- `lift_ce64(edges:&[u64;8], polarity)` (:370-395) reads CE64 bits 40..63 only, refuses mixed witness / reserved Epi5, writes nothing on refusal. `ce64_upper(witness)` (:213-219) rebuilds bits 40..63 with direction=0; "Bits 0..39 (S/P/O, F, C) are the caller's" (:211). So F and C are DISCARDED by the lift. VERIFIED-IN-CODE.
- tests/moore_tenant_isa_equivalence.rs:95 rebuilds `(edges[k] & PAYLOAD40) | lane.ce64_upper(w)` — the equivalence (0 divergences / 256 rounds, :113-115; wrong witness observable :119-122) holds BECAUSE F/C are taken from the ORIGINAL edge, not from the tenant. tests/moore_nars16_probe.rs MooreTenant keeps `payload: [u64;8] // SPOFC40 per lane` (:242 region) outside the lane; header :1-13 says the claim is "MooreContext + SPOFC40 + MooreNars16 ≡ CE64 under the ISA". Consequence: **no canonical tenant stores the F/C a revision reads or writes**. VERIFIED-IN-CODE.

### CE64 truth + revision (VERIFIED-IN-CODE)
- Field spans (isa.rs Field::span): Spo 0..24, Frequency 24..32 (u8), Confidence 32..40 (u8), Pearl 40..43, Direction 43..46, Inference 46..50, Plasticity 50..53, Witness 53..59, Epistemic 59..64.
- `truth::evidence_weight(c) = c/(1-c)`, saturates to f32::MAX at c ≥ 0.999 (isa.rs ~:216-222).
- `truth::revision(f1,c1,f2,c2)`: w1+w2 pooled, f = (f1w1+f2w2)/ws, c = ws/(ws+1); None if ws ≤ ε (isa.rs ~:252-262). k = 1 — the SAME horizon the learner uses (`c = w/(w+1)` at probe :824).
- `CausalEdge64::revision(rhs)` writes only F/C via `set_frequency/set_confidence` = `(x*255).round() as u8` (edge.rs:371-379, :723-734). `learn()` (edge.rs:827-) also revises and additionally freezes plasticity at c > 0.9 / 0.7.
- `NarsTables::revise(f1,c1,f2,c2)` (tables.rs:126-131) bins confidence into `c_levels` (16 in the chain tests) — a second, coarser quantization.

### EpistemicState5 (VERIFIED-IN-CODE epistemic_state5.rs :1-60, enums :125, :173, `new` :325)
- raw5 = Topology2 (2b) | Certification3<<2 (3b); 24 meaningful states. The module doc states: "Observation is NOT a coordinate ... lives in the evidence path (witness, SupportLedger, revision input)" and reads go through declared `(classid, rail) → Epi5Reading`.
- The learner's `Ev::state` (probe :827-841) maps statistical (f, c) thresholds onto Certification3 {Open, Associated, Supports, CausalCandidate} — never Related, never Causes. The contract defines Supports = "supports in every declared stratum", CausalCandidate = "supports, and exposure precedes outcome" (epistemic_state5.rs fact consts). The probe's thresholds (f<0.7 or c<0.8) are an AD-HOC projection, not the declared semantics. Grade: VERIFIED-IN-CODE that the mapping is threshold-based; the claim that this IS certification is UNSUPPORTED.

### Duplicate-evidence behaviour of the canonical path (VERIFIED-IN-CODE, test-pinned)
- tests/chain_confidence.rs (lance-graph-planner): `replay_step` revises each hop, confidence 200→224→237→237… (:72-82); self-revision raises confidence (:89-106); a c=0 weight still adds confidence 128→137 through 16-bin NarsTables (:113-125). No stamp / evidential-base check on this path.
- tests/gadamer_isa_revision.rs: `CausalEdge64::revision` alone inflates an echo (10× same evidence → c > 200, :265-271); with `GadamerRevision` above it, echo and closed cycles never reach the register (c stays 100, :230-246); new independent roots each round are admitted (:250-260). Roots are an `EvidenceMask` (u64 or [u64;N], revision.rs:27-44); the policy is self-described as "a textual-hermeneutic policy, not a universal truth algorithm" (revision.rs:20-22).
- causal-edge/tests/ce64_op_contract.rs: `forward` silently zeroes W and Epi5 (:142-148); `learn` and single-field setters preserve them (:115-134); syllogize conclusions carry neither (:153-175); Counterfactual weight faults (:85-96).

### Masked reductions (VERIFIED-IN-CODE lance-graph-mask-risc/src/ir.rs)
- `Terminal::MaskedSumI32`, `GroupSumI32{mask,key,val}`, `GroupReduce{mask, key: GroupKey::{Lane,Via,Pair{hi,lo,stride}}, fold: GroupFold::{Count,MinI32,MaxI32,SumSymI32}}` (:290-445, :488-524), `GroupPowerSumsI32` → `PowerSums{n:u64,sum:i64,sum_sq:u128}` (:446-470; ndarray simd_masking_ops.rs:2118-2125), `GroupCrossPowerSumsI32` (:471-486). Merge law `merge_group_sink` (:663). Exact up to 2^32 rows.
- Bounded u8 power sums into a `Register128` (4 LE u32 words) per group row, widened losslessly to `PowerSums` across 65,536-row tiles: crates/jc/tests/register128_bounded_stats.rs (header :1-5), ndarray `fold_bounded_power_sums_tiles` (simd_masking_ops.rs:3030), `BOUNDED_TILE_ROWS = 1<<16` (:2633). Register0/Register1 are tenants 16/17 (canonical_node.rs:1124-1128).

## 4. Answers

### (a) Does the learner read or write tenant 18-20 / CE64 / EpistemicState5 state?
- Tenants 18-20: **ABSENT.** The probe imports `causal_edge::{CausalEdge64, CausalMask}`, `PLAST_MASK/PLAST_SHIFT`, and `epistemic_state5::{Certification3, Epi5Gen, EpistemicState5, Topology2}` (:31-33); it never imports `moore_tenant`, `canonical_node`, `ValueTenant`, `Register128` or any slab. Searched the full file. 
- CE64: **WRITE-ONLY.** `record()` (:950-958) packs one CE64 per credit into `trace`; `trace.clear()` after every game (:1344); nothing reads it back except the two structural tests (`the_register_never_names_the_recipe`, `w_follows_the_pending_slots`). The F/C bytes of the record are never set (CausalEdge64(0) base; only mask, direction, mantissa, plasticity, W, epi5).
- EpistemicState5: **WRITE-ONLY projection** of the f64 state into the CE64 record (`Ev::state`, :827-841); never read for a decision.
- The authoritative state is `Learner.stats: HashMap<u64,[Ev;9]>` (f64 pos/neg) + `folds:[(u64,u64);9]`. VERIFIED-IN-CODE.

### (b) Field → canonical carrier → exact / lossy / missing
| learner field (where) | candidate carrier | verdict |
|---|---|---|
| context key u64 (signature 11 bits, coarse with bit 40, board hash 64 bits, or 0) :717-734, :1200-1205 | NodeRow key / group id in a `GroupKey::Lane`/`Pair{hi=key_id, lo=recipe, stride=9}` | EXACT for Structural (signature < 2^11; coarse needs a disjoint id range instead of bit 40). Literal board hash: needs a dictionary; it's the control arm anyway. No canonical tenant holds it: tenants 18-20 are slot-indexed, not situation-indexed. |
| recipe index 0..9 | group minor key | EXACT. Not a Moore slot; 9 ≠ 8 lanes. |
| pos (f64 = Σ a/7 over a>0, ×decay) :909-914 | (i) CE64 F/C u8 via revision; (ii) Nars16x8 u16 lane; (iii) GroupSumI32 / bounded PowerSums over an integer lane `max(a,0)` | (i) LOSSY and biased (see sim below); (ii) reading undeclared, 8 lanes < 18 needed per key, u16 overflows at 65,535/7 ≈ 9,362 unit events — and it is a slot axis; (iii) EXACT when decay = 1.0: store pos7 = Σ max(a,0) ∈ ℤ (activation is integer −7..7 :936), Σ fits i64 / bounded-u8 Register128 per tile. |
| neg (same, a<0) | same | same; second lane `max(−a,0)` or second Register rail. |
| decay (multiplicative, per update of THAT cell) :909-910 | none | **MISSING.** Not a sum: the weight of an event depends on how many LATER updates hit the same cell. Expressible only with a per-cell rank/scan (weight = d^(later count)) — no canonical rank-weighted fold. BASE uses decay 1.0, and H5 (decay) was FALSIFIED, so the base arm does not need it. |
| fold cost per recipe (u64 count, u64 total) :1227-1230 | GroupSumI32/Count keyed by recipe | EXACT (integers). |
| expectation e = c(f−.5)+.5 with c = w/(w+1) (k=1) :818-826 | consumer arithmetic over (pos7,neg7) | EXACT as a rational of integers; argmax + `ln(1+mean)` cost term is f64 consumer math — no canonical "expected-value argmax over a group row" op (not needed; it is a 9-element read). |
| Thompson Beta(pos+½,neg+½) draw :885-887 | consumer RNG | not a representation question; deterministic under seed. |
| backoff (fine key miss → coarse key) :874-880 | two group rows | EXACT (two reads). |
| Pending ring (64 slots), W slot :1132-1261 | CE64 W bits 53..58 (6 bits = 64) | the W slot number fits; the Pending CONTENT (key, back, op, before, immediate, pred) has no carrier — it's transient control state. W is tenant-scoped in MooreNars16 (one W for 8 lanes), which does not match one W per pending step. |
| EpistemicState5 projection | CE64 59..63 / MooreNars16 Epi5 | representable (5 bits), but SEMANTICALLY WRONG to use as statistical confidence (see (d)). |

Replica measurement of the CE64-F/C carrier (python, `scratchpad/recovery/sim.py`: replays `truth::revision` + `set_confidence` round-to-u8 exactly as read at isa.rs:252-262 and edge.rs:371-379; NOT a cargo run — grade: MEASURED-BY-REPLICA, not repo-pinned):
- 2000 random events, |a| ∈ 1..7, P(+)=0.55: true w 1155.9, f 0.551, c 0.9991 → u8 path ends F=124 (f 0.486), C=244 (w 22.2). **The frequency crosses 0.5 to the wrong side**, so e flips below 0.5 — this changes the argmax.
- P(+)=0.70: true f 0.720 → u8 f 0.765, again C=244, w ≈ 22.
- C saturates at 244 (w ≈ 22) for unit-weight evidence: revising c=244 with a ≤ 1-weight observation rounds back to 244. The u8 register is an implicit EXPONENTIAL-FORGETTING learner with an effective window of ~22 unit events, not the persistent learner. (So Arm B through CE64 F/C is closer to the falsified decay arm than to BASE.)
- 20 identical +7 echoes: u8 C reaches 244 just like 20 independent events — the register cannot tell them apart (matches gadamer_isa_revision.rs:265-271).

### (c) Which operation connects them
- The learner's credit IS NARS revision in real arithmetic: with belief (f,c) ↔ (pos,neg), w = pos+neg, and an observation of frequency 1 (a>0) or 0 (a<0) with weight |a|/7, `truth::revision` yields w' = w + |a|/7, f' = (f·w + [a>0]·|a|/7)/w' — identical to `pos += a` / `neg -= a` at probe :911-915 (k = 1 on both sides). Algebraically VERIFIED (both formulas read). So the SEMANTIC operation exists (`Opcode::Revision`, `CausalEdge64::revision`).
- What is broken is the CARRIER of that operation: u8 F/C (lossy, saturating, sign-flipping f), and NarsTables 16-bin c (adds confidence from c=0 weights, chain_confidence.rs:113-125).
- The EXACT connection available today is a masked grouped reduction over an event table (one row per credited step: key_id u32, recipe u32, pos_act i32/u8, neg_act i32/u8, plus provenance lanes): `GroupSumI32` (or `GroupReduce`/`GroupPowerSumsI32`) with `GroupKey::Pair{hi: key_id, lo: recipe, stride: 9}`, mask = "credited, learn on, not frozen". Exact for decay = 1.0, order-independent, mergeable (`merge_group_sink`), and bounded-u8 into Register0/Register1 group rows.
- The Moore shift (`mask_shift_morton`, plasticity probe) is NOT the connecting op: the learned table is keyed by a population-level signature of the whole candidate set, not by a cell or a slot. There is no Moore-local learned state anywhere in the learner.
- GENUINELY MISSING:
  1. A declared reading that maps a group row (or Nars16x8 lane) to `(pos7, neg7)` evidence counts — Nars16x8's reading is undeclared; Register128's bounded-stats reading is declared only as power sums.
  2. An ONLINE incremental fold: the decision at step t needs the state through t−1. The mask-risc terminals are batch folds over a plane; re-running per decision is O(T²). Per-row incremental accumulate into a persistent group sink (equivalently: GroupSum over a one-row extent + `merge_group_sink`) may be expressible — must be TESTED, not assumed.
  3. Decay ≠ 1 (rank-weighted fold) — missing, but not required by BASE.
  4. A wide (non-u8) truth carrier for revision if B insists on CE64: missing; would need an F/C-as-counts reading, which would be a new reading, not a CE64 change.

### (d) Duplicate evidence / echo, and keeping statistical confidence separate from CE64 epistemic state
- Canonical path: YES, duplicates raise confidence. `CausalEdge64::revision` pools whatever it is handed (gadamer_isa_revision.rs:265-271: 10× same evidence → c > 200); `replay_step` self-revision raises c (chain_confidence.rs:89-106) and even a c=0 weight raises it via 16 bins (:113-125). Only a gate ABOVE the register (GadamerRevision, roots as EvidenceMask) stops echo/closed cycles (:230-246). VERIFIED-IN-CODE / TEST-PINNED.
- The HashMap learner has the same property: replaying the same game adds the same evidence again; nothing in `credit()` checks provenance. Within one game each credited step is a distinct event, but across seeds/blocks a replayed game (same seed) would double-count. VERIFIED-IN-CODE (no stamp field in Ev or Pending).
- Separation proposal (not implemented): three distinct objects, never the same bits —
  1. **Statistical evidence** (pos7, neg7, n) in group rows / Register rails, written ONLY by the masked fold over the event table; every event row carries a provenance stamp (seed, game, move index) so duplicate rows are detectable and an echo arm can be built by duplicating rows deliberately.
  2. **The chooser's expectation** (e, Beta draws) = consumer projection of (1), never stored.
  3. **CE64 EpistemicState5** = declared certification only (Open/Associated/... with the contract's meanings), written by an explicit certification step, never derived from c thresholds. The probe's `Ev::state` threshold mapping should be treated as a diagnostic label, not canon.
  The experiment's statistical confidence (paired sign-flip p, Holm, CIs from shared/lab.rs) is computed over BLOCKS and never written into any CE64 field.

## 5. Proposed three-arm experiment (do not implement)

### Arms (same probe, same `run_op`, same opponents, same signature)
- **Arm A — existing HashMap learner** (`BASE`: Learned, Retained(1), Structural, decay 1.0, cost 0.02). Unchanged.
- **Arm B — canonical state.** Each credited step appends one event row to a column table (lanes: `key_id: U32` (fine and coarse as two rows or a remapped id), `recipe: U32`, `pos: I32 = max(a,0)`, `neg: I32 = max(−a,0)`, `stamp: U64` (seed, game, ply)). The chooser's row for (key, recipe) is read from a group sink produced by `GroupSumI32`/`GroupReduce` with `GroupKey::Pair{hi:key_id, lo:recipe, stride:9}` (or bounded power sums into Register0/Register1 group rows), maintained incrementally (one-row extent + `merge_group_sink`). e computed from (pos/7, neg/7) with the SAME k=1 formula. The CE64 record and EpistemicState5 are written exactly as today, write-only.
  - **Sub-arm B′ (the representation probe):** drive the decision from CE64 F/C via `CausalEdge64::revision` per (key, recipe) cell. Prediction from the replica: diverges (w capped ≈ 22, f biased across 0.5); compare against A and against A-with-decay to locate it.
- **Arm C — negative controls (three, each needed):** C1 no learning (`learn:false`, untrained → always HPM via tie); C2 shuffled credit (permute activations across steps within a block, keeping the marginal distribution); C3 echo (each credited event appended twice / replay each game twice) — tests whether the arm is sensitive to duplicate evidence; for B, also with a stamp-dedup mask, which must restore A's behaviour.

### Matched scenarios
- 9×9 five vs Threat, 16 blocks, fresh seeds (not 11/22/33, 2001.., 4001..), 300 games, endpoint last 100 window; opponent switch Threat→Lookahead for adaptation (Aggressor was uninformative, board entry). Same per-step rng stream (step_seed) in all arms so decisions are comparable move-for-move.

### Independent outcome oracle
- Game result only (win=1, draw=.5) as the primary endpoint; the one-reply regret oracle is NOT independent (same signal as training) and is reported as secondary. Optional external: frozen evaluation vs Lookahead on unseen openings (Stage 4 shape).

### Metrics
- **Decision agreement A vs B:** fraction of reasoning steps choosing the same recipe on the identical trajectory (run B as a shadow chooser on A's trajectory — avoids divergence after the first disagreement); pre-register "B ≡ A" as 100% under decay 1.0 (it is the same arithmetic; any miss = a carrier bug).
- Held-out score (frozen, unseen seeds), post-switch adaptation (score over first 2 windows after switch).
- Folds per decision (reasoning folds, and separately the bookkeeping cost of the reduction in rows touched).
- Materialized bytes: HashMap rows×(9×16 B) vs event table + group sink bytes; heap allocations per decision (counting allocator in the probe).
- Replay determinism (byte-identical stats across two runs, as `replay_is_deterministic` :2354-2365).
- Calibration: ECE of e vs observed productive, all and non-neutral (lab::reliability).
- Provenance: every B decision traceable to the set of event stamps that formed its row; echo arm must show detectable duplicate stamps.
- Statistics: paired sign-flip p + Holm over blocks (lab.rs), never stored in CE64.

### If B cannot be expressed — the exact capability to test
1. "Incremental keyed fold": append one event row and update one (key, recipe) slot of a persistent group sink without rescanning the plane, bit-identical to a batch `GroupSumI32` over all rows so far. Falsifier: after every step, batch fold == incremental sink.
2. A declared reading of a group row (Register rails or Nars16x8) as `(pos7, neg7)` counts. Falsifier: round-trip + saturation refusal at the u8/u16/u32 bound instead of silent wrap.
3. (Only if decay is wanted) a rank-weighted fold; out of scope for BASE.

### Premise problems flagged
- 16-bit lanes cannot hold f64 sums; but the sums are integers ×1/7 (activation ∈ ℤ, :936), so exactness needs integer counters, not floats — f64 is an implementation choice, not a requirement (decay ≠ 1 aside).
- Tenants 18-20 are indexed by Moore slot; the learner is indexed by (board-level signature, recipe). "Moore-local recipe learning" is Moore-local only in HYDRATION (candidate cells), not in learned state. Mapping recipes onto 8 slot lanes would be a category error (9 recipes, no geometric meaning).
- MooreNars16 carries no F/C at all; "lift_ce64" drops bits 0..39. The ISA-equivalence result does not cover revision state.
- CE64 u8 F/C revision is not the persistent learner: it saturates at w ≈ 22 and biases f (replica). Using it for Arm B tests a different learner.
- EpistemicState5 thresholds in `Ev::state` overload certification with statistical confidence; contract says observation/evidence is not a coordinate.
- The certified-minimax "arm" is a realization flag (H4), not a chooser arm; it does not interact with the representation question except through the cost term.
- Gadamer roots are a bitmask EvidenceMask (u64 or [u64;N]); per-game roots at thousands of games need [u64;N] with N ≥ games/64, and the policy is declared textual-hermeneutic, so applying it to game outcomes is itself a hypothesis.

## Grades summary
- VERIFIED-IN-CODE: learner state type/update/decision (probe :811-927, :869-899); CE64 write-only (:950-958, :1344); no tenant use; MooreNars16 has no F/C (moore_tenant.rs:107-121, :211); revision formula + u8 rounding (isa.rs, edge.rs:371-379, :723-734); duplicate inflation (gadamer_isa_revision.rs:265-271, chain_confidence.rs:89-125); mask-risc terminals (ir.rs).
- MEASURED-BY-REPLICA (python, not repo-pinned): u8 F/C saturation at C=244 / w≈22 and f sign-flip.
- CLAIMED (board entries): score numbers, H/T verdicts, ECE.
- ABSENT: any tenant 18-20 read/write in the learner; a declared counts reading for Nars16x8/Register rails; an incremental keyed fold op; a rank-weighted (decay) fold; a provenance stamp in the learner.
