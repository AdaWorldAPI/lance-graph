# Perturbationsfeld probe — v1 (PROPOSAL, 2026-09-29)

> **Status:** PROPOSAL — no code written, nothing wired. Grounded against
> lance-graph `origin/main` **5282dfa3** (forensic pass run on **f52497e7**;
> the one intervening merge, #1299, touches only `deepnsm-v2`).
>
> **Prefix: `D-PFP-*`** (Perturbationsfeld probe). Rows in `STATUS_BOARD.md`.
>
> **One question:** does current code have the pieces for
>
> ```
> mask/program → non-materialized logical field → fold/resonance
>              → perturbation → cross-cycle consequence
> ```
>
> — and if it does, does the perturbation field's ADDRESS carry information
> into the next cycle when lowered through `lance-graph-mask-risc`? This is
> the photolithography / Perturbationsfeld question, asked as a falsifier.
>
> **CE64 is deliberately NOT the centre of this plan.** It stays the hot-path
> reasoning register; nothing here repairs, re-reads or promotes it.
>
> **Sibling arc:** `waben-fold-execution-loop-v1.md` (`D-WFL-*`). WFL builds
> the loop *mask → fold → result produces the next focus → publish* from the
> fold side. This plan asks whether the thinking-engine's perturbation field
> can be a *producer* of that next focus. It adds no carrier, no lane and no
> loop of its own; if it passes, its consequence enters WFL at D-WFL-W5.

Evidence labels used throughout: **[H]** current HEAD fact · **[M]**
mechanically verified implementation fact · **[HI]** historical
implementation · **[HC]** historical concept/metaphor · **[RV]** reverted
experiment · **[SH]** supported hypothesis · **[CH]** contradicted
hypothesis · **[U]** unresolved.

---

## 1. Forensic basis (what the code does today)

Five read-only investigations (RISC-mask; scales + materialization; engine_bridge
+ cycle trace; DTOs; lithography archaeology) plus a pass against
`.claude/v3/*`. Load-bearing lines were re-read by the orchestrator:
`driver.rs:320-342, 480-500, 655-664`, `engine_bridge.rs:113-150, 784`,
`p64-bridge/src/lib.rs:25-34`, `helix/examples/fire_forget_replay_probe.rs:41,110-114`,
`thinking-engine/src/engine.rs:160-166`.

### 1.1 Current execution map

```
 ═══ solid = production   ┄┄ = test/lab/conjectural   ✗ = no connection ═══

 serve.rs / grpc.rs (LAB) ──ingest_codebook_indices──► BindSpace (Arc, RAM, singleton)
                                                            │ edge(row)
 ┌──────── cognitive-shader-driver::ShaderDriver::run ──────▼──────────────────┐
 │ CE64(stored) ─s_idx()─► p64 cascade (planes [[u64;64];8], 4 KB)            │
 │     └─► NarsTables.revise ─► _revised_truth   ✗ DISCARDED (driver.rs:340)   │
 │ top-8 → cycle_fp (XOR-braid) → F / MUL / gate                               │
 │ fresh CE64' = pack(s=row%256, o=(row/4)%256, f=c=resonance)  (:480-495)     │
 │ awareness.revise (:661) + rung_elevator  ◄── ONLY cross-cycle state (RAM)   │
 └───────────────────────────────┬─────────────────────────────────────────────┘
                                 ▼ ShaderBus → sink.on_bus → NullSink/test only
                                 ├┄► WireBus (lab JSON/gRPC)
                                 ├┄► engine_bridge::persist_cycle (tests only)
                                 ├┄► EngineBusBridge→BusDto┄►commit_to_l4 (tests only)
                                 ✗ Lance

 thinking-engine (optional `with-engine`):
   perturb(&[u16]) ─► MatVec over MATERIALIZED 4096² u8 table (16 MB)
     ─► PerturbationDto{energy: Vec<f32> 4096, top_k:[(u16,f32);8]}
     ─► dispatch_from_top_k → ColumnWindow{start,end}   (dense energy DROPPED)

 lance-graph-mask-risc ✗──✗ shader / CE64 / p64 / thinking-engine (no dep either way)
   ◄── quack, report, sap, d-diamond-1-probe, lgj-abi plan_eval  (production)
```

### 1.2 What `lance-graph-mask-risc` is [M]

A **borrowing, straight-line mask-program ISA with exactly one evaluator** —
data, operation and address kept separate, not mixed:

- **data:** borrowed `&[u64]` bit-per-row planes (`ir.rs::Planes` :73), never owned;
- **operation:** `Program = ops: Vec<MaskOp> + one Terminal + scratch slots`
  (`ir.rs:511`); `MaskOp` = gated `Pred` / And / Or / Xor / AndNot / Not /
  `Ternlog{imm}` / `Gather` (:200-259);
- **address/view:** `Operand::{Plane,Scratch}` slot names; `LaneRef::Strided`
  field view over unmoved record bytes (:16, :42);
- **lowering:** `fuse.rs` truth-table → ternlog chains, generated 256-arm
  dispatch, `exec.rs` → `ndarray::simd` facade words; `forbid(unsafe)`, names
  no ISA.

Materialization is forbidden by doctrine and test: "a caller-owned bitmap is
still materialisation"; `Keep` is the one elected bitmap; one materialiser
(`materialize_rows`, guarded by `exactly_one_materialiser`). Production tile
= 256 words (16,384 rows), chosen by measurement. No 64×64, no Morton, no
resonance, no CE64 token anywhere in the crate. **Every operand is a borrowed
RESIDENT slice** — `LaneRef::{I32,U32,U64,Strided}`; there is no
generator / function-of-address operand.

Origin: IR skeleton `a7518e52` (PR #1225, 2026-09-14), executor PR #1226;
motivating plan `lance-graph-java/.claude/plans/mask-risc-lowering-v1.md`
(`3225c6c`, 2026-09-03), DuckDB-lineage, not shader-lineage.

### 1.3 Quack ↔ mask-risc convergence [M]

Complete, by construction: Quack "may build a `Program` and must never
evaluate one" (`quack/src/lib.rs:21`).

| DuckDB anchor | HEAD mechanism |
|---|---|
| population | borrowed resident lanes + `Planes.masks` |
| selection | gated `Pred` (`under` a survivor plane), ternlog, `Gather` semijoin |
| streamed pass | tiled executor; `execute_extent` — work ∝ extent, nothing copied or rebased |
| fold | Count / Any / All / MaskedSum/Min/Max / GroupSum / `GroupReduce`; fused terminals write no mask |
| no intermediate | one materialiser; `Keep` is an explicit election |

### 1.4 Scales and materialization [M unless marked]

| scale | where | logical / physical | allocated? |
|---|---|---|---|
| 8×8 | one u64 word = 8×8 block (`simd_masking_ops.rs:3940`) | physical | 1 word |
| 64×64 bits | `Palette64{rows:[u64;64]}`; shader planes `[[u64;64];8]` | physical | 512 B / 4 KB |
| 64×64 u64 cells | ndarray PR-X3 `BlockedGrid` (different object) | physical | 32 KB |
| 256×256 | p64 `sparse256` BSR 4×4 blocks; bgz17 256² tables | tables physical, matrix logical | tables only |
| 1024²/2048² | ndarray framebuffer PyramidShader | physical u8 | yes |
| 4096² | thinking-engine `TABLE_SIZE` (`engine.rs:166`) | physical u8 | **yes, 16 MB** |
| 64K×64K | helix `fire_forget_replay_probe.rs:42`, `render(addr)` | **virtual** | **never** (row-major `addr = y*AXIS+x`, doc says "Morton-ish") |
| "64k"/"256k" | morton_cascade ladder = 256²/512² **cells** | logical | no — unit differs from the helix axis |
| 64K rows | mask-risc tiles, lgj, ladybug BindSpace | physical 1-D | yes |
| 256K×256K | doc prose only | [HC] | no |
| 10K×10K | TECH_DEBT "glitch matrix" (2026-04-19, per user) | claimed | **never located in code** [U] |

Every "10K" in code is a 10,000-bit per-record vector, not a 10K² field.
p64 / p64-bridge use contiguous `/4` bucketing (row-major), not Morton;
`edge_to_block`'s only justification is "256 palette indices map to 64 blocks
of 4" — no geometric argument exists in source. 64×64 = 8×u64x8 is implicit
(scalar code LLVM autovectorizes) in p64 and stated as both lowering and
ontology in p64's module doc; mask-risc has no such structure.

### 1.5 Materialization evolution

- 2026-01-30 ladybug `942d6a5`: eagerly materialized 65,536 × 10K-bit field,
  ~80 MB [HI].
- 2026-03-22 ndarray `85f5fb4b` `hpc/holo.rs:1712` "Lithographic Gating" of
  one 2 KB container — the only executable "lithographic" code; **no callers**
  [HI, dead].
- 2026-04-03 `086047ff`: materialized 4096² table, "16M **RISC** thought
  engine" — the word RISC first sat on a materialized MatVec [HI, still HEAD].
- 2026-05-18 PR-X9 doc "virtual grid views — NEVER materialized" [HC, plan-only].
- 2026-06-14 first Morton cascade code (still allocates its leaf field).
- 2026-07-19 helix `fb10bfd9`: **first executed virtual field** [M].
- 2026-09-14+ `mask_shift_morton`, mask-risc IR, tiled execution [M].

Verdict: materialized → virtual → mask-driven is a real **trend** [SH], not
one lineage; lithography as the middle step is [CH]; Morton as the carrier
of the transition is [CH].

### 1.6 Lithography archaeology

- ladybug-rs: **zero** hits for lithography/reticle/wafer in all history; its
  "exposure" is *Belichtungsmesser*, a camera-light-meter Hamming prefilter [HC].
- ada-docs `SHARED_LITHOGRAPHY.md` (2026-01-22) is cited, not verified [U].
- In lance-graph / lgj the vocabulary first appears 2026-08-30 onward, as a
  label on already-shipped mask algebra; the lgj plan scopes it to one
  load-bearing place ("the wafer is immutable" ⇒ cached masks evict, never
  invalidate) [HC + design intent].
- **Convergence as a lineage claim is unsupported.** What holds is a
  structural analogy: select-by-mask over resident bytes on immutable
  versions — ordinary columnar practice, not inherited from the old docs.

### 1.7 DTOs [M]

| DTO | evidence-supported meaning | producer | consumer | status |
|---|---|---|---|---|
| `StreamDto` | codebook indices + timestamp | none | none | dead type |
| `PerturbationDto` (`dto.rs:66`) | **settled** normalized energy (4096 f32, full snapshot) + top-8; `converged = cycles<10`; `entropy()` = −Σe ln e | `think*()` / `commit()` **after** cycling | same crate; `dispatch_from_top_k` keeps top_k only | live, crate-internal |
| `BusDto` | argmax + top-8 | `commit()` | tests only | produced, unconsumed |
| `ThoughtStruct` | — | none | none | dead |
| perspectival `ResonanceDto` | 3-value agreement + heuristic labels | `from_superposition` | never-constructed `MomentDto` | test-only (contradicts `MODULE-TABLE.md:372`) |
| `FastBusDto` | `repr(C)` ≤24 B summary | `from_thought` | cfg(test) | test-only |
| `ShaderBus` | one completed cycle | `driver.rs` | lab `WireBus`, tests | live, no production sink |
| `CycleFrame` | storage identity only | planner | planner Lance path | live, unrelated to shader |

**PerturbationDto is OBSERVED from execution, never injected** — injection is
`perturb(&mut self, &[u16])` (`engine.rs:510`) taking raw indices. No DTO has
LE encoding. "Shannon"/"proprioception" appear in no DTO file.
`perturbation-sim` is a power-grid outage simulator, unrelated.

### 1.8 engine_bridge

Created `da88a547` (PR #205, 2026-04-18); `persist_cycle` body unchanged
since; **never had a production caller**; unusable on a live driver
(`ShaderDriver.bindspace: Arc<BindSpace>`). Its doc line "[6] emitted_edges
feed commit_to_l4" is contradicted by the types from day one (`commit_to_l4`
takes `&BusDto.top_k`). Classification: test/minimal bridge [SH] + fossil of
the singleton BindSpace [SH] + abandoned CE64-persistence intent [HC/RV via
`ce64-spofc-learning-v1`, #1294 reverted by #1296]. Not a learning membrane
as built [CH]. `acb21839` (PR #1051, 2026-08-26) records in code
(`engine_bridge.rs:113-119`) that the dense energy never reaches the mask ALU
and forbids lowering until `ISS-PERTURBATION-P64-ADDRESS-IDENTITY-UNPROVEN`
closes.

### 1.9 Cycle boundary

- **Shader cycle:** only in-RAM `awareness` + `rung_elevator` cross [M].
  CE64 is transient hot-path state — rebuilt per cycle and discarded; the
  ruled future (#477, `E-EVERYTHING-WIRES-TO-SOA-V3-CE64-IS-ALU-LEGACY-1`) is
  an owner-stamped `edges` column value, never a handoff [SH, unbuilt].
- **Mask/fold world:** immutable Lance versions carry the population;
  caller-held resident planes (lgj generation-checked `Mask` handles, `Keep`
  sinks) outlive a query; fold results return as scalars. No production mask
  cache (`mask_cache_hit_probe` is a probe).
- **Between the two worlds:** nothing crosses. Result: "both layers intended,
  neither wired" — the CE64 write-back (per-owner, per-version) and the
  perturbation/L4 model update are distinct layers and must not be collapsed.

## 2. `.claude/v3` vs HEAD

**v3 establishes:** the pipeline `thinking-engine → p64 → cognitive-shader-driver
→ SoA`; Ψ `PerturbationDto` = "MECHANICAL Morton-tile inverse-pyramid
perturbation field"; the L4 learning loop (residue → owner-stamped tenant lane
→ next cycle's template reads the row) with landing lanes `LearnedStyle` (11)
and `ExploreStyle` (12, "from the P64 perturbation ladder") — D-V3-W4b
**Queued**; `persist_cycle`/`dispatch_busdto` BLOCKED→W4a; `commit_to_l4`
BLOCKED (possible orphan write); p64-bridge stateless; the 1BRC lane F/G/R
measurements (route-and-write 3× over the classic map; Morton address ~10%
over radix; "mailbox = OWNER boundary, tile = ADDRESS boundary"); field
certification gated by D-MTS-2/3 (Queued).

**Deltas at HEAD:**

1. v3 never mentions mask-risc or Quack — the inventory predates the mask/fold
   execution anchor. Everything after the 09-14 IR skeleton (GroupReduce,
   fused Range∩plane, fused ternlog Count/Any, Tern2/Tern3, strided views,
   tiled 256-word executor, `execute_extent`, `Program::compile`, the
   aperture / cache-hit / HHTL-order probes) is unrecorded there.
2. Shader path net-unchanged: the 09-25 SPOFC series landed and was reverted
   (#1294 → #1296/#1298).
3. Contradiction: Ψ is "Morton-tile" in v3 but a flat `Vec<f32>` in code.
4. Contradiction: v3's L4 loop is recorded as a mechanism; in code
   `LearnedStyle` is written only by `planner/examples/probe_sudoku_teacher.rs:727`.
5. `engine_bridge.rs:113-119` now forbids the lowering in code — v3's
   "perturbation ladder → ExploreStyle" is gated behind an open issue.

## 3. Does a non-materialized Perturbationsfeld exist?

**Partially, as disjoint pieces; absent as a path.**

| chain link | HEAD |
|---|---|
| mask/program | exists (mask-risc) |
| non-materialized logical field | absent for mask-risc (resident operands only); the only executed virtual field is the helix probe, standalone |
| fold/resonance | exists twice, unconnected: mask-risc terminals; thinking-engine MatVec over a materialized table |
| perturbation | exists, dense and materialized, observed after cycling; only `top_k` survives |
| cross-cycle consequence | absent in production |

**Smallest missing executable seam:** perturbation → next cycle's mask
aperture. `dispatch_from_top_k` already reduces the field to a contiguous row
interval (`ColumnWindow{start,end}`); mask-risc already consumes exactly that
shape without touching a word (`Pred::Range`, `execute_extent`). Nothing hands
one to the other — and the code forbids doing so until the address identity
is proven. So the first executable step is a probe, not wiring.

## 4. The proposal — D-PFP-1, the three-arm Perturbationsfeld probe

### 4.1 Why this and nothing else

- It is the only step that DECIDES the Perturbationsfeld question; every other
  candidate (a production `ShaderSink`, `persist_cycle`, the L4 lane write)
  persists a consequence before anyone knows the field means anything.
- The arms are already registered in `ISS-PERTURBATION-P64-ADDRESS-IDENTITY-UNPROVEN`
  (control / experiment / sabotage) and have never been run.
- **mask-risc removes the hard part of that issue.** Q1 (row-major vs Morton
  vs permuted `codebook_id ↔ (row,col)`) exists only because the 4096-entry
  field was going to be read as 2-D p64 cells (`S/4 × O/4`). Lowered instead
  onto mask-risc over a **1-D population of 4096 rows where row = codebook
  id**, the identity `energy[i] ↔ row i` holds by construction. The 2-D
  question leaves the gate and becomes an optional variant (§4.5).
- Every part exists: `ThinkingEngine::perturb` / `think` → `PerturbationDto`;
  mask-risc `Pred` / `Ternlog` / `Keep` build the aperture; `Count` /
  `GroupReduce` / `MaskedSum` fold; feeding the fold's selected ids back into
  `perturb` is the cross-cycle consequence. No contract, ABI, lane or carrier
  change.

### 4.2 Arms (two cycles, n → n+1, one fixed corpus, fixed seeds)

| arm | cycle-n field → cycle-n+1 input |
|---|---|
| **Control** | today's path: `top_k` → `min..max` window (`dispatch_from_top_k`) → `Pred::Range` / `execute_extent` |
| **Experiment** | dense `energy` → order-preserving i32 key lane (§4.3) → `Pred::GtI32` threshold mask over 4096 rows → fold → selected ids → `perturb` for n+1 |
| **Sabotage** | Experiment with the `energy[i] ↔ row` addressing permuted by a fixed seed |

### 4.3 Pre-registration (D-PFP-0, written and committed BEFORE any run)

- metrics on the n+1 energy: top-k overlap (|A∩B|/k) and L1 distance;
- numeric thresholds for "measurably different" fixed in the pre-registration
  file, never adjusted after reading results;
- energy threshold for the aperture, corpus, seeds, codebook table and the
  ENGINE VARIANT (which of u8 / BF16 / i8 / f32 produced `energy`) fixed;
- **the f32 → i32 lowering, fixed and exact.** mask-risc has no f32
  predicate: `LaneRef` is `I32 | U32 | U64 | Strided` and the ordered
  predicates are `GtI32` / `LtI32` (`crates/lance-graph-mask-risc/src/ir.rs:129,131`).
  `PerturbationDto.energy` is `Vec<f32>` and `from_energy_f32` does not
  guarantee `e ≥ 0` (the signed engine exists), so the lowering is the
  standard total-order key, not a quantization:
  `k(e) = { b = e.to_bits() as i32; if b < 0 { b ^ 0x7FFF_FFFF } else { b } }`.
  It is strictly monotone over all finite f32 (no result-affecting rounding),
  so `e > θ ⇔ k(e) > k(θ)` except that `-0.0` and `+0.0` receive distinct
  keys — θ must therefore be pre-registered as a nonzero value. NaN in
  `energy` ⇒ INVALID run. The key lane is a probe-local derived copy
  (4096 × i32 = 16 KB), stated here as a materialization the probe accepts;
  it is not a production pattern and adds no new `Pred` or `LaneRef`;
- **lowering oracle:** each run also computes the aperture with a scalar f32
  comparison (`e > θ`) and asserts it equals the mask-risc mask bit-for-bit
  (§4.4 can-it-fire);
- any significance claim cites Jirak 2016 (I-NOISE-FLOOR-JIRAK), never
  classical Berry–Esseen; hand-set thresholds are labelled hand-set.

### 4.4 Outcomes — exhaustive (every run lands in exactly one row)

Validity checks run FIRST; a run that fails either is INVALID and no
outcome row is read from it:

- **Can-it-stay-silent twin (mandatory):** two unpermuted runs of the same
  arm must read identical under the metric (the metric must not fire on
  everything). Both twins use non-trivial inputs. Fails ⇒ **INVALID**
  (nondeterminism or a metric that fires on noise) — fix, re-pre-register,
  re-run; never read the arms.
- **Can-it-fire check:** the sabotage permutation must change the aperture
  mask itself, and the lowering oracle (§4.3) must agree bit-for-bit.
  Fails ⇒ **INVALID** (the disable did not apply / the lowering lies).

Then, with "≠" meaning "differs by the pre-registered thresholds" and "≈"
meaning "does not":

| E vs S | E vs C | outcome | disposition |
|---|---|---|---|
| ≠ | ≠ | **PASS** | the field is an address AND the dense aperture changes the next cycle beyond today's window. Unlocks D-PFP-2; a consequence may be proposed for D-WFL-W5 / D-V3-W4b |
| ≈ | any | **FAIL** | the field is not an address under this lowering; the lithography/Perturbationsfeld framing has no mechanical content here; stop before wiring any seam. D-PFP-2 stays deferred |
| ≠ | ≈ | **ADDRESS-WITHOUT-GAIN** | the addressing carries information, but the dense aperture buys nothing over the existing `top_k` → window path. Recorded as a finding; does NOT unlock wiring (today's path already delivers the same consequence) and does NOT unlock D-PFP-2. The only licensed follow-up is re-examining the aperture choice (§6 OPEN), as a new pre-registration |

No other combination exists; an outcome not in this table is a defect in
the harness, not a result.

- Read the true exit status and the full result lines of every run, never a
  grep of assertions.

### 4.5 Optional variant (D-PFP-2, only after D-PFP-1 PASSES)

The 2-D addressing question as three arms over the same 4096 cells:
row-major (`id>>6, id&63`) vs Morton deinterleave (12 → 6+6) vs permuted.
This is the actual Q1 of the open issue, answered by measurement rather than
archaeology. Still a probe; still no p64 wiring.

### 4.6 Placement

`thinking-engine` is `workspace.exclude`, so the probe is a **standalone
probe crate** depending on both `thinking-engine` and `lance-graph-mask-risc`,
the `d-diamond-1-probe` pattern. Workspace-excluded. No production crate
gains a dependency.

### 4.7 What a PASS buys / what a FAIL saves

- PASS: the first proven executable loop mask/program → fold → perturbation →
  cross-cycle consequence. Its consequence then has a legitimate home: the
  focus-producer slot of D-WFL-W5, and later the owner-stamped
  `LearnedStyle`/`ExploreStyle` write of D-V3-W4b.
- FAIL: the whole wiring arc, for the price of one probe crate.

## 5. Explicitly NOT authorized

- a 64K×64K / 256K×256K buffer "to make the architecture work";
- a new `LaneRef` generator variant, carrier, lane, DTO or universal
  representation;
- any p64 lowering of the perturbation field (the in-code gate stands);
- promoting Morton coordinates, RISC-mask instructions, or W-slot into CE64
  semantics; treating S/P/O ordinal adjacency as geometry;
- wiring `persist_cycle`, a production `ShaderSink`, or `commit_to_l4`;
- renaming anything "cascade" or "perturbation";
- citing lithography language as evidence.

## 6. Open points (stay open)

- OPEN: the correct metric for "the field carried information" — top-k
  overlap + L1 is a proposal, not a derivation.
- OPEN: whether a threshold aperture (vs top-k-as-set, vs quantile) is the
  honest lowering of a dense energy field into a mask.
- OPEN: 10K×10K "glitch matrix" location; ada-docs `SHARED_LITHOGRAPHY.md`
  and rustynum-holo `focus.rs` contents.
- OPEN: whether `ce64-spofc-learning-v1` returns.
- OPEN: ternlog dispatcher ownership (lgj `kernels.rs:360` duplicate).
- STALE DOCS (not corrected here): `MODULE-TABLE.md:372` (perspectival
  ResonanceDto "WIRED-HOT-PATH"); `engine_bridge.rs` module doc [6];
  v3's "Morton-tile" Ψ wording.

## 7. Convergence prompt for the parallel session

> Paste the block below into the other session unchanged.

```
You have been working on the same architecture question in parallel. A
proposal now exists at .claude/plans/perturbationsfeld-probe-v1.md
(D-PFP-*). Please CONVERGE your additional ideas into it rather than
starting a parallel plan.

Read, in order: that plan (all of it), .claude/v3/knowledge/v3-substrate-primer.md §3,
.claude/plans/waben-fold-execution-loop-v1.md (D-WFL-W3..W5 rows in
STATUS_BOARD.md), ISSUES.md ISS-PERTURBATION-P64-ADDRESS-IDENTITY-UNPROVEN,
and crates/cognitive-shader-driver/src/engine_bridge.rs:105-150.

Then, for each idea you hold that is NOT already in the plan, report ONE row:
  idea | which plan section it extends or contradicts | evidence label
  ([H]/[M]/[HI]/[HC]/[RV]/[SH]/[CH]/[U]) | file:line or commit | does it
  change D-PFP-1's arms, metrics, pass/fail, or placement? (yes/no + how)

Rules:
- Do not re-inventory producers/consumers or the SoA layout; .claude/v3 and
  §1 of the plan are the map.
- Do not make CE64 the centre; it is not the hypothesis under test.
- Do not add archaeology; an old occurrence of a word has no authority.
- If an idea contradicts a §1 finding, cite the line that falsifies it — a
  grep hit is not evidence until the file is read.
- If you believe the probe itself is the wrong next step, say so in one row
  with the specific cheaper-or-more-decisive alternative, not a redesign.
- Keep OPEN items open; UNKNOWN is a valid answer.
- No implementation, no new carrier/lane/DTO, no wiring.

Deliver the rows plus at most one paragraph of synthesis. The orchestrating
session merges accepted rows into the plan as an append-only v1 addendum.
```

---

## 9. Addendum 2026-09-29 — ratified v3 of D-PFP-0/1 (5+3 council)

> Append-only. This section SUPERSEDES §4.2–§4.6 where they differ; §1–§3
> and §5–§7 stand. Council run: 5 savants (prior-art, iron-rule, code-truth,
> cascade-impact, different-views) → draft v2 → 3 reviewers
> (overclaim-auditor, dilution-collapse-sentinel, firewall-warden) → this v3.
> Verdicts: 0 BLOCK; 7 P1 and 17 P2 fixes, all applied (ledger §9.8).

### 9.0 What the probe can and cannot decide

In a 1-D population where row index == codebook id, relabelling the injected
ids changes the engine's input, so a relabelled (sabotage) arm is EXPECTED to
differ from the experiment arm for any input-sensitive engine — expected, not
proven (a symmetric table or a shared attractor could still coincide). The
probe therefore decides only:

1. **Is the engine's output sensitive to its input on the tracked tables at
   the δ level?** (INPUT-INSENSITIVE vs not.) No probe measuring this was
   found by the prior-art search (it is an absence of a search hit, not a
   proof of absence).
2. **Given it is, does the E-set (dense-energy selection) retain the cycle-n
   top set into cycle n+1 more, less, or no differently than the C-set
   (today's `top_k` window)?** Retention is a SELF-CONSISTENCY measure, not
   fidelity or information gain. Because E is the exact active set and C is
   a hull that may include inactive rows, E ≥ C is the EXPECTED direction;
   the informative parts are the magnitude, a reversal, and the
   cardinality-matched arm E_m, which separates "more ids" from "which ids".

The sabotage arm is a SANITY control with its own outcome. The 2-D
address-identity question (ISS-PERTURBATION-P64-ADDRESS-IDENTITY-UNPROVEN)
is NOT decided here (D-PFP-2), nor is D-WFL-W5.

### 9.1 Correction to §4 (R1): N = 256, not 4096

Only 256² tables are git-tracked (`crates/thinking-engine/data/*/distance_table_256x256.u8`);
`ThinkingEngine::new` infers size from table length (`engine.rs:204-231`).
The population is therefore N = 256 rows; the key lane is 256 × i32 = 1 KiB.
Every "4096 rows / 16 KB" in §4 and in the 2026-09-29 INTEGRATION_PLANS entry
reads as N = 256. D-PFP-2 must choose its own table size.

### 9.2 Frozen decisions

F1 probe only, no production crate gains a dependency · F2 no p64 lowering
(`engine_bridge.rs:113-119`) · F3 no new `Pred`/`LaneRef`/carrier/lane/DTO ·
F4 falsifiability rule; a check implied by the code it tests is labelled
TRIPWIRE and never counted as discriminating · F5 no σ / significance claim;
thresholds hand-set, single seed, descriptive; results never say
"significant" · F6 real tables and lens indices · F7 exhaustive outcomes ·
F8 exact f32→i32 total-order key, θ nonzero, NaN ⇒ INVALID, scalar oracle ·
F9 top-set overlap primary, L1 reported only · F10 control path = `top_k`
active rows → window → `Pred::Range` → `Keep` · F11 no model identifier in
any artifact; AGENT_LOG written by the orchestrating thread only · F12
PREREG.md committed strictly before results; the results entry quotes the
PREREG branch-commit SHA AND the sha256 of PREREG.md (survives a squash
merge; local ordering is evidence, not tamper-proofing).

### 9.3 Inputs (verified at 155909a5; line numbers may drift)

thinking-engine (built with `default-features = false`): `engine.rs:204`
`new`; `:427` `think` (loops `cycle()`; its doc at :425-426 says `cycle_auto`
— doc defect, not fixed here); `:517` `perturb` (+1 per id < size,
renormalize if total > 1e-10); `:533` `reset`; energy after `think` on a u8
table is ≥ 0 and finite and CAN be all-zero; `dto.rs:89` `from_energy_f32`
always fills 8 `top_k` slots — on a sparse field the tail is padded with
zero-energy entries. `codebook_index.rs:28` `CodebookIndex::new` (public;
asserts idx < table_size). Data: Jina v5 table + index (151,936 LE u16, max
255); BGE-M3 table + index (250,002, max 255). Prior art cited, not
duplicated: `examples/chunker_falsifier.rs`.
mask-risc (public API): `Planes`, `LaneRef::I32`, `MaskOp::Pred{pred, under:
None, dst}`, `Pred::{GtI32, Range}`, `Terminal::Keep`, `Program`,
`Scratch::for_program`, `execute(.., None)` → `Value::Mask(Scratch(0))`,
`Scratch::slot(0)` (single tile at N = 256), `materialize_rows`.
cognitive-shader-driver (reproduced, not depended on):
`SCAN_WORTHY_ENERGY = 0.01`, window + empty fallback `[0, min(N,64))`.

### 9.4 Procedure

- Lenses: PRIMARY Jina v5, REPLICATION BGE-M3, via `include_bytes!` from the
  probe crate. PRIMARY decides. Disagreement ⇒ "<PRIMARY outcome>,
  LENS-SPECIFIC". REPLICATION INVALID with PRIMARY valid ⇒ "<PRIMARY
  outcome>, REPLICATION INVALID".
- Stimuli: Q = 32 stimuli, each m = 8 token ids, SplitMix64 seed
  `0x9E3779B97F4A7C15`, uniform in [0, vocab), mapped by
  `CodebookIndex::lookup_many`.
- Cycle n: `reset(); perturb(stim); P_n = think(10)`.
- Arms (θ = 0.01, hand-set = `SCAN_WORTHY_ENERGY`):
  - **C** window over `{id ∈ P_n.top_k : e > θ}` (empty ⇒ `[0, min(N,64))`)
    via mask-risc `Pred::Range` + `Keep`.
  - **C'** the same window as a scalar id list (TRIPWIRE for the C lowering).
  - **E** key lane `K[i] = k(e_n[i])` (probe-local, "probe only — not a lane
    pattern") via `Pred::GtI32{t: k(θ)}` + `Keep`.
  - **E_m** (reported only) the top-|ids_C| rows of `e_n` by energy among
    `e > 0` (ties by lower id) — cardinality-matched to C.
  - **S** `π(ids_E)`, π a fixed Fisher–Yates permutation of 0..N, SplitMix64
    seed `0x5EED_0000_0000_0001`.
- Cycle n+1 per arm X: `reset(); perturb(ids_X); P_{n+1}^X = think(10)`.
  RESET: only the id set carries across; CONTINUE is named and unrun.

### 9.5 Metrics (integer units)

- `act(P)` = ids in `P.top_k` with `e > 0` (padding removed); padding rate
  reported.
- `I(a,b) = |act(a) ∩ act(b)|` (0..8); `O = I/8` (fixed denominator).
- Retention of arm X: `R_X = Σ_s I(P_n, P_{n+1}^X)` over the stimuli valid
  for X, reported also as a mean in [0,1].
- Difference `D(X,Y) = Σ_s (8 − I(P_{n+1}^X, P_{n+1}^Y))`.
- Empirical null (anchor): `N0 = mean over stimulus pairs s≠t of
  I(P_n^s, P_n^t)/8` (the cross-stimulus overlap); 8/256 ≈ 0.031 printed
  for reference only.
- δ = 0.25 (= 2 of 8 ids per stimulus), hand-set. A mean-difference test
  `ΔR ≥ δ` is evaluated exactly as `Σ ΔI ≥ 2·Q_valid` in integers.
- L1 means reported only.

### 9.6 Validity, degeneracy, outcomes (evaluated in this order; exhaustive)

Degenerate: a stimulus with `e_n` all-zero, OR `ids_E` empty; and per arm X,
`e_{n+1}^X` all-zero excludes that stimulus from every comparison involving
X. > 25% of stimuli excluded in any required comparison ⇒ INVALID.

1. **INVALID** — any of: NaN in any energy (near-vacuous for u8 tables; kept);
   V3 lowering oracle fails (scalar `e > θ` ≠ mask-risc E mask, or scalar
   window ≠ mask-risc C mask, any stimulus); θ inertness fails for E (the E
   set must shrink at 2θ on ≥ 1 stimulus AND grow at θ/2 on ≥ 1 stimulus);
   TRIPWIRES fail (V1 determinism: each arm run twice identical; C vs C'
   `D = 0`); degeneracy ceiling exceeded.
2. **INPUT-INSENSITIVE** — non-collapse fails: (a) positive control — the two
   single-id stimuli {a},{b} of minimal table similarity give
   `I(P^a, P^b) > 8 − 2` (overlap above 1 − δ), OR (b) `1 − N0 < δ`.
   Worded "output insensitive to input at the δ level on these tables".
3. **RELABEL-INSENSITIVE** — sanity fails: over stimuli with
   `|ids_E| ≤ N/2`, `D(E,S) < 2·Q_eligible` (fewer than 2 of 8 ids differ on
   average). Not a collapse verdict. If fewer than 8 stimuli are eligible the
   sanity is reported "not evaluable" and step 4 proceeds.
4. On `ΔR = R_E − R_C` (integers, per valid stimulus):
   - **HIGHER-RETENTION** if `ΔR ≥ 2·Q_valid`,
   - **LOWER-RETENTION** if `ΔR ≤ −2·Q_valid`,
   - **NO-RETENTION-DIFFERENCE** otherwise.
   Always printed beside it: `R_S`, `R_{E_m}`, mean |ids_C|, |ids_E|,
   padding rate, N0. Reading aid (not a verdict): `R_{E_m} ≈ R_C` suggests a
   count effect; `R_{E_m} > R_C` suggests membership matters.
   θ inertness for C is REPORTED ("θ is decoration in the control path" if
   inert), never INVALID — it reproduces production.

Mapping to §4.4 (155909a5): PASS → HIGHER-RETENTION; FAIL →
INPUT-INSENSITIVE; ADDRESS-WITHOUT-GAIN → NO-RETENTION-DIFFERENCE; new:
RELABEL-INSENSITIVE, LOWER-RETENTION. Discriminating checks are the
non-collapse pair, the sanity, the oracle and θ inertness; V1 and C' are
TRIPWIRES.

Harness-bug clause: if a run is INVALID because of a probe defect (not the
data), the defect is fixed, constants stay unchanged, and BOTH runs are
quoted in the result entry.

### 9.7 Scope statements (pre-registered wording)

- INPUT-INSENSITIVE is scoped to the tracked 256² Jina v5 / BGE-M3 tables
  with the p75 floor and 10 cycles under RESET. It would cast DOUBT (not a
  verdict) on thinking-engine "unwired gems" that assume input-dependent
  energy (`.claude/v3/FUTURE-DESIGN.md`).
- No outcome decides D-WFL-W5, CONTINUE, or the 2-D address identity.
- No outcome is attributed to "address" alone; E_m is the only attribution
  aid, and it is reported, not decisive.

### 9.8 Gates

G1 (local, not CI-gated — CI builds only explicit manifest paths):
probe `cargo build --release`, `cargo test`, `cargo clippy --all-targets --
-D warnings` green. G2 root `cargo metadata --no-deps` succeeds after the
`exclude` edit. G3 exactly one outcome per lens from 9.6. G4 PREREG commit
precedes the results commit; entry quotes SHA + sha256. G5 board: commit 1
(PREREG + crate + exclude + this addendum + INTEGRATION_PLANS correction line
+ STATUS rows) and commit 2 (entry + `entries_index.py --write` + STATUS flip
+ AGENT_LOG) each regenerate SUPERSESSION-INDEX LAST; every ledger prepend
uses read-then-write (never open-for-write while reading) and a `wc -l`
post-check. G6 constants never change after the first run.

### 9.9 Change ledger v1 → v2 → v3

v1→v2 (savants): sabotage demoted from PASS leg (S5-Q1, S2-Q1 ×2) · δ
anchoring + descriptive wording (S2-Q2) · broader collapse check (S2-Q1,
S5-Q5) · V1 relabelled tripwire (S2-Q1) · θ inertness reported for C (S2-Q1)
· all-zero energy degenerate (S3-Q2) · public `CodebookIndex` (S3-Q1) ·
explicit `under: None` / lanes (S3-Q3) · RESET scope (S5-Q3) · LENS-SPECIFIC
(S5-Q4) · COLLAPSED scope (S5-Q5) · PREREG constant test + SHA (S2-Q5) · mod
registration (S2-Q5) · N=256 correction (S4-Q4) · local-only G1 (S4-Q2) ·
prior art cited (S1) · probe-local helpers (S1-Q2..Q4).

v2→v3 (reviewers; stricter verdict won everywhere):
- P1 `top_k` padding makes 0.03 a wrong null → `act()` drops zero-energy
  padding; empirical cross-stimulus null N0 (overclaim R7).
- P1 "GAIN / re-evokes better" overclaims; E ≥ C expected → retention
  relabelled HIGHER/LOWER/NO-RETENTION-DIFFERENCE, self-consistency wording,
  cardinality-matched E_m reported (overclaim R9; dilution R5).
- P1 NC(a) `O < 1` near-vacuous → threshold `> 8 − 2` (overclaim R10).
- P1 NC(c) conflated relabel-sensitivity with collapse, false on large E →
  separate RELABEL-INSENSITIVE outcome, eligibility `|ids_E| ≤ N/2`
  (dilution R9/R10; §6 wording follows).
- P1 per-arm all-zero `e_{n+1}` → per-arm pair exclusion counted in the
  ceiling (dilution R8).
- P2: "by construction" → "expected, not proven"; absence = "not found by
  search" (overclaim §0) · "aperture" → E-set/C-set (dilution §0) · S
  notation → Q stimuli (both R3) · R_S reported, integer comparisons
  (dilution R7) · V2(C') relabelled TRIPWIRE (all three) · squash-safe
  SHA + sha256 (overclaim §5, firewall §1) · harness-bug clause (dilution
  §5) · INTEGRATION_PLANS correction moved into commit 1, supersession LAST
  per commit, `wc -l`, AGENT_LOG orchestrator-only (firewall R12/§5/§8) ·
  L1 rejection narrowed: "same-size relabel of E" rejected; the
  independent different-stimulus baseline is adopted as N0 (overclaim §8,
  dilution §8).
