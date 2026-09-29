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
| **Experiment** | dense `energy` → threshold mask over 4096 rows (mask-risc `Pred` on an energy lane) → fold → selected ids → `perturb` for n+1 |
| **Sabotage** | Experiment with the `energy[i] ↔ row` addressing permuted by a fixed seed |

### 4.3 Pre-registration (D-PFP-0, written and committed BEFORE any run)

- metrics on the n+1 energy: top-k overlap (|A∩B|/k) and L1 distance;
- numeric thresholds for "measurably different" fixed in the pre-registration
  file, never adjusted after reading results;
- energy threshold for the aperture, corpus, seeds, codebook table fixed;
- any significance claim cites Jirak 2016 (I-NOISE-FLOOR-JIRAK), never
  classical Berry–Esseen; hand-set thresholds are labelled hand-set.

### 4.4 Pass / fail

- **PASS:** Experiment ≠ Sabotage on the n+1 field **and** Experiment ≠
  Control, by the pre-registered thresholds.
- **FAIL:** Sabotage indistinguishable from Experiment ⇒ the field is not an
  address; the lithography/Perturbationsfeld framing has no mechanical content
  here; stop before wiring any seam.
- **Can-it-stay-silent twin (mandatory):** two unpermuted runs of the same
  arm must read identical (the metric must not fire on everything). Both
  twins use non-trivial inputs.
- **Can-it-fire check:** the sabotage permutation must be shown to change the
  aperture mask itself (else the disable did not apply).
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
