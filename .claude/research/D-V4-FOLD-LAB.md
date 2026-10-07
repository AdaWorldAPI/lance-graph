# D-V4-FOLD-LAB — OGAR-R2IL as V4 ISA, folding as execution law

> **READ BY:** anyone proposing an R2IL opcode, a fold/planner IR layer, a
> JITSON/CubeCL/MLIR backend, or an adaptive carrier. Also answers the
> D-V4-ISA-LAB brief (same questions, narrower scope).
> **Status:** source-backed research, 2026-10-07. Production code was read
> first; plans/docs are cited only as intent. Every verdict below names its
> evidence grade: **[G]** measured or read in executing code, **[H]** read in
> declared-but-unexecuted code or bounded by literature, **[S]** reasoned.
> **No Rust was changed for this report.** Literature sources and full
> citations: research notes behind this file (scratchpad, not committed);
> the load-bearing citations are inlined.

## 1. Current architecture map (as it actually is)

```
SOURCE                Cypher text            report def         machine AST (r2sleigh, out of tree)
  │                       │                      │                        │
SEMANTIC              lance-graph parser +   lance-graph-report    —
                      SemanticAnalyzer +     ReportPlan
                      LogicalPlanner
  │                       │                      │
DEMAND                cypher-quack           (fixed by MeasureKind)
                      demand::classify  [experimental]
  │                       │                      │
"BIND"                cypher-quack Binding   report boundary Catalog / quack bind::Draft (test-only Binder)
  │                       │                      │
QUERY                 quack::Query {filter, agg}  ◀── both reach here
  │
LOWER                 quack::lower | lower_fused | lower_group_by
  │
MICRO-OPS             mask-risc Program { ops: Vec<MaskOp>, terminal }
  │
EXECUTE               mask-risc execute_into / execute_extent over borrowed lanes (ndarray::simd)

OGAR-R2IL             0x90..=0xE1 machine band + 0xE2..=0xED fold band, one table under two
                      PROVISIONAL concept ids (0xC400 MACHINE / 0xC401 FOLD)
                      └─ only executable reader: FoldDialect in r2il-mask-abi-probe/tests/row_bridge.rs
                         (crate EXCLUDED from the workspace) → lowers to mask-risc → bit-identical to quack::lower
```

**The decisive fact:** today nothing in a workspace member reaches R2IL. Quack →
mask-risc is the live path; R2IL is a parallel spelling of mask-risc that one
test file reads. [G: `lance-graph/Cargo.toml:101-111` excludes both
`r2il-mask-abi-probe` and `lance-graph-ogar`; neither quack's nor mask-risc's
`Cargo.toml` depends on ogar/loco/r2il.]

## 2. R2IL ISA assessment — **MIXED, leaning executable ISA in shape, physical IR in content**

| Property | Verdict | Evidence |
|---|---|---|
| Ops executable, not domain-named | ISA ✓ | machine + fold mnemonics only; no business names [G `ogar-r2il/src/lib.rs` MNEMONICS] |
| Meaning in operands / addresses | ISA ✓ | LOAD space immediate selects lane kind; GROUP_SUM terminal chosen by key address [G row_bridge.rs] |
| Same vocabulary over unrelated domains | MIXED | three frontend shapes run; only the quack-shaped one is oracle-checked [G] |
| Multiple readers implement one op | ✗ WEAK | one in-tree reader (fold, test-only); it never consults the registry or the classid [G] |
| Storage layout absent from op names | mostly ✓ | `SUM` not `SumI32`, but the reader only accepts i32 lanes [G] |
| Backend does not leak upward | ✗ VIOLATED | fold band mirrors mask-risc terminals 1:1; unsigned compares refused *because mask-risc lacks them*; LOAD spaces 0–3 copy `LaneRef` + `Planes.masks` [G lib.rs ~352-418] |
| New business concept needs no opcode | ✓ | IAM lift (§3) needs none [G-by-construction, executable subset] |

Violations worth naming (not all are defects; each is a boundary leak):
1. **Backend naming the ISA** — the fold band is mask-risc's terminal list.
2. **Backend gap as vocabulary rule** — no unsigned ordered compare.
3. **One byte, two semantics** — PopCount/IntEqual/IntAnd are scalar under MACHINE
   and population ops under FOLD; the classid that should disambiguate is
   ignored by the only reader.
4. **VIA's meaning is unstable** — docs say "indexed gather", the reader uses it
   as a join-address constructor; mask-risc `Gather` (semijoin) is unreachable.
5. **Immediates are u8** — literals ≥ 256 cannot be written; CONSTANT is unwired.
6. **Both concept ids PROVISIONAL**, not minted in `ogar-vocab`.

## 3. V3 → V4 lift table

| Domain | Lifts with existing ops? | Notes |
|---|---|---|
| IAM: active ∧ dept=Stuttgart ∧ location≠Berlin ∧ ¬E5 → candidates | **YES** | `LOAD/EQ` · `VIA/EQ` · `IntAnd` · `IntEqual+IntNot` (≠; not NULL-safe) · `LOAD plane/IntNot/IntAnd` · `KEEP`. Limits: ValueIds < 256 (u8 imm); if E5 is a (user,sku) relation not a resident plane, "¬E5" is a produced population → two programs combined by the consumer (existing `ImplyGroup` pattern, `lance-graph-dir-sim/src/rule.rs:95-111`). No IAM opcode. [G executable subset, H for SCATTER_OR] |
| Cypher Q0–Q3, Q5, Q6, Q9 | **YES via Quack** | Quack→mask-risc covers them (cypher-quack differential, 3-way agreement). R2IL path: covered only where the 5 executed fold ops reach. [G] |
| Cypher Q4, Q7, Q8, QV, Q10 | **NO** — gaps are in Quack, not R2IL | foreign-value sum, ordered compare through fk, computed-frontier hop chain [G refusals] |
| Machine semantics | shared TABLE, separate READER | §4 |
| Report/pivot | partial | `ReportPlan` reuses quack, but **re-lowers per pass and per fold state** (`lance-graph-report/src/exec.rs:554,619-628,755`); literals baked in `Pred`; no parameter slot. The "compile once, only period/tenant dynamic" property does NOT hold today. [G] |

## 4. Quack semantic boundary (where meaning must stop)

Quack/planner must still own, and R2IL must never be asked to reconstruct:
bag vs set · WALK vs TRAIL · variable rebinding · earlier-variable identity ·
`count(*)` vs `count(DISTINCT x)` · fanout multiplicity · group ownership.
The #1305 witness is the falsifier: on the DAG, 2-hop `count(*)` = 4 paths,
terminal support = 3. A mask can carry the 3 and not the 4. cypher-quack
decides this **before** lowering (`demand::classify`) and refuses the 2-hop
count rather than emitting a support program [G
`two_hop_counts_are_refused_never_answered_with_support`]. Earliest safe
boundary: **semantic analysis → demand → carrier selection → lowering**. R2IL
(or quack::Query, its live equivalent) receives an already-chosen carrier.

## 5. R2IL/Fold coverage of mask-risc (41 variants, `mask-risc/src/ir.rs:127-470`)

| group | EXACT | COMPOSITION | MISMATCH | MISSING |
|---|---|---|---|---|
| Pred (16) | 7 | 3 | 0 | 6 |
| MaskOp (8) | 4 | 3 | 1 (Gather/Semijoin) | 0 |
| Terminal (17) | 10 | 1 | 1 (ScatterOr: no `out_rows` operand) | 5 |
| **total** | **21 (51 %)** | **7 (17 %)** | **2 (5 %)** | **11 (27 %)** |

Fold band status: **executed (test-only)** VIA, RANGE, SUM, GROUP_SUM, KEEP ·
**declaration only** (refused `Unimplemented`) MIN, MAX, KEY_RUNS, ANY, ALL,
SCATTER_OR, BLEND · **not minted** FIRST, MATCH, SCATTER_COUNT (correctly held
by their falsifier discipline). [G row_bridge.rs]

## 6. JITSON archaeology — folding found, but not in the JIT

- **Config fold (partial evaluation).** What the Cranelift path actually bakes:
  `threshold` (compare vs constant), `record_size` (multiply by constant),
  `top_k` (early-exit bound), and a direct call to a link-time-bound distance
  kernel; the noise kernel unrolls octaves into f64 constants. **Claimed but
  never emitted:** `prefetch_ahead`, `focus_mask` (hashed into the cache key
  only), `CpuCaps` (detected, never read; ISA from `isa::lookup` with
  `opt_level=speed` hardcoded). The 2048-byte distance call does all the work
  and stays opaque with a runtime `len`. **Verdict: it trims ~8 loop/dispatch
  ops per record and reduces no dynamic work** — no bytes skipped, no candidate
  pruned. No benches exist. JIT and AOT also disagree (`<` vs `<=` threshold;
  first-k vs best-k). [G `ndarray/src/hpc/jitson_cranelift/scan_jit.rs`,
  `engine.rs`]. F10 holds: JITSON is not structurally simplifying execution.
- **Survivor fold.** The real staged fold lives in `hpc/packed.rs` +
  `hpc/cascade.rs`: stroke-major data, S1 sequential scan → compacted
  `Vec<(idx, partial)>` → S2 over survivors only → S3 → top-k, carrying the
  partial distance forward. Carrier is a **materialised index Vec** (exactly
  what Quack's R1 ruling forbids); thresholds are fixed or σ-calibrated (Welford
  drift detection), selectivity is assumed, never fed back. Three different S1
  rules coexist. **Reusable as a law, not as code.** [G]
- **mask-risc already has the mask-native form of survivor gating:**
  `MaskOp::Pred { under: Some(gate) }` evaluates a predicate only in 64-row words
  where the gate has survivors (`pack_under` skips dead words before loading);
  `quack::lower` chains each conjunct on the running accumulator. `lower_fused`
  cannot skip (ternlog has no `under`). Ordering is worth up to 99.61 pp of
  skipped words on a clustered conjunction under `lower`, zero under
  `lower_fused` (quack lib.rs docs, measured). **So the survivor law is already
  implemented at word granularity, without materialising survivors.** [G]
- **Collapse fold.** The SD gate (Flow < 0.15 ≤ Hold ≤ 0.35 < Block) is the
  **SAME MECHANISM** as the contract's `gate_state`/`Tactic::gate` recipe
  eligibility (`lance-graph-contract/src/recipe_kernels.rs`) — a real ancestor.
  `RecipeIR`/`PhilosopherIR`/`CollapseParams` are **types only**, no evaluator:
  STRUCTURAL RHYME. Survivor bands → collapse gate: UNRELATED. One boundary
  inconsistency: planner `physical/collapse.rs` treats SD = 0.35 as Block, the
  other two as Hold. [G]

Harvest candidates: the kernel-cache freeze (compile under `&mut`, freeze into
`Arc`, lock-free reads); `template_hash` as a stable content key **after**
adding length separators and the missing `backend_key`; the `extern "C"`
trampoline table as the minimal FFI seam. Not worth harvesting: `CpuCaps`
(duplicates `simd_caps()`), `PrecompileQueue`.

## 7–12. Optimizer survey (literature; physical choices only)

None of the surveyed mechanisms needs a new semantic op [H]:

- **Adaptive carrier** (GraphBLAS hypersparse/sparse/bitmap/full; Roaring
  array/bitmap/run). GraphBLAS `bitmap_switch` is a table by the smaller
  dimension (0.04 … 0.40) [G, read in `GB_Global.c`]; Roaring switches at 4096
  values per 2¹⁶ chunk (8 KiB either way) [G]. Maps to: per 64K-row window,
  choose dense words / sorted u16 list / runs, with hysteresis. **Conflicts
  with quack R1 (no selection vector)** — so this is a measured question, not a
  free win (§16, probe).
- **Direction** (Beamer SC'12 α=14 β=24; Ligra |U|+Σdeg ≤ |E|/20; GraphIt
  schedule language). Maps to: one VIA, scheduler picks gather vs scatter vs
  reverse-membership. GraphIt is the closest template: algorithm written once,
  schedule separate and result-preserving.
- **Late materialisation** (DuckDB 1.3: 3–10× on LIMIT/Top-N, vendor-reported;
  SLM VLDB'25: +14.7 % vs early, +8.9 % vs late on JOB). mask-risc's gated `Pred`
  is already the word-level form; the lane read happens only for live words.
- **Gunrock invalid markers** — legal only for idempotent folds (any/or/min);
  wrong for count/sum. A legality rule, not a feature.
- **CSR5 / SELL-C-σ / merge-path** — kernel choices for segmented reduce under
  degree skew; apply only where multiplicity/contributions must survive.
- **Yannakakis / predicate transfer** — a rewrite ORDER over the semijoin we
  already have; Bekkers et al. (arXiv 2411.04042) fuse the passes.
- **WCOJ / LFTJ / Free Join** — only for cyclic patterns; acyclic hop chains
  are served by semijoin masks. Would add a physical multi-way intersect, still
  not a semantic op.
- **Factorized (Kùzu CIDR'23)** — a carrier + fold rule:
  count(A×B) = count(A)·count(B); legal for count/sum/min/max/group, **not** for
  count-distinct unless the key sits in one factor. This is exactly the missing
  "foreign-value sum" gap (§3): 2-hop `count(*)` = Σ_b in(b)·out(b) — the probe's
  `TwoFoldsAndDot` already computes it from two GroupReduce folds.
- **BitWeaving / ByteSlice** — lane encoding at load + predicate kernels writing
  the mask directly; backend layer.
- **Morsel / vector-at-a-time** — needs mergeable partials; count-distinct over
  a key-ordered lane needs a run-boundary fix-up across morsels.

## 13. CubeCL target assessment [H, README/crate graph checked]

Backends CUDA/HIP/Metal/Vulkan/WGSL/CPU; CPU path = "plane size 1, cubes
sequential"; `cubecl-cpu` depends on `cubecl-llvm` + `pliron` (no C++ MLIR);
`#[cube]` builds IR when called; comptime specialises plane/cube dim and
vector width; autotune cached. A tiny adapter (Fold program → per-shape
comptime kernel, or one interpreter kernel) fits the model **as a target**.
Breakage to plan for: WGSL has no u64 (u32 mask words); popcount/atomics vary;
float group-sum order breaks bit-exact parity (integers fine); the semijoin's
foreign mask must be device-resident; **CubeCL owns its buffers — a borrowed
zero-copy API is unverified, which conflicts with "lance-graph owns the only
canonical copy"**; LLVM is heavy (feature-gated only). Verdict: CubeCL can be a
pure target **if** the copy-to-device is accepted as an explicit, named
materialisation at the backend membrane.

## 14. MLIR control comparison

| MLIR concept | Here |
|---|---|
| dialects | ALREADY COVERED — classid → vocabulary table (one table, two readers) |
| partial conversion | USEFUL IDEA — refusal-typed lowering is the same "only illegal ops lower" shape |
| pattern rewriting | ALREADY COVERED locally (quack fuser → ternlog); not needed as a framework |
| SSA | ALREADY COVERED — numbered scratch slots in mask-risc |
| canonicalization | ACTUALLY MISSING — no normal form before lowering (e.g. GtI32 vs swapped LtI32 both reachable) |
| vector dialect | UNNECESSARY — `ndarray::simd` is the vector layer |
| memref | ALREADY COVERED — borrowed `LaneRef` + `Planes` |
| gpu dialect | UNNECESSARY unless CubeCL lands |
| LLVM dialect | UNNECESSARY — no LLVM in the architecture |

Prior MEASURED witness for the same-bytes-lens law (Law C):
`board/entries/2026-10-06-v3-v4-dual-reading-round3.md` — one resident
`[u8; 2 × 512]` read as V3 facets and as V4 calls at the same addresses, zero
heap bytes, an owner write seen by both readings at once.

"Same program + reading + KnownSet" replaces dialect conversion where the shape
is invariant (MACHINE/FOLD share the table). It fails where the reader needs a
different *value type* on the stack (scalar vs population) — that is real
semantics, and today the classid that should carry it is ignored.

## 15. Cost model

`stage_cost ≈ population × bytes_per_candidate × op_cost`,
`next_population = population × selectivity`, plus conversion and setup terms.
In mask-risc the "population" of a gated stage is **live 64-row words**, not
rows — the model must be stated in words, or it mispredicts clustered
survivors (one live row keeps a whole word alive). Not yet validated against
measurement; the carrier probe is the first data point.

## 16. Experiment designs (ranked)

1. **D-FOLD-CARRIER** (highest value — tests quack R1 directly): one filter +
   count over 1M rows, density ladder 100 %…0.01 %, uniform vs clustered;
   carriers = gated dense mask (`lower`), fused dense (`lower_fused`), sorted u32
   ordinals (probe-local). Same answer asserted; measure ns, bytes touched,
   allocation.
2. **D-BIND-BUNDLE probe** — the same `Query` through ≥ 2 physical programs (see
   `D-BIND-BUNDLE-0.md`); the in-tree `lower`/`lower_fused` pair is already a
   two-bundle witness for dense carriers.
3. **D-FOLD-FACTOR** — 2-hop `count(*)` via Σ in·out vs DF join vs refusal;
   `count(DISTINCT c)` must stay refused or go through a non-factorized carrier.
4. D-FOLD-DIRECTION, D-FOLD-LATE — after 1.
5. JITSON native, CubeCL — only after a fold shows dynamic work worth
   specialising.

## 17. Falsifier results

| | Result |
|---|---|
| F1 domain opcode explosion | **SURVIVES** — IAM, Cypher (covered part), report need none |
| F2 semantic reconstruction | **SURVIVES** — demand decided before lowering; 2-hop count refused |
| F3 mandatory materialisation | **SURVIVES** for the covered set; the forbidden Keep→Semijoin shape is a refusal, not a materialisation |
| F4 backend leaks upward | **FAILS (partially)** — the R2IL fold band encodes mask-risc's terminals, its unsigned-compare gap and its LaneRef spaces |
| F5 Fold is only renaming | **CURRENTLY TRUE** — the fold band adds no optimisation, no second backend, no partial evaluation today; it is a second spelling of mask-risc read by one test |
| F6 adaptive carriers lose | **SURVIVES for this workload** — a sorted-ordinal carrier beats ungated masks below ~25 % density, but a mask-native gated gather beats the ordinal carrier at every density below 100 % (`D-BIND-BUNDLE-0.md`, probe result). The gap is `Gather` lacking `under`, not the carrier |
| F7 direction switching loses | OPEN |
| F8 late materialisation loses | SURVIVES at word level (gated Pred) |
| F9 factorization can't preserve answers | SURVIVES for count/sum; count-distinct excluded by legality |
| F10 JITSON reduces no dynamic work | **TRUE** — keep JITSON as optional codegen/cache machinery |
| F11 CubeCL needs semantic duplication | not forced; buffer ownership is the real cost |

## 18. Recommendation

| Question | Verdict |
|---|---|
| Is OGAR-R2IL plausibly the V4 ISA? | **PARTIAL** — right shape, physical content, one test reader |
| Can Fold remain a reading rather than new IR? | **YES** (one table, two readers already declared) — but the reader must start honouring the classid |
| Can Quack lower to it after demand analysis? | **PARTIAL** — 68 % exact+composition; Quack→R2IL does not exist; and it would add a layer with no consumer |
| Can carrier representation stay below V4? | **YES** in principle (no surveyed mechanism needs an op) |
| Adaptive Dense/Sparse/Run worth it? | **NO (measured, one workload)** — gated dense dominates sorted ordinals once the gather is gated; R1 stands |
| Push/pull worth it? | OPEN (no multi-hop executor to switch yet) |
| Selective late materialisation? | **YES, already** (gated Pred, word level) |
| CSR5/SELL solves a measured fanout problem? | **NO** measured problem yet |
| Factorized multiplicity needed? | **YES** for 2-hop count(*) — as a fold rule, not a carrier type |
| JITSON structurally simplifies? | **NO** |
| CubeCL a pure target? | **YES**, with buffer copy as a named membrane cost |
| Another persistent IR required? | **NO** |

- **KEEP:** Quack `Query` → mask-risc `Program` as the live V4-equivalent;
  demand-before-lowering; word-gated survivor execution; typed refusals.
- **BUILD:** `MaskOp::Gather { under }` — the survivor law for the one op that
  lacks it (measured: 5.2 ms → ~0.2–1.1 ms at ≤ 10 % density, emulated). It is an
  executor extension behind an existing op, not a new opcode. (D-FOLD-CARRIER
  ran and did not overturn R1.)
- **KILL:** elevating JITSON into the architecture (F10), and adding a Quack→R2IL
  lowering *now* — it would be a second spelling with no second backend
  (F5). R2IL earns its place the day a second reader (CubeCL, or a real MACHINE
  interpreter honouring the classid) executes the same bytes.
