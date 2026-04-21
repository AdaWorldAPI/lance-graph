# Coding Practices — Patterns for the Thinking Engine Stack

**Source:** Extracted from EmbedAnything (StarlightSearch, Apache 2.0, 1.1K stars),
validated against our architecture constraints. These are patterns, not dependencies.

**Audience:** Claude Code agents working on lance-graph / ndarray / ada-rs.

---

## Checklist for New Modules

Before merging a new module, verify:

```
[ ] Does it auto-detect model type, or hardcode model names?
[ ] Does commit() use the sink pattern, or return-and-pray?
[ ] Is there a builder, or does the caller assemble raw structs?
[ ] Are heavy deps behind feature gates?
[ ] Does it work with BOTH u8 and i8 tables (or is it type-generic)?
[ ] Are per-role scale factors preserved, or does it assume uniform range?
[ ] Is the boundary between calibration and runtime clean?
[ ] Does it avoid forward passes at runtime? (codebook lookup only)
```

## Anti-Patterns (what NOT to copy)

```
1. 48KB lib.rs → Keep lib.rs as module declarations only.
2. Clone-heavy structs → BusDto references codebook indices, not cloned content.
3. Python-first API design → Rust-first. Python binding later.
4. Forward pass at every query → Codebook lookup. Never go back.
5. f32 everywhere → Precision-aware types (BF16 calibration, i8 runtime).
```

## Implemented Patterns

```
Pattern                  Module                  Status
───────                  ──────                  ──────
Auto-detect config.json  auto_detect.rs          6 tests, routes by architecture
Builder (fluent API)     builder.rs              7 tests, Lens/TableType/Pooling/Sinks
Pooling strategies       pooling.rs              6 tests, ArgMax/Mean/TopK/Weighted
Commit sinks (adapter)   builder.rs on_commit()  callback-based, multiple sinks
Tensor bridge            tensor_bridge.rs        7 tests, F32/I8/U8/Tensor enum
Semantic chunking        semantic_chunker.rs     4 tests, convergence-jump = boundary

Feature gates:
  default      = runtime only (codebook lookup + MatVec)
  tokenizer    = HuggingFace tokenizers 0.22
  calibration  = candle 0.9 + hf-hub (forward pass + training)
```

---

## SoA + Object-Does-The-Work Patterns (this workspace, 2026-04-20 shipped)

Complement the EmbedAnything patterns above. Extracted from the codec-sweep
lab-infra work (PRs #225–#239); validated against `cognitive-shader-architecture.md`
7-layer stack + `ripple-dto-contracts.md` + the lab-vs-canonical surface invariants.

### Checklist for new DTOs / kernels / caches

```
[ ] Is construction sealed behind a Builder that validates?
    (raw struct literals bypass the checks — lock the fields pub(crate))
[ ] Does the object carry a stable signature()/kernel_hash for cache keying,
    and does that signature EXCLUDE runtime drift (seeds, timestamps, RNG state)?
[ ] Are typed errors returned instead of strings / Option::None?
    (precision-ladder, overfit guards, dim mismatches all get their own variant)
[ ] Is the cache generic-over-handle-type (Cache<H>) so stub tests + real
    kernels share the same cache semantics?
[ ] Is there a stub flag in result DTOs for Phase-N-before-Phase-N+k?
    (machine-checkable anti-#219: assert!(!result.stub) trips pipelines
     that confuse stub output for real measurements)
[ ] Is the feature matrix tested?
    (default / serve / grpc / lab must all `cargo check` + `cargo clippy`;
     `--features serve` alone has hidden gaps like grpc-only dep missing)
[ ] Is serialisation at edges only? (Rule F — decode at REST ingress,
    encode at response/Lance egress; zero serde between stages)
[ ] Are DoS ceilings enforced AT CONSTRUCTION (method check) not AT
    ENUMERATION (after work started)?
```

### Anti-patterns (surfaced by session corrections)

```
6.  Stateless-shader vs stateful-engine MISFRAMED as competing
    → they're views of one SoA. p64 topology, bgz17 palette distance,
      4096² scan kernels, 16K fingerprints all live as BindSpace
      columns; the shader DRIVES through them. Don't build a new
      crate for a "higher layer" — extend by column.

7.  Hallucinating ndarray's surface from docs comments
    → jitson_cranelift::JitEngine with 2-phase BUILD/RUN Arc-freeze
      was ALREADY shipped. Don't invent a worse RwLock scaffold.
      Probe `/home/user/ndarray/src/hpc/jitson_cranelift/` FIRST.

8.  Feature-matrix blindness
    → `cargo test --features serve` is NOT the full check. A DTO
      added to `wire.rs` fails grpc.rs:LINE with E0063 when the
      grpc feature compiles. Run default / serve / grpc / lab
      before declaring a DTO change complete.

9.  Epiphany-dumping orientation-as-discovery
    → EPIPHANIES.md is for NEW findings from code/measurement.
      Architecture explained by the user is ORIENTATION, not
      epiphany. Tag ORIENTATION entries separately so future
      sessions can distinguish discovery from onboarding.

10. Raw struct literals bypassing builders
    → `WireCodecParams { subspaces: 6, .. }` lets callers skip
      precision-ladder checks. Lock fields pub(crate), make
      builder the only path. Inherent optimization — the invariant
      rides inside the type.
```

### Patterns shipped (reference implementations in this repo)

```
Pattern                           File                                               Tests   Notes
───────                           ──────────                                         ─────   ─────
SoA columns (read-only, Arc)      contract::cognitive_shader + driver::bindspace      16    FingerprintColumns / QualiaColumn / MetaColumn (u32 packed: thinking 6b + awareness 4b + nars_f 8b + nars_c 8b + free_e 6b) / EdgeColumn / temporal / expert
Builder + typed errors            contract::cam::CodecParamsBuilder                   14    .build() runs precision-ladder + overfit guard BEFORE any JIT compile
Stable signature cache key        contract::cam::CodecParams::kernel_signature()       3    Excludes seed; JIT cache stays hot across seed variations inherently
Generic-over-H cache              cognitive_shader_driver::codec_kernel_cache          9    CodecKernelCache<H: Clone>; hosts StubKernel for tests, real kernels post-D1.1b
Stub flag anti-#219               wire::WireTokenAgreementResult::stub                 3    `stub: bool` + `backend: "stub"` default; `!stub` assertions trip silent-fail pipelines
Three-DTO pipeline lifecycle      StreamDto → ResonanceDto → BusDto → ThoughtStruct   (n/a)  Per ripple-dto-contracts.md — same data, four maturity stages; serialise at edges only (Rule F)
Object-safe trait + Box<dyn>      rotation_kernel::RotationKernel, DecodeKernel       24    Send + Sync + Debug bounds enable `Box<dyn T>` in ResidualComposer for recursive composition
Spec-drift guard test             wire::sweep_request_yaml_shape_deserializes         1     Inline JSON fixture mirrors checked-in YAML; breaks if YAMLs drift from DTOs
DoS ceiling at construction       WireSweepGrid::cardinality() pre-check              1     Early-return on > MAX before enumerate(); not a try/catch after work started
Feature matrix _lab-dtos          cognitive-shader-driver/Cargo.toml                  —     _lab-dtos = [serde + serde_json + base64 + bytemuck]; serve + grpc both pull it
Substrate proof-in-code           crates/jc/examples/prove_it.rs                      6     3/5 pillars pass in ~5s: E-SUBSTRATE-1 bundle assoc, φ-Weyl discrepancy, Jirak Berry-Esseen
```

### Principles (why the patterns cluster)

```
1.  The object does the work.
    Validation rides inside the type — Builder::build() checks precision
    ladder, Rotation::new() checks pow2, CodecParamsError has one variant
    per impossible combination. Downstream code doesn't re-validate.

2.  SoA over AoS, always.
    One allocation per COLUMN TYPE; pack meta into u32s for cache-line
    reads. 16D qualia as `Box<[f32]>` length N×18, not `Vec<[f32;18]>`.
    Zero-copy SIMD views via `&[u64]` cast as U8x64 / U64x8.

3.  Same substrate, different view.
    AGI(topic,angle,style,edge,qualia,temporal,expert,cycle) ≡
    Test(input,codec,invariant,meta,icc,session,operator,sweep_id) ≡
    Think(style,angle,nars,entropy,gestalt,qualia).
    Three questions asked of the same SoA; foreground different columns
    per view. Don't build a new data structure for a new question.

4.  Stream/Resonance/Bus DTO is the data lifecycle.
    StreamDto = ingress (preserves order + ambiguity).
    ResonanceDto = superposition (preserves contradiction).
    BusDto = explicit compile into p64 + CognitiveShader.
    ThoughtStruct = durable revisable object on blackboard.
    Serialise ONLY at Stream ingress + BusDto egress/persistence.

5.  Weights are seeds.
    GGUF → palette + Fingerprint<256> + holographic residual + CausalEdge64.
    Inference = Hamming cascade + table lookup. No matmul. No FP inner
    loop. `Weights are parameters to run matmul on` is the wrong mental
    model — BindSpace columns are the model.

6.  Scaffold-before-codegen.
    Ship the cache/trait/composition layer with Stub implementations
    FIRST. Test cache semantics + composition in microseconds without
    the real JIT engine. Replace Stubs with real impls behind the same
    trait when the heavy work lands.

7.  Feature matrix is part of the contract.
    A DTO added under `#[cfg(feature = "serve")]` that's referenced by
    `grpc.rs` breaks `--features grpc` alone. Always run:
      cargo check                      # default
      cargo check --features serve
      cargo check --features grpc
      cargo check --features lab       # umbrella
      cargo clippy --features lab -- -D warnings
    before declaring complete.

8.  Pin your toolchain.
    rust-toolchain.toml at repo root. 1.94 stable today. 1.95+ has
    destabilisations (JSON target specs → unstable; mut ref patterns
    feature-gated). Bump deliberately in its own PR, never as part of
    feature work.
```

### Read order for new sessions

Before touching anything in this workspace:

1. `.claude/BOOT.md` + `LATEST_STATE.md` + `PR_ARC_INVENTORY.md` (CLAUDE.md mandatory)
2. `.claude/knowledge/cognitive-shader-architecture.md` (7-layer stack + SoA column types + 4 data patterns)
3. `.claude/knowledge/lab-vs-canonical-surface.md` (invariants I1-I11 + six rules A-F)
4. `.claude/contracts/ripple-dto-contracts.md` (StreamDto / ResonanceDto / BusDto / ThoughtStruct — the actual contract, not re-derivable)
5. `.claude/knowledge/encoding-ecosystem.md` (MANDATORY before codec work)
6. `.claude/CODING_PRACTICES.md` (this file) + `.claude/agents/BOOT.md` (delegation targets)

Before grepping source: consult the matching `.claude/agents/*.md` card.
Before inventing a type: grep `LATEST_STATE.md` Contract Inventory.
Before naming something an "epiphany": check whether it's ORIENTATION
already in the architecture docs; if so, tag it ORIENTATION, not FINDING.

---

## MANDATORY: `ndarray::simd::*` canonical import

Per lab-vs-canonical invariant **I2** and Rule B of the six JIT kernel
rules. Every SIMD access across `lance-graph` / `cognitive-shader-driver`
/ `thinking-engine` / `bgz-tensor` / `p64-bridge` goes through one of
three canonical surfaces — **nothing else**.

```rust
// ─── Canonical SIMD lane types (always prefer these) ─────────────────
use ndarray::simd::{F32x16, U8x64, F16x32, F64x8, BF16x32};
use ndarray::simd::{Fingerprint, hamming_distance_raw, popcount_raw};

// ─── AMX sibling module (Intel AMX on Sapphire Rapids+) ──────────────
use ndarray::simd_amx::{amx_available, vnni_dot_u8_i8, vnni_matvec, matvec_dispatch};

// ─── AMX tile primitives (inline-asm stable path) ───────────────────
use ndarray::hpc::amx_matmul::{
    tile_loadconfig, tile_zero, tile_load, tile_store, tile_release,
    tile_dpbusd, tile_dpbf16ps, vnni_pack_bf16,
};

// ─── Runtime caps detection ─────────────────────────────────────────
use ndarray::hpc::simd_caps::{simd_caps, SimdCaps};

// ─── WRONG (private backend reach — reviewer MUST reject) ──────────
use ndarray::hpc::simd_avx512::F32x16;    // private impl; namespaces can refactor
use ndarray::hpc::simd_avx2::F32x8;        // same
use std::arch::x86_64::_mm512_loadu_ps;    // raw intrinsic, not a kernel
use std::simd::Simd;                       // stdlib portable_simd, not canonical here
```

**Polyfill hierarchy** per Rule C — `ndarray::simd::*` primitives resolve
to the right tier at dispatch; consumers never emit raw intrinsics:

```
Tier 1 — Intel AMX tiles (Sapphire Rapids+, OS-enabled XCR0+prctl)
         matmul-heavy paths only: tile_dpbusd, tile_dpbf16ps
Tier 2 — AVX-512 VNNI: vnni_dot_u8_i8, vnni_matvec
Tier 3 — AVX-512 baseline: F32x16, U8x64, F64x8 (mandatory floor
         per target-cpu=x86-64-v4)
Tier 4 — AVX-2 fallback: F32x8, F64x4 (cfg-gated)

There is NO consumer-visible scalar tier. If a SoA path scalarises,
the SoA invariant is broken. ndarray::simd handles any non-x86
scalar fallback INTERNALLY; the consumer never hand-rolls.
```

**Reviewer trigger:** any PR with `std::arch::*` or
`ndarray::hpc::simd_avxNNN::*` or a scalar loop on a column-sweep
hot path → reject + cite this section. Exception: the ndarray
crate itself implements backends, not a violation.

---

## The 3-Way BindSpace Mutation Scheme

BindSpace is **read-only** — `Arc<[u64; 256 * N]>` columns, never mutated
in place. Writers hold owned `Copy` microcopies (CausalEdge64, Band,
TruthValue, ThinkingStyle — all ≤ 16 B, stack-only). All mutations cross
the airgap through `contract::collapse_gate::{GateDecision, MergeMode}`.
Three merge modes, three use cases — **never interchange them**.

```
┌─────────────────┬────────────────────┬────────────────────────────────┐
│ MergeMode       │ Semantics          │ Use when                        │
├─────────────────┼────────────────────┼────────────────────────────────┤
│ MergeMode::Xor  │ target ^= delta    │ SINGLE writer. XOR is           │
│                 │ self-inverse       │ self-inverse → reversible, no   │
│                 │ commutative+assoc  │ locks needed. Good for:         │
│                 │                    │ ─ per-expert A2A blackboard     │
│                 │                    │   posts (one expert at a time)  │
│                 │                    │ ─ single-row edge flips         │
│                 │                    │ ─ individual CausalEdge64       │
│                 │                    │   updates on an owned cursor    │
├─────────────────┼────────────────────┼────────────────────────────────┤
│ MergeMode::     │ saturating-add     │ MULTIPLE concurrent writers     │
│ Bundle          │ into int8          │ to the SAME target. Associative │
│                 │ awareness register │ and commutative IN EXPECTATION  │
│                 │                    │ (E-SUBSTRATE-1 guarantees       │
│                 │                    │ Chapman-Kolmogorov by           │
│                 │                    │ construction in d=10_000;       │
│                 │                    │ JL concentration suppresses     │
│                 │                    │ deviations at rate e^(-d)).     │
│                 │                    │ Good for:                       │
│                 │                    │ ─ codec sweep result aggregation│
│                 │                    │ ─ thinking-engine lens          │
│                 │                    │   composition                   │
│                 │                    │ ─ multi-expert consensus        │
│                 │                    │ ─ soaking CollapseGate deltas   │
├─────────────────┼────────────────────┼────────────────────────────────┤
│ MergeMode::     │ keep all variants  │ AMBIGUITY worth preserving. No  │
│ Superposition   │ present until      │ writer has enough confidence to │
│                 │ CollapseGate       │ commit; next cycle may          │
│                 │ resolves           │ disambiguate with new evidence. │
│                 │                    │ Good for:                       │
│                 │                    │ ─ WechselAmbiguity coref        │
│                 │                    │   candidates                    │
│                 │                    │ ─ ResonanceDto family_candidates│
│                 │                    │ ─ contradiction as structure,   │
│                 │                    │   not noise                     │
└─────────────────┴────────────────────┴────────────────────────────────┘
```

**Iron rule (CLAUDE.md I-SUBSTRATE-MARKOV):** `MergeMode::Xor` on a
**multi-writer** path breaks the Markov guarantee. Two concurrent
XOR writes cancel — racing writers lose their deltas. Always use
`Bundle` when two or more writers can reach the same target
simultaneously. Reversibility vs associativity is the trade-off;
pick based on writer cardinality, not convenience.

**Reviewer trigger:** any PR that uses `MergeMode::Xor` on a path
where concurrent writers exist → reject + require justification
that the path is truly single-writer (typically via an owned
cursor or serialized commit).

---

## Adhering-Agent Review Checklist

This document doubles as the **review checklist each specialist agent
consults** when approving code in its domain. Spawn the right agent,
hand it the PR scope, and it walks these sections looking for the
patterns it owns:

| Agent                      | Owns these checks                                       |
|----------------------------|--------------------------------------------------------|
| `family-codec-smith`       | SoA over AoS; precision-ladder in builder; stub flag   |
|                            | on Phase-N-before-Phase-N+k DTOs; HEEL/HIP/BRANCH/TWIG/ |
|                            | LEAF progression; role-aware codebooks; no one-codebook |
|                            | for all families.                                       |
| `bus-compiler`             | StreamDto/ResonanceDto/BusDto/ThoughtStruct lifecycle;  |
|                            | BusDto compilable into `CausalEdge64`; style maps to    |
|                            | explicit bus knobs; no vague summaries.                 |
| `palette-engineer`         | `ndarray::simd::*` canonical import ONLY; 256×256       |
|                            | palette distance tables for bgz17; no std::arch reach.  |
| `certification-officer`    | Feature matrix complete (default/serve/grpc/lab); spec- |
|                            | drift guards present; stub flag honored end-to-end;     |
|                            | 4-decimal metric reporting; SplitMix64 seed 0x9E37…     |
|                            | for deterministic sampling.                             |
| `truth-architect`          | The object does the work; three-view SoA isomorphism    |
|                            | (AGI ≡ Test ≡ Think); no new crate for a new question.  |
| `integration-lead`         | Rule A–F compliance on every JIT kernel; feature gates  |
|                            | coherent; board-hygiene commits paired with code.       |
| `ripple-architect`         | Ripple architecture (stream → resonance → bus → thought);|
|                            | contradiction preserved; no collapse into generic       |
|                            | confidence.                                             |
| `host-glove-designer`      | AGI-as-glove doctrine; structured payloads over         |
|                            | prompt-only mediation; minimal useful interface.        |
| `resonance-cartographer`   | ResonanceDto superposition preserved; coherence +       |
|                            | contradiction + pressure + drift all tracked; no early  |
|                            | collapse.                                               |
| `trajectory-cartographer`  | BusDto trajectories across cycles; cycle_fingerprint    |
|                            | as cache key + retrieval key + replay seed + cursor.    |
| `container-architect`      | Container = [u64; 256] 16 Kbit width; SoA column        |
|                            | layout (one Arc per column, packed u32 meta); no        |
|                            | per-row heap allocations.                               |
| `contradiction-cartographer` | Contradictions as first-class (phase + magnitude);    |
|                            | Staunen markers fire; epiphany vs error-correction      |
|                            | classification.                                         |
| `mirror-kernel-synthesist` | Mirror/reflection primitives respect the 64×64/256×256/ |
|                            | 4096×4096/16K resolution ladder.                        |
| `perspective-weaver`       | Topic/angle/style/edge columns independently Hamming-   |
|                            | sweepable; AND-across-cascades query model.             |
| `thought-struct-scribe`    | ThoughtStruct durable + revisable (not replacement-     |
|                            | only); storable on blackboard; distinct from raw text.  |
| `savant-research`          | Real measurements over synthetic corpus (Rule 23);      |
|                            | NaN scan at every pipeline stage; two-universes         |
|                            | firewall (Basin 1 continuous ≠ Basin 2 discrete).       |
| `adk-coordinator`          | Multi-agent orchestration; A2A blackboard via           |
|                            | BindSpace expert column.                                |
| `adk-behavior-monitor`     | Agent behavior against board invariants; handover      |
|                            | protocol discipline.                                    |
| `workspace-primer`         | New-session orientation; mandatory pre-reads honored.   |

**Spawn pattern:** when authoring a PR, identify the 1–2 agents whose
domains the code touches, hand them the scope with a pointer to this
doc, and let them return approval (PASS) or a blocking-diff list (FAIL
with specific lines + which section was violated). Agents read this
doc as their rubric, not their personality.
