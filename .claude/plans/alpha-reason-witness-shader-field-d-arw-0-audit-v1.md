# D-ARW-0 shader-field source audit v1

> **Status:** SOURCE AUDIT / PLAN-ONLY. No production wiring in this file.
> **Date:** 2026-08-30.
> **Owner:** #1078 `alpha-reason-witness-cognitive-fabric-v1` D-ARW-0.
> **Audited lance-graph baseline:** `main` at `de1d0c2fe54f36bf0d4d3f1393c55d0cea0c3ae9`.
> **Purpose:** recover the real current and historical wire underneath the operator-recalled `StreamDto → Resonance/Perturbation → P64/cognitive-shader-driver → CE64 → BusDto` field architecture before reconnecting anything.

## 0. Evidence grades used here

- **SOURCE FACT:** current executable/source establishes it.
- **HISTORICAL SOURCE FACT:** committed historical source/PR/doc establishes it.
- **OPERATOR-RECOVERED INTENT:** original design intent recalled by the operator, not independently recovered verbatim from source.
- **BROKEN WIRE:** both sides exist but the producer/carrier/consumer path drops, collapses, or changes identity.
- **UNPROVEN IDENTITY:** shapes are compatible but no source establishes that their addresses name the same locus.
- **MODERN DRIFT:** current implementation materially differs from the recovered original intent.
- **NUMERICAL COINCIDENCE:** same dimensions/number with no proven semantic identity.

## 1. One-screen verdict

The archaeology did **not** recover one pristine old pipeline. It recovered **two live halves separated by a missing address identity**:

```text
CURRENT THINKING FIELD                         CURRENT P64 / CE64 FIELD

StreamDto.codebook_indices                    CausalEdge64
        │                                           │
        ▼                                           ▼
BindSpace rows                              edge_to_block(S/4,O/4)
        │                                           │
        ▼                                           ▼
PerturbationDto.energy[~4096]              [[u64;64];8] topology planes
        │
        X  dense energy never crosses
        │
        └─ top_k only
              │
              ▼
       threshold + min..max
              │
              ▼
         ColumnWindow
              │
              ▼
      Vec<u32> passing rows
              │
              ▼
        ShaderDriver
              │
              ▼
          ShaderHits
              │
              ▼
     CausalEdge64 emission
```

**BUY:** source archaeology and a read-only address-identity/sabotage probe.

**HOLD:** production reconnection, EWA integration, R2IL field programming, Alpha lineage, or a new mask carrier until identity is proven.

**NO BUY:** another DTO, another 4096 carrier, or a guessed `codebook_index ↔ P64 cell` conversion.

## 2. Current DTO field is real

`crates/thinking-engine/src/dto.rs` defines the current four-stage vocabulary:

```text
Φ StreamDto       sensor output enters the field
Ψ PerturbationDto ripple field, Vec<f32> energy, canonical size 4096
B BusDto          committed thought
Γ ThoughtStruct   stabilized thought
```

`PerturbationDto::from_energy_f32` copies the energy slice and derives a top-8 by sorting indexed energies.

The file explicitly records the mechanical rename:

```text
ResonanceDto  →  PerturbationDto
```

while a distinct perspectival `ResonanceDto` remains in `awareness_dto.rs`.

**Classification:** SOURCE FACT.

Primary source: `crates/thinking-engine/src/dto.rs`.

## 3. Source-proven broken seam: perturbation never reaches P64

`crates/cognitive-shader-driver/src/engine_bridge.rs` states that two DTO pipelines existed separately:

```text
thinking-engine:          StreamDto → PerturbationDto → BusDto → ThoughtStruct
cognitive-shader-driver:  ShaderDispatch → ShaderResonance → ShaderBus → ShaderCrystal
```

The bridge currently does:

```text
StreamDto.codebook_indices → BindSpace fingerprints
PerturbationDto.top_k       → ShaderDispatch.rows
ShaderBus top hit           → BusDto.codebook_index
```

The critical source comment is explicit:

- `dispatch_from_top_k` is a **control-plane WINDOW heuristic, not a mask**;
- `{7,42,900}` becomes the dense window `7..901`;
- `PerturbationDto.energy` **never reaches this seam at all**;
- source itself names `energy[i] ↔ p64 (S/4 × O/4)` identity as **UNPROVEN**.

**Classification:** BROKEN WIRE + UNPROVEN IDENTITY.

Primary source: `crates/cognitive-shader-driver/src/engine_bridge.rs`.

## 4. CE64 ↔ P64 is current source fact

`crates/p64-bridge/src/lib.rs` currently implements:

```text
CausalEdge64
   ↓ edge_to_block
row = S / 4
col = O / 4
   ↓
64 × 64 palette block
```

It also builds:

- `[u64;64]` palette masks from edge batches;
- `[[u64;64];8]` predicate-layer planes from edge batches;
- style `layer_mask`, combine, contra and density parameters.

The current `CognitiveShader` documentation distinguishes:

```text
Mask      WHICH pairs interact
Distance  HOW FAR
Compose   WHAT composition means
```

and states that the mask says **WHERE** to look.

**Classification:** SOURCE FACT.

This corrects the earlier weak reconstruction: CE64/P64 coupling is not merely historical. The missing wire is upstream address identity from perturbation/codebook space into this CE64/P64 locus.

Primary source: `crates/p64-bridge/src/lib.rs`.

## 5. Current shader is genuinely the driver, but row admission is not fully lithographic

`crates/lance-graph-contract/src/cognitive_shader.rs` records the role reversal:

```text
Before: thinking-engine drives, shader helper
Now:    CognitiveShader drives, dispatches cycles, commits via sinks
```

Current `driver.rs` executes:

```text
meta prefilter
→ style
→ P64/bgZ cascade
→ cycle signature
→ CausalEdge64 emission
→ FreeEnergy gate
→ sink
```

However `crates/cognitive-shader-driver/src/backing.rs::prefilter` returns a dense `Vec<u32>` of passing row indices. The payload fingerprints are borrowed zero-copy afterwards, but row admission itself is currently materialized as an index vector.

**Classification:** SOURCE FACT + MODERN DRIFT relative to the operator-recalled mask/lithography rule.

This is not permission to replace the Vec yet. First prove the address identity and a mask-native buyer.

## 6. Current ThinkingEngine contradicts the recalled “never materialize the cognitive LUT” intent

`crates/thinking-engine/src/engine.rs` says literally:

```text
energy_next = distance_table × energy_current
The distance table is precomputed ONCE.
It IS the brain.
```

The canonical engine owns a `Vec<u8>` N×N distance table. For 4096 atoms that is a 4096² table, described as 16 MB / L3-resident, with row-sweep, VNNI and AMX tiling strategies.

Therefore the operator-recalled architecture:

> continuously process the resident/implicit field; do not materialize the giant cognitive LUT

is **not** the current ThinkingEngine contract.

**Classification:** OPERATOR-RECOVERED INTENT + MODERN DRIFT/CONFLICT.

Do not rewrite history in either direction. Small hot lookup tables and precomputed scientific kernels may still be lawful; the unresolved question is whether the *global cognitive relation field* should be materialized N².

Primary source: `crates/thinking-engine/src/engine.rs`.

## 7. Current CE64 emission does not prove a 12-bit field-cell identity

The current compare harness reproduces the real driver edge-emission recipe:

```text
s_palette = row % 256
o_palette = (row / 4) % 256
p_palette = 0
f,c       = resonance
```

That emitted CE64 is later interpretable by `p64-bridge::edge_to_block` as `(S/4,O/4)`.

This is a real current path, but it does **not** source-prove that a thinking-engine `codebook_index` is the same 12-bit coordinate as the resulting 64×64 P64 cell. `engine_bridge.rs` correctly fences that identity as unproven.

Further, `edge_v3_compare.rs` proves the staged V3 can drop the duplicate in-edge SPO and resolve SPO from the target node CAM-PQ facet while preserving syllogistic thinking.

**Classification:** SOURCE FACT for the emission recipe; UNPROVEN IDENTITY for codebook→P64; IMPORTANT FUTURE CONSTRAINT for “CE64 vertical mantissa.”

Interpret “vertical mantissa” as recovered architecture intent, not a commitment that future V3 must physically retain three SPO bytes in the edge.

Primary source: `crates/cognitive-shader-driver/src/edge_v3_compare.rs` plus `p64-bridge`.

## 8. SPO 2³ exists in current machinery, but cache-amortization meaning is not recovered

Current shader source contains Pearl/SPO 3-bit projection masks and precomputed NARS tables. The driver documents `Pearl 2³ + DK + Plasticity + Truth` lookup, and `RungLevel::causal_mask_bits()` maps current Pearl readings onto O / PO / SPO masks.

That proves **SPO 2³ is a live computational vocabulary**.

It does **not** independently prove the operator-recalled original meaning:

> CE64 as vertical mantissa informed by SPO 2³, amortized as eight cheap readings/cycles over L1-resident state.

**Classification:** SOURCE FACT for current 2³ projection; OPERATOR-RECOVERED INTENT for the 8-cycle/L1 amortization rationale.

## 9. Morton cascade is real and immaterialized; top-k ownership is unproven

`AdaWorldAPI/OGAR/docs/DISCOVERY-MAP.md` records:

```text
64 → 256 → 1024 → 4096 → 16k → 64k → 256k
```

as an **immaterialized Morton enumeration**, one extra nibble per level, and separately states that the cascade is a coordinate transform rather than a stored grid.

No current producer→consumer trace found in this audit establishes that `PerturbationDto.top_k` chooses among those cascade levels.

**Classification:** HISTORICAL SOURCE FACT for the cascade; UNPROVEN IDENTITY for `top_k = Morton aperture selector`.

Falsifier remains F-ARW-SHADER-2: if no shared owner/path is recovered, record only compatible geometry.

## 10. Resonance → Perturbation rename is source-backed

Merged lance-graph PR #1043 records `ResonanceDto` as `REPURPOSE → PerturbationDto`, with migrated code and stale plans. Current `dto.rs` preserves a deprecated alias and explains the mechanical vs perspectival name split.

**Classification:** HISTORICAL SOURCE FACT confirmed by current source.

## 11. Stockfish-rs has the two roles the operator recalled

### 11.1 Systems/geometry oracle

`stockfish-harvest-64x64-v1.md` explicitly starts from:

- chess-native `64×64 = 4096` from×to address space;
- incremental NNUE make/unmake updates;
- Stockfish C++ as oracle + harvest source, never runtime dependency;
- leaf-by-leaf parity against real Stockfish.

It explicitly relates NNUE incremental accumulation to perturbation incrementality.

### 11.2 Expert-iteration oracle

Merged stockfish-rs PR #12 defines the teacher-stream hypothesis:

```text
Strict-time policy proposes/ranks
→ Retro deep search supplies hindsight labels
→ repeated cycles distil later search into faster present response
```

Its golden probe measured stable exact teacher labels, policy headroom and oracle-order node reduction; the named downstream path is NARS revision → selector/policy → held-out replay.

**Classification:** HISTORICAL SOURCE FACT for both roles.

Stockfish remains a systems/reference teacher and learning oracle. It is **not** evidence that Stockfish itself uses Morton hierarchy.

## 12. Alpha lineage remains unresolved because “Alpha” names different current mechanisms

Current lance-graph shader code has a live `AlphaComposite` / front-to-back merge over shader hits and qualia.

MedCare separately has the same-address thin-provisioned `AlphaOverlay` attention/residual plane audited in the parent plan.

Those are not the same carrier by name alone, and no source trace in D-ARW-0 establishes:

```text
Resonance/Perturbation → AlphaOverlay
```

or proves that shader `AlphaComposite` is the successor of the old perturbation field.

**Classification:** SOURCE FACT for both mechanisms; lineage UNPROVEN.

F-ARW-SHADER-4 remains active.

## 13. AttentionMaskSoA is another mask meaning, not the P64 plane

Current `cognitive-shader-driver/src/attention_mask.rs` stores flat `(mailbox_id,w_slot)` attention entries with activity, cycle and residual plus LRU eviction.

It is not the P64 `[u64;64]` topology mask and not MedCare's base-ordinal AlphaMask.

**Classification:** SOURCE FACT; vocabulary collision to preserve, not collapse.

## 14. Reconstructed intent, kept explicitly non-canonical

The operator-recovered original picture is recorded because it guides archaeology, but none of these phrases are promoted to current contract without source/probe evidence:

```text
StreamDto                  incoming perturbations / continuously thinking stream
Resonance/Perturbation     EWA-like active field, not truth
Morton cascade             variable immaterialized aperture
mask                       lithography: selected substrate IS the work surface
P64                        bounded hot field ALU
CE64                       vertical semantic/causal mantissa at an active locus
SPO 2³                     eight amortized readings over resident state
R2IL/loco                  cognitive shader program
Alpha                      residual / reversible working history
Revision/Rubicon           authority and durable writeback
Stockfish                  systems geometry teacher + expert-iteration oracle
```

The audit supports several descendants of that picture, but not the missing shared address identity.

## 15. D-ARW-0 classification table

| Relation | Verdict |
|---|---|
| `StreamDto → BindSpace` | SOURCE FACT |
| `PerturbationDto.energy[4096]` exists | SOURCE FACT |
| `PerturbationDto.energy → P64 mask` | BROKEN WIRE |
| `PerturbationDto.top_k → sparse mask` | BROKEN WIRE; currently dense `min..max` window |
| `CausalEdge64 → P64 64×64 block` | SOURCE FACT |
| P64 mask = “WHERE” topology | SOURCE FACT |
| shader driver owns dispatch loop | SOURCE FACT |
| shader row admission fully mask-native | FALSE today; `Vec<u32>` prefilter |
| 4096² global distance LUT materialized | SOURCE FACT |
| never-materialize-global-cognitive-LUT | OPERATOR-RECOVERED INTENT, conflicts with current ThinkingEngine |
| `ResonanceDto → PerturbationDto` | HISTORICAL SOURCE FACT + current alias |
| Morton 64→…→256k immaterialized cascade | HISTORICAL SOURCE FACT |
| `top_k` chooses Morton cascade level | UNPROVEN |
| CE64 “vertical mantissa” | OPERATOR-RECOVERED INTENT, partially echoed by CE64→P64 but not current contract |
| SPO 2³ current mask/projection | SOURCE FACT |
| SPO 2³ = eight-cycle L1 amortization rationale | OPERATOR-RECOVERED INTENT |
| Alpha = continuation of Resonance/Perturbation | UNPROVEN |
| Stockfish 64×64 + incremental systems teacher | HISTORICAL SOURCE FACT |
| Stockfish expert-iteration oracle | HISTORICAL SOURCE FACT |

## 16. Smallest next executable probe: ADDRESS-IDENTITY, not architecture

Do this in a separate bounded probe PR, not production code and not inside #1078's plan branch.

### A. Address reading arms

For a corpus of **independently grounded relations** that can be represented both as a codebook/field input and as a canonical CE64/V3 target relation, record without changing production:

```text
A0 current bridge row/codebook index
A1 current CE64 emission recipe → p64 edge_to_block
A2 row-major 12-bit → 6+6 candidate reading
A3 Morton 12-bit → 6+6 candidate reading
A4 seeded random permutation sabotage
```

Do not select a winner from internal self-consistency. The oracle must be an independently grounded relation/address source.

### B. Sparse-work arm

For the same active sets, compare work cardinality only:

```text
current top_k → min..max ColumnWindow
vs
exact sparse bitmask over the SAME proven address identity
```

This is a measurement arm, not a production mask proposal.

### C. Required falsifiers

1. **Permutation sabotage:** if random permutation does not damage the locality-sensitive metric, Morton is decorative for this buyer.
2. **Identity mismatch:** if no candidate mapping reproduces independently grounded relation identity, STOP. Do not reconnect Perturbation to P64.
3. **Sparse silence:** if exact sparse selection does not reduce meaningful work or changes semantics, NO BUY for mask lowering here.
4. **No authority:** none of these field/layout measurements may change evidence, ReasoningBand, Revision or Rubicon semantics.

### D. Gate before EWA/R2IL

Only after one address reading survives the independent oracle and sabotage should a second probe test:

```text
same masked field
→ one existing local P64/EWA-shaped primitive
→ one existing R2IL/loco composition
→ same typed result/receipt
```

Until then EWA, R2IL and Alpha are consumers waiting on a coordinate contract, not evidence for one.

## 17. D-ARW-0 verdict

**PARTIAL BUY / RESTORATION HOLD.**

The source establishes a real field lineage, a real current CE64→P64 topology path, a real mechanical perturbation field, a real shader-driver role reversal, an immaterialized Morton cascade, and Stockfish's dual systems/teacher role.

The source also establishes the break:

> **the perturbation field never reached the mask ALU, sparse top-k collapses to a bounding window, and current code does not establish that a 4096 perturbation/codebook index names the same locus as the CE64/P64 64×64 cell.**

That missing identity is now the only honest first question. No representation is bought before it answers.