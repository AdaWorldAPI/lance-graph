# D-ARW-0 shader-field archaeology — pass 1

> **Status:** SOURCE AUDIT / PLAN ONLY. No production wiring in this file.
> **Date:** 2026-08-30.
> **Current-main snapshot audited:** `de1d0c2fe54f36bf0d4d3f1393c55d0cea0c3ae9`.
> **Owner plan:** `.claude/plans/alpha-reason-witness-cognitive-fabric-v1.md`.
> **Lineage addendum:** `.claude/plans/alpha-reason-witness-shader-field-lineage-addendum-v1.md`.
>
> This pass exists to answer one narrower question before any implementation BUY:
> **what parts of the historical Stream / resonance / perturbation / shader / mask / CE64 / EWA field machine still exist as current executable source, what is merely historical, and where is the wire actually torn?**

---

## 0. Evidence vocabulary

This pass deliberately separates:

- **SOURCE FACT** — current executable source at the snapshot above establishes it.
- **HISTORICAL SOURCE FACT** — committed historical source establishes a former implementation or architecture statement.
- **PLAN FACT** — a PR / plan / board entry states intent; not runtime proof.
- **OPERATOR-RECOVERED INTENT** — the operator remembers the original architectural meaning; recorded as such when no independent historical source has yet been recovered.
- **BROKEN WIRE** — both sides exist, but the inspected current producer → carrier → consumer path does not join them.
- **UNPROVEN** — attractive relation, insufficient evidence.
- **NO-BUY** — do not mint a new carrier / mapping / controller yet.

GitHub code search is not treated as an absence oracle in this pass: it returned zero for identifiers that are visibly present in fetched source. Negative claims therefore use direct inspected paths and are scoped accordingly.

---

## 1. Current producer: the dense perturbation field is real

### SOURCE FACT — DTO family

`crates/thinking-engine/src/dto.rs` defines the current doctrinal family:

```text
Φ  StreamDto        sensor / intake enters field
Ψ  PerturbationDto  active ripple field
B  BusDto           committed thought transport
Γ  ThoughtStruct    persisted stabilized form
```

`PerturbationDto` currently carries:

```rust
energy: Vec<f32>
cycle_count: u16
converged: bool
top_k: [(u16, f32); 8]
```

`from_energy_f32` derives `top_k` by sorting the dense energy vector descending and taking the largest eight entries. The deprecated mechanical alias `ResonanceDto = PerturbationDto` remains in this module; a different perspectival `ResonanceDto` exists elsewhere and must not be conflated.

**Verdict:** the dense field is not archaeological prose. It is a current carrier.

### SOURCE FACT — current field engine

`crates/thinking-engine/src/engine.rs` currently executes a repeated field update over `CODEBOOK_SIZE = 4096`:

```text
energy_next = distance_table × energy_current
```

The engine returns a full `PerturbationDto` after the cycle loop. `perturb(indices)` changes energy; `commit()` derives the BusDto from the resulting field / top-k.

The current implementation also materializes a `4096 × 4096` distance table (`TABLE_SIZE = 4096²`) and describes that table as the core of the current MatVec mechanism.

**Important correction to the recovered lineage:** the operator-recalled idea “always processing / do not materialize the cognitive LUT” cannot honestly be read as “no lookup/table is ever materialized” across all historical implementations. The current thinking-engine explicitly materializes the 4096² operator table, and the April cognitive-shader document explicitly used a read-only distance table as a texture. A narrower reading survives source better: **working cognition remains resident and is transformed by operators rather than repeatedly materializing object/row worklists or a table of all possible thoughts.** If the stronger rule really was “no distance LUT either”, then the current engine is a divergence and needs a separate falsifier, not retrospective rewriting.

---

## 2. The torn metre: dense field → shader mask

### SOURCE FACT — current bridge drops the dense field

`crates/cognitive-shader-driver/src/engine_bridge.rs` explicitly describes two DTO pipelines and the bridge between them.

The relevant current path is:

```text
StreamDto.codebook_indices
    → ingest_codebook_indices(...)
    → BindSpace content fingerprints

PerturbationDto.top_k
    → dispatch_from_top_k(...)
    → active indices above SCAN_WORTHY_ENERGY
    → min(index)..max(index)+1
    → ShaderDispatch.rows: ColumnWindow
```

The bridge documentation is explicit that this is a **control-plane WINDOW heuristic, not a mask**. For a sparse active set `{7, 42, 900}`, the consumer scans the contiguous interval `7..901`.

The full `PerturbationDto.energy` vector does not cross this inspected bridge into the P64 mask surface.

### BROKEN WIRE — field algebra collapses to a bounding interval

That establishes the current tear precisely:

```text
DENSE FIELD                     SHADER / P64
energy[0..N]                    real mask algebra exists
      │                               ▲
      └─ top 8 ─> min..max window ────┘
```

The semantic field survives on the producer side. The mask ALU survives on the consumer side. The current bridge reduces the field to a contiguous window before the mask can do the selective work.

This re-confirms the #1051 finding without relying on #1051 prose alone.

**D-ARW-0 verdict:** `BROKEN WIRE`, not missing concept and not numerical coincidence.

---

## 3. Current P64 surface: the lithographic half exists

### SOURCE FACT — CE64 → 64×64 field

`crates/p64-bridge/src/lib.rs` currently maps `CausalEdge64` into a 64×64 working surface:

```text
row = S_index / 4
col = O_index / 4
```

Thus the 256×256 S/O space is grouped into 64×64 blocks. `edge_to_layer_mask` maps the edge’s causal/inference reading into one or more of eight predicate layers. The bridge can produce:

```rust
[u64; 64]       // one 64×64 plane
[[u64; 64]; 8]  // eight predicate planes
```

The current `CognitiveShader` consumes those planes and a semiring; its own documentation says, in substance, the mask says **where to look**.

### SOURCE FACT — P64 does not currently consume PerturbationDto

The inspected current p64 source consumes CE64-derived mask topology. It does not take the thinking-engine dense `energy` carrier as an input.

### OPERATOR-RECOVERED INTENT — masking as lithography

The operator restores the original reading:

> **masking was supposed to make the work** — a descriptive `where()` / `method()` surface over resident data, zero-copy, like lithography rather than a row-wise homunculus.

This exact historical word is not independently recovered in current lance-graph source, but it is mechanically consistent with the present mask-native doctrine in `lance-graph-java`: semantic facade methods name operations; packed masks are native execution currency; row-id materialization is an explicit terminal.

The useful architectural statement is therefore:

```text
WHERE   = expose / narrow a resident population as a mask
METHOD  = apply a substrate operation to that population
```

not:

```text
WHERE   = build a collection of rows
METHOD  = iterate the collection in the facade
```

**NO-BUY:** do not add a second selection representation to resurrect the metaphor. The mask currency already exists.

---

## 4. CE64: what is source fact vs restored “vertical mantissa” meaning

### SOURCE FACT — physical layout

Current `crates/causal-edge/src/{edge,layout}.rs` retains the 3×8-bit S/P/O base at bits 0..23.

The current v2 layout also uses a signed i4 inference mantissa at bits 46..49. Bits 53..58 are W-slot territory. Bits 59..60 and 61..63 have additive modern readings for topology / ReasoningBand layered over historical compatibility constraints.

### OPERATOR-RECOVERED INTENT — CE64 as vertical mantissa

The operator restores a broader original role:

> **CausalEdge64 was intended as the vertical semantic / causal mantissa at the hot field locus, informed by SPO `2^3` amortization, with eight cheap readings of resident state and an intended L1-cache amortization across those cycles.**

Current source independently proves the 3×8 S/P/O fossil and a smaller inference mantissa field, but this pass has **not** recovered a historical source sentence saying “CE64 is the vertical mantissa” or proving the exact “8-cycle L1” contract.

Therefore:

- `3 × 8 S/P/O bits` = **SOURCE FACT**.
- SPO `2^3` / same-address eight-reading design is already recorded historically in the #1051 board archaeology, but there as operator-restored intent.
- “vertical mantissa” as the architectural name = **OPERATOR-RECOVERED INTENT**.
- exact 8-cycle/L1 performance contract = **OPERATOR-RECOVERED INTENT, MEASUREMENT REQUIRED**.

**STOP:** do not reinterpret unrelated CE64 bits to force this semantics. First prove the field-locus mapping and measure the reuse economics.

---

## 5. Morton cascade: real address geometry, not yet the P64 bridge

### SOURCE FACT — OGAR’s current discovery map

`AdaWorldAPI/OGAR/docs/DISCOVERY-MAP.md` currently states:

```text
64 → 256 → 1024 → 4096 → 16k → 64k → 256k
```

as an **immaterialized Morton enumeration**, every level adding one nibble. It explicitly calls the cascade a **coordinate transform, not a stored grid** and states the substrate goal as an “immaterialized Morton cascade with templated payloads”, with non-amortized per-query cost forbidden.

This is much stronger than a numerical rhyme. The cascade is real current documented address geometry in OGAR.

### UNPROVEN — top_k is the Morton aperture

Nothing inspected in the current thinking-engine / engine bridge proves that `PerturbationDto.top_k` selects a level of that Morton cascade.

Current `top_k` means simply “largest eight energy entries”. Current `dispatch_from_top_k` converts them to a min/max row window. No Morton deinterleave or cascade-level identity is established there.

### UNPROVEN — energy index ↔ P64 cell identity

The old unresolved address question remains:

```text
energy[i]
    ?=
P64 cell[row][col]
```

Possible mappings still include row-major, Morton 12→6+6, or a codebook-specific permutation. Count equality (`4096 == 64×64`) is not identity.

The decisive sabotage remains excellent:

```text
CONTROL     current top_k → min/max ColumnWindow
EXPERIMENT  energy → candidate spatial mapping → exact mask → P64 ALU
SABOTAGE    same values, deterministic random permutation of cell identity
```

If EXPERIMENT ≈ SABOTAGE on the locality-sensitive metric, the proposed geometry is decorative and receives **NO BUY**.

---

## 6. Historical cognitive-shader source: the framebuffer lineage is real

### HISTORICAL SOURCE FACT — `SESSION_COGNITIVE_SHADER.md`, PR #130 era

Committed historical source at `63e312073b5e2874fb0519a223901d338c5ce2e1` explicitly mapped:

```text
Distance table = texture
Energy vector  = framebuffer
MatVec         = shader dispatch
L4 accumulator = persistent storage buffer
```

It described the energy buffer as double-buffered, resident across cycles, and the whole thought as a series of dispatches with data staying in shared/storage memory. It also explicitly linked the Rust engine semantics back to `ada-consciousness` VSA superposition / collapse semantics.

This upgrades “cognitive shader” from a later metaphor to a **historically explicit execution model**.

### Important historical correction

That same source also explicitly loaded a read-only distance table into shared memory and sampled it repeatedly. Therefore the old shader lineage itself supports:

```text
resident operator table + resident mutable field + tiny outward result
```

more directly than the stronger statement “no LUT exists anywhere”.

The load-bearing historical idea is **resident execution with minimal movement**, not necessarily the abolition of all lookup tables.

---

## 7. EWA: three nearby mechanisms, currently distinct

This is the most important correction in this pass.

### 7.1 SOURCE FACT — real EWA-Sandwich kernel

`crates/lance-graph-contract/src/sigma_propagation.rs` currently implements the actual 2×2 SPD propagation:

```text
Σ' = M · Σ · Mᵀ
```

with `Spd2`, `ewa_sandwich`, `ewa_inverse`, `log_norm_growth`, and `pillar_5plus_bound`.

The module itself says the shader-driver propagation consumer is planned between edge emission and FreeEnergy gating.

### 7.2 SOURCE FACT — BindSpace carries a sigma index

`crates/cognitive-shader-driver/src/bindspace.rs` currently carries one `sigma: u8` per row and exposes `sigma_at` / `write_sigma`.

Its comments say the byte indexes a 256-entry Σ codebook. However, in the inspected `sigma_propagation.rs` current source, the public surface is the SPD kernel and tests; this pass did **not** find a `SigmaCodebook` implementation there. Treat “codebook owned here” in BindSpace comments as a wiring/documentation claim needing its own follow-up, not as established current source.

### 7.3 SOURCE FACT — current ShaderDriver does not visibly execute the Sandwich

The inspected current `crates/cognitive-shader-driver/src/driver.rs` hot path contains no `sigma` use. Its relevant stages remain P64/content hits, edge emission, FreeEnergy/MUL-style gate work, sink, then NARS/style revision.

Thus the B4 “shader-driver Σ-propagate” consumer named in the contract docs has not been established in the current inspected driver.

### 7.4 SOURCE FACT — AlphaFrontToBack is a different operation

The current shader contract / driver also has `MergeMode::AlphaFrontToBack` and an `AlphaComposite` result. The implementation performs front-to-back alpha accumulation over hit qualia with early saturation.

That is useful, but it is **not the same operation as `Σ' = MΣMᵀ`**. Do not call the alpha composite proof that the EWA-Sandwich field is wired.

### 7.5 SOURCE FACT — SPLAT is a third, distinct surface

`crates/lance-graph-contract/src/splat.rs` defines CAM-plane splat deposition:

```text
ReasoningWitness64 / centers / q8 amplitude / q8 width / channel
    → CamPlaneSplat
    → SplatPlaneSet::deposit
    → one of six 16K awareness planes
```

It preserves witness/replay identity and separates support, contradiction, forecast, counterfactual, style, source channels. The hot deposition shown here is bit-plane / q8 pressure geometry, not itself the SPD Sandwich.

### BROKEN WIRE — a promising but incomplete triangle

Current source therefore contains all three vertices:

```text
SPLAT deposition       BindSpace sigma index       EWA Sandwich kernel
       \                       |                       /
        \                      |                      /
         +---------- potential field path ----------+

                     ShaderDriver
                         ?
```

but the inspected production shader connection is not established.

This is exactly the kind of seam D-ARW-0 should recover before inventing another field abstraction.

---

## 8. Alpha: do not conflate two unrelated names

Two Alpha surfaces are currently relevant and must stay separate:

1. **MedCare AlphaOverlay / AlphaMask** — thin-provisioned same-address cognitive attention/residual overlay over `NodeRow`, with `AlphaStamp {cycle, seq, rung, visits}` and mask-native attention.
2. **Shader AlphaFrontToBack / AlphaComposite** — a merge mode that composites hit qualia with front-to-back alpha saturation.

They share the word Alpha and a reversible/working-surface flavor, but this pass has not established common producer identity, storage identity, or lineage.

**STOP:** `MedCare AlphaOverlay == shader AlphaComposite == old Resonance/Perturbation field` is currently an attractive story, not a source fact.

---

## 9. Stockfish-rs stays a dual reference, not Morton evidence

The stockfish-rs plans remain useful in two orthogonal ways:

1. **systems reference:** learn how a naturally 64×64 / 4096 domain keeps compact state hot, uses gather/lookup/scatter style patterns, incremental updates, deterministic fallbacks, and measures hardware reality;
2. **expert-iteration oracle:** deep exact root search labels candidate decisions; Strict observations are physically separated from Retro teacher labels; learned selectors must beat Frozen on held-out replay.

Stockfish itself does **not** provide evidence that Morton hierarchy is the correct mapping for the cognitive field. The mapping must survive the permutation sabotage independently.

The future reinforcement target can eventually include **work economy** as well as decision agreement, but only after the field address identity exists:

```text
same quality decision
with fewer admitted cells / fewer aperture expansions / fewer exact replays
```

Do not train against a spatial metric before the space is proven.

---

## 10. Pass-1 evidence table

| Claim | Grade after this pass |
|---|---|
| `PerturbationDto.energy` is a real dense current field | **SOURCE FACT** |
| `top_k` is eight largest energies | **SOURCE FACT** |
| current engine bridge drops dense energy and emits min/max window | **SOURCE FACT / BROKEN WIRE** |
| P64 is a real 64×64 × 8 mask surface derived from CE64 | **SOURCE FACT** |
| current P64 consumes PerturbationDto directly | **NOT ESTABLISHED** |
| CE64 retains 3×8 S/P/O bits | **SOURCE FACT** |
| CE64 overall role = “vertical mantissa” | **OPERATOR-RECOVERED INTENT** |
| SPO 2³ exact 8-cycle L1 amortization | **OPERATOR-RECOVERED INTENT; MEASURE** |
| OGAR 64→…→256k Morton cascade exists and is immaterialized | **SOURCE FACT (OGAR docs)** |
| `top_k` is the Morton aperture selector | **UNPROVEN** |
| energy index ↔ P64 cell map is canonical | **UNPROVEN** |
| historical Cognitive Shader used energy-as-framebuffer / table-as-texture | **HISTORICAL SOURCE FACT** |
| current EWA-Sandwich `Σ'=MΣMᵀ` kernel exists | **SOURCE FACT** |
| BindSpace carries `sigma:u8` | **SOURCE FACT** |
| current ShaderDriver consumes that sigma via EWA-Sandwich | **NOT ESTABLISHED / planned in kernel docs** |
| current AlphaFrontToBack is the EWA-Sandwich | **FALSE AS AN OPERATIONAL EQUIVALENCE** |
| SPLAT deposition exists with witness/replay + channel planes | **SOURCE FACT** |
| SPLAT→sigma→EWA→Shader is one production chain | **NOT ESTABLISHED** |
| MedCare Alpha is descendant of old Perturbation/Resonance field | **UNPROVEN** |
| Stockfish proves Morton | **NO** |
| Stockfish is useful as systems + expert-iteration oracle | **HISTORICAL SOURCE FACT / reference arm** |

---

## 11. What this changes in D-ARW-0

D-ARW-0 should no longer ask vaguely whether “the field” exists. It exists in several separately mature fragments.

The next source question is now narrower:

> **Can one real dense/current activity field be lowered into the existing mask-native working surface without losing address identity, while using the already-shipped EWA/SPLAT/CE64 machinery only where each has a proven producer and consumer?**

That is not yet an implementation request.

The first permitted executable probe is gated by address identity:

```text
real field fixture
    → candidate identity mapping
    → exact sparse mask (not min/max window)
    → existing P64 combine/contra/style operation
    → optional EWA 2×2 arm only if sigma producer is real
    → typed measurements

CONTROL   current min/max window
EXPERIMENT exact mask using candidate mapping
SABOTAGE  fixed random permutation of the mapping
```

Measure at minimum:

- result / ranking equivalence where expected;
- admitted cell count and words touched;
- cache / work cost if instrumentation already exists;
- locality-sensitive outcome under sabotage;
- whether EWA arm changes a predeclared geometry metric rather than merely adding compute.

**NO BUY** if permutation does not hurt the locality-sensitive arm.
**NO BUY** if exact masks do not reduce work or improve correctness over the window baseline.
**NO BUY** for new DTO, field carrier, provenance bit, rung tenant, or scheduler in this probe.

---

## 12. Pass-1 conclusion

The architecture is not missing. It is **disarticulated**.

Current source already contains:

```text
live dense perturbation field
real mask-native 64×64 working ALU
CE64 semantic carrier
immaterialized Morton cascade in OGAR
real 2×2 EWA-Sandwich kernel
per-row sigma byte
witness-aware SPLAT pressure planes
shader driver / resident BindSpace
Alpha surfaces
```

The inspected broken points are more specific:

```text
Perturbation energy → mask identity     TORN
Morton cascade → top_k aperture         UNPROVEN
sigma byte → EWA → ShaderDriver         NOT ESTABLISHED
SPLAT → EWA → production shader         NOT ESTABLISHED
old field → MedCare Alpha lineage       UNPROVEN
CE64 “vertical mantissa” exact runtime  OPERATOR-RECOVERED, unmeasured
```

That is enough to **stop designing representations** and continue archaeology / measurement.

The machine is not waiting for another abstraction. It is waiting for the correct wires to be proven.