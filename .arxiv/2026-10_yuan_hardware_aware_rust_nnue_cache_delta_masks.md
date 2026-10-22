# Hardware-aware Rust / NNUE: move less data, update less state, then prove the win

**Primary source:** Alan Yuan, [*Hardware-Aware Programming in Rust*](https://medium.com/@alan0408yuan/hardware-aware-programming-in-rust-6e68a70c1535), Medium, consulted 2026-10-22. The linked implementation is [alanyuan08/Chess-App](https://github.com/alanyuan08/Chess-App). This is a **blog/article excerpt and engineering reading**, not an arXiv paper or a peer-reviewed benchmark.

**Companion source:** Bo-Yuan Huang et al., [*Application-level Validation of Accelerator Designs Using a Formal Software/Hardware Interface*](https://doi.org/10.1145/3639051), ACM TODAES 29(2), Article 35 (2024). Consulted in the user-supplied 25-page PDF; its existing shelf treatment is [3LA / formal-interface note](2026-10_3la_formal_interface_flexible_matching_application_validation.md).

**Harvest policy and status:** Original paraphrase, source figures clearly attributed, no article reproduction. The **LG/DR/R2IL mapping below is an architectural inference and falsifiable proposal**, not a claim that the article benchmarked lance-graph, that the hardware features are universally available, or that proposed paths already ship.

## Source-grounded excerpt

**Y1. The running example is a Rust chess NNUE evaluator.** Yuan profiles a chess engine on an Apple M4 Pro and treats CPU-memory locality and instruction throughput as separate engineering problems. The article reports three approximate workloads: over **20M full position evaluations/s**, **40M incremental accumulator updates/s**, and **1.6M accumulator rebuilds/s**. These are different operations and must not be compared as if they had the same work per call.

**Y2. Incrementality is more important than a clever full recomputation.** In NNUE, a chess move usually changes only a few features. Rather than rereading every feature row and rebuilding the whole representation, the engine adjusts a retained accumulator using the weights for added/removed features. Yuan describes 512-byte feature rows and a much larger feature-weight corpus (about 24 MB), alongside a roughly 34.3 KB set of later-layer evaluation weights. The key separation is **large mostly cold model data** versus **small repeatedly touched inference state**.

**Y3. Alignment can avoid split cache-line reads.** His Rust example uses an accumulator of 256 `i16` values (512 bytes) with explicit 128-byte alignment on the profiled Apple target. If the line size truly is 128 bytes and the object is aligned, the array occupies exactly four lines rather than potentially five. `#[repr(C, align(N))]` establishes a type's alignment; it does **not** establish the cache-line size of arbitrary target machines, guarantee a value lives in L1, or establish optimality of that layout.

**Y4. Thread sharing can defeat an otherwise good layout.** His Lazy-SMP chess search uses a shared transposition table keyed by board position and depth. He describes grouping candidate entries into cache-line-sized buckets with depth/age/replacement preferences, motivated partly by false sharing and unnecessary line replacement. This is a cache and eviction-policy lesson, **not** proof that a 128-byte bucket is best for every machine or workload.

**Y5. SIMD changes the cost of repeated lane arithmetic.** Yuan contrasts native-vectorized and SIMD-disabled build configurations and reports approximately **27.93M versus 4.13M nodes/s** (about **6.8×**) for his tested evaluator. Those numbers belong to *his build and platform*. Disabling Neon/SVE changes code generation; without a controlled experiment and assembly/performance counters, the result cannot be attributed solely to one vector loop or generalized to mask-risc.

**Y6. Masks and indexed tables can reduce control-flow costs.** The article illustrates chess bitboards, mask intersection, set-bit iteration, arithmetic selection, and table indexing in place of some scalar branches. These techniques can expose contiguous arithmetic and predictable access patterns. The useful rule is **measure the branch and data-access behavior**, not “always remove `if`”: predictable branches can be cheap, branchless expressions can add instructions, and a compiler may already lower branches to selects.

**Y7. Keep the synchronization claim target-specific.** The article discusses packed 16-byte transposition entries and potential `AtomicU128` storage. A 16-byte aligned struct is **not automatically** a supported lock-free 128-bit atomic on all Rust targets. Check `target_has_atomic = "128"`, alignment, operation availability, and the memory-order contract; `Relaxed` gives atomicity of the individual operation where supported, not multi-field or multi-record publication correctness. Also, uncontended locks do not inherently make a kernel syscall.

## Chronology and existing architecture: SoA preceded this comparison

**Operator clarification (2026-10-22):** lance-graph's SoA substrate has used a **32 × 128-bit** geometry for **over a year**. This is **4,096 bits = 512 bytes** as a logical payload, the same total packed storage as 64 × 64 DR64 and the 256 × i16 NNUE accumulator. The source of the SoA design is **not** Alan Yuan's article or Apple's SDK; NNUE/Apple provide an **independent hardware-locality comparison**, not a novel design suggestion for this project. A 512-byte logical footprint alone does not guarantee alignment, cache residency, or identical access patterns across these three representations.

**Scope of this correction:** project history/geometry is supplied by the operator; the paper/article makes no claim about lance-graph, and this excerpt makes no claim of independently verifying the deployment date from version history. Work below should **reuse and measure the existing SoA layout**, rather than propose constructing it.

## Transfer into lance-graph: existing SoA and a potentially reusable NNUE-style delta

The architecture worth exploring is **delta-driven, cache-resident, locally addressed folds**, rather than transplanting a chess evaluator or adding a second execution engine.

```text
source snapshots (global identity, version, semantics)
     |
     v
cold binding: global ID -> versioned local ordinal
     |                    one-time / reused per focus group
     v
local row/feature indices + prefetch plan
     |
     v
DR mask / SoA lane window + compact delta
     |
     v
R2IL lawful fold -> Quack -> mask-risc execute_into
     |
     +--> tiny retained accumulator / sufficient statistic
     |
     +--> independently checked observable result
```

**The critical analogy:** the NNUE accumulator retains the contribution of unchanged chess features; a DR/R2IL fold could retain the contribution of **unchanged, version-identified rows/features**, then process only its delta. A global-to-local map makes the memory coordinates cheap to address during the execution window. It does **not** make the underlying source identity, source generation, semantic contract, or joins disappear.

For an **additive, exact-integer** accumulator with unchanged weights and no side effects:

```text
A_next = A_prev + sum(weight[added_features]) - sum(weight[removed_features])
```

For a **Boolean presence mask**, one may instead use set/clear or XOR delta operations under the proper invariants. **Bag multiplicity, COUNT, SUM, GROUP_SUM, SQL NULL validity, logical UNKNOWN, noninvertible saturation/clamping, floating-point rounding, and source revision are not all representable by the same XOR rule.** Maintain separate sufficient statistics or deliberately perform a rebuild when the incremental law is not admissible.

### Physical scales (conceptual bit-matrix figures, not measured cache residency)

| Representation | Logical bits | Packed storage | Observation |
|---|---:|---:|---|
| Existing SoA 32 × 128 bits | 4,096 bits | **512 B** | Pre-existing project layout (>1 year per operator); 128-bit coordinate/lane meaning is project-specific |
| DR64 64 × 64 bit matrix | 4,096 bits | **512 B** | Four 128-B lines or eight 64-B lines if packed and suitably aligned |
| DR256 256 × 256 bit matrix | 65,536 bits | **8 KiB** | Potentially compact L1 working set; actual footprint includes metadata, source lanes, outputs and other live state |
| NNUE example accumulator | 256 × 16-bit values | **512 B** | Same byte scale as DR64, but very different arithmetic and semantics |

The distinction between **packed mask size**, **active SoA payload**, **working-set size**, and **resident cache footprint** must remain visible in every performance claim. Packing 8 KiB of mask does not magically reduce unrelated source vectors to 8 KiB.

### Connection to mathematical compilation and the existing 3LA note

The Yuan article supplies the **physical optimization hypothesis**: data geography, accumulator reuse, SIMD lanes, masks, and avoided coherence traffic.

The attached 3LA paper supplies a **different, semantic discipline**: describe observable instruction behavior through a formal interface, use semantics-guided flexible matching to discover legal offloads, and check full application results. The existing [3LA excerpt](2026-10_3la_formal_interface_flexible_matching_application_validation.md) already maps that idea onto an aspirational egglog → R2IL → Quack → mask-risc seam.

These are complementary, not interchangeable:

- **Cold/compiler side:** discover and prove an algebraic equivalence, record numeric domain, Bag/Set meaning, validity/UNKNOWN semantics, source version, effects, and required feature layout.
- **Bind side:** construct a versioned global-to-local ordinal map, choose contiguous SoA windows and prefetch opportunities, and decline reuse on stale or incompatible bindings.
- **Hot side:** execute the already admitted R2IL/mask-risc operation on mask/lanes with the least observable movement; optionally maintain exact, justified deltas.
- **Oracle:** compare the entire typed result against an independent scalar/reference implementation. A mathematical proof of an algebraic law is **not** a proof of cache residence, speedup, hardware support, or source freshness.

Do not route hot execution through egglog, introduce a second SIMD VM, or pretend `target-cpu=native` is a portable deployment setting.

## Minimal falsifier/benchmark plan

All experiments should pin CPU model, target feature flags, Rust compiler, frequency/power constraints, input sizes, semantics, thread count, branch selectivity, source generation, repetitions, and warm/cold behavior. Record throughput and ns/op **plus** L1/L2 misses, instructions and branch misses where supported; do not invent counter values. Compare end-to-end binding/execute/output costs, not just a kernel microbenchmark.

| Probe | A/B arms | Decisive observation |
|---|---|---|
| **HW1: alignment** | 64-/128-byte aligned packed mask/SoA vs unaligned baseline, per target | Lower cache-line splits/misses and improved full-kernel latency without excessive padding; otherwise keep default |
| **HW2: incremental fold** | Full recompute vs exact delta add/remove over 0%, 1%, 5%, 20%, 100% changed rows | Outputs exactly match scalar oracle, including removals; identify delta density where rebuilding wins |
| **HW3: mask vs branch** | Mask intersection/set-bit walk vs scalar filter/branch across sparse/dense/predictable/random predicates | Include preparation and result materialization; verify whether SIMD/branch changes explain the win |
| **HW4: locality/binding** | Repeated local ordinal window and bounded prefetch vs repeated global indirection | Same identities under fresh generations; no accidental join on equal but unrelated ordinals; include map construction/rebind cost |
| **HW5: false sharing** | Shared cache-line-adjacent per-worker counters/entries vs isolated padded shards | Tail latency, remote invalidations if available, bytes of padding, and total-throughput effect |
| **HW6: 128-bit entry** | Conditionally supported atomic 128-bit packed entry vs existing correct fallback | Compile/behavior on supported and unsupported targets, torn-read oracle, generation/ABA contract |
| **HW7: end-to-end** | Existing R2IL/Quack/mask-risc baseline vs best proven combination above | p50/p95 plus total preparation and execution; permit “no optimization” as outcome |

**Smallest verification step (not a greenfield design):** locate the existing **32 × 128-bit SoA** path, its history, current parity tests and benchmark evidence before naming a gap. Run **HW2** only if a specific incremental fold/freshness case is untested; otherwise classify it as already covered. Use **HW4** only to test an observed binding/prefetch cost. Do not claim the project has only now discovered cache-local fixed-size data geometry. No production change until a missing case and total-cost evidence are demonstrated.

## Mistakes this excerpt must not license

1. “Fits in L1” is a working-set hypothesis, not a performance result; cache capacity is not wholly available to one structure.
2. A *512-byte* NNUE accumulator and a *512-byte* DR64 bit matrix are analogous in storage footprint, **not** equivalent representations.
3. Branchless is not automatically constant-time, faster, vectorized, or side-channel safe.
4. One `AtomicU128` load/store does not make a whole concurrent algorithm correct or portable.
5. A mask intersection is not by itself a relational join with Bag cardinality, source identity, NULLs, and provenance.
6. Reusing a delta across a changed snapshot/weights/ABI can be incorrect even if its mathematical formula was valid on the prior snapshot.
7. Hardware-specific tuning should be an **optional, measured specialization with a known-good baseline fallback**.

## Source pointers

- [Alan Yuan, *Hardware-Aware Programming in Rust* (original article)](https://medium.com/@alan0408yuan/hardware-aware-programming-in-rust-6e68a70c1535)
- [Chess-App (author-linked reference implementation)](https://github.com/alanyuan08/Chess-App)
- [Huang et al., 3LA, ACM DOI 10.1145/3639051](https://doi.org/10.1145/3639051)
- [Existing 3LA research shelf excerpt](2026-10_3la_formal_interface_flexible_matching_application_validation.md)

**Classification:** source-harvest / performance hypothesis / no shipped integration. **Follow-up owner:** a narrowly scoped HW2+HW4 benchmark decision, only after checking current main paths and task ownership.
