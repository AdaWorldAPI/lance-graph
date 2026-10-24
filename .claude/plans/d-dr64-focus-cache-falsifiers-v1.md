# D-DRF-1 falsification preregistration — lease, prefetch, foreign gather and writeback

**Status:** PRE-REGISTERED, NOT RUN (2026-10-24). Companion to [D-DRF-1](d-dr64-focus-lease-foreign-cache-v1.md). This is a **proposed** lab matrix, not a declaration that a DR focus cache, third relationship axis or ONNX learning stream exists.

## 1. All-in decision rule

We compare three **correct**, source-pinned execution strategies for the same typed query and the same 64-thought workload:

- **A (reference):** direct immutable DR64 u16 map -> borrowed SoA, with optional software lookahead prefetch.
- **B (materialized):** one legal/gated foreign facet gather -> compact resident cache -> repeated native ndarray folds.
- **C (materialized + writeback):** B plus dirty-local marking, duplicate-destination deterministic reduction, owned dirty payload handoff to the existing CycleWriter.

The winner is not allowed to change Boolean/Distinct vs Bag vs SQL K3, the source scope, ONNX numeric semantics, contribution coverage or the version horizon to obtain a lower time. Compare a CPU scalar oracle and existing native ndarray/mask-risc behavior first, *then* costs. The legal outcome may be **REFUSE**, direct baseline, or fallback.

Cost breakdown per strategy: initial source/index bind; version/scope admission; cache miss and build; focus lease acquisition; cold page faults; prefetch requests and faults; execution; dirty reduction; full-row materialization; writer submit and backpressure; durability latency separately; memory retention; eviction; cross-focus churn. Do not amortize A differently from B/C or quote post-warm execution alone.

Sweep reuse N = 1, 8, 64, 256 and switching patterns AAA... / ABCABC... / adversarial unique tables. For the operator's expected case use 64 thoughts and up to three mixed foreign views (Disease / Phenomenology / LOINC). Report *actual* reuse counts for each MapId. A cache cannot save random reads if each foreign facet is used only once.

## 2. Geometry and identity falsifiers

| ID | Setup | Expected decision / observation | Disable that must fail |
| --- | --- | --- | --- |
| G1 | 32 x 128-bit, 64 x 64-bit, 64x64 u16 mapping, and 64^3 Boolean cube independently constructed | Equal **4096-bit** Boolean tile views, 512 B; pointer map **8192 B**; Boolean cube **32768 B**; combined 40960 B when both allocated; no 64^3 pointers or extra SoA rows | Conflate bitmask bytes with u16 bytes |
| G2 | DR64 map with repeated ordinals and 4,096 different ordinals; target row ABI 512 B | Max distinct targets 4,096 = **2 MiB footprint**, not mandated RAM allocation or bytes transferred. Full DR256 map 65,536 distinct = **32 MiB** | Multiply by relationship axis or by 64 thoughts |
| G3 | Same u16 ordinal in two different SoA datasets / branches at equal numeric version | Never infer equal identity; cache reuse must refuse without dataset/branch/index witnesses | Key only on local ordinal and bare version |
| G4 | Sealed map A, append/reseal/compact source; map B has same focus-slot numerals but different row-address seal | Old map remains replayable with old snapshot; B requires new binding; stale A->B reuse refused | Drop row-index seal / stable-ID requirement |
| G5 | MapPalette slot P -> Q -> P; thought slot reused after squash; identical shape and facets | Reject stale palette/ThoughtCommitId association despite reused slot and shape | Cache key only on palette slot / thought slot |
| G6 | Same physical pointer, different authorized tenant/scope | Reject cross-scope cache reuse before exposing bytes | Remove authorization from cache admission |
| G7 | Active focus omits a dependency contributor, or closure exits 64-local aperture | Expand only via an explicitly authorized and measured binding, otherwise refuse; do not silently turn missing into FALSE or zero | Pretend focused support is global support |

## 3. Lifetime, safety and prefetch falsifiers

| ID | Setup | Expected decision / observation | Disable that must fail |
| --- | --- | --- | --- |
| L1 | Two simultaneous threads demand identical witnessed Focus A | At most one completed source gather per cache-entry construction; both read the same immutable contents; lease begins when focus loads, not first fold | Each reader creates a new uncontrolled gather |
| L2 | Reader 1 holds Focus A; focus manager loads/activates B and retires A; reader 1 continues | A remains valid to reader 1, B available to reader 2, no use-after-free; A reclaimed only after final pin and dependent writer references are detached | Reclaim A immediately on focus switch |
| L3 | Load focus B speculatively but never use it; memory budget asks for eviction | Eviction can reclaim B without changing durable map or fact data; no leaked forever-static LazyLock | Retain B in a global static |
| L4 | Borrow NodeRowPacket from real canonical SoA, then try mutable owner write while borrowed | The **Rust compiler** refuses the aliased mutation; avoid inventing per-fold FrameStore checks | Introduce unsafe aliasing or interior writer |
| P1 | Random, clustered, repeated and adversarial 64K-target ordinals; lookahead distances 0,4,8,16,32,64 | Identical scalar/native results with and without prefetch; time and L1/L2/L3/DRAM misses reported by distribution; no presupposed speedup | Skip bounds on prefetch address or assume locality |
| P2 | Predict Focus B then use C; or make 64 fast focus switches | Prefetch cancellation/mis-prediction must not corrupt A/C or leak leases; track wrong-prefetch bytes, hit rate and eviction churn | Treat prefetch as a semantic commit |
| P3 | Prepare DR64 8 KiB map; prefetch 512-byte-strided SoA facet offsets from u16 ordinals | Prefetch targets declared facet offsets only, never arbitrary payload-derived foreign addresses; no cold page-fault costs hidden | Use unadmitted value as address |

**Note:** the repository's separate dataset idle-flush plan rejected a hot per-read lease protocol for its >3-day, >300 MB cold-dataset eviction policy. L1–L3 describe **DR focus cache lifetime**, not a retrofit of that storage policy.

## 4. Materialization and writeback falsifiers

| ID | Setup | Expected decision / observation | Disable that must fail |
| --- | --- | --- | --- |
| M1 | Three foreign 16-byte facets across 4,096 local pixels, repeat 64 mixed-ontology thoughts | Charge 3 x 64 KiB = **192 KiB** cached facet data plus references, metadata, masks and build; compare against direct on whole run and with cold/warm/focus switches | Report only 64 KiB or omit gather cost |
| M2 | Same base relationship question represented by direct borrowed view and materialized facet (including out-of-focus, NULL/unknown and Bag inputs) | Correctness/typed refusals identical to scalar and native baseline; materialization never broadens admission policy | Convert presence to truth or erase multiplicity |
| W1 | 4,096 local pixels, 37 unique dirty canonical destinations, 64 thoughts, 512-byte row ABI | **37 distinct full-row images = 18,944 B** of final row payloads at persistence, not 4,096 x 512 or 64 x 37 x 512. Count metadata and intermediate costs separately | Write each touched position/thought |
| W2 | Two local pixels alias one canonical row; two thoughts modify **different** facets; order intentionally shuffled | Final row preserves both updates via typed reduction against sealed Vn; stable outcome for all permitted interleavings | Apply raw last-payload-wins to independent partial patches |
| W3 | Two noncommuting same-facet updates with declared stream positions | Applied result exactly matches deterministic oracle, otherwise typed conflict/refusal; never accidental associative rewrite | Swap update order, silently add operands |
| W4 | Producer submits owned casts to bounded CycleWriter, WAL is delayed; thought continues on same source | Source owner borrow is released after truly owned handoff; reader lifetime depends only on outstanding reads, not WAL completion; durability mark does **not** advance prematurely | Await durability inline or report submit as durable |
| W5 | Alternative descriptor-only path points into live SoA and is queued | Focus/source lease must survive until writer captures/pins bytes. Early release forbidden; test invalidation race | Drop lease at mere descriptor enqueue |
| W6 | Queue full / writer poisoned / WAL precommit failure / lost acknowledgement | Ownership returned on submit failure; precommit regeneration reads sealed Vn; ambiguity resubmits SAME frozen batch; no duplicate published version; no stale focus re-read | Resample live source bytes during ambiguous retry |
| W7 | Intent-only or no artifact change, and a mixed artifact-backed cycle | Intent-only does not write a new version; changed dirty rows coalesce and publish one sealed successor version; historical DR map version unchanged | Force an empty WAL append |
| W8 | Active view still pinned after cache eviction under heavy memory pressure | Bound resident usage may exceed soft budget while live readers exist, but result remains correct and evicted inactive views rebuild deterministically | Evict live underlying SoA snapshot |

**The three amortization fractions are measured, not assumed.** The foreign-key cache exception rests on them, so DRF-P2 reports each per workload, with its denominator:

| ID | Quantity | Definition | Pre-registered expectation |
| --- | --- | --- | --- |
| A1 | dirty fraction | distinct canonical rows written back ÷ distinct rows read, per focus version | small (W1's 37 / 4,096 shape); no target |
| A2 | read-hit fraction | facet reads served from the resident cache ÷ all facet reads, after the cold build | 70–90% |
| A3 | value-free learning fraction | learning steps answered from Boolean relationship planes alone ÷ all learning steps | "many"; no target |

If A2 lands below the expected band, or A3 is near zero, report it as the reason before reporting any timing; the exception's basis is then weaker than stated and the break-even below decides it.

Break-even test: C wins only when its **full** gather+64 folds+dirty+submit cost is lower than the legitimate A baseline for equivalent work. Report the lower and upper crossover N and the conditions (density, locality, cache budget, dirty fraction, focus churn) under which the win reverses.

## 5. Relationship and learning separation falsifiers

| ID | Setup | Expected decision / observation | Disable that must fail |
| --- | --- | --- | --- |
| R1 | Typed identity x property x relationship-selector 64^3 Boolean cube; select one 64^2 relation slice | Slice and independent scalar ternary oracle agree; no forced allocation of full 32 KiB cube for a single sparse relation | Assume every cube dimension means the same type |
| R2 | Try reinterpreting relationship selector as a SoA ordinal, CE64 P/W map ID, or thought occupancy lane | Reject; mapping index, relation selector and speculative thought axis remain separate | Collapse axes into one numeric namespace |
| R3 | Relational composition over local focus with missing contributing intermediate | Exact only with complete dependency closure; else authorized expansion or refusal | Mask missing contribution as absent |
| R4 | Proposed plastic overlay learns new strength but base relation/map revision V stays pinned | Old V remains bit-identical and historical reads unchanged; overlay carries independent commit/model/source-root identities; admitted proposal produces V+1 only on explicit commit | Mutate base DR in place |
| R5 | Duplicate learning source roots / correlated evidence + tanh or palette saturation | Refuse unwarranted independent-evidence accumulation; numerical boundedness does not certify truth or NARS/Bayes equivalence | Count repeated evidence as independent |
| R6 | ONNX model/weight shape collision but different graph or codebook generation | Refuse cached compiled result unless complete source/model/layer/weight/calibration identity matches; independently check typed approximate/exact contract | Use shape-only/d_head key |
| R7 | Persist candidate learning artifacts in hypothetical S3-like object storage and reload | Object storage is an **optional hydration/versioned artifact path**, not a random-read tensor store or proof engine; replay from pinned local snapshot, not live remote page faults | Treat ONNX format or S3 as automatically executable, durable or certified |

## 6. Measurement and status reporting

Pre-register results as **PASS / FAIL / BLOCKED / NOT RUN**. For each cell include seed and workload generator, machine/CPU/ISA, compiler flags and build profile, source snapshot and MapId identity, operations/semiring, exposed bytes, allocation counts, cold versus warm times, cache-miss and focus-switch distribution, p50/p95 and tail, writeback reduction cardinalities and durability completion separately. Report model quality if comparing learned approximate outcomes, with an independent held-out oracle and explicitly declared tolerance.

**Non-goals in this PR:** no CI implementation, no claim of actual latency, no DR64^3 ABI, no CE64 extension, no extra mutable table, no runtime Tensor/ONNX interpreter, no new storage lease policy and no EXO-50 post.

**Follow-up after review:** one small falsifier-led probe PR at a time, then create the coding prompt and EXO-50 handoff from the accepted reference rather than re-deriving semantics from chat.
