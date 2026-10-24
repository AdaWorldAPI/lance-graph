# D-DRF-1 — immutable DR64 focus, leased foreign caches and amortized dirty-row writeback

**Status:** REFERENCE DESIGN / PRE-REGISTRATION ONLY (2026-10-24). This document creates no Rust types, modifies no wire format, proves no speedup, and does not authorize a new executor. Its purpose is to make the operator's clarified design reviewable **before** a coding prompt or EXO-50 coordination.

**Canonical authority:** D-RMAP-1A (separate RelationshipMap tables and thought palettes), D-NFC (native-first dependency direction), D-NFR (representation status), D-MBX persistence, and the Rust-owned SoA. This note specializes their **focus residency / cache / writeback / optional learning** gap; it does not replace their semantic ownership.

## 0. Operator decision and thesis

1. **One DR64 focus is the common case.** It is not a mandate to allocate DR256 or a complete cubic tensor.
2. **The original DR mapping is immutable and versioned.** A map is an address/provenance reference, not a mutable cache index. Any changed map means a distinct sealed revision; no in-place map edit or reused ordinal silently inherits identity.
3. A 16-bit pixel stores the ordinal of **one SoA position** in a 64K-position address space. The value lives in the canonical 512-byte SoA row, not inside DR.
4. The active focus may mix up to 64 identities (Disease, Phenomenology, LOINC, etc.), 64 properties, and potentially 64 **distinct relationship selectors**. These axes are typed and are not automatically interchangeable.
5. Relationships are cheap enough to process before values. The third relationship axis may be an independent **Boolean** plane, not a third full-value copy or a third pointer cube.
6. **Load creates a lease** over the mapping and source snapshot. Lookahead prefetch prepares both subsequent referenced cache lines and the next focus. The lease is removed with the last reader after retirement; it does not change durable DR history.
7. Zero-copy is the default borrowed path. The one exception is the DR64³ foreign-key cache, scoped and conditioned in §5. Repeated 64-thought joins may profit from one gathered, compact cache of selected facets.
8. A cache records local dirtiness, reduces duplicate global destinations under declared update semantics, and hands off a small owned dirty-row batch. Repeated thought operations do not each trigger a WAL commit.
9. Optional ONNX-derived learning output may live in a **separate relationship/plasticity overlay**, as proposed evidence or candidate changes. ONNX remains an external vocabulary/serialization/admission leaf; no internal ONNX graph interpreter, model, proof or guaranteed performance follows from this note.

## 1. Geometry: do not conflate a bit tile, a pointer map and a relationship cube

| Logical view | DR64 focus | DR256 global |
| --- | ---: | ---: |
| Population pixels | 64 x 64 = 4,096 | 256 x 256 = 65,536 |
| Boolean presence tile at one bit/pixel | **512 B** | **8 KiB** |
| u16 pixel-to-SoA ordinal mapping at 2 B/pixel | **8 KiB** | **128 KiB** |
| Max distinct referenced 512-byte SoA rows | **2 MiB** | **32 MiB** |
| Dense Boolean 3-axis relation volume at 1 bit/cell | **32 KiB** (64^3) | **2 MiB** (256^3) |

- The 32 x 128-bit register view and the 64 x 64-bit DR64 Boolean view contain the same 4,096 bits: one 512-byte tile.
- A **64 x 64 u16 pointer map is 8 KiB**, sixteen times the Boolean tile. Its 16-bit ordinals each select an entire 512-byte row from up to 65,536 existing rows; they do not address arbitrary bytes.
- A **64 x 64 x 64 Boolean relation cube is 32 KiB**. If stored *in addition to* the 8 KiB pointer map, their combined allocated payload is **40 KiB**, excluding headers, occupancy/validity, version witnesses and caches. The phrasing "8 KiB becomes 32 KiB" is meaningful for the change of representation, **not** a claim that both arrays fit in 32 KiB.
- DR64^3 has **262,144 logical Boolean cells**. It is not 64^3 pointer entries, not 64^3 SoA rows, not 64 copies of a 2 MiB backing dataset, and not the separate 64-thought occupancy plane.
- DR256^3 is optional global validation/reference geometry, not an automatic execution allocation.
- The **2 MiB / 32 MiB figures are addressable distinct-row footprints**, not necessarily simultaneously resident cache bytes or transferred traffic. A focus may reference fewer distinct rows and repeated ordinals.
- Axis semantics must be declared by the caller: e.g. (identity, property, relationship selector) or (source identity, target identity, predicate). Choosing one layout must not silently turn relationship selectors into property IDs. A Boolean cell means support/presence under a declared semiring; it cannot invent Bag counts, SQL NULL, epistemic evidence or numeric strength.

## 2. Ownership and identities

| Layer / owner | Authority and lifetime |
| --- | --- |
| Canonical SoA owner | Mutable values only through its real Rust ownership boundary; native borrowed NodeRow/SoaEnvelope views. No second FrameStore, Table or per-fold mutation guard. |
| Sealed DR / RelationshipMap | Immutable global-to-local reference and relationship topology at a particular **dataset/branch/version + index seal + mapping identity**. Changing relationships creates a successor revision; it never edits the old one. |
| ThoughtMapPaletteBinding | Separate selected map palette and generation per thought. CE64 S/P/O/W bytes do not become MapIds or physical cache selectors. |
| Focus binding | Typed, authorized local aperture and source snapshot witness. One ordinal shared numerically across datasets is not evidence of identical identity. |
| Focus lease | **Ephemeral resident ownership** of a map, compatible source snapshot and any cached gathered facets. Acquired at **focus load**, not at first fold; released only when dependent readers and writer inputs are detached. Not an extra durable DR generation. |
| Native executor | Existing ndarray / mask-risc / Quack / R2IL operations on borrowed or compact aligned buffers. No new tensor VM or per-op dynamic router. |
| CycleWriter + WalSink | Independent owned dirty payloads, cycle ordering and durable publication. The writer never needs a mutable SoA-owner borrow across I/O. |
| Temporal / Thought-Git | Version-aware historical projection, provenance, candidate-vs-committed distinction, deterministic replay. No per-focus-lease durable micro-event requirement. |

**Identity key:** not merely a raw u16, focus number, local palette slot, d_head, shape or bare u64 version. A cache hit needs the relevant original map/RelationFacts durable reference, row-index seal or stable-row-ID witness, source SoA dataset revision, focus identity/aperture, palette assignment revision, projection/facet layout, operation/semiring, and access scope. Where model-specific compiled palettes are involved, include model/layer/weight/codebook identity and calibration generation. Reuse existing identity authorities: D-RMAP-1A's DurableRef and ThoughtCommitId remain *specified*, **not universally shipped Rust types**.

Immutable mapping **does not** automatically imply immutable target values. Historical replay requires the exact SoA snapshot(s), not simply ordinal equality. Index compaction that changes physical row offsets requires a stable-row-ID mapping or a new witnessed index seal; an old physical-ordinal mapping is never silently rebound.

## 3. Focus load is the lease boundary

Proposed state transitions (not an implemented Rust state machine):

- **UNLOADED -> LOADING:** resolve map version, palette, authorized focus, index seal and SoA source snapshot. Initiate lazy preparation. Coalesce concurrent requests for the *same fully witnessed key* to avoid duplicate gathers.
- **LOADING -> RESIDENT:** materialize the 8 KiB u16 map if not already present and pin its read-only source snapshot. **Acquire the focus lease here**, even before the first operation. Refuse any failed witness/scope/binding.
- **RESIDENT -> PREFETCHING_NEXT:** prepare Focus B and optionally prefetch its selected SoA facets while Focus A remains borrowed. Do not mutate the immutable mapping in place to become B.
- **RESIDENT -> ACTIVE:** execute via Rust borrows and original native kernels. A materialized foreign facet is a derived cache only, never authoritative fact storage.
- **ACTIVE -> RETIRED:** publish the next focus independently of outstanding writer I/O. Existing readers can finish on A; new readers bind B.
- **RETIRED -> EVICTED:** reclaim A's disposable map/cache allocation once the last reader is gone and any writer payload depending on A is **owned/detached**. Cache eviction does **not** delete the sealed DR or SoA history.

Use a **focus-scoped lazy initialization entry** with a reference-counted or otherwise proven reader lifetime. A global static LazyLock is appropriate for persistent method/kernel registry but **not** for an evictable focus cache. Existing Rust &[NodeRow] / &mut NodeRow exclusion does the mutation safety work: do not introduce a repeated synthetic frame guard in the fold hot path.

A cache may retain a map and source snapshot longer than strictly needed for hot reuse. The runtime residency lease is not a distributed storage lease, not the deployment writer claim, and not the 3-day idle-flush policy's *explicitly rejected* read-lease protocol. Those are separate lifecycles and cost models.

## 4. Two kinds of lookahead prefetch

1. **Focus anticipation:** predict likely next Disease / Phenomenology / LOINC map bindings; initialize Focus B's lease and begin gathering selected facets while A is active. Predictions may be wrong; they must never change query semantics or pin unlimited memory.
2. **Intra-focus memory lookahead:** the 8 KiB map gives bounded ordinals; request cache lines at SoA base + ordinal * 512 + **declared facet offset** ahead of demand. Bounds and snapshot validity are established by the focus binding, not rechecked per pixel. Software prefetch is advisory, portable fallback is a direct load.

Prefetch distance is **not fixed by the architecture**. Compare no prefetch, hardware-only, and explicit lookahead (0/4/8/16/32/64 references) across uniform random, clustered, repeated and adversarial ordinals. A random 512-byte-stride SoA access can incur cache-line latency even though the 8 KiB index fits near CPU cache. Budget cached foreign composites independently of the canonical SoA and do not claim the full 2 MiB is actually in L2.

## 5. Allowed materialization: amortized foreign-table cache for 64 thoughts

**DECISION (2026-10-24):** the DR64³ foreign-key cache is a scoped exception to the zero-copy law (`.claude/knowledge/zero-copy-lens-law.md` § "The DR64³ foreign-key cache exception"). It is not an elevation under the law's rung test: a gathered facet is a cast-reproducible projection, and the exception says so.

- **SCOPE:** foreign-key operations over a DR64 focus whose referenced rows come from several ontologies (the motivating case is about 8, e.g. Disease / Phenomenology / LOINC and others), reused across many thoughts, with relationship learning over the result feeding live-generated ONNX as the mathematical compiler substrate. Nothing outside that shape inherits the exception: a single-ontology fold, a one-shot query, or a value path that a borrowed lens already serves stays under the law.
- **BASIS:** without the cache, every thought pays scattered 512-byte-stride reads into each foreign table, repeated per ontology, per thought and per learning iteration. That work does not stay in the nanosecond fold; a gather once per focus version, reused many times, is the only economical form. This is a cost argument, made explicitly for this scope, which is why it is recorded as a decision rather than derived from the law.
- **CONDITIONS:** the cache is disposable and never authoritative; clearing it and recomputing from the exact sealed references must give the same answers; its key is the full witness of §2 (map revision, index seal, SoA source snapshot, focus, palette generation, facet layout, operation, scope); it never holds uncommitted truth; its build cost is reported separately, never folded into post-warm timings. The gather itself still uses the native kernels, and in-fold execution stays under the no-millisecond rule.
- **REVISIT WHEN:** DRF-P2 measures it. If the gather-and-reuse arm (B or C) does not beat the direct arm (A) on whole-run cost at the operator's workload, including cold build, the exception lapses for that workload and the cache is removed rather than tuned.

A pre-fold gather may be worthwhile **when the admission rules allow it** and repeated execution amortizes random-access cost. Do not silently repeal D-RMAP-1A, H14c/H14d or the no-foreign-gather firewall: the first implementation must show a typed, explicitly admitted coordinate translation, complete contributors, required authority and a separate measured cold preparation cost.

Examples for a DR64 focus of 4,096 pixels:

- Mapping: 8 KiB, retained as immutable provenance.
- Gathered 16-byte facet: **64 KiB** per materialized foreign projection.
- Three different 16-byte projections for Disease / Phenomenology / LOINC: **192 KiB** excluding keys, mapping and masks.
- Entire 512-byte rows: 2 MiB if all 4,096 destinations are distinct. **Do not default to that copy** when only a few facets are needed.
- Optional 64^3 Boolean relation cube: **32 KiB**; can be materialized, compressed or evaluated as a set of existing 64x64 slices under typed relation identity. It is not a replacement for the 8 KiB pointer map.

The **cache is expendable**. It must be legal to clear all resident foreign views and recompute from the exact sealed references. It must not store uncommitted semantic truth, silently over-approximate a focused reduction, or make model/palette weights valid only by matching shape.

Cost per thought for N reuses is roughly: local_fold + (gather + cache_build + dirty_reduction + writeback) / N, **when** the operation's effects and numerical result contract permit this amortization. For 64 thoughts, compare actual per-table reuse, not just a blanket divide-by-64 claim.

## 6. Dirty-row propagation without a second writer

At each local operation, mark changed local positions (4,096-position Boolean dirty plane = **512 B**). On flush:

1. Take dirty local positions; translate **through the exact original immutable DR mapping** into canonical SoA row ordinals and owner/dataset keys.
2. **Coalesce by canonical destination**. Multiple pixels can point to one row, and several thoughts can update different facets of that row. Coalescing cannot use an unqualified "last payload wins" policy for partial deltas; specify additive, overwrite, Boolean union, guarded replacement or explicit refusal **per declared operation**. Preserve deterministic thought/stream ordering and source roots where required.
3. Build **one complete 512-byte final row image per changed destination**, against the sealed predecessor version Vn, or a properly typed sparse patch only if its storage ABI is explicitly admitted. Unchanged rows are inherited.
4. Move detached owned payloads into **BatchWriter -> collect_casts -> CycleWriterHandle::try_submit -> DetachedCycleBatch::freeze -> WalSink::commit_cycle**. The thought loop resumes on successful queue handoff; durability is **not** asserted until the sealed outcome is known. A rejected/full queue returns ownership; an ambiguous commit reconciles using the **same frozen batch**, never newly sampled mutable SoA bytes.
5. Release a source-dependent focus lease as soon as writer inputs are genuinely **independent of that source**, not merely when a descriptor names a future live SoA read. The current CycleWriter path moves owned Vec<SweepSlot> payloads; the current LanceCycleWriter still copies frozen payloads into Arrow builders. Do not call that physical seam end-to-end zero-copy.
6. A successful artifact-backed seal creates Vn+1 and permits replay/transition application. Intent-only/no-change cycles are not forced into the WAL. Never publish in-flight Vn+1 values as if they were part of Vn.

**Correctness before throughput:** sparse row updates from an unrelated ontology must not overwrite a concurrently valid different facet merely because both resolve to one 512-byte record. Version conflicts either reconcile under an admitted deterministic reduction or refuse/rebase using the existing owner/writer rules.

## 7. Relationship-first learning as a separate future overlay

An optional future learner may treat the 64^3 relationship coordinate system as a **cheap address/support substrate**:

- Immutable DR version identifies WHERE the canonical relationships/values live; it is **not trained**.
- Boolean relation plane identifies WHICH bounded candidate triples are present or addressable. This plane is distinct from confidence/weight magnitude and from the 64-thought occupancy register.
- A separate sparse or quantized **plasticity/evidence overlay** holds learnable values and their provenance. It can be revised and versioned **without rewriting the original DR mapping**.
- ONNX/Optimum can serve as **offline or external model interchange, candidate-update stream and oracle adapter**. They are not native storage engines, not certified proof generators and not a new internal graph/interpreter. An S3-like cheap durable artifact stream is a *storage architecture hypothesis*: object storage should hydrate/version relatively large artifacts, not service per-pixel random reads or replace local mmap-backed execution.
- ndarray/BGZ/Helix/Palette256 supply existing numerical methods; approximate compiled tables remain on a separately calibrated quality frontier. ONNX candidate updates are **not certified** merely because they came from a model.
- Thought-Git may associate each candidate delta with source roots, parent commit, model/calibration identity and deterministic replay witness; approved relation changes land only as a new sealed revision, preserving the old map. No inferred automatic approval of probabilistic rules.

A discrete u16 address lookup has no generic differentiable gradient. Back-propagation, if admitted, is through **declared value/weight paths and explicit aggregation of duplicate references**, not by treating the ordinal selection itself as a smooth ONNX operation. The third relationship axis can reduce unnecessary value gathers, but that win remains to be measured.

## 8. Audit boundary

Record **durable mapping/relation revisions and admitted updates** at the existing sealed persistence/Thought-Git boundary with enough identity and data (or deterministic reconstruction inputs) to replay. A hash proves identity/integrity under its chosen mechanism; **a digest alone cannot reconstruct a discarded map**. Query-time temporal.rs can project the resulting versioned history with a concrete Strict reference. It is not itself a WAL or an event publisher.

Lease acquire, prefetch, miss, hit, retirement and eviction are **runtime metrics** rather than mandatory durable semantic commits. A watcher with always-latest notifications can drive cache invalidation but cannot replace an audit trail because slow readers can skip intermediate events.

Keep historical frames independently readable. A new focus does not mutate old revisions. Authorization/ACL revocation and cross-tenant scope changes remain an explicit admission policy for new/reused leases; a historical version is not a bypass for current access restrictions.

## 9. Explicit ownership of next actions (no new implementation authorized here)

- **D-DRF-1 / this PR:** reference semantics, geometry, lifetime, storage and falsification contract only; add no new Rust production API.
- **DRF-P1 (separate future lab PR):** two-focus lazy-load / lease / prefetch / last-reader-eviction probe using existing owner/snapshot/borrow seams. Negative race and stale-map fixtures first.
- **DRF-P2 (separate future lab PR):** 64-thought foreign-table A/B/C comparison (direct+prefetch versus gather+reuse versus gather+reuse+dirty writeback). Include cold build and realistic memory. The reported winner is physical, never semantic.
- **DRF-P3 (separate future lab PR):** admitted relationship cube/slices and typed dirty-row reduction oracle. Demonstrate mixed ontology domain seals and no accidental Bag/NULL/evidence loss.
- **DRF-P4 (OPTIONAL, separate research):** plasticity overlay and ONNX serialized candidate stream with independent oracle, provenance and failure gates. No change to native-first architecture.

**No EXO-50 message, coding-agent prompt, new DR field, map registrar, CE64 bit redefinition or production cache API is part of this PR.** Use the landed discussion/reference as the source when coordination is explicitly requested.

## 10. Independent falsifiers and measurement matrix

Read [D-DRF-1 falsification preregistration](d-dr64-focus-cache-falsifiers-v1.md). Required gates: geometry; map revision replay and compaction; focus eviction under concurrent readers; lookahead correctness; legal pre-fold gather; 64-thought total economics; duplicate-target dirty reduction; queued/asynchronous failure and reconciliation; ONNX candidate/proof separation.

## Source crosswalk

- [D-RMAP-1A ownership and versioning](d-rmap-separate-tables-thought-palettes-v1.md), especially §§2, 6a, 7 and 8; [D-RMAP-1](d-rmap-implicit-relationship-register-v1.md).
- [D-NFC dependency correction](d-native-first-dependency-correction-v1.md); [D-NFR native representation](d-native-repr-contract-v1.md). [#1755](https://github.com/AdaWorldAPI/lance-graph/pull/1755) is the separate SoA borrow probe (merged 2026-10-24); do not copy its withdrawn FrameStore design.
- [DR64^3 holographic geometry](d-dr3-holo-onnx-s3a-s3c-v1.md), [physical comparison/reuse cost](d-onnx-relalg-s3d-physical-shootout-v1.md).
- Existing source: [NodeRow/SoaEnvelope](../../crates/lance-graph-contract/src/soa_envelope.rs), [temporal.rs](../../crates/lance-graph-planner/src/temporal.rs), [BatchWriter](../../crates/lance-graph-planner/src/batch_writer.rs), [persistence sink](../../crates/lance-graph-planner/src/persist_sink.rs), [CycleWriter](../../crates/lance-graph-supervisor/src/cycle_writer.rs), [LanceCycleWriter](../../crates/lance-graph/src/graph/cycle_sink.rs).
- [Existing Palette256 plasticity probe](../board/entries/2026-10-22-pal256-hebbian-overlay-plasticity.md), [NARS/Bayes evidence boundary](../board/entries/2026-10-22-pal256-nars-bayes-evidence-contract.md), [S3 hydration doctrine](../board/entries/2026-08-06-e-object-store-hydrates-it-does-not-store-1.md).
