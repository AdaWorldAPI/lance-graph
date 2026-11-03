# D-DRISA-0: DR remote async register / Abakus ALU / cognitive shader driver

> **2026-11-04. Status: PRE-REGISTERED ARCHITECTURE, DOCS ONLY.** No new runtime, RPC endpoint, public ABI, wire protocol, GPU shader, source map, F### word, ISA implementation, or speedup is authorized or claimed. **Operator thesis:** replace repetitive hash-table-like lookups and transported/materialized payloads with a sealed DR register-address basis, a typed ClassView/DT64 method lens, existing Quack/R2IL/mask-risc instructions and ndarray/Abakus ALU words. Submit work asynchronously to the source-owning executor and return only the necessary result.
>
> This adds a **new hypothesis and a remote-execution boundary** to merged D-DRAM-0 (#1949), D-GLOBAL-WHERE P2 (#1950), and D-DRAM-P3 (#1951). It does **not** alter their evidence or lift the open owner decisions.

**Tracking:** [EXO-153](https://linear.app/exo-red/issue/EXO-153); [EXO-50](https://linear.app/exo-red/issue/EXO-50) is append-only handoff, not approval. [Staged-PR charter](d-drisa-staged-pr-charter-v1.md); [falsifiers](d-drisa-falsifier-matrix-v1.md); [board](../board/entries/2026-11-04-d-drisa-0-remote-register-contract.md).

## 1. The sharp thesis and its boundaries

A normal remote database interaction often fetches records or columns and executes client-side work. **D-DRISA sends a typed program and witnessed coordinate context to the data-local executor.** The executor binds canonical borrowed value lanes if needed and runs the existing masking algebra there. DR coordinates remain the *virtual register-address space*. A compact scalar, witnessed mask or candidate relation returns to the caller. Abakus is a candidate **bit-sliced arithmetic** realization of certain ALU operations, **not** the key/identity map itself.

The research question is whether this yields lower **end-to-end** latency, payload movement and repeated preparation without loss of semantics, isolation or zero-copy local borrow laws. It is **not** already a proven distributed cognitive processor.

**Not:** a new hash table; a copied facet cache; a new mask store; a GPU backend; arbitrary executable shaders; a second Quack/R2IL interpreter; a DR→Lance global row-ID table; a remote Rust pointer; a magic claim of zero-copy network I/O.

```text
REMOTE CALLER                           DATA-OWNING EXECUTOR
  intent + method + result demand        |
     |                                   |  verify source/frame/schema/scope
     +-- versioned descriptor ---------->|  bind existing DR frame/tiles
     |                                   |    (MAP-only may finish here)
     |                                   |  resolve ClassView demanded fields
     |                                   |  optional admitted locator receipt
     |                                   |  local canonical SoA BORROW/read
     |                                   |  Quack/RelAlg or R2IL → mask-risc
     |                                   |  ndarray::simd + optional Abakus
     |<------- compact result -----------|  COUNT/EXISTS/mask/candidate/refusal
     |         async completion          |  release owned scratch/source lease
```

**Shader** means a small, bounded read-only data-local program over one witnessed register window, not graphics shader code or unbounded distributed execution.

## 2. Owners: the already-existing ISA is unusually close

| Concept | Existing actual owner / status | Remote boundary required |
| --- | --- | --- |
| Identity / coordinate register | `lance-graph-dt64/src/dr.rs`: `MapTable`, `LocalMap`, `Frame`, `Guards`, `Axis::for_survivors`. SHIPPED. Identity-only; no values | Opaque, versioned register *name*, never an exported pointer or second identity authority |
| Typed class and behavior | `lance-graph-dt64/src/lib.rs`: immutable `MethodBinding`, `MethodId`, `SigRev`, ON×WHEN×DO, `ValueDemand`. SHIPPED store | Source/field permission and read-only remote admission; DT64 store alone is not authorization |
| Property scope | `lance-graph-contract/src/class_view.rs`: `ClassView` / `FieldMask` / `WideFieldMask`. SHIPPED | Resolve property IRI to physical source column under schema/source seal, no silent widening |
| Algebra/compiler | Quack/RelAlg; R2IL and OGAR body ownership. SHIPPED partial execution pathways | Executor validates allowlisted described program; do not mint new F### or own OGAR body |
| Virtual mask register | `mask-risc/src/ir.rs` `Operand::{Plane,Scratch}`; `Planes`, `LaneRef::{I32,U32,U64,Strided}`, `Foreign`. SHIPPED LOCAL | Safe remote *descriptor*, not transported `&[T]` / `StridedRef` memory pointer |
| Execution/ISA | `mask-risc/src/exec.rs`, caller-owned reusable scratch, `execute_into` delegating to ndarray. SHIPPED | An admission-aware dispatch/finish boundary, not a second op loop |
| Arithmetic ALU | ndarray `half_add_mask` / `full_add_mask` (#402) and native Boolean/comparison words. SHIPPED | Capability/predicate-signature check; no public `Abacus<BITS>` or `U24x5` claimed (#1951 census) |
| Source I/O | #1633 and #1950: `SealedFacetLane` / `OrderedLaneWitness`, `RowAddrMask`, `Scanner::project`. LAB | D4 single DR→Lance `_rowid` authority and S-INGRESS measured copy remain BLOCKED |
| Cached *location* receipt | D-DRAM-0 `2; #1951: **no production owner or lease found** | EXO-46/89/151 gate; never mutate `LocalMap::globals` |
| Grammar cognitive consumer | #1948 co-addressed `head_frame`, two read-only methods. LAB | Producer is derived semantic lane, not a locator; EXO-46/140 admission and source revision |
| Remote scheduler / transport | **NO IMPLEMENTATION ESTABLISHED BY THIS REVIEW** | Typed descriptor, async state, result lifecycle, security and locality; independent lab first |

The 5×24 anecdote is now settled more narrowly: #1951 located examples-only `U24x5` microtile packing; no shipped `ndarray::simd::5x24::*` family. Never import lab example code as an alleged production ISA.

## 3. Four meanings of zero-copy

**(A) LOCAL-BORROW: established default.** Resident source-owning executor borrows canonical `&[T]` or `LaneRef::Strided` with a live source lease. mask-risc owns no payload bytes. Reusable scratch is *mask scratch*, not copied SoA.

**(B) NO-INTERMEDIATE-PAYLOAD: the target.** Between remote caller and data-local executor pass only an admitted instruction descriptor and a compact result, not row/object payloads. This can yield a strong reduction in transferred bytes, **even if network request/result serialization exists**.

**(C) PHYSICAL/NETWORK ZERO-COPY: unproven.** No assumed RDMA, pinned pages, NIC ownership, device access, mmap sharing, zero-copy compression, or guaranteed zero-copy transport. Arrow C Data Interface describes **same-process** sharing, and explicitly excludes distinct processes; Arrow Flight uses RPC/IPC. No transport chosen in this plan.

**(D) STORAGE/INGRESS COPIES: explicitly accounted.** Lance page reads, Arrow batch creation, `RowAddrTreeMap` and `S_INGRESS_COPY` are separate. A source may not be resident; selective reads are allowed but costed. D4 remains `BLOCKED_D4` until one owner signs the attested mapping.

**Non-negotiable:** A faster copied cache does not authorize changing BORROW into a generic GATHER physical plan. D-DRF-1 `5's exceptional foreign-cache admission remains exceptional.

## 4. Logical register addressing and lifetime

Register naming is a conceptual descriptor, **not an ABI definition**:

```text
RegisterWindowRef:
  coordinate_source: map_identity + Frame.issue + tile-range + route_witness
  data_source: dataset identity + branch/version + index/rowaddr seal
  class_basis: ClassId + schema revision + field-basis digest
  method: MethodId + SigRev + guard digest + result kind
  scope: authenticated principal + permission/scope + expiry + tenant
  semantics: Boolean/Bag + TwoValued/Kleene + numeric/sentinel rules
  program: validated existing-IR digest + bounded operand description
```

- Global ID is not a local DR ordinal; a DR ordinal is not a Lance `_rowid`; field position is not a physical column ordinal. Mixed dimensions receive typed refusals.
- A source-location memo is a **disposable, sparse receipt under an existing binding**. It can be forgotten with no loss of truth. It never becomes the new relation owner or is appended to the identity list.
- A binding may retain coordinates across value-only changes while **predicate planes derived from changed values** become stale. Separate versions: coordinate frame, source snapshot, field schema, permission scope, derived truth, program vocabulary.
- Reads from an old frame are lawful only with the corresponding historical source version and scope witness; otherwise refuse.
- Retired/overwritten source, compaction, stable-row-id change, index-rebuild, permission revocation, focus-slot reuse and tenant switch must invalidate or refuse stale physical descriptors.
- Live local references must never outlive their Arrow/source owner, pin/guard or release callback. Raw addresses never cross the RPC boundary.
- Result masks are tagged with their coordinate universe; a downstream method cannot reuse them on another frame solely because bit lengths agree.
- ClassView may select fewer fields. Dynamic **scope** does not license reinterpretation of presence/sign bits, NULL/unknown or codebook semantics.

## 5. Execution request / result sketch (not public wire structs)

```text
ExecuteDescriptorV0:
  protocol_schema_version, request_id, operation_digest
  target_dataset/branch/version/index_seal
  frame/map/tile_range + source_class/field_schema revisions
  method_id/sig_rev/guard_digest + fields_needed
  program_ref (existing Quack/R2IL/mask-risc semantics; locally validated)
  authentication/authorization claim (server-verifiable, never caller-declared)
  Boolean|Bag, TwoValued|Kleene, null/absence, signed/overflow rules
  result_kind (Scalar | MaskReceipt | RelationReceipt | Refusal)
  read_only = true
  deadline, quotas (max_tiles, max_slots, max_ops, max_memory, max_result_bytes)
  optional locality/prefetch hint (ADVISORY, no authority)

CompleteDescriptorV0:
  request_id + operation_digest + effective source/frame/scope witnesses
  status (Completed | Refused | Cancelled | Indeterminate)
  scalar or bounded result + release token if returned handle
  measured stage costs, bytes, allocations, source-read mode
```

The wire encoding, capability signatures, authentication, principal binding, revocation policy, version handshake, replay protection and opaque-handle issuance are **not designed or authorized** here. Explicitly require an owner decision and a threat-model review before an open TCP endpoint. The source-owning executor **validates the program and authorization**; a program digest and a caller-supplied scope string are not sufficient.

### 5.1 Async state and resource rules

```text
SUBMITTED → ADMISSION → BOUND → QUEUED → RUNNING → COMPLETED
                |        |        |         |
              REFUSED  STALE    CANCEL    EXPIRED / INDETERMINATE
                                      \
                                       RELEASED after safe completion
```

- Source snapshot/frame/scope must remain valid while queued *and* running; either pin a lawful immutable source snapshot or re-check and refuse.
- Cancellation request is not rollback. An unknown network outcome remains **Indeterminate**, never logical FALSE or empty mask.
- Retry is allowed only for read-only deterministic operation against exactly the same effective source/frame/scope. No exactly-once guarantee.
- Bound queue depth, per-request operation count, scratch bytes, result bytes and in-flight leases. Denial of service and unbounded fan-out are refusals, not accepted workload.
- No effectful programs in stages 0–6, no automatic hydration, no undisclosed callback or script execution.
- Connection loss must not leak source lease or result receipt. TTL, explicit release and race tests required before promotion.
- Cross-owner joins are **not** automatically data-local; either require a separately admitted and costed data movement/join plan, or refuse.
- A true k-of-n must count satisfied predicates *per identity*, not multiply Bag derivations; SQL NULL/UNKNOWN cannot disappear.

## 6. Abakus ALU / adaptive cognitive lens

**Existing opcode words**: `ndarray::simd::half_add_mask`, `full_add_mask`; native `mask_and`, `mask_or`, `ternary_match_u32_to_mask`. Abakus is a bit-sliced **arithmetic** technique on identity-aligned significance planes. A carry moves to a higher significance *plane for the same identity*, not to a neighbor identity or a new row.

**First positive consumer**: over one sealed population and predicates `P0..P4` compute `K3 = (P0 + P1 + P2 + P3 + P4 >= 3)` using existing half/full-add words, compare against an independent scalar-per-identity oracle and a native Boolean network. Verify all 64 tail/one-hot/alternating/random patterns and count/EXISTS. Benchmark n=1,2,4,5,16,24 with k=0,1,half,n,n+1. Report cold plane preparation and hot execution separately, with disable-carry tests.

**Negative consumer:** two-mask AND must stay a native Boolean mask word; Abakus cannot justify adding operations. A packed unsigned-u8 comparison inside u32 is an **independent missing facade operation** identified by #1950, not proof that Abakus's carry adder should implement it. `U24x5` is lab packing, not the needed ordered-byte ISA.

**Adaptive means:** lower only the admitted demanded fields from ClassView/WideFieldMask, select the existing word/facade compatible with the semantic contract, reuse frame-aligned masks while valid; do not reinterpret type/overflow/sentinel or bypass ndarray ownership. Numeric approximate U9/Palette512 and exact Boolean/Bag truth remain separate.

**Cognitive shader** is a constrained, read-only method program over one leased DR population; it is not magical learned inference. #1948 grammar's `head_frame` is a **derived semantic lane**, not a memoized source locator. XLOG/NAL callers use the existing semantic polyfill boundary, no new XLOG runtime or recursion admitted by a transport adapter. THOUGHT hypotheses remain speculative; WORLD writes need the existing hydration/Rubicon authority.

## 7. Exact baselines, not marketing extrapolation

For each same-source, same-predicate, same-scope, same-result fixture:

| Arm | Description | What it proves |
| --- | --- | --- |
| B0 | data-local borrowed planes, local call | local executor floor |
| B1 | remote fetch rows/columns, evaluate locally | actual transfer/materialization baseline |
| B2 | remote issue, data-local borrowed execution, scalar/mask result | remote no-intermediate-payload hypothesis |
| B3 | B2 + reused DR frame/selection binding | reuse without repeated routing |
| B4 | B3 + optional, validated **locator** receipt | address memoization, **not** copied value cache |
| B5 | B4 + prefetch lookahead (including wrong predictions) | prefetch economics |
| B6 | B2/B4 + justified Abakus k-of-n ALU | arithmetic physical-plan benefit |

All arms account for: `C_total = C_request_serialization + C_transport + C_auth + C_admission + C_frame_bind + C_route + C_locator + C_source_io + C_mask_ingress + C_prefetch + C_fold + C_result_encoding + C_transport_return + C_release`. Count network bytes, disk/page-cache bytes, source allocations, result/receipt bytes, request count, queue delays, peak RSS, p50/p95/p99, CPU cycles. Compare Q=1,2,8,16,64,256, 64/256/multi-tile, scatter/cluster, 0.1%,1%,10%,50%,95%, k=1/2/4/5/16/24 and 1M/10M where feasible. Do not use a 1M-COUNT win to claim remote performance.

**Critical #1950 correction:** `Binding::prepare` caches `gather_lane` copied `Vec<Vec<i32>>` as well as CSR and masks. Its Q≥2 timing is **bundled prepared-binding amortization**, not independent proof of locator memoization. The 50%-selectivity borrowed path being 2.3–2.5× slower remains a required falsifier. #1951's `range_count * 4096` is a **fragmentation proxy**, NOT the number of distinct pages touched, and the 4096 constant is a policy pin, not a calibrated general planner law.

Keep <1000µs warm-in-fold gate **where already defined**; report network and IO cost separately rather than incorrectly claiming remote requests all meet it. Only measured whole-query benefit plus exact parity can justify an optimization decision.

## 8. Gate map, staged PR sequence

The precise PR charter and independent falsifier inputs are in the companion plans. This is the intended **order**, not automatic authorization:

```text
D-DRISA-0 DOCS/BOARD/LINEAR (this PR)
  ├── 1: in-process borrowed-register equivalence (LAB)
  ├── 2: async loopback lifecycle / refusal (LAB, security-read first)
  ├── 3: Abakus threshold consumer (LAB, ndarray owner)
  ├── 4: two-process data-local transport comparison (LAB, transport/security gate)
  ├── 5: admitted source-location receipt + prefetch (EXO-46/89/151 BLOCKED)
  ├── 6: DR→Lance single authority / mask handoff (D4/S-INGRESS BLOCKED)
  ├── 7: grammar/XLOG/NAL cognitive read-only consumers (EXO-140/73/78 GATED)
  └── 8: fault/performance evidence register and promotion decision (NO AUTO MERGE)
```

Stages 1 and 3 are parallelizable **only as lab code** with pinned oracles; stage 2 does not grant permission for a network listener. The existing #1951 P3 plan continues unchanged; D-DRISA stages are a separate research direction. A failed stage reports `REFUSED / DEFERRED` and does not require adding a new engine.

## 9. Scientific and security falsification

See the [independent test matrix](d-drisa-falsifier-matrix-v1.md) for exact R1–R16 (register/lease), A1–A8 (ALU), N1–N14 (remote/fault), C1–C10 (cognition/semantics), P1–P9 (cost) fixtures, oracles, deliberate disable conditions, measurements and stopping points. Each result is graded VERIFIED-CODE, MEASURED-LAB, PROPOSED, GATED or UNVERIFIED.

**Unconditional STOP**: new global→row-ID authority; wire-transmitted raw pointer; copy of canonical values silently cached as normal path; cross-scope/tenant leak; stale frame silently accepted; default value materialization; claiming U24x5 shipped; new F###/opcode ownership; setting UNKNOWN=FALSE except under an admitted SQL WHERE; THOUGHT→WORLD promotion; retry changing source version; auto merge; rewriting transport into Quack.

## 10. Source and published-reference inventory

- [DT64 DR map / frame](../../crates/lance-graph-dt64/src/dr.rs), [DT64 bindings](../../crates/lance-graph-dt64/src/lib.rs), [mask-risc IR](../../crates/lance-graph-mask-risc/src/ir.rs), [mask-risc execution](../../crates/lance-graph-mask-risc/src/exec.rs), [R2IL dialect](../../crates/lance-graph-r2il/src/lib.rs), [ClassView](../../crates/lance-graph-contract/src/class_view.rs), [ordered-lane witness](../../crates/lance-graph-contract/src/ordered_lane.rs).
- [#1948 grammar](https://github.com/AdaWorldAPI/lance-graph/pull/1948), [#1949 D-DRAM plan](https://github.com/AdaWorldAPI/lance-graph/pull/1949), [#1950 P2 measured labs](https://github.com/AdaWorldAPI/lance-graph/pull/1950), [#1951 P3 census](https://github.com/AdaWorldAPI/lance-graph/pull/1951), [ndarray #402](https://github.com/AdaWorldAPI/ndarray/pull/402).
- [D-DRAM-0](d-dram-abacus-address-memoization-v1.md), [D-DRAM-P3](d-dram-p3-integrated-v1.md), [D-DRF-1](d-dr64-focus-lease-foreign-cache-v1.md), [D-RMAP-1A](d-rmap-separate-tables-thought-palettes-v1.md), [D-DRAV-1](d-dr-address-first-value-demand-v1.md).
- [EXO-153](https://linear.app/exo-red/issue/EXO-153); [EXO-50 bus](https://linear.app/exo-red/issue/EXO-50); [EXO-46](https://linear.app/exo-red/issue/EXO-46); [EXO-89](https://linear.app/exo-red/issue/EXO-89); [EXO-151](https://linear.app/exo-red/issue/EXO-151); [EXO-121](https://linear.app/exo-red/issue/EXO-121); [EXO-140](https://linear.app/exo-red/issue/EXO-140); [EXO-114](https://linear.app/exo-red/issue/EXO-114); [EXO-73](https://linear.app/exo-red/issue/EXO-73); [EXO-78](https://linear.app/exo-red/issue/EXO-78).
- [Apache Arrow C Data Interface](https://arrow.apache.org/docs/format/CDataInterface.html): same-process zero-copy, release callbacks, **not** interprocess sharing. [Arrow Flight](https://arrow.apache.org/docs/format/Flight.html): IPC/gRPC, streaming and custom actions as **possible transport** only. [Arrow C Device Data Interface](https://arrow.apache.org/docs/format/CDeviceDataInterface.html): **experimental**, no device guarantee.
