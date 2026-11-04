# D-DRIVER-ZC-0: Virtual zero-copy access driver + address-preserving bus algebra

> **2026-11-04 · PROPOSAL / PRE-REGISTRATION / DOCS ONLY.** No runtime, public driver API, format, compiler rewrite, socket, second source identity authority, SIMD opcode, benchmark or theorem proof is shipped here. This document refines, **without superseding**, [D-DRISA-0](d-drisa-remote-register-shader-v1.md) and the operator's [zero-copy lens law](../knowledge/zero-copy-lens-law.md). D-DRISA-1 (#1955) and D-DRISA-3 (#1956) are now **merged lab evidence**, not future stages.
>
> **Operator north star:** Apply the *ndarray SIMD polyfill architectural move* to **data access**. Instead of trying to audit roughly 17,000 independently written calls, require admitted consumers to pass through **one narrow, typed virtual driver/lens boundary**. A single witnessed DR address supplies local tile or admitted foreign/global-to-local access, the driver borrows canonical SoA data once under a source lease, and existing algebra reads the same address through logical XY/XZ/XX/YY/ZZ buses without duplicating value columns or constructing row objects. Remote calls request *data-local execution*, not remote Rust references.
>
> **Tracking:** [EXO-154](https://linear.app/exo-red/issue/EXO-154) owns this narrower zero-copy/bus research program; [EXO-153](https://linear.app/exo-red/issue/EXO-153) owns its broader D-DRISA execution program; [EXO-50](https://linear.app/exo-red/issue/EXO-50) is append-only message coordination. [Staged PR charter](d-driver-zc-staged-pr-charter-v1.md), [falsifiers](d-driver-zc-falsifier-matrix-v1.md), [board entry](../board/entries/2026-11-04-d-driver-zc-0-virtual-bus-contract.md).

## 1. New architectural ruling proposed, NOT granted

**Driver = address-preserving execution membrane, not an ISA or new ALU.** All admitted read paths should use one capability-bound contract with five conceptual steps:

1. **BIND:** admit a DR Frame / source incarnation / tile-local position and class/scope witness. A single costly global→local bind is amortized *outside the hot fold*.
2. **PROJECT:** produce typed virtual *views* onto demanded canonical SoA lanes/relationships; never create a second row/column owner in the admitted hot path.
3. **STREAM:** consume the same source address across tile-sized operations; preserve masks' coordinate universe; reuse the bound frame without per-method foreign lookup.
4. **EXECUTE:** existing Quack/RelAlg/R2IL → mask-risc → ndarray SIMD. Abakus carry words are an optional physical realization, **not** mandatory ALU.
5. **RELEASE:** close source, frame, scope and scratch lifetimes; return scalar/compact witnessed mask or explicit admitted egress.

This is **a proposed facade and type/lifetime discipline**, not a statement that a new `Driver` trait already exists. Names below are *conceptual operation names*, not Rust/API signatures.

~~~text
logical caller (DT64/ClassView/Quack, demanded fields, scope)
   |
   v
single logical address: (source, DR Frame, tile, local, facet, layout)
   |
   v
+--------------------------------------------------------+
| VIRTUAL ACCESS DRIVER  [research contract, not code]  |
| BIND -> PROJECT -> STREAM -> EXECUTE -> RELEASE        |
| - one coordinate/layout authority, a validated lease  |
| - zero implicit duplicate canonical payload           |
| - no row-object/tuple construction within hot fold    |
| - local or data-owner remote execution location       |
+--------------------------------------------------------+
     |             |                   |
   XY/XZ         XX/YY/ZZ           existing ALU
   lens           lens               mask-risc/ndarray
     \             |                   /
       same canonical borrowed SoA tile/source owner
                          |
           scalar / mask / result receipt / refusal
~~~

### 1.1 What the comparison with ndarray actually says

| SIMD polyfill architecture | Proposed zero-copy driver architecture |
| --- | --- |
| One mathematical operation expressed once | One logical witnessed address and requested operation expressed once |
| ndarray::simd hides scalar/AVX2/AVX-512/NEON lowering | Driver hides admitted local, tiled and data-local foreign source-owner access |
| Per-backend kernels, same vector semantics | Per-source binding implementation, same relationship/value semantics |
| SIMD fidelity and performance verified by test oracles | Coordinate, source-layout, lifetime, no-owned-payload and bus-composition verified by independent oracles |
| Callers cannot choose a private ISA implementation by accident when using facade | **Migrated** callers cannot choose another payload ownership path by accident when only the sealed capability is exposed |

**Important limitation:** A facade does not magically make preexisting public raw accessors inaccessible, nor can ordinary Rust types prevent all possible copying of public scalar values. Enforce the law at an owned, testable boundary, phase migration, and do not claim that 17,000 historical callers are already proven clean.

## 2. Non-negotiable meaning of "zero copy"

The existing [zero-copy lens law](../knowledge/zero-copy-lens-law.md) says the lens **is** the read floor, the array is already a projection, and there are no implicit payload-copy escape hatches. This program respects that operator ruling.

### 2.1 Three proof scopes, never conflated

| Claim | What would prove it | What it does NOT mean |
| --- | --- | --- |
| **ZC-A: resident borrowed execution** | Source-owner borrows canonical buffer; no new owned value/record column is made on admitted hot path; alias/pointer + instrumentation evidence | No scratch/result bits; no CPU loads/stores; no cache misses |
| **ZC-B: virtual layout** | Multiple typed bus projections share same source buffer/address binding; no physical XY/XZ/XX/YY/ZZ arrays or row object replicas | All pairwise results for arbitrary N are O(1) |
| **ZC-C: remote data-local operation** | Remote client sends typed descriptor; source-owning worker resolves and executes; no payload values sent to caller except explicit authorized egress | RDMA, network zero-copy, universal no disk I/O, low latency by definition |

**Accounted boundaries, not hidden exceptions:** storage page fetch / decompression / Arrow batch, Lance `RowAddrTreeMap` S-INGRESS mask translation, network framing, and explicitly demanded results may move/copy bytes. They are **not** reclassified as "zero-copy hot execution." A remote request that must return an entire result population necessarily transmits an output proportional to its result size. For storage-backed "Thinking Hard Drive" claims, count all page reads, cache fills, decompression, ingestion copies and results.

### 2.2 Enforceable guarantee vs slogan

A usable guarantee requires all of:
- The source-locating operation resolves to **one** canonical owner, with a lease valid for the entire fold and a layout witness minted/checked **by that owner**.
- Only a **sealed, typed view/capability** for the requested fields and scope crosses from the access membrane to execution. Runtime forbids indiscriminate full-column gather.
- Binds, lease acquisitions and foreign hops are **explicitly accounted**, so they cannot hide inside an apparently pure arithmetic fold.
- A bounded borrowed view can be used immediately or through an executor callback without allowing an owned dataset replica by default. Internal source storage may be packed/strided; neither mandates a copy.
- Existing ndarray/mask-risc executes computation. The driver **does not** implement a new elementwise loop, hash index, relation store, serializer or ALU.
- All **migrated reachable** production paths go through the membrane; bypasses fail compile or a required CI policy. Unmigrated legacy paths are marked EXCLUDED/NOT-YET-GUARANTEED, never silently certified.

**False proof:** Counting only allocator calls or testing pointer equality on one demonstration is insufficient to prove an entire call graph has no copies. A stack copy, source decoder, output serializer, a call to `to_vec()` in another module, and a hidden second cache are independent failure modes.

## 3. The virtual bus: five requested views, one physical owner

Let a canonical **same-identity** SoA, under a sealed layout witness `W`, expose typed x,y,z lanes: `S=(X[],Y[],Z[])`. For a fixed, already-bound ordinal `i`:

```text
  Bus_XX(i) := (read X[i], read X[i])
  Bus_XY(i) := (read X[i], read Y[i])
  Bus_XZ(i) := (read X[i], read Z[i])
  Bus_YY(i) := (read Y[i], read Y[i])
  Bus_ZZ(i) := (read Z[i], read Z[i])
  # YZ optional symmetry control; not required as a sixth stored bus
```

These are **view descriptors with typed operand roles**, not automatically numeric products or persistent arrays. The higher-level meaning of XX, XY etc may be a relation, projection, orientation, quadratic form, variance component, or grammar edge; that interpretation belongs to its *existing semantic owner*, not to this driver. Same-name pair XX may require only **one physical lane load** reused in two operand roles.

For a requested operator `F` and a bounded population mask `M_W`:

\[
\operatorname{Fold}_{W,M}(F,\text{Bus}_{ab}) =
\bigoplus_{i \in M_W} F(S_a[i], S_b[i]).
\]

The core implementation can use two borrowed operand lanes and one existing mask-reduction operator; it **does not** have to build `Vec<(a,b)>` or `Vec<PSCustomObject>` before folding.

**Foreign XY is a different law.** If `Y` refers to another identity, dataset, source version or a relationship edge, this is **not** `Y[i]` by definition. Its global→local hop must be admitted under EXO-46/140, bound and witnessed once, potentially executed at the source owner; arbitrary cross-source Cartesian products are not free.

### 3.1 Compositional lens fusion

A strong target is to run a chain such as `XY → XZ → ZZ` while retaining its address basis:

- **Static semantic composition:** Quack/DT64 lowers operator composition into one typed program and one source/frame lease where lawful. Virtual outputs can remain scratch masks/planes rather than owned rows.
- **Repeated shared-address input:** X from XY and XZ need not be read from a different store or copied into separate value vectors; the same borrowed X lane participates in both.
- **Fusable program:** keep intermediate predicates in existing mask-risc scratch under its epoch. Do **not** claim a scratch slot survives across calls; EXO-153 M3 is open.
- **Barrier:** if the next step needs a different global ID, joins different owners, changes frame/scope, or demands an actual returned row object, it must execute an admitted bind/egress transition or refuse.

### 3.2 Complexity: the precise O(1) claim

A *constant-size lens descriptor* and **fixed number of resident SoA field accesses at an already-resolved local ordinal** can be O(1) in population size. This excludes: finding the global ID in a set of tiles, storage-page reads, decompressing an on-disk representation, runtime fan-out, computing all pairwise relations or emitting N outputs. A query over K selected identities generally needs at least Ω(K) work or a separately proven aggregate shortcut.

The virtual buses avoid **physical representation expansion**, not the laws of physics or information-theoretic output bounds. The strong test is no duplicate canonical storage and address-preserving composition, with exact oracle parity.

## 4. Small driver contract, not a new ISA

### 4.1 Conceptual operations (not promised Rust symbols)

| Operation | Required input | Result / ownership | Failure |
| --- | --- | --- | --- |
| `bind` | DR frame/map version, source-incarnation and layout witness, scope, selected IDs or co-addressed mask | One bounded, source-valid binding; no copied values | stale frame, wrong source, layout mismatch, unauthorized scope |
| `project` | ClassView basis and demanded field positions/roles `(a,b)` | borrowed, typed local view for the leased source; no new value owner | absent/unknown/ambiguous/unauthorized field |
| `stream` | already-bound population/relationship mask, tile/field demands | caller-owned scratch and source-borrowed lanes, tile-by-tile | dimension mismatch, expired lease, unsupported foreign map |
| `execute` | approved Quack/RelAlg/R2IL/mask-risc plan and authorized terminal | existing ndarray execution, no private engine | unsupported opcode, invalid truth mode, overflow/NULL policy |
| `release` | lease/owner token, scratch scope, optional result receipt | no outstanding reads beyond owner lifetime | stale or reused handle refused |

There must be exactly one **logical** caller contract; local and remote are **two admissible implementations** under the same semantics, not two competing APIs.

- **Local:** expose an in-process borrowed view with a lifetime no longer than the source snapshot. Use `LaneRef::Strided` when appropriate, no forced contiguous value gathering.
- **Remote:** never transmit a raw pointer or Rust reference. A typed descriptor addresses the source owner, which validates source/frame/scope and borrows locally before evaluation. Cross-process command and result messages are costed, not falsely called zero-copy.
- **Storage:** the source owner may need actual disk I/O. Full DR→Lance integration remains `BLOCKED_D4` and `S_INGRESS_COPY` under EXO-114/#1633. A page is not automatically the same dimension as a DR local tile.
- **Legacy:** a `PSCustomObject`-like consumer object is a projection capability / lazy *view*, not a bag of copied properties. If a consumer explicitly asks for a value object, that is result egress requiring separate intent/owner.

### 4.2 Witness tuple (design only)

```text
W = (
  canonical_source_identity,
  source_incarnation + value_revision + storage/index_seal,
  DR_map_identity + Frame.issue/route_version + tile/ordinal_basis,
  source-owned layout/field-schema/facet digest,
  class_id + ClassView.field_basis + demanded fields,
  scope/permission/principal/tenant + expiry,
  selected relation_semiring (B / N), truth_mode, method_sig_rev,
  driver-capability version + execution owner
)
```

A local FNV TileMap digest helps compare layout; **it is not collision-resistant remote authentication**. A remote descriptor sent by a client is untrusted until the source-owning executor checks its own versioned snapshot and authorization. A value-only write may leave coordinate validity unchanged while invalidating value-derived predicates. A different SoA incarnation can reset its local numeric revision and still be stale. Historical frame replay is valid for old source values **only if the matching historical source snapshot exists**.

## 5. New "Thinking Hard Drive" claim, framed for proof instead of marketing

**Proposed theorem schema, NOT a mathematical theorem or a shipped result**:

Given a finite witnessed source with one canonical SoA layout, a finite bounded set of pure admitted operations, valid source/frame/scope leases, and a driver whose implementation borrows the source lanes, an independently specified oracle and a virtual bus driver evaluate the *same relation* while **the resident executor constructs no secondary owned canonical value/row representation**.

The proof has two independent obligations:

1. **Extensional parity:** identical results, including Boolean vs Bag, NULL/UNKNOWN, absent vs zero, tail bits, multi-tile and foreign/source refusals. Same address universe and timestamps.
2. **Representation bound:** resident execution's *newly owned canonical payload bytes* = **0**; scratch/selection/result memory is separately counted, not prohibited. Prove no unauthorized value reconstruction inside the admitted graph and measure actual I/O and network boundaries separately.

**Counterexample:** A graph that computes `XY` from one frame and `XZ` from a different population, then silently uses the same local ordinal, may allocate zero bytes **yet be wrong**. Zero copies alone do not prove valid cognition. Likewise a physical source-page read can dominate latency despite zero payload copies. A theorem can only cover the declared execution path and cost model.

This is the precise research connection to the "thinking hard drive": persistent data should supply *the operands of computation in place* (when resident and lawful), rather than first becoming a second reconstructed object world. The disk is not doing CPU arithmetic; data-local computation still uses ordinary processors, memory and storage I/O.

## 6. Evidence, current status and relation to existing programs

| Earlier source | Verified / measured evidence | What D-DRIVER-ZC does NOT inherit |
| --- | --- | --- |
| [Zero-copy lens law](../knowledge/zero-copy-lens-law.md) | Operator law: no implicit copy escape hatch; ClassView/array is projection; specific transparent-stride examples | No blanket performance proof for every source/storage layout |
| [#1955](https://github.com/AdaWorldAPI/lance-graph/pull/1955) merged | D-DRISA-1 lab: borrowed in-memory canonical SoA, frame/layout/source checks, route once, no owned value arrays in register file | No virtual XY/XZ bus, no disk or remote guarantee |
| [#1956](https://github.com/AdaWorldAPI/lance-graph/pull/1956) merged | D-DRISA-3 lab: carry-ALU semantics exact, but 1.5–4.8× slower than native alternatives on eight measured shapes | No mandate for a carry-based ALU, no new public ISA |
| [#1950](https://github.com/AdaWorldAPI/lance-graph/pull/1950) merged | Source mask read and ingress costs with user-provided row IDs; prepared-bind Q≥2 includes copied `gather_lane` values | No end-to-end DR→Lance D4 or zero-copy prepared receipt |
| [#1949](https://github.com/AdaWorldAPI/lance-graph/pull/1949) merged docs | Optional source locator receipts and dynamic ClassView lens proposal | No production location-lease owner |
| [#1953](https://github.com/AdaWorldAPI/lance-graph/pull/1953) merged docs | D-DRISA remote-instruction contract, 57 falsifier pre-registration | No automatically authorized remote endpoint |
| [EXO-104](https://linear.app/exo-red/issue/EXO-104) | Existing `tuples_over_mask` issue: row copies, enumeration, sorting, latency violation | A tuple oracle is not a production implementation |
| [EXO-105](https://linear.app/exo-red/issue/EXO-105) | Reference executor/public-API licensing decision open | Do not mutate frozen mask-risc source from this docs plan |
| [EXO-109](https://linear.app/exo-red/issue/EXO-109) | Oracle independence decision | Reference implementation may not reuse the candidate's algorithm |

**Open admission decisions:** EXO-46 (counted global→focus binding), EXO-89 (RelationshipMap), EXO-151 (DR focus/lease), EXO-140 (foreign semijoin/gather), EXO-121 (ClassView/field contract), EXO-114 (ordered source), D4/S-INGRESS. Source-owner layout witness and result-mask continuation also require ownership decisions from [D-DRISA M1/M3](d-drisa-decision-memos-v1.md).

**Deferred by operator instruction:** Any audit or consolidation of migrated ladybug-rs/cognitive-shader-driver, P64, StreamDTO, BusDTO or PerturbationDTO. The eventual goal of **one canonical surface** is recorded, but the current plan does not inspect or choose among those residues. No new per-op web/REST/gRPC surface is authorized.

## 7. Acceptance gate for the docs and first executable probe

Read [staged PR charter](d-driver-zc-staged-pr-charter-v1.md) and [independent falsifier matrix](d-driver-zc-falsifier-matrix-v1.md). Passing ZC-1 means **one existing SoA source, one framed binding, at least five distinct virtual bus views, no owned value replicas, exact independent results and deliberate disable tests**. A passing zero-allocation microbenchmark is insufficient.

**Explicit NO-GO**: new universal hash table, full-field gather as default, hidden owned `Vec` of source values, fake FNV cryptographic source proof, repeated per-call foreign lookup, mixing schema/source/scope witnesses, unadmitted cross-source joins, unaccounted disk/network copies, new SIMD backend or independent cognitive/shader driver.
