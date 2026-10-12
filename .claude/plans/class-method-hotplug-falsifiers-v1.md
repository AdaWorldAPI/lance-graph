# D-CMH executable acceptance suite — Class activation, schema mask and method dispatch

> **RESEARCH / proposed tests only (2026-10-12). Nothing in this plan has been executed by this PR.** Execution owners and original test directories are named; no new test, runtime, permission, codebook or ABI has been merged. Parent: [class-method-schema-hotplug-v1.md](class-method-schema-hotplug-v1.md).

## 0. Acceptance doctrine

The test target is **one prebound class-method activation**, not a generic runtime. For every falsifier record: pinned repo SHA and feature flags, source fixture, source-to-canonical mapping grade, classid and app prefix, schema seal, method key and local FnAddr, exact slab/row generation, actor/tenant grants, original scalar oracle, Quack and R2IL outputs, error code when refusing, hot-path allocations, mask bytes, fold count and any external mock count.

A test is *invalid/vacuous* if the negative control does not actually change a result or force a refusal when the violated invariant matters. Disable-check a real guard and record red→green. A broad "CI green" result with no disabled arms is not semantic parity.

Do not demand that Quack and R2IL emit **identical mask-risc opcode sequences**: Quack `lower` can skip rejected mask words while `lower_fused` can produce ternlog, and R2IL can emit multiple terminal epochs. Demand an equal **typed answer under an independent oracle** at identical data/source/permission versions. Only test physical-plan byte identity where equivalent canonical lowering was deliberately selected.

### Baseline fixtures

- **F-SIMAF-CATS:** existing `lance-graph-sap` 23-field DTO/`CatsBatch`, technical-name and C# name aliases, known decimal scale, `work_day`/`billable` derived lenses, nullable optional field with sentinel. This is **not** CATSDB or a BAPI integration.
- **F-ODOO-WIDE:** harvested Odoo `account.move` with **216** fields in `sap/tests/view_convergence.rs` (mask mechanics only), plus an explicit `account.analytic.line` fixture where fields from `hr_timesheet`/`project` may be absent.
- **F-CROSS-GLOVE:** OGAR `BILLABLE_WORK_ENTRY=0x0103`, Quack/OGIT canonical URIs, conversion grade provenance from D-XGP; SAP hours vs WoA minutes and optional Odoo unit_amount. Current claims "Hypothesized" must refuse binding.
- **F-ACTION-MOCK:** synthetic OGAR method with pure selector + `ActionDef`, blocked/allowed ActorContexts, source generations v,v+1, simulated provider effect ledger keyed by idempotency.
- **F-HOTPLUG:** `OgarAuthority.activate` with reading-only capability set, known and unknown classid, V3 and malformed/legacy slabs; deliberately modified register law.

## 1. Test matrix

| ID | Claim + fixture | Oracle / required result | Disable / adversarial failure |
|---|---|---|---|
| **CMH-P01** | Reading-only `HotPlug { classids:[0x0103], covered:[] }` resolves reading even without an ActionDef table | existing `OgarAuthority` + `ResolvedReading` | force capability-join for reading-only seat: false `NoCapabilitiesFor` should be detected |
| **CMH-P02** | A plugged unknown id or claimed palette seat owned by another consumer fails closed | `ActivationDrift::{UnknownClassid,UnexpectedConsumer}` or exact current error | remove claimed-seat/unknown guard: unauthorized activation goes green → FAIL |
| **CMH-P03** | V3 current read versus old existing slab physical reading are separately preserved | byte-exact source slab decode; `SlabDeclaration` value schema wins; key/tail reading still resolved lawfully | override old slab with current new byte reading; produce wrong result/refusal |
| **CMH-P04** | Unknown slab tag, layout version, signed register law, missing rail and no reading refuse | precise `ActivationDrift` arms | inject implicit `ReadMode::DEFAULT`/V1 fallback and detect silent acceptance |
| **CMH-P05** | Canon high u16 concept + low u16 appid are not reversed or conflated; methods bind full ID while OGAR validates concept | explicit encoded full `u32` and expected owner | swap halves, share method plans across appid; field/method isolation fails |
| **CMH-P06** | `WideFieldMask::from_universe_present` is wrapped with strict unknown-name precheck | `CatsSchema::realize_projection` reference + explicit errors | unknown present name silently discarded by raw constructor: detect false green |
| **CMH-P07** | Wide mask bits 0/63/64/108/215 for harvested Odoo `account.move` 216-field basis; zero lower-bit collisions | independent array/bitset per ordinal + full 216-field provenance | drop >=64 bit, wrap to low word, alias different schema ordinal: red |
| **CMH-P08** | Universe cap >256 fails; 256 max accepted; optional >256 class splits only with approved sub-schema | `WideMaskCapError::UniverseExceedsSocCap` or explicit split contract | silent wrap/truncation: red |
| **CMH-P09** | Same native field technical/C# alias resolves same numeric `Col` and canonical URI where grounded | existing `CatsSchema::resolve`, OGIT URI mapped source | re-order fields without changing seal or map duplicate aliases to different IDs: red |
| **CMH-P10** | Native Odoo/SAP/WoA fields with equal lane kind but **Hypothesized** equivalence are not bound to shared canonical meaning | D-XGP mapping grade oracle; `Converted` only with reviewed unit proof | auto-join by label/type/ordinal and report false cross-glove parity |
| **CMH-P11** | SAP batch-local hours decimal scale, WoA integer minutes, and Odoo unit_amount explicitly convert | exact rational/int arithmetic under bound; no premature rounding | mix scale 2 with scale 3 / treat minutes as hours: nonzero oracle delta |
| **CMH-P12** | Quack binder nullable comparison semantics over `BoundField.validity`; SIMAF optional sentinel without plane **refuses** | scalar SQL 3VL `TRUE/FALSE/UNKNOWN`, ensure WHERE only true | drop validity / treat NULL as 0; `NOT EQ` admits false row |
| **CMH-P13** | Quack Binder name→Col and R2IL LOAD use **identical BoundFieldMap** per prepared class | same value oracle across masks: empty/1/63/64/65/256/257/1000 rows | deliberately swap one lane for R2IL, must mismatch |
| **CMH-P14** | Quack `lower`, `lower_fused` and R2IL preserve answers but may differ in program length and fold epochs | exact mask/scalar oracle; report costs separately | require opcode-byte equality or reuse scratch mask after terminal; test catches malformed comparison |
| **CMH-P15** | Inventory keyed MethodKey resolves to body, local FnAddr lookup bounded and stable only for that inventory | `ogar_loco::VecInventory::{push_keyed,addr_of}`, Interpreter::with_inventory | reorder registration and persist FnAddr; wrong method must be detected |
| **CMH-P16** | Missing/duplicate/incompatible method, unencodable body ref, unsupported source method **refuse** | typed registry lookup and body-reference validity oracle | default to first method, entry address 0, or arbitrary source body: fail |
| **CMH-P17** | `ComputeEdge` target/input position >=64 is refused until a wide checker exists; 64-field DAG detects cycles | independent topo sort over declared full field universe | current 64-position checker used blindly on 216-field Odoo: adversarial >63 cycle escapes → FAIL |
| **CMH-P18** | `required ⊆ available ∩ allowed` before reading; RowMask type + generation + tenant scope separate | independent field-permission and row-policy oracle | request forbidden field; alias/NULL/method plan bypasses RBAC: fail, no data to Rig |
| **CMH-P19** | `Pure/ReadVersion/Propose/ExternalMutate` method effect boundary; OGAR ActionInvocation gate required | mock handler count 0 when denied; 1 when authorized with stable idempotency | direct write from Quack/R2IL/LLM model call reaches handler: fail |
| **CMH-P20** | Prepared schema+body version cache invalidates on changed semantic field mapping, method body or auth context; old slab read stable | two immutable activation generations + snapshot pinned external oracle | stale cache hits or new ClassView decodes old slab incorrectly: fail |
| **CMH-P21** | Crash after provider effect before local receipt, replay same logical action produces 1 observable external effect | mock provider idempotency/outbox ledger, observed target state | invocation identity tied to arrival time or repeated action sends twice: fail |
| **CMH-P22** | Class-method activation does not allocate row-sized buffers or a second runtime; hot pure method low-overhead | measure scratch/reads/calls and scalar reference; no row materialization | bridge allocates Vec<Row> or name-strings per row; fail cost/architecture gate |
| **CMH-P23** | Quack `Registrar` never becomes a second method/class authority; no new global classid mint | compare authority activation / production table binding | create_table(name,width) silently registers a new semantic class: fail |
| **CMH-P24** | Reading-only activated class may later acquire ActionDef without changing field/read semantics; DO still denied absent actor grant | two activation snapshots, same read method, different capability presence | new capability makes previously unauthorized actor able to execute: fail |

## 2. Oracle rules and required negative data

### 2.1 F-SIMAF-CATS scalar reference

Generate rows for `n ∈ {0, 1, 63, 64, 65, 256, 257, 1_024}`, fields by current `CatsBatch::lanes()`:
- billable true/nonbillable/unknown, one nonmapped value;
- hours scaled under at least two **different** batch-local decimal scales;
- derived `work_day` across UTC boundary;
- optional approval null sentinel and real 0 value to make NULL/false visibly different;
- empty/all/mixed live mask, including tail bits.

Answer oracle `SumHours(billable=true, authorized_only, exact_scale)` by scalar iteration over source values with **exact conversion before aggregation**. Quack query and bound R2IL procedure must match **numeric answers**, not necessarily bytecode. Confirm they borrowed existing columns and did not mint an undocumented technical concept.

### 2.2 F-ODOO-WIDE ordered projection

For 216-field fixture, compute reference `BitVec(216)`. Include positions `{0, 63, 64, 108, 215}` plus an unknown name. The mask must preserve all five selected fields. Confirm positional identities are qualified by **schema seal** and `classid`. A re-ordered schema with same unqualified ordinal must be rejected. Never misidentify the fixture as `account.analytic.line`.

### 2.3 F-CROSS-GLOVE semantics

A same-name/same-lane synthetic SAP and Odoo field with different units or business meaning is *the negative control*. Do not compare only data shape or convert "already invoiced" (WoA `abgerechnet`) into "eligible for invoice" (`billable`). The first success must pin:
- exact source record class and mapping grade `Converted` for each field;
- unit/scale and normalization formula;
- null/unknown behavior;
- company/tenant scope;
- owner/actor permission;
- proof that source and target semantics refer to the same business quantity.

Until then report `UnsupportedSemanticMapping`; **refusal is a valid outcome**.

### 2.4 F-INVENTORY identity versus address

Create two `VecInventory` instances with the same 16-byte method keys and bodies but different insertion order. Verify `key_of(addr)` and `addr_of(key)` resolve local addresses correctly. A cached `FnAddr(1)` from inventory A must not be blindly used in B. Check entry address `0` and body refs byte width separately; if a method key cannot be represented by current body-reference immediate, activation must either introduce a verified local trampoline with its own proof or refuse. Do **not** widen branch immediates to make the test easy.

### 2.5 F-DO gate, hypothetical plan to real-world mutation

```text
read candidate set -> evaluate pure method -> build inert ActionProposal
-> resolve authoritative ActionDef -> create ActionInvocation
-> verify current actor, class, target, state, source version, RBAC, approval
-> commit_via / GatedOgarHandler -> mock provider -> receipt -> read-back
```

Test injects:
- forged MethodKey / ActionDef id mismatch;
- unauthorized role, wrong tenant, missing field, stale source;
- same invocation redelivered, provider effect then crash before receipt;
- model-generated instructions inserted into proposal payload;
- a pure method trying to return `AuthorizedAction`.

Any external call before hard gate is a P0 failure. Post-action read-back can fail even when provider returned success: not automatically a second identical write or positive evidence.

## 3. Production integration and risk gates

**Performance record**: for each matching setup report hot/cold setup time, compiled instruction count, terminal epochs, scanned/dereferenced row counts, bytes read, decoded bytes, allocated bytes/call, scratch peak, actor-permission checks, materialization bytes and API calls. Compare:
A. existing hand-written CATS Binder path;
B. prepared class lookup + Quack bound query;
C. prepared class lookup + existing R2IL procedure;
D. independent scalar reference (parity oracle, **not** necessarily faster).

Must not infer "zero-cost" from object sizes. `WideFieldMask` grows a heap representation after position 63. Cold activation may allocate; per-call string search or Vec<Row> creation in the hot path is an explicit failure. A generic method dispatch with larger cost and no improvement to reuse/maintainability can be rejected.

**Security criteria:** temporary project/workflow plans and generated method declarations cannot silently override OGAR entity ownership or current RBAC/field grants. Capability *names* are not authorization tokens. In-flight activation hot-swap must not change semantics of the already running plan; stale handles, schema seals and source generations must be rechecked or explicitly rejected.

**Contract drift criteria:** no changes to `lance-graph-contract` zero-dep nature; no global count fuse; no altered FieldMask bits 0..63; no V1 fallback; no classid/appid reversal; no persisted CE64 reinterpretation; no ractor workflow dependency in HubSPO.

## 4. Waves and required reports

| Phase | Minimal artifact | Test gates |
|---|---|---|
| **CMH-W0** this research PR | RFC, evidence inventory, falsifiers, board index | static doc sanity only |
| **CMH-W1** existing consumer / mock authority | PreparedClass sidecar + strict field map | P01–P13, P23 |
| **CMH-W2** existing R2IL procedures | one method resolved as Quack and R2IL | P13–P16, P22 |
| **CMH-W3** wide Odoo + cross-glove | converted field identity with unit proof | P07, P08, P10, P11, P17 |
| **CMH-W4** DO and schema hot-swap mock | stable MethodKey, effect boundary, versioned activation | P18–P21, P24 |
| **CMH-W5** optional Rig / graph-flow consumer | bounded authorized read materialization; resumable call | same security and generation tests, no row/context copy |

### Review checklist

- [x] Existing code and missing joins listed separately in [evidence inventory](../knowledge/class-method-hotplug-existing-contracts.md).
- [x] No operational APIs or changes advertised as shipped by these documents.
- [x] Every P01–P24 has an independent expected result or current exact contract, plus a failure/disable arm.
- [x] SAP CATSDB/`BAPI_CATS_INSERT` parity explicitly NOT claimed; CATS is SIMAF DTO.
- [x] Source field semantics and authority scoped to OGAR/OGIT; consumer declares only plugged IDs.
- [ ] Executable probes, compile checks, CI, local benchmarks and customer integration **not executed in this PR**.
- [ ] Operator's review of method identity/version, wide-compute dependency and DO outbox contract **not yet obtained**.
- [ ] No merge/deployment approval until wave-specific test PRs demonstrate the invariants.
