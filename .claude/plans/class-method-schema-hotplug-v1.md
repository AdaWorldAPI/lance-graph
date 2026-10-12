# D-CMH-0..12 — Class × Schema × Methods hot-plug activation, with Quack and R2IL

> **Draft RFC / documentation only, 2026-10-12.** This file authorizes **no** Rust code, OGAR Class/ActionDef or codebook mutation, new opcode, re-interpretation of persisted bytes, extension of the zero-dependency contract, or production ERP/IAM/CRM mutation. The first implementation requires named parity and disable tests.
>
> **Sibling semantic RFC:** [OGAR #357](https://github.com/AdaWorldAPI/OGAR/pull/357); *this* plan is the **physical binding/activation seam**, not another logic engine.
> **Evidence inventory:** [class-method-hotplug-existing-contracts.md](../knowledge/class-method-hotplug-existing-contracts.md); **falsifiers:** [class-method-hotplug-falsifiers-v1.md](class-method-hotplug-falsifiers-v1.md). Each [G]/[H]/[ABSENT] label is scoped to the cited code or explicit gap.

## 0. Problem and decision

A consumer currently may separately:
1. declare `HotPlug { consumer, classids: &[u16], covered }`;
2. receive `OgarAuthority.activate(&plug) -> Activation` containing concept/capability and contextual `ReadMode`;
3. bind ClassView / `WideFieldMask` structural fields;
4. implement Quack `Binder`: textual table/field/literal → numeric `TableId`, `Col`, `Mask` and validity;
5. author an R2IL procedure or call Quack `lower`;
6. separately resolve OGAR `MethodDecl` / `ActionDef`.

**The missing link is one validated, per-plug, class-scoped *binding composition*, not a global class table, new catalog or runtime.** The desired runtime unit is a method *resolved against* a class reading and a versioned data source. Every consumer reuses the same process.

### Proposed invariant D-CMH-0

`(OGAR class authority × OGIT canonical field identity × actual slab declaration × schema/field projection × method identity/effect) -> PreparedClass`, with **zero new hot-path name resolution** and typed, fail-closed query/procedure/action dispatch.

One physical bulk engine: `mask-risc::Program -> ndarray::simd`. Quack and R2IL are separate **frontends/composition surfaces**, not duplicate row evaluators. Every required field is resolved or activation refuses; DO never lowers into a write-masking kernel.

**Do not alter** the existing `HotPlug`, `CapabilityAuthority`, `Activation` and `ResolvedReading` contracts in W0/W1. In particular:
- `HotPlug.classids` uses the **canonical concept high u16**; application identity/namespace is bound separately as the low u16 of full `u32` class identity (`concept << 16 | appid`). Do not conflate `ClassId = u16` in ClassView with complete app-specific `u32` classid.
- An activation is **per plugged set**. It is never a newly hard-coded global lookup, and no global COUNT_FUSE/consumer-roster lockstep is reintroduced.
- An explicit `SlabDeclaration` wins for **persisted physical reading/value schema**, not the semantic interpretation of a current ClassView or the key tail. Old slabs remain readable under their declared physical format.
- `read_mode_for` and signed register law binding **bang** on absence/mismatch; never silently choose V1 or guess class semantics.
- **Reading-only** hot-plugs may have `covered = []` and no capability table. Do not treat every lack of ActionDef as an activation error. Permissions are not created by `HotPlug.covered`.

## 1. Proposed activation diagram

```text
SOURCE ONCE / COLD PLUG
  Consumer HotPlug + app prefix (concepts/capability declarations)
       │
       ├─ OgarAuthority (class identity, minted concepts, capability set, reading)
       │        └── Activation / ResolvedReading (existing; unchanged)
       ├─ OGIT OntologyRegistry (canonical field URIs, aliases, bridge ids)
       ├─ OGAR Class + MethodDecl + ActionDef (source semantic surface)
       ├─ ClassView ordered FieldRefs + WideFieldMask (field universe)
       ├─ physical SlabDeclaration + source generation (bytes already written)
       └─ ogar-loco Inventory (keyed function bodies; local FnAddr)
               │
        [ClassActivationCompiler]  << proposed composer, cold path
          1 identity & reading check
          2 ordered field + native/semantic mapping check
          3 type/null/units/RBAC binding
          4 methods + body/effects link
          5 lower/certify pure plans
          6 publish one immutable PreparedClass or refuse
               │
        PreparedClass { schema seal, bound fields, method slots, read rules }
               │
             HOT CALL:
     MethodSlot + generation-checked Planes + ActorContext
               ├── Query -> Quack Query / lower(_fused) -> mask-risc
               ├── Procedure -> Inventory/FnAddr -> R2IL FoldDialect -> mask-risc
               ├── Compute -> checked compute DAG / typed derived read
               └── Action -> inert OGAR ActionInvocation -> AUTHORITY GATE
                                                      -> CapabilityExecutor
               │
             Versioned result/receipt, no row-sized Context materialization
```

**No duplication:** OGAR *authoritatively* owns class/definition/action semantics. OGIT *authoritatively* resolves imported ontology field identity/URIs. The binder does not mint either, and Quack `Registrar` does not become a class registry.

## 2. API sketch (illustrative Rust, NOT shipped)

```rust
// A sidecar in the execution/binding layer, NOT a new zero-dep contract type.
struct PreparedClass {
    identity: FullClassId,             // canonical concept high + appid low
    schema_seal: SchemaSeal,           // deterministic semantic revision
    reading: ResolvedReading,          // existing HotPlug + SlabDeclaration
    fields: BoundFieldMap,             // field-key -> Col, LaneKind, validity
    methods: BoundMethods,             // [MethodKey] -> PreparedMethod
    field_universe: WideFieldMask,     // per-class projection basis
}

struct PreparedMethod {
    key: MethodKey,                    // stable [u8;16] identity; OGAR/substrate minted
    arity_and_types: MethodSignature,
    required: WideFieldMask,           // property positions in this schema
    produces: ResultSort,
    effect: EffectClass,
    body: MethodBodyRef,
    law: Option<VerifiedLawCertificate>,
}
enum MethodBodyRef {
    Quack(BoundQueryTemplate),
    R2il { inventory_key: [u8;16], local_addr: FnAddr },
    Compute(ComputeGraphRef),
    OgarAction(ActionDefId),           // never executable without an Invocation
    Unsupported(ExplicitGapCode),
}
enum EffectClass { Pure, ReadVersion, Propose, ExternalMutate }

fn prepare_class(
    plug: &HotPlug,
    authority: &impl CapabilityAuthority,
    schema: &impl CanonicalFieldSource,
    method_inventory: &impl MethodSource,
    slab: Option<&SlabDeclaration>,
) -> Result<PreparedClass, PrepareError>;

fn call_pure(
    prepared: &PreparedClass,
    method: MethodKey,
    data: BorrowedPlanes<'_>,
    context: &ReadContext,
) -> Result<TypedResult, CallError>;
```

`PrepareError` must name the failure, never degrade to an empty/default program: `UnknownClass`, `UnmintedCanonicalField`, `AmbiguousAlias`, `SchemaSealMismatch`, `FieldOverflow`, `UnmappedRequired`, `MissingValidity`, `UnsupportedLane`, `MissingBody`, `MethodSignatureMismatch`, `UncertifiedRewrite`, `SlabDrift`, `NoReadScope`, `EffectNotPure`, `CapabilityMismatch`.

W0–W2 should implement the **composer as an additive sidecar** in a layer allowed to depend on OGAR, Quack and R2IL. Do not add those dependencies to `lance-graph-contract`. Actual crate path to be selected in W1 after verifying the excluded-workspace integration constraints: `lance-graph-ogar` is a candidate, but it is workspace-excluded and must not induce a Cargo workspace-load dependency cycle. If the executable binder belongs in a consumer-facing existing crate, keep the same contracts and a separate test target; do **not** mint `lance-graph-class-registry` as a new crate for convenience.

### 2.1 Method identity: key, slot, interpreter address are THREE things

- **Stable identity:** keyed 16-byte function node identity from OGAR's `ogar-loco::Inventory::key_of`; a genuine key must be minted by the substrate/owner. `MethodKey` binds owner Class, schema/semantic revision and signature as a *separate verified registry relation*, not from random hashing truncated into a global classid.
- **Local dispatch:** `ogar_loco::FnAddr(u16)` is a transient ordinal inside one inventory; `VecInventory::push_keyed` and `addr_of` already work, and `Interpreter::with_inventory` resolves bodies. The inventory does **not** allocate a globally stable method ordinal.
- **CALL bytes:** current program body-reference positions use byte-sized targets and the entry address 0 is reserved. A 65,536-capacity inventory does **not** prove any one CALL can name all 65,536 methods. Validate reachable/callable address encoding or refuse.
- **C++ signatures:** `lance_graph_contract::codegen_manifest::{MethodSig,ClassMethods,methods_for}` are already a generated **signature manifest** for C++; they are not OGAR `MethodDecl` or an executable R2IL body. A preparation adapter may reuse the signature pattern, but must not silently merge three distinct authorities.
- **OGAR `MethodDecl`:** holds source name/kind/record semantics and *opaque* `body_source`. Merely harvesting a method never means it can be compiled into R2IL. Failed lifting yields `Unsupported`, not a fake implementation.
- **NARS:** `ogar-loco/src/nars.rs` already registers 34 recipe codes, but Control `body_refs=0` in its current table; recipe names alone do not create bodies.

### 2.2 Field bindings: one canonical basis, multiple domains

`ClassView::fields(canon_classid)` fixes an ordered per-class basis. Build stable **field identity before ordinals**:
1. Resolve source technical name, alias and URI using OGIT `OntologyRegistry` (`bridge_id, public_name` → `SchemaPtr` / canonical URI).
2. Require a **reviewed semantic mapping** from each native source field to the OGAR field identity. Same position or type in SAP/Odoo is not evidence of same meaning. Record grade `Proposed | Hypothesized | Converted | Refused` and conversion proof/version.
3. Bind each eligible canonical field onto `Quack::BoundField { col, kind, validity }` with physical carrier, null rule and explicit unit/scale. Bind derived lenses separately, with declared provenance and signature.
4. Canonicalize the **ordered** field universe *once*, mint the `WideFieldMask` from requested field positions, and record a schema seal covering IDs/order, types, conversions, validity, reading and method signature; no labels on hot path.
5. At execution, borrow `Planes::lanes` / `Planes::masks` from the dataset. Ensure plan source generation, tenant/role grants and Slab declaration match; do not materialize a row tuple for the binder.

`FieldKey = (FullClassId, CanonicalFieldId, SchemaRevision)` is conceptual. Do not use `(classid,ordinal)` as a portable semantic identity *across* schema versions, classes or bridge mappings. An ordinal is stable only within the declared ordered field basis.

### 2.3 What WideFieldMask canonizes, and what it does NOT

- `WideFieldMask` is a **FIELD-position mask**. Small (<65 positions) matches `FieldMask(u64)` bit-for-bit and starts without heap allocation; >64 uses wider representation. Its `from_universe_present` constructor caps the universe at **256 positions**. An imported class with >256 fields must be split per existing OGAR-SOC rule or fail closed, not silently truncated.
- `from_universe_present` ignores present names absent from the universe by construction; the caller **must check every requested name** against the declared universe before calling it. `CatsSchema::realize_projection` does this and is the immediate precedent.
- Presence, required fields, grants and actual data-value validity are FOUR distinct notions. Proposed typed field masks: `required`, `available`, `read_allowed`, `write_declared`, plus per-field **row validity** planes. Never treat a presence bit as RBAC or a missing row value as a missing field declaration.
- Cold permission/availability criterion for a pure method:
  `required ⊆ (available ∩ read_allowed)`, plus all requested fields mapped under one schema seal. The field algebra here is *not* the row-selection mask of `mask-risc`.
- Hot read mask:
  `R_result = R_class ∩ R_tenant ∩ R_RBAC ∩ R_live ∩ R_method_predicate`, all referencing the **same row universe and dataset generation**. Quack SQL NULL semantics uses `BoundField.validity` (row plane): `UNKNOWN` cannot be treated as `FALSE` by arbitrary `NOT` rewriting.
- For DO, `write_declared` is only an **effect inventory** for an `ActionDef`. It never grants an actor permission. OGAR `ActionInvocation` must still pass current RBAC, guard, approval/MUL and idempotency checks.

### 2.4 Schema revision vs snapshot identity

Proposed `SchemaSeal` must be a deterministic hash of *semantic* schema facts: ordered canonical field IDs, native → semantic mapping/version, carrier kinds, logical types, NULL/Unknown and row-validity policy, unit/scale conversion law, derived-field provenance, schema/declaration versions, ClassView reading, method signatures and verified result/effect types. Exclude localized labels/UI cosmetics and runtime row counts. If a method body changes without a schema change, **also key the executable plan by its MethodKey/body version**.

Cache bound code with tuple `(FullClassId, SchemaSeal, MethodKey, BodyVersion, ReadingLaw)`; bind execution against `(LanceGeneration, SlabDeclaration, Tenant/ActorContext)`. No cache entry can supply permissions. On a schema update, validate a new class activation **off-path** then atomically publish the new generation; in-flight reads either finish against their pinned slab/source or refuse/replan. Existing persisted slabs continue to decode according to their stored declaration. No implicit recarving or key migration.

## 3. TWO lowering paths, ONE physical backend

### 3.1 Quack direct query (existing and suitable for filtering/reporting)

```text
method field names (cold)
  -> Quack::Draft.bind(&impl Binder)
  -> Query { Filter, Agg } (fixed-width; no text)
  -> Quack::lower / lower_fused / grouped lowering
  -> mask-risc::Program -> execute_into(borrowed planes)
```

The shipped Quack `Binder` supports table name, live row mask, name→`BoundField {Col, FieldKind, validity}` and literal→numeric codebook ID. The current `FieldKind` in that Binder supports `I32` and `Code/U32`; it does not mean the general mask-risc executor is limited to those lane types. It is illegal to guess U64/strided/optional fields by losing information. `Quack::Registrar` currently registers only name+width and has **no production implementation**; do not promote it into the canonical Class registry.

### 3.2 R2IL named method (existing interpreter and missing link)

```text
MethodKey -> OGAR-validated function inventory key
  -> FnAddr (local, checked)
  -> ogar-loco Interpreter::with_inventory(...)
  -> FoldDialect::call (R2IL calls / existing procedures)
  -> mask-risc Program per terminal / pure scalar composition
```

Current `lance-graph-r2il::procs` already contains `select_lt`, `filter_lt`, `sum`, `group_sum` and `margin`. R2IL is a *program composer*; its `FoldDialect` uses mask-risc for each bulk terminal, not another row loop. **One mask-risc Program has one terminal**; `margin = SUM(revenue) - SUM(cost)` requires two folded epochs and reselection. The method planner may cache legal input plan components, but it must not retain an invalid scratch mask across terminal epochs.

The **proposed bridge** has to bind the same ordered fields/validity/read scope to both Quack `Query` and R2IL `LOAD/Predicate` call operands, then prove answer parity. A bit-identical `Program` is **not mandatory**: Quack `lower` (survivor gated) and `lower_fused` (TERNLOG) intentionally produce different instruction sequences; the contract is **identical typed results** under exact oracle, with resource counts measured separately. Only assert byte-for-byte equality for explicitly identical normalization policies.

No generic magic conversion from arbitrary Quack `Expr` into loco exists today. No automatic body parser/PROC_DEF/PROC_CALL semantics should be claimed. The first bridge may be a static method catalog of *known working* pure R2IL procedures.

### 3.3 Derived/computed methods

`ClassView::ComputeEdge` and compute DAG are existing descriptors. The existing `compute_dag_is_acyclic` checks **64 positions** while `WideFieldMask` accepts **up to 256**. This mismatch is a hard guard: do not silently accept Odoo computed methods above position 63 or ignore a dependency cycle; either a separately tested **generic** wide DAG checker or a declared sub-schema partition must be provided first. A plain source-language method read is **not** the same as an `@api.depends` recomputation trigger; respect OGAR's differentiated reads/writes/calls and source-specific compute/inverse rules.

### 3.4 Action methods: a protected, separate path

```text
MethodBodyRef::OgarAction(ActionDefId)
   -> ActionProposal (stable object ID, witness, intent, source version)
   -> graph-flow-action::ActionInvocation::commit_via(...)
   -> OGAR RBAC + state guard + capacity/MUL + required approval
   -> graph-flow-action-ogar::GatedOgarHandler
   -> external capability executor
   -> idempotent receipt -> independent observed outcome
```

`do()` is **not** `BlendI32` and is **not** the physical masking executor writing a field. Reading/evaluating a candidate set and actually mutating a target are different authority levels. The method dispatcher must refuse `call_pure(Action)`; an agent/model cannot construct an authorized action directly. Source generations, actor scopes, guard context and dedupe identity are revalidated at the DO membrane. Missing provider idempotency requires outbox+reconciliation, not a claim of exactly-once calls.

## 4. End-to-end examples and source-truth caveats

### SIMAF time-entry DTO (called "CATS" in crate): the existing prototype

Source: `crates/lance-graph-sap/{schema.rs,binder.rs,bind.rs,query.rs}`.
- `CatsSchema` has **23 fields**, append-only ordinals, technical-name/C# aliases and `ClassView::fields`, plus `realize_projection()` that prevalidates unknown field names before `WideFieldMask::from_universe_present`.
- `CatsBinder` implements `Quack::bind::Binder`; maps native NUMC/ABAP_BOOL/dictionaries and separately derived `work_day` / `billable` lenses.
- Optional CATS fields without validity planes and some U64/strided reads are **explicitly refused**; losing sentinel/NULL or lane kind is not a "helpful" fallback.
- The DTO source is **SIMAF's time-tracking DTO, not proof of actual SAP CATSDB/BAPI_CATS_INSERT runtime parity**. `hours` conversion is **batch-local scale**; a cross-batch sum needs an explicit rescale law.

Proposed W1 method:
```text
class concept BILLABLE_WORK_ENTRY (0x0103, when resolved by appropriate port)
 native CatsSchema → semantic FieldMap with grade and unit witness
 method sum_billable_hours : Pure Sum
   required = {billable, hours_logged}
   query = live & billable==true -> Group/Sum on scaled integer lane
   body = Quack query OR R2IL procedure; same scalar oracle
```

**Failure is valid:** the current canonical basis lacks some temporal roles; 0x0103 may have no ActionDef capabilities, so a reading-only plug stays useful. Never mint a CATS capability to satisfy the demo.

### Odoo billing and account.move wide projection

- Existing `lance-graph-sap/tests/view_convergence.rs` used Odoo `account.move` with **216 fields** in its harvested reference fixture; this is a **wide-mask mechanics test**, not evidence that Odoo's fields match a SAP CATS field.
- `account.analytic.line` is a different Odoo model. Its base extraction may lack `employee_id`/`project_id`/`task_id` until `hr_timesheet` / `project` expansions are harvested. Do not assume them in the registered schema.
- W2 tests: bit 0, 63, 64, 108, 215 on an ordered 216-field universe; alias equality and unknown alias refusal; computed dep>63 refusal; method sum over converted `duration` with SAP hours vs WoA minutes exactly checked, no implicit label-based correspondence.
- `D-XGP-0..8` records cross-glove basis under `0x0103`, current statuses and unmapped temporal roles. Source-to-canonical claims marked "Hypothesized" remain **non-bindable** until a demonstrated `Converted` map exists. Registry activation must **not** auto-upgrade them.
- For WoA, the hours live on `TimeSheet`, not the `TimesheetActivity` description child; see D-XGP-8. No silent team/member or parent-row conflation.

### IAM / HubSPO pure guard, then guarded DO

```text
IAM user class: active AND NOT licensed AND independent_grounding
HubSPO record: allowed_owner_set AND due AND stage_eligible
Rubicon: goal_accepted AND context_triggered AND NOT blocked

==> each is a typed field-to-row mask compilation (NOT shared physical lanes).
==> simulate and propose ActionInvocation; only OGAR hard gate may perform DO.
```

For HubSPO, H-ADR-7 keeps workflow state and timers in Lance rows and forbids `ractor` actor-per-record as a dependency. A logical actor is a stable address plus versioned continuation, not a new runtime process. A Common Boolean form does not grant identical business semantics.

## 5. Hotplug lifecycle and call contract

| Moment | Owner | Action | Refusal |
|---|---|---|---|
| DISCOVER | consumer | requests only used canon concepts / covered capabilities | unknown/foreign claimed seat |
| ACTIVATE | OGAR `OgarAuthority` | resolve concept, action surface, reading law for this plug | `ActivationDrift`, never implicit V1 |
| RESOLVE SLAB | contract `Activation` | class concept × current/old slab declaration × supported layout | wrong version/law/carrier |
| BIND FIELDS | OGIT/OGAR adapter | canonical URI, source mapping grade, aliases, units, NULL, ordered schema | ambiguous/unknown, field>256, missing validity |
| LINK METHODS | OGAR + loco | link method identity/signature to body/`FnAddr` or ActionDef | duplicate/unresolved/unsupported body |
| CERTIFY | physical plan binder | verify input/output sorts, mask universes, method effect and algebraic laws | invalid rewrite, unsupported lane |
| PUBLISH | consumer | atomic immutable `PreparedClass` for schema/body version | incomplete activation never published |
| READ | Quack/R2IL + Lance | generation-bound authorized borrowed Planes | stale source, revoked access, wrong slab |
| DO | OGAR capability executor | commit-time current actor/guard/approval, deduped external effect | deny/hold/block, no direct write |
| UPDATE | OGAR producer + slab writer | new semantic activation for new writes; old slabs read by declaration | no downgrade/reinterpretation |

**Security:** capability *availability* (`HotPlug.covered`), field-level projection, current RBAC grants, source row scope and `ActionInvocation` authorization remain orthogonal. Hot reload cannot make a previously denied field visible by mutating the method table without a new activation and fresh current authorization. Stable plan handles require cross-process identity+generation if a workflow resumes elsewhere.

## 6. Wave plan (D-CMH-1..12) and STOP tests

| D-ID | Work package | Owner / existing home | Exit condition |
|---|---|---|---|
| **D-CMH-0** | this RFC + inventory + falsifiers | lance-graph docs, OGAR #357 linked | no code/ISA edits |
| **D-CMH-1** | one composed PreparedClass sidecar for a reading-only plug | existing consumer/`lance-graph-ogar`, **not contract** | no second authority; 0x0103 no-capability path works |
| **D-CMH-2** | checked canonical field/alias→Quack binder with seal | OGIT registry + Quack Binder | unknown mapping/alias refusals and field-order pin |
| **D-CMH-3** | WideFieldMask read/required/allowed projection + null/row scope | ClassView + Quack | cases >64; no authorization inferred from presence |
| **D-CMH-4** | method inventory bind by 16-byte key + local FnAddr | OGAR loco + activation sidecar | missing/oversized/unreachable body refuses |
| **D-CMH-5** | a single SAP/SIMAF pure method via Quack and R2IL | lance-graph-sap + r2il/procs | same independent scalar oracle; no extra bulk executor |
| **D-CMH-6** | Odoo/WoA cross-glove conversion semantics, 216 fields | D-XGP existing glove-parity | field identity not guessed, no unit/scale drift |
| **D-CMH-7** | compute DAG wide-field boundary | ClassView/consumer | never silently drop dependency >=64 |
| **D-CMH-8** | typed method purity/effect split and OGAR action dispatch | OGAR action / graph-flow-action-ogar | denied actions never reach handler |
| **D-CMH-9** | schema/body versioned atomic reactivation and old-slab read | HotPlug reading + Lance versions | no V1 fallback, no retroactive recarve |
| **D-CMH-10** | Rig/read-only bounded method export | graph-flow-lance/Rig | no raw row/SoA Context; source+field auth |
| **D-CMH-11** | optional wide numeric or binding operations **only after insufficiency proof** | smallest physical owner | oracle failed before/green after; separate PR |
| **D-CMH-12** | decision review / acceptance of general hotplug schema-method pattern | cross-repo maintainers | all named adversarial tests green, no unreviewed ID mint |

**Minimal first slice:** implement only D-CMH-1..5 for a **read-only** SIMAF/CATS method with a mock authority, no new opcodes or global registry, and compare Quack vs R2IL vs plain-loop answers. Do not implement W6/DO/method bytecode until this fails or passes with evidence.

## 7. Decision ledger: do not silently settle

1. Should `PreparedClass` live in the existing workspace-excluded `lance-graph-ogar` or a consumer-side feature? **Never** make `lance-graph-contract` depend on OGAR.
2. Who issues the stable 16-byte `MethodKey` and how does it relate to `ClassId(u16)` vs full `u32` app identity and `FnAddr(u16)`? OGAR owner mints; no hash collision by truncation.
3. How are valid R2IL bodies linked to an OGAR `MethodDecl` without interpreting arbitrary source-language bodies? "Unsupported" is permitted and must be visible.
4. Can Quack's name-based `Binder` and R2IL's numeric `LOAD` share *one* bound `FieldMap` without duplicate sources of truth? This is the core first probe.
5. What exact method result sorts (`Scalar, RowMask, GroupFold, Truth, BindingMultiset, ActionProposal`) already have legal physical representation?
6. Who certifies source→canonical mapping `Converted` and unit/scale semantics? Type equality alone cannot.
7. What is the null/validity plane encoding for CATS nullable fields, U64 and strided terms before unrefusing them?
8. Where is the canonical 256-field compute dependency checker; when should an oversized class split according to OGAR-SOC?
9. When does an existing slab bind to an old semantic schema vs new ClassView metadata, and how are migrations explicit?
10. How do we prevent generation/tenant/field ACL cache reuse after resume or read-replica changes?
11. How are Quack/R2IL method plan costs compared when one canonical answer admits different instruction sequences?
12. What is the minimal DO adapter; how are effect receipts, provider idempotency and postaction observation represented?
13. What is the exact role of codegen_manifest::ClassMethods in this binder? It is a *signature-only* C++ source today.
14. For goal-oriented NARS/Rubikon methods, ensure goal selection != OGAR technical commit, preserving lance-graph #1496/#1497/#1498 research gates.

## 8. Sources and cross-repo boundaries

- `crates/lance-graph-contract/src/{hotplug.rs,class_view.rs,codegen_manifest.rs}`
- `crates/lance-graph-quack/src/{bind.rs,lib.rs}` and `crates/lance-graph-r2il/src/{lib.rs,procs.rs}`
- `crates/lance-graph-sap/src/{schema.rs,binder.rs,bind.rs}`
- `crates/lance-graph-ogar/src/lib.rs` (`OgarAuthority`)
- `crates/lance-graph-ontology/src/registry.rs` (OGIT canonical bridge names)
- `crates/lance-graph-glove-parity/` and [cross-glove plan](cross-glove-business-parity-v1.md) (D-XGP, graded source mapping)
- [OGAR MethodDecl/ActionDef](https://github.com/AdaWorldAPI/OGAR/blob/main/crates/ogar-vocab/src/lib.rs), [loco Inventory](https://github.com/AdaWorldAPI/OGAR/blob/main/crates/ogar-loco/src/inventory.rs), [Interpreter](https://github.com/AdaWorldAPI/OGAR/blob/main/crates/ogar-loco/src/interpret.rs)
- [OGAR #357](https://github.com/AdaWorldAPI/OGAR/pull/357) typed-logic proposal (**draft**); [lance-graph #1496](https://github.com/AdaWorldAPI/lance-graph/pull/1496) merged NAL planning, [#1497](https://github.com/AdaWorldAPI/lance-graph/pull/1497) PLN attention draft, [#1498](https://github.com/AdaWorldAPI/lance-graph/pull/1498) Hyperon draft.
- [graph-flow-lance](https://github.com/AdaWorldAPI/rs-graph-llm/blob/main/crates/graph-flow-lance/README.md), [graph-flow-action-ogar](https://github.com/AdaWorldAPI/rs-graph-llm/blob/main/graph-flow-action-ogar/src/lib.rs), [HubSPO-rs](https://github.com/AdaWorldAPI/HubSPO-rs).

**One-line target:** *HotPlug announces an existing concept; OGAR resolves the class and methods; ClassView/WideFieldMask seals the field projection; one binder links Quack/R2IL to the same numeric lanes; OGAR alone admits do().*
