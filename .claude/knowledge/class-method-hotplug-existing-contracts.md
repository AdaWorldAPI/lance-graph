# Class × Method × Schema HotPlug: code-grounded seam inventory (2026-10-12)

> **Grading:** [G] exact type/behavior read in source; [P] proposal or documentary plan; [ABSENT] seam/code path not shown by inspected sources. *Code-existence is not integration/performance parity*. Read with [the activation proposal](../plans/class-method-schema-hotplug-v1.md) and [the falsifiers](../plans/class-method-hotplug-falsifiers-v1.md). Companion OGAR [#357](https://github.com/AdaWorldAPI/OGAR/pull/357) is **draft**.

## 1. Critical existing bricks

| Brick | What exists now | What the bridge still needs |
|---|---|---|
| **[G] HotPlug socket** | [`hotplug.rs`](../../crates/lance-graph-contract/src/hotplug.rs): `HotPlug {consumer, classids: &[u16], covered}`; `CapabilityAuthority::activate(&HotPlug) -> Activation`; `Activation::read_mode_for` and `register_law_for` fail closed. `resolve_for_context` composes current OGAR reading with an *optional physical* `SlabDeclaration`. `ResolvedReading::bind_register128/bind_signed_register` validate rails/laws. | A sidecar that binds a declared **field schema and executable methods** onto this activation. Do **not** add methods to `HotPlug` or OGAR dependencies to zero-dep contract to claim completion. |
| **[G] Real authority** | [`lance-graph-ogar/src/lib.rs`](../../crates/lance-graph-ogar/src/lib.rs): `OgarAuthority` implements `CapabilityAuthority` via OGAR capability registry and handles V3 reading-only domains/palette seats separately. Empty covered list is valid for reading-only. | Compose authority's existing per-consumer activation with OGAR source `Class/MethodDecl/ActionDef` + OGIT field identities after activation; no new global list. |
| **[G] ClassView fields** | [`class_view.rs`](../../crates/lance-graph-contract/src/class_view.rs): `ClassView::fields(ClassId=u16) -> &[FieldRef]` uses stable per-class ordered field basis. `FieldMask(u64)` remains unchanged, `WideFieldMask` small/wide dual representation supports 0..255 positions; `from_universe_present` max 256 and computes membership in the declared universe. | Need a shared strict alias/basis+schema-version **binder**; `WideFieldMask` is field selection, NOT row selection or RBAC. A present name missing from universe is **ignored** by raw constructor, so prevalidate requested names. |
| **[G] C++ MethodSig** | [`codegen_manifest.rs`](../../crates/lance-graph-contract/src/codegen_manifest.rs): `MethodSig { name, params, ret, is_const, is_static, overrides }` and `ClassMethods{classid:u32,methods}`, `methods_for`. Generated constant manifest, **signature only**; returned empty for missing class. | Different harvested source + no executable body; cannot claim this is OGAR MethodDecl or loco Inventory. Reuse pattern if proven, not table contents/semantics. |
| **[G] OGAR MethodDecl** | [`ogar-vocab`](https://github.com/AdaWorldAPI/OGAR/blob/main/crates/ogar-vocab/src/lib.rs): `MethodDecl{name,kind,body_source,decorators,semantics}` plus `ActionDef/ActionInvocation/KausalSpec` with declared reads/writes/calls/guards. | AST body is opaque, no automatic executable R2IL translation; link typed body ref without duplicating authority. |
| **[G] ogar-loco Inventory** | [`inventory.rs`](https://github.com/AdaWorldAPI/OGAR/blob/main/crates/ogar-loco/src/inventory.rs): `FnAddr(u16)`, `Inventory::{body,key_of,len}`, `VecInventory::{push_keyed,addr_of}`. [`interpret.rs`](https://github.com/AdaWorldAPI/OGAR/blob/main/crates/ogar-loco/src/interpret.rs) has `Interpreter::with_inventory` actually consuming it. **This is SHIPPED, not just a concept.** | Semantic link `OGAR MethodDecl ↔ stable function key ↔ source/target signature ↔ local FnAddr ↔ verified R2IL body` is missing. Call-target byte encoding is narrower than FnAddr's u16 capacity. |
| **[G] Quack Binder** | [`quack/src/bind.rs`](../../crates/lance-graph-quack/src/bind.rs): `Binder::{table,live,field,code}`, `BoundField {col,kind,validity}`, `Draft::bind -> Query`. Text disappears at boundary; SQL nullable comparisons use resident validity mask. `FieldKind` at this binder currently `I32/Code`. | Bind automatically from activated OGAR ClassView/OGIT field identity into Quack numeric keys; do not reinterpret unsupported U64/strided lanes or sentinel NULL as known values. |
| **[G] Quack physical lowering** | [`quack/src/lib.rs`](../../crates/lance-graph-quack/src/lib.rs): `Filter/Agg/Query`, `lower` and `lower_fused` to `mask_risc::Program`; `GroupBy` uses grouped lowering and equivalent folds when available. No row evaluator in Quack. | Method-shaped invocation with bound fields, checked laws and versioned source. Quack and R2IL need answer parity, **not** necessarily identical instruction bytes. |
| **[G] Quack Registrar narrowness** | `quack/src/bind.rs` `TableDeclaration -> Registrar::register(name,width)` supports only table name and row width; module says no production implementation. | It is **not** an OGAR class/method registry, and a new metadata catalog on top would create a second authority. |
| **[G] R2IL bulk dialect** | [`lance-graph-r2il/src/lib.rs`](../../crates/lance-graph-r2il/src/lib.rs): `FoldDialect` interprets R2IL over borrowed planes and invokes mask-risc for each terminal. `Val` holds scalar/slot/address, not row data. One mask-risc Program has one terminal. | Field binding to same `BoundFieldMap` Quack uses; preserve epoch/scratch validity across multiple folds. |
| **[G] R2IL procedures** | [`r2il/src/procs.rs`](../../crates/lance-graph-r2il/src/procs.rs): `select_lt/filter_lt/sum/group_sum/margin`; named Rust emitters call `FoldDialect::call` without new opcodes or hot row loops. | Registered method key -> known/verified emitter, not arbitrary `MethodDecl::body_source`. `PROC_DEF/PROC_CALL` general VM-linker semantics remain open per module doc. |
| **[G] SIMAF CATS DTO** | [`sap/schema.rs`](../../crates/lance-graph-sap/src/schema.rs): 23 fields; `CatsSchema::resolve` technical/C# aliases; `realize_projection` strict unknown precheck before WideFieldMask; [`sap/binder.rs`](../../crates/lance-graph-sap/src/binder.rs): `CatsBinder` Quack binder with `LIVE`, `work_day` and `billable` derived lenses; optional null fields and some U64 refused. | One Registered Class/Method entry instead of independent schema/binder/procedure construction. **SIMAF's DTO is not SAP CATSDB/BAPI parity**; scale is batch-local and must be converted before cross-batch or cross-glove aggregation. |
| **[G] OGIT field identity** | [`ontology/registry.rs`](../../crates/lance-graph-ontology/src/registry.rs): idempotent TTL hydration, `(bridge_id,public_name) → SchemaPtr` and public/OGIT URI lookups, optional Lance cache. | Checked native alias → canonical identity and mapping grade, with semantic unit and source provenance. Don't mint a field identity from a matching string alone. |
| **[G] Source name ↔ row mapping** | [cross-glove plan](../plans/cross-glove-business-parity-v1.md), `lance-graph-glove-parity` tests/board `D-XGP`: candidates for 0x0103, source-to-canonical mapping grades, SAP/Odoo/WoA differences. | Bind only **Converted** mappings with proof; several previously hypothesized relations remain unsupported. |
| **[G] Physical mask** | [`mask-risc/src/ir.rs`](../../crates/lance-graph-mask-risc/src/ir.rs): AND/OR/XOR/TERNLOG/NOT/ANDNOT; predicates, gated Gather/via, group folds, Keep/Count/Any/All; borrowed planes and caller-owned scratch. | Compiler chooses which of these to use; there is no license to invent an opcode per class/method. |
| **[P] Typed logic sibling** | [OGAR #357](https://github.com/AdaWorldAPI/OGAR/pull/357) RFC for algebraic logic, class/method/schema and DO effects. | This PR's *concrete activation binder*; independent implementation PR gates. |

## 2. Important corrections to protect against false progress

### 2.1 Inventory exists now, update the older narrative

Previous drafts described `LocoConcept::Inventory` as minted but not implemented. That was true at the earlier gap doc version, **not** at this read. `ogar-loco::Inventory`, `VecInventory::push_keyed`, `addr_of` and `Interpreter::with_inventory` exist today. The still missing piece is method-to-body **semantic linking** with checked class/schema/authority, plus matching field bindings and plan execution. This needs a correction on review of OGAR #357; do **not** create a second inventory.

### 2.2 Codegen manifest is a different method plane

`MethodSig` in lance-graph-contract is C++ harvested signature metadata, intentionally `&'static` and `const`-constructible. `OGAR::MethodDecl` is language-independent harvested source method metadata and contains opaque source, while `ogar-loco::Inventory` is an executable body-address service. The final composed adapter may reference all three, but none is equivalent to the others.

### 2.3 Same mask basis ≠ same SAP/Odoo business semantics

[D-XGP-0](../board/entries/2026-10-10-d-xgp-0-cross-glove-inventory.md) explicitly found that the wide-mask example is Odoo `account.move`, **216 fields**, not `account.analytic.line`. SIMAF DTO is not CATSDB. A matching U32/I32 lane is not equivalent to matching a canonical business field, currency/scale, datetime timezone, analytic-line plugin provenance, account authorization or workflow action. D-XGP-8 finds WoA hours on `TimeSheet` rather than a child `TimesheetActivity`; the label "Stundenzettel" is a DTO label of an OGIT concept, not an independently minted conceptual class.

### 2.4 Schema field masks and population masks live in different universes

`WideFieldMask` indexes **field positions of a class**. Quack's `Mask` indexes an array of row-population mask planes. The two may share bitwise algebra without sharing sort or address identity. Proposal to bind the same scalar operator shape for NARS/IAM/CRM **must** keep their class/tenant/version/evidence semantics separate.

### 2.5 Method dispatch must not become authority

The available `HotPlug.covered` is capability *availability*. `ClassView`, `WideFieldMask` and OGAR declared ActionDef writes are neither permission nor approval. External DO continues through `graph-flow-action-ogar::GatedOgarHandler` / `ActionInvocation::commit_via`. A pure query must never be allowed to invoke its own action handler based on method naming or a positive reasoning confidence.

### 2.6 The 64/256 difference is real

`WideFieldMask` currently accepts up to 256 positions (per field universe), and its small representation is allocation-free. However `ClassView::ComputeEdge` and the existing `compute_dag_is_acyclic` implement a 64-position checker; a wide class can exceed that. A generic binding that claims automatic Odoo computed-method support must first handle or refuse positions >=64. Any plan that silently ignores them is unsound.

### 2.7 One Program, one terminal; one bulk runtime

Quack can fold a filtered query into one `Program`; R2IL procedures with multiple scalar folds execute multiple programs and combine scalars. This is already documented in `procs::margin`. Comparing instruction sequences as a global equality requirement is wrong; compare observable typed results and quantify measured cost. Neither path should store a row-sized intermediate in graph-flow Context.

### 2.8 ClassId widths and canonical/app fields are different scopes

`ClassView::ClassId = u16` addresses the canonical concept field universe; OGAR `ActionDef.object_class` and `codegen_manifest::ClassMethods.classid` have other source spellings/full class identity. The full V3 classid concept/appid split is class-specific. Do not route methods from SAP into an unrelated Odoo app simply because their canonical high half is `0x0103`. Bind both conceptual class basis and application/actor/source scope.

## 3. Scope decision and first executable experiment

**The minimal missing joint:**

```text
HotPlug / OgarAuthority / ResolvedReading
         +
ClassView.fields() / WideFieldMask / OGIT canonical URI
         +
Quack Binder (TableId/Col/validity/code)
         +
OGAR MethodDecl -> existing loco Inventory key -> FnAddr
         =
PreparedClass (semantic+versioned *sidecar*, no global registry)
         ├ pure read method -> Quack or known R2IL procs -> mask-risc
         └ action method -> OGAR hard gate -> external effect
```

First experiment should be **read-only SIMAF/CATS**, a selected pure method with a matching scalar oracle, no new contract dependency/opcode, using existing data/bitmasks. Reuse `CatsBinder` and `CatsSchema` as control. Reject unknown fields and unsupported optional NULL. Then bind a known R2IL `sum` procedure to exactly the same field mapping and measure answer parity. Successful results in this subset do not imply production SAP capability.

**Source status and falsification checklist** live in [the companion](../plans/class-method-hotplug-falsifiers-v1.md). Link to [OGAR #357](https://github.com/AdaWorldAPI/OGAR/pull/357), [#1496](https://github.com/AdaWorldAPI/lance-graph/pull/1496) (NAL plans merged), [#1497](https://github.com/AdaWorldAPI/lance-graph/pull/1497) (PLN draft) and [#1498](https://github.com/AdaWorldAPI/lance-graph/pull/1498) (Hyperon draft) for higher-order rule work; do not claim those experimental logic subsets are implemented as part of a generic HotPlug.
