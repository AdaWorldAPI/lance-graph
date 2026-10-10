# cross-glove-business-parity-v1 — Odoo × SAP over one Quack algebra

> Status: PROPOSAL (2026-10-10). Addendum to `sap-glove-quack-v1.md`. Nothing
> below is built. No code is authorized until the inventory (§1) and the
> contract (§3) have been reviewed.
> D-ids: D-XGP-0..7. Grades: `[G]` read in source at the pins below; `[H]`
> design hypothesis that needs its probe; `[S]` operator ruling required.

> **⊘ CORRECTION (2026-10-10, same day, operator review): read §C first.**
> §2, §3.1 and §3.3 proposed a new `lance-graph-contract::glove` module (a
> canonical basis type plus a pivot type). That module duplicates what is
> already shipped:
>
> - `OntologyRegistry` already joins `(bridge_id, public_name)` to one OGIT URI.
>   The same URI gets the same `entity_type_id` for every kind, attributes
>   included (`lance-graph-ontology/src/registry.rs:585-618`; test `:815`).
> - Quack `Binder` is the executable side, with a shipped production
>   implementation in `lance-graph-dir-sim/src/bind.rs:58`.
> - The CATS DTO→SoA→Quack path in `lance-graph-sap` is the reference
>   implementation. It never goes through `Binder`.
>
> The search was for a finished SAP↔Quack DTO registry, when it should have
> traced the bricks and the binding boundary between them. §C replaces §2,
> §3.1 and §3.3. The rest stays as it is unless §C says otherwise, and nothing
> above is deleted.

## 0. Pins and what changed since the baseline

| repo | sha | read |
|---|---|---|
| lance-graph | `6e40f43b` | `lance-graph-{sap,quack,mask-risc,report,report-ogar,ontology,contract}` |
| lance-graph-java | `cb02972` | `native/lgj-abi/{Cargo.toml,src/exports.rs,src/plan_lower.rs,src/exports/tests/}`, `docs/abi.md`, `java/src/main` |
| OGAR | `2fbae6e` | `ogar-vocab/{lib,ports,residual}.rs`, `ogar-from-ruff`, `ogar-from-schema`, `ogar-class-view`, `ogar-loco`, `ogar-r2il`, `ogar-proposal`, `ogar-dir-sim`, `ogar-action-handler` |
| ruff | `9407c748` | `ruff_python_spo/src/lib.rs`, `ruff_spo_triplet/src/ir.rs` |
| odoo-rs | `be322fa` | `od-ontology/src/{ogar,ogar_actions,recompute_dag}.rs`, `tests/`, `data/` |
| SIMAF / SIMAFPort | `a420b48` / `381d6c2` | file inventory; `Core/Database/zsimaf_tables.abap`, `Classes/zcl_time_dto_processor.txt` |

The baseline was read at lance-graph `0c7d4e1` and lgj `0b835f1`. Both are now
older than the pins above. Every `[G]` row of the baseline used below was
re-read at the new pins. Two baseline statements no longer hold:

1. **§6, "`0x000B` is the next free `APP_PREFIX`"** is stale. `0x000B` is
   `HubSpoPort` (`OGAR/crates/ogar-vocab/src/ports.rs:678`;
   `docs/APP-CLASS-CODEBOOK-LAYOUT.md:185`). The next free value at
   `2fbae6e` is `0x000C`. The ruling stays `[S]`.
2. **§3.3, "Home: start inside `lance-graph-sap` (`pivot.rs`). Promote to
   `lance-graph-contract` when a second consumer needs it."** Odoo is that
   second consumer. It may not depend on an SAP crate (the mission's
   constraint), so `Pivot` must start in `lance-graph-contract`. This is the
   only W0–W6 correction the addendum makes; see §6.

W0–W6 are otherwise preserved unchanged: their order, gates, STOP rule (G1)
and pins all stand.

## 1. Inventory (Phase 0)

The labels mean:

- **SHIPPED**: there is code, and a test exercises it.
- **PARTIAL**: there is code without a test, or the code is stubbed or narrower than the name suggests.
- **PROPOSED**: it exists only in docs or plans.
- **MISSING**: the search was made and found nothing; the search space is named.

### 1.1 Execution (lance-graph)

| capability | status | evidence |
|---|---|---|
| One evaluator: Quack lowers `Filter`/`Agg` → `mask_risc::Program` | SHIPPED | `lance-graph-quack/src/lib.rs:1383` `lower`; `Cargo.toml` "no second evaluator" |
| Text membrane: `Binder` (name → `BoundField{col, kind, validity}`, literal → code) | SHIPPED | `lance-graph-quack/src/bind.rs:77-104`; report `tests/quack_bind.rs:47` |
| `TableDeclaration` / `Registrar` | PARTIAL | `bind.rs:364-412`: name + width only, **no fields**; "no production implementation" (`bind.rs:53-55`) |
| Lane kinds `I32`/`U32`/`U64`/`Strided` | SHIPPED | `lance-graph-mask-risc/src/value.rs:80` |
| `Planes{n_rows, masks, lanes: &[LaneRef]}` (borrowed lanes) | SHIPPED | `mask-risc/src/ir.rs:73-80` |
| Grouped folds `Count/MinI32/MaxI32/SumSymI32` | SHIPPED | `mask-risc/src/ir.rs:510-528` |
| i64 / money fold | MISSING | `GroupFold` above; baseline G1 stands |
| CATS schema, bind, fold, pivots, BAPI sink | SHIPPED | `lance-graph-sap/src/*`, 6 test files |
| CATS hours scale | PARTIAL | `sap/src/bind.rs:65-98`: **batch-local** exact scale (max decimals in the batch), so two batches can carry different scales |
| "SAP and Odoo use the same field ABI" | PARTIAL | `sap/tests/view_convergence.rs` uses Odoo **`account.move`** (216 fields), not `account.analytic.line`. It proves both sides use `WideFieldMask`, not that any field corresponds |
| Report pivot / rotate as metadata | SHIPPED | `lance-graph-report/src/plan.rs:399-411`, `result.rs:321-332`; `tests/agnostic.rs:29,72,403`, `tests/zero_copy.rs:97` |
| Report lowers through Quack, not DataFusion | SHIPPED | `report/src/exec.rs:74-78`; `tests/reference_workload.rs:157` |
| Report destination | SHIPPED | `report/src/boundary.rs:129`; `tests/destination.rs` |
| OGAR class → report dimension binding | MISSING | `lance-graph-report-ogar/src/lib.rs`: the report is one opaque OGAR leaf (`present_mask = WideFieldMask::EMPTY`) |
| `Register128` (content-blind 128-bit register; meaning from SPOG context) | SHIPPED | `lance-graph-contract/src/register128.rs:56`; `hotplug.rs:231-245`; `crates/jc/tests/register128_bounded_stats.rs` |

### 1.2 Identity and schema (contract, OGAR, ontology)

| capability | status | evidence |
|---|---|---|
| `BILLABLE_WORK_ENTRY = 0x0103` | SHIPPED | `OGAR/crates/ogar-vocab/src/lib.rs:1977`, codebook `:2406` |
| Odoo `account.analytic.line` → `0x0103` | SHIPPED | `ports.rs:549`; test `planning_and_erp_converge_on_billable_work_entry` (`ports.rs:1286-1300`); odoo-rs `tests/classid_pins.rs:35-80` |
| SAP `CATS` → `0x0103` (`SapPort`) | MISSING | no SAP port or alias in `ogar-vocab/src/` |
| Field identity = (class, position) | PARTIAL | `contract/src/class_view.rs:1065` `ClassView::fields`; positions append-only (`ogar-class-view/src/lib.rs:33-47`). No single value type pairs class and ordinal |
| `ClassId` width | PARTIAL | `contract/src/class_view.rs:54` is `u16`; `ActionDef.object_class` is a `u32` classid. Two widths for one notion |
| Units as a field facet | MISSING | no unit type in contract (`property.rs`) or OGAR (`AttributeOptions` has `digits`/`currency_field` only) |
| Nullability | PARTIAL | OGAR `AttributeOptions.required` (`ogar-vocab/src/lib.rs:~964`); Quack `BoundField.validity`; none in contract |
| Field cardinality | MISSING | `Cardinality` exists for links only (`contract/src/property.rs:310`) |
| Enum / codebook domains | SHIPPED | OGAR `EnumDecl`/`EnumSource` (`ogar-vocab/src/lib.rs:860-886`) |
| Odoo `account.analytic.line` schema (base `analytic` module) | SHIPPED as data | `lance-graph-ontology/src/odoo_blueprint/extracted/analytic.rs:482`: `name, date, amount(Monetary), unit_amount(Float), product_uom_id, partner_id, user_id, company_id, currency_id, category, …` |
| Odoo `employee_id` / `project_id` / `task_id` on analytic lines | MISSING | they come from `hr_timesheet`/`project`; no extraction in `odoo_blueprint/extracted/` |
| Mapping-status vocabulary (proposed / hypothesized / unsupported) | PARTIAL | `ogar-proposal` `ProposalDraft.confidence: f32` (`lib.rs:64`); `ResidualRepresentation` (`ogar-vocab/src/residual.rs:30`); no grade enum |

### 1.3 Harvest and behavior (ruff, odoo-rs, OGAR)

| capability | status | evidence |
|---|---|---|
| Odoo fields, `compute=`, `@api.depends`, relations, `_inherit` → `ModelGraph` | SHIPPED | `ruff_python_spo/src/lib.rs:245-281`; tests `:536,:606,:656,:708` |
| Body facts (reads, writes, raises, calls, guarded writes) | SHIPPED | `ruff_python_spo/src/lib.rs:9-23`; test `:907` |
| `constrains` / `onchange` / `stored` | SHIPPED (struct only; not emitted as triples) | tests `:780,:795,:841`; `:851` |
| Selection values | MISSING | `ruff_python_spo/src/lib.rs:35-37` ("a remaining follow-up") |
| Method bodies as code | PROPOSED | `odoo-rs/crates/od-ontology/src/lib.rs:34-39` |
| ruff → OGAR `CompiledClass{class, facet, actions}` | SHIPPED | `OGAR/crates/ogar-from-ruff/src/mint.rs:50,81-227`; odoo-rs `ogar.rs:53`, tests `:175,:195,:300` |
| `ActionDef`, `ActionInvocation`, `KausalSpec` | SHIPPED | `ogar-vocab/src/lib.rs:396,515,610` |
| Authorization on `ActionDef` | MISSING | no role field (`lib.rs:396`); field-level `AttributeOptions.groups` only |
| Company scope | PARTIAL | `LokalSpec.company` on invocation (`lib.rs:695`); `company_dependent` on attributes; not on `ActionDef` |
| `RecomputeDag` | SHIPPED | `odoo-rs/crates/od-ontology/src/recompute_dag.rs:109`; 9 tests. `MethodKind::classify` keys on name prefix only (`:84`) |
| analytic-line model semantics tested in odoo-rs | MISSING | no Python fixture and no test; `examples/classid_pull.rs:64` says "not in the corpus" |
| odoo-rs → quack / report | MISSING | `od-ontology/Cargo.toml`: contract only, behind `fieldmask` |
| loco / r2il program + interpreter | SHIPPED | `ogar-loco/src/{program.rs:49,interpret.rs:197}`; `ogar-r2il` 25 tests. Every `Dialect` impl is test-only; neither depends on quack or mask-risc |
| `ogar-dir-sim` `Change` → `ExecutionPlan` | SHIPPED (types) | `ogar-dir-sim/src/plan.rs:23,87,110,197`; 8 tests |

### 1.4 Java and SAP sources

| capability | status | evidence |
|---|---|---|
| lgj lowering into `mask_risc::Program` | SHIPPED | `lgj-abi/src/plan_lower.rs:1,43`; 30 exports (`exports.rs:104-2537`) |
| lgj depends on quack | NO (by design) | `lgj-abi/Cargo.toml`: dev-dependency only, "the membrane must not depend on a consumer of the IR it serves" |
| lgj ↔ quack differential | SHIPPED | `src/exports/tests/lowering_convergence.rs` |
| lgj group sum | SHIPPED (one i32 measure, one key) | `exports.rs:2425` `lgj_plan_group_sum_i32`; `View.sumByGroup` |
| lgj pivot / `sql()` | MISSING | no `pivot` in native or Java sources; no `sql` method |
| SIMAF CATS DTO (the source `lance-graph-sap` harvested) | SHIPPED as source text | `sap/sources.tsv`; SIMAF `Schema/UniversalDtoPoc.TimeTracking.txt` |
| `BAPI_CATIMESHEETMGR_INSERT` call | PARTIAL | `SIMAF/Classes/zcl_time_dto_processor.txt:211`, mapping `:189-199` still says "[Implement Logic Here]" |
| CATSDB / `BAPI_CATS_INSERT` field list | MISSING | referenced by name only; no DD03L extract in either repo |
| IDoc | MISSING | the only hits are C# `IDocumentPickup`/`IDocumentDestination` |
| Any SAP runtime evidence | MISSING | the `.docx` UAT and "VM Simulation" files are narrative, not runs |

### 1.5 Two premises in the brief, corrected

- **"4×SPOG / Register128".** `Register128` is shipped and content-blind: its meaning comes from the SPOG context of the row it sits in (`register128.rs:1-40`). "4×SPOG" is defined nowhere. Searched: `OGAR` (crates, docs, board) and `lance-graph` (crates, `.claude`). OGAR docs name **3× SPOG quads** for Odoo in the 12-byte facet (`OGAR/docs/DISCOVERY-MAP.md:848-851`). So the register is a working representation for the reasoning side. It is **not** the carrier for deterministic business-rule execution, and this contract does not use it (§3.6).
- **The "SAP schema" in the fixture is not CATSDB.** `lance-graph-sap` harvested SIMAF's middleware DTO (`ty_s_time_entry_details` etc.): 23 fields, including security and sanitization metadata. Its parity is with SIMAF's time-tracking DTO, which is a stand-in for SAP. Claims about SAP proper need a CATSDB / `BAPI_CATS_INSERT` field list, which neither repo has.

## 2. The answer, short

> **The smallest stable landing contract is a versioned canonical field basis
> per concept, plus one graded pivot per glove onto it, executed by
> re-viewing borrowed lanes in canonical order so that ONE Quack `Query`
> value runs unchanged over either glove's population.**

The contract has three parts:

1. **Types** (zero-dep, `lance-graph-contract::glove`):
   - `CanonicalBasis`: an append-only field list for one concept.
   - `GlovePivot`: native position → canonical ordinal, graded `Exact`, `Converted`, `Hypothesized` or `Unsupported`.
2. **Data** (OGAR, data-as-config):
   - The basis for `0x0103` is minted next to `ClassView` in `ogar-class-view`, which already depends on the contract.
   - Each glove's pivot lives with the glove: SAP's in `lance-graph-sap` as numbers only (still OGAR-free), and Odoo's in `odoo-rs`, built from its ruff harvest.
3. **Execution** (no new evaluator):
   - A `Planes` value whose `lanes` slice is permuted into canonical order. These are pointer permutations whose size is fixed by the basis; nothing grows with row count.
   - A `CanonicalBinder` implementing Quack's existing `Binder`, so literals still resolve through each glove's own codebook.

The external interfaces stay independent because a glove sees only its own
native schema and its own pivot. Quack never sees a glove name. OGAR never sees
a lane.

## 3. The canonical binding contract (proposed, D-XGP-1)

### 3.1 Types — `lance-graph-contract/src/glove.rs` `[H]`

```rust
/// One field of a concept's canonical basis. Append-only within a version line.
pub struct CanonicalFieldSpec {
    pub ordinal: u16,
    pub name: &'static str,          // the stable key; never re-used
    pub domain: Domain,
    pub nullable: bool,
    pub cardinality: FieldCardinality,
}
pub enum Domain {
    Code { codebook: u32 },          // OGAR concept id of the value domain
    Integer,
    Quantity { unit: u32, scale_pow10: i8 },   // unit = OGAR concept id
    Money { scale_pow10: i8 },       // currency is a separate Code field
    Date,                            // calendar date, no zone
    Instant,                         // UTC instant
    Text,                            // edge-only; never a lane
}
pub enum FieldCardinality { One, Many }

pub struct CanonicalBasis<'a> {
    pub concept: u16,                // canon-high concept, e.g. 0x0103
    pub version: u16,
    pub fields: &'a [CanonicalFieldSpec],
}

pub enum MappingGrade {
    Exact,                           // same meaning, same domain
    Converted { conversion: u16 },   // same meaning; a named, total conversion
    Hypothesized,                    // proposed; never bound for execution
    Unsupported,                     // no counterpart; never bound
}
pub struct PivotEntry { pub native: u16, pub canonical: u16, pub grade: MappingGrade }

pub struct GlovePivot<'a> {
    pub app_prefix: u16,             // OdooPort 0x0002; SapPort [S]
    pub concept: u16,
    pub basis_version: u16,
    pub entries: &'a [PivotEntry],
}
```

Why `&'a` slices: SAP's pivot is a `const`, and Odoo's is built at codegen time
from a harvest. Both borrow. No glove needs `Vec` in the contract.

### 3.2 Laws (each one is a unit test, D-XGP-1)

1. **Validity.**
   - `canonical < basis.fields.len()` for every entry.
   - `native` values are unique.
   - `basis_version` matches the basis.
2. **Domain.**
   - `Exact` requires `native domain == canonical domain`.
   - `Converted` requires a registered conversion whose input matches the native domain and whose output matches the canonical domain.
3. **Execution gate.** Only `Exact` and `Converted` entries are ever bound. Binding a name whose entry is `Hypothesized`/`Unsupported`, or absent, is `BindError::UnknownField`, never a silent default.
4. **Composition.**
   - `odoo → canonical → sap` is defined only through the basis, never as a direct mapping.
   - The grade of a composition is the weaker of the two, in the order `Exact > Converted > Hypothesized > Unsupported`.
5. **Append-only.**
   - A basis version may only add fields.
   - Renaming or retyping a field is a new `version`, and every pivot declares the version it was written against.

### 3.3 Where the data lives

| artifact | home | depends on |
|---|---|---|
| `glove` types | `lance-graph-contract` | nothing (zero-dep stays) |
| `BILLABLE_WORK_ENTRY` basis v1 | `OGAR/crates/ogar-class-view` (`basis.rs`) | contract (already, git `main`) |
| SAP CATS pivot | `lance-graph-sap/src/pivot.rs` | contract only (OGAR-free, as the baseline requires) |
| Odoo analytic-line pivot | `odoo-rs/crates/od-ontology` | contract (`fieldmask` feature exists), `ogar-class-view` |
| pivot ↔ basis validation for SAP | the probe crate (§5) | both |
| conversions (scale, unit, date↔instant) | `lance-graph-contract::glove` as `const` descriptors; cold execution at bind | nothing |

### 3.4 Execution landing — why `Binder`, not `Registrar` `[G]`

`Registrar` registers a table name and a byte width, mints a `TableId`, and has
no production implementation (`quack/src/bind.rs:53-55,364-412`). It carries
no field identity and cannot express a mapping.

`Binder` is already the per-field resolution membrane:

- `field(table, name) -> BoundField{col, kind, validity}`
- `code(table, col, literal) -> u32`

Report already implements it over its catalog (`report/tests/quack_bind.rs:47`).
A `CanonicalBinder` over (`CanonicalBasis`, `GlovePivot`, the glove's codebook)
is a narrow adapter on an existing seam, with no new trait.

**`Registrar` stays as it is.** It becomes relevant only when a storage
contract exists (the brief's hard constraint).

### 3.5 One `Query`, two lane views `[H]`

> **⊘ Corrected per review (PR #1441).** A lane view only permutes borrowed
> `LaneRef`s, and `Binder` never sees row data, so neither one can convert a
> value. Conversion (instant→date, decimal rescale to the canonical scale, the
> UoM gate) therefore happens in **each glove's bind**. CATS already does its
> own conversions there (`sap/src/bind.rs:85-162`). The bind emits
> **source-side normalized columns**, owned by the batch, that are already in
> the canonical domain. The lane view then permutes pointers to those columns.
> Allocation contract: bind-time columns are O(rows), paid once per batch and
> owned by it, exactly as `CatsBatch` owns its columns today. Per-query state
> (scratch, lane view) is independent of row count. No conversion runs per
> query. One shipped fact needs a change for this: CATS currently uses a
> **batch-local** scale. To meet the canonical domain, the bind must rescale to
> the basis's fixed scale, or refuse a batch that cannot be represented at it.

```
canonical Draft  ──bind via CanonicalBinder(odoo)──►  Query_o ─┐
 (names, text)   ──bind via CanonicalBinder(sap)───►  Query_s ─┤ lower → Program_o / Program_s
                                                               │
Planes_o.lanes = [odoo lane for canonical 0, 1, …]  ◄─ pivot ──┤ (borrowed LaneRefs,
Planes_s.lanes = [sap  lane for canonical 0, 1, …]  ◄─ pivot ──┘  size fixed by basis)
```

Because both bindings address **canonical** `Col`s, `Program_o` and
`Program_s` differ only in the literal codes that each glove's codebook issued.
For a query with no textual literals they are the **same value**, and the
probe asserts that.

Group keys over `Code` fields come out as glove-local codes. Parity therefore
compares **decoded canonical values at the sink, never raw codes**. Two batches
can assign the same code to different values, so a raw-code comparison would
be a coincidence detector.

### 3.6 What deliberately stays out

- **`Register128` / SPOG context.** That is the reasoning carrier, and keeping
  it out of the business-rule path preserves the CE64/SPOFC separation. A later
  plan may give a canonical basis a register reading. That would be a separate
  `SlabReading` and would not change this contract.
- **CE64 / ARM evidence.**
  - A `Hypothesized` grade may cite evidence through a `ProposalDraft`
    (`ogar-proposal`), whose `confidence` field exists.
  - Evidence never promotes a grade. Promotion to `Exact`/`Converted` requires
    the parity probe (§5) on a reviewed fixture.
  - Statistical association is not equivalence.
- **Storage.** Quack runs over borrowed lanes. It does not persist, and
  transactional persistence stays in each ERP until a storage contract exists.
- **New ORM, query language, interpreter, pivot engine, report algebra.**
  - Report pivots are reused as they are (§4.2).
  - `GlovePivot` maps fields. It is not a report pivot: a report pivot assigns
    roles to cells, and the two must not be merged.

## 4. Native API convergence (Phase 3)

### 4.1 Java

lgj lowers through its own `plan_lower.rs` into the same `mask_risc::Program`
IR. Quack is a dev-dependency used for a differential test only, by design
(`lgj-abi/Cargo.toml`). The shared algebraic contract **already exists: it is
`mask_risc::Program`**. Routing lgj through Quack would add a dependency the
membrane deliberately refuses, and nothing has been benchmarked to justify it.

Proposal (D-XGP-6), in this order and stopping at the first no:

1. Extend `src/exports/tests/lowering_convergence.rs` with one case per
   canonical-bound query from the probe. lgj's lowering of the same predicate
   set must equal Quack's `Program`, byte-for-byte. This needs no new symbol
   and no ABI minor.
2. Only if a consumer needs it: a `GlovePivot`-shaped **lane permutation**
   applied when a store is opened over real data. No such constructor exists
   today. Every `lgj_rowstore_open*` export at `cb02972` builds a seeded
   fixture (`exports.rs:1505`: `n_rows, seed, edge_*`), so this is a new ABI
   minor and goes through `abi-membrane-warden`. The permutation is metadata
   and allocates nothing per row. Java sees field names, never ordinals
   (E3/E6).
3. Benchmark before any change to a shipped lowerer, with the
   `query-stage-profiler` gate. "Replacing a proven lowerer" is out of scope
   until a measurement shows a cost.

### 4.2 Report

Report's `CoordSpec::Field(FieldId)` is a bare ordinal (`report/src/ids.rs:11`).
A canonical basis gives that ordinal a meaning that two gloves share. The seam
(D-XGP-5, after the probe) is a report `Catalog` built from a
`CanonicalBasis`. Then a report plan written against canonical names executes
over either glove's `Planes` without changing `plan.rs`, `result.rs` or the
pivot/rotate code. The OGAR class → dimension binding stays MISSING until this
lands.

## 5. First vertical parity probe — `BILLABLE_WORK_ENTRY` (D-XGP-3)

### 5.1 The candidate basis v1 (`[H]`, for review)

The left column is the proposed canonical field. The two right columns show the
grade each glove's field gets against the base `analytic` module and the SIMAF
DTO. Nothing here is graded `Exact` until the probe and review agree.

| # | canonical | domain | Odoo `account.analytic.line` (base) | SIMAF CATS DTO |
|---|---|---|---|---|
| 0 | `work_date` | `Date` | `date` (Date) — Exact | `work_date_utc` (UTC instant) — **Converted** instant→date; zone rule is part of the conversion |
| 1 | `quantity` | `Quantity{unit, scale}` | `unit_amount` (Float, unit = `product_uom_id`) — Converted only when the UoM is time; else Unsupported | `hours_logged` (`catsquantity`, batch-local scale) — Converted (rescale to the basis scale) |
| 2 | `worker` | `Code` | `user_id` (res.users) — **Hypothesized** (a user is not an employee) | `employee_number` (PERNR) — Exact against an HR-employee domain |
| 3 | `customer` | `Code` | `partner_id` — Hypothesized (crosswalk res.partner ↔ KUNNR) | `customer_number` (KUNNR) — Exact |
| 4 | `company` | `Code` | `company_id` — Exact | `tenant_id` (string) — Hypothesized |
| 5 | `cost_amount` | `Money` | `amount` (Monetary) — Exact | — Unsupported |
| 6 | `activity` | `Code` | — Unsupported in base (`hr_timesheet`/product fields not harvested) | `activity_type` (LSTAR) — Exact |
| 7 | `project` | `Code` | — MISSING (needs `project`/`hr_timesheet` harvest) | `project_code` (`ps_posid`) — Exact |
| 8 | `task` | `Code` | — MISSING (needs `project`/`hr_timesheet` harvest) | `task_code` (`aufnr`) — Exact |
| 9 | `description` | `Text` | `name` — Exact | `notes` — Hypothesized |

(Corrected per review, PR #1441: `project` and `task` were one row. They are independent codes with independent lanes and need separate canonical fields; one ordinal would alias or lose one of them.)

**Honest result:** at these pins the only fields that can be bound on **both**
sides are `work_date` and `quantity` (both Converted on at least one side), and
`company` on the Odoo side. That is enough for one algebra-parity query and not
enough for a business claim.

### 5.2 Three parity levels, kept apart

| level | what passes | what is required | status at pins |
|---|---|---|---|
| **Schema** | every bound entry passes laws 1-5; domains, units, nullability, cardinality agree after conversion | basis v1 reviewed; both pivots validated | PROPOSED |
| **Algebra** | the same canonical `Draft` (`where work_date in [d0,d1]`, `sum quantity by work_date`) yields identical lowered `Program`s and identical decoded results on a fixture pair that encodes the same facts on both sides | §3.5; a reviewed fixture pair | PROPOSED |
| **Behavior** | the same preconditions, guards, state transitions and observable effects | an **independent oracle** for each side | Odoo: harvestable (KausalSpec exists) · **SAP: UNPROVEN.** There is no SAP runtime; the SIMAF BAPI mapping is unfinished (`zcl_time_dto_processor.txt:189-199`) |

Business parity is claimed for nothing in this plan. Behavior parity for SAP is
marked unproven and stays so until a real SAP system (or a recorded trace from
one) is available.

### 5.3 Positive tests

- P1 Basis + both pivots validate (laws 1–5).
- P2 Canonical `Draft` binds through both `CanonicalBinder`s. For a
  literal-free query the two lowered `Program`s are **equal**.
- P3 On the fixture pair (same N facts, entered once as Odoo rows and once as
  CATS DTO rows), `GroupSum(quantity by work_date)` equals the hand-computed
  oracle, and the two are equal after decoding at the sink.
- P4 Per-query scratch and lane-view allocation are independent of row count; bind-time normalized columns are O(rows), owned by the batch, built once (§3.5 correction) (mirror
  `sap/tests/no_alloc.rs`).

### 5.4 Negative controls (each one must fail, red-then-green)

- N1 **Wrong unit.** Odoo rows in a non-time UoM bound as `quantity` must be
  refused (`Unsupported`), never summed.
- N2 **Scale drift.** Two CATS batches with scales 1 and 2. Summing raw lanes
  without the basis rescale must disagree with the oracle; with the conversion
  it must agree.
- N3 **Wrong mapping.** Swap `worker` ↔ `customer` in one pivot: law 2 rejects
  it, or P3's oracle diverges.
- N4 **Hypothesized field.** A query naming `worker` must fail to bind on the
  Odoo side (grade `Hypothesized`) instead of silently using `user_id`.
- N5 **Raw-code comparison.** Comparing group keys as raw codes across the two
  gloves must be shown to give a wrong answer on a constructed fixture, so the
  decode-at-sink rule is load-bearing.
- N6 **Timezone.** A CATS instant near midnight UTC, in a zone where the
  calendar date differs. The conversion rule must be explicit, and a test pins
  which date it lands on.
- N7 **Execution order (behavior side, Odoo only).** For
  `_compute_general_account_id` vs `_check_general_account_id`, a
  `RecomputeDag` that orders the check before the compute must be detected.
  This is a falsifier of the DAG, not a parity claim.

### 5.5 Home of the probe

The probe gets its own workspace crate `crates/lance-graph-glove-parity`,
shaped like `lance-graph-sap`. It has path deps on `contract`, `quack`,
`mask-risc` and `lance-graph-sap`, and a git dep on `ogar-class-view`. Its
fixtures:

- `fixtures/odoo_analytic_line.tsv`: Odoo rows.
- `fixtures/cats_dto.txt`: the same facts as CATS DTO rows.
- `fixtures/odoo_pivot.tsv`: produced by the Odoo session.

It is test-only, with no library surface beyond `CanonicalBinder`, which
promotes to `lance-graph-quack/src/bind.rs` once a second consumer (report or
lgj) needs it.

## 6. Dependency graph (permitted edges)

```
                        lance-graph-contract  (zero-dep; + glove.rs types)
                       ▲        ▲        ▲          ▲            ▲
                       │        │        │          │            │
        lance-graph-mask-risc   │   ogar-class-view │       lgj-abi (contract, mask-risc;
                 ▲     ▲        │   (+ basis data)  │        quack = dev-dep only)
                 │     │        │        ▲          │
        lance-graph-quack ──────┘        │          │
           ▲      ▲                      │          │
           │      │                      │          │
  lance-graph-  lance-graph-sap          │      odoo-rs/od-ontology
  report        (contract, mask-risc,    │      (ogar-vocab, ogar-from-ruff,
                 quack; NO OGAR;         │       ruff_*, contract[fieldmask],
                 + pivot.rs numbers)     │       ogar-class-view)
                       ▲                 │
                       │                 │
               lance-graph-glove-parity ─┘  (probe; test-only)
               lance-graph-sap-sim (baseline W4; OGAR + sap)
```

Forbidden edges, each a STOP:

- `odoo-rs → lance-graph-sap` or any SAP crate.
- `lance-graph-sap → OGAR`. This is baseline §5, unchanged.
- `lance-graph-contract → anything`.
- `lgj-abi → lance-graph-quack` as a normal dependency.
- `quack → OGAR`.
- `report → ontology`. Today report does not depend on ontology, and the
  basis must reach report through contract types, not through ontology.

## 7. Business-logic workstealing (Phase 4) — what the Odoo session submits

Every harvested behavior item lands in exactly one class. Each class maps to
a type that already exists:

| class | existing carrier | lands where | executes |
|---|---|---|---|
| Declarative SPOG fact | ruff `ModelGraph` triples (`depends_on`, `emitted_by`, relations) | odoo-rs data, then OGAR `Class` | no |
| Reusable OGAR primitive | `KausalSpec::{Depends, Constrains, Onchange, StateGuard}` | `ActionDef.kausal` | gate only |
| Executable ActionDef | `ActionDef` with `exec` target | OGAR `CompiledClass.actions` | via `ogar-action-handler` / graph-flow, **not** Quack |
| Residual framework wrapper | `ResidualRepresentation` (`ogar-vocab/src/residual.rs:30`) | OGAR | no (hand port) |
| Hypothesized mapping | `PivotEntry{grade: Hypothesized}` + optional `ProposalDraft` | pivot file | never bound |
| Unsupported behavior | `PivotEntry{grade: Unsupported}` / no `ActionDef` | pivot file | never |

Boundaries preserved:

- **Transaction.** Quack computes populations and folds. It never commits.
- **Authorization.** `ActionDef` has no role field. RBAC stays with the
  `GatedOgarHandler` / rs-graph-llm gate, and this plan adds none.
- **Company scope.** `company` is a canonical `Code` field. Every canonical
  query that crosses companies must name it, and the probe includes one that
  does not (to show it returns the union, not a scoped answer).
- **Side effects.** `writes` / `calls` stay name-level facts and are never
  executed by the parity probe.

ARM discovery may produce candidate `Hypothesized` entries, and CE64 may record
their evidence. Neither may change a grade.

## 8. Gap table

| gap | effort | risk | falsification criterion |
|---|---|---|---|
| G-X1 `glove.rs` types + laws in contract | S (≈250 LOC + tests) | low; additive, zero-dep | law tests fail when each check is removed (disable runs) |
| G-X2 `BILLABLE_WORK_ENTRY` basis v1 in `ogar-class-view` | S | medium: getting the basis wrong poisons every pivot | review with both pivots attached; append-only test |
| G-X3 SAP CATS pivot (`lance-graph-sap/src/pivot.rs`), merged with baseline W1 | S | low | existing CATS suites unchanged + law tests |
| G-X4 Odoo analytic-line pivot from ruff harvest | M (Odoo session) | medium: base vs `hr_timesheet` fields | pivot validates; N4 refuses `worker` |
| G-X5 `CanonicalBinder` + canonical lane view | S | low; reuses `Binder`, `Planes` | P2 equal programs; N5 shows raw codes are wrong |
| G-X6 Conversions (instant→date with zone, decimal rescale, UoM gate) | M | **high**: silent wrong numbers | N1, N2, N6 |
| G-X7 Units as a domain facet (OGAR concept id) | S | medium: no unit vocabulary exists yet | `Quantity` with an unknown unit id fails validation |
| G-X8 Odoo `hr_timesheet`/`project` field harvest | M | low | `employee_id`/`project_id`/`task_id` present in extraction with source pins |
| G-X9 CATSDB / `BAPI_CATS_INSERT` field list | S if an export exists; blocked otherwise | high: currently a SIMAF stand-in | a DD03L or SE11 export with a source pin |
| G-X10 SAP behavior oracle | **blocked** | high | a real system or a recorded trace; until then behavior parity = UNPROVEN |
| G-X11 lgj convergence cases | S | low | `lowering_convergence.rs` byte-equality per case |
| G-X12 report `Catalog` from `CanonicalBasis` | M | low | a report plan runs over both lane views with identical cells |
| baseline G1 money / i64 fold | per baseline | per baseline | per baseline; `cost_amount` is not bindable for folds until it closes |

## 9. Ordered draft-PR boundaries

Each PR lands with its board rows in the same commit.

1. **PR-A (this one).** The plan docs, the baseline committed verbatim, and the
   handoff. No code.
2. **PR-B (lance-graph).** `contract/src/glove.rs`: types and laws only
   (G-X1). The zero-dep check stays green.
3. **PR-C (OGAR).** `ogar-class-view/src/basis.rs` with basis v1 for `0x0103`
   (G-X2, G-X7). Reviewed against the §5.1 table. Requires the operator to
   rule on the canonical field list.
4. **PR-D (lance-graph).** Baseline W1 (`pivot.rs`) re-expressed onto
   `glove::GlovePivot` (G-X3). CATS suites unchanged.
5. **PR-E (odoo-rs, the Odoo session).** The analytic-line pivot +
   `fixtures/odoo_pivot.tsv` + behavior classification (G-X4, G-X8). See the
   handoff.
6. **PR-F (lance-graph).** `lance-graph-glove-parity` probe: P1–P4, N1–N7
   (G-X5, G-X6).
7. **PR-G (lgj).** Convergence cases (G-X11).
8. **PR-H (lance-graph).** Report catalog from the basis (G-X12).

Baseline W2–W6 proceed independently of PR-E..H. W4's `lance-graph-sap-sim`
may consume `GlovePivot` once PR-D lands.

## 10. Open rulings `[S]`

1. **The `SapPort` `APP_PREFIX`.** `0x000C` is the next free value, not
   `0x000B` (see §0).
2. **Canonical basis v1 for `0x0103`.** Is `worker` an HR employee (PERNR /
   `hr.employee`) or a login user? This decides whether Odoo needs the
   `hr_timesheet` harvest before any worker parity.
3. **Whether `Quantity.unit` is an OGAR concept id.** `unit_of_measure` is
   `0x020B` (`contract/src/ogar_codebook.rs:625`); the alternative is a
   separate unit vocabulary.
4. **Whether the SIMAF DTO counts as "SAP" for schema parity.** The
   alternative is to block on a CATSDB export (G-X9).

## C. Correction: the DTO → registry → Quack path as it actually runs

> Read at lance-graph `6e40f43b`. Every row below cites a line that was opened
> in full. This section supersedes §2, §3.1 and §3.3. Three things are kept
> apart throughout:
>
> 1. the shipped CATS-specific DTO → Quack implementation;
> 2. the shipped generic registry and binder interfaces;
> 3. the proposed `WireSchema` / `WireBatch<S>` generalisation from the
>    baseline, which is **not shipped**.

### C.1 The current data path (source-grounded)

```
SIMAF ABAP/C# DTO (text columns)
   │  lance-graph-sap::schema   FIELDS[23]: ordinal, abap_group, technical_name,
   │                            csharp_name, native_type, optional, width, carrier   schema.rs:12-21,23
   │                            CatsSchema::new(class, category) → FieldRef "urn:simaf:cats:<group>:<name>"  :278-292
   │                            CatsSchema::resolve(name) → Col   (technical OR C# name)   :294-299
   │                            impl ClassView for CatsSchema                                :301-315
   ▼
   │  lance-graph-sap::bind     CatsBatch::bind(schema, [&[Option<&str>]; 23])               bind.rs:55-170
   │                            per-field adapters: decimal (batch scale) :85-105, PERNR numc :106-112,
   │                            ABAP_BOOL :113-123, UTC → U64 :124-131, else first-occurrence dict code :132-154
   │                            derived WORK_DAY lane (YYYYMMDD) :157-162
   │                            lanes() → [LaneRef; 24]   (borrowed, zero-copy)                :180-182
   │                            edge_value(ordinal,row) → text   (sink-only decode)            :194-224
   ▼
   │  lance-graph-sap::query    CatsQuery::prepare: Filter built from HARD-CODED Col constants
   │                            (EMPLOYEE, WORK_DAY) + numc/utc literals; NO Binder             query.rs:21-62
   │                            lower(Query{filter, Agg::GroupSumI32}) / Agg::Rows              :38-50
   │                            execute_into(Planes{masks: &[], lanes}) → i64 sums              :76-99
   ▼
   │  lance-graph-sap::edge     HASH_ORDINALS / BAPI_ORDINALS (selections of FIELDS)          edge.rs:10,18
   │                            bapi_sink(batch, kept) → only row materialisation              :171-206
   ▼
BAPI_CATIMESHEETMGR_INSERT parameter records (no SAP call)

                ─── the generic interfaces, not used by CATS ───

lance-graph-ontology::OntologyRegistry
   append_mapping(MappingProposal{bridge_id, public_name, ogit_uri, kind: Attribute{..}})   registry.rs:214
   URI ⇒ shared entity_type_id across bridges, kind-agnostic                                 :585-618, test :815
   resolve(bridge, public_name) → SchemaPtr :246;  resolve_uri(uri) :256;  rows_with_entity_type(id) :356
   SchemaSource trait (producer of proposals)                                                 schema_source.rs
lance-graph-quack::bind
   Draft{table, preds(Eq|Ne), Count|Rows} ──bind(&dyn Binder)──► Query                        bind.rs:195-357
   Binder{table, live, field → BoundField{col, kind: I32|Code, validity}, code}              :92-104
   TableDeclaration{name,width} ──register(&mut dyn Registrar)──► ResolvedTable{id,width}    :364-434
```

### C.2 Status, by brick

| brick | status | evidence | note |
|---|---|---|---|
| SAP DTO field table + `ClassView` | SHIPPED | `sap/src/schema.rs:12-315`; test `source_order_and_aliases_are_one_basis` :321 | class id is the caller's ("supplied by the caller's registry", :255); tests pass `42` |
| Native name → `Col` | SHIPPED | `CatsSchema::resolve` :294 | half of `Binder::field`, under another name |
| DTO → SoA lanes, codes, null sentinels | SHIPPED | `sap/src/bind.rs:55-182`; `tests/binding.rs`, `vocabulary.rs` | nulls are **sentinels** (code 0, `u32::MAX`, U64 0), not validity planes |
| Forward literal → code | PARTIAL | the forward `HashMap` is discarded after bind (`bind.rs:134-136`); only reverse labels are kept | `Binder::code` needs a forward lookup; a cold linear scan of `dictionaries[ordinal]` is enough |
| SoA → Quack fold | SHIPPED | `sap/src/query.rs:21-117`; `tests/fold.rs`, `no_alloc.rs` | builds `Filter` directly and **bypasses `Binder`** |
| SoA → SAP sink | SHIPPED | `sap/src/edge.rs:154-206`; `tests/edges.rs` | |
| Quack `Binder` (generic, field-level) | SHIPPED | `quack/src/bind.rs:92-104,288-357` | production implementation: `lance-graph-dir-sim/src/bind.rs:58` (`UserBinder`); test implementations: report `tests/quack_bind.rs:20`, quack `bind.rs:462` |
| `Draft` surface | PARTIAL | `Op{Eq,Ne}` only (`bind.rs:137`); `Want{Count,Rows}` only (:187) | no range predicates, no grouped aggregates, so `CatsQuery`'s `Ge/Le` + `GroupSumI32` **cannot** be expressed as a `Draft` today |
| Quack `Registrar` / `TableDeclaration` | PARTIAL | name + width only; no production implementation (`bind.rs:53-57,364-434`) | storage allocation, not field binding; it stays out of this seam until a storage contract exists |
| `OntologyRegistry` cross-bridge identity | SHIPPED | `registry.rs:214,246-262,356,585-618`; test `same_uri_across_bridges_and_namespaces_shares_one_template_id` :815 | this is the semantic registry. It already does what §3.1's "canonical basis" was meant to do |
| `MappingProposalKind::Attribute{predicate, semantic_type}` | SHIPPED | `ontology/src/proposal.rs:54-65` | field-level rows exist as a kind |
| `SchemaSource` (producer trait) | PARTIAL | trait at `schema_source.rs`; **zero implementations** (searched `lance-graph/crates`, `odoo-rs`, `OGAR/crates`) | `ogar-proposal` sketches one (`lib.rs:24`) |
| CATS attributes registered in the registry | MISSING | no `append_mapping` caller in `lance-graph-sap`; sap has no ontology dependency | |
| Odoo `account.analytic.line` attributes in the registry | MISSING | `data/ontologies/odoo/odoo-core.ttl` has no `analytic` and 0 properties; the Rust blueprint `odoo_blueprint/extracted/analytic.rs:482` is never appended | |
| Canonical attribute URIs for `0x0103` | MISSING | none registered | operator ruling (§C.5) |
| Units | PARTIAL | QUDT is hydrated as a ContextBundle (`hydrators/qudt.rs`); `SemanticType` (`contract/src/property.rs:796`) has no quantity/unit variant | corrects §1.2's "MISSING" at registry level |
| Registry row → `BoundField` translation | **MISSING** | no code joins `SchemaPtr`/`MappingRow` to a `Col` | **the gap** |
| `WireSchema` / `WireBatch<S>` | PROPOSED | baseline §3.1 | not shipped; not needed for the first probe |

### C.3 The exact missing binding

The semantic side and the executable side meet at **one function**, which does not exist yet:

```
canonical attribute name (Draft text)
   │  OntologyRegistry::resolve_uri(canonical_uri) → SchemaPtr.entity_type_id     (shipped)
   │  rows_with_entity_type(id).find(bridge_id == "sap") → MappingRow.public_name   (shipped)
   ▼
native field name ("hours_logged")
   │  CatsSchema::resolve(name) → Col                                               (shipped)
   │  FieldDescriptor.carrier → FieldKind (I32 → I32, U32 → Code)                  (one match)
   │  FieldDescriptor.optional → validity                                            (needs a live/validity plane; see below)
   ▼
BoundField{col, kind, validity}        ← this assembly is the missing adapter
```

Concretely, the missing adapter is two pieces. Neither one is a new registry, ORM, trait or evaluator:

1. **`impl quack::bind::Binder for` a CATS batch view** (in `lance-graph-sap`,
   which already depends on quack). It has the same shape as dir-sim's
   `UserBinder`:
   - `field` = `CatsSchema::resolve`, plus the carrier→kind mapping;
   - `code` = a cold forward scan of the batch dictionary;
   - `live` = `Mask(0)`, with the executor supplying an all-live plane 0.

   This is about 60 LOC, and it keeps `lance-graph-sap` OGAR-free and
   ontology-free.
2. **A canonical-name front** that turns a canonical attribute name into the
   glove's native name through the two shipped registry calls above, then
   delegates to (1). It lives where both the registry and the glove are
   visible: the probe crate, and later `lance-graph-sap-sim` (baseline W4).

Two shipped facts constrain the adapter. Neither is solved by adding a type:

- **Null representation.** CATS stores nulls as sentinels; `Binder` expects a
  validity `Mask`. For the first probe, bind only required fields
  (`validity: None`). An optional field gets a validity plane derived once at
  bind (cold, one bit per row) when it is needed, and not before.
- **`Draft` is Eq/Ne + Count/Rows only.** Ranges and grouped folds are a
  **frontend** gap in `quack/src/bind.rs`, not a lowering gap: `Filter` and
  `Agg` already have them (`lib.rs:218,1127`). The first probe therefore uses
  `Binder` for field and literal resolution, and builds the `Agg` from the
  bound `Col`s, as `CatsQuery` does today.

### C.4 Minimal parity probe (replaces §5.3–5.5 as step 1)

The probe does not need an Odoo batch. The sharpest first falsifier is:
**does the canonical-name path reproduce the shipped CATS path exactly?**

- **Q1 (identity of the binding).** Take the `CatsQuery::prepare` inputs.
  Bind the employee field through the registry front + CATS `Binder` using
  only canonical names. Use the registry fixture with the CATS attributes
  appended under bridge `"sap"`, with canonical URIs.
  - The resulting `Col`s equal `EMPLOYEE`/`WORK_DAY`/`HOURS`/`ACTIVITY`.
  - The lowered `Program` equals `CatsQuery::plan()`.
  - The sums equal `execute_into`'s.
- **Q2 (registry join).** Append the Odoo analytic-line attributes under bridge
  `"odoo"` with the same canonical URIs.
  - `rows_with_entity_type` returns both rows for each shared field.
  - A field present on one side only (`activity_type`; `amount`) resolves on
    that side and returns `None` on the other. It is never defaulted.
- **Negative controls.**
  - N-a: pointing the `"sap"` row for `quantity` at `employee_number` changes
    the `Program`, and the Q1 equality fails.
  - N-b: a literal absent from the dictionary gives `BindError::UnknownValue`
    and mints nothing (the same guarantee as `bind.rs:345-348`).
  - N-c: a canonical name with no `"sap"` row gives `BindError::UnknownField`.
  - N-d: binding an optional field without a validity plane is refused,
    rather than reading sentinel `0` as a value.
- **Home:** a test file in a crate that sees the registry and `lance-graph-sap`.
  The smallest is a new test-only crate `crates/lance-graph-glove-parity`
  (own workspace, like sap). It depends on contract, quack, mask-risc, sap
  and ontology; ontology's default features are light (`Cargo.toml`). It
  needs no OGAR dependency.

Algebra parity *across* gloves (§5.2) remains step 2. It needs an Odoo
population binder, which does not exist (odoo-rs has no lanes). Behavior parity
for SAP stays UNPROVEN, unchanged.

### C.5 What changes elsewhere in this plan

- **§2 / §3.1 / §3.3 are superseded.**
  - The canonical basis is a set of registry attribute rows sharing one URI per
    field, not a new contract type.
  - The "pivot" is the set of `(bridge, public_name) → uri` rows per glove.
  - Grades: rows with `confidence < 1.0` or `active == false` are never bound
    (`MappingRow`, `proposal.rs:81-95`). Conversion stays in the glove's bind
    (CATS already does exact decimals and UTC).
  - D-XGP-1 is re-scoped from "glove.rs types" to "CATS `Binder` + registry
    front + probe Q1/Q2".
- **§3.4 stands.** `Binder`, not `Registrar`. The addition is that `Binder` has
  a shipped production implementation (dir-sim) to copy.
- **§3.5 stands as a later step.** It needs Odoo lanes.
- **Open ruling (replaces §10.2–3).** Who mints the canonical attribute URIs
  for `0x0103`'s fields (OGIT NTO namespace vs OGAR `vocab/imports/ogit`)?
  The registry needs URI strings; it does not mint them.
- **The Odoo handoff is reduced.** The Odoo session emits
  `MappingProposal::Attribute` rows under bridge `"odoo"` (ideally as the first
  `SchemaSource` implementation, fed from the ruff harvest / blueprint), and
  the behavior classification. It does not build a `GlovePivot`. See the
  handoff's correction block.

### C.6 Two contract clarifications (operator review, 2026-10-10)

**C.6.1 A glove field mapping is not a report `pivot()`.** The two are used together but obey different laws, and the plan must never merge them.

| | glove field mapping (this plan) | `lance-graph-report` `pivot()` / `rotate()` |
|---|---|---|
| what it is | a mapping of **fields**: native `(bridge_id, public_name)` → canonical attribute URI → the glove's `Col` (`OntologyRegistry` rows + `Binder`, §C.3) | an assignment of **coordinates to roles** over an already folded cell space: `ReportPlan::pivot(rows, columns)` (`report/src/plan.rs:399`), `rotate` (`:411`), `ReportResult::with_roles` (`result.rs:332`) |
| when | cold, at bind, before lowering | after the fold; metadata over `Arc<CellSpace>` |
| what it changes | which lane a name reads; never values | which axis a coordinate is shown on; never cells. `PhysicalKey` excludes roles, so re-roling never re-folds (`plan.rs:313`) |
| laws | partial function; injective per glove (`native` unique); composes only through the shared URI, never directly glove-to-glove; no inverse for a selection; a missing or `confidence < 1.0` row is a bind error, never a default | a permutation of roles; on a plan, `rotate ∘ rotate` restores the roles by construction (Row↔Column, Page fixed; `plan.rs:411-419`, not separately test-pinned); on a result, `rotate` also clears `row_order` (`result.rs:321-329`), so a double rotation restores the roles but **not** a top-k row ordering; cell `(a,b)` equals rotated cell `(b,a)` (`report/tests/agnostic.rs:29`); measures and fold state unchanged |
| what it may not do | aggregate, filter, reorder rows | rename fields, change a lane, change a measure |
| how they compose | a field mapping produces the `Col`s that a report `CoordSpec::Field` / `Measure` reads. The report pivot then arranges the folded result. The order is fixed: **map → fold → pivot**, never pivot → map. |

The baseline's `Pivot<const N>` (`sap-glove-quack-v1.md` §3.3, positions of a wire shape → canonical ordinals, e.g. `BAPI_ORDINALS`) is a **field mapping** in this sense. Its "composition" law is the field-mapping law above, not the report's role permutation. Any code or doc that names both must keep the names distinct: here *field mapping*, there *report pivot*.

**C.6.2 `Converted` means a tested conversion, not a claim about two types.** A mapping entry may be bound as converted only when all of these hold:

1. **The conversion is named and lives in the glove's bind.** For example, CATS `decimal` (`sap/src/bind.rs:237`) with the batch rescale (`:85-105`), and `utc` (`:274`) with the derived `WORK_DAY` (`:157-162`). Registry rows hold no conversion code.
2. **An oracle test pins it on real values.** The test covers both sides of every boundary the conversion has: scale 0/1/2 across batches, the midnight-UTC date edge, and the ±range limits (the C# `[0.01, 24.00]` range, `bind.rs:252`).
3. **A disable run proves the test is load-bearing.** Removing the conversion must turn the test red. This follows the repo rule: commit first, then disable, assert the disable patch applied, then restore.
4. **Unit and identity questions are conversions too, and are not settled by type.**
   - `unit_amount` is a time quantity only when `product_uom_id` is a time unit. That is a row predicate, not a field type, so the mapping is `Converted` only with a UoM gate that refuses non-time rows (negative control N1).
   - An HR employee (PERNR, `hr.employee`) and a login user (`res.users`) are **different identities**. No conversion between them exists without an explicit, tested crosswalk table, so `worker` stays `Hypothesized` (`confidence < 1.0`, never bound) until the ruling and a crosswalk exist.

Any entry that fails one of the four points is `Hypothesized` by definition. In the registry design (§C.5) that means `confidence < 1.0`, and the canonical-name front refuses to bind it. The negative controls N1–N7 (§5.4) and N-a…N-d (§C.4) are the acceptance gate for every `Converted` entry. The positive demo is not.

### C.7 Order of work after this PR (operator review)

1. **This docs PR.** Merged after reviews.
2. **D-XGP-1.** The minimal adapter of §C.3: a CATS `impl Binder`, the canonical-name front, and the clarification laws of C.6 as tests. It reuses `OntologyRegistry` and `Binder`; no `glove.rs` module unless review asks for one again.
3. **D-XGP-2.** The canonical attribute URIs for `0x0103`, after the §C.5 ruling.
4. **D-XGP-3.** The probe: Q1, Q2, N1–N7 and N-a…N-d.
5. **D-XGP-4.** Only then does the Odoo session start from its handoff. The handoff stays gated until steps 2 and 3 have merged.

### C.8 D-XGP-1 landed on a branch: what is now true (2026-10-10)

- **Shipped (test-pinned, unmerged):**
  - `lance-graph-sap::binder::CatsBinder`, an `impl Binder` over a bound `CatsBatch`, plus `CatsBatch::dictionary_code`.
  - `lance-graph-glove-parity::{native_field, CanonicalBinder}`.
  - Probe Q1, Q2, N-a and N-c (§C.4) as tests.
  - N-b (an unknown value mints nothing) and N-d (optional fields refused) as tests in `lance-graph-sap/tests/binder.rs`.
  - Every guard disable-verified: 6 in sap, 5 in the front.
- **The fold half of Q1 is still built by hand.** `Draft` cannot express ranges or grouped folds, so the probe writes them against the lanes the binder resolved. Closing this is a Quack frontend change (`bind.rs` `Op`/`Want`), not a lowering change.
- **Registry identity is keyed by `(bridge_id, public_name)`.** A second proposal for the same native field is idempotent when its checksum matches. Producers (the Odoo `SchemaSource`) must make the checksum cover the URI, or a re-mapping is silently a no-op.
- **`MappingRow::active` is not consulted.** No registry path sets it false, so a guard on it could not be tested.
- **Correction to §0.1 / §10.1.** `0x000C` is claimed by `HiroPort` (OGAR #333 and the contract mirror on `claude/hiro-app-prefix`, both unmerged at this writing). The next free app prefix for a `SapPort` is `0x000D`; `0x0006` stays deliberately unallocated. The ruling remains the operator's.
- **Still open:**
  - the canonical attribute URIs (§C.5);
  - an Odoo population binder for cross-glove algebra parity (step 2, not built);
  - SAP behavior parity, still UNPROVEN.

### C.9 Lockstep is deprecated: everything plugs through `hotplug.rs` (operator, 2026-10-10)

> **⊘ This supersedes the lockstep framing in §0.1, §6, §10.1, §C.5 and §C.8 ("next free `APP_PREFIX`", "who mints the canonical attribute URIs").**
> Those questions assumed paired OGAR + contract-mirror allocations made in step. That pattern is deprecated. A consumer declares what it plugs; the authority resolves it.

**The pattern, as read in source** (`lance-graph-contract/src/hotplug.rs:1-48,50-62,700-705`; authority `lance-graph-ogar/src/lib.rs:535`; reference consumer `HubSPO-rs/crates/hubspo-port/src/lib.rs`):

- The consumer declares one `HotPlug { consumer, classids, covered }` const, and an `RbacPlug` when it has roles. Both take canonical concept ids read from `ogar_vocab::class_ids`, never literals.
- `OgarAuthority::activate` resolves the plug into an `Activation`: concepts, capabilities and a storage reading per plugged id. It fails closed with a named `ActivationDrift`.
- The app prefix never enters a plug. A consumer composes `concept << 16 | its prefix` itself for rendering, so the prefix is not a gate for parity work.
- Field identity is the authority's `ClassView` of the concept, by position and `predicate_iri` (`ogar-class-view/src/lib.rs:345-357`). It is not a separately minted URI list.

**What this changes, measured at OGAR `main` `7aaf824`:**

1. **The canonical basis for `0x0103` already exists.** It is OGAR's promoted `BillableWorkEntry` (`ogar-vocab/src/lib.rs:3515`), lifted by `ogar-class-view` (`lib.rs:79,197`). Its fields in `ClassView` order:
   - field 0 is the attribute `billable` (boolean);
   - fields 1-12 are the family edges `project → Project`, `about → ProjectWorkItem`, `performed_by → Worker`, `duration → Duration`, `priced_by → RatePolicy`, `cost_center → CostCenter`, `classified_by → TaxPolicy`, `materializes_as → InvoiceLineCandidate`, `approval_state → ApprovalState`, `tenant → Tenant`, `audit_trail → AuditTrail`, `posted_by → PostingAction`.

   D-XGP-2 therefore mints nothing. Each glove maps its native fields onto these 13 positions.
2. **Gap: the class has no temporal role.** There is no work date and no period. The date-range half of probe Q1 (`work_day`) therefore has no canonical field to bind to. Adding one is an OGAR change to the promoted class, made by the authority. It is not a consumer-side mint, and this plan does not make it.
3. **Most canonical fields are edges, not scalars.** `performed_by`, `duration`, `tenant` and `approval_state` point to other concepts. Mapping CATS `employee_number` (a PERNR code) onto `performed_by` binds the code lane of an edge. Mapping `hours_logged` onto `duration → Duration` needs the `Duration` concept's own reading of a quantity. Those mappings are `Converted` only under §C.6.2, and `Hypothesized` until then.
4. **No capability table covers `0x0103`.** The domain tables are OCR, geo, healthcare and document (`ogar-vocab/src/capability_registry.rs:193-209`). A SAP or Odoo `HotPlug` on `BILLABLE_WORK_ENTRY` therefore activates as `NoCapabilitiesFor(0x0103)`. HubSPO pins the same state in a test: capabilities are declared only once the consumer implements them.
5. **The fixtures in `lance-graph-glove-parity` must follow the authority.** The test basis `ogar.GloveFixture:{worker,workDay,quantity,activity,costAmount}` does not match `BillableWorkEntry`. The next probe step binds through the authority's `ClassView` field names instead, and keeps only the binder mechanics proven so far.

**Revised D-XGP-2:** declare the SAP and Odoo plugs (`HotPlug` consts) in a crate that sees OGAR, pinning today's `NoCapabilitiesFor` the way HubSPO does. Then map CATS and Odoo native fields onto the 13 `BillableWorkEntry` positions, each graded per §C.6.2. `lance-graph-sap` stays OGAR-free.

**Open for the operator:** whether `BillableWorkEntry` gains a temporal role (work date / period), and of which kind. That is an OGAR authority change.
