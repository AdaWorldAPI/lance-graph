# cross-glove-business-parity-v1 — Odoo × SAP over one Quack algebra

> Status: PROPOSAL (2026-10-10). Addendum to `sap-glove-quack-v1.md`. Nothing
> below is built. No code is authorized until the inventory (§1) and the
> contract (§3) have been reviewed.
> D-ids: D-XGP-0..7. Grades: `[G]` read in source at the pins below; `[H]`
> design hypothesis that needs its probe; `[S]` operator ruling required.

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
| 7 | `project` / `task` | `Code` | — MISSING (needs `project`/`hr_timesheet` harvest) | `project_code` / `task_code` — Exact |
| 8 | `description` | `Text` | `name` — Exact | `notes` — Hypothesized |

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
- P4 Scratch and lane-view allocation are independent of row count (mirror
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
