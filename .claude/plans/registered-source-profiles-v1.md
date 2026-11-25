# Registered source profiles — register a known source's meaning once, discover only the rest (v1)

**Status:** PROPOSED / DOCS ONLY (2026-11-25). Research plan; nothing built.
**Census:** `.claude/knowledge/registered-source-profiles-census.md`. Every claim there cites a path and line, or is labelled HYPOTHESIS or OPEN.
**Neighbours:** `.claude/plans/adaptive-exact-decimal-binding-v1.md` (D-AED). Physical width and exponent are owned there and stay out of any semantic profile. D-OMR-AMEND-1 makes `BindStore` the method registry. D-LANESOCK covers the socket, frames and scope.

## 0. Why, and what was verified

Microsoft has already defined the meaning of Graph, Exchange Online, Azure and AD properties: `skuId` is an identity, `consumedUnits` is a seat count. Every license report, log aggregation and PSCustomObject-like projection should not have to rediscover that from each payload. The meaning should be registered once, independent of any particular report, and reused across reporting, aggregation, governance and reasoning. Discovery is the fallback for unknown or drifted fields.

**This plan does not bypass validation, physical layout adaptation or frame-qualified binding.**

The orchestrator re-read these claims at their source before relying on them:
- **Configuration overrides discovery but does not skip it.** intake `Shape::build` calls `detect_column` for every column (`lance-graph-intake/src/config.rs:139-140`), then merges the config. report-rs's workbook lanes declare every column and still pay for detection on each revision.
- **`Verdict`** is exactly `Agrees | Differs { detected } | DetectedOnly` (`config.rs:33-40`). `ColumnShape.detected` has one reader (`config.rs:184`).
- **Field identity depends on the table name.** Intake mints IRIs as `intake:{table}/{col}` (`lib.rs:528-531`) and creates a `BindStore` per table (`lib.rs:566`).
- **Quack** has `FieldKind::{I32, Code}` only (`lance-graph-quack/src/bind.rs:66-73`). Under `EmptyText` it maps `''` to code 0 (`intake/src/bind.rs:137`). VRL's `code_for` has no such branch.
- **report-rs reads a missing `prepaidUnits` as 0.** A test pins that behaviour and labels it a known limitation (`report-rs/src/lab.rs:1110-1130`).
- **Intake `append` already validates without discovery.** `observe` runs detection only on a misfit (`lib.rs:1103-1133`).

## 1. Framing (operator direction, 2026-11-25)

The plan keeps three lifetimes distinct:

| Lifetime | What it is | Existing owner | Changes when |
|---|---|---|---|
| **Semantic source profile** | Field identity, semantic type, identity/dimension/measure/relationship role, unit, resolution, exactness policy, absence semantics, temporal kind (snapshot, gauge, counter, event), source endpoint, API version and property path | `ClassView` / `FieldRef.predicate_iri`; `PropertySpec` + `field_meaning`; OGAR `ogar-az` `SchemaId`/`AttrDef` for endpoint and version | The source changes a meaning (versioned) or adds a property (extension) |
| **Registered mathematical method** | Lawful aggregation and behaviour, expressed as ON/WHEN/DO | `BindStore` `MethodBinding` (D-OMR-AMEND-1) | A new lawful method is registered |
| **Physical source binding** | Lanes, code spaces, frame, placement, slice width and exponent | DR `FrameStore` / `CodeTable` / `MapTable`; `LaneBinding`; the D-AED slice descriptor | Every append, promotion or exponent change |

Rules:
- **An append, a new observation, a magnitude promotion or an exponent change refreshes only the physical binding's witnesses.** It never triggers semantic rediscovery or method re-registration. Census §10 lists, per event, which witness refreshes and which must not.
- **Tenant independence.** Class identity, field semantics, units and methods do not depend on tenant values. Authorization, code spaces, coordinates, source ownership and frames stay per source (census §9).
- **Measures matter more than datatypes.** A seat count is a per-SKU snapshot, so summing it across dates is unlawful without a declared law. A SKU id is never summable. A counter may reset. A cost sums only within one currency.
- **No profile registry beside ClassView, BindStore and DR. No second runtime, codebook, schema engine or permanent materialized population.**

## 2. Reuse matrix (summary; full table in census §2)

- **Exists:**
  - field identity (`FieldRef.predicate_iri`, IRI-only basis digest);
  - physical type (`PgColumn`);
  - methods (`MethodBinding`);
  - frames and code tables (DR);
  - Graph endpoint, version and attributes (OGAR `ogar-az`).
- **Partial:**
  - semantic type (`SemanticType` has no quantity, duration, bytes, percent or count);
  - identity versus property (`AxisKind`), with no measure-versus-dimension distinction;
  - absence (`AbsentAs`, presence planes; report-rs `Observed` lives only consumer-side);
  - exactness policy (exact-decimal probe only);
  - scope (a bare `u64`, no authority).
- **Missing:**
  - unit;
  - temporal kind;
  - lawful aggregation axes;
  - a witness for a changed meaning under the same IRI (the digest excludes `PropertySpec`);
  - a source-scoped IRI supplied by the caller;
  - a declared-shape intake path.
- **Must stay out of a semantic profile:** physical width, exponent and slice bounds (D-AED), code values and code spaces, frames, scope, row counts.

## 3. Phases

Each phase stops at its gate, and a negative result is a valid outcome. Every phase is research or probe first.

### D-RSP-0 — census and plan
This file plus `.claude/knowledge/registered-source-profiles-census.md`, with board rows. **Shipped with this file.**

### D-RSP-1 — declared-and-validated intake path
**Question.** Census §3, path B. When every column has a fully declared `PgColumn` (`ty`, `encoding` and `absent` all set) and the key is configured, skip `detect_column` and `detect_key`. Still run `admit`, which checks that the declaration is lawful, and `lane_value` per cell, which is validation.

**Verdict.** Do not manufacture discovery agreement: a declared column whose detection did not run must not report `Agrees`. HYPOTHESIS:
- add `Verdict::Declared { validated_rows, profile_rev }`;
- make `ColumnShape.detected` an `Option<Detected>`.

Affected: the conservation assert, `key_column`, intake's tests and `select_bench`. No consumer outside intake reads `Verdict` (census §12).

**Path C (drift).** Detect only the column that fails, as `observe` already does.

**Gate.**
- A counter shows zero `detect_column` calls on a declared table.
- Results equal those of the discovery path on the same input.
- A declaration that no longer fits is refused (`ShapeChanged`), never silently re-detected.
- Disable-verify the skip.

### D-RSP-2 — source-scoped field identity and tenant reuse
**Question.** Let the caller supply source-scoped IRIs, so one Graph property has one identity under any table name. Let intake take a borrowed `BindStore` and `ClassView`, so one registered class (classid, fields, methods) serves many tenants' tables. Each tenant keeps its own `FrameStore`, `CodeTable` and `MapTable` (census §9 risks). This is a new API shape, not a new registry.

**Gate.**
- Two tenants share one profile with two stores and two code tables.
- A frame from one tenant is refused on the other (`ForeignFrame`).
- Codes never mix.
- Scope stays enforced. The scope issuer itself remains D-LANESOCK's open item.

### D-RSP-3 — measure semantics (decision needed)
**Question.** Where do unit, temporal kind and lawful aggregation axes live? Census §5 gives two candidates, which can be combined:
- **(a) Declare** them beside `semantic_type` in `PropertySpec`, read through `field_meaning` / `mask_where`. Open: whether `PropertySpec`'s `const` constructors can grow fields.
- **(b) Enforce** them by registering lawful aggregates as `MethodBinding`s. An unregistered aggregate over a gauge is then refused as an unknown method, following DT64's "UNKNOWN is never empty".

First read OGAR's `ActionDef` and ontology for any existing measure law (census OPEN 7).

**Gate.**
- Summing a snapshot across dates is refused.
- Summing seats across disjoint SKUs is admitted only under a registered law.
- `skuId` has no sum method.

### D-RSP-4 — meaning witness
**Question.** A changed meaning under the same IRI is undetected today. HYPOTHESIS (census §10): version meaning by IRI, so a changed meaning gets a new IRI. That reuses the basis digest and `PredicateChanged`, and avoids a second digest over `PropertySpec`.

**Gate.** An incompatible meaning change is refused at bind and is never silently accepted.

### D-RSP-5 — one field seam for Quack and VRL
**Question.** Both frontends consume the same lookup, owned by the source owner (intake): `predicate_iri → (column, code space, exponent, validity)`. Today Quack's `Binder` reads intake's shape by column name, while VRL reads `ClassView` labels plus a `code_of` closure (census §4). Neither frontend owns a source's datatype interpretation.

This is not a new IR: `Query` stays Quack's and `Recipe` stays VRL's.

**Semantic differences to document, never merge:**
- an unknown `=` literal is refused in Quack and matches no row in VRL;
- `''` under `EmptyText` gives code 0 in Quack, while in VRL an absent cell stays undecided. **Decided by the row type** (DECISION, operator, 2026-11-25; see below), so this is not a disagreement to settle between the frontends;
- VRL text has only `==` and `!=`, while Quack has LIKE;
- VRL keeps UNKNOWN as a separate count, while Quack folds validity into WHERE.

**Empty and absent literals are decided by the row type** (DECISION, operator, 2026-11-25):

| Row type | Value | Reads as |
|---|---|---|
| text | `''` (empty literal) | NaN: absent, never an empty-string value |
| boolean | `1` | true |
| boolean | `0` | false |
| boolean | NaN (absent) | false |

The seam carries this rule per row type, so Quack and VRL apply the same one. Open: how a boolean NaN reading false squares with VRL's three-valued UNKNOWN count for drop terms. The gate must test it on both frontends.

**Gate.** Both frontends bind through the seam and give identical counts and sums on one admitted source. Each remaining difference above is pinned by a test on both sides. The row-type rule for empty and absent literals is pinned on both frontends.

### D-RSP-6 — harvesting declarations
**Question.** Seed `PgColumn` (type, absence) and `FieldRef` IRIs from existing declarations:
- OGAR `ogar-az` `AttrDef` + `SchemaId { MsGraph, 2 }`;
- `MAILBOX_SCHEMA_V1`;
- report-rs's field policy and `mapping.rs`;
- Graph CSDL `$metadata`, Azure OpenAPI and PowerShell `[OutputType]` where they exist. None was found in report-rs (census §11).

A harvest never seeds measure semantics. Unit, gauge/counter/snapshot, counter resets, aggregation axes, decimal resolution, "0 means none or not reported" and cross-source identity each need separate evidence.

**Gate.** A harvested declaration equals the hand-declared one on the license fixture. No measure field comes from a harvest.

### D-RSP-7 — cross-source measure equivalence
**Question.** Graph, EXO and Az properties for one conceptual measure (mailbox storage, say) resolve to one canonical IRI and method only with all four of the following. Never by label match.
- (a) a canonical IRI, with each source property declared as a reading of it, for example through a lens `ClassView`;
- (b) an admitted exact conversion with provenance. Bytes ↔ KB is exact; 2-decimal GB is not (R39);
- (c) evidence of the same object identity (the ogar-az edges) and the same observation time;
- (d) a differential oracle on paired samples.

**Gate.** Without (a)–(d), the sources stay separate IRIs.

### D-RSP-8 — smallest vertical slice (census §7, §14)
**Where.** A lab crate, `crates/source-profile-probe`, outside the workspace. It composes intake, `quack::bind`, `lance-graph-vrl`, `dt64` and the exact-decimal probe's threshold.

**Fixtures.**
- F1, subscribedSku-shaped: one row with `prepaid_*` absent.
- F2, telemetry: `duration_ms` as decimal text.

**Arms.**

| Arm | What it is |
|---|---|
| A | Discovery (today's path) |
| B | Declared and validated |
| C | B, plus one width or exponent change |
| D | B, plus one unknown property |

Quack and VRL both run over the same admitted source.

**Proofs, each disable-verified:**
- no `detect_column` call on B;
- identical results across A and B;
- no extra population;
- no silent `f64`;
- absence is UNKNOWN, never 0;
- field and method identity are stable after promotion;
- a stale prepared layout is refused;
- an incompatible meaning change is refused;
- tenant frames are refused across tenants;
- D re-detects only the unknown column.

**Measured separately** (ns or µs per row, N ∈ {1, 64, 4096}, an asserted answer, cold phases "reported, not gated"):
- discovery;
- binding;
- validation and placement;
- physical adaptation;
- allocations and bytes.

The baseline is `select_bench`, which reports 0.46 µs/row including detection.

**Objective.** Amortize knowledge acquisition: register and bind once per profile, discover once per unknown property, and validate every cell always.

### D-RSP-9 — license report as consumer (report-rs, separate PR)
**Question.** A registered subscribedSku / user / assignment-state / service-plan profile, with the report-rs license logic catalogue (68 rules, the #77 scrubbed PowerShell reference) as the oracle.
- A missing `prepaidUnits` becomes UNKNOWN, not 0. This fixes the `lab.rs` limitation its own test pins.
- Mailbox and archive sizes stay exact bytes, with GB as presentation only.
- No finding infers purchased seats, costs or savings without evidence (`rpt-license-1.md`).

Source profile registration stays separate from tenant authorization and data availability.

## 4. Out of scope

- Physical width and exponent mechanics (D-AED).
- The scope issuer and authorization (D-LANESOCK, rbac).
- Any production change before its phase's gate.
- The report-rs XVERWEIS work.

## 5. Open (from census §8)

1. **Borrowed store for tenant reuse.** Whether intake may take a borrowed `BindStore` and `ClassView` (D-RSP-2).
2. **Measure-semantics home.** Where measure semantics live, and whether `PropertySpec` can grow fields (D-RSP-3).
3. **Meaning versioning.** Versioning by IRI or by a meaning witness (D-RSP-4).
4. **Slice descriptor home.** Where the slice descriptor lives (D-AED open item 2).
5. **Revision movement on append.** Whether `relation_map` / `coordinates` revisions move on intake append.
6. **Real data.** Real Graph, EXO and Az units and ranges; this needs operator-named samples.
7. **OGAR measure law.** Whether OGAR `ActionDef` or ontology already carries measure law.
8. **Empty literal: settled by row type** (D-RSP-5). Text `''` is NaN (absent); a boolean reads 1 as true and 0 or NaN as false. Still open: how boolean NaN → false meets VRL's three-valued UNKNOWN count.
9. **Harvest source.** Whether OGAR has a CSDL or OpenAPI harvest elsewhere is unknown.
