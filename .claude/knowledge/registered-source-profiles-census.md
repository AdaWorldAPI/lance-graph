# Source profiles: register a known source's semantics once, discover only the rest

READ BY: anyone designing source schema profiles, intake declared shapes, Quack/VRL field binding, measure semantics or cross-source identity; consumers of Microsoft Graph / Exchange Online / Azure / AD data (report-rs license report, log aggregation).

STATUS: READ-ONLY census (2026-11-25), input to `.claude/plans/registered-source-profiles-v1.md`. The orchestrator re-read the load-bearing claims against source before committing (see the plan's §0): `Shape::build` detects every column (`config.rs:137-147`); `Verdict` is `Agrees`/`Differs`/`DetectedOnly`; `ColumnShape.detected` has one reader (`config.rs:184`); intake mints `intake:{table}/{col}` (`lib.rs:528-531`) and a `BindStore` per table (`:566`); Quack `FieldKind` is `I32`/`Code` only; Quack maps `''` to code 0 under `EmptyText` (`intake/src/bind.rs:137`); report-rs reads a missing `prepaidUnits` as 0 and pins it (`RR/src/lab.rs:1110-1130`).


READ-ONLY census. Read at lance-graph `338283b` (origin/main), report-rs `0c49444`, and
OGAR `ac14414` (read for `ogar-az` / `ogar-dir-core` / `ogar-flow-m365` only). No cargo
was run. Every factual line cites `path:line`; the rest is labelled HYPOTHESIS or OPEN.
Paths: `LG/` = `/home/user/lance-graph/`, `RR/` = `/home/user/report-rs/`,
`OG/` = `/home/user/OGAR/`.

**Short answer.** Partly. Most of what a "profile" would hold already has an owner:
- identity is the ClassView `FieldRef.predicate_iri` plus the DT64 basis digest;
- meaning is `PropertySpec` / `SemanticType`, joined by `Resolved::field_meaning`;
- methods are `BindStore`;
- the physical shape is intake's `PgColumn`;
- frames and scope belong to DR.

Four things are missing:
1. **A way to hand intake a declared shape without running detection.**
   `Shape::build` always detects (`LG/crates/lance-graph-intake/src/config.rs:137-147`).
2. **A field identity that does not depend on the table name.** Intake mints
   `intake:{table}/{column}` (`LG/crates/lance-graph-intake/src/lib.rs:528-531`).
3. **Measure semantics.** Nothing records whether a value is a gauge, a counter or a
   snapshot, its unit, or whether it may be summed across dates or SKUs.
4. **A witness that the meaning has not changed.** The basis digest covers predicate IRIs
   only (`LG/crates/lance-graph-dt64/src/lib.rs:305-339`), so a changed `PropertySpec`
   under the same IRI is not detected.

None of the four needs a new registry. Where each one would belong is in §2 and §10.

---

## 1. WHY: what each consumer rediscovers today

1. **Intake runs detection on every column, even when every column is declared.**
   - `Shape::build` gathers each column's cells (`config.rs:139`), calls `detect_column`
     (`config.rs:140`), then merges the config over the result (`config.rs:141-147`).
   - report-rs's workbook lanes declare `ty`, `encoding` and `absent` for every column
     and also name the key (`RR/src/workbook_delta/lanes.rs:92-98`, `:135-153`).
   - They still pay for detection on every revision: each part is a fresh
     `Table::from_rows` (`lanes.rs:173-179`).
   - What detection costs: per column, one owned `Vec` of cells (`config.rs:139`) and a
     `present` vector (`detect.rs:158-162`), plus an `i64` vector for integer candidates
     (`detect.rs:178`).
2. **Every intake `Table` builds its own class and its own store.**
   - IRIs are `intake:{name}/{col}` (`lib.rs:528-531`), so the same Graph property under
     two table names (for example `rev-p0` and `rev-p1`, `lanes.rs:174`) gets two field
     identities.
   - A new `BindStore` is created and registered per table (`lib.rs:566-578`).
   - Fields are pure `FieldRef`s with no meaning attached (`lib.rs:556-564`).
3. **report-rs interprets the same Graph JSON in several places, each its own way.**
   - `lab.rs` reads `subscribedSkus` seats with `int()`, which turns a missing field into 0
     (`RR/src/lab.rs:213-215`, `:224-236`).
   - A test pins that missing and explicit zero cannot be told apart, and that `free()`
     then reports 0 (`lab.rs:1111-1130`).
   - `tenths()` goes through `f64` (`lab.rs:243-247`).
   - `licreport.rs` reads the same endpoint for names only (`RR/src/licreport.rs:128-160`).
   - `gov/observation.rs` has its own `Observed::{Known, Unavailable}`
     (`RR/src/gov/observation.rs:72-77`).
   - `field_mapping.rs` has a closed `OBSERVED` list of `ReportField`s
     (`RR/src/field_mapping.rs:46-50`; the enum is at `OG/crates/ogar-flow-m365/src/vocab.rs:140-159`).
   - The M365 plan states the problem itself: "each its own way … instead of adding a
     fourth way to call Graph" (`RR/docs/plans/rpt-m365-primitives-v1.md:10-23`).
4. **Quack and VRL each resolve names and literals separately.**
   - Quack goes through `Binder::field` and `Binder::code` by name
     (`LG/crates/lance-graph-quack/src/bind.rs:92-112`).
   - VRL goes through `ClassView::fields` labels and a caller-supplied `code_of` closure
     keyed by IRI (`LG/crates/lance-graph-vrl/src/lib.rs:458-479`, `:551-566`).
   - Intake implements both seams: `impl Binder for Table` (`intake/src/bind.rs:96-149`) and
     `Table::code_for` for VRL (`lib.rs:704-714`). That is two code paths over one code
     table.
5. **Decimals are not recognised at all.** Decimal text becomes text
   (`detect.rs:157-209`). The exponent is declared by a probe-local stand-in, `FieldExps`
   (`LG/crates/exact-decimal-probe/src/lib.rs:584-590`), whose doc says the production home
   is "intake's column shape".

---

## 2. Reuse matrix: who owns each responsibility

| Responsibility | Existing owner | State |
|---|---|---|
| Field identity | `FieldRef { predicate_iri, label }` (`LG/crates/lance-graph-contract/src/ontology.rs:467-479`); position = ClassView order (`class_view.rs:1061`); basis digest over the IRIs (`dt64/src/lib.rs:305-339`); `LaneBinding.refs` and `PredicateChanged` (`dt64/src/lane.rs:259-280`) | **PARTIAL.** The contract is right, but intake mints table-scoped IRIs (`lib.rs:528-531`). A profile needs source-scoped IRIs supplied by the caller. |
| Semantic type | `PropertySpec.semantic_type: SemanticType` (`contract/src/property.rs:41-59`, `:796-826`); joined per field by `Resolved::field_meaning` (`dt64/src/lib.rs:733-749`) | **PARTIAL.** `SemanticType` has `Currency`, `Date` and `Email`, and nothing for quantity, duration, bytes, percent or count. Intake attaches no `PropertySchema`. |
| Measure / dimension / identity / relationship | `AxisKind::{Identity, Property, Community, Capability}` (`dt64/src/dr.rs:340-347`); `PropertyKind` is obligation only (`property.rs:18-31`) | **PARTIAL.** Identity versus property exists as an axis kind. "Measure versus dimension" exists nowhere. Relationships are class views (`classid-on-do-call-shape.md` §0). |
| Unit and resolution | — | **MISSING.** The decimal resolution was planned on "the field" (`adaptive-exact-decimal-binding-v1.md:44`). No unit field exists anywhere. |
| Exactness and conversion policy | `Provenance` and `F64Policy` in the probe (`exact-decimal-probe/src/lib.rs:155-174`); `Arith::{Exact, Wrapping, Checked, Ieee}` (`contract/src/set_rules.rs:31-44`) | **PARTIAL** (probe only). A semantic profile may carry "decimal admitted" or "f64 admitted as a method" (plan :29-31). |
| Absence semantics | `AbsentAs::{Null, EmptyText}` (`contract/src/pg_row.rs:168-175`); presence plane `Arg::Present` (`dt64/src/lib.rs:169-181`); `Truth` (`set_rules.rs:49-55`); report-rs `Observed` (`gov/observation.rs:72-77`) | **PARTIAL.** Covers "absent". "Not collected" or "source not delivered" lives only in report-rs; lab.rs reads missing as 0 (`lab.rs:213-215`). |
| Temporal semantics (snapshot, gauge, counter, event) | — | **MISSING.** No gauge, counter, reset or snapshot marker was found in contract, dt64, quack, vrl or report (grep `gauge`/`Counter`/`summable`; hits in `algebra_law.rs` and `set_rules.rs` are about merge algebra). |
| Lawful methods | `MethodBinding { on, when, reads, effects, population, truth }` (`dt64/src/lib.rs:229-252`); `AlgebraLaw` (`contract/src/algebra_law.rs:36-79`); `set_rules::Semantics` (`set_rules.rs:72-81`) | **PARTIAL.** Laws describe the merge, never the measure. Nothing says "SUM of this field over dates is unlawful". |
| Provenance | `Provenance` (probe); report-rs `SourceObservation`, `EvidenceRef`, `SourceRevision` (`RR/src/source_graph.rs:15-19`) | **PARTIAL**, consumer-side. |
| Source endpoint, API version, property path | OGAR `SchemaId { family: MsGraph, version }` + `SCHEMA_V1: &[AttrDef { name, slot, kind, since }]`, `$select` derived from it (`OG/crates/ogar-az/src/lib.rs:15-16`, `:34-67`); `SchemaFamily::{AdDs, MsGraph, ExchangeOnline}` (`OG/crates/ogar-dir-core/src/schema.rs:22-29`) | **EXISTS** for directory attributes, in OGAR. This is the nearest thing to a registered source profile today. Not linked to ClassView IRIs. |
| Physical type and lane encoding | `PgColumn { ty, encoding, absent }` (`pg_row.rs:178-182`) | **EXISTS** (intake). |
| Physical width and exponent per sealed slice | planned: D-AED-2/3 (`adaptive-exact-decimal-binding-v1.md:83-114`); the descriptor's home is OPEN (`:194`) | **MISSING.** **Must NOT go in a semantic profile.** These are source-owned layout facts, witnessed with the frame (plan §1 `:54`). |
| Tenant or reader scope | `dr::Scope { id }` (`dr.rs:523-525`), a separate guard (`dr.rs:541-559`); `LaneRequest.scope` (`lane.rs:153-163`) | **PARTIAL.** The scope carries no authority (`d-lanesock…-v1.md:122`, `:147`). |

**What must stay out of a semantic profile:** physical width, exponent and slice bounds
(D-AED-3); code values and code spaces (`CodeSpaceId { class, property }`,
`dr.rs:1349-1352`, per store, §9); frames; scope; row counts.

---

## 3. Discovery, registered, drift: paths A, B and C against `Shape::build` and `merge`

What runs where today:
- **Schema discovery** = `detect_column` (`detect.rs:157-209`) plus `detect_key` when
  `config.key` is `None` (`config.rs:187-190`).
- **Value validation** = `lane_value` per cell (`lib.rs:139-170`), reached through
  `place` (`lib.rs:1245`). On append, `observe` runs it as well (`lib.rs:1103-1133`).
  `from_rows` calls `append_rows` directly, which skips `observe`; a misfit cell is then a
  `Cell` refusal (`lib.rs:1159-1164`, `:616`).
- **Physical range detection** = only the i32 / u32 split inside detection
  (`detect.rs:178-188`). For decimals it would be D-AED-7 (exponent and coefficient range,
  plan `:153-166`).

| Step | A: unknown source (discovery) | B: registered, data fits | C: registered, data drifted |
|---|---|---|---|
| Name check (`config.rs:122-127`), ragged check (`:128-135`) | run | run | run |
| `detect_column` per column | run | **skip.** Nothing reads `detected` except the verdict and `key_column`'s `..c.detected.clone()` (`config.rs:179-186`). | run, but only on the column that failed, as `observe` already does (`lib.rs:1121-1131`) |
| `merge` / `admit` (`config.rs:78-106`) | run | run `admit()` on the declared column (the profile may itself be stale) | — |
| `detect_key` | if no key is configured | skip: the key is declared | — |
| `lane_value` per cell | run | **must run**: validation is not discovery | fails on the first misfit, giving `ShapeChanged { was, now }` (`lib.rs:1127-1131`) |
| Range or exponent detection (D-AED) | run | **must run** per sealed slice: a physical fact, not semantics | promotion or new slice (D-AED-4, `plan:116-136`), not a semantic change |
| Verdict | `DetectedOnly` / `Agrees` / `Differs` | needs a state that is not `Agrees` (§12) | refusal; "a new shape is a new classid" (`consumer-surface…md:236`) |

What a skip loses (B):
- `ColumnShape.detected` with `support`, `total` and `witness` (`detect.rs:23-31`,
  `config.rs:44-49`).
- Any `Agrees` / `Differs` signal. That signal only says "config versus this batch". It is
  not drift evidence for the next batch, which `observe` covers.

HYPOTHESIS: B is safe exactly when every column has a fully declared `PgColumn`
(`ty`, `encoding` and `absent` all `Some`) and the key is configured, which is what
report-rs's lanes already supply (`lanes.rs:92-98`, `:135-140`).

---

## 4. Quack and VRL: what they share, what must stay apart

| Piece | Quack | VRL | Shared owner today |
|---|---|---|---|
| Field resolution | `Binder::field(table, name)`, giving `BoundField { col, kind, validity }` (`quack/src/bind.rs:76-98`) | label lookup in `ClassView::fields` (`vrl/src/lib.rs:458-479`); identity re-checked by IRI (`idents`, `:526-529`; `Refusal::OtherField`, `:121-123`) | the ClassView basis. Intake's `Binder::field` reads `self.shape` by column name (`intake/src/bind.rs:105-121`), not the view. |
| Field kinds | **`FieldKind::{I32, Code}` only** (`quack/src/bind.rs:66-73`); `Literal::{Text, Int}` (`:193-198`); unsigned, bool and date are not offered (`intake/src/bind.rs:12-15`, `:110-114`) | `i32` compare / sum and `u32` codes projected (`vrl/src/lib.rs:50-55`) | confirmed: only I32 and Code. |
| Codebook literal | `Binder::code` never mints; an unknown literal is `UnknownValue` (`quack/src/bind.rs:99-102`); LIKE with an empty cover is an answer (`:131-136`) | `resolve_text(frame, code_of)`; an unknown string "matches no row" (`vrl/src/lib.rs:541-566`) | DT `CodeTable` through `Table::code` (`intake/src/bind.rs:123-132`) and `Table::code_for` (`lib.rs:704-714`) |
| Empty-as-absent | `''` gives code 0 under `EmptyText` (`intake/src/bind.rs:125-127`) | `code_for` has no such branch (`lib.rs:704-714`) | **divergence**, see below |
| Decimal literal | none | probe: `DecimalRecipe::compile` rewrites the literal to an i32 threshold at the declared exponent and refuses `ExpChanged` (`exact-decimal-probe/src/lib.rs:625-670`) | `FieldExps` stand-in (`:588-590`) |
| Presence | `BoundField.validity` (`quack/src/bind.rs:84-90`) | presence lanes; the UNKNOWN count (`vrl/src/lib.rs:723-735`) | intake presence planes (`lib.rs:6-16`) |
| Frame / layout witness | `TypedQuery<Frame>` + `CodeWitness` (`quack/src/bind.rs:615-685`); `BoundQuery` frame check (`intake/src/bind.rs:77-93`) | `LaneBinding::borrow_all` checks frame, route, basis and scope (`vrl/src/lib.rs:14-18`); `Prepared.frame` (`:690-692`) | DR `FrameStore` |
| DT64 methods | none on the query path; WHEN is "a Quack WHERE" (`dt64/src/lib.rs:12-16`) | none | `BindStore` |

The seam both should share is a single `(predicate_iri → column, code space, exponent,
validity)` lookup, owned by the source owner (intake), which both `Binder` and VRL's
`code_of` / `FieldExps` call. HYPOTHESIS. This is not a new IR: `Query` stays Quack's
(`quack/src/bind.rs:15-20`) and `Recipe` stays VRL's.

**Semantic differences to document, never merge:**
1. An unknown `=` literal is refused in Quack and matches no row in VRL.
2. `''` under `EmptyText` gives code 0 in Quack; VRL's `code_for` has no such branch, so
   `== ""` resolves to "no code" and an absent cell stays undecided. Orchestrator reading:
   a semantic difference (VRL is three-valued over absence, Quack matches absence through
   code 0), not yet shown to be a defect; pin both with a test before changing either.
3. VRL text supports `==` and `!=` only, with no order (`vrl/src/lib.rs:67-70`). Quack
   has LIKE (`quack/src/bind.rs:570`).
4. VRL's three-valued UNKNOWN is a separate count (`vrl/src/lib.rs:723-735`). Quack folds
   validity into WHERE (`bind.rs:84-90`).

---

## 5. Measures and lawful aggregation

- **Exists:**
  - `AlgebraLaw` describes the merge only (`algebra_law.rs:1-29`, `:36-79`).
  - `set_rules::Semantics { arith, truth, sums, population }` guards rewrites
    (`set_rules.rs:72-81`); `Population::{Distinct, Bag}` (`:67-70`).
  - `MethodBinding.population` and `.truth` (`dt64/src/lib.rs:248-250`).
- **Not found:** gauge, counter, reset, snapshot-versus-flow, "not additive across time"
  or "not additive across SKU". OPEN whether OGAR's `ActionDef` has any; not read.
- **The ON / WHEN / DO contract.** `MethodBinding` has `on` (`dt64/src/lib.rs:238-239`),
  `when: Option<Guard>` (`:240-241`) and `effects` as DO (`:243-244`). The doctrine: the
  left classid is ON and the right `0xF###` is DO; WHEN is supplied by `register`
  (`LG/.claude/knowledge/classid-on-do-call-shape.md` §0).
  The BindStore is the registry (D-OMR-AMEND-1,
  `LG/.claude/board/entries/2026-11-18-d-omr-bindstore-is-the-registry.md:7-31`).
- **Where measure law would belong.** HYPOTHESIS, two candidates. Both reuse existing
  types.
  - (a) Field meaning. A "temporal kind" and a "summable over" set beside `semantic_type`
    in `PropertySpec`, read through `field_meaning` / `mask_where`
    (`dt64/src/lib.rs:733-761`). Then "fields lawfully summable over dates" is a
    `mask_where` mask.
  - (b) Methods. A lawful aggregate is a registered `MethodBinding` whose `reads` and
    `Effect.algebra` name the fold. An unregistered aggregate over a gauge is then refused
    as "unknown method", which follows from DT64's "UNKNOWN is never empty"
    (`dt64/src/lib.rs:43-47`).
  - (a) declares and (b) enforces. Neither is a new store.
  - OPEN: whether `PropertySpec` (a zero-dep contract struct, `const fn` constructors,
    `property.rs:61-80`) may grow fields without breaking `const` users.

---

## 6. The license report as oracle

Source: `RR/docs/reference/license-report/logic-catalogue.md`, 68 rules, the scrubbed
PowerShell reference that the "#77" references point to
(`README.md:1-17`; `RR/docs/plans/rpt-address-console-aduc-license-v1.md:123` says #77 is
"a scrubbed PowerShell reference catalogue, not runtime").

| Role | Fields or rules (catalogue line) |
|---|---|
| Identities | Graph `id`, UPN, onPrem UPN / Sam / DN (`:90`); AD record keyed by UPN (`:139`); `ExternalDirectoryObjectId` (`:171`); SKU GUID, service plan id (`:182-188`); group id (`:89`) |
| Seat counts (aggregatable, snapshot) | `subscribedSkus` (`:92`); report-rs reads `prepaidUnits.enabled/suspended/warning` and `consumedUnits` (`RR/src/lab.rs:224-236`). A per-SKU snapshot: summing across SKUs or across dates is not lawful without a declared law. HYPOTHESIS. |
| Relationships | `licenseAssignmentStates{skuId, assignedByGroup, state}`, `assignedPlans{servicePlanId, capabilityStatus}` (`:182-183`); manager (`:363`); transitive members (`:101`); AD ↔ Graph matching (R32 `:349`) |
| Assignment and provisioning states | `LicState::{Active, Disabled, Error, Other}` (`RR/src/gov/observation.rs:81-90`); `AssignmentOrigin::{Direct, Group}` (`:105-113`); R33 attribution (`:351-356`); `isLicenseReconciliationNeeded` (R44 `:397`) |
| Aggregatable quantities | mailbox `StorageBytes`, `ItemCount` (R26 `:321-324`); archive bytes parsed from text (R27 `:326-327`); usage report counts (`RR/src/source_graph.rs:24-31`, aggregate D7) |
| Lossy conversions in the oracle | `MailboxSizeGB = round(bytes/1GB, 2)` (R39 `:371-373`); `ArchiveSizeGB` likewise (R27). The profile must keep bytes exact and treat GB as presentation. |
| Evidence per finding | LIC-001: two or more paths to one SKU, `billable_seats: Some(1)` (`RR/src/gov/rules_lic.rs:329-342`); LIC-011..013: SKU catalogue facts, Teams-effective needs an `Active` path with no Teams plan disabled (`:277-287`); unknown input gives `Outcome::Unknown` (`:296-298`, `:1-5`) |

**Must not infer:**
- No "Confirmed" finding: it would need purchased seats or prices
  (`RR/docs/plans/rpt-license-1.md:13-19`).
- Six "nicht erhoben" fields (`rpt-license-1.md:20-23`).
- Never infer savings from redundant paths, expiry from seat counts, or departure from a
  disabled flag (`rpt-license-1.md:45`).
- Today's `lab.rs` violates the absence rule for seats (`lab.rs:1111-1130`): a missing
  `prepaidUnits` reads as 0 free seats. A registered `subscribedSku` profile must declare
  the absent policy so that this becomes UNKNOWN.

---

## 7. Smallest vertical slice (HYPOTHESIS; measurement design)

**Where:** a lab crate beside `exact-decimal-probe` (for example `source-profile-probe`).
It composes `lance-graph-intake` (`Table`, `ShapeConfig`), `lance-graph-quack::bind`,
`lance-graph-vrl`, `lance-graph-dt64` (`BindStore`, `PropertySchema` through
`field_meaning`) and the decimal probe's threshold. It adds no crate to production.
The tests live in its `tests/`.

**Fixtures (synthetic, committed):**
- F1, `subscribedSku`-shaped: `skuId`, `skuPartNumber`, `capabilityStatus`,
  `prepaid_enabled`, `consumed`. One row has `prepaid_*` absent.
- F2, telemetry: `ts`, `host` (code), `duration_ms` as decimal text, `bytes`, `status`.

**Arms:**

| Arm | What it is |
|---|---|
| A | `from_rows` with an empty `ShapeConfig` (today's discovery) |
| B | a fully declared `ShapeConfig` and key, with detection skipped (the §3 B path) |
| C | B, plus one batch whose `duration_ms` needs a finer exponent or a wider magnitude |
| D | B, plus one batch carrying an unknown property `x_new` |

**Proofs, each with a disable run:**
1. **No repeated discovery on a known path.** Count `detect_column` calls with a counter
   feature like `predicate_runs` (`lib.rs:653`). B gives 0 for F1 and F2.
2. **Identical results.** Quack `count` and `select`, and VRL `count`, `sum` and `rows`,
   are equal across A and B on the same admitted source. VRL through `Witness` as in
   `LG/crates/lance-graph-vrl/tests/intake.rs:24`.
3. **No extra population.** One `Table` per source and no second lane set. Assert the
   number of lane slices.
4. **No silent `f64`.** `duration_ms` arrives as a coefficient. A float-artefact column is
   refused (D-AED-7 gate, `plan:162-166`).
5. **Absence handling.** A missing `prepaid_enabled` is UNKNOWN in VRL
   (`prepare_unknown`) and three-valued in Quack, never 0.
6. **Stable field and method identity after promotion.** In C, the basis digest and the
   `MethodRef` are unchanged, and only the frame or descriptor moves.
7. **Stale layout refused.** A query or recipe prepared before C is refused: `Stale`
   (`intake/src/bind.rs:77-82`) and `ExpChanged` (probe `:654-670`).
8. **Incompatible semantic change refused.** Changing an IRI's `PropertySpec` refuses.
   This needs a meaning witness (§10).
9. **Tenant scope enforced.** Two tenants share one profile with two stores and two code
   tables. A frame of one is refused on the other (`ForeignFrame`, `dr.rs:367-369`).
10. **Unknown property (D).** Arm D routes `x_new` to discovery only. The known columns
    are not re-detected.

**Measure separately** (ns or µs per row, "reported, not gated" for the cold phases,
following LG CLAUDE.md):
- discovery: `Shape::build`, split into the detect and merge parts;
- binding: `register`, `route` and `bind_lanes`;
- placement and validation: `place` with `lane_value`;
- allocations: a counting allocator;
- bytes moved;
- execution.

The baseline is `examples/select_bench.rs`. It reports admission as 0.46 µs per row
"including detection" (`LG/.claude/knowledge/consumer-surface-data-and-class-as-config.md:316`)
and does not split detection out.

---

## 9. Profile reuse across tenants

**What is tenant-independent today:**
- The classid is the caller's: "the intake mints no concept"
  (`consumer-surface…md:129-131`).
- ClassView fields are keyed by `ClassId` alone (`class_view.rs:1061`).
- `BindStore` holds schema and methods by classid, immutable once registered
  (`dt64/src/lib.rs:31-40`, `:849-873`).
- The basis digest is over IRIs only (`dt64/src/lib.rs:305-339`).

So one classid, one ClassView and one set of `MethodBinding`s can serve every tenant, as
long as the IRIs carry no tenant value.

**What is per source (tenant) today:**
- `FrameStore` / `Frame`: issuer identity, no public constructor, two stores never equal
  (`dr.rs:485-505`; `d-lanesock…-v1.md:147`).
- `CodeTable`, bound to one `FrameStore` by identity (`dr.rs:1381-1396`). So
  `CodeSpaceId { class, property }` (`dr.rs:1349-1352`) shared across tenants still gives
  separate code tables. Codes never mix across tenants, provided each tenant has its own
  store.
- `LaneSource::frames()` is checked by identity (`d-lanesock…-v1.md:167-169`).
- `MapTable` (DR placement), owned per `Table` (`lib.rs:517-519`).
- `dr::Scope`, a separate guard that is never part of frame identity (`dr.rs:521-525`,
  `:541-559`).

**Where tenant scope is enforced:**
- `LaneRequest.scope` must equal every lane's scope (`lane.rs:159-160`;
  `LaneRefusal::Scope` `:189-194`).
- That check is equality on a bare `u64`. "A numeric `Scope` alone authorizes nothing"
  (`d-lanesock…-v1.md:122-137`). The sealed reader-scope issuer is not built
  (`:139-161`).
- Authorization kernels: `lance-graph-rbac` `authorize_scoped` / `authorize_memberships`,
  and `RbacAuthority::bind` (`contract/src/rbac_plug.rs`). Cited from
  `d-lanesock…-v1.md:126-130`; not re-read.
- Intake hard-codes `READER: Scope { id: 1 }` (`lib.rs:68`).
- report-rs's tenant fence lives in its own code (`RR/src/iam.rs`, cited at
  `d-lanesock…-v1.md:155-157`).

**Risks:**
- Intake's `BindStore` is per `Table` (`lib.rs:566`). Sharing one registered class across
  tenants means one `BindStore` per deployment, with per-tenant `FrameStore` / `CodeTable`
  / `MapTable`. HYPOTHESIS: this needs a `Table` constructor that takes a borrowed store
  and view. That is a new API shape, not a new registry.
- Intake builds `IntakeView` from table-local IRIs (`lib.rs:528-564`), so today two
  tenants' tables are two classes in effect, even under one classid.

---

## 10. Three lifetimes

| Event | Must refresh | Must NOT refresh |
|---|---|---|
| Append (rows fit) | `FrameRevisions.source` (`dr.rs:360-364`; "advance the data version", `lib.rs:26-29`); `CodeTable` advance for new strings (`lib.rs:1186-1210`, `dr.rs:1381-1388`); every `SelectionRef` / `BoundQuery` refused (`intake/src/bind.rs:77-82`) | classid, ClassView, `BindStore` registration, basis digest, methods |
| New observation, same shape | the same as append | detection (`observe` validates; it detects only on a misfit, `lib.rs:1121-1131`) |
| Magnitude promotion (physical) | the slice descriptor witness (D-AED-3, `plan:101-114`; home OPEN `:194`); frame | `SchemaRev`, basis digest, `MethodBinding`. HYPOTHESIS: "promotion or new slice", not `ShapeChanged`. Today an i32 overflow does give `ShapeChanged`, which the D-AED plan would change. |
| Exponent change (physical resolution) | the same; prepared decimal recipes refused (`ExpChanged`, probe `:612-617`) | meaning, methods |
| New property | a new basis, so a new classid ("a new shape is a new classid", `consumer-surface…md:145-148`; `BehaviourImmutable` `dt64/src/lib.rs:873`) or a new `SchemaRev` with `fields_at` history (`class_view.rs:1086-1105`); `LaneBinding` rebind (`StoreBasisMismatch` / `PredicateChanged`, `lane.rs:196-210`) | the existing fields' IRIs and methods (carried to the derived class: `register_derived`, `dt64/src/lib.rs:859-861`) |
| Changed meaning (same IRI) | **nothing detects it today.** The digest excludes `PropertySpec` (`dt64/src/lib.rs:305-311`); `field_meaning` takes the schema as an argument (`:735-749`) | — |
| Changed meaning (new IRI) | basis digest, so `PredicateChanged` (`lane.rs:205-210`) | — |

`relation_map` and `coordinates` revisions (`dr.rs:360-364`) move with DR mapping or
coordinate reassignment, not with semantics. OPEN which intake path moves them; only
`source` advancing is documented (`lib.rs:26-29`).

Recommendation (HYPOTHESIS): bind meaning by IRI versioning (a changed meaning gets a new
IRI). That reuses the digest and avoids a second digest over `PropertySpec`.

---

## 11. Harvesting source contracts

**What exists:**
- OGAR `ogar-az` declares Graph `user` attributes as `AttrDef { name, slot, kind, since }`
  under a versioned `SchemaId { MsGraph, 2 }`, and derives `$select` from it
  (`OG/crates/ogar-az/src/lib.rs:15-16`, `:34-67`).
- `AttrKind::{Str, Bytes, MultiStr, U32, Bool, …}`, where absence is the presence bit
  (`OG/crates/ogar-dir-core/src/schema.rs:54-69`).
- `ogar-az/src/mailbox.rs` (schema `MAILBOX_SCHEMA_V1`, `:12-14`) and `license.rs` exist.
- report-rs harvested the PowerShell script's 13 Graph calls with their `$select`
  (`logic-catalogue.md:86-101`) and the permissions per call (`:106-116`).
- The workbook field policy is a declaration: `field_id, predicate, type, min_role,
  headers` (`RR/fixtures/workbooks/customers-field-policy-v1.csv:1-5`).
- The customer mapping is a versioned header with validation (`RR/src/mapping.rs:22-36`).

**Not found:**
- No CSDL `$metadata` or OpenAPI harvesting. `grep -i 'metadata|csdl|openapi'` over
  `RR/src` hit only `Get-Mg*` labels in `flow_scratch.rs:55-77`, `identity_ws.rs:666` and
  the `lab.rs:18` comment.
- Azure REST specs and PowerShell `[OutputType]` declarations were not searched in OGAR
  beyond `ogar-az`. OPEN.

**What harvesting yields automatically versus what needs evidence:**

| Automatic from CSDL / OpenAPI / PS types (HYPOTHESIS) | Needs separate evidence |
|---|---|
| property name and path, EDM type (`Edm.Int64`, `Edm.Decimal`, `Edm.DateTimeOffset`, `Edm.Guid`), nullability, collection, navigation (relationship) | unit (bytes or KB; Graph reports mix them); gauge, counter or snapshot; counter resets; lawful aggregation axes; exact resolution of a decimal; whether `0` means "none" or "not reported" (`lab.rs:1111-1130`); cross-source identity equivalence (§13); report anonymisation (R26, `logic-catalogue.md:324`) |

A harvest can seed `PgColumn` (type and absent) and `FieldRef` IRIs. It must not seed
`PropertySpec` measure semantics.

---

## 12. A verdict for a registered intake

- The enum today: `Agrees` = config given and equal to the detection;
  `Differs { detected }`; `DetectedOnly` (`config.rs:32-40`).
- `merge` sets `Agrees` only after comparing `col == detected.col` (`config.rs:98-104`).
- The column count is conserved and asserted (`config.rs:156-165`).

If path B skips detection, reporting `Agrees` would invent evidence that never existed.

**HYPOTHESIS:** a fourth variant, for example
`Declared { validated_rows: usize, profile_rev: SchemaRev }`. Meaning: the shape was taken
from a registered declaration, `admit()` passed, and `lane_value` accepted every present
cell of this batch. No detection ran, so there is no `detected: Detected` field. That needs
`ColumnShape.detected` to become `Option<Detected>` (`config.rs:44-49`).

Affected consumers:
- the conservation assert (`config.rs:156-165`);
- `key_column`, which reads `c.detected` (`config.rs:179-186`);
- `IntakeRefusal::ShapeChanged` uses `detect_column` independently (`lib.rs:1127-1131`) and
  is unaffected;
- intake's tests (`LG/crates/lance-graph-intake/tests/intake.rs:218-248`, `:334-336`,
  `:650`, `:1657-1659`) and the `select_bench` example.

No consumer outside intake reads `Verdict`: report-rs imports `ShapeConfig` and `Table`
only (`RR/src/workbook_delta/lanes.rs:23`); vrl and exact-decimal tests import `Table` and
`ShapeConfig`.

---

## 13. Reusing one meaning across sources

**Identity mechanisms that exist:**
- **Property identity is `predicate_iri`.** Equal IRIs are equal identity; labels never
  are (the digest excludes labels, `dt64/src/lib.rs:305-311`).
- **Concept identity:** `canonical_concept_id(concept: &str)` maps a codebook concept name
  to its id (`contract/src/ogar_codebook.rs:833-837`). This is class-level, not
  measure-level.
- **Object identity across sources is evidence, not a merge:**
  - Graph `id` = EXO `ExternalDirectoryObjectId` is stated as the WHO of a mailbox record
    (`OG/crates/ogar-az/src/mailbox.rs:9-10`).
  - AD ↔ Entra are "two nodes, related by a `DirEdge`" (`OG/crates/ogar-az/src/lib.rs:5-7`;
    `sync_edges` `:542`).
  - `primaryMailboxId` decoding is strict and "never guessed" (`mailbox.rs:28-36`).
  - Report-rs's customer mapping refuses ambiguity (`RR/src/mapping.rs:12-15`).

**No measure-equivalence mechanism was found** (for example "Graph `storageUsedInBytes` ≡
EXO `TotalItemSize` ≡ Az metric X").

What an equivalence would need (HYPOTHESIS):
- (a) one IRI chosen as canonical, with each source property declared as a reading of it;
- (b) an admitted exact conversion with its own `Provenance`, of the
  `F64ShortestRoundTrip` kind (probe `:159-164`). Bytes ↔ KB is exact; GB at 2 decimals is
  not (R39);
- (c) evidence: the same object identity (via the edges above) and the same observation
  time;
- (d) a differential oracle on paired samples.

Without (a)–(d), the sources stay separate IRIs. A match by label (`MailboxSizeGB`) is
never evidence.

Where (a) lives: OPEN. Candidate: a lens ClassView whose `FieldRef`s name the canonical
IRI, per `lens_fields` (`class_view.rs:1084`). That is not a new registry.

---

## 14. Continuous log ingestion

**Today:**
- `append` → `observe` (validation only; `detect_column` only on failure, `lib.rs:1103-1133`)
  → `append_rows` → `place` + `lane_value`, CodeTable advance, frame advance
  (`lib.rs:1149-1210`).
- So intake's **append is already "registered, validate only"**. Discovery runs only in
  `from_rows`.
- A shape change refuses, with "a new class" (`lib.rs:1099-1102`). Under D-AED that would
  instead be a new sealed slice (`plan:116-136`).
- The binding is made once in `from_rows` (`lib.rs:579-597`). Whether `append` rebinds or
  re-checks the `LaneBinding` per read is OPEN (not read). The VRL witness checks the frame
  per run (`vrl/src/lib.rs:14-18`).

**Measurement design** (no existing benchmark separates these; `select_bench` reports
admission including detection, `consumer-surface…md:316`):

| Mode | What to time | Allocations / bytes |
|---|---|---|
| per-event discovery | `from_rows` with one row per event (detect, register, bind, place) | allocations per event; `BindStore` created per event (`lib.rs:566`) |
| per-batch discovery | `from_rows` once per batch of N | the same per batch |
| validation only | `append(batch)` on a table built once (`observe` + `place`) | cells: zero `detect_column` calls (a counter); string allocations only for fresh codes |
| physical adaptation | `append` that crosses an exponent or width boundary (D-AED-4; not built) | slice count, descriptor bytes, re-prepare cost of recipes |

Report per row in ns, with N in {1, 64, 4096} and an asserted answer, labelling each cold
phase "reported, not gated". The quantity to amortise is knowledge acquisition: register
and bind once per profile, discovery once per unknown property, and validation per cell
always.

---

## 8. OPEN

1. Whether intake may take a borrowed `BindStore` and ClassView, so that one registered
   class serves many tenants' tables (§9). This is an API shape, not built.
2. Where measure semantics live (`PropertySpec` fields or `MethodBinding` only), and
   whether `PropertySpec` can grow fields (§5).
3. Whether meaning changes are versioned by IRI or by a meaning witness (§10).
4. The home of the slice descriptor (D-AED open item 2, `plan:194`).
5. Whether `relation_map` / `coordinates` revisions move on intake append (§10).
6. Real Graph, EXO or Az units and coefficient ranges: need operator-named samples
   (D-AED-6, `plan:176-183`).
7. OGAR `ActionDef` or ontology for measure law: not read.
8. The VRL empty-absent divergence (§4 item 2): confirm with a test before calling it a
   defect.
9. CSDL / OpenAPI harvest: none was found, and whether OGAR has one elsewhere is UNKNOWN.
