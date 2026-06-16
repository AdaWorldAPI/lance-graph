# OpenProject AR-shape extraction — 100 %-coverage `ruff_ruby_spo` + `op-surreal-ast` bridge (v1)

> **Status:** PROPOSAL (2026-06-15). Branch: `claude/openproject-ar-shape-extraction-v1`.
>
> **Scope:** make `AdaWorldAPI/ruff::ruff_ruby_spo` emit `Triple{s,p,o,f,c}` ndjson
> for **100 %** of the 78 distinct class-body DSL names that appear in
> `OpenProject/app/models` (measured exhaustively this session, §2), land the
> predicate-vocabulary extension in `ruff_spo_triplet`, wire `op-surreal-ast`
> in `adaworldapi/openproject-nexgen-rs` as the SurrealQL `DEFINE TABLE/FIELD`
> emitter that consumes the resulting ndjson. The lance-graph spine
> (`graph::spo::odoo_ontology::parse_triples`) absorbs the extended predicate
> set unchanged — ndjson is the firewall.
>
> **Why now:** user directive 2026-06-15 — *"include everything from AR shape,
> no leftovers, any leftovers take multitude of time"*. The
> `ruff_ruby_spo::lib.rs` header explicitly names this as its purpose
> (*"Point `extract` at an OpenProject `app/models/` tree"*) and the current
> `RubyClass.associations: Vec<String>` covers only ~21 % of declarations
> (351 / 1696) while dropping 305 nested association options to flat names.

## 1. Canonical references (read instead of re-derive)

- `.claude/knowledge/aerial-arm-ruff-spo-codegen-synergies.md` —
  the three-frontend / one-substrate / two-codegen bracket; names `ruff_spo_triplet`
  as the truth/triple contract, `ruff_ruby_spo` as the Ruby/Rails proposer leg,
  `ruff_python_dto_check` as the Python sibling, and **D-ARM-SYN-1**
  (the council-gated predicate-vocab extension this plan instantiates).
- `.claude/plans/normalized-entity-holy-grail-v1.md` — the trunk that the
  AR-shape extraction feeds: 5 verbs (`resolve/hydrate/classify/align/think`),
  `NormalizedEntity` carrier with OGIT slot, three execution contexts
  (interactive / bulk / periodisch). **The AR shape IS the typed pipeline that
  replaces rows-and-joins.**
- `.claude/plans/polyglot-container-query-membrane-v1.md` §2.2 — the
  surrealdb-ast / C16b DDL builders / `op-surreal-ast` consumer surface in
  openproject-nexgen-rs; **D-PG-5** (DDL ⇄ registry bridge) is what D-AR-5/6
  below complete.
- `.claude/knowledge/odoo-extraction-tools-v1.md` — the precedent sibling
  pattern (Python AST → `OdooEntity` consts → `lance-graph-contract::callcenter::ogit_uris`).
  The Ruby/Rails leg mirrors this exactly, language-swapped.
- `AdaWorldAPI/ruff/crates/ruff_ruby_spo/src/lib.rs` — SCAFFOLD with
  `todo!()`s + a doc-comment naming the exact Rails construct each
  extractor must read.
- `AdaWorldAPI/ruff/crates/ruff_spo_triplet/src/` — language-agnostic SPO
  expansion; `Triple{s,p,o,f,c}` mirrors `lance_graph::graph::spo::odoo_ontology::OntologyTriple`
  field-for-field; ndjson loads via `parse_triples` with **no transform**.

## 2. The 100 %-coverage surface — 78 distinct class-body DSL names

Measured exhaustively from `OpenProject/app/models/**/*.rb` (941 files, 1696
class-body DSL declarations). Full census saved at
`/tmp/cov-repro/openproject-78-name-surface.txt`. Reproducible via
`find app/models -name '*.rb' -print0 | xargs -0 grep -hE '^  [a-z_]' | grep -vE '^  (def|if|else|elsif|case|when|unless|begin|rescue|ensure|end|return|raise|yield|super|nil|true|false|do)'` then `sort | uniq -c | sort -rn`.

**Classification — 67 emit predicates + 11 non-fact scope markers:**

| Category | Names (frequency) | Predicate emitted |
|---|---|---|
| **Associations (5)** | `belongs_to`(143) `has_many`(125) `has_one`(20) `has_and_belongs_to_many`(12) `accepts_nested_attributes_for`(4) | **`declares_association`** *(NEW — Round 1 fix; existing `traverses_relation` is `Inferred`/body-walk, subject = fn; class-level declaration needs a separate `Authoritative` predicate, subject = class)* + **option-level**: `class_name` `dependent` `optional` `inverse_of` `through` `polymorphic` `foreign_key` `as` `source` `touch` |
| **Validations (5)** | `validates`(159) `validate`(52) `normalizes`(5) `validates_associated`(2) `validates_each`(1) | **`validates_constraint`** *(Round 1 rename — verb form, disambiguates from declarative `has_*` predicates)* + **`normalizes_attribute`** |
| **Callbacks (12)** | `before_save`(10) `before_destroy`(10) `after_create`(8) `after_destroy`(6) `before_create`(5) `before_validation`(4) `after_save`(4) `after_commit`(2) `after_update`(1) `after_validation`(1) `after_initialize`(1) `after_destroy_commit`(1) `around_destroy`(1) | **`has_callback{phase, target}`** |
| **Concerns / module composition (6)** | `include`(311) `extend`(59) `class_methods`(22) `included`(35) `prepend`(4) `prepended`(1) | **`includes_module`** **`extends_module`** **`prepends_module`** **`concern_class_methods`** **`concern_included_block`** |
| **Attributes — DB + virtual + visibility (13)** | `attribute`(18) `attr_accessor`(24) `attr_reader`(15) `attr_readonly`(2) `alias_attribute`(4) `alias_method`(4) `alias`(17) `undef_method`(1) `serialize`(10) `enum`(10) `store_attribute`(13) `store_accessor`(1) `define_attribute_method`(1) | **`has_attribute`** **`aliases_attribute`** **`aliases_method`** **`column_override`** |
| **Delegation (1)** | `delegate`(38) | **`delegates_to`** |
| **Scope / query DSL (3)** | `scope`(95) `default_scope`(11) `scopes`(26 — OpenProject plural) | **`has_scope`** **`has_default_scope`** |
| **Ruby visibility / class structure (8)** | `private`(183) `protected`(33) `private_class_method`(16) `private_constant`(2) `class_attribute`(3) `module_function`(1) `self`(89) `class`(176) `module`(205) | *non-fact scope markers — consumed by the parser, not emitted* |
| **OpenProject `acts_as_*` family (10)** | `acts_as_list`(10) `acts_as_attachable`(9) `acts_as_watchable`(6) `acts_as_searchable`(6) `acts_as_journalized`(6) `acts_as_event`(6) `acts_as_customizable`(5) `acts_as_tree`(3) `acts_as_favoritable`(3) `acts_as_url`(1) | **`acts_as{name, options}`** *(Round 1 rename — was `mixes_in_acts_as`; the `acts_as_*` family is its own idiom and the prefix duplicated `includes_module` semantics)* (one fact per variant) |
| **OpenProject custom registrations (6) — Round 1 SPLIT** | `register_journal_formatter`(27 — own predicate) · `register_journal_formatted_fields`(13 — own predicate) · `register_query`(1) `activity_provider_for`(6) `deprecated_alias`(6) `associated_to_ask_before_destruction`(1) `has_details_table`(1) — catch-all | **`registers_journal_formatter`** + **`registers_journal_formatted_fields`** *(promoted — 27+13 = 40/54 = 74 % of mass; restores iron-rule type-safety on bulk)* + **`has_dsl_call{name, args}`** (long-tail only — 5 names ≤ 6 each) |
| **Third-party gem DSL (5)** | `mount_uploader`(12 *carrierwave*) `has_paper_trail`(4) `has_closure_tree`(2) `counter_culture`(1) `auto_strip_attributes`(2) | **`mounts_uploader`** **`has_paper_trail`** **`has_closure_tree`** **`counter_cultures`** **`auto_strips`** |
| **Metaprogramming (1)** | `define_method`(24) | **`defines_method{name, body_source}`** *(Round 1 rename — body_source slot already signals dynamism)* — body kept verbatim (per `scope` precedent); per-edge `Provenance::Inferred(0.85, 0.75)` override on these 24 sites (not `OpenProjectExtracted`; 1.4 % of declarations) |
| **Refinements (1)** | `using`(2) | **`uses_refinement`** |

Total: **78 names → 67 emit + 11 scope markers**. Coverage proof in §5
D-AR-4: every class-body line in `app/models/**/*.rb` lands in one of the
above, or test fails.

## 3. Deliverables

### D-AR-1 — `ruff_spo_triplet::Predicate` vocab + `Provenance::OpenProjectExtracted` (target: `AdaWorldAPI/ruff`) *— Round 1 LOCKED via savant consolidation*

Add **27 new predicates** *(Round 1 final count, with the §2 fixes)* to the closed-vocab enum (`Predicate::*`, `as_str`, `from_str`, `default_provenance`). Add
`Provenance::OpenProjectExtracted{file: PathBuf, line: u32}` variant
distinct from existing `Provenance::Authoritative(0.95, 0.90)` *(the actual existing tier name — corrected from the original handover's "Extracted")*, `Provenance::Inferred(0.85, 0.75)`, and `Provenance::Structural(1.0, 1.0)`.

**Locked (f, c) — hand-tuned per `I-NOISE-FLOOR-JIRAK`:** `OpenProjectExtracted` = **`(0.95, 0.88)`**. Frequency tracks `Authoritative` (Python `@api.depends` is as truth-functionally certain as Ruby `belongs_to :project`); confidence drops 0.90 → 0.88 (two NARS revision-counts) to encode the Ruby metaprogramming surface delta (`class_methods do`, `included do`, `define_method`, `class << self`, dynamic constants) — small fraction of declarations unresolvable at static-AST time. Single tier (NOT split into `Static`/`DynamicallyResolved`) — the 24 `define_method` sites (1.4 %) use per-edge `Provenance::Inferred(0.85, 0.75)` override instead. Annotation comment text drafted by truth-architect (see `AGENT_LOG.md`).

**Council gate:** dto-soa-savant + truth-architect must both ACK before merge,
per `aerial-arm-ruff-spo-codegen-synergies.md` D-ARM-SYN-1. **Round 1 status: ACK from both.**

**Acceptance:** `ndjson::from_ndjson` round-trips all 27 new predicate names;
the existing `mined_rules_serialise_to_spo_ndjson` test stays green.

**PR shape: lands with D-AR-2 in ONE ruff PR** per integration-lead consolidation (the predicate enum + the `Declaration → Triple` expansion test are coupled; council reviews vocab+expansion atomically).

**Round-2 status (PR X):** ✅ **BRANCH PUSHED** to `AdaWorldAPI/ruff` at `claude/ar-shape-coverage-ruff` (commit `21c828d`, 2026-06-15). 36 tests pass in `ruff_spo_triplet` (was 14); 3 in `ruff_ruby_spo` (was 2). `Predicate::ALL.len() == 34` test-asserted. Operator action: review + open PR. Errata banked: round-1 consolidation called M = 23 (savant carry-over from kickoff plan); canonical §2 table = **27**; total emit predicates = **34** (7 core + 27 AR-shape). Plan + AGENT_LOG reconcile to 27.

### D-AR-2 — `ruff_ruby_spo::RubyClass` + `ruff_spo_triplet::Model` IR expansion (target: `AdaWorldAPI/ruff`) *— Round 1 LOCKED*

**Frontend IR** (`ruff_ruby_spo::RubyClass`): replace `associations: Vec<String>` with structured `declarations: Vec<Declaration>` discriminated union over the 67 emit categories. Per-association `AssocOptions` carries the 10 nested options (`class_name`, `dependent`, `optional`, `inverse_of`, `through`, `polymorphic`, `foreign_key`, `as`, `source`, `touch`). STI parent chain via `inherits_from: Option<String>` + `abstract_class: bool` + `inheritance_column: Option<String>`.

**Shared IR** (`ruff_spo_triplet::Model`, per prior-art-savant Round 1): `ModelGraph` itself adds **zero new fields** (still `{namespace, models: Vec<Model>}`); growth lands on `Model` as 13 sibling-shape `Vec<…>` fields: `associations: Vec<AssocDecl>`, `validations: Vec<Validation>`, `callbacks: Vec<Callback>`, `concerns: Vec<ConcernRef>`, `attributes: Vec<AttrDecl>`, `delegations: Vec<Delegation>`, `scopes: Vec<ScopeDecl>`, `acts_as: Vec<ActsAs>`, `dsl_calls: Vec<DslCall>`, `gem_dsl: Vec<GemDsl>`, `dynamic_methods: Vec<DynMethod>`, `refinements: Vec<UsingRef>`, `sti: Option<StiInfo>`. `expand()` extends with new match arms only — **no new trait**, the crate has no trait surface and `expand` stays a free fn over IR structs (prior-art-savant verdict: ADDITIONS-ONLY, zero drift).

**Acceptance:** unit tests cover each of the 67 categories with a hand-built `Declaration` → `Triple` expansion; the locked-shape test (already in ruff_ruby_spo) still passes.

**PR shape: lands with D-AR-1 in ONE ruff PR** per integration-lead consolidation.

### D-AR-3 — `ruff_ruby_spo` extractor implementation — the `todo!()` replacement (target: `AdaWorldAPI/ruff`)

Add `lib-ruby-parser` as a dep. Implement `parse_models` (file → Vec<RubyClass>)
and `extract_fields`/`extract_functions`/`extract_declarations` against the
real Ruby AST. Walk every class-body statement at indent depth = `class<n+1>`
and dispatch on the 78-name surface. Recurse into `module Foo::Bar` namespaces
to carry the full FQN. Recurse into `include FooConcern` to merge concern
declarations into the owning class.

**Acceptance:** running on `OpenProject/app/models/` produces ≥ 1696 +
option-level declarations as ndjson; every line validates against `Triple`
schema; predicate distribution matches §2 frequencies ±5 %.

### D-AR-4 — Coverage proof test (target: `AdaWorldAPI/ruff`)

A test that walks `app/models/`, runs the extractor, and asserts:
1. **Zero `Unclassified`** — every class-body line at indent depth 2 either
   produces ≥ 1 Triple or matches the 11-name scope-marker allowlist.
2. Predicate-frequency table matches §2 within tolerance (regression on
   ruff_ruby_spo silently dropping a category fails the build).
3. Round-trip: ndjson → `parse_triples` → lance-graph SPO store consumes
   without error.

**This is the 100 %-coverage gate.** Until D-AR-4 is green, downstream legs
do not land.

### D-AR-5 — `op-surreal-ast` consumer skeleton (target: `adaworldapi/openproject-nexgen-rs`)

Branch: `claude/op-surreal-ast`. Read the ndjson from D-AR-3, group Triples
by subject (= OpenProject model name), emit SurrealQL `DEFINE TABLE` per
class + `DEFINE FIELD` per `has_attribute` + `DEFINE INDEX` per
`traverses_relation`+`uniqueness`. Use the surrealdb fork's C16b
`new_for_ddl` builders (`catalog::TableDefinition::new_for_ddl`,
`.with_field(...)`, `→ ToSql`); the database-free emit surface is the API.

**Acceptance:** generated SurrealQL parses via `surrealdb-ast::parse_query`;
the count of `DEFINE TABLE` ≈ 1696 class declarations minus abstract bases.

### D-AR-6 — C16c bridge — `From<op_surreal_ast::*> for catalog::*` (target: **`AdaWorldAPI/surrealdb`** fork — LOCKED by integration-lead Round 1: Rust orphan rule + C16b builders live at fork `core/src/catalog/{table.rs, schema/field.rs, schema/index.rs}` per polyglot plan §2.2 line 64; the `From` impls must sit beside the `catalog::*` target types, not in `openproject-nexgen-rs` where only `op_surreal_ast::*` lives)

Plumb the `op_surreal_ast::{TableDef, FieldDef, IndexDef}` into the surrealdb
fork's `catalog::{TableDefinition, FieldDefinition, IndexDefinition}` so the
DDL emitted by D-AR-5 lands as registry catalog entries (the
`lance-graph-contract::callcenter::ogit_uris` slot).

**Acceptance:** the `polyglot-container-query-membrane-v1` D-PG-5 acceptance
("DDL ⇄ registry bridge round-trips dedup-by-URI mint") passes.

### D-AR-7 — Board hygiene (this commit)

`.claude/plans/openproject-ar-shape-extraction-v1.md` (this file), this
plan's INTEGRATION_PLANS prepend, STATUS_BOARD `D-AR-{1..7}` rows, the wave
handover under `.claude/handovers/`. TECH_DEBT entry for the
`define_method` dynamic-name case (the only category where the extractor
can't resolve a static method name without evaluating Ruby — body source
kept verbatim, NARS confidence drops accordingly).

## 4. The autoattended wave — Round 1 research (parallel)

Per `.claude/knowledge/autoattended-multiagent-pattern.md`: 4 savants, each
loads Tier-0 (`LATEST_STATE.md` + `PR_ARC_INVENTORY.md`) + the canonical
refs in §1, produces one output, prepends an `AGENT_LOG.md` entry on
return. Main thread does the atomic-consolidation pass.

| Savant | Iron rule | Output |
|---|---|---|
| **dto-soa-savant** | every predicate is a discriminated method, never a stringly-typed catch-all; the closed vocab is the type-safety mechanism | finalised 22-predicate enum proposal + naming review against existing 7 + collision check |
| **prior-art-savant** | the chain `RubyClass → ModelGraph → Triple → parse_triples` is canonical; no parallel invention | confirm the §2 67-emit / 11-scope split reuses the existing chain end-to-end; flag any new traits / types not justified |
| **truth-architect** | NARS (f, c) calibration grounded per `I-NOISE-FLOOR-JIRAK`; hand-tuned must say so | propose (f, c) defaults for `Provenance::OpenProjectExtracted` (separate from `Extracted` Python and `Aerial+::Mined`); justify against Jirak rate |
| **integration-lead** | cross-repo PR set must land in dependency order; no consumer ahead of producer | sequence: D-AR-1 (predicates) → D-AR-2 (IR) → D-AR-3 (extractor) → D-AR-4 (coverage proof) → D-AR-5/6 (consumer). Stage gates per repo (ruff PR, lance-graph-contract validation, op-nexgen PR). |

## 5. Sequencing *(Round 1 LOCKED — integration-lead amendment applied)*

- **Round 1 (parallel, complete 2026-06-15 ~20:55 UTC):** 4 savants research → main consolidates. **Status: DONE.** All 4 ACK + 4 amendments applied above.
- **Round 2 (sequential, post-consolidation):**
  - **PR X (`AdaWorldAPI/ruff`, branch `claude/ar-shape-coverage-ruff`):** ✅ **SHIPPED `21c828d` 2026-06-15.** D-AR-1 + D-AR-2 in ONE PR — predicate vocab (27 new; total 34) + `Provenance::OpenProjectExtracted(0.95, 0.88)` + `RubyClass.declarations: Vec<Declaration>` + `Model` IR expansion (12 Vec + 1 Option fields) + Declaration→Triple unit tests. Gate: D-ARM-SYN-1 council ACK + ndjson 27-predicate round-trip. 36 tests pass in `ruff_spo_triplet` (was 14); 3 in `ruff_ruby_spo` (was 2). Operator action: review + open PR.
  - **PR Y (`AdaWorldAPI/ruff`, same branch, stacked or fast-followed on PR X):** D-AR-3 + D-AR-4 — `lib-ruby-parser` dep + real extractor over `app/models/` + **the 100 %-coverage proof test**. Gate: D-AR-4 green = THE 100 % GATE.
- **Round 3 (parallel, can start any time after PR X merges):**
  - **PR Z (`adaworldapi/openproject-nexgen-rs`, branch `claude/op-surreal-ast`):** D-AR-5 skeleton — parallelizes with PR Y per integration-lead (ndjson contract = firewall; consumer match-arm-extensibly against existing 7 predicates, new vocab grows handled cases never breaks contract). Gate: SurrealQL parses + `DEFINE TABLE` count matches.
  - **PR W (`AdaWorldAPI/surrealdb` fork, branch `claude/c16c-op-surreal-ast-bridge` against `op-codegen-bridge` initiative):** D-AR-6 — `impl From<op_surreal_ast::{TableDef, FieldDef, IndexDef}> for catalog::{TableDefinition, FieldDefinition, IndexDefinition}`. Gate: D-PG-5 DDL ⇄ registry round-trip via dedup-by-URI mint.

## 6. Out of scope

- Aerial+ rule mining over the extracted SPO (separate `streaming-arm-nars-discovery-v1`).
- Generating Rust code from the SurrealQL DDL (separate codegen step;
  `ruff_python_codegen` precedent applies once the DDL is canonical).
- Extending coverage beyond `app/models/` — controllers / services / lib are
  Phase 2.
- Updating `normalized-entity-holy-grail-v1` to consume the new ndjson —
  separate plan iteration once D-AR-5/6 lands.

## 7. References (read order)

1. `AdaWorldAPI/ruff/crates/ruff_ruby_spo/src/lib.rs` — the scaffold this completes
2. `AdaWorldAPI/ruff/crates/ruff_spo_triplet/src/{triple.rs,lib.rs}` — the contract
3. `AdaWorldAPI/ruff/crates/ruff_python_dto_check/src/lib.rs` — Python sibling extractor (precedent shape)
4. `.claude/knowledge/aerial-arm-ruff-spo-codegen-synergies.md` — D-ARM-SYN-1 + the bracket
5. `.claude/plans/polyglot-container-query-membrane-v1.md` §2.2 + D-PG-5 — DDL bridge
6. `.claude/plans/normalized-entity-holy-grail-v1.md` — trunk plan consuming the ndjson
7. `/tmp/cov-repro/openproject-78-name-surface.txt` — measured ground truth
