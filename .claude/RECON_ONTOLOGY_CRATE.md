# RECON_ONTOLOGY_CRATE.md

> **Scope:** Phase 1 reconnaissance for `lance-graph-ontology` crate (revision 4 of the
> ontology-crate scope). Read-only; no code is permitted before this document and
> `DECISION_SPO_ARIGRAPH.md` are committed.
>
> **Branch:** `claude/create-graph-ontology-crate-gkuJG`
> **Date:** 2026-05-07
> **Author:** Session-class agent on Opus 4.7 (1M)

The plan promised that a long list of substrates was already shipped and that this
session's only deltas are one new crate plus two architectural decisions. The findings
below verify or refute each claim with a path and a short quote.

## 1. "Already shipped" verification

### 1.1 `lance-graph-contract::mul` and the shader-driver wiring

`crates/lance-graph-contract/src/mul.rs` is 344 lines and contains
`MulAssessment::compute(&SituationInput)` as a carrier-method (no free functions on
state). Confirmed.

`crates/cognitive-shader-driver/src/driver.rs` is 1,195 lines. The MUL wiring sits at
lines 271-320 (the plan said "271-320" — verified). The driver builds a
`SituationInput` from observable shader state, calls `MulAssessment::compute`, and
gates Flow→Hold via `mul.is_unskilled_overconfident()`:

> ```
> 295        let situation = SituationInput {
> 296            felt_competence: top_resonance.clamp(0.0, 1.0) as f64,
> 297            demonstrated_competence: (1.0 - free_energy.total).clamp(0.0, 1.0) as f64,
> 298            environment_stability: (1.0 - std_dev_clamped).clamp(0.0, 1.0),
> 299            challenge_level: std_dev_clamped,
> 300            skill_level: awareness_skill,
> 301            ..SituationInput::default()
> 302        };
> 303        let mul = MulAssessment::compute(&situation);
> ...
> 308        let gate = if free_energy.is_catastrophic() {
> 309            GateDecision::BLOCK
> 310        } else if mul.is_unskilled_overconfident() {
> 311            // MUL veto: the system "feels confident" while DK / trust
> 312            // textures flag the gap. Hold rather than commit.
> 313            GateDecision::HOLD
> ```

The publisher-side defaults (`calibration_accuracy`, `allostatic_load`,
`max_acceptable_damage`, `sandbox_available`) flow from `SituationInput::default()`
exactly as the plan stated. Tightening those is publisher-side polish, not this
session's work.

### 1.2 `lance-graph-contract::ontology`

`crates/lance-graph-contract/src/ontology.rs` is 646 lines. The first 120 lines
declare `Locale`, `Label`, `EntityTypeId` (Foundry Object-Type equivalent),
`Ontology` with `schemas: Vec<Schema>`, `links: Vec<LinkSpec>`, `actions:
Vec<ActionSpec>`, plus an `OntologyBuilder`.

`PropertySpec`, `Marking` (Public/Internal/Pii/Financial/Restricted), `SemanticType`,
`Schema`, `LinkSpec`, `ActionSpec` live in
`crates/lance-graph-contract/src/property.rs`. They are imported into `ontology.rs`
at line 20 and composed via `Ontology` and `OntologyBuilder`. Confirmed.

A `SchemaExpander` trait already exists and is exercised by
`crates/lance-graph/src/graph/spo/ontology_bridge.rs` — that bridge already turns
a contract `Ontology` plus an entity instance into `ExpandedTriple`s for the SPO
store. This is critical for the new crate: TTL hydration produces
`Ontology`/`Schema` values, and the existing SchemaExpander path takes them from
there.

### 1.3 Polyglot parsers in `lance-graph-planner`

`crates/lance-graph-planner/src/strategy/` contains:

```
arena_ir.rs        chat_bundle.rs     collapse_gate.rs   cypher_parse.rs
dp_join.rs         extension.rs       gql_parse.rs       gremlin_parse.rs
histogram_cost.rs  jit_compile.rs     mod.rs             morsel_exec.rs
rule_optimizer.rs  sigma_scan.rs      sparql_parse.rs    stream_pipeline.rs
truth_propagation.rs                  workflow_dag.rs
```

Cypher / Gremlin / SPARQL / GQL strategies are present and each implements the
`PlanStrategy` trait with a `plan(&self, input, arena)` method. Confirmed.

**Caveat (PARSER-1 in entropy ledger):** `cypher_parse.rs` is a 72-line *stub* that
detects features by uppercased substring match; the real nom parser lives at
`crates/lance-graph/src/parser.rs:23` (`pub fn parse_cypher_query(input: &str) ->
Result<CypherQuery>`). The comment at the bottom of the strategy says explicitly:

> ```
> // Real implementation: call lance-graph's parser::parse_cypher_query()
> // to produce a full AST. For now, feature detection is the output.
> ```

This is documented in `.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md` row PARSER-1
("Wired ×1 real + Stub ×3 + parallel ×1 excluded", entropy 5). The plan to wire is
known but unowned: *"Wire `cypher_parse::CypherParse::plan` to call
`lance-graph::parser::parse_cypher_query` (real nom)."*

Implication for this session's Phase 7 (`woa-rs` integration test): the test should
call `lance_graph::parser::parse_cypher_query` directly, not the strategy. The
strategy-dispatched path is gap'd by design and out of scope here.

### 1.4 SPO store and ARiGraph

`crates/lance-graph/src/graph/spo/` has `builder.rs`, `merkle.rs`, `mod.rs`,
`nsm_bridge.rs`, `ontology_bridge.rs`, `semiring.rs`, `store.rs`, `truth.rs`.
Fingerprint-keyed columnar SoA layout with `HammingMin` truth-semiring as advertised.
Confirmed.

`crates/lance-graph/src/graph/arigraph/` has `episodic.rs`, `language.rs`, `mod.rs`,
`orchestrator.rs`, `retrieval.rs`, `sensorium.rs`, `triplet_graph.rs`, `xai_client.rs`.
`triplet_graph.rs` is the string-keyed `HashMap<String, Vec<usize>>` (1,072 LOC per
the entropy ledger) holding warm episodic state. Confirmed.

`AdjacencyStore` (CSR/CSC) and `batch_adjacent` are referenced by
`crates/lance-graph-planner/src/adjacency/` per the workspace dep graph. Three-layer
ARiGraph transcode (semantic Concept + Relation; episodic Episode + EXTRACTED/FOLLOWS;
cognitive ThinkingStyle + ACTIVATES) maps to the files above. Confirmed.

### 1.5 `CausalEdge64` and `cognitive-shader-driver`

`crates/causal-edge/src/` has `edge.rs`, `lib.rs`, `network.rs`, `pearl.rs`,
`plasticity.rs`, `tables.rs`. `edge.rs` defines the 64-bit edge with palette indices,
NARS (frequency, confidence), Pearl 2³ `causal_mask`, direction/inference/plasticity/
temporal bits per the plan. Confirmed.

`crates/cognitive-shader-driver/src/` has `bindspace.rs` (BindSpace SoA columns),
`driver.rs` (1,195 lines including the MUL gate at 271-320, CausalEdge64 emission
loop at 322 onwards), `planner_bridge.rs` (translates contract↔planner SituationInput
shapes per the plan). Confirmed.

### 1.6 `smb-ontology` declarative pattern

`smb-office-rs/crates/smb-ontology/src/` is **2,079 lines** across 8 files:
`customer.rs` (364), `lib.rs` (130), `mahnung.rs` (176), `markings.rs` (151),
`rechnung.rs` (187), `remaining.rs` (632), `schuldner.rs` (149), `woa_artikel.rs`
(290). 13 German Steuerberater entities expressed as declarative
`PropertySpec`/`SemanticType` Rust. Confirmed. This is the proven Foundry-shape
pattern. It stays untouched per the plan as the OGIT-skeptical-customer fallback.

### 1.7 Existing OGIT / `ontology-crate` work

```
$ rg -i "ogit|open.*graph.*it|almatoai|ontology_dict" --type rs --type md --type toml
```

found nothing in `crates/lance-graph-contract/src/*.rs`, and one hit each in
`crates/lance-graph-callcenter/`, `crates/lance-graph-contract/src/ontology.rs`,
`crates/lance-graph/src/graph/spo/ontology_bridge.rs`, and
`crates/lance-graph-callcenter/src/transcode/ontology_table.rs` — all of those are
*pre-existing* ontology-shaped code (callcenter DTO, contract ontology builder, SPO
bridge), none of them reference OGIT.

`find crates -name "*ontology*" -type d` returns nothing — there is no
`crates/lance-graph-ontology/` directory yet. Confirmed: the new crate does not exist.

`grep -i "OGIT\|almatoai"` against `.claude/**/*.md` returns nothing. Confirmed: no
prior OGIT integration work exists in this workspace.

### 1.8 `woa-rs` and `WoA` state

`woa-rs/` contains exactly `CLAUDE.md`, `NOTES.md`, `PROMPT.md`, `README.md`,
`rfcs/` — bare scaffolding as the plan claimed. The `CLAUDE.md` declares a chunked-
write rule, names `AdaWorldAPI/WoA` as canonical source, and says behavioural parity
is the spec. No Rust source is present yet. Confirmed.

`WoA/` (Python source) contains `RUST_TRANSCODE_PLAN.md` (58,054 bytes — matches
"58KB"), `RUST_TRANSCODE_LEDGER.md` (5,036 bytes), `models.py` (527 lines),
`app.py`, `pdf_gen.py`, `mail_send.py`, `migrate_data.py`, `import_keepass.py`,
`vault_io.py`, `wsgi.py`, plus `static/`, `templates/`, `requirements.txt`. The
transcode plan and ledger are present and authoritative for Phase 6. Confirmed.

`models.py` declares 16 SQLAlchemy models we will need to TTL-emit:

```
Tenant, User, Customer, Article, WorkOrder, Position, Activity, Picture,
HistoryEntry, LogbookEntry, NumberSequence, Setting, CustomerPortalUser,
PasswordEntry, TimeSheet
```

(Plus the `WorkOrder.doc_type` enum: `workorder | offer | order | invoice | credit |
gutschrift`.) The shape is German-language WaWi/handwerk: customer + article + work-
order with positions, activities, pictures, history. Phase 6 will transcode these
into TTL files under `OGIT/NTO/WorkOrder/`.

### 1.9 OGIT fork state

`/home/user/OGIT/` is the AdaWorldAPI fork. Top-level: `NTO/`, `SDF/`, `SGO/`,
`bin/`, `docs/`, `pdf/`, `versioning/`, plus `ogit.ttl`, `validate.sh`,
`namespace.sh`, `singleTTL.sh`, `verbToEntity2.py`, `LICENSE.md`, `README.md`.

`NTO/` contains 66 namespace directories sorted alphabetically: `Accounting`,
`Advertising`, `Audit`, `Auth`, `Automation`, `Botany`, `ClassificationStandard`,
`Compliance`, `Cost`, `Credit`, `CustomerSupport`, `Data`, `DataProcessing`,
`Datacenter`, `Documents`, `EmailCorrespondance`, `Examples`, `Factory`,
`FinancialAccounting`, `FinancialMarket`, `Forms`, `Forum`, `GeoProfile`, `HR`,
`Health`, `Knowledge`, `Legal`, `Location`, `MARS`, `ML`, `MRO`, `MRP`,
`MaterialManagement`, `Meteorology`, `Mobile`, `Network`, `OSLC-arch`,
`OSLC-asset`, `OSLC-automation`, `OSLC-change`, `OSLC-core`, `OSLC-crtv`,
`OSLC-ems`, `OSLC-perfmon`, `OSLC-qm`, `OSLC-reqman`, `PLM`, `PTF`, `Politics`,
`Price`, `Procurement`, `Project`, `Publications`, `RDDL`, `RL`, `RPA`, `Religion`,
`SaaS`, `SalesDistribution`, `Schedule`, `Security`, `ServiceManagement`,
`Software`, `Statistics`, `Survey`, `Transport`, `UserMeta`, `Version`. No
`WorkOrder/`, no `Healthcare/`, no `Steuerberater/`, no `Q2Gotham/` directories
exist yet. Confirmed: phase 6 adds `WorkOrder/` and is the first AdaWorldAPI
extension to the fork.

A representative entity TTL (`NTO/Network/entities/IPAddress.ttl`) shows the
convention used by all OGIT entity files: `@prefix` declarations, an
`ogit.<Namespace>:<EntityName>` subject typed `a rdfs:Class; rdfs:subClassOf
ogit:Entity`, then `rdfs:label`, `dcterms:description`, `dcterms:valid`,
`dcterms:creator`, `ogit:scope "NTO"`, `ogit:parent ogit:Node`, three RDF lists
for `ogit:mandatory-attributes`, `ogit:optional-attributes`,
`ogit:indexed-attributes`, and an `ogit:allowed` block enumerating the verbs and
target entity types this entity may participate in. Phase 6 emits TTL in exactly
this shape.

`ogit.ttl` (root vocabulary) declares `ogit:Entity`, `ogit:Attribute`, and the
verb/scope vocabulary — same conventions, with `ogit:scope "SGO"` for the root.

### 1.10 Already-shipped summary table

| Plan claim                                 | Status     | Evidence                                                        |
|--------------------------------------------|------------|-----------------------------------------------------------------|
| `lance-graph-contract::mul`                | Confirmed  | `crates/lance-graph-contract/src/mul.rs` (344 LOC)              |
| Shader-driver MUL veto at driver.rs:271-320| Confirmed  | `cognitive-shader-driver/src/driver.rs:271-320`                 |
| `lance-graph-contract::ontology`           | Confirmed  | `crates/lance-graph-contract/src/ontology.rs` (646 LOC)         |
| `PropertySpec`/`Marking`/`SemanticType`    | Confirmed  | `crates/lance-graph-contract/src/property.rs`                   |
| `SchemaExpander` trait + spo bridge        | Confirmed  | `lance-graph/src/graph/spo/ontology_bridge.rs`                  |
| Polyglot parsers (Cypher/GQL/Gremlin/SPARQL)| Confirmed* | `lance-graph-planner/src/strategy/{cypher,gql,gremlin,sparql}_parse.rs` (\*PARSER-1 stub gap) |
| SPO store (fingerprint, HammingMin)        | Confirmed  | `lance-graph/src/graph/spo/`                                    |
| ARiGraph triplet_graph (string-keyed)      | Confirmed  | `lance-graph/src/graph/arigraph/triplet_graph.rs`               |
| `CausalEdge64`                             | Confirmed  | `crates/causal-edge/src/{edge,pearl,plasticity,tables}.rs`      |
| BindSpace SoA columns + driver             | Confirmed  | `cognitive-shader-driver/src/{bindspace,driver}.rs`             |
| `smb-ontology` (declarative Rust, 13 ents) | Confirmed  | `smb-office-rs/crates/smb-ontology/` (2,079 LOC, 8 files)       |
| `lance-graph-ontology` does NOT exist      | Confirmed  | Workspace `Cargo.toml` members + `find` returns no match        |
| OGIT references in lance-graph             | Confirmed: none | rg returns no `OGIT|almatoai` hit in workspace            |
| AdaWorldAPI/OGIT NTO upstream parity        | 66 namespaces present, no AdaWorldAPI extensions yet          |
| `woa-rs` is bare scaffolding               | Confirmed  | 4 markdown files, 1 empty `rfcs/` dir, no Rust source           |
| `WoA/RUST_TRANSCODE_PLAN.md` (~58 KB)      | Confirmed  | 58,054 bytes; ledger 5,036 bytes; `models.py` 527 LOC, 16 models|

The plan's "already shipped" list holds. The PARSER-1 gap is documented and out of
scope; we will route Phase 7 around it via the real `parse_cypher_query` entry point.

## 2. SPO-1 evidence

`.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md:70` declares the SPO-1 row:

> | **SPO-1** | R7/R6 | Two SPO stores | Wired (×2 distinct) | 2 | Med |
> `lance-graph::graph::spo::*` (fingerprint-keyed, HammingMin truth-semiring) +
> `lance-graph::graph::arigraph::triplet_graph` (string-keyed `HashMap<String,
> Vec<usize>>`, 1,072 LOC). Share only `TruthValue`. **No bridge fn** between them
> — `to_fingerprints()` is a derive, not a writer. | none | Missing | **4** |

`.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md:245` declares the disposition:

> | **SPO-1** | Stage 3 (×2 distinct purposes, **not duplicates by design**) |
> `triplet_graph` Smart (string-keyed methods); `spo::store` Smart (fingerprint-keyed
> methods) | Add `arigraph::SpoBridge::promote_to_spo(&TripletGraph, gate, &mut
> SpoStore)` — promotes warm string-keyed entries into cold fingerprint-keyed store |

So the workspace has already named the answer: the two stores are not duplicates,
they are two cache layers (warm, string-keyed working memory; cold, fingerprint-keyed
durable). The blocker for closure is the missing one-way `promote_to_spo` writer.
`DECISION_SPO_ARIGRAPH.md` adopts this directly and discusses what (if anything) the
ontology crate has to do about it.

## 3. Other findings worth pinning

### 3.1 No TTL parsing dependency yet

`Cargo.lock` and per-crate `Cargo.toml` files contain no reference to `oxttl`,
`oxrdf`, or `sophia`. The new `lance-graph-ontology` crate will be the first
introduction of a TTL parser dependency in this workspace.

### 3.2 `AGENT_LOG.md` is missing

Entropy ledger row AGENT-LOG-1 (entropy 3) reports that `.claude/board/AGENT_LOG.md`
is referenced by CLAUDE.md but does not exist. This session creates it as a side-
effect of doing Layer-2 A2A handover correctly. (The Mandatory Board-Hygiene Rule
in CLAUDE.md says any PR adding work prepends an entry to AGENT_LOG.md in the same
commit — so creating the file is forced once we land any artifact at all.)

### 3.3 Existing `ontology_dictionary` Lance table

`grep -rn "ontology_dict\|ontology_dictionary"` against `lance-graph/` returns no
hits in source. The Lance dictionary table specified in the plan does not exist yet.
Confirmed: Phase 4 creates it.

### 3.4 `SchemaExpander` is the contact point

The new crate does *not* invent its own EntityStore. It produces `Ontology` /
`Schema` / `LinkSpec` values via TTL hydration and hands them to existing
`SchemaExpander` paths. Specifically:

- TTL → `MappingProposal { schemas: Vec<Schema>, links: Vec<LinkSpec> }`
- `OntologyBuilder::schema(...).link(...).build()` → `Ontology`
- `Ontology::expand_entity(...)` (already exists, see `spo/ontology_bridge.rs`
  test) → `ExpandedTriple` for SPO writes

So the new crate is a *parser + cache + scoping facade*, not a new storage layer.
This matches the plan's framing ("scoped views, not stores").

## 4. Out-of-scope confirmations

This session does not produce: a new SPOG quad store, janus-driver, new SPARQL/
Gremlin/GQL/Cypher parsers, a new SPO store, new MUL wiring or new MUL publishers,
new CausalEdge64 variants, new BindSpace columns, smb-ontology TTL migration,
callcenter-bridge, MySQL/MSSQL `SchemaSource` impls, or a customer admin form. Those
are explicit non-goals from the plan. None of the recon above changes that boundary.

## 5. What this recon authorises

The plan's premises hold. Phases 2–7 may proceed. The two adjustments are (a)
Phase 7 calls `lance_graph::parser::parse_cypher_query` directly because PARSER-1
stub is unowned, and (b) Phase 6 TTL emission targets `OGIT/NTO/WorkOrder/` (the
fork has no `WorkOrder/` directory yet, so this is an additive PR with no
upstream conflict).

The next deliverable is `DECISION_SPO_ARIGRAPH.md`. Then the crate.
