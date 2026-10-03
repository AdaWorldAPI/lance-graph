# ORKG reference wiring: harvest of the Open Research Knowledge Graph (2026-10-03)

**Status:** REFERENCE / COMPARATOR — NOT DEPENDENCY, NOT AUTHORITY.
Architecture reconnaissance only. No code, schema artifact, fixture or prose was
copied from ORKG; identifiers, endpoint names and type names are quoted for
reference.

**READ BY:** anyone designing a literature-evidence ingest, a comparison / meta-study
projection, an interoperability export, or a facet surface over an evidence
population. Read § 1 and § 12 before citing anything below as a design input.

---

## 0. Provenance of this harvest

| Source | Revision measured | License |
|---|---|---|
| `orkg-backend` (Kotlin, Spring, Neo4j + PostgreSQL) | `9b24878a940cc38a57a2480620ff4d064ffff1c8`, v0.102.2, 2026-09-24 | MIT |
| `orkg-frontend` (Next.js / TypeScript) | `beb8902dc82155a5a712287b1c2b1c46eb45afa0`, 2026-10-02 | MIT |
| `orkg-ontology` (one Turtle file) | `724d929095a7e37d132bc977bb9817d4253089ce`, 2023-06-20 | MIT |
| `nlp/orkg-nlp-api` | `bd63c9a4`, 2025-10-09 | MIT |
| `orkg-simcomp/orkg-simcomp-api` | `955894f2`, 2026-03-19 | MIT |
| `orkg-similarity` (legacy comparison builder) | `0ff4f4ac`, 2023-05-25 | MIT |
| `smart-filters` | `97626d84`, 2025-07-09 | MIT |
| `agentic-loop` | `3dd35e8b`, 2026-06-19 | MIT |
| `orkg-ask/backend` | `7c968d86`, 2026-08-26 | MIT |
| `orkg-comparisons-creator` | `ed76d8a7`, 2026-02-24 | MIT |
| `annotation` (legacy NER annotator) | `b6b8885f`, 2023-06-19 | **Apache-2.0** |
| Paper arXiv:2107.05738 | v1, 2021-07-05 | CC-BY 4.0 |
| Live API `https://orkg.org/api` | queried 2026-10-03 | — |

Permalink patterns:

- backend: `https://gitlab.com/TIBHannover/orkg/orkg-backend/-/blob/9b24878a940cc38a57a2480620ff4d064ffff1c8/<path>`
- frontend: `https://gitlab.com/TIBHannover/orkg/orkg-frontend/-/blob/beb8902dc82155a5a712287b1c2b1c46eb45afa0/<path>`
- API docs: `https://tibhannover.gitlab.io/orkg/orkg-backend/` (source:
  `documentation/src/antora/` in the backend repo)

Path abbreviations below: `CT/` = `content-types/`, `G/` = `graph/`, and `FE:` = a
frontend path. Backend Kotlin paths omit the `src/main/kotlin/org/orkg/...` middle.

AdaWorldAPI side measured at lance-graph `06061fe`, OGAR `a37383e`, MedCare-rs
`29116c5` (all origin/main, 2026-10-03). `ndarray` was **not** checked out in this
session, so the full CLAM/CHAODA engine and `ndarray::hpc::reliability` were not
read. Nothing in this document was executed; "test" means the test function exists.

**License note.** Every ORKG repository read is MIT, except the legacy `annotation`
service (Apache-2.0). MIT would permit reuse with attribution, but this harvest
deliberately vendors nothing. Interoperability (§ 10) targets ORKG's **public API and
import formats**, which an independent implementation can produce without code
ancestry.

---

## 1. Scope

> ORKG is harvested as a scholarly-knowledge bookkeeping and comparison reference.
> It is not adopted as the reasoning substrate and is not an implementation
> dependency.

ORKG terms (Paper, Contribution, Template, Comparison, Facet) are used here only to
describe ORKG. They do not become internal canonical vocabulary. Internal concepts
keep their own names and are minted through OGAR classids, never through a parallel
schema.

Sibling reference: `.claude/harvest/indra-reference-wiring.md` (INDRA, causal
knowledge assembly and model construction). ORKG is the scholarly-contribution and
comparison reference; INDRA is the mechanistic-statement and assembly reference.

---

## 2. ORKG system map

```
 SOURCE INGESTION            AUTHORING / EXTRACTION                 GRAPH PERSISTENCE
 ─────────────────           ──────────────────────                 ─────────────────
 DOI / title (manual)  ──►  Paper form, grid editor, CSV import ──► Neo4j: Thing nodes
 PDF ─► GROBID (agentic) ─► NLP services → SUGGESTIONS ────────────► + RELATED edges
 Paper CSV (/api/csvs)       user click / approve = the write       (statement_id,
 OLS / Wikidata / GeoNames   LLM plan → CSV → import (AI_GENERATED)   predicate_id,
   term import (/api/import)                                          extraction_method)
                             Templates (SHACL NodeShape) — enforced  PostgreSQL: snapshots,
                             only on template-instance writes         comparison tables,
                                                                      community, CSV jobs
                                         │
 COMPARISON                              ▼
 ──────────
 Comparison resource ─ compareContribution ─► sources (contributions / any Thing,
                       compareRosettaStone…    or Rosetta-Stone contexts)
 selected ComparisonPath tree (predicate ids, depth ≤ 10)
   ─► Cypher path traversal per source ─► ComparisonColumnData
   ─► ComparisonTable.from(): columns = sources, rows = paths, multi-value = extra rows
                                         │
 SEARCH / FACETING                       ▼                    PRESENTATION / PUBLICATION
 ─────────────────                                            ──────────────────────────
 per-comparison column filters (FE, URL state)                comparison page, smart review,
 observatory SearchFilter(path, range, values, exact)         literature list
 LLM "smart filters" over search abstracts (read-only)        publish → frozen table + DOI
                                                              exports: CSV, LaTeX, PDF,
                                                              RDF Data Cube, JATS, N-Triples
```

Six layers, each with its own code home:

| Layer | Where | What it does |
|---|---|---|
| source ingestion | `data-import/`, `G/graph-adapter-output-web/` (OLS, Wikidata, GeoNames), frontend paper form, `agentic-loop` | Bibliographic entry is manual or CSV. There is **no** Crossref or DOI-metadata fetch in the backend. |
| knowledge authoring / extraction | frontend; `orkg-nlp-api`; `agentic-loop` | Every NLP service returns **suggestions**. Writes happen on a user action. |
| graph persistence | `G/`, `CT/`, Neo4j + PostgreSQL | Statements are binary `RELATED` edges. Rosetta-Stone statements are n-ary, versioned structures. |
| comparison | `CT/content-types-core-model/.../ComparisonTable.kt`, `ComparisonTableService.kt` | Projects recorded values into a table. It computes nothing over them. |
| search / faceting | `G/graph-core-model/.../SearchFilter.kt`; `FE: src/components/Comparison/ComparisonTable/RowHeader/FilterPopover/` | Predicate-path filters over papers; label filters over comparison columns. |
| presentation | frontend; `data-export/`; JATS / CSV adapters in `CT/content-types-adapter-input-representations/` | Publish, version, DOI, export. |

---

## 3. Core data model

| Concept | Implementation (backend unless marked) | Shape | Persistence | Cardinality / relations |
|---|---|---|---|---|
| **Thing** | `G/graph-core-model/.../Thing.kt` `sealed interface Thing` | `id: ThingId`, `label`, `createdAt`, `createdBy`, `modifiable` | Neo4j `:Thing` | Supertype of Resource, Class, Predicate, Literal. `ThingId` matches `^[a-zA-Z0-9:_-]+$` (`common/core-identifiers/.../ThingId.kt`). Generated ids are `R<n>`, `P<n>`, `C<n>`, `L<n>`; semantic ids such as `P31` or `sh:path` are also legal. |
| **Resource** | `Resource.kt` | + `classes: Set<ThingId>`, `extractionMethod`, `observatoryId`, `organizationId`, `visibility`, `verified` | `:Resource`; **classes become extra Neo4j labels** (`@DynamicLabels`) | One observatory and one organization per resource. |
| **Class** | `Class.kt`, `ClassHierarchyController` | + `uri: IRI?`, `extractionMethod` | `:Class` | Has a subclass hierarchy. |
| **Predicate** | `Predicate.kt` | + `uri: IRI?`, `extractionMethod` | `:Predicate`, but statements reference it by **string `predicate_id`**, not by node | Roughly 90 well-known ids in `G/graph-core-constants/.../Constants.kt`. |
| **Literal** | `Literal.kt`; datatypes in `Literals.XSD` (36 entries) | `label`, `datatype` (a free string, default `xsd:string`) | `:Literal` | Numbers are stored as **labels**. |
| **Statement** | `GeneralStatement.kt` | `id: StatementId("S…")`, `subject`, `predicate`, `object`, `createdBy`, `createdAt`, `extractionMethod`, `modifiable`, `index` | `(:Thing)-[:RELATED {statement_id, predicate_id, created_by, created_at, extraction_method, index, modifiable}]->(:Thing)` | Binary. **No source document, confidence, qualifier, reification or named graph.** Updates keep no `updatedBy`. |
| **ExtractionMethod** | `G/graph-core-model/.../ExtractionMethod.kt` | `AUTOMATIC`, `MANUAL`, `UNKNOWN`, `AI_GENERATED`, `AI_GENERATED_WITH_MANUAL_REVIEW` | `extraction_method` on every node and edge | Set once per write command and stamped onto every element it creates (`SubgraphCreator.kt`, `ContributionCreator.kt`). `canBeChangedTo` allows only AI ↔ AI-reviewed transitions; a label edit on an AI element without a method demotes it to `MANUAL` (`G/graph-core-services/.../Extensions.kt`). |
| **Paper** | `CT/.../Paper.kt`, `PaperService.kt`, `PaperController` (`/api/papers`) | title, research fields, identifiers (`doi` `P26`, `isbn`, `issn`, `open_alex`), authors (ordered List), publication info, contributions, versions | Resource of class `Paper` | `P31 hasContribution` → 0..n contributions. **No min/max count.** |
| **Contribution** | `CT/.../Contribution.kt`, `ContributionValidator.kt` | `label`, `classes`, `properties: Map<ThingId, List<ThingId>>` | Resource whose classes include `Contribution` | Needs a non-empty label and ≥ 1 statement. Its subtree uses free, user-chosen predicates. |
| **Template** | `CT/.../Template.kt`; actions `CT/.../actions/templates/` | `targetClass`, `properties: List<TemplateProperty>` (string / number / other-literal / resource / untyped), `isClosed`, relations to research field / problem / predicate | Resource `NodeShape` with `PropertyShape` children (`sh:path`, `sh:minCount`, `sh:maxCount`, `sh:datatype`, `sh:class`, `sh:pattern`, `sh:minInclusive`, `sh:maxInclusive`, `sh:order`); numbers stored as literals | **Enforced only** on `/api/templates/{id}/instances` (`AbstractTemplatePropertyValueValidator`) and the Rosetta-Stone endpoints. **Advisory** on raw statement writes and paper creation. |
| **Rosetta-Stone statement** | `CT/.../RosettaStoneStatement.kt`, `RosettaStoneTemplate.kt` | n-ary: `subjects`, `objects: List<List<Thing>>` per position; per version `certainty: LOW \| MODERATE \| HIGH`, `negated: Boolean`, soft-delete fields | Separate structure (`VERSION`, `TEMPLATE`, `CONTEXT`, `METADATA`, `SUBJECT`, `OBJECT`, `VALUE` edges), **not** `RELATED` | Append-only versions; `contextId` links it to a paper or contribution. ORKG's only first-class epistemic qualifier. |
| **ComparisonPath** | `CT/.../ComparisonPath.kt`, `SimpleComparisonPath.kt`, `LabeledComparisonPath.kt` | tree of `(id, type ∈ {PREDICATE, ROSETTA_STONE_STATEMENT, ROSETTA_STONE_STATEMENT_VALUE}, children)`; labeled form adds `label`, `description`, `sources: Int?` | stored inside the comparison table (JPA `comparison_tables`) | `MAX_PATH_DEPTH = 10` (`ComparisonTableService.kt`). Rosetta paths: statement at level 1, value at level 2 only. |
| **Comparison** | `CT/.../Comparison.kt` | `type ∈ {SYSTEMATIC, RELATED_WORK, STATE_OF_THE_ART, RESOURCE, UNKNOWN}`, `sources: List<ComparisonDataSource(id, THING \| ROSETTA_STONE_STATEMENT)>`, `searchProtocol` (inclusion / exclusion criteria, search engines, search strings, research questions, studies returned / retained), visualizations, references, versions | Resource of class `Comparison`; sources via `compareContribution` / `compareRosettaStoneContribution` | Publishing needs ≥ 2 distinct sources (`ComparisonPublishableValidator`). |
| **Published comparison** | `actions/comparisons/` chain | copy resource classed `ComparisonPublished` + `LatestVersion` | JPA `comparison_tables` keyed by the version id | Table frozen (§ 4.6). Contributions are **re-linked, not deep-copied**. Optional DataCite DOI, prefix `10.7484`. |
| **Literature list** | `CT/.../LiteratureList.kt` | sections: list (entries → paper / link) or text | `HasSection` edges ordered by `createdAt` | Published versions archive a subgraph snapshot in PostgreSQL. |
| **Smart review** | `CT/.../SmartReview.kt` | sections: comparison, visualization, resource, property, ontology, text | as above | Editorial container; no numeric work. |
| **ORKG ontology** | `orkg-ontology/orkg-core.ttl` | 7 OWL classes (`Paper`, `ResearchContribution`, `ResearchProblem`, …), 6 object properties (`addresses`, `employs`, `utilizes`, `yields`, …) | — | **Unlinked**: the backend never references `http://orkg.org/core` and exports under `http://orkg.org/orkg/{class,predicate,resource}/`. |

The full field-level breakdown is in the backend report this harvest was built from
(scratchpad only, not committed). Every row above was re-read against the source
cited.

---

## 4. Comparison mechanics (the core)

### 4.1 Source resources

`ComparisonDataSource(id, type)`. `THING` sources are any Thing reached via
`compareContribution`. The Cypher root is `node("Thing")`, not `Contribution`, so
**arbitrary resources can be compared** (`ComparisonType.RESOURCE_COMPARISON`).
`ROSETTA_STONE_STATEMENT` sources are contexts whose Rosetta statements are read
through their templates (`ComparisonTableService.findComparisonColumnDataByDataSourcesAndPaths`).

### 4.2 Selecting paths

1. `GET /api/comparisons/{id}/table-paths` lists every reachable path:
   `Neo4jComparisonAuxiliaryRepository.findAllComparisonTablePredicatePathsByComparisonId`
   walks `custom.subgraph(roots, {maxLevel: 10, relationshipFilter: "RELATED>"})` and
   collects `(subjectId, predicateId, objectIds, label, description)`.
   `LabeledComparisonPathBuilder.buildTree` (in `SpringDataNeo4jComparisonAuxiliaryAdapter.kt`) turns that into a tree, with a `sources`
   count per path (how many sources reach it).
2. The user ticks and orders paths (`FE: .../FirstColumnHeader/TablePathsModal/`).
3. `PUT /api/comparisons/{id}/contents` → `ComparisonTableService.update` validates
   typing and depth (`validateComparisonPathTypingAndDepth`), resolves labels, rejects
   paths that no longer exist (`ComparisonPathNotFound`), and stores the selection.
   **A published comparison rejects the update** (`ComparisonAlreadyPublished`).

Hiding a property means deselecting it. The selection is stored state of the
comparison, not a view.

### 4.3 Path traversal

`SpringDataNeo4jComparisonAuxiliaryAdapter.findComparisonColumnDataByRootIdsAndPaths`
builds one Cypher query. Per root, `buildQueryTree` emits nested
`OPTIONAL MATCH (s)-[:RELATED {predicate_id: <path id>}]->(n:Thing)` blocks, one
level per path depth, UNION-ed across sibling paths and recursing into children.
Values come back as `ComparisonTableValue(value: Thing, children: Map<ThingId,
List<ComparisonTableValue>>)`. The column title is the owning `Paper` (found via
inbound `P31`) when one exists, otherwise the root itself; the contribution becomes
the subtitle.

### 4.4 Row and column construction

`ComparisonTable.from(comparisonId, selectedPaths, columnData)`:

- **Columns = sources.** `titles[i]`, `subtitles[i]` per source.
- **Rows = selected paths**, as a tree: `values: Map<pathId, List<ComparisonTableRow>>`,
  where `ComparisonTableRow(values: List<Thing?> (one slot per column), children)`.
- **Multiple values:** `insert()` places a source value in the first row of that path
  whose slot for this column is empty **or already holds the same Thing id**;
  otherwise it appends a new row. So several values for one property become stacked
  rows, and **identical value ids across columns align on one row**. The frontend
  collapses after 6 rows (`FE: SelectedPath.tsx`, `MAX_ITEMS = 6`).
- **Missing values:** the slot stays `null`. Rows whose slots are all null are
  dropped in `build()`. The frontend renders an empty cell (`FE: Cell.tsx`), with no
  "not reported" / "not applicable" distinction.
- **Literal vs resource:** both are `Thing`. A literal shows its label; a resource is
  a link. Formatted labels are resolved afterwards (`FormattedLabelUseCases`).
- **Class / range handling:** none at table time. No datatype coercion, no unit.
- **Rosetta-Stone sources:** the latest statement version becomes the row value; its
  position inputs (`hasSubjectPosition`, `hasObjectPositionN`) become children.

### 4.5 Sorting and filtering

- Rows are ordered by the stored path selection (`ComparisonTable.sorted`).
- Columns follow `sources` order, reorderable by drag in edit mode.
- There is **no per-column sort**.
- Filtering is frontend-only (§ 5.2) and hides columns.

### 4.6 Publication and export

`ComparisonService.publish` runs:
`ComparisonPublishableValidator` → `ComparisonVersionCreator` →
`ComparisonVersionTableCreator` → `ComparisonVersionHistoryUpdater` →
`ComparisonVersionDoiPublisher`.

`ComparisonVersionTableCreator` freezes the **resolved** table into
`comparison_tables` under the new version id. On read, `ComparisonTableService.findByComparisonId`
returns that stored table for a published comparison and recomputes it live for an
unpublished one.

Exports:

- backend: CSV (`text/csv`, optionally transposed) and JATS XML;
- frontend: LaTeX, PDF (respects active filters), and RDF Data Cube
  (`FE: .../ComparisonHeader/hooks/useRdfExport.ts`, `qb:` vocabulary);
- a version diff page (`FE: src/app/comparisons/diff/[oldId]/[newId]/`).

### 4.7 Does a comparison calculate evidence?

**No.** The table is a projection of recorded values. Searched space: all backend
`*.kt` (excluding tests) and all frontend `src/**/*.{ts,tsx,js,jsx}` for effect
size, meta-analysis, confidence, SD / variance, median, average, aggregation,
normalisation, QUDT, unit. No statistical computation over cell values exists. The
one numeric-semantics site is the Papers-with-Code benchmark read model
(`CT/.../BenchmarkService.kt`, QUDT-shaped subgraph), which returns scores as
**strings** and never compares or aggregates them.

### 4.8 Measured on a live comparison

`GET https://orkg.org/api/comparisons/R44930/contents` (Accept
`application/vnd.orkg.comparison.v3+json`), "COVID-19 Reproductive Number Estimates",
published, head `R761413`, queried 2026-10-03:

- 31 columns (papers); 5 root paths: `location`, `Time period` (→ `has beginning`,
  `has end`), `Basic reproduction number` (→ `Has value`, `Confidence interval (95%)`),
  `Method`, `Approaches`.
- 186 root-level cells, 54 of them `null` (29 %).
- **R0 point estimates: 29 of 31 typed `xsd:string`, 2 typed `xsd:decimal`.**
- The 95 % CI is a nested resource whose bounds sit one level deeper than the stored
  selection, so the published table shows only the label "Confidence interval (95%)".

A request **without** the v3 Accept header returns HTTP 406. That matches
`ComparisonControllerV2`, which answers 406 on every route as a tombstone for the
retired v2 media type.

---

## 5. Faceted search: 2021 design vs 2026 implementation

### 5.1 The 2021 paper (historical design intent)

arXiv:2107.05738, *"Demonstration of Faceted Search on Scholarly Knowledge Graphs"*,
Heidari, Ramadan, Stocker, Auer; WWW '21 Companion; DOI 10.1145/3442442.3458605.
It is a 2-page demo paper with **no formal facet model and no evaluation**.

What it states:

- facets are per comparison and "inferred automatically from the property type",
  with templates supporting "the dynamic and automated construction of facets";
- per type: string multi-select with autocomplete; numeric exact value or `>` / `<`
  range; date range; **exclusion** of values for every type;
- the filtered subset plus its configuration can be saved "as a new comparison …
  with a permanent URL";
- the demo used a COVID-19 comparison of 31 papers.

**Correction to the brief:** the `path / range / values / exact` representation does
**not** come from this paper. It is the current observatory filter (§ 5.3).

### 5.2 Current per-comparison column filters (the paper's descendant)

`FE: src/components/Comparison/ComparisonTable/RowHeader/FilterPopover/`:

- `FilterType = 'category' | 'number' | 'date' | 'text'`.
- The type is **guessed from value labels** (`useFilters.tsx::getType`), not from the
  template or the literal datatype.
- Matching: category membership; `Number(label)` inside min/max; case-insensitive
  substring; date range.
- Filters AND across properties; a column passes a filter if **any** of its row
  values matches. The effect is to **hide columns**.
- State lives only in the URL `filters` parameter.
- **Lost since 2021:** exclusion, and saving a filtered subset as a new comparison.
- Inferred, not tested: `applyFilters` collects values by path id only, so the same
  predicate under two parent paths may be merged.

### 5.3 Current observatory paper filter (`SearchFilter`)

`G/graph-core-model/.../SearchFilter.kt`:

```
typealias PredicatePath = List<ThingId>
SearchFilter(path: PredicatePath, range: ThingId, values: Set<Value(op, value: String)>, exact: Boolean)
Operator ∈ { EQ, NE, LT, GT, LE, GE }
```

- Used by **one** endpoint: `GET /api/observatories/{id}/papers?filter_config=…`.
- Curators store facet *definitions* (`community/.../ObservatoryFilter.kt`: path,
  range, exact, featured; **no values**); users supply values.
- Query (`G/graph-adapter-output-spring-data-neo4j/.../SpringDataNeo4jStatementAdapter.kt`,
  `findAllUnpublishedPapersByObservatoryIdAndFilters`, ~L727–850): anchored at the
  paper's contribution (`P31`); `exact = false` prefixes `-[:RELATED*0..(10-|path|)]->`
  so the path may begin anywhere below the contribution; `range` picks the terminal
  node label / datatype; values OR within a filter, filters AND.
- **The compared value is `n.label` against a `String` parameter**, so `LT`/`GT` on
  numbers is lexicographic. Verified by reading the query builder; not executed.
- The API docs' operator table swaps `GE` and `GT` relative to the code
  (`documentation/.../appendix/filter-configs.adoc`).

### 5.4 Search "smart filters"

`FE: src/app/search/` → `smart-filters` service (Mistral via Ollama + DBpedia
Spotlight). It returns `facet_value_pairs` with paper ids per facet value, derived
from abstracts. It is read-only and writes nothing to the graph.

### 5.5 Is a facet `property-path × value-domain × predicate`?

For § 5.3, yes, literally: `path` (predicate path) × `range` (value domain as a
class / datatype) × `values` (a disjunction of `(op, value)` predicates), with
`exact` selecting anchored vs suffix match. The result is a **subset of papers**.

Grades against AdaWorldAPI concepts (the facet's *function* is "select a subset of a
population by a predicate over a path-reached value"):

| AdaWorldAPI concept | Relation | Why |
|---|---|---|
| ontology rail / graph path | ANALOGOUS | Both name a route to a value. ORKG's route is a free predicate-id list over a mutable graph; a rail is a fixed carving of a canonical address (`le-contract.md` § 3). A rail is not traversed; a path is. |
| `ClassView` (`lance-graph-contract/src/class_view.rs:1065`) | ANALOGOUS to `range` + template | A `ClassView` fixes a class's field basis at compile time. ORKG's `range` is checked per query and enforced nowhere else. |
| mask (`FieldMask` `:70`, `WideFieldMask` `:243`; `lance-graph-mask-risc`) | ANALOGOUS (same operation shape, different mechanism) | Both produce "the members that satisfy a predicate". ORKG computes a result set by string comparison in Cypher; a population mask is a bit vector over a canonical row population, composable without materialising members. |
| fold | NO MATCH | ORKG has no fold. Ours is also not built: "fold result at v into v+1" is D-LXC-6, Queued. |
| "epistemic basin" | NO MATCH | The term does not exist in either codebase. Episodic basins are shipped (`deepnsm-v2/src/basin.rs:42`), but they measure a subject's neighbourhood width, not a facet. |
| causal edge (`CausalEdge64`) | ORTHOGONAL | A facet selects; an edge asserts. |

---

## 6. Extraction boundary

| Question | Answer | Evidence |
|---|---|---|
| What can ORKG extract automatically today? | NER on titles / abstracts, research-field classification, template and predicate recommendation, LLM property / value checks, PDF table extraction, SciKGTeX PDF metadata, and an LLM extraction plan executed into a CSV of strings | `orkg-nlp-api/app/routers/{annotation,clustering,nli,text,pdf}.py`; `agentic-loop/app/agents/{planner,executor}.py` |
| What requires authoring / curation? | Every graph write. Templates, path selection, comparison sources, observatory filter definitions | frontend write sites (§ D of the frontend report); `ObservatoryFilter` |
| Does NLP produce final assertions or suggestions? | **Suggestions.** The NLP API's only backend calls are GET lookups. Writes happen in the frontend on a click. The agentic loop needs the user to **approve** the plan; its import tags statements `AI_GENERATED`, and publishing is **blocked until every AI cell is reviewed** (`FE: .../Publish/hooks/usePublish.ts`; `FE: src/components/Comparison/ComparisonTable/AiReview/`). One exception: the standalone `orkg-comparisons-creator` writes contributions tagged `AUTOMATIC`; the frontend does not call it. | `orkg-nlp-api/app/services/backend.py`; `orkg-comparisons-creator/api/comparisons/simple.py` |
| How are extraction method / provenance recorded? | `ExtractionMethod` per node and edge, stamped per write batch, plus `createdBy` / `createdAt`. Most frontend writes after a suggestion leave it `UNKNOWN`. Rosetta-Stone versions add `certainty` and `negated`. | § 3 |
| Does ORKG calculate statistical evidence from underlying observations? | No | § 4.7 search space |
| Does it revise contradicting claims? | No revision. Contradiction is representable only as separate (optionally `negated`) Rosetta statements side by side. | `RosettaStoneStatementVersion` |
| Does it normalise study populations / effect measures? | No | § 4.7; R44930 stores R0 as strings |
| Does it identify duplicate cohorts? | No. "Duplicate" code is paper / entity dedup only (`createPaperMergeIfExists`, DOI uniqueness). | frontend `duplicate` hits |
| Causal inference? | No | `causal` = 0 hits in frontend `src/` and backend `*.kt` |
| Meta-analysis? | No | `meta-?analys` = 0 hits |

How to read each "no":

- **Statistics / meta-analysis / effect normalisation:** architecturally absent from
  backend and frontend. Not delegated to any service read. The comparison's
  `searchProtocol` records PRISMA-style metadata (studies returned / retained) but
  nothing computes over it. Two NLP experiment projects (`r0-estimates`,
  `virology-dashboard-*`) were **not inspected** and may hold domain numerics.
- **Revision / contradiction:** out of the data model. `certainty` is a three-level
  label set by the author, not a computed quantity.
- **Duplicate cohorts / causal inference:** not found within the closed search spaces
  above. Nothing suggests either is in scope.

ORKG's design treats a scholarly claim as **a curated record about a paper**, made
machine-actionable for comparison and retrieval. Absence of evidence computation is
the scope of that design, not a defect.

---

## 7. Cross-map to AdaWorldAPI

| ORKG | AdaWorldAPI current candidate | Relation | Evidence |
|---|---|---|---|
| Paper (+ DOI) | provenance / source identity | **NO MATCH** today | MedCare `provenance.rs:270` `ProvenanceWitness` has `source: WitnessSource ∈ {OracleModule, ObservedEpisode, LearnerRun}`. There is no document / DOI variant, and the type has no caller outside its own module. `CausalEdge64` and AriGraph `Triplet` carry no source field. |
| Contribution | evidence episode / claim population | **NO MATCH** | No contribution / claim / study classid is minted in `ogar-vocab`. Nearest: `deepnsm-v2/src/evidence.rs:40` `EvidenceBasin` (a subject neighbourhood in a corpus) and MedCare `Cohort` (`medcare-cohorts/src/lib.rs:2220`, synthetic). Neither is "one study's reported findings". |
| Statement | SPO / `CausalEdge64` / AriGraph edge | ANALOGOUS | ORKG: string predicate id, author, time, extraction method, **no truth**. `Triplet` (`arigraph/triplet_graph.rs:16`): string relation, NARS truth, timestamp, **no author / source**. `CausalEdge64` (`causal-edge/src/edge.rs:161`): 8-bit palette S/P/O, NARS f/c u8, Pearl mask, **no provenance**. Each side has an axis the other lacks. |
| Rosetta-Stone statement (n-ary, `certainty`, `negated`) | NARS-graded edge / `ontology_warrant::Quorum` | ANALOGOUS; ORKG's qualifier STRICTLY WEAKER | `certainty` is a three-level authored label. `NarsTruth` (`lance-graph-contract/src/exploration.rs:89`, revision `:110`) is a revisable (f, c). `Quorum` (`ontology_warrant.rs:59`) grades a claim from counts of corroborating / dissenting / abstaining sources and deliberately cannot name them. ORKG's n-ary positions have no AdaWorldAPI counterpart beyond fixed S/P/O. |
| Predicate | ontology relation / CE64 predicate coordinate | ANALOGOUS | ORKG predicate = a global graph node with label and optional URI. CE64 P = an 8-bit palette index into a local codebook; it is not a global identity. OBO relations (RO) baked through `ogar-obo` are the closer match, but no ORKG-like free predicate registry exists. |
| Class | OGAR / OBO / DOLCE class identity | ANALOGOUS | ORKG class = a free node (optional URI), and classes are Neo4j labels on resources. OGAR classid = a minted `(concept << 16) \| app` address resolved through a `ClassView`. DOLCE is real only on the Odoo path; `ogar-class-view/src/lib.rs:414` `dolce_category_id` returns 0. |
| Template (SHACL NodeShape) | `ClassView` field basis | ANALOGOUS | Both declare the expected fields of a class. ORKG enforces only on the template-instance endpoints; a `ClassView` is the compile-time shape every row of the class is read through. `elixir-template` (`crates/elixir-template/src/lib.rs:106`) is a **step pipeline**, ORTHOGONAL to a schema template despite the name. |
| Property path (`ComparisonPath`, depth ≤ 10) | ontology rail / graph path / mask projection | ANALOGOUS to a graph path; ORTHOGONAL to a mask | A path selects *where to look*; a mask selects *which members*. Our nearest "where to look" is a `ClassView` field position, which is flat, not a tree. |
| Comparison (sources × paths → table) | evidence-population projection | ANALOGOUS shape; ORKG **computes nothing** | Sources ≈ a population; selected paths ≈ a field mask; the table ≈ a projection. A projection with computed evidence (pooled estimates, revised truth) is **not built** on our side either (§ 9). |
| Facet (`SearchFilter`) | mask over a canonical population | ANALOGOUS | § 5.5. |
| Published snapshot (frozen table + DOI) | Lance version / replayable projection | ANALOGOUS | ORKG freezes the resolved table but re-links live contributions. Lance versions (`lance-graph/src/graph/versioned.rs:499` `at_version`, `:522` `tag`, `:541` `diff`) freeze the whole dataset. ORKG has external citability (DataCite DOI); a Lance version has none. Note that `versioned.rs` is not yet wired to `temporal.rs`. |
| `ExtractionMethod` | MedCare `EpistemicOrigin {Oracle, Observed, Learned}` (`provenance.rs:82`) | ANALOGOUS | Both label the channel a fact came through. ORKG's is per write batch and fixed; MedCare's is per witness, and the edge truth comes from revision over witnesses. ORKG adds a human review state (`AI_GENERATED_WITH_MANUAL_REVIEW`) that we do not model. |
| `searchProtocol` (inclusion / exclusion, studies retained) | — | NO MATCH | Nothing models a literature search protocol. |
| Research field, observatory, organization | — | NO MATCH | Community curation structures; out of our scope. |

---

## 8. What ORKG does that we should reuse conceptually

Contracts and interoperability ideas, not frontend features:

1. **AI-review gate before publication.** AI-extracted values carry their own state,
   are reviewed cell by cell, and **block publication** until reviewed; rejected
   values are deleted, accepted ones re-tagged. This is the human-scale version of
   "similarity proposes / CAM addresses": a proposal must not reach the published
   record unexamined. Worth mirroring as a state on the propose side of any
   literature ingest.
2. **Auto-demotion on edit.** Editing an AI-generated element without restating the
   method demotes it to `MANUAL`, so a label never outlives the content it
   described.
3. **First-class epistemic qualifiers on n-ary statements** (`certainty`, `negated`,
   append-only versions, soft delete). We already go further on certainty with NARS,
   but **`negated` as an explicit field** and **n-ary positions** are worth noting:
   a reported finding is rarely binary (exposure, outcome, population, estimate,
   interval).
4. **A comparison's source is typed** (`THING` vs `ROSETTA_STONE_STATEMENT`). The table
   builder does not care where a column came from. A projection over mixed origins
   (literature claims and observed cohorts) needs the same indifference.
5. **The value-identity row merge** (§ 4.4): identical value ids across sources share
   a row, so agreement is visible structurally. With canonical ontology addresses on
   our side, the same rule would align studies on the same concept for free.
6. **Search-protocol metadata** on a comparison (inclusion / exclusion criteria,
   search strings, studies returned / retained). Any meta-study projection should
   carry its selection protocol next to its result.
7. **Publish = freeze the resolved table + changelog + version diff + DOI.** Our Lance
   versions provide replay. A *citable* identifier for a published projection is a
   separate, missing concern.
8. **Typed path selection with bounded depth and validation** (`MAX_PATH_DEPTH`,
   level-typed Rosetta paths, `ComparisonPathNotFound` on stale paths).
9. **External term import by URI or short form** through OLS / Wikidata
   (`/api/import/{resources,predicates,classes}`): a small, boring contract for
   pulling single terms rather than whole ontologies.
10. **Export targets for interoperability:** paper CSV import schema
    (`data-import/.../Schemes.kt`, § 10), RDF Data Cube for comparison tables, JATS.

Lessons from what ORKG left open, framed as constraints on our side rather than
criticism:

- **A numeric value stored as a label cannot be ranged.** R44930 holds 29 of 31 R0
  estimates as `xsd:string`, and the facet compares strings. Typed numerics with
  units must be fixed at ingest; they cannot be recovered at query time.
- **"Missing" needs more than one state.** A `null` cell conflates "not reported",
  "not applicable" and "not extracted". deepnsm-v2 already keeps `Option<u64>` so
  that unknown is never zero (`lexical.rs:700` `missing_counts_stay_missing`).
- **Advisory schemas drift.** Templates are enforced on one endpoint and bypassed by
  raw writes, and the ontology repo has gone unreferenced since 2023. A shape that is
  checked only on one path is documentation.
- **Type guessing from labels** (frontend filter `getType`) is the downstream cost of
  untyped ingest.

---

## 9. What our substrate does below ORKG

Status per stage, measured at the SHAs in § 0. SHIPPED = code on main with a test
function (not executed here). PLANNED = plan / board only.

### 9.1 Text leg (raw text → claims)

| Stage | Status | Evidence |
|---|---|---|
| COCA lexical evidence per (surface, PoS, lemma), unknown ≠ 0 | SHIPPED | `deepnsm-v2/src/lexical.rs:332` `LexicalEvidence`; test `:594` `homograph_keeps_every_pos_reading` |
| Competing readings kept through the parser (`PosSet`, reading mask) | PLANNED (lexicon keeps them; parser takes one) | `fsm.rs:13` "deliberately commits to ONE coarse SPO reading"; D-LXC-2 Queued |
| Parser-configuration population | PLANNED | sizing only, `.claude/plans/deepnsm-v2-lexical-evidence-consumer-v1.md` |
| Relative clauses | SHIPPED, one level | `deepnsm-v2/src/fsm.rs:118` `parse_to_spo`; tests `:276`, `:288` |
| Episodic basins + held-out gate | SHIPPED | `deepnsm-v2/src/basin.rs:42`, `:185`; contract `episodic_basin.rs:86` |
| Corpus-level evidence against a shuffle null | SHIPPED | `deepnsm-v2/src/evidence.rs:234` `shuffle_beliefs_null` |
| Text-derived SPO → ontology address | **NOT SHOWN** | No code joining deepnsm-v2 triples to OBO / OGAR addresses was found in this audit. |
| Reading **effect size + sample size** out of a publication | ABSENT | None of the three evidence legs reads it. |

### 9.2 Observation leg (raw observations → evidence)

| Stage | Status | Evidence |
|---|---|---|
| Cohort statistics (Pearson, Spearman, Cronbach α, ICC, κ, ω, η², t-tests, ANOVA) | SHIPPED | `jc/src/reliability.rs:94/172/217/284`; `jc/src/stats.rs`; MedCare `cohort_stats.rs:592` `disease_statistics` |
| …over **live** patients | PARTIAL | `/views/studie/statistik` runs on 1,200 **synthetic** patients; real patients enter only through the optional MySQL airgap |
| Observation → ontology-addressed evidence edge with provenance | SHIPPED (MedCare) | `medcare-cohorts/src/loinc_evidence.rs:108` `criterion_edges`, `:208` `patient_evidence`; test `:370` |
| Fisher-z | SHIPPED, transform only | `lance-graph-contract/src/distance.rs` `fisher_z`; `jc/src/stats.rs:1193` `fisher_2z` |
| Inverse-variance pooling, CI on r, heterogeneity, random effects | **ABSENT** | closed search (`n - 3`, `inverse_variance`, `meta_analys`, `DerSimonian`, `random_effects`, `cochran`) over all three repos |
| d-family effect sizes | out of scope in `jc` by design (`stats.rs:35-42`); a struct exists in `lance-graph-cognitive/src/search/scientific.rs:79` `EffectSize` | — |
| CLAM / CHAODA | lite SHIPPED (`perturbation-sim/src/chaoda.rs:72`); full engine in ndarray, **not read** | — |

### 9.3 Shared reasoning space

| Stage | Status | Evidence |
|---|---|---|
| Ontology normalisation (ICD / MeSH / SNOMED / … → MONDO-trie address) | SHIPPED | OGAR `ogar-obo/src/crosswalk.rs:47`, `:97` `resolve_icd10`; MedCare `obo_store.rs:120` |
| Study-metadata ontologies (BFO, IAO, OBI, OBCS, **SEPIO**, **ECO**) | REGISTERED, **not bakeable** | OGAR `ogar-obo/src/registry.rs:221` `META_STUDY_SPINE`, gated by a pad-collision issue |
| NARS revision | SHIPPED, nine separate truth structs | contract `exploration.rs:110`; planner `nars/truth.rs:57`; D-BBB-NARS-4 open |
| Claim warrant from independent-source counts | SHIPPED | `ontology_warrant.rs:59` `Quorum`, `:112` `warrant()` |
| Rules from co-occurrence → NARS | SHIPPED; wiring to `SpoStore` **Blocked** | `lance-graph-arm-discovery/src/translator.rs:68`; D-DNV-3 |
| Per-witness provenance with revision | SHIPPED, **no callers** | MedCare `provenance.rs:270` |
| Replayable versions | SHIPPED, separate from `temporal.rs` | `lance-graph/src/graph/versioned.rs:499` |
| Population masks / field masks | SHIPPED | `class_view.rs:70/243`; `lance-graph-mask-risc` |
| Comparison / meta-study projection | **ABSENT** | — |

### 9.4 Verdict on the distinction

The target pipeline is

```
raw text + raw observations
  → normalise onto shared ontology coordinates
  → preserve provenance
  → compute / revise evidence
  → expose comparison / meta-study projections
```

Every stage has **some** shipped component, on one leg or the other. The
distinction from ORKG is **not proven end to end**, because three joins do not exist:

1. text-derived claims are not normalised onto the same ontology addresses as
   observations (no join found);
2. no provenance type names a **publication** as a source, and the one rich
   provenance type has no callers;
3. nothing pools reported effects or projects a population into a comparison view.

What *is* shown is narrower and real: each leg has components that compute and
revise evidence (lexical counts with missing ≠ zero, held-out basins, shuffle nulls,
reliability statistics over cohorts, quorum-graded warrants, NARS revision), and
ORKG has no counterpart to any of them.

---

## 10. Reference projection (design only, not implemented)

**Purpose:** interoperability and testing. One way, Ada → ORKG. Loss-aware.

### 10.1 Smallest target: ORKG's paper-CSV import

ORKG already accepts a paper CSV (`/api/csvs`, schema `data-import/.../Schemes.kt`
`paperCSV`) with closed `paper:` and `contribution:` header namespaces, free
`orkg:<predicateId>` columns, typed cells (`text`, `string`, `decimal`, `integer`,
`boolean`, `date`, `url`), and a **per-row `contribution:extraction_method`** column.
Emitting that CSV needs no ORKG client code and exercises ORKG's own validation.

| Ada concept | ORKG column / element | Rule |
|---|---|---|
| evidence source (a publication, once a source variant exists) | `paper:title`, `paper:doi`, `paper:authors`, `paper:publication_year`, `paper:research_field` | one row per (source, finding) |
| evidence claim / study episode | one contribution per row | |
| canonical concept (OGAR classid / OBO address) | `orkg:<predicate>` cell holding a **resource** imported by URI (`/api/import/resources` via OLS) | never a bare label |
| canonical relation | the `orkg:<predicateId>` header, resolved by URI | |
| measured value / effect | typed `decimal` / `integer` cell | never `string` for numerics |
| interval, unit | separate typed columns (`…_ci_low`, `…_ci_high`, `…_unit`) | ORKG has no unit type |
| NARS (f, c) | Rosetta-Stone `certainty` bucket if Rosetta is used; otherwise a `decimal` column with an explicit predicate | lossy (§ 10.3) |
| origin channel | `contribution:extraction_method = AUTOMATIC` | deterministic, not an LLM, so **not** `AI_GENERATED` |
| comparison dimensions | the selected `orkg:` columns → `PUT /contents` selected paths | |

### 10.2 Richer target, if certainty and negation matter

Post Rosetta-Stone statements (`/api/rosetta-stone/statements`) against a template
with positions (population, exposure, outcome, estimate, interval). This keeps
n-ary structure, `negated`, and a coarse certainty, at the cost of defining an ORKG
template first.

### 10.3 What ORKG cannot represent without extension

- NARS (f, c), and its revision history: at best a three-level `certainty` or a
  free decimal with no semantics.
- Per-witness provenance (several sources revising one edge): ORKG has one
  `createdBy` per statement.
- Epistemic-origin masks (`OriginMask`, oracle / observed / learned together).
- Episodic-basin membership and width; held-out gate results.
- Sufficient statistics (n, Σx, Σx², r with n) needed to re-pool later.
- CLAM / CHAODA geometry and anomaly scores.
- Temporal version ranges (`QueryReference::at(v, rung)`); a published comparison is
  one snapshot.
- Competing readings / parser-configuration populations (planned on our side).
- Population and fold state (masks over a canonical population).
- Canonical address identity: ORKG ids are graph-local (`R…`); our identities
  survive only as URIs on imported resources.

### 10.4 The reverse direction

ORKG → Ada (ingest ORKG statements as sourced claims) is the more useful path for the
benchmark below, and it sits squarely on the **propose** side of the firewall:
imported claims would be evidence to revise, never addressed facts. It is noted here
and not designed.

---

## 11. Candidate benchmark

**Target:** ORKG comparison **R44930**, "COVID-19 Reproductive Number Estimates"
(published, head `R761413`, 31 papers, 5 paths; § 4.8). It is the same scale as the
2021 demo comparison, it is clinical, and its R0 point estimates, CIs, locations and
time periods are exactly the dimensions a pooled estimate needs.

**Steps:**

1. Freeze R44930's table (`/contents`, v3) and the 31 paper DOIs as the reference.
2. Obtain the 31 papers (open-access subset first; record which are unobtainable).
3. Reconstruct each dimension from our pipeline: location → place address, time
   period → interval, R0 → typed decimal, CI → typed interval, method → concept.
4. Project our result into the § 10.1 CSV shape and into ORKG's table shape.
5. Compare per cell: entity identity, property identity, value (numeric equality
   within reporting precision), missing (and *which kind* of missing), provenance.
6. Report separately what our side knows that the table cannot hold (§ 10.3), for
   example sufficient statistics and a pooled estimate with heterogeneity.

**This is a reference benchmark, not a superiority demo.** Expected outcome: step 3
fails for text extraction today (no effect-size reader, § 9.1), and step 6 has no
pooling to report (§ 9.2). The benchmark's first value is to make those two gaps
concrete against a real, externally curated table.

---

## 12. Mission thesis, tested

> ORKG is primarily a structured scholarly contribution and comparison system.
> MarkovNSM/lance-graph aims to make the structured contribution itself a projection
> of a deeper evidence substrate in which literature-derived claims and directly
> observed cohort data can occupy the same ontology-grounded, provenance-preserving
> reasoning space.

**First sentence: true, with two corrections.**

- ORKG is not only bookkeeping. It has first-class epistemic qualifiers (Rosetta-Stone
  `certainty`, `negated`), an LLM extraction pipeline, and a human review gate that
  blocks publication. What it lacks is **computation over evidence**, not epistemic
  vocabulary.
- Its faceted exploration is weaker than the 2021 paper's intent: exclusion and
  save-as-comparison are gone, and numeric facets compare strings.

**Second sentence: an aim, not yet a property of the code.** The boundary lies
exactly at the three missing joins in § 9.4. Literature-derived claims do not yet
enter the reasoning space at all: there is no publication source type, no
contribution / study / claim classid, the study-metadata ontologies (SEPIO, ECO)
are registered but unbakeable, and no reader extracts reported effects. Observed
cohort data does enter it, but the shipped study page runs on synthetic cohorts.

Corrected statement, matching current code:

> ORKG records curated scholarly claims and projects them into comparisons; it does
> not compute over them. AdaWorldAPI has shipped components that compute and revise
> evidence from raw text and from cohort observations on ontology addresses. Joining
> literature claims and observations into one provenance-preserving space, and
> projecting comparisons out of it, is still to be built.

---

## 13. Anything in ORKG that should alter our design

1. **Provenance must name a publication.** Add a document / DOI source variant
   before any literature ingest, beside `WitnessSource`'s module / episode / run
   variants. Without it, an ORKG-shaped projection has nothing to put in `paper:`.
2. **Typed numerics with units and intervals at ingest.** R44930 shows what happens
   otherwise.
3. **Several missing states**, at least "not reported" vs "not applicable" vs "not
   extracted".
4. **A review state on the propose side**, modelled on ORKG's AI-review gate.
5. **`negated` and n-ary findings** deserve explicit fields; a reported finding is
   not a triple.
6. **A citable identifier for a published projection** is a separate concern from
   replay, and Lance versions do not provide one.

## 14. Already built that makes an ORKG layer a projection, not a substrate

- Canonical ontology addresses with external-code crosswalks (OGAR `ogar-obo`): a
  comparison column could key on an address rather than a label, which is what ORKG's
  value-identity row merge would need.
- `ClassView` + field masks: the "selected paths" of a comparison are a field mask
  over a class basis.
- Population masks (`lance-graph-mask-risc`): a facet is a mask, computed without
  materialising members.
- NARS revision and `Quorum` warrants: a cell can carry a revised truth, not just a
  value.
- Lance versioning: a published comparison could be a tagged version of the
  population it projects.
- The reliability battery in `jc`: per-cohort statistics already exist to be
  projected.

Each is a component. The projection that composes them is not built (§ 9.4).
