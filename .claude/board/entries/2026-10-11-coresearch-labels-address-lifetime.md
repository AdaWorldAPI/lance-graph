# Coresearch — concept addresses, labels per language, source-id mapping and id lifetime (2026-10-11)

**Status:** OPEN — exploration map. It ratifies nothing; the operator chooses.
Plan context: `v3-mandatory-hotplug-reading-v1.md` rules 3–5, O1, O2.

## The question, after the premise audit

The draft asked where "Stundenzettel / TimeSheet / Zeiterfassung" live as
labels of one concept. The audit found four category errors and re-asked it:

- `0x0103` is the cross-app concept `billable_work_entry`.
  `ogit.WorkOrder:TimeSheet` is WoA's own class. They are two entities related
  by a **mapping** (O2), not two spellings of one id.
- OGAR's port alias column is not a label table. Its rows mix words in a
  language, source-class ids, another app's class and collapsed subtypes.
- The `0x1000` marker and the render prefixes sit on the **reading** axis.
  They left this question.
- The immutable address today is the **u16** concept. The u32 classid's low
  half is app-owned.

Three questions remain. **Q1 lexical:** where do per-language labels of a u16
live? **Q2 mapping:** how does a source id (class name, OGIT CURIE) relate to a
u16, with which relation kinds, and who owns the mapping? **(c) lifetime:** how
is an id deprecated but still resolvable, and what separates an address from a
positional ordinal?

## The shape the rows converge on

1. **One lifecycle ledger in `ogar-vocab`** (X2, X3, X21). It is a golden list
   of every shipped `(concept, u16)` pair, plus a successor list keyed by the
   u16 (`ReplacedBy` / `Consider` / `Tombstone`). A renumber or a reuse becomes
   a red test. The ledger is data in OGAR, never a mirrored column in the
   contract (that would be a lockstep table) and never `#[deprecated]`.
2. **`PortSpec` is the mapping set** (X14, X17, X18, X19, X20). The port's
   namespace identifies the set. Each alias row gains a relation kind, and
   CURIEs resolve as `ogit.<NS>:<local>` → the port with that namespace →
   `class_id(local)`. An unmapped name stays `None`, and no mapping row ever
   feeds `is_a`.
3. **Labels as u16-keyed rows in `ogar-vocab`** (X8, X9, X13, X10): `(address,
   BCP-47 tag, pref|alt|hidden, text)`, with fallback at lookup time. They are
   consumed at projection; the contract gets only the type and a lookup trait.
   A customer binary gets labels as codegen output, never by linking OGAR.
4. **Ordinals stay per-load and are named as such** (X5, X7, X22, X23). The
   three u16 spaces get distinct names: the codebook address, the registry
   surrogate and the load ordinal. Every load records the git ref of its
   source tree as a provenance receipt, never as a gate.

## Verdicts

`F` is the firewall verdict: PASS, C (fits only through a named seam), or TRAP.
"probe" names the falsifier result, where one was run.

| row | idea | F | verdict | probe / reason |
|---|---|---|---|---|
| X1 | the u16 is an opaque surrogate except for the domain byte | PASS | OPERATOR (with X6) | **KILLED.** One production reader gives the low byte a second meaning: `is_palette_seat` (`lance-graph-ogar/src/lib.rs:504`) reads `0x17xx` low byte ≥ `0x17` as a consumer palette seat. OGAR's own doc says the slot is the shared concept (`ogar-vocab lib.rs:1576-1580`). The 6 domain-byte readers are permitted. |
| X2 | never renumber: test shipped ids against a frozen list | PASS | ADOPT-NOW | Not run (cross-repo). Today a lockstep renumber passes every gate. The test lives in `ogar-vocab`; the contract's value pins check the same list. |
| X3 | deprecation as data with a successor pointer | C | ADOPT with X2 | **Half KILLED.** "Deprecated, kept = resolvable" holds in the default build: 9/9 fossils read like their successor. Under `--no-default-features` the 5 V3 fossils read `DEFAULT`, the same as a never-minted key. No successor function exists. `flip_classid` is an order flip, not a successor, and misses 4 of 9 because the V3 appid was renormalised (`0x0100_1000` vs `0x0101_1000`). |
| X4 | an obsolete id resolves but leaves the rails | C | PARK (codebook) / PROBE (ogar-obo) | No production `is_a_parent` exists, so the codebook has no rail to leave (P-X19 precondition). `ogar-obo` already drops obsolete terms from rails and crosswalk; it lacks `replaced_by`. P-X4 not run. |
| X5 | record the source version per load | C | PROBE | Provenance receipt only (OGIT's version fields are placeholders). Gating on it would be an internal pin. P-X5 not run (needs history). |
| X6 | writer namespaces (two writers on one u16) | C | OPERATOR | **SURVIVED as a measured collision.** 3 of 114 codebook ids are also code-minted appid heads: `0x0101` project / OpenProject, `0x0202` commercial_document / Odoo, `0x0A01` anatomical_structure / q2. Live consequence: `concept_override` (`lance-graph-ogar lib.rs:489-492`) gives any consumer plugging these **concepts** the head's storage reading (`PROJECT_V3`, `ERP_V3`, `FMA_V3`), not `PLUG_AND_PLAY_V3`. Ports alias `0x0202` from WoA, SMB, Odoo and HubSpo. Latent today (no plug names them). |
| X7 | address and ordinal are different types | PASS | ADOPT-NOW (rename first) | New: the contract declares `ClassId` twice, `u32` (`rbac.rs:331`) and `u16` (`class_view.rs:54`). P-X7 not run. |
| X8 | language-tagged labels on the concept | C | ADOPT shape, new type | **KILLED as "absorb the three shapes":** 0/3. `Label` has no u16, `LabelDTO` has no language and holds a key, `FieldRef` is a field label. The twin holds (`"Kunde"` → `0x0204, de, alt`). The label row is new data, not a merge of existing shapes. |
| X9 | fallback at lookup, never stored copies | PASS | ADOPT-NOW (latent) | **SURVIVED.** `Label::en_only` yields a German DTO equal to the key with no fallback flag. Production ontologies pass real German, so the defect is latent. |
| X10 | shared pool, ClassView selects | C | OPERATOR | **KILLED premise:** the `ClassView` trait has no class-level label and is keyed by the u16, not the u32 (`E0599` / `E0308`). The ruled label home ("ClassView carries labels", keyed by the full classid) does not exist in the trait. |
| X11 | a label never carries a relation | PASS | ADOPT-NOW | Curation rule; see X18. |
| X12 | external per-language lexicon (OntoLex) | C | PARK | **KILLED:** German coverage is 5/114 (6 lenient), no third language. |
| X13 | absence per language is representable | PASS | ADOPT with X8 | **SURVIVED:** `en_only("project") == new("project","project","project")`. The closed `Label` cannot express absence. |
| X14 | name lookup is scoped by source | C | ADOPT in `PortSpec` | Not run. Homonym known by reading: "User" → project_actor globally, OSM_USER in `OsmPort`. |
| X15 | locale reaches entities | C | PROBE after X7 | **SURVIVED:** locale reaches only the ontology name; entity names are the same in De and En. New: `EntityTypeDto.id` (`ontology_dto.rs:193`) exports the registry's append-order counter. The cache re-mints it on invalidation (`lance_cache.rs:119-133`), so an outside app persisting it holds a cache-lifetime ordinal. |
| X16 | an ingest that keeps language tags | C | PARK until X8 | Both readers drop tags (verified by reading). OGIT has 0 tagged `rdfs:label`. |
| X17 | typed, one-way, provenance-bearing mapping rows | C | ADOPT in OGAR | Not run. |
| X18 | split the alias column by relation kind | C | ADOPT, with explicit re-pins | **Four kinds KILLED as complete; vocabulary SURVIVED.** 188 rows: source class 128, subtype 5, discriminator value 8, other app 3, word in a language 32, OGIT IRI 9, unknown 3. WoA's Quote/Order/Invoice/CreditNote and their German forms are `doc_type` values of one class, not subtypes. OpenProject is 25/28 source classes (the twin). |
| X19 | mappings stay out of the hierarchy | PASS | PARK (rule for the first mapping store) | **SURVIVED:** no production `is_a_parent` and no mapping store exist. The twin sees a real `is_a` edge. |
| X20 | an unmapped entity carries `None` | PASS | ALREADY-HAVE | Not run. |
| X21 | mapping lifetime follows the source release | PASS | PARK | **KILLED:** 0 of 6 WoA source-class rows are stale. One is semantically wrong: `TimesheetActivity` holds no hours; they live on `TimeSheet`, which has no alias. |
| X22 | ordinals are per-scope positions, never addresses | PASS | ALREADY-HAVE (rule) | Not run. OgitTables documents ids as stable per tree. |
| X23 | the registry's append log as a stable surrogate | TRAP | SKIP | **SURVIVED mechanically, failed in substance:** TimeSheet kept id 1926, but one inserted file moved 14 other entities. Hydration walks `read_dir` order (LIFO stack), not names, so ids depend on the filesystem. The cache also deletes itself on any schema change. |
| X24 | lifecycle on the identity row, not the label | PASS | ADOPT (deferred guard) | No OBO label ingest exists yet. |

## Defects found on the way (no decision needed)

- `ClassId` is two types in one contract (`u32` and `u16`).
- `EntityTypeDto.id` exports a cache-lifetime ordinal to PostgREST, Phoenix
  and downstream apps.
- `Label::en_only` fabricates a German label (latent).
- Both TTL ingest paths drop language tags; OGAR keeps only the first
  `rdfs:label`.
- No test pins the history of shipped codebook ids.

## Open points for the operator

- **Two writers on three u16s (X6, X1).** `0x0101`, `0x0202` and `0x0A01` are
  both concepts and consumer appid heads, and since D-HPR-2 the head's storage
  reading wins for anyone plugging the concept. Options:
  - (a) key the storage readings by a consumer-owned value, not the concept;
  - (b) move the appid heads off occupied concept slots;
  - (c) accept the collision and document that plugging these concepts means
    that domain's storage model.
  `0x17` palette seats are the same shape in a domain with no codebook rows.
- **Label home (X10).** The trait cannot carry the ruled home. Choose between a
  u16-keyed pool in `ogar-vocab` that the ClassView selects from, and a
  class-level label method on `ClassView` keyed by the full classid.
- **O2 refined (X18, X21).** The hours row is `TimeSheet`; `TimesheetActivity`
  holds none. Does `ogit.WorkOrder:TimeSheet` map to `0x0103` as exact or
  close, and is "Stundenzettel" a German label of `0x0103` or only of WoA's
  class?
- **Narrowing (X18).** WoA's document words are `doc_type` values. Do they stay
  one concept (`0x0202`), or do `invoice` / `quote` / `credit` become their own
  concepts?

## What was not searched

The W3C OntoLex and SSSOM specs were read through papers only. SNOMED CT and
UMLS were used as structural patterns only; nothing about their terms is
recorded here. OGIT history was not fetched (depth 1), so X5 is unmeasured.
Probes P-X2, X4, X5, X7, X11, X14, X17, X20, X22 were specified but not run.

---

## Crosswalk — labels per language, source-id mapping, id lifetime (coresearch Phase 2, 2026-10-10)

Built by the main thread from 50 scout items (code, prior art, literature, systems, concepts). Self-contained: each row carries the mechanism and the evidence. Grades: outside READ-IN-FULL / SECTION-READ / ABSTRACT-ONLY / SECONDHAND; inside VERIFIED-IN-CODE / CLAIMED / ABSENT (closed search named). Relation: ALREADY-HAVE / PARTIAL / NEW / CONFLICTS-ANCHOR.

### Corrections the scouts made to the brief (all from reads)
- `OntologyDto::project(registry, namespace, key, label, locale)` EXISTS in lance-graph (`lance-graph-callcenter/src/ontology_dto.rs:80-112`), not in OGAR; its locale picks only the ONTOLOGY's own name through the closed 2-locale `Label{key,en,de}`; entity names are never localised. RDF-OWL-ALIGNMENT §5 ("SKOS-prefLabel projection") is overstated, not absent.
- OGIT: the 12 `@en` tags are all on `dcterms:description` in `ogit.ttl`; 0 of 3693 `rdfs:label` carry a language tag; 0 `@de`. `ogit:hide` = hidden-from-documentation (true on the meta-classes Entity/Attribute/Verb), NOT an obsolete marker. OGIT has 0 `owl:deprecated`, 0 `isReplacedBy`, 0 `end=`, 0 `owl:versionIRI`; `owl:versionInfo "__OGIT_VERSION__"` is a placeholder. Deprecation in OGIT appears only in prose (MARSNode) or Turtle `#` comments (isoCode, LegalSection, Region), which are not in the RDF graph.
- SKOS does NOT keep mappings apart from the hierarchy: `skos:broadMatch` is a sub-property of `skos:broader` (S41). Keeping mappings out of `is_a` is our design choice, not a SKOS guarantee.
- OgitTables' `KNOWN_REJECTED` includes `NTO/WorkOrder/verbs/`, so `LogsTime.ttl` is not loaded. Side finding (scratch run): OGIT's own `validate.sh` fails on an undefined `ogit.Compliance` prefix.
- Two independent u16 assignment spaces exist besides the codebook: the registry's append-order `EntityTypeId` and OgitTables' sort-order `TypeId`.
- The brief's constraint "a label names exactly one address" is STRICTER than every outside lexical model read (SKOS restricts prefLabel per concept per tag, not across concepts; Wikidata requires uniqueness only of label+description for items). Homonyms are real here too ("User" → PROJECT_ACTOR in OpenProject/Redmine/HubSpo, → OSM_USER in OsmPort).

### Rows

| id | row | idea | mechanism and evidence (outside, grade) | nearest inside surface (grade) | relation |
|---|---|---|---|---|---|
| X1 | IDENTITY | Opaque, meaning-free id; meaning lives in metadata | Local id carries no meaning; labels are metadata for humans; a type may be embedded only if indisputable, unchangeable and never reused across types (McMurry 2017 PLoS Biol L4, SECTION-READ). An id changes when information inside it changes, so classification/status stay out of it (Cool URIs 1998, READ-IN-FULL; OBO ID policy). SNOMED concept id carries no term (SECTION-READ). | u16 codebook address is a surrogate key whose high byte is a domain (the "embedded type" McMurry permits only if never reused) — `ogar_codebook.rs` (VERIFIED). OGIT identity is a name-bearing PURL `purl.org/<ns>/<itemname>`: a rename is a new identity (`NTO/PURL_ID_Registration.md`, READ). | PARTIAL (the u16 is the opaque key; the OGIT PURL is a natural key; the domain byte is out of scope per A4) |
| X2 | IDENTITY | Never delete, never reuse, never renumber — enforced against history | Deprecate, never delete or reassign; removed record resolves to a tombstone (McMurry L7, SECTION-READ). PostgreSQL enum labels: only ADD/RENAME, no removal; identity = OID, not position (SECTION-READ). OBO fp-003: a namespace must not have been used before (READ-IN-FULL). | Ruled: E-OGAR-NORTHSTAR-1 "append-only ClassIds, mint once, never renumber, tombstone" (CLAIMED — no tombstone carrier). Enforced only as uniqueness, non-zero, present-tense mirror parity, `MirrorDrift` by concept NAME, and a few value pins (`0x0103` at OGAR `lib.rs:2588,5958`); no test compares against a frozen list of shipped ids, so a lockstep renumber passes every gate (`ogar_codebook.rs:577-588,998-1039`, `hotplug.rs:600-649`, VERIFIED). Counter-precedent: OSINT rows `0x0700/0x0701` were DELETED (67→65) and u16 `0x0701` now means an appid (D-OSINT-APPID-NOT-CONCEPT; VERIFIED). | PARTIAL (rule exists; history not enforced; one deletion+reuse precedent) |
| X3 | IDENTITY | Deprecation as queryable DATA with a typed replacement pointer | `owl:deprecated true` is an annotation with no logical effect (OWL 2 §5.5, SECTION-READ). OBO fp-019: obsolete = flag + `IAO:0100001 term replaced by` (one exact) or `oboInOwl:consider` (several) + reason (READ-IN-FULL). SNOMED: inactive component stays; typed association REPLACED BY / SAME AS / POSSIBLY EQUIVALENT TO / PARTIALLY EQUIVALENT TO / MOVED TO / WAS A (SECTION-READ). Wikidata merge → redirect (`owl:sameAs`), never deleted or reused; inbound statements rewritten later by a bot (SECTION-READ). Cool-URIs-for-FAIR-KGs: tombstone (no successor / split, human reason) vs morph/merge with `dcterms:isReplacedBy` both directions (arXiv 2407.09237, SECTION-READ). KGCL: NodeObsoletion ≠ NodeDeletion (arXiv 2409.13906, READ-IN-FULL). | Only prose: `CLASSID_*_V3` "Deprecated, kept" doc notes; `*_LEGACY` pre-flip consts still resolve via `BUILTIN_READ_MODES`; `Membership` kept as a synonym of `Member` by comment + test (OGAR `ports.rs:176-185`); `ogar-obo` `obsolete: bool`; `lance-graph-ontology` `MappingRow.active` (soft-delete). Closed search (`deprecated:`/`replaced_by`/`superseded_by`/`obsolete`/`#[deprecated` over contract src + ogar-vocab src): no replacement pointer anywhere (VERIFIED). | PARTIAL (resolvable-kept practice exists; no machine-readable marker, no successor kind) |
| X4 | IDENTITY | An obsolete id still resolves but leaves the rails (hierarchy, mappings) | OBO fp-019: an obsolete term loses ALL logical axioms incl. `is_a`, every use is rewritten to the replacement; the dashboard ERRORs on an obsolete term with axioms (READ-IN-FULL). SSSOM: obsolete terms must not appear in mappings (arXiv 2112.07051, SECTION-READ). | `ClassView::is_a_parent` has no production implementation (closed search, ABSENT); kept fossils keep reading and rendering (A3 is reading-axis only, out of scope). No notion of "resolves but inert". | NEW |
| X5 | IDENTITY | A version id for the source ontology | OWL ontology IRI names a series, versionIRI one version; old versions stay reachable; an import closure must not hold two versions of one series (OWL 2 §3.1-3.4, SECTION-READ). SSSOM records subject/object source version per mapping set (SECTION-READ). | OGIT has no version IRI (`__OGIT_VERSION__` placeholder; `versioning/version.info` id + placeholder + git ref). OgitTables loads a floating checkout; its determinism test runs two loads of ONE checkout (`tests/ogit_tables.rs:196-203`, VERIFIED). | NEW |
| X6 | IDENTITY | Writer namespaces for independent minters (the two-writer question) | SNOMED: meaning-free namespace id per authorised issuer inside the identifier, so writers mint in disjoint spaces (SECTION-READ). SAP: customers append to SAP-owned fixed-value ranges in place (SECTION-READ). Opposite answers, both real. | C.9.3 makes appid ≡ concept byte, so u16 `0x0101` = concept `project` AND OpenProject's own appid (two writers on one address); OSINT `0x0701` concept→appid precedent (VERIFIED). | PARTIAL (domain byte partitions; the low byte is shared by two writers) — raised to the operator, not decided |
| X7 | IDENTITY | Address and ordinal are different TYPES | PostgreSQL: identity = OID, order = separate `enumsortorder` float, so position is not identity (SECTION-READ). | `ClassId` (`class_view.rs:54`) and `EntityTypeId` (`ontology.rs:81`) are both bare `pub type = u16`; `EntityTypeId` has two assignment rules (positional `idx+1` in `ontology.rs:75-91` / `ontology_dto.rs:116-123`, and append-order in the registry); only OgitTables uses newtypes (`TypeId/AttrId/VerbId`, which are ordinals) (VERIFIED). | PARTIAL |
| X8 | LEXICAL | Language-tagged labels on the concept: one preferred + n alternates per language | SKOS: labels are BCP-47-tagged literals; ≤1 `prefLabel` per tag per concept (S14); pref/alt/hidden pairwise disjoint (S13); `hiddenLabel` search-only; a concept may lack a prefLabel (READ §5.1-5.6). Wikibase fingerprint per language: ≤1 label, ≤1 description, n aliases (SECTION-READ). Odoo 16+: translatable field = jsonb `{lang: value}` on the row, en_US base, open language set, one value per language (SECTION-READ, LGPL-3). SAP text table keyed by check-table key + LANG (SECTION-READ, proprietary pattern only). SNOMED: descriptions are separate components with ids; language refsets mark preferred/acceptable per language/dialect/context (SECTION-READ). Gillis-Webber & Keet: three label-layer variants; a language that splits a concept (Fleuve/Rivière) makes distinct concepts, not labels (arXiv 2210.02807, SECTION-READ). | `ontology::Label{key,en,de}` is one closed EN/DE pair per whole ONTOLOGY, no per-concept label (`ontology.rs:28-60`, VERIFIED); `FieldRef.label` English only (`ontology.rs:463-478`); `ClassView::field_label` only, no class-label method (`class_view.rs:1066-1083`, closed search); `LabelDTO` has no language (`ogar_codebook.rs:812-840`); OGAR per-field `label: Option<String>` + `translate: Option<bool>` + `EnumSource::Static (key,label)` all untagged (`ogar-vocab lib.rs:884-889,963-981`). OGIT: one untagged `rdfs:label`; "Stundenzettel" only inside `dcterms:description` (`TimeSheet.ttl:10-11`). | PARTIAL (field/ontology labels exist; no per-concept, per-language label anywhere) |
| X9 | LEXICAL | Fallback across languages happens at LOOKUP, never by storing copies | SKOS §5.6.5: fallback through BCP-47 matching at lookup (READ). Odoo: read falls back from context language to en_US base (SECTION-READ). | `Label::en_only` writes the key into the `de` slot — a German label nobody gave (`ontology.rs:44-69`, VERIFIED). Draft plan `unified-bridge-consumer-migration-v1.md` §4.4 copies a parent's label into a child `FamilyEntry` (Draft, never built; ABSENT) — one label would name two addresses. | CONFLICTS (with our own constraint, not an anchor): `en_only` and the draft inheritance store copies |
| X10 | LEXICAL | Shared label pool + context-scoped PREFERENCE sets | SNOMED language reference sets: the same descriptions, marked preferred/acceptable per language, dialect or narrower context (e.g. a specialty) (SECTION-READ). | Operator-locked slot-purity rule (le-contract §2, E-V3-FACET-4-PLUS-12): "the ClassView carries labels", never a payload slot (CLAIMED in docs; label carriage not verified in the trait). APP-CLASS-CODEBOOK-LAYOUT §3 puts the label set in the ClassView keyed on the FULL u32 (app/render side, SPEC). `LabelDTO.label` is consumer-local. | PARTIAL — and a TENSION to raise: the ruled label home is the app-side ClassView (u32), while the ruling "languages are labels of one concept" points at the concept (u16). A shared u16-keyed pool that the ClassView SELECTS from (SNOMED shape) satisfies both; nobody has ruled it. |
| X11 | LEXICAL | A label never carries a relation: scoped synonyms (narrow/broad) are mappings, not labels | OBO/KGCL synonym scope exact/narrow/broad/related: only the exact one is a label in the SKOS sense; narrow/broad synonyms are sub/supertype relations riding on a lexical row (arXiv 2409.13906, READ-IN-FULL). SKOS-XL makes a label a resource when it needs its own metadata (provenance, "DTO label") (SECTION-READ). | `WOA_ALIASES` collapses 9 subtypes (Quote/Angebot/Order/Auftrag/Invoice/Rechnung/CreditNote/Gutschrift/RecurringInvoice) onto COMMERCIAL_DOCUMENT; `canonical_concept` mixes source-class ids, German words, spelling/plural variants and collapsed subtypes with a lexical fallback that can produce a non-codebook string (`ogar-vocab lib.rs:2898-2950,3258-3276`, VERIFIED). | NEW (no relation kind on any name row) |
| X12 | LEXICAL | External per-language lexicon that denotes the concept by reference | OntoLex-Lemon: one lexicon per language outside the ontology; a sense has exactly one reference; cost: more files, sync, per-language grammar at render (arXiv 2210.02807 §3.2, SECTION-READ; W3C spec not read). | none | NEW |
| X13 | LEXICAL | Per-language labels go absent and stale independently | Wikidata: tail entities lack most languages; ~20% of sampled names judged incorrect; languages drift independently (arXiv 2311.15781, SECTION-READ). | Closed `Label{en,de}` cannot represent an absent or stale language label (VERIFIED). | NEW (design constraint) |
| X14 | LEXICAL | Name lookup returns candidates scoped by source, not one address | SKOS has no cross-concept label uniqueness; Wikidata enforces uniqueness only on label+description (items) (SECTION-READ). | Global `canonical_concept("User")` → project_actor vs port-scoped `OsmPort` "User" → OSM_USER; `canonical_concept_in_domain` gates between them (`ogar-vocab`, VERIFIED). | PARTIAL (scoping exists for routing; the brief's lexical constraint is stricter than practice) |
| X15 | LEXICAL | A locale parameter on the projection | (outside: SKOS lookup-time fallback, X9) | `OntologyDto::project(…, locale)` exists, localises only the ontology name (`ontology_dto.rs:80-112`, VERIFIED). | ALREADY-HAVE (the plumbing) / PARTIAL (no per-entity reach) |
| X16 | LEXICAL | An ingest path that keeps tagged labels | (SKOS/OWL label literals are plain literals with tags) | OGAR TTL reader keeps the FIRST `rdfs:label` only, strips `@lang`, and `into_class` DROPS label and description from `Class` (`ogar-from-schema ttl.rs:85-135`, `lib.rs:169-210`, VERIFIED). lance-graph `OwlHydrator` never interns literals, so tagged labels are dropped at hydration (`hydrators/owl.rs:185-245`, VERIFIED). | PARTIAL (both ingest paths exist; both drop the label) |
| X17 | MAPPING | Typed, one-direction, provenance-bearing mapping rows | SSSOM: subject_id, predicate_id (SKOS exact/close/broad/narrow/related, OWL equivalentClass/subClassOf), object_id, match_type, author/creator/reviewer, source versions, versioned mapping set; walking untyped xrefs is unsound (OxO limb→extremities) (arXiv 2112.07051, SECTION-READ). UMLS: each source string is an atom with its source (SAB) and source code (CODE/SCUI/SAUI), grouped under one CUI; closed REL kinds SY/RB/RN/RO/RQ/XR (SECTION-READ). Odoo `ir.model.data`: (module,name) → (model,res_id), many-to-one (SECTION-READ, LGPL-3). OGIT `ogit:original` = "less strict than owl:sameAs", 369 uses (361 OSLC), no kind (READ-IN-FULL). | `PortSpec::aliases()`: 188 `(public_name,u16)` rows across 11 ports, exact-string scan, no relation kind; "TimeSheet" in no table, so `WoaPort::class_id("TimeSheet") == None` (`ports.rs:51-131,358-397`, VERIFIED). `ogar-adapter` canonical→target string map, map direction only (VERIFIED). `odoo_alignment::resolve_odoo` = one `owl:equivalentClass` pivot, None when unmapped (VERIFIED). `LabelDTO.label` holds source-class names ("account.analytic.line"). | PARTIAL (mapping tables exist; none states a kind, a version or an owner) |
| X18 | MAPPING | Split the alias column by relation kind | Combine X11 + X17: a word in a language → LEXICAL label; a source-class id → exact/close match; a subtype → broadMatch (or is_a if a concept). | `WOA_ALIASES` 25 rows → 6 ids: 10 German words, 15 English names, 1 other app's class (TimeEntry), 9 collapsed subtypes (VERIFIED). | NEW |
| X19 | MAPPING | Keep the mapping graph out of the concept hierarchy | SKOS S41: broadMatch ⊂ broader, so a match asserted in the same graph as is_a entails hierarchy (WoA:Quote broadMatch 0x0202 would pull a source class into is_a); §10.6.1 calls mappings a provenance convenience, "no substitute for provenance mechanisms" (READ). | `lance-graph-ontology` SKOS hydrator loads exact/close/broad/narrow/relatedMatch + pref/alt/hiddenLabel only as edge-type IRIs into its own G slot; no code asserts a match from a source class to a codebook u16 (closed search, `hydrators/skos.rs:1-113`, VERIFIED). | NEW (constraint for any mapping store) |
| X20 | MAPPING | An unmapped source entity carries None, never a mint | (UMLS: every atom sits under some CUI — counter-model) | `resolve_odoo` None test-pinned for `stock.move` = "author an alignment axiom, never mint" (Option B LOCKED); `WoaPort::class_id("TimeSheet")` None; plug/reading/ClassView are u16-keyed, so a CURIE-only entity can be mapped and labelled but not plugged, read or rendered (VERIFIED). | ALREADY-HAVE (None semantics) |
| X21 | MAPPING | Mapping lifetime follows the SOURCE release, the address does not | Odoo: an external id missing from updated module data is deleted with its record (not a never-delete address) (SECTION-READ). SSSOM pins subject/object source version (SECTION-READ). | Alias tables are compile-time consts with no source version; OGIT has no version IRI (X5) (VERIFIED). | NEW |
| X22 | REPRESENTATION | Dictionary codes / ordinals are per-scope positions, never addresses | Arrow: dictionary id per stream/file; a non-delta batch replaces and renumbers (spec example moves 'C' 2→1) (SECTION-READ, Apache-2.0). Lance 13: dictionary chosen and built per page (source read, Apache-2.0). DuckDB ENUM codes = definition order (MIT). Monotone MPHF returns rank, which shifts on a smaller insert (arXiv 2506.06536, SECTION-READ). SAP DD07T keys fixed-value TEXT by VALPOS, a position (SECONDHAND). | OgitTables `TypeId` = row index after sorting by `(domain_of(curie), curie)`; any earlier-sorting insert (a domain before WorkOrder, the SGO core pinned first, a WorkOrder name before TimeSheet, or a change in accepted files) shifts TimeSheet's id; the test pins no TypeId (`ogit.rs:201-305,559-608`, `tests/ogit_tables.rs:14-62`, VERIFIED). `LabelCodebook` token ids are corpus-sort ordinals (`soa_bake/label_codebook.rs:180-197`). `IdentityCodebook` ordinals (ISS-IDENTITY-CODEBOOK-ORDINAL-STABILITY OPEN). Positional `entity_type_id`/`from_ontology idx+1` still live, flagged "v1 path to gate" in E-MINT-TRACE-1, never gated; `OwlHydrator` first-seen interning depends on file order. Rule already ruled: ClassView doc forbids a stored ordinal as a location (`class_view.rs:1226-1228`); E-W2 DECISION (2026-10-10) moves space ordinals out of the classid. | ALREADY-HAVE (the rule) / PARTIAL (positional ids still live, ungated) |
| X23 | REPRESENTATION→IDENTITY | A persisted, append-only CURIE→id log turns an ordinal into a stable surrogate | (pattern: PostgreSQL OID per label; Odoo res_id surrogate + external-id table) | `lance-graph-ontology` registry mints `EntityTypeId` in append order: dedup by OGIT URI, monotone, never reused, gaps allowed, tests pin this; stability rests on the append-only `lance_cache`, not on the tree (`registry.rs:586-607,847-875`, `lance_cache.rs:11-12`, VERIFIED). OgitTables does NOT use it. | ALREADY-HAVE (in lance-graph-ontology) — synergy: OgitTables could key off it |
| X24 | IDENTITY | Lifecycle belongs on the identity row, not in the label | OBO prefixes an obsolete term's label with "obsolete " (fp-019, READ-IN-FULL) — lifecycle leaking into the lexical row. | Our filing rule keeps lifecycle on IDENTITY (brief). | CONFLICTS (outside convention vs our filing rule; not an anchor) |
