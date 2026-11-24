# D-OJC-0..8 — Loading an ontology is the compile step

**Status:** PROPOSED / DOCS ONLY. Nothing here is built or measured.
**Owner input (2026-11-23/24, paraphrased in the session's words):** the
ontology cache is the just-in-time compile moment for behaviour. Loading an
OWL / RDF ontology must produce the executable form directly, fast and
deterministic enough to run business logic (Odoo loaded from RDF, used like
SAP), with ON / WHEN / DO coming from the ontology and intervention and
counterfactual acting as compiler checks.
**Subordinate to:** `d-dt64-7-behaviour-addressing-registration-v1.md`
(the `BindStore` is the method registry; WHEN is what `register` supplies) and
`ontology-method-registry-v1.md` §8–§9. This plan adds no second registry. It
covers what the load produces before anything is registered.

## 1. Overview

A producer (an OWL/TTL file, a Schematron file, OGIT plus Knowledge Items,
SNOMED RF2, the OBO core, WordNet, EU legal ontologies, a code graph) is read
once. The read yields:

1. **addresses**: every IRI resolved once to a class address; after the load
   no lookup goes through a string;
2. **hierarchy as layout**: `rdfs:subClassOf` (and any axis a law declares
   transitive) laid out as a coded path, one byte per level with the depth
   beside it, in path order, so a subtree is one prefix mask or one contiguous
   range; the reference shape is OGAR `ogar-dir-core::Dn128`;
3. **laws, not stored edges**: an axiom a law can read (transitivity,
   inverse, domain, range) becomes a declared law on the class, read over the
   rows; a relation no law expresses stays a relation;
4. **a frozen frame**: the result is immutable; a new ontology version is a
   new frame, as `lance-graph-intake` versions its reads.

What the positions *mean* is decided by the law the class declares
(`entries/2026-11-23-stride-positions-carry-no-names.md`), never by the bytes.

Registration (ON / WHEN / DO into the `BindStore`) then consumes this frame.
Checking a proposed action is two existing reads:

- an intervention on the data: does the action's ON class and WHEN hold
  under the closure (`ogar-elk` `Closure::entails`);
- an intervention on the ontology: is adding these axioms sound
  (`ogar-elk` `Closure::merge`), evaluated without committing.

## 2. Checklist

### Phase A — tests first (no implementation)

- [ ] A1. Oracle fixture: a small OWL file with a 6-level `subClassOf` chain,
      a sibling branch, an `owl:TransitiveProperty`, an `owl:inverseOf` pair
      and a `rdfs:domain`/`rdfs:range`. Oracle: a naive fixpoint closure in the
      test, compared answer by answer.
- [ ] A2. Falsifier for addresses: a lookup after the load allocates nothing
      and takes no lock (counting allocator; `RwLock` absent from the read
      path type).
- [ ] A3. Falsifier for layout: "every class under C" equals the oracle and is
      read as one prefix mask plus the secondary-parent edges (§8); a 257th
      child under one parent is refused, not hashed or truncated; a diamond
      (C subclass of incomparable A and B) is found under both A and B.
- [ ] A4. Falsifier for determinism: two loads of the same file give
      byte-identical frames; reordering the triples in the file gives the same
      frame.
- [ ] A5. Silence twins: an axiom no law reads stays a relation (no law
      fabricated); an unsound merge is refused and the frame is unchanged.

### Phase B — the compile path in `lance-graph-ontology`

- [ ] B1. Frozen positional dictionary: IRI → address once at load;
      reads by address borrow the row (`&MappingRow`), no clone; the
      `entity_type → NiblePath` maps become direct tables indexed by the
      `u16`.
- [ ] B2. Hierarchy layout per the chosen DN type (decision 1).
- [ ] B3. Axiom → law for the four axioms in A1.
- [ ] B4. Frame and version: one frame per load; the Lance cache
      (`lance-cache`) persists the frame, it is never read on a lookup.

### Phase C — producers through the same path

- [ ] C1. DOLCE / FIBO / schema.org (already hydrated) re-run through B; parity
      with today's registry rows.
- [ ] C2. Schematron (EN 16931 / XRechnung): rules become WHEN on their classes.
- [ ] C3. OGIT Automation: `ActionApplicability` (ON), `Trigger` (WHEN),
      `KnowledgeItem` / `ActionHandler` (DO). Knowledge Item bodies remain
      blocked as in D-OMR-5 (no reader, no samples).
- [ ] C4. OBO / SNOMED through OGAR `ogar-obo` (baked rows) and `ogar-elk`
      (entailment); procedure `method` → DO, site / substance → ON; WHEN from a
      declared rule set (CPIC is the existing one in MedCare-rs).
- [ ] C5. A code graph (Kythe facts / ruff SPO): `childof` and `extends` /
      `overrides` as hierarchy laws; `calls` and `ref` stay relations.
- [ ] C6. WordNet: hypernymy / troponymy, meronymy; morphology as a declared
      law over lemma → form; a surface form carries its set of senses (feeds
      deepnsm-v2 `parse_readings`).
- [ ] C7. EU law: ELI / Akoma Ntoso structure as the hierarchy; ODRL /
      LegalRuleML norms as WHEN; per-member-state transposition as separate
      worlds (precedent: `mistaber`).

### Phase D — cost

- [ ] D1. Load time per triple (µs, whole batch) and subtree read (ns) on the
      C1 ontologies and on a large one (SNOMED or the OBO core). Reported with
      the machine and target-cpu; no ms figure is a result.

## 3. Deliverables

| D-id | Deliverable | Gate |
|---|---|---|
| D-OJC-0 | This plan; decisions in §4 ruled | owner |
| D-OJC-1 | Phase A tests, red against today's registry | each falsifier fails for the reason it names |
| D-OJC-2 | B1 frozen positional dictionary | A2 green; disable run red |
| D-OJC-3 | B2 hierarchy layout | A3 green; disable run red |
| D-OJC-4 | B3 axiom → law | A1, A5 green against the oracle |
| D-OJC-5 | B4 frame and version | A4 green |
| D-OJC-6 | C1–C2 producers | parity with current rows; Schematron WHEN fires on a failing invoice and stays silent on a valid one |
| D-OJC-7 | C3–C7 producers, each its own PR | per producer: oracle parity and a refusal named for what it cannot express |
| D-OJC-8 | D1 cost | numbers recorded with method |

## 4. Decisions needed

1. **Where the DN type lives.** `Dn128` is in OGAR `ogar-dir-core` (no
   dependencies); `lance-graph-ontology` and `lance-graph-callcenter` are
   workspace members, and the workspace stays free of OGAR crates. Options:
   move it to `lance-graph-contract` with `ogar-dir-core` re-exporting it;
   depend on `ogar-dir-core` by git; or use the contract's `NiblePath`
   (16 × 4-bit, 16 children per level).
2. **Which axioms become laws in B3** beyond the four in A1.
3. **Callcenter `dn_path.rs`** (FNV-hashed segments folded to a 1-byte scent):
   replace by the decision-1 type in this plan, or separately.
4. **Name-keyed lookups** (`resolve(bridge_id, public_name)`): keep as the
   entry-time resolver only, or retire with the bridges
   (`CONSUMER-BRIDGE-DEPRECATION.md`).

## 5. Details

- **Prior art read:** `lance-graph-ontology` (`registry.rs`: string-keyed
  `HashMap`s, `resolve` allocates two `String`s per call, `row_for_uri`
  clones, every read takes an `RwLock`; loads Turtle via `oxttl` and RDF/XML via
  `oxrdfxml`; `hydrators/` for DOLCE, FIBO, schema.org, SKOS, PROV-O, QUDT,
  OWL-Time, XSD, Schematron, ZUGFeRD, SKR/DATEV, Odoo); OGAR `ogar-dir-core`
  (`Dn128`: 16 levels, one byte per level, depth beside it, refuses a 257th
  child, `subtree_mask()`); OGAR `ogar-elk` (`entails`, `merge`); OGAR
  `ogar-obo` (OBO core baked to 512-byte `NodeRow`s, EL subset, RF2 reader);
  OGIT `NTO/Automation/entities/` (ActionApplicability, Trigger,
  KnowledgeItem, ActionHandler, ActionCapability, GraphRule, Intent).
- **Not verified:** load speed of anything; whether `ogar-elk`'s closure is
  fast enough to answer WHEN per call or must be read from the baked frame;
  SNOMED's OWL refset against `ogar-obo`'s EL subset; WordNet, ODRL,
  LegalRuleML, ELI and Akoma Ntoso were not read in this session.
- **Limit:** only what a source states as structure or rule compiles this
  way. Arithmetic in DO (tax amounts, dosage, depreciation) needs declared
  laws beyond `lane_law`'s affine ops or a written `ActionDef`; how much of a
  business system that leaves is the measurement this plan should produce.

## 6. Amendment (2026-11-24): lazy loading, the ontology as intermediary IR

**The pattern, stated once.** A loaded ontology is an intermediary IR between
its sources (OWL / RDF files, Schematron, OGIT Knowledge Items, SNOMED, OBO,
WordNet, EU law, a code graph) and execution: the producers lower into it, and
registration and reads consume it. The ontology cache is where this IR is
compiled just in time. Compiling it lazily is the general case; compiling a
whole ontology at once is the special case where every class is referenced.

**Why lazy compilation is sound here.** A class's compiled form (its address,
its hierarchy code path, its laws) depends only on its own ancestor path and its
own axioms, not on the rest of the graph. So a class can be compiled on first
reference, together with any ancestors not yet compiled, and frozen from then
on. What is never referenced is never compiled.

**Wikidata is the large case.** The ~115M-entity load is not run whole; classes
are compiled as they are reached. This plan does not design the runtime for
that. It already exists as a plan:
`wikidata-lazy-spine-hydration-v1.md` (D-LWS-*, the `NiblePath`-keyed tiered
hydration manager, I/P/B frames over Lance versioning), with its design in
`.claude/knowledge/agnostic-lazy-world-spine.md` and
`.claude/knowledge/delta-card-addressing-integration-map.md`. What this section
adds is the compile view of it, as three constraints any lazy producer must
meet:

1. **Codes are fixed per snapshot, never by load order.** A child's code under
   its parent must come from a rule fixed for the snapshot (for example derived
   from its identifier), not from the order in which classes were first asked
   for. Otherwise the same class gets a different address depending on query
   history, and the determinism falsifier A4 fails. The rule must also give
   **distinct codes to siblings**: two children of one parent never share a
   code, so they never share a hierarchy address. A rule that would collide is
   refused for that snapshot, not resolved by arrival order.
2. **Overflow escapes, it does not hash or truncate.** More than 256 children
   under one parent, or a path deeper than 16 levels, becomes an explicit
   reference to the registry (OGAR: depth beyond the native levels is "the
   hierarchy's job (registry resolve + ref-escape)"), the same refusal rule as
   `Dn128::from_ou_hhtl`.
3. **A lazily compiled class equals the eagerly compiled one.** Compiling class
   C on first reference must give the same frame entry as compiling the whole
   snapshot and reading C.

**Deliverable added.**

| D-id | Deliverable | Gate |
|---|---|---|
| D-OJC-9 | Lazy compile of one class with its ancestors, against the D-LWS manager | constraint 3: lazy entry equals eager entry for every class of a fixture, in every reference order tried; constraint 1: two reference orders give byte-identical entries; constraint 2: the 257th child yields a registry reference, not a code; sibling collision: two siblings whose rule would give the same code are refused, and no two siblings in the fixture share a code or address |

**Not verified:** the D-LWS plan and its two knowledge docs were read for scope
only (the plan's header and sections), not in full; no lazy load exists to
measure.

## 7. Amendment (2026-11-24): dimension tables and CSR instead of foreign keys

**The compiled layout.** The compile step (§1) lays the frozen frame out in two
forms that DR binds and every reader borrows:

- **Dimension table:** per class address, its attributes at fixed positions,
  read as `dim[addr].field`. O(1) by address, no join, no foreign key. It is the
  class view compiled into position order (`classid-on-do-call-shape.md` §0:
  "There is no foreign key. A relation reached through classid and ontology is a
  class view.").
- **CSR per relation:** `offsets[addr] .. offsets[addr + 1]` into a `targets`
  lane. Because classes are laid out in hierarchy order (§1 item 2), a subtree
  is one contiguous run of addresses and its relations one contiguous run of
  `targets`.
- **Lazy, immutable binding:** a class's dimension row and CSR block are
  compiled on first reference (§6) and frozen in the frame. DR binds a field
  mask to those positions and reads in place.

**What replaces the foreign key.**

| Star schema | Compiled ontology |
|---|---|
| a fact row stores `customer_id` | a fact row stores the class address: the coordinate, not a pointer to resolve |
| join through the dimension on FK = PK | positional read `dim[addr]`, no hash, no match |
| many-to-many through a bridge table | the CSR block of the address |
| a dimension change rewrites the table | a new frame; rows stay readable under the frame they were written in |

**Reasoners over procedural memory.** With the ontology cache as procedural
memory (the compiled ON / WHEN / DO), Datalog-family engines become readers of
the same frame instead of owners of copies of it:

| Engine | Adds |
|---|---|
| Datafrog | WHEN derived by semi-naive fixpoint over entailment chains |
| Ascent | monotone lattice aggregates per subtree (lowest cost, earliest deadline) |
| egglog | equivalence of DO sequences, the cheapest equivalent chosen before it runs |
| Scallop | provenance and confidence carried along each derivation |
| XLOG | the in-house witnessed reasoning compiler and cube, already reading lanes in place |

A sorted-by-key relation is CSR with its offsets implicit, so a semi-naive step
over the compiled CSR is an aligned pass over shared coordinates, not a copy
into an engine's tuple store. Where their semantics overlap, the engines must
agree on the same frame; XLOG's D-XLOG-KNOWLEDGE-G1 Datafrog comparator is the
first such gate.

**Boundary.** The dimension and CSR lanes are the single canonical copy of the
compiled IR, owned by the frame; every reader borrows them. An engine that
copies them into its own storage brings back the tuple problem and is used
only as an oracle, not as the execution path.

**Deliverables added.**

| D-id | Deliverable | Gate |
|---|---|---|
| D-OJC-10 | Dimension table and CSR lanes in the compiled frame; DR binds and reads them in place | a subtree's relations are one contiguous `targets` run; a read allocates nothing; equal to the oracle closure of A1 |
| D-OJC-11 | One external engine (Datafrog first) reading the frame's CSR without copying, compared with XLOG | the two agree on every query of the A1 fixture; a disable run that copies into tuples is detected by the allocation falsifier |

**Not verified:** Datafrog's, Ascent's, egglog's and Scallop's internal
relation layouts were not read in this session; "sorted relation = CSR with
implicit offsets" is stated from memory and is the first thing D-OJC-11 must
check.

## 8. Amendment (2026-11-24): multiple inheritance and dense ordinals

Two gaps in §1, §6 and §7, raised in review on #2163.

**Multiple inheritance.** OWL allows a class C to subclass both A and B where
neither is an ancestor of the other. One coded path cannot have both A's and
B's paths as prefixes, so a prefix read alone misses C under one of them.

- Each class gets one **primary parent**, chosen by a rule fixed for the
  snapshot (the same kind of rule as §6 constraint 1, never arrival order). Its
  coded path runs through that parent.
- Every other `subClassOf` parent is a **secondary edge**, stored as a CSR
  relation (§7) from the class to each secondary parent, and the reverse
  direction from each parent to its secondary children.
- A subtree read under A is the least set S with: every class under A's prefix
  is in S; every secondary child of a class in S is in S; every class under the
  prefix of a class in S is in S. The two steps repeat **to a fixed point**:
  if D is a primary child of A and C a secondary child of D, C and C's own
  subtree are under A. With a frozen frame this set is computed once per class
  at compile time and stored as one mask, so the read stays one mask.
- **Secondary edges are indexed for the whole snapshot at ingest**, in the same
  ordering pass that assigns the dense ordinals below: one CSR of
  `parent ordinal → secondary children` built from identifiers and
  `subClassOf` statements alone, without compiling any class. Compiling A
  lazily (§6) therefore finds A's secondary descendants in that index even
  when none of them has been compiled; computing A's subtree mask needs only
  ordinals and that index, not the descendants' compiled entries.

**Dense ordinals.** `dim[addr]` and `offsets[addr]` (§7) index arrays, so `addr`
must be a dense ordinal, distinct from the coded path:

- The ordinal is the class's rank in a depth-first walk of the primary-parent
  tree, children visited in code order. It is fixed per snapshot and
  independent of load and reference order, and a subtree under the primary tree
  is one contiguous ordinal run.
- The ordinals are assigned from the snapshot's class list once, at ingest of
  the snapshot (an ordering pass over identifiers and primary parents, without
  compiling any class); lazy compilation then fills `dim` rows and CSR blocks
  at those fixed ordinals.
- A class whose path escaped to a registry reference (§6 constraint 2) still
  has an ordinal, a `dim` row and a CSR offset. Only its coded path is replaced
  by the reference; its place in the arrays is the same as for any class.

**Gate changes.** D-OJC-10 is checked against these ordinals: a primary
subtree's ordinals form one run, every class including registry-reference ones
has exactly one `dim` row, the diamond of A3 is found under both parents, and a chained case (D a primary child of A, C a secondary child of D, E a primary child of C) returns C and E under A, with A compiled first and C, D, E not yet compiled.

## 9. Amendment (2026-11-24): the frame as retrieval source (RAG / RIG)

The compiled frame (§7: dimension tables and CSR, bound through DR) can serve
as the retrieval source for a model, in place of nearest-neighbour search over
text chunks:

- **Exact reads.** A retrieval-interleaved query from the model resolves to a
  class address; the answer is `dim[addr]` or a CSR run: a fact with its frame
  version and the axiom it came from, not an approximate passage.
- **A pinned DR version works like a token cache.** With the frame version
  pinned for a conversation, the same query returns the same facts; the address
  is the cache key. Across conversations on one frame the retrieved part does
  not drift.
- **What comes back can be executable.** A retrieval can return a DO with its
  WHEN, checked as an intervention or counterfactual (§1) before the model acts.
- **Names only at entry.** The step from the model's words to an address is the
  one place errors enter. An ambiguous name is refused or returned as a
  candidate set, never guessed.

Vector similarity stays available as a way to *find* candidate addresses; it is
navigation, never the retrieved fact.

## 10. Amendment (2026-11-24): lance-context wired through DR, data as context as config

**Goal (owner, 2026-11-24, paraphrased):** use lance-context without taking on
its development and without bypassing the architecture: context data is
config, mapped through DR as a CSR / dimension-table-like codebook table
object, versioned in the Quack `BindStore`, duplicating nothing we have.

**What lance-context is (read at `AdaWorldAPI/lance-context` `main`, fork of
`lance-format/lance-context`):** a storage engine for agent memory and RL
rollouts on Lance. Relevant surfaces:

- `lance-context-api::schema_spec::SchemaSpec` / `ColumnSpec`: a schema
  declared as a value, persisted in the dataset's schema metadata, refused on
  reopen with a different declaration. Depends only on `arrow-schema` and
  `serde`. An `id: Utf8` column is mandatory (the LSM merge key).
- `lance-context-core::generic_store::GenericStore`: a store over a
  caller-declared `SchemaSpec`; all storage behaviour (WAL, merge, compaction,
  versioning, `checkout(version)`) inherited from its `StorageBase`.
- `TraceRecord`: deduplicated conversation turns (`session_id`, ordered
  `turn_id`, role, content).

**The division of labour.**

| Concern | Owner |
|---|---|
| Durable context rows: turns, blobs, embeddings, rollouts; append, WAL, compaction, snapshots | lance-context, unchanged |
| Which class a context row belongs to; its place in the hierarchy; relations between context and ontology classes | the compiled frame (§7), bound through DR |
| The mapping itself | one codebook table object: `context row id → class ordinal` (dimension side) and `class ordinal → context rows` (CSR side) |
| Versioning of that mapping | the `BindStore`, as a pair `(frame version, lance-context snapshot version)` |
| Queries | Quack over the frame's masks; lance-context is read by id for the rows a mask selected |

**Data as context as config.** The binding is declared, not coded: a
`SchemaSpec` names the columns, and a declaration names which column carries
the class reference. Registering a context store is then a `register` of that
declaration, the same call shape as any other class
(`classid-on-do-call-shape.md`). No per-store adapter code.

**What is not duplicated.**

- No second copy of context rows: the codebook holds ids and ordinals, never
  content. Content is read from lance-context by id after the mask is known
  ("filter IDs first, hydrate raw values last").
- No second registry: the `BindStore` is the registry (D-DT64-7).
- No second versioning scheme: Lance versions on the lance-context side, the
  frame version on ours, joined as one pair.
- lance-context's `Relationship { target_id: String, relation: String, weight }`
  on `ContextRecord` is an ingestion view. It is resolved once into the CSR at
  registration and not consulted on the read path
  (`d-rmap-2-context-tile-algebra-probe-v1.md` already records it as "NOT the
  hot numeric RelationMap").

**The boundary that keeps this within the architecture.** lance-context
vendors and patches Lance (`vendor/lance`, `vendor/lance-io`,
`vendor/lance-core`, `vendor/lance-encoding`, wired by `[patch.crates-io]`) and
depends on DataFusion 54. lance-graph's rule is that the Lance family is
consumed from crates.io upstream, never from a fork
(`E-LANCE-IS-UPSTREAM-AUTHORITATIVE-1`), and DataFusion is in its grace period.
So:

- lance-context is **not** a dependency of any lance-graph workspace crate.
- The wiring lives in a crate outside the workspace (the shape of
  `lance-graph-dir-sim` and `lance-graph-ogar`), or runs lance-context as its
  own process behind its existing server.
- The only type that may cross into the workspace is `SchemaSpec` /
  `ColumnSpec` from `lance-context-api`, if that crate proves free of the
  patched Lance (its doc says it depends only on `arrow-schema` and `serde`;
  to be verified by `cargo tree`).

**Deliverables added.**

| D-id | Deliverable | Gate |
|---|---|---|
| D-OJC-12 | Retrieval over the frame: a query resolved to an address returns the `dim` row or CSR run with its frame version | same query on the same pinned version returns byte-identical facts; an ambiguous name returns its candidate set; a version bump changes the answer only where the frame changed |
| D-OJC-13 | lance-context codebook table: declared binding of a `SchemaSpec` store to class ordinals, versioned in the `BindStore` | the codebook holds no content column; a mask over the frame selects ids and lance-context returns exactly those rows; reopening under a different declaration is refused |
| D-OJC-14 | Dependency fence: lance-context stays outside the workspace graph | `cargo tree` of every workspace member shows no `lance-context-core` and no patched Lance; a disable run that adds the dependency fails the fence |

**Decisions needed (added to §4).**

5. In-process outside the workspace, or lance-context as a separate process
   behind its server.
6. Whether `lance-context-api` may be a workspace dependency for `SchemaSpec`,
   or the declaration is mirrored as our own type with a parity test.
7. Which lance-context stores are bound first (`TraceRecord` turns are the
   smallest).

**Not verified:** lance-context was read from its README, `Cargo.toml`,
`schema_spec.rs` and `generic_store.rs` headers and signatures only; nothing
was built. Whether `GenericStore::get_many` is a zero-copy read is not known
(`d-rmap-2` flags the same doubt). The fork's vendored Lance is 14.0.2, the
same line as lance-graph's `14.*`; whether the patches change on-disk format
was not checked.

### 10a. A context is a bag over the id lane

Read at `AdaWorldAPI/lance-context` `c9a638b`. Every lance-context store
carries a mandatory `id: Utf8` column (its LSM merge key). That column is the
key lane:

- **Key lane.** `identity[index].key` is the store's `id`, through an exact
  dictionary `id → ordinal` built once per snapshot. Exact, never a hash: a
  collision would return the wrong row.
- **A context is a bag.** "This session", "the turns relevant to this question"
  is a selection of ordinals over the key lane: one mask under one frame, lent
  in place the way `LaneBinding::borrow_all` lends (there is no separate bag
  type; the bag is what `borrow_all` lends). An id lookup returns a bag; row
  content is fetched from Lance last, by row address, for the positions in the
  mask.
- **Version.** A bag is valid under one frame, and the frame names the pair
  (ontology frame version, lance-context snapshot). A bag presented under
  another frame is refused, the witness rule of `borrow_all_at`.
- **Two forms of one table in DR.**
  - Dimension: `ordinal → (row address, class ordinal, session ordinal)`.
  - CSR: `session → its turns`, `class → its context rows`, each one run.

Missing today: a `LaneSource` over lance-context's `id` column; and row
addresses that survive compaction (rebuild the dimension per snapshot, or use
Lance stable row ids if the vendored Lance supports them; not checked).

### 10b. Migration path: DataFusion and Quack side by side as oracle

> **lance-context half (2026-11-24):** the seam and the comparing wrapper are
> proposed in `AdaWorldAPI/lance-context` PR #1,
> `docs/design/id-lookup-seam-and-quack-oracle.md`
> (https://github.com/AdaWorldAPI/lance-context/pull/1). The call-site line
> numbers below were read at `c9a638b`; that document re-reads them at
> `5adf73f`, where `annotation_staging.rs` is 712 / 777 and `document_store.rs`
> 204 / 229 / 246. It also finds `enable_stable_row_ids` in use in that crate's
> tests, which bears on the open point in §10a about row addresses surviving
> compaction.

**Where DataFusion is used in lance-context** (25 lines, two crates):

| Role | Where | Plan |
|---|---|---|
| Lookup by id: `id = x`, `id IN (…)` | `generic_store.rs` (373, 423, 636), `rollout_append.rs` (868), `annotation_staging.rs` (710, 756), `document_store.rs` (198, 222) | route through the seam below |
| Caller-supplied filter | `generic_store.rs` (561, 876–888) | stays on DataFusion until an expression → Quack lowering exists |
| Merge-key read (`ScalarValue`) | `store_base.rs` (3211) | not a query; unchanged |
| A query engine: SQL parser, `SessionContext`, `MemTable`, `SortExec`, `GlobalLimitExec` | `rollout_store.rs` `query_sql()` | not called from our side; filter / sort / limit go to Quack, joins to the XLOG cube |
| Ingest materializer | `lance-context-ingest`, optional feature `materializer` | left off |

DataFusion cannot leave the binary while Lance is used: Lance depends on it
and re-exports it (`lance::deps::datafusion`). "No DataFusion" here means none
of our query logic runs on it.

**The seam.** One small trait in lance-context, roughly
`IdLookup: ids → row addresses`, with today's scanner filter as the default
implementation, so behaviour is unchanged. It is additive and could go
upstream. The Quack implementation lives in the wiring crate outside the
workspace, because lance-context cannot depend on lance-graph crates (§10
boundary).

**Two tables.**

- DataFusion side: the Lance dataset with its scalar index on `id`, as today.
- Quack side: the key lane and dimension table of §10a. An id match gives a
  mask; rows are taken from Lance by address.

**Oracle run (Parallelbetrieb).** Both sides answer every id lookup on the same
pinned dataset version. Phase 1: DataFusion's answer is served, Quack's is
compared, and every difference is recorded with the ids and the version.
Phase 2, after agreement has held over a stated number of lookups and
snapshots: Quack is served and DataFusion is the shadow. Phase 3: the
DataFusion arm is kept only as a test oracle.

**What makes the comparison fair.**

- lance-context reads through its write-ahead log, so recent rows may not be
  in the base dataset. Compare at a pinned version, or build the key lane over
  the unmerged generations too; otherwise the oracle reports false differences.
- Both arms must return the same row set, compared as sets of ids, not as
  ordered row lists.

**Deliverables added.**

| D-id | Deliverable | Gate |
|---|---|---|
| D-OJC-18 | `LaneSource` over a lance-context `id` column; exact `id → ordinal` dictionary and the dimension / CSR forms per snapshot | an id lookup returns a mask with no row fetched; two distinct ids never share an ordinal; a bag under another frame is refused |
| D-OJC-19 | `IdLookup` seam in lance-context with the scanner filter as default | lance-context's own test suite unchanged with the default; the seam is called at every id-lookup site listed above |
| D-OJC-20 | Oracle run: both arms on a pinned version, differences recorded | a planted wrong ordinal is reported (can fire); a clean snapshot reports none (can stay silent); unmerged WAL rows cause no false difference |

**Decisions needed (added to §4).**

8. How long agreement must hold (lookups and snapshots) before phase 2.
9. Whether the `IdLookup` seam is offered upstream to `lance-format/lance-context`
   or kept on the fork.

## 11. Amendment (2026-11-24): a grounded code reviewer for lance-graph

A consumer of §9, §10 and C5, governed by the Claude Teacher protocol of
#2077 / #2078 (LG-COG-001; both open and docs only at the time of writing).
Nothing here adds an executor: Quack / R2IL / mask-RISC stays the only
execution path, and the model's output is a cold, untrusted, structured
hypothesis.

**Parts.**

| Part | What it is | From |
|---|---|---|
| Code graph | lance-graph's own source loaded as an ontology: items as classes, `childof` / `extends` / `overrides` as hierarchy laws, `calls` / `ref` as relations | C5, D-OJC-7 |
| Review memory | past PRs, review threads, board entries and decisions as context rows, bound to code-graph classes by the codebook table | §10, D-OJC-13 |
| Rules | the repo's own laws as WHEN conditions registered on code classes (examples: no tuple materialization; Lance only from upstream; no new DataFusion code; stride positions carry no names) | §1, registration under D-DT64-7 |
| Derivation | a Datalog pass over the frame derives violations, each with the facts and the rule that produced it | §7 |
| Teacher | the model proposes findings and explanations over what the frame returned | #2077 / #2078 |

**The rule that keeps it honest.** Every finding the reviewer publishes must
resolve to addresses in the frame and to a rule or a retrieved fact, at a named
frame version. A finding that does not resolve is dropped, not posted. Judgement
calls the frame cannot derive (is this design right?) are labelled as the
model's hypothesis and are kept apart from derived findings.

**What it may do.** Read, and post review comments. It never approves, merges,
resolves a thread, or pushes. Reward and rollout events grant no authority
(#2077).

**Evaluation.** The four arms of #2077: no context; retrieval; retrieval plus
config; retrieval plus config plus feedback. The oracle is past PRs whose
review findings were later confirmed by a fix commit: a finding counts as found
when the reviewer, run on the pre-fix head, names the same defect at the same
place. Findings with no confirming fix are scored separately as unverified, not
as false. The API cost limit of #2077 applies.

**Deliverables added.**

| D-id | Deliverable | Gate |
|---|---|---|
| D-OJC-15 | lance-graph's code graph and its review history loaded through the compile path and the codebook table | "who calls X" equals an oracle built from the compiler's own output for a fixture crate; a past ruling on a module is returned with its source and version |
| D-OJC-16 | Rule set: a first handful of repo laws as registered WHEN conditions, with derivations | each rule fires on a known past violation and stays silent on the fixed commit |
| D-OJC-17 | Reviewer run under LG-COG-001's four arms against the confirmed-findings oracle | every published finding resolves to frame addresses; an unresolvable finding is dropped (can-fire and can-stay-silent); arm results recorded with cost |

**Depends on:** D-OJC-7 (code graph producer), D-OJC-12, D-OJC-13, and the
owner decisions of #2077 / #2078 (which method and bag references exist to
reuse, where the thin facade lives).

**Not verified:** #2077 and #2078 were read from their PR descriptions only,
not their plan files; no code graph of lance-graph has been loaded; no review
has been run.
