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
      read as one prefix mask; a 257th child under one parent is refused, not
      hashed or truncated.
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
   history, and the determinism falsifier A4 fails.
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
| D-OJC-9 | Lazy compile of one class with its ancestors, against the D-LWS manager | constraint 3: lazy entry equals eager entry for every class of a fixture, in every reference order tried; constraint 1: two reference orders give byte-identical entries; constraint 2: the 257th child yields a registry reference, not a code |

**Not verified:** the D-LWS plan and its two knowledge docs were read for scope
only (the plan's header and sections), not in full; no lazy load exists to
measure.
