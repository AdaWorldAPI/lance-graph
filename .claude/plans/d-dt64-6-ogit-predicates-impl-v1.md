# D-DT64-6 predicates: implementation plan for the OGIT predicate expansion (stages 6-1 to 6-4)

**Date:** 2026-10-28. **Status:** PROPOSED / DOCS ONLY. **Parent:** [D-DT64-6](https://github.com/AdaWorldAPI/lance-graph/blob/main/.claude/plans/d-dt64-6-routing-bindstore-behavior-v1.md) ([#1831](https://github.com/AdaWorldAPI/lance-graph/pull/1831), squash-merged into `main` as `42b4b83`; `32e8d07` is a commit on the PR branch and is not on `main`), §5 stages D-DT64-6-1 to D-DT64-6-4. **Not:** code, test, ABI or merge authorization. Each stage below needs its own owner approval before any code.

This plan turns #1831's one-line stage descriptions into implementable work: file and symbol, the change, the tests that must fail before and pass after, and the acceptance numbers. Measurements were taken on 2026-10-28 against lance-graph `main` `eeaaafb`, OGAR `main` `6b43a48` and AdaWorldAPI/OGIT `master` `b5731a8`. OGIT is read unpinned (#1831 D7); the commit is recorded as a provenance receipt, never as a gate.

Evidence states as in #1831: VERIFIED-IN-CODE (read in full here), MEASURED (command given), CURRENT-CONTRACT, PROPOSED, OPEN.

## 0. What the expansion has to read (MEASURED)

| Measure | Value | Command (OGIT checkout) |
|---|---|---|
| verb TTL files | 523 | `git ls-tree -r --name-only origin/master \| grep -c '/verbs/.*\.ttl$'` |
| … declared `a owl:ObjectProperty; rdfs:subPropertyOf ogit:Verb` | 511 | `git grep -l 'owl:ObjectProperty' origin/master -- '*/verbs/*.ttl' \| wc -l` |
| … declared `a rdfs:Class; rdfs:subClassOf ogit:Verb` (all in `NTO/WorkOrder/verbs/`) | 12 | `git grep -l 'subClassOf ogit:Verb' origin/master -- '*/verbs/*.ttl' \| wc -l` |
| verb files under `SGO/sgo/verbs/` (root `ogit:` namespace) | 183 | `git ls-tree -r --name-only origin/master \| grep -c '^SGO/.*verbs/'` |
| files with `ogit:allowed` | 960, none under `verbs/` | `git grep -l 'ogit:allowed' origin/master -- '*.ttl'` |
| files with `ogit:from-to` | 12 (the WorkOrder `rdfs:Class` verbs) | `git grep -l 'ogit:from-to' origin/master -- '*.ttl'` |
| entity TTL files / with `subClassOf ogit:Entity` | 953 / 931 | `git ls-tree …`, `git grep -il 'subClassOf ogit:Entity'` |
| allowed edges whose verb is root `ogit:` | ≈2,330 of ≈2,932 (79%) | Appendix A script (regex, approximate) |
| allowed edges with src, verb and dst all in `ogit.<Domain>:` | ≈550 (19%) | same script |

Shape of one allowed block (`NTO/Schedule/entities/Calendar.ttl`): the entity is the source; each list cell is a blank node `[ verb target ]`. In the file each cell is on its own line, the first indented with spaces and the second with a tab:

```turtle
  ogit:allowed (
    [ ogit:contains  ogit.Schedule:Event ]
	[ ogit:belongs  ogit.Auth:Account ]
  );
```

Shape of the 12 `from-to` verbs (`NTO/WorkOrder/verbs/Issued.ttl`): `ogit:from-to ( [ ogit:from ogit.WorkOrder:Customer ; ogit:to ogit.WorkOrder:Order ] )`.

## 1. Where the two readers stand (VERIFIED-IN-CODE)

**lance-graph-ontology `ttl_parse.rs`.**
- Parses with oxttl into `RawTriple`s, groups them by subject, and resolves RDF lists with `walk_rdf_list`. Blank-node subjects are skipped as proposals, but their triples stay in `by_subject`, so the `[ verb target ]` cells are available.
- `classify` accepts a verb only as `rdf:type rdfs:Class` + `rdfs:subClassOf ogit:Verb`, so 511 of 523 verbs fall to `SubjectKind::Other`.
- `OgitUri::parse` (`namespace.rs`) requires an `ogit.<Domain>:` prefix, so every root `ogit:` subject is skipped ("root vocabulary terms etc."). That covers the 183 SGO verbs and the SGO core entities such as `ogit:Person`. OGAR's `tests/ontology_cache.rs` pins this as current behaviour.
- `walk_rdf_list` collects only IRI list items. A list of blank nodes, which is what `ogit:allowed` is, yields nothing.
- Verbs become `MappingProposalKind::Edge { LinkSpec { subject_type: "ogit.Node", object_type: "ogit.Node", ManyToMany } }`, one per verb.

**OGAR.**
- `ogar-from-schema/src/ttl.rs` `extract_allowed_block` (a line-oriented walker, not a Turtle parser) fills `EntityDecl.allowed: Vec<(verb, target)>`.
- `ogar-action-handler/src/ogit.rs` `OgitTables` turns these into `Table::Allowed` with `u32` lanes `src`, `verb`, `dst`. Ids are row ordinals, and rejected, duplicate and unresolved entries are reported.
- The SGO core is included.
- OGAR depends on lance-graph-ontology behind the feature `ontology-cache`. `tests/ontology_cache.rs` already compares entity names between the two readers, with four known exceptions.

## 2. Stages

### D-DT64-6-1a — `ttl_parse` reads every OGIT verb and the root namespace (NEW, lance-graph-ontology)

**Change.**
1. `classify`: add the verb arm `rdf:type owl:ObjectProperty` + `rdfs:subPropertyOf ogit:Verb`, keeping the existing `rdfs:Class` arm. Add `OWL_OBJECT_PROPERTY` and `RDFS_SUBPROPERTY_OF` to the constant pool.
2. Root namespace: a root `ogit:<name>` subject that classifies as Entity or Verb is kept, not skipped. Two options:
   - **(a)** `OgitUri` gains a root form whose `namespace()` returns `None` and whose `name()` returns the local name;
   - **(b)** the root names are mapped to a reserved pseudo-namespace `SGO`.

   OPEN: the owner chooses. Option (a) is smaller but changes `OgitUri`'s invariant; option (b) keeps the invariant and needs a namespace-registry row. Either way, `ontology_cache.rs`'s assertion that `ogit:dependsOn` is absent from the cache has to flip in OGAR, in the same reconciliation round.
3. Anything still classified `Other` that has `rdf:type` or `rdfs:subClassOf`/`rdfs:subPropertyOf` pointing at an `ogit:` IRI is reported in a new `unclassified: Vec<(source_uri, subject)>`, never dropped silently.

**Tests (`crates/lance-graph-ontology/tests/`, registered in the crate's integration test layout).**
- `object_property_verb_is_a_verb`: inline TTL for an `owl:ObjectProperty` verb → exactly one Edge proposal. Must fail on today's `main`.
- `rdfs_class_verb_still_a_verb`: `Issued.ttl`-shaped inline TTL → one Edge proposal. Silence twin: passes before and after.
- `root_namespace_verb_kept`: `ogit:contains` → one proposal under the chosen root form. Fails on `main`.
- `datatype_property_is_not_a_verb`: an `owl:DatatypeProperty` stays an Attribute, so the new arm cannot swallow attributes.
- Real-data arm (skips without `OGIT_FORK_PATH`): `hydrate_real_ogit.rs` gains `every_verb_file_yields_one_verb`, asserting `== 523` Edge proposals and `unclassified.is_empty()`. The expected number is measured from the checkout in the test (count `verbs/*.ttl` files), not hard-coded, because OGIT is unpinned.
- Disable run: remove the new arm → `object_property_verb_is_a_verb` and the real-data count go red. Commit before disabling.

**Acceptance.** Every verb file and every entity file under an `entities` directory (953 on `b5731a8`, §7.3), root namespace included, classified on OGIT `master`, with zero unclassified. No change to existing Entity and Attribute proposal counts other than the added root-namespace ones; this is checked by diffing proposal counts per namespace before and after.

### D-DT64-6-1 — two-reader partition reconciliation (OGAR, read-only)

**Where.** OGAR `crates/ogar-action-handler/tests/ontology_cache.rs`, feature `ontology-cache`. It is the only place that links both readers, because the reverse dependency would be a cycle.

**Change.** A new test, `verbs_and_entities_partition_identically`. Both readers load the same `ogit_checkout::root()` in one run, and each prints its partition as accepted / rejected / duplicate / unresolved, per kind (entity, verb). Every set difference is listed with the file and the reason. Known differences live in a named const with one line of reason each, the same pattern as `ONLY_IN_THE_TABLES`. Unexplained differences fail the test.

**Explains.** The 523 / 2,894 count that #1831 §5 cites versus 518 / 2,756 (eaw-5 line 24, OGAR #363) gap. OGIT file counts are identical at `2315167` and `b5731a8`, so the gap is on the reader side. The test records the observed OGIT commit as a provenance line in its output only.

**Gate.** D-DT64-6-1a merged and consumed by OGAR. Until then, the test expects 511 + 183 verbs missing on the lance-graph side and says so.

### D-DT64-6-2 — the `(src, verb, dst)` allowed table in lance-graph-ontology

**Precondition (OPEN, §6).** The change below is a second `ogit:allowed` parser. Merged #1831 makes "Two parsers produce the table" a falsifier of D-DT64-6-2 (§5) and rules out a second parser (§7), so this design fails its parent's acceptance unless the owner first chooses §6 option B **and** amends that guard. Under #1831's default, option A, D-DT64-6-2 is only the standing parity test in OGAR plus a consumer reading `OgitTables::Allowed`, and items 1–3 below do not apply. Under option C neither reader parses `ogit:allowed`. The rest of this stage describes option B only.

**Change (option B only).**
1. Entity pass: in `parse_into_proposals`, for `SubjectKind::Entity`, read the `ogit:allowed` list with a new `walk_rdf_list_nodes` that returns blank-node list items (today's `walk_rdf_list` returns only IRIs). Each cell must have exactly one `(predicate, Iri)` pair, giving `(entity, verb, target)`. A cell with any other shape is reported, not skipped.
2. Verb pass: for the 12 `from-to` verbs, read `ogit:from-to ( [ ogit:from A ; ogit:to B ] )` into `(A, verb, B)`.
3. Output: a new `AllowedEdge { src: String, verb: String, dst: String, source_uri, origin: AllowedOrigin::{EntityAllowed, VerbFromTo} }` collection beside the proposals, as a parse result field, not a new `MappingProposalKind` (#1831 §7: "No second registry"). It holds canonical names, no classids (#1831 D-DT64-6-2: classid-free). Duplicates are counted, not deduplicated silently.
4. The per-verb placeholder `LinkSpec`: merged #1831 decides this. Its §5 row D-DT64-6-2 says the placeholder "is replaced from" the one table, and its §7 says "D-DT64-6-2 replaces the placeholder `LinkSpec` from the one table". This plan does not reopen that. What stays open is which reader owns the one table (§6).
5. Unresolved, resolved across the whole checkout. `parse_ttl_directory_with_provenance` calls `parse_into_proposals` once per file, and OGIT declares entities and verbs in separate files, so a per-file check would report almost every valid reference (the Calendar cells' `ogit:contains` and `ogit.Auth:Account` are declared elsewhere). Resolution is therefore a second pass after the directory walk: collect every declared subject from all files first, then report as `unresolved` each `verb` or `dst` that no file declares, using the same three-way partition OGAR reports.

**Tests.**
- `allowed_block_yields_one_edge_per_cell`: Calendar-shaped inline TTL → exactly 2 edges with the right (src, verb, dst). Fails on `main`.
- `allowed_cell_with_two_pairs_is_reported` and `empty_allowed_list_yields_nothing`: can-fire and can-stay-silent.
- `from_to_verb_yields_edge`: `Issued.ttl` shape → `(Customer, issued, Order)`.
- `undeclared_target_is_unresolved_not_dropped`.
- Real data: the edge count equals the distinct `(src, verb, dst)` count from an independent oracle. Use OGAR's `OgitTables::Allowed` via the 6-1 test, not a second lance-graph helper, so the oracle is independent of the candidate.
- Pair-for-pair parity lives in OGAR's 6-1 test: `allowed_edges_match_pair_for_pair`, keeping it standing rather than run once.
- Disable run: make `walk_rdf_list_nodes` return empty → the inline tests and the parity test go red.

**Acceptance.** Pair-for-pair equality with `OgitTables::Allowed` on OGIT `master`, every residual difference named in the known-differences const, `dropped = 0`.

### D-DT64-6-3 — OGIT verbs as DT64 predicates (catalogue per ClassView frame)

**Conflict to settle first (OPEN).** DT64-1 (`d-dt64-1-predicate-catalogue-source-crosswalk-v1.md`) defines one versioned catalogue of 64 slots with fixed meanings (eight groups of eight). OGIT's 378 used verbs do not fit one catalogue. They are also polymorphic: `generates` has 158 source classes and `contains` 111 targets. #1831's "64-wide blocks" must therefore mean one catalogue per ClassView frame, where `catalogue_id` = OGIT verb catalogue for class C, and not more slots in DT64-1's catalogue.

**Proposed shape (PROPOSED).**
- Each slot is one verb that touches C as source. The DT64-1 descriptor fields are filled from the allowed table:

  | descriptor field | value |
  |---|---|
  | `symbol` | the verb CURIE |
  | `domain_class` | C |
  | `range_class` | the set of targets for (C, verb) |
  | `direction` | out |
  | `authority_source` | `ogit:allowed` + file path |
  | `source_snapshot` | the OGIT commit receipt |
  | `cardinality` | UNKNOWN (OGIT's `ogit:cardinality` is `"many2many"` everywhere, which carries no information) |
  | `observed_or_derived` | observed-schema |
  | `closed_world_scope` | none (schema whitelist, not instance completeness) |

- Ordinals are per frame, never global (#1831 D-DT64-6-3 falsifier).
- Per class, p95 is 10 verbs and the maximum 123; 2 classes exceed 64. Those 2 get a second block in the same frame, with the block index part of the frame key, and are never truncated.
- Inbound verbs (C as target) form a separate direction-in catalogue. They are not mixed into the same slots (DT64-1 question 7: an inverse is explicit, not symmetric).

**Tests (lab, wherever the DT64 lab crate lives; DT64 has no code today).**
- Every `(C, verb)` in the allowed table maps to exactly one slot in C's frame.
- A slot's descriptor round-trips to the source file.
- The two classes above 64 are split, not truncated.
- A global verb id never appears in a descriptor.
- A verb not allowed for C has no slot (can-stay-silent).

**Gate.** D-DT64-6-2 green, plus the DT64-1 owner's confirmation that per-frame catalogues are admissible (DT64-1 says "a production registry may choose fewer than 64 active predicates", but does not address many catalogues).

### D-DT64-6-4 — consumer read-only access through HotPlug

**Change.**
- A consumer declares `HotPlug { consumer, classids, covered }`.
- `CapabilityAuthority::activate` (OGAR, the authority) returns an `Activation` whose read path lends the frame catalogue for the consumer's classids. The read is through `OgitTables::lane` slices, so there is no consumer copy.
- No new capability name is minted without OGAR's mint check, which verifies the expected consumer, coverage in both directions and ids minted once (`hotplug.rs` role 2).

**Tests.**
- A consumer that holds its own copy of the table fails a test. The check is structural: the consumer crate has no `Vec<AllowedEdge>`-shaped field, and the read type is a borrowed slice.
- Activation with an uncovered classid returns `ActivationDrift`.

**Gate.** D-DT64-6-3 green.

## 3. Order and ownership

| Stage | Repo | Owner | Depends on |
|---|---|---|---|
| 6-1a | lance-graph | lance-graph-ontology | none |
| 6-1 | OGAR | OGAR + lance-graph-ontology | 6-1a |
| 6-2 | lance-graph (code), OGAR (parity test) | lance-graph-ontology; OGAR for the test | 6-1 green |
| 6-3 | lance-graph lab | DT64 catalogue owner | 6-2 green; DT64-1 owner ruling |
| 6-4 | OGAR + consumer | consumer owner with OGAR authority | 6-3 green |

## 3a. Update 2026-10-28: AdaWorldAPI/OGIT#11 (open)

OGIT#11 changes inputs this plan reads. §0 stays as measured on OGIT `master` `b5731a8`; re-measure on the branch before any stage starts.

- `ogit:assesses` was undeclared. It becomes `ogit.Healthcare:assesses`, and its target `AnatomicalStructure` is declared. Other undeclared allowed targets in `NTO/Healthcare` and `NTO/WorkOrder` are fixed.
- `tools/import_check.py` (in OGIT#11) reports 0 errors on the branch head `0bf9a80` and writes the verb catalogue (531 verbs: 519 `owl:ObjectProperty`, 12 `rdfs:Class`) and the allowed table (2,965 rows: 2,948 from `ogit:allowed`, 17 from the 12 verbs' `ogit:from-to`). Its counts include verbs declared outside `verbs/` directories, so they differ from §0's file counts. The table is a candidate cross-check for 6-2; it is not a second parser to adopt (§5).
- The SGO layout is unchanged: root `ogit:` files stay in `SGO/sgo` and `SGO/core`. The OGIT validator's directory check reads the canonical path, so they validate only in a checkout whose path contains `/ogit/` (GitHub CI's `OGIT`, OGAR CI's `ogit`).
- `tools/ogit_validate.py` (in OGIT#11) runs the OGIT validator's checks without Java.
- `NTO/Accounting` loses three files (two verbs and one attribute) and one attribute is renamed (`counterpartyCountryCode`). OGAR#364 lowers the Accounting row floor in `ogar-from-schema` `ttl_emit.rs` from 36 to 33 to match.
- OGAR#364 re-measures `ontology_cache`'s `ONLY_IN_THE_TABLES` for the branch: the three Healthcare Assessment subclasses, each now in its own file, are the only entities OGAR's reader takes and the cache misses. 6-1's partition test starts from that list.

## 4. Open

1. Root namespace representation in `OgitUri`: option (a) or (b) in 6-1a.
2. Which reader owns the allowed table (§6). The `LinkSpec` replacement itself is decided by #1831.
3. Per-frame catalogues versus DT64-1's single catalogue (6-3).
4. Whether `AllowedEdge` is exposed through `OntologyRegistry` or only from the parse result.
5. Law candidates from schema mirroring (inverse, symmetric) stay unadmitted (#1831 §7); none of these stages admits a law.

## 5. Non-goals

No new `MappingProposalKind`, no second registry, no classids in the allowed table, no OGIT pin, no consumer copy, no law admission, no change to `ClassView::FieldMask`, no runtime DT64 engine.

## 6. Reconciliation with merged #1831: who owns the allowed table (OPEN)

Merged #1831 (`d-dt64-6-routing-bindstore-behavior-v1.md`, §5 row D-DT64-6-2 and §7) says one reader owns the table, that OGAR `OgitTables::Allowed` already is that table, and that the default is to read it rather than grow a second expansion in `lance-graph-ontology`. §2 stage D-DT64-6-2 above proposes `walk_rdf_list_nodes` and `AllowedEdge` in `lance-graph-ontology`, which is that second expansion. The two plans disagree, and the owner decides.

There is a dependency cycle to respect: `ogar-action-handler` depends on `lance-graph-ontology` (feature `ontology-cache`), so `lance-graph-ontology` cannot depend on OGAR.

| Option | Who parses `ogit:allowed` | Who checks | Cost |
|---|---|---|---|
| **A** OGAR owns | OGAR `OgitTables` | the standing OGAR test (§2, 6-1) | `lance-graph-ontology` cannot read the table without inverting the dependency; a consumer that needs it reads OGAR's. OGAR's reader must first be fixed (§7.1, §7.2). |
| **B** lance-graph-ontology owns | `lance-graph-ontology` `ttl_parse` (oxttl) | OGAR's `ontology_cache.rs` compares it with `OgitTables::Allowed` | Two parsers exist, but one is the authority and the other only a parity guard. Contradicts #1831 §7 ("No second … `ogit:allowed` parser") unless the owner amends it. |
| **C** generated snapshot (suggested, PROPOSED) | one generator writes `verbs.tsv` and `allowed.tsv` from the OGIT tree; OGAR and `lance-graph-ontology` both read the files | a regenerate-and-diff check, as for `SUPERSESSION-INDEX.md` | No Turtle parsing in either reader and no dependency edge between them. The snapshot carries the OGIT commit as a provenance line, never as a gate (#1831 D7). Fits OGIT being changed mostly by this organization. Needs an owner for the generator. `tools/import_check.py` in OGIT#11 already writes both files; a Rust generator in ruff (an OGIT RDF/OWL harvester) is the alternative. |

The snapshot's columns, as `import_check.py --emit-allowed` writes them: `source, verb, target, file, origin` with origin `allowed` or `from-to`; `--emit-verbs`: `verb, file, shape, uses`.

## 7. Measurements for the D-DT64-6-1 partition (2026-10-28)

Measured on OGIT `master` `b5731a8`, in a checkout directory named `OGIT`, with OGAR `main` `6b43a48`'s reader code (OGAR#364 changes no reader code) and OGIT#11's `tools/import_check.py` as the full Turtle parse. The OGAR numbers come from a throwaway example that called `OgitTables::load` and `ttl::parse_file` and was deleted after the run; it is not committed.

### 7.1 OGAR's verb and class rejections

`OgitTables::load` on `b5731a8`: 939 types, 1,940 attributes, 518 verbs, 2,771 allowed rows, 0 duplicates, 508 unresolved, **102 rejected files**.

| Rejected | Count | Why | Paths |
|---|---|---|---|
| verbs declared `a rdfs:Class; rdfs:subClassOf ogit:Verb` | 12 | `parse_verb` (`ogar-from-schema` `sgo.rs`) requires the text `a owl:ObjectProperty` | `NTO/WorkOrder/verbs/` |
| files whose type is written with column alignment (`a      owl:DatatypeProperty`) | 90: 74 attributes, 14 entities, 2 verbs | `detect_kind` (`ttl.rs`) and `parse_verb` match the type by substring with exactly one space after `a` | `NTO/CustomerSupport` 40, `NTO/Mobile` 47, `SGO/sgo` 3 (attributes `public`, `source`, `timeZone`) |

A sibling OGAR stage to D-DT64-6-1a, here called **D-DT64-6-1b (OGAR, PROPOSED)**, would change both readers. The counts below come from applying each change locally and reverting it:

| Change | rejected | types | attributes | verbs | allowed | unresolved |
|---|---|---|---|---|---|---|
| none (`b5731a8`) | 102 | 939 | 1,940 | 518 | 2,771 | 508 |
| A: `parse_verb` also accepts the `rdfs:Class` verb form | 90 | 939 | 1,940 | 530 | 2,771 | 508 |
| A + B: the type match also tolerates any whitespace after `a` | 0 | 953 | 2,014 | 532 | 2,823 | 495 |

Change A alone does not move `allowed`, because the 12 `rdfs:Class` verbs carry `ogit:from-to`, which `OgitTables::Allowed` does not read. Whether `Allowed` gains the from-to rows is part of the §6 decision. Not measured on the OGIT#11 branch.

### 7.2 Limits of `extract_allowed_block`

`extract_allowed_block` (`ogar-from-schema` `ttl.rs`) takes the first occurrence of `ogit:allowed`, the first `(` after it and the first `)` after that, and reads the first two words of each line in between. Compared file by file with the full parse over the 953 files under an `entities` directory:

| Case | Files | Effect in OGAR |
|---|---|---|
| more than one cell on a line | 0 | none on `b5731a8` |
| a cell spanning lines | 0 | none on `b5731a8` |
| a comment inside the list that contains `)` (each has `(2026-05-02)`) | 4: `NTO/Location/entities/Address.ttl`, `NTO/Politics/entities/Position.ttl`, `SGO/sgo/entities/Organization.ttl`, `SGO/sgo/entities/Person.ttl` | the block ends at that `)`; the cells after it are lost (1, 2, 37 and 71 cells) and one bogus pair `# ---` is read |
| `ogit:allowed` written in a description before the real block | 1: `NTO/Integration/entities/ConnectorRun.ttl` | the block is read from `(concept doc Section 7.1)` in the description: the bogus pair `concept doc` is read, and all 5 real cells are lost |
| a second allowed list in the same file | 1: `NTO/Healthcare/entities/Assessment.ttl` (four subjects, four lists) | only the first subject is read; this is the known `ONLY_IN_THE_TABLES` case |
| file rejected as an entity (§7.1) | 14 | the file's cells are not read at all |

In total OGAR reads 2,797 per-file pairs where the full parse reads 2,947: 116 cells are lost in the five files above, 5 bogus pairs are added, and the 14 rejected files hold the other 39. Each case is either a known-difference entry in the 6-1 test or an OGAR fix; the line-oriented walker cannot be fixed for comments and descriptions without tokenizing Turtle strings and comments.

### 7.3 Which files the counts mean

| Universe | Files | Read by |
|---|---|---|
| files containing the text `ogit:allowed` | 960 | lexical count (§0) |
| … under a directory named `entities` (any case, so `NTO/Credit/Entities` counts) | 952 | `kind_of` |
| … under `NTO/Healthcare/enumerations/` | 7 | neither reader's entity pass; `kind_of` ignores the directory. These are `rdfs:Class` enumerations with allowed lists. |
| … the root `ogit.ttl` | 1 | neither |
| `.ttl` files under an `entities` directory | 953 (952 above plus `NTO/WorkOrder/entities/Tenant.ttl`, which has no allowed list) | `kind_of` → `parse_file` |
| … accepted as entities by OGAR | 939 (953 − 14 rejected) | `OgitTables` |
| … with a non-empty allowed list as OGAR reads it | 686 | `OgitTables` |
| entity files containing `subClassOf ogit:Entity` with one space (any case) | 931 | the §0 grep. It misses the 8 `NTO/Meteorology` entities, which write `rdfs:subClassOf     ogit:Entity` with column alignment, and the 14 rejected files; 939 − 8 = 931 |

Row counts, by category, never mixed:

| Rows | `b5731a8` | OGIT#11 `0bf9a80` | Read by |
|---|---|---|---|
| `ogit:allowed` cells, full parse, all files | 2,950 | 2,948 | `import_check.py` |
| `ogit:from-to` cells (12 verbs) | 17 | 17 | `import_check.py`; not `OgitTables::Allowed` |
| `OgitTables::Allowed` rows (resolved, deduplicated) | 2,771 | NOT MEASURED | OGAR |

### 7.4 Corrections to merged documents

- #1831, `d-dt64-6-routing-bindstore-behavior-v1.md` §2, row on OGAR's verb reader: "`extract_allowed_block` reads one `[ verb target ]` pair per line from the first `ogit:allowed (`…`)`". That is the mechanism, but a `)` in a comment and an `ogit:allowed` in a description both end or misplace the block (§7.2). The same row says `parse_verb` rejects the `rdfs:Class` verbs; it also rejects column-aligned `owl:ObjectProperty` verbs (§7.1).
- #1831, same file, §5 rows D-DT64-6-1 and D-DT64-6-3, and this plan's §0: the 2,894 and 523 figures came from a consumer-side count, not from either reader. They should be replaced by the 6-1 partition when it runs.
- #1833, `d-sag-6-1-audit-reference-ledger-v1.md` §2, second row: it says "#1831 plan §2 and board entry" call crai-rs "the successor to #1760". Merged #1831 says the opposite: its §2 row reads "a consumer of #1760, not its successor", and its board entry says #1760 "is consumed by crai-rs CR-H0 / CR-H1". The row's right-hand column is correct; its attribution is wrong. The left column should read: "earlier drafts".

## 8. Description versus ON / WHEN / DO

An `ogit:allowed` row says what may be linked to what. It is a description, not behaviour: it names no event (ON), no guard (WHEN) and no effect (DO).

- ruff `ruff_spo_triplet` `role.rs`, `PredicateRole`: `DESCRIPTION` is `DESCRIBES | STRUCTURE | RELATES | PRESENTS`; `DO_ARM` is `EFFECTS | TRIGGERS`; `GUARD` is `CONSTRAINS | REQUIRES | AUTHORIZES`. The test `a_description_is_not_an_action_or_a_guard` pins that a description predicate has neither a DO nor a guard role. An allowed row is `RELATES`.
- OGAR `docs/HIRO-DO-ARM-LIFT.md` §1: behaviour has three slices (identity, contract, body), and DO is lossless only when the `ActionDef` points to the body. An allowed row has none of the three.
- #1814 `cfa-1-typed-incidence.md`, "Logical shape": ON is an event edge, WHEN a typed three-valued guard, DO an effect descriptor that grants nothing.
- #1819 `on-do-watched-where-factored-dispatch-plan-20261027.md` §1: a candidate DO-presence mask never proves that a mapping applies.
- #1831 §1 (WHEN × ON × DO attaches per DR#3 cell) and §3 (dispatch order).

Description is therefore **not a fourth axis** beside ON, WHEN and DO. It is what a DT64 descriptor already records about a predicate (§2, 6-3: `observed_or_derived` = observed-schema). An OGIT verb placed in a DT64 slot carries description only. Its ON, WHEN and DO cells are UNKNOWN until a source attests behaviour for it (#1831 §5, D-DT64-6-5), and an empty cell is never read as "no behaviour".

## 9. D-SAG-6-1 references

#1833 (`d-sag-6-1-audit-reference-ledger-v1.md`, merged) is the reference list for the D-SAG-6-1 audit; §1 lists what #1760 §7 does not, §2 the stale statuses. This plan adds:

| Reference | State |
|---|---|
| crai-rs `docs/ROADMAP.md`, section "crai-rs staged harvest roadmap", stage table row **CR-H2** (conditional): its dependency column lists "D-SAG-6-1 Rust HIR/MIR/Charon audit", so CR-H2 waits on the D-SAG-6-1 audit (crai-rs `main` `378a24a`, private) | MISSING from #1760 §7; read in full here |
| #1833's own §2 second row | STALE: attribution wrong (§7.4) |
| this plan's §7 measurements | MISSING from both; relevant where D-SAG-6-1 compares the Ruff → OGAR bridge with OGIT readers |

## 10. Corrections to merged #1831 from the D-CFA and EAW research docs (2026-10-28)

Each note names the merged parent file `d-dt64-6-routing-bindstore-behavior-v1.md` (#1831) and the section it corrects; the parent is not edited. Every source below was read in full on `main` `f98da6b`. Research paths are under `docs/research/`.

**10.1 #1813 is the start of the ON×WHEN×DO series (§2 grounding table).** #1831 cites #1814 and #1819 and the `counterfactual-behavior/` directory, never #1813. #1813 (`counterfactual-behavior/README.md`, sections "Claim to falsify" and "Non-negotiable semantics"; `thought-dependency-scratchpad.md`, "Breadcrumb keys") is the first document: it fixes the 64³ incidence relation as logical, separates thought slots from cube coordinates, and keeps each cell able to reference several methods.

**10.2 #1814 is the source-harvesting contract the per-cell registry follows (§1 D3, §2, §3).** #1831 cites `cfa-1-typed-incidence.md` only for its summary and kill test I03. Its sections "Question", "Logical shape, no Rust ABI commitment" and "Kill tests" are the contract: ON, WHEN and DO are harvested from the source owners (OGAR, Ruff, the ERP harvest, rust-analyzer), never manufactured. "Logical shape" also gives D3's field lists: REGISTRATION (MethodId, signature, source identity, revision and schema, kind and phase, declared precedence or supersession, optional recipe reference, cells, permissions) and WITNESS (focus, snapshot, scope, codebook generation, knownness). THOUGHT is never a fourth axis.

**10.3 #1819 gives the binding tuple and the watch rules (§1 D3, D4, D8; §2; §3; §6 Q3).** From `on-do-watched-where-factored-dispatch-plan-20261027.md`:
- "Operator correction: methods are mapped onto algebra": a binding is `r = (source MethodId, ON, WHEN, DO, focus, scope, schema/version/snapshot witness, compiled_polyfill_ref)`; two bindings of one polyfill are distinct, never redundant; `selected(r, e, source)` needs ON to match, DO within `ON_DO[e]` and `WHERE_TRUE(WHEN, scope, borrowed_source)`. This is the shape D3/D4's registry carries.
- "Normal convergence": many mappings to one qualified outcome, and one outcome to several typed destinations, are both kept; a real conflict needs incompatible source-qualified results under one coordinate and snapshot plus a domain law. #1831 §6 Q3 paraphrases this.
- §2 "watch as a possible source-owned invalidation": warm reuse of a Keep holds only while the snapshot and every predicate and scope witness stay valid; a compiled recipe may be reused on a fresh binding when its old mask may not; repair needs a complete change set from the owner, otherwise full re-evaluation or REFUSE; a watch signal is never a version witness. D8's frame-keyed register copy should cite this beside `FrameStore::check`.
- §2 items 5 and 6: DO records capability identity, not authorization; `ScatterOrU32` output stays terminal-only.
- §1 "Existing owners": Quack `Filter::sql_where`, `lower_fused` and mask-risc terminals; #1792; #1790/#1791 (workload-specific); `dir-rmap-probe::pop::Pop` as an existence example only; the K3 validity planes of #1334/#1677/#1687.

**10.4 #1815 grounds ordered behaviour and the override rule (§1 D5, D8; §2; §3; §6).** From `cfa-2-ablation-replay.md`:
- "What a comparison must preserve" (order, multiplicity, trace, read and write sets, full observation boundary), with kill tests A06 and A12, is the in-repo basis for #1831 §3's "references ordered behaviour"; crai-rs stays a second, private source.
- D5's immutability and re-route rule: A01 (T0 refuses interventions), A09 (ablating one of two methods in a cell removes only that one), A11 (a changed source under unchanged coordinates needs rebinding, recompute or refusal; also grounds D8's re-prefetch).
- DO has no effect authority: A10.
- "Thesis": 64³ intervention addresses are not 64³ concurrent programs, and Thought-Git slots are independent of the ON/WHEN/DO ordinals.
- **OPEN:** cfa-2's interventions include "suppress one registered MethodId" and "suppress all methods at a cell"; #1819's operator correction re-scopes those as hypothetical changes to bindings, guard valuations or input populations, leaving the method inventory intact. #1831 states neither. Interventions on registry cells take #1819's form unless the owner rules otherwise.

**10.5 OPEN: is the per-cell registry inside DT64? (§1 D3, D4).** D3/D4 place the ON×WHEN×DO registry in DT64 ("DT64 is the DTO bind store", WHEN×ON×DO per DR#3 cell). Three sources keep them apart: `thought-dependency-scratchpad.md` B1 ("DT64-Q0 is typed witness/binding and a borrowed input seam, not the new S1 ON/WHEN/DO cube"), cfa-1 "Question" (S1 is "not equivalent" to the DT64-0 catalogue) and cfa-1 "Logical shape" (the mapping stays "a cold proposed view until its owner agrees"). #1831's own §2 rows and §7 already repeat the first two, so the tension is inside #1831. The user decisions stand; the owner of DT64-1..3 rules. Add `epistemic-awareness/dr64c-epistemic-awareness-scratchpad-20261027.md`, "Research safety / epistemic firewall" ("No new runtime, physical DR³ ABI, binding store, …"), to the sources #1831 §4 lists against a persistent store.

**10.6 OGIT-derived predicates carry no guards or effects (D-DT64-6-3).** cfa-1 "Corpus options" item 2 names `OgitTables` `ogit:allowed` as schema facets only ("represent UNKNOWN rather than invent"), and kill tests I04 and I07 apply. Every OGIT-derived descriptor therefore has ON, WHEN and DO UNKNOWN, as §8 says.

**10.7 The CausalEdge64 evidence view is missing from D1 (§1 D1).** dr64c scratchpad item 5 and `eaw-1-relationship-presence-and-dr64c-coordinates.md` "Two overlapping address views" give two views over one base address: Identity × Property/Edge × Predicate (rule binding) and Identity × Property/Edge × CausalEdge64 evidence, where the edge record is a payload, not 64 more coordinates. `eaw-2-…` "A necessary TWO-DOMAIN mapping": the Epi5 `u32` code selector is not a DR64 row mask.

**10.8 Presence is five quantities (§1 D3; D-DT64-6-2 reporting).** EAW-1 "Hypothesis, explicitly distinct from authority" separates P (attested present), K (scoped completeness), Unknown, A (K3 applicability) and M (mapping occurrences). A D3 registration is M, never P, and a 0 bit proves nothing. EAW-5 "What the merged source data actually shows" and kill test E5-5: OGAR's 102 rejected files and 508 unresolved references (§7.1) are UNKNOWN coverage, never negative edges.

**10.9 Immutability is a goal (§1 D5, D8).** dr64c scratchpad item 6: "Version labels alone never prove physical immutability", and versioned meaning and replay "are goals with open admission gates". EAW-1: a source-changing replay "must bind a new frame or refuse". D5 is a policy decision whose physical guarantee is open.

**10.10 The ternlog fold has three operands (§3 dispatch step 4).** `eaw-3-…` "Scope and semantics": Eligible = AND3(P, A_TRUE, E_TRUE); a fourth operand needs another operation or a precomposed plane, and a TRUE-only mask is not K3 negation.

**10.11 Worlds stay separate (§3).** `eaw-4-…`, section on the #45 sibling landing: `next_version` over sibling worlds is a comparison instrument, not one world. Any registry work that uses Thought-Git puts the world boundary in the caller.

**10.12 D-DT64-6-1 has a parent (§5 D-DT64-6-1).** `eaw-5-…` "Suggested source-harvest ladder" H4 is the OGIT corpus step (accepted and rejected files, an independent raw-corpus coverage read, no inverse inferred from names such as `consistsOf`/`isPartOf`). D-DT64-6-1 is that step, not a standalone stage.

**10.13 Dated status (stale, not edited).** All of #1813–#1830 are merged on `main` (`git log origin/main`). `counterfactual-behavior/stack-receipts.md` calls #1813–#1818 draft PRs, and the dr64c scratchpad's "Live audit correction" and EAW-5 call #1813–#1819 open; #1831 §2 says "#1814 and #1819 have merged since". All are dated 2026-10-27 and stale.

## Appendix A — namespace split of the allowed edges (approximate)

Regex over the TTL tree, not a Turtle parser, so the totals differ slightly from OGAR's reader (≈2,932 here versus the 2,894 that #1831 §5 cites). It is enough to show the share of root-namespace verbs; D-DT64-6-1 replaces it with exact counts from both readers. Run as `python3 -I allowed_ns.py <OGIT checkout>`.

```python
import re,sys,pathlib,collections
root=pathlib.Path(sys.argv[1])
blk=re.compile(r'ogit:allowed\s*\((.*?)\)\s*;',re.S)
pair=re.compile(r'\[\s*([\w.:-]+)\s+([\w.:-]+)\s*\]')
subj=re.compile(r'^(ogit[\w.]*:\w+)\s*$',re.M)
edges=set();n=0
for f in root.rglob('*.ttl'):
    t=f.read_text(errors='replace')
    m=blk.search(t)
    if not m: continue
    s=subj.search(t); s=s.group(1) if s else '?'
    for v,d in pair.findall(m.group(1)): edges.add((s,v,d))
root_ns=lambda x:x.startswith('ogit:')
c=collections.Counter((root_ns(s),root_ns(v),root_ns(d)) for s,v,d in edges)
print('distinct edges',len(edges))
for k,v in sorted(c.items()): print('src_root=%s verb_root=%s dst_root=%s'%k, v)
print('edges with root verb',sum(v for k,v in c.items() if k[1]))
print('edges with root src or dst',sum(v for k,v in c.items() if k[0] or k[2]))
```
