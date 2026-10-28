# D-DT64-6 predicates: implementation plan for the OGIT predicate expansion (stages 6-1 to 6-4)

**Date:** 2026-10-28. **Status:** PROPOSED / DOCS ONLY. **Parent:** [D-DT64-6](https://github.com/AdaWorldAPI/lance-graph/blob/claude/clone-ndarray-lance-graph-dqfjlx/.claude/plans/d-dt64-6-routing-bindstore-behavior-v1.md) ([#1831](https://github.com/AdaWorldAPI/lance-graph/pull/1831), open; the link points at its branch until it merges), §5 stages D-DT64-6-1 to D-DT64-6-4. **Not:** code, test, ABI or merge authorization. Each stage below needs its own owner approval before any code.

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

Shape of one allowed block (`NTO/Schedule/entities/Calendar.ttl`): the entity is the source; each list cell is a blank node `[ verb target ]`.

```turtle
ogit.Schedule:Calendar a rdfs:Class; rdfs:subClassOf ogit:Entity;
  ogit:allowed ( [ ogit:contains ogit.Schedule:Event ] [ ogit:belongs ogit.Auth:Account ] );
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

**Acceptance.** 523 verbs and 931 + root entity subjects classified on OGIT `master`, with zero unclassified. No change to existing Entity and Attribute proposal counts other than the added root-namespace ones; this is checked by diffing proposal counts per namespace before and after.

### D-DT64-6-1 — two-reader partition reconciliation (OGAR, read-only)

**Where.** OGAR `crates/ogar-action-handler/tests/ontology_cache.rs`, feature `ontology-cache`. It is the only place that links both readers, because the reverse dependency would be a cycle.

**Change.** A new test, `verbs_and_entities_partition_identically`. Both readers load the same `ogit_checkout::root()` in one run, and each prints its partition as accepted / rejected / duplicate / unresolved, per kind (entity, verb). Every set difference is listed with the file and the reason. Known differences live in a named const with one line of reason each, the same pattern as `ONLY_IN_THE_TABLES`. Unexplained differences fail the test.

**Explains.** The 523 / 2,894 (odoo-rs §8) versus 518 / 2,756 (eaw-5 line 24, OGAR #363) gap. OGIT file counts are identical at `2315167` and `b5731a8`, so the gap is on the reader side. The test records the observed OGIT commit as a provenance line in its output only.

**Gate.** D-DT64-6-1a merged and consumed by OGAR. Until then, the test expects 511 + 183 verbs missing on the lance-graph side and says so.

### D-DT64-6-2 — the `(src, verb, dst)` allowed table in lance-graph-ontology

**Change.**
1. Entity pass: in `parse_into_proposals`, for `SubjectKind::Entity`, read the `ogit:allowed` list with a new `walk_rdf_list_nodes` that returns blank-node list items (today's `walk_rdf_list` returns only IRIs). Each cell must have exactly one `(predicate, Iri)` pair, giving `(entity, verb, target)`. A cell with any other shape is reported, not skipped.
2. Verb pass: for the 12 `from-to` verbs, read `ogit:from-to ( [ ogit:from A ; ogit:to B ] )` into `(A, verb, B)`.
3. Output: a new `AllowedEdge { src: String, verb: String, dst: String, source_uri, origin: AllowedOrigin::{EntityAllowed, VerbFromTo} }` collection beside the proposals, as a parse result field, not a new `MappingProposalKind` (#1831 §7: "No second registry"). It holds canonical names, no classids (#1831 D-DT64-6-2: classid-free). Duplicates are counted, not deduplicated silently.
4. The per-verb placeholder `LinkSpec` stays as it is until a consumer needs typed links. A per-verb `LinkSpec` can only summarise many `(src, dst)` pairs and is not the table. Replacing it is a separate decision recorded in OPEN below.
5. Unresolved: a `dst` or `verb` that is not a declared subject in the same parse is reported as `unresolved`, using the same three-way partition OGAR reports.

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

## 4. Open

1. Root namespace representation in `OgitUri`: option (a) or (b) in 6-1a.
2. Whether the per-verb placeholder `LinkSpec` is replaced, summarised or kept once the allowed table exists.
3. Per-frame catalogues versus DT64-1's single catalogue (6-3).
4. Whether `AllowedEdge` is exposed through `OntologyRegistry` or only from the parse result.
5. Law candidates from schema mirroring (inverse, symmetric) stay unadmitted (#1831 §7); none of these stages admits a law.

## 5. Non-goals

No new `MappingProposalKind`, no second registry, no classids in the allowed table, no OGIT pin, no consumer copy, no law admission, no change to `ClassView::FieldMask`, no runtime DT64 engine.

## Appendix A — namespace split of the allowed edges (approximate)

Regex over the TTL tree, not a Turtle parser, so the totals differ slightly from OGAR's reader (≈2,932 here versus 2,894 in odoo-rs §8). It is enough to show the share of root-namespace verbs; D-DT64-6-1 replaces it with exact counts from both readers. Run as `python3 -I allowed_ns.py <OGIT checkout>`.

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
