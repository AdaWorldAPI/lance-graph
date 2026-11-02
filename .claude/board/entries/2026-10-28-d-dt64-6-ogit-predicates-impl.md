# D-DT64-6-1a to 6-4: implementation plan for the OGIT predicate expansion

**Status:** PROPOSED / DOCS ONLY. Plan: [d-dt64-6-ogit-predicates-impl-v1.md](../../plans/d-dt64-6-ogit-predicates-impl-v1.md). Companion to [#1831](https://github.com/AdaWorldAPI/lance-graph/pull/1831) (D-DT64-6), which stages the OGIT work in one line per stage. No code, test, ABI or merge authorization.
Read at lance-graph `main` `eeaaafb`, OGAR `main` `6b43a48`, AdaWorldAPI/OGIT `master` `b5731a8` (provenance receipt, not a pin).

## Findings that shape the stages (MEASURED or VERIFIED-IN-CODE)

- `ttl_parse` `classify` accepts a verb only as `rdfs:Class` + `subClassOf ogit:Verb`. OGIT has 523 verb files: 511 are `owl:ObjectProperty` + `subPropertyOf ogit:Verb` and fall to `SubjectKind::Other`; only the 12 `NTO/WorkOrder` verbs match.
- `OgitUri::parse` needs an `ogit.<Domain>:` prefix, so root `ogit:` subjects (183 SGO verbs, SGO core entities) are skipped. About 79% of allowed edges use a root `ogit:` verb; about 19% have source, verb and target all in a domain namespace (regex count, approximate).
- `ogit:allowed` is on entity files (960 files, none under `verbs/`), one `[ verb target ]` blank node per cell; `ogit:from-to` only on the 12 WorkOrder verbs. `walk_rdf_list` returns IRI items only, so blank-node cells are never read.
- OGAR reads the same block with `ogar-from-schema` `extract_allowed_block` into `OgitTables::Allowed`; OGAR depends on lance-graph-ontology, so the parity test belongs in OGAR `tests/ontology_cache.rs`, which already compares entity names between the two readers.
- DT64-1 is one fixed 64-slot catalogue; OGIT's 378 used, polymorphic verbs need one catalogue per ClassView frame (p95 10 verbs per class, max 123, 2 classes above 64).

## Stages added or detailed

D-DT64-6-1a (new: verb classification and root namespace in `ttl_parse`), D-DT64-6-1 (two-reader partition test in OGAR), D-DT64-6-2 (`(src, verb, dst)` table from entity `ogit:allowed` plus `from-to`, pair-for-pair against `OgitTables::Allowed`), D-DT64-6-3 (per-frame catalogues filling DT64-1's descriptor), D-DT64-6-4 (HotPlug read access, no consumer copy). Each with tests that fail on today's `main`, silence twins and a disable run.

## Open

Root-namespace form in `OgitUri`; which reader owns the allowed table (OGAR, lance-graph-ontology, or a generated TSV snapshot both read; plan §6); per-frame catalogues versus DT64-1's single catalogue; whether the allowed table is exposed through `OntologyRegistry`. The `LinkSpec` replacement is decided by #1831.

## Update (same day): measurements and reconciliation, plan §§6–9

- OGAR rejects 102 OGIT files on `b5731a8`: the 12 `rdfs:Class` verbs and 90 files written with column-aligned whitespace after `a`. Accepting both forms (a proposed OGAR stage, D-DT64-6-1b) measured 102 → 90 → 0 rejected (MEASURED, changes reverted).
- `extract_allowed_block` loses 116 cells in 5 files: a `)` in a comment inside the list (4 files) and `ogit:allowed` written in a description (1 file). No file has two cells on a line or a cell over two lines (MEASURED).
- File universes 960 / 953 / 952 / 939 / 931 and row counts per category (allowed versus from-to) are separated in plan §7.3.
- Corrections recorded for merged #1831 §2 and #1833 §2 (the "successor" attribution); description is not a fourth axis beside ON, WHEN and DO (plan §8); crai-rs CR-H2 waits on the D-SAG-6-1 audit (plan §9).
