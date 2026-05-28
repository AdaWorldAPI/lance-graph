# ruff-py-dto + Odoo POC — empirical report

**Date.** 2026-05-28 (in-session, auto-attended).
**Patch.** `vendor/ruff-patches/` — D-RPYDTO-2a class-body recursion.
**Binary.** `/tmp/ruff-fork/ruff-main/target/release/ruff-py-dto` (build 1m29s).
**Config.** `tools/ruff-py-dto-odoo-bin/odoo.config.json`.
**TTL emitter.** `tools/ruff-py-dto-odoo-bin/bundles_to_ttl.py`.

## Before vs after the class-body-recursion patch

Same input (`odoo/addons/account/`, ~3000 `.py` files), same binary, same
config. Only difference: top-level loop → recursive walker into class bodies.

| metric                                        | before |   after |
| ---                                           |    ---: |    ---: |
| bundles produced (`harvest`)                  |       0 |     363 |
| families produced                             |       0 |      32 |
| `decorator_by_attribute.depends` (preflight)  |       0 |     351 |
| `decorator_by_attribute.model`                |       0 |     305 |
| `decorator_by_attribute.classmethod`          |       0 |     130 |
| `decorator_by_attribute.constrains`           |       0 |      66 |
| `decorator_by_attribute.onchange`             |       0 |      45 |
| `decorator_by_attribute.depends_context`      |       0 |      38 |
| `decorator_by_attribute.model_create_multi`   |       0 |      24 |
| `decorator_by_attribute.ondelete`             |       0 |      21 |
| `candidate_misses.defs_with_decorators_but_unmatched` |   2 |    1043 |

`candidate_misses` jumps because the walker now sees ~1000 decorated methods
on classes that don't match the proposed `route` rule — exactly the
diagnostic signal the preflight is designed to surface.

## Scale across full `odoo/addons/` (622 addons)

```text
totals.bundles  = 3555
totals.families = 388
```

Top-20 families by method count:

```
   264  account_move
   106  res_config_settings
   101  sale_order
    82  sale_order_line
    80  res_partner
    75  product_template
    70  stock_picking
    67  stock_move
    60  account_move_line
    51  account_journal
    50  res_company
    50  mrp_production
    50  hr_employee
    50  event_event
    48  account_tax
    45  project_task
    44  crm_lead
    44  account_payment
    40  project_project
    ...
```

## TTL emitter — 7-tuple grammar over the bundles

`tools/ruff-py-dto-odoo-bin/bundles_to_ttl.py` reads the NDJSON and emits one
TTL file per family with the dual-axis + transitivity classification per
`E-BUSINESS-LOGIC-IS-GRAMMAR-1`.

Distribution across all 3555 methods:

| facet                       |   count |  fraction |
| ---                         |    ---: |     ---: |
| `axis:Deterministic`        |  3,166  |    89.1% |
| `axis:Heuristic`            |    389  |    10.9% |
| `verb:Transitive`           |  3,087  |    86.8% |
| `verb:Intransitive`         |    468  |    13.2% |

Causal regulation-marker hits (text-scan only — D3 LLM extraction will add the
rest):

```
4 × UStG#18-Voranmeldung
3 × GoBD#Unveraenderbarkeit
2 × HGB#239-Festschreibung
```

The 89/11 deterministic/heuristic split lines up with the prior on
Odoo: most `@api.depends` / `@api.constrains` ARE deterministic computes /
checks; the heuristic 11% are `@api.onchange` + name-pattern-matched
`_resolve_*` / `_guess_*` / `_find_*` / `_match_*` etc.

The 87/13 transitive/intransitive split reflects how many methods carry
`_check_*` / `_validate_*` semantics (raise without return) vs.
compute/return semantics.

## OWL 2 conformance (per Keet 2025-11-15 / 2025-11-17)

Maria Keet's two blog posts ("No, an ontology isn't 'just RDF'", 2025-11-15;
"Just Turtle and RDF vs OWL examples: the CPEV and FIBO", 2025-11-17) document
the typical failure mode: a `.ttl` file that parses as Turtle but drops to
OWL 2 Full (undecidable) because of undeclared classes/properties, blank-node
lists, `rdf:langString` as a data property range, or reserved-vocabulary
misuse (e.g. `owl:minQualifiedCardinality` as an annotation property in FIBO).

`bundles_to_ttl.py` emits a TBox preamble (`OWL_TBOX` in the source) that
addresses each of those failure modes for the OGIT vocabulary:

| Keet failure mode                                  | OGIT remediation                                                                  |
| ---                                                | ---                                                                               |
| Undeclared `owl:Class` instances                   | `ogit:Method`, `axis:Classification`, `verb:Transitivity` declared as `owl:Class` |
| Undeclared annotation properties                   | Every `ogit:*` / `axis:*` / `verb:*` / `tekamolo:*` predicate has `a owl:AnnotationProperty` |
| `rdf:langString` as a data property range         | Not used                                                                          |
| Blank-node lists (`[...]` patterns → anonymous individuals) | Not used — every method gets a direct IRI subject                                 |
| Reserved-vocabulary-as-annotation misuse           | No `owl:*` predicate is reused as annotation                                       |
| No explicit OWL profile declaration                | `owl:Ontology` + `owl:versionInfo "OWL 2 EL profile (serialized as Turtle 1.1)"` |

Audit against the per-family TTL (rdflib parser + structural check):

```
OK: parsed 1379 triples (account_move family)
owl:Restriction instances (EL forbids most):  0
OWL types used:  Class, NamedIndividual, AnnotationProperty, Ontology
undeclared user predicates:  0
blank-node subjects:  0
owl:Ontology declarations:  1
```

Caveat per Keet: rdflib parses RDF, not OWL. The real DL-conformance gate
is Protégé / OWL API / a DL reasoner running the RDF→OWL mapping spec
(§3 of the OWL 2 Mapping to RDF Graphs). The audit above only confirms
the necessary conditions; a full DL classifier run is the sufficient
condition and remains future work for the engine-integration phase.

## Sample TTL — `account_move._check_invoice_currency_rate`

```turtle
### account_move._check_invoice_currency_rate  [Deterministic / Intransitive]
odoo:account_move._check_invoice_currency_rate a ogit:Method ;
    ogit:family odoo:account_move ;
    ogit:methodName "_check_invoice_currency_rate" ;
    ogit:matchId "odoo_constrains" ;
    axis:classification axis:Deterministic ;
    verb:transitivity verb:Intransitive ;
    ogit:decorator """@api.constrains('invoice_currency_rate')""" ;
    ogit:signature """self""" ;
    ogit:bodyLines 12 ;
    rdfs:comment """[Python body source preserved verbatim]""" ;
    dcterms:source """models/account_move.py:L2845-2856""" .
```

Full file: `tools/ruff-py-dto-odoo-bin/sample-output.account_move.ttl`.

## Known imperfections (refined in D-RPYDTO-5 / D-RPYDTO-6)

1. **HYBRID under-detection.** `_compute_abnormal_warnings` is classified
   `Deterministic / Transitive` because the heuristic axis assigner only
   looks at decorator + method-name; its docstring ("Bell curve... we warn the
   user") would classify it `Hybrid` under the proper priority rules. The
   real D-RPYDTO-5 classifier integrates LLM-extracted L-doc evidence
   (`E-LITERATURE-IS-INGRESS-3-1`) plus body-text NLP markers.

2. **Causal-ref recall is low.** Pure text matching catches 9 markers; the
   real D3 path runs the LLM literature extractor across HGB/UStG/AO/GoBD
   PDFs to populate `tekamolo:causal` from regulation IRIs the body
   *implements* but doesn't name.

3. **No quantity extraction yet.** `tekamolo:quantities` (the 7th tuple
   slot) is empty because the prototype doesn't yet parse magic-number
   literals or fiscal threshold constants out of the body. Trivial follow-up.

4. **One TTL block per method, no graph edges yet.** Cross-family edges
   (`ogit:invokes`, `ogit:reads_field`, `ogit:delegates_to`) require a
   second pass over the Python AST — these are the `DelegationTuple` rows
   from D-RPYDTO-2a step 4 in `.claude/odoo/ruff-patches/README.md`.
   **(UPDATE 2026-05-28: now done as a POC second pass via
   `.claude/odoo/extract_delegation.py` — reads the original `.py` files
   via bundle file+line metadata and walks the AST. 100% parse success
   on 3555/3555 methods, emitting 831 invoke + 2162 read + 519 write +
   456 raise + 11 traverse + 697 reads_env edges. Rust path remains the
   proper home; this POC validates the schema.)**

## Artifacts (in `/tmp/`, ephemeral)

- `/tmp/ruff-fork/ruff-main/` — patched ruff working tree.
- `/tmp/ruff-fork/ruff-main/target/release/ruff-py-dto` — binary.
- `/tmp/odoo-extract/harvest-account2/` — account-only harvest (363 bundles).
- `/tmp/odoo-extract/harvest-full/` — full odoo/addons harvest (3555 bundles).
- `/tmp/odoo-extract/preflight-account2/` — preflight report.
- `/tmp/work/ttl-account/` — TTL for account-only (32 files, ~400 KB).
- `/tmp/work/ttl-full/` — TTL for full tree (388 files, 4.4 MB).
- `/tmp/work/*.log` + `/tmp/work/*.json` — build logs, manifests, distributions.

These are regenerable from the committed patch + config + emitter; not
committed.

## What this proves

1. **D-RPYDTO-2a is critical-path.** Zero → 3555 bundles from one ~50-line
   patch. Every downstream stage (typed `OdooMethod` consts, NARS revision
   inputs, generative-DSL inputs, KG ingest) depends on this.

2. **The 7-tuple grammar is operational, not just a manifesto.** End-to-end
   pipeline: Python AST → NDJSON bundles → OGIT TTL with axis +
   transitivity + causal markers + provenance. Runs in 30 s for a full Odoo
   tree on a release-mode single core.

3. **Cross-source validation is now mechanically possible.** Each TTL
   method carries `dcterms:source` pointing to `<file>:<line_start>-<line_end>`
   in the Python tree, and `axis:classification` deterministic vs
   heuristic. Feed this alongside the L-doc OWL/TTL outputs and run SHACL
   shapes that say "every `ogit:Method` whose `tekamolo:causal` resolves to
   regulation:HGB#239 MUST be `axis:Deterministic`" — this is the
   triangulation `E-OWL-IS-THE-UNIVERSAL-INGRESS-1` calls for, and the
   substrate now exists.

## Next steps (priority order)

1. **Land D-RPYDTO-2a upstream**, formally — split into the three PRs sketched
   in `vendor/ruff-patches/README.md`.

2. **Add a TargetSpec for OGIT-TTL** in ruff-py-dto so the emitter logic in
   `bundles_to_ttl.py` moves into the Rust binary and ships with the same
   schema-version discipline as the NDJSON path.

3. **Implement D-RPYDTO-5 priority classifier** to fix the `_compute_abnormal_warnings`
   HYBRID-miss class of bugs. Uses LLM-extracted L-doc cross-evidence per
   `E-LITERATURE-IS-INGRESS-3-1`.

4. **Wire D3 LLM literature extraction** so causal markers come from regulation
   PDFs, not text-matching on `HGB` substrings.

5. **Add `DelegationTuple` extraction** (second AST pass) to populate
   cross-family `ogit:invokes` / `ogit:reads_field` / `ogit:delegates_to`
   edges. Without these edges the TTL is a bag of methods, not a graph.
