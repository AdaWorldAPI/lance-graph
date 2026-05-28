# OGIT meta-DTO schema — alignment for the Odoo retarget

> **Status:** FINDING (2026-05-28). Audit of `AdaWorldAPI/OGIT` HEAD against the
> in-session `bundles_to_ttl.py` output. Proposal — not executed.
>
> **Reads:** the keet posts (`keet.wordpress.com/2025/11/15/` +
> `2025/11/17/`) flagged me to check what an ontology actually IS in
> OGIT terms, not just whether it parses as Turtle.

## Headline finding

`bundles_to_ttl.py` emits TTL that conforms to **OWL 2 EL** (per the previous
commit `cdcbc349`) but does NOT conform to **OGIT's meta-DTO schema** — the
concrete declarative pattern that every OGIT entity / attribute / verb file
follows. The audit below shows what OGIT actually requires and where the
emitter diverges.

## OGIT's meta-DTO schema (verified against repo HEAD `fc2c9bd`)

**Canonical namespace.** `<http://www.purl.org/ogit/>` — NOT
`https://ogit.adaworldapi.com/` (which is the namespace my emitter currently
hardcodes; that URL does not resolve and is not used anywhere in OGIT).

**Three meta-types**, defined in the root `ogit.ttl`:

| meta-type | OWL kind | rdfs:label | what instances are |
| --- | --- | --- | --- |
| `ogit:Entity`    | `rdfs:Class`            | "Entity"    | the noun nodes in the graph (JournalEntry, SalesOrder, Customer, FiscalJurisdiction, ...) |
| `ogit:Attribute` | `owl:DatatypeProperty`  | "Attribute" | the typed scalar properties on entities (accountingStandard, iso3166Alpha2, ...) |
| `ogit:Verb`      | `owl:ObjectProperty`    | "Verb"      | the typed edges between entities (contributesTo, hasFiscalCountry, ...) |

Every domain artifact lives in `NTO/<Domain>/{entities,attributes,verbs}/<Name>.ttl`
or `SGO/{core,ogit,sgo}/{entities,attributes,verbs}/<Name>.ttl` (one file per
artifact). The three sibling directories enforce the trichotomy at the
filesystem level.

**The meta-DTO declaration block** every Entity uses, demonstrated by
`NTO/Accounting/entities/JournalEntry.ttl` (which is the OGIT projection of
Odoo `account.move`, authored 2026-05-28 by a prior Claude session this
afternoon):

```turtle
ogit.Accounting:JournalEntry
    a rdfs:Class;
    rdfs:subClassOf ogit:Entity;
    rdfs:label "JournalEntry";
    dcterms:description "A double-entry accounting journal entry — the canonical
        OGIT projection of Odoo `account.move` (and the deeply analogous SAP
        `BSEG` document line, MS Dynamics `JournalLine`, DATEV `Buchung`). ...";
    dcterms:valid "start=2026-05-28;";
    dcterms:creator "Claude (AdaWorldAPI/lance-graph 3-hop optim)";
    ogit:scope "NTO";
    ogit:parent ogit:Node;
    ogit:mandatory-attributes (
        ogit:id
        ogit:currency
        ogit:status
    );
    ogit:optional-attributes (
        ogit:name
        ogit:type
        ogit:summary
        ogit:createdAt
        ogit:lastUpdatedAt
        ogit.Accounting:fiscalCountryCode
        ogit.Accounting:commercialPartnerCountryCode
    );
    ogit:indexed-attributes (
        ogit:id
        ogit:status
        ogit.Accounting:fiscalCountryCode
        ogit.Accounting:commercialPartnerCountryCode
    );
    ogit:allowed (
        [ ogit.Accounting:hasFiscalCountry ogit.Accounting:FiscalJurisdiction ]
        [ ogit:references ogit.FinancialMarket:Country ]
        [ ogit:reports ogit:Timeseries ]
        [ ogit:generates ogit:Timeseries ]
    ) .
```

**The seven mandatory slots** every Entity carries:

1. `a rdfs:Class` + `rdfs:subClassOf ogit:Entity` — the meta-type.
2. `rdfs:label` — short identifier (matches the filename stem).
3. `dcterms:description` — English description (LLM-readable explanation).
4. `dcterms:valid "start=YYYY-MM-DD;"` — DCMI Period for validity range.
5. `dcterms:creator` — author attribution.
6. `ogit:scope` — partition code ("NTO" / "SGO" / "SDF").
7. `ogit:parent` — usually `ogit:Node`, but can refer to a more specific parent.

**The four attribute-classifier lists**:

- `ogit:mandatory-attributes` — required at every instance (RDF list).
- `ogit:optional-attributes` — may be present (RDF list).
- `ogit:indexed-attributes` — should be backed by a graph-DB index (RDF list).
- `ogit:allowed` — list of `[verb target_entity]` pairs declaring allowed
  outgoing edges (uses the very blank-node pattern Keet flags as the CPEV
  anti-pattern; OGIT explicitly accepts this trade-off because the meta-DTO
  expressivity matters more than OWL 2 DL conformance).

**Attribute declarations** are simpler (`NTO/Accounting/attributes/accountingStandard.ttl`):

```turtle
ogit.Accounting:accountingStandard
    a owl:DatatypeProperty;
    rdfs:subPropertyOf ogit:Attribute;
    rdfs:label "accountingStandard";
    dcterms:description "...";
    dcterms:valid "start=2018-12-12;";
    dcterms:creator "Viktor Voss" .
```

**Verb declarations** mirror attributes (`NTO/Accounting/verbs/contributesTo.ttl`):

```turtle
ogit.Accounting:contributesTo
    a owl:ObjectProperty;
    rdfs:subPropertyOf ogit:Verb;
    rdfs:label "contributesTo";
    dcterms:description "Verb associates which entity contributes to another.";
    dcterms:valid "start=2018-12-05;";
    dcterms:creator "Viktor Voss" .
```

## Scope partitions seen in the wild

`grep -rh 'ogit:scope' AdaWorldAPI/OGIT` count tally:

| scope | count | meaning |
| --- | ---: | --- |
| `"NTO"` |   ~590 | Namespace Type Ontology (business domains: Accounting, SalesDistribution, Procurement, SharePoint, ...) |
| `"SGO"` |    ~68 | Semantic Graph Ontology (graph-level concepts: Node, Edge, Timeseries, ...) |
| `"SDF"` |     ~0 in current grep, but the directory exists with Automation + MARS |

NTO is the right scope for the Odoo retarget — Odoo IS a business-domain
ontology covering Accounting / SalesDistribution / Procurement / HR /
Manufacturing / etc.

## Prior-session 2026-05-28 Odoo-mapping work

Eleven files authored this afternoon by a prior `Claude` session under attribution
`"Claude (AdaWorldAPI/lance-graph 3-hop optim)"`:

| file | meaning |
| --- | --- |
| `NTO/Accounting/entities/JournalEntry.ttl`         | maps `account.move` |
| `NTO/Accounting/entities/FiscalJurisdiction.ttl`   | new fiscal-jurisdiction entity |
| `NTO/Accounting/attributes/fiscalCountryCode.ttl`  | promoted 3-hop attr |
| `NTO/Accounting/attributes/commercialPartnerCountryCode.ttl` | promoted 3-hop attr |
| `NTO/Accounting/attributes/iso3166Alpha2.ttl`      | ISO country code |
| `NTO/Accounting/attributes/isEuMember.ttl`         | EU membership flag |
| `NTO/Accounting/attributes/vatRateStandard.ttl`    | standard VAT rate |
| `NTO/Accounting/attributes/productCategoryComplete.ttl` | category attr |
| `NTO/Accounting/verbs/hasFiscalCountry.ttl`        | 3-hop traversal verb |
| `NTO/Accounting/verbs/hasPickingType.ttl`          | stock-related verb |
| `NTO/Accounting/verbs/hasProductCategory.ttl`      | product traversal verb |

The 3-hop doctrine encoded in JournalEntry's description: instead of
forcing the graph consumer to traverse
`entity.company_id.account_fiscal_country_id.code` (5+ Odoo sites)
or `move.commercial_partner_id.country_id`, the leaf attributes are
promoted onto the entity directly AND the intermediate hops remain
accessible via shortcut verbs. Both routes are kept coherent by the
EdgeColumn ComputeRecompute cascade.

## Other relevant Odoo→OGIT mappings present in the repo (not necessarily 2026-05-28)

```
NTO/SalesDistribution/entities/SalesOrder.ttl       (2019-07-10, Marek Meyer) — maps sale.order
NTO/SalesDistribution/entities/SalesOrderItem.ttl
NTO/SalesDistribution/entities/Customer.ttl
NTO/SalesDistribution/entities/Payment.ttl
NTO/SalesDistribution/attributes/salesOrderId.ttl
NTO/Procurement/attributes/productionOrderId.ttl
SGO/sgo/entities/Person.ttl
```

`SalesOrder.ttl` already declares `ogit:allowed (... ogit:triggers
ogit.Procurement:ProductionOrder ... ogit:triggers ogit.Procurement:PurchaseOrder ...)`
— the canonical sale→procurement edge that any Odoo
`sale.order → mrp.production` / `sale.order → purchase.order` mapping
should reuse rather than reinvent.

## Where `bundles_to_ttl.py` diverges

| OGIT meta-DTO requirement | `bundles_to_ttl.py` current output | severity |
| --- | --- | --- |
| Namespace `<http://www.purl.org/ogit/>` | uses `<https://ogit.adaworldapi.com/>` — made-up, doesn't resolve | **CRITICAL** — every IRI is wrong |
| Entities typed `rdfs:subClassOf ogit:Entity` | emits `a ogit:Method` (invented type) | **CRITICAL** |
| `dcterms:valid "start=YYYY-MM-DD;"` per artifact | not emitted | high |
| `ogit:scope "NTO"` per artifact | not emitted | high |
| `ogit:parent ogit:Node` per artifact | not emitted | high |
| `ogit:mandatory-attributes ( ... )` RDF list | not emitted | high |
| `ogit:optional-attributes ( ... )` RDF list | not emitted | high |
| `ogit:indexed-attributes ( ... )` RDF list | not emitted | high |
| `ogit:allowed ( [v t] ... )` for outgoing edges | not emitted (delegation edges are flat triples, not the OGIT `allowed` schema) | medium |
| One file per Entity / Attribute / Verb under `NTO/<Domain>/{entities,attributes,verbs}/` | emits one TTL per family with all methods bundled | medium (different file organization, but content-recoverable) |
| Existing `JournalEntry` reused for `account.move`, `SalesOrder` for `sale.order`, etc. | emits new IRIs `odoo:account_move`, `odoo:sale_order` — fails to reuse the canonical entries | **CRITICAL** — fragments the graph |

## Where the trichotomy puts methods

OGIT has no native `ogit:Method` meta-type. A Python method on
`account_move` is conceptually NOT an Entity (it has no graph-node identity)
— it's a *behavioral specification* attached to the JournalEntry entity. The
three plausible mappings:

1. **As attributes** — `ogit.Accounting:invariantCheck`,
   `ogit.Accounting:computeFormula`, `ogit.Accounting:onchangeRule` (one
   attribute per Odoo decorator family). The method's body source becomes
   the attribute value (a string literal). Loses cross-method graph
   structure.

2. **As entities in a new `NTO/Logic` scope** — `ogit.Logic:ValidationConstraint`,
   `ogit.Logic:ComputeFormula`, `ogit.Logic:OnchangeRule`. Each method
   instance is an Entity carrying body / decorators / axis / transitivity /
   provenance attributes. Verbs like `ogit:Logic:constrains` /
   `ogit:Logic:computes` connect Logic entities to the business Entities
   they govern. Preserves cross-method graph structure (the
   DelegationTuple edges become `ogit:invokes` between Logic entities).

3. **As a new `SDF/Method` scope** — same as (2) but under SDF (Schema
   Definition Format) since "methods" are schema-level not domain-level.
   SDF already houses Automation + MARS so this is the natural home.

**Recommendation:** option (3). Methods are SDF, not NTO. The Odoo extract
populates `NTO/Accounting`, `NTO/SalesDistribution`, ... by REUSING existing
Entity IRIs where they exist (JournalEntry, SalesOrder, Customer) and
declaring SDF-scope Method entities that reference those NTO entities via
`ogit:constrains` / `ogit:computes` / `ogit:onchanges`.

## Migration sketch (for execution under user direction)

The minimal alignment lift on `bundles_to_ttl.py`:

1. **Namespace fix.** Replace `OGIT = "https://ogit.adaworldapi.com"` with
   `OGIT = "http://www.purl.org/ogit"`. Add `@prefix ogit.SDF: <http://www.purl.org/ogit/SDF/> .`
   for the new Method scope.

2. **Family-to-Entity table.** Hand-curated map from Odoo family name to
   canonical OGIT Entity IRI:
   ```
   account_move   -> ogit.Accounting:JournalEntry
   sale_order     -> ogit.SalesDistribution:SalesOrder
   sale_order_line-> ogit.SalesDistribution:SalesOrderItem
   res_partner    -> ogit.SGO:Person   (or new ogit.MasterData:Partner)
   account_payment-> ogit.SalesDistribution:Payment
   ...
   ```
   The first time a family is encountered without a canonical mapping, the
   emitter emits a TODO and skips (vs inventing an `odoo:family_name` IRI).

3. **Method entities.** Emit one entity per method under
   `SDF/Method/entities/<family>__<method>.ttl`:
   ```turtle
   ogit.SDF:JournalEntry__check_invoice_currency_rate
       a rdfs:Class;
       rdfs:subClassOf ogit.SDF:ValidationConstraint;
       rdfs:label "_check_invoice_currency_rate";
       dcterms:description "[body source verbatim]";
       dcterms:valid "start=2026-05-28;";
       dcterms:creator "Claude (AdaWorldAPI/lance-graph ruff-py-dto)";
       ogit:scope "SDF";
       ogit:parent ogit.SDF:ValidationConstraint;
       ogit:mandatory-attributes (
           ogit:id
           ogit.SDF:bodySource
           ogit.SDF:decoratorPattern
       );
       ogit:optional-attributes (
           ogit.SDF:axisClassification
           ogit.SDF:transitivity
           ogit.SDF:causalReference
       );
       ogit:indexed-attributes ( ogit:id ogit.SDF:decoratorPattern );
       ogit:allowed (
           [ ogit.SDF:constrains ogit.Accounting:JournalEntry ]
           [ ogit.SDF:reads ogit.Accounting:fiscalCountryCode ]
           [ ogit.SDF:raises ogit.SDF:ValidationError ]
       ) .
   ```

4. **New attribute declarations** in `SDF/Method/attributes/`:
   `bodySource`, `decoratorPattern`, `axisClassification`, `transitivity`,
   `causalReference`. Each a one-file declaration following the standard
   attribute pattern.

5. **New verb declarations** in `SDF/Method/verbs/`: `constrains`,
   `computes`, `onchanges`, `invokes`, `reads`, `writes`, `raises`,
   `traversesRelation`, `readsEnvironment`. Each a one-file declaration.

6. **Three new meta-types** for the axis classification, in
   `SDF/Method/entities/`: `ValidationConstraint` (for `@api.constrains`),
   `ComputeFormula` (for `@api.depends`), `OnchangeRule` (for `@api.onchange`),
   plus parent `Method`.

The total new-artifact count: ~12 schema files (the meta-types + 5 attrs +
9 verbs) + N method-entity files (~3555 with current harvest).

The schema files are small and stable — they go in OGIT directly. The
method-entity files are bulk and regenerable — they probably stay in
lance-graph's `.claude/odoo/` (where the bundles already live) and only
the schema files PR to AdaWorldAPI/OGIT.

## What this means for the immediate POC

The POC's `bundles_to_ttl.py` proves the EXTRACTION mechanics (matcher works,
delegation edges extracted, TTL parseable). It does NOT yet produce
OGIT-conformant content. Three honest paths forward:

1. **Keep the POC as-is** and document that it's a "shape demonstration", not
   a feed into the OGIT graph. Decoupled from the ontology.

2. **Wire the migration sketch above** as D-RPYDTO-6 (the original "TTL emit
   target" deliverable) and produce OGIT-conformant output. This is what the
   `OWL` REPORT.md section was hinting at when it called the rdflib audit
   "necessary but not sufficient".

3. **Hybrid: keep the in-band POC TTL** for engine-level grammar exercises
   (it's already OWL 2 EL conformant per `cdcbc349`), but ALSO emit a
   parallel `*.ogit.ttl` stream using the OGIT meta-DTO schema for the
   ontology-feed path.

Recommendation: (2). The OWL 2 EL TTL the POC currently emits is a sunk-cost
proof-of-shape; the real downstream consumer is OGIT, and OGIT's schema is
the gating constraint, not OWL 2 DL profile conformance. The grammar slots
(axis / transitivity / causal) all have natural homes as
`ogit:optional-attributes` on Method entities.

Owner direction needed before executing migration. Path (2) is ~1 day of
emitter rewrite + ~12 schema-PR files to OGIT (which we now have a PAT for).
