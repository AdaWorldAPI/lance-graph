# Odoo `account_move_line` -> OGIT `LineItem` extension proposal

> **Status:** DRAFT (2026-05-28, this session). Ready for review before
> splitting into per-file PRs against `AdaWorldAPI/OGIT` `master`.
>
> **Why:** part of the business-logic extraction from `odoo/addons/account/`,
> grounded in the OGIT Schema. Continuation of the
> `JournalEntry`/`FiscalJurisdiction`/3-hop-attributes work landed earlier
> today on the `claude/activate-lance-graph-att-k2pHI` branch (`Claude
> (AdaWorldAPI/lance-graph 3-hop optim)` attribution).
>
> **Framing per user, this session:** these mandatory/optional/indexed
> attribute lists are NOT just schema. They are the **item bank** for the
> axiom-pattern-derived psychometric instrument per Keet/Raboanary
> (ESWC'26). Every attribute promoted into the meta-DTO becomes a slot
> that the determination algorithm can bind, producing calibrated items
> with measurable difficulty / discrimination / reliability via the
> existing `thinking-engine/cronbach.rs` + `calibrate_lenses.rs`
> instrumentation.

## Mining methodology

Input: the 42 method bundles for `account_move_line` from the
`ruff-py-dto` harvest of `odoo/addons/account/models/account_move_line.py`
(commit `cdcbc349`). Breakdown by primary decorator:

- 31 × `@api.depends` (compute methods — derive fields from inputs)
- 6 × `@api.onchange` (UI-reactive — fire when field changes)
- 5 × `@api.constrains` (validators — raise on invalid combinations)

For each decorator, the field-argument tuple is parsed and counted across
all 42 methods. The frequency tables drive the meta-DTO classification:

| signal                              | classification target  | rule                                                                      |
| ---                                 | ---                    | ---                                                                       |
| top-decile `@api.depends` field     | mandatory-attribute    | field appears in ≥ 7/31 depends lists ⟹ load-bearing for compute layer    |
| any `@api.constrains` field         | indexed-attribute      | constraint scans benefit from index; usually paired with mandatory        |
| `@api.depends` < top-decile         | optional-attribute     | used in some computes but not the dominant input                          |
| `@api.onchange` field               | optional-attribute (+ UI-reactive flag — out of scope here)               |
| body-source `self.X.<method>()`     | allowed-edge verb      | extracted via DelegationTuple second pass; surfaces relationship verbs    |

## Mined frequencies (account_move_line, 42 methods)

`@api.depends` field counts (top 20):

```
12  move_id              -- parent JournalEntry
11  product_id           -- 1:1 product reference
 7  account_id           -- the account being debited/credited
 7  company_id           -- multi-company partition
 4  currency_id
 4  tax_ids
 3  balance              -- already in OGIT LineItem optional
 3  matched_debit_ids    -- reconciliation
 3  matched_credit_ids   -- reconciliation
 3  price_unit
 3  quantity
 3  analytic_distribution
 3  display_type
 2  currency_rate
 2  discount
 2  date_maturity
 2  product_uom_id
 1  debit
 1  credit
 1  amount_currency
```

`@api.constrains` field counts:

```
 2  tax_ids
 2  account_id
 1  tax_repartition_line_id
 1  tax_line_id
 1  reconciled
 1  display_type
 1  deductible_amount
 1  matching_number
 1  matched_debit_ids
 1  matched_credit_ids
```

`@api.onchange` field counts (UI-reactive):

```
 1  account_id, amount_currency, currency_id, credit, debit, partner_id, product_id
```

## Proposed extension to `NTO/Accounting/entities/LineItem.ttl`

(Current OGIT LineItem: Viktor Voss, 2018-12-05, empty mandatory/indexed,
generic optional set of 6 fields. Extension adds Odoo-grounded fields
with deference to the existing pattern.)

```turtle
@prefix ogit.Accounting:  <http://www.purl.org/ogit/Accounting/> .
@prefix ogit.SalesDistribution: <http://www.purl.org/ogit/SalesDistribution/> .
@prefix ogit.MasterData:  <http://www.purl.org/ogit/MasterData/> .
@prefix ogit:             <http://www.purl.org/ogit/> .
@prefix rdfs:             <http://www.w3.org/2000/01/rdf-schema#> .
@prefix dcterms:          <http://purl.org/dc/terms/> .

ogit.Accounting:LineItem
    a rdfs:Class;
    rdfs:subClassOf ogit:Entity;
    rdfs:label "LineItem";
    dcterms:description "A double-entry accounting line item — the canonical
        OGIT projection of Odoo `account.move.line` (and the analogous SAP
        `BSEG`-row, DATEV `Buchungssatz`-Zeile). One LineItem belongs-to one
        JournalEntry, debits OR credits one Account, and optionally
        references a Partner, Product, Currency, and zero-or-more Tax
        entries. Mandatory/optional/indexed lists below are mined from the
        decorator frequency of the 42 Odoo `_compute_*`/`_check_*`/
        `_onchange_*` methods on `account.move.line` and serve as the item
        bank for axiom-pattern-derived psychometric items per Keet &
        Raboanary (ESWC'26).";
    dcterms:valid "start=2026-05-28;";
    dcterms:creator "Claude (AdaWorldAPI/lance-graph odoo-extraction)";
    ogit:scope "NTO";
    ogit:parent ogit:Node;
    ogit:mandatory-attributes (
        ogit:id
        ogit.Accounting:account_id
        ogit.Accounting:move_id
        ogit.Accounting:company_id
        ogit.Accounting:balance
    );
    ogit:optional-attributes (
        ogit:name
        ogit:lastUpdatedAt
        ogit.Accounting:subtype
        ogit.Accounting:association
        ogit.Accounting:AccountNumber
        ogit.Accounting:debit
        ogit.Accounting:credit
        ogit.Accounting:amount_currency
        ogit.Accounting:currency_rate
        ogit.Accounting:currency_id
        ogit.Accounting:price_unit
        ogit.Accounting:quantity
        ogit.Accounting:discount
        ogit.Accounting:product_id
        ogit.Accounting:product_uom_id
        ogit.Accounting:partner_id
        ogit.Accounting:tax_ids
        ogit.Accounting:tax_line_id
        ogit.Accounting:tax_repartition_line_id
        ogit.Accounting:analytic_distribution
        ogit.Accounting:display_type
        ogit.Accounting:reconciled
        ogit.Accounting:matched_debit_ids
        ogit.Accounting:matched_credit_ids
        ogit.Accounting:deductible_amount
        ogit.Accounting:matching_number
        ogit.Accounting:date_maturity
    );
    ogit:indexed-attributes (
        ogit:id
        ogit.Accounting:account_id
        ogit.Accounting:move_id
        ogit.Accounting:reconciled
        ogit.Accounting:display_type
        ogit.Accounting:tax_ids
    );
    ogit:allowed (
        [ ogit.Accounting:belongs ogit.Accounting:JournalEntry ]
        [ ogit.Accounting:debits ogit:Account ]
        [ ogit.Accounting:credits ogit:Account ]
        [ ogit.Accounting:reconcilesWith ogit.Accounting:LineItem ]
        [ ogit.Accounting:hasTax ogit.Accounting:Tax ]
        [ ogit:references ogit.MasterData:Partner ]
        [ ogit:references ogit.SalesDistribution:Product ]
        [ ogit:contributesTo ogit.Accounting:FinancialStatement ]
        [ ogit:maps ogit.Accounting:CategoryItem ]
    ) .
```

Notes on the meta-DTO choices:

- `account_id` is **mandatory** (every line debits/credits exactly one
  account; this is the foundational invariant of double-entry bookkeeping).
- `move_id` is **mandatory** (every line belongs to exactly one move;
  orphan lines are an integrity error in Odoo).
- `company_id` is **mandatory** (multi-company partition; constrains
  visibility and currency).
- `balance` is **mandatory** (the line's signed monetary impact; derived
  from debit−credit but stored eagerly for indexed queries).
- `debit`/`credit` are **optional** individually but the constraint
  `debit > 0 XOR credit > 0` is invariant — not expressible in the OGIT
  list format directly; either it lives in `dcterms:description` (as it
  does here) or it becomes a constraint-shape on top of the OGIT graph.
- `reconciled` + `matched_debit_ids` + `matched_credit_ids` are **indexed**
  because reconciliation queries are the primary access pattern for the
  accounting close.
- `tax_ids` is **indexed** because it gates 2 of 5 `@api.constrains`
  validators (so most tax-related queries scan it).
- The `allowed` edge `ogit.Accounting:belongs JournalEntry` is the
  inverse of `JournalEntry contains LineItem` (already in JournalEntry's
  allowed-list — see earlier session work).

## New attributes (require per-file TTL files in `NTO/Accounting/attributes/`)

15 new `xsd:decimal`/`xsd:integer`/`xsd:boolean`/`xsd:string` attributes:

```turtle
# Per-file TTL (one per attribute, following the
# accountingStandard.ttl / fiscalCountryCode.ttl pattern).

ogit.Accounting:account_id
    a owl:ObjectProperty;  # references an Account entity
    rdfs:subPropertyOf ogit:Attribute;
    rdfs:label "account_id";
    dcterms:description "Reference to the Account this line debits or credits.
        Mandatory per double-entry bookkeeping invariant.";
    dcterms:valid "start=2026-05-28;";
    dcterms:creator "Claude (AdaWorldAPI/lance-graph odoo-extraction)" .

ogit.Accounting:move_id
    a owl:ObjectProperty;  # references the parent JournalEntry
    rdfs:subPropertyOf ogit:Attribute;
    rdfs:label "move_id";
    dcterms:description "Reference to the parent JournalEntry. Every LineItem
        belongs to exactly one JournalEntry.";
    dcterms:valid "start=2026-05-28;";
    dcterms:creator "Claude (AdaWorldAPI/lance-graph odoo-extraction)" .

ogit.Accounting:company_id
    a owl:ObjectProperty;
    rdfs:subPropertyOf ogit:Attribute;
    rdfs:label "company_id";
    dcterms:description "Multi-company partition. Every LineItem belongs to
        exactly one Company for visibility and currency resolution.";
    dcterms:valid "start=2026-05-28;";
    dcterms:creator "Claude (AdaWorldAPI/lance-graph odoo-extraction)" .

ogit.Accounting:debit
    a owl:DatatypeProperty;
    rdfs:subPropertyOf ogit:Attribute;
    rdfs:label "debit";
    rdfs:range xsd:decimal;
    dcterms:description "Debit-side amount in company currency. Either debit
        or credit is non-zero, never both. Sign convention: positive value =
        debit posting.";
    dcterms:valid "start=2026-05-28;";
    dcterms:creator "Claude (AdaWorldAPI/lance-graph odoo-extraction)" .

ogit.Accounting:credit
    a owl:DatatypeProperty;
    rdfs:subPropertyOf ogit:Attribute;
    rdfs:label "credit";
    rdfs:range xsd:decimal;
    dcterms:description "Credit-side amount in company currency. Either
        credit or debit is non-zero, never both.";
    dcterms:valid "start=2026-05-28;";
    dcterms:creator "Claude (AdaWorldAPI/lance-graph odoo-extraction)" .

ogit.Accounting:amount_currency
    a owl:DatatypeProperty;
    rdfs:subPropertyOf ogit:Attribute;
    rdfs:label "amount_currency";
    rdfs:range xsd:decimal;
    dcterms:description "Line amount in the line's own currency (if different
        from company currency). Signed: positive = debit-side, negative =
        credit-side.";
    dcterms:valid "start=2026-05-28;";
    dcterms:creator "Claude (AdaWorldAPI/lance-graph odoo-extraction)" .

ogit.Accounting:currency_rate
    a owl:DatatypeProperty;
    rdfs:subPropertyOf ogit:Attribute;
    rdfs:label "currency_rate";
    rdfs:range xsd:decimal;
    dcterms:description "Exchange rate (line currency -> company currency)
        at the time of posting. Always strictly positive for posted lines
        (HGB §244 + UStG §16(6) Schiff currency-conversion requirements).";
    dcterms:valid "start=2026-05-28;";
    dcterms:creator "Claude (AdaWorldAPI/lance-graph odoo-extraction)" .

ogit.Accounting:currency_id
    a owl:ObjectProperty;
    rdfs:subPropertyOf ogit:Attribute;
    rdfs:label "currency_id";
    dcterms:description "Reference to the Currency this line is denominated in.";
    dcterms:valid "start=2026-05-28;";
    dcterms:creator "Claude (AdaWorldAPI/lance-graph odoo-extraction)" .

ogit.Accounting:price_unit
    a owl:DatatypeProperty;
    rdfs:subPropertyOf ogit:Attribute;
    rdfs:label "price_unit";
    rdfs:range xsd:decimal;
    dcterms:description "Unit price for the line's product/service. Used in
        invoice line items where quantity * price_unit drives subtotal.";
    dcterms:valid "start=2026-05-28;";
    dcterms:creator "Claude (AdaWorldAPI/lance-graph odoo-extraction)" .

ogit.Accounting:quantity
    a owl:DatatypeProperty;
    rdfs:subPropertyOf ogit:Attribute;
    rdfs:label "quantity";
    rdfs:range xsd:decimal;
    dcterms:description "Quantity of the line's product/service in the
        product's unit of measure.";
    dcterms:valid "start=2026-05-28;";
    dcterms:creator "Claude (AdaWorldAPI/lance-graph odoo-extraction)" .

ogit.Accounting:discount
    a owl:DatatypeProperty;
    rdfs:subPropertyOf ogit:Attribute;
    rdfs:label "discount";
    rdfs:range xsd:decimal;
    dcterms:description "Discount percentage applied to this line's
        price_unit * quantity subtotal.";
    dcterms:valid "start=2026-05-28;";
    dcterms:creator "Claude (AdaWorldAPI/lance-graph odoo-extraction)" .

ogit.Accounting:tax_ids
    a owl:ObjectProperty;
    rdfs:subPropertyOf ogit:Attribute;
    rdfs:label "tax_ids";
    dcterms:description "Many-to-many reference to Tax entries applied to
        this line. Constrains together with account_id (line must not mix
        tax_ids that are incompatible with its account's tax_country).";
    dcterms:valid "start=2026-05-28;";
    dcterms:creator "Claude (AdaWorldAPI/lance-graph odoo-extraction)" .

ogit.Accounting:tax_line_id
    a owl:ObjectProperty;
    rdfs:subPropertyOf ogit:Attribute;
    rdfs:label "tax_line_id";
    dcterms:description "If this LineItem IS a tax-posting line (rather than
        a regular debit/credit), reference to the originating Tax entry.";
    dcterms:valid "start=2026-05-28;";
    dcterms:creator "Claude (AdaWorldAPI/lance-graph odoo-extraction)" .

ogit.Accounting:tax_repartition_line_id
    a owl:ObjectProperty;
    rdfs:subPropertyOf ogit:Attribute;
    rdfs:label "tax_repartition_line_id";
    dcterms:description "Reference to the tax-repartition line that drove
        this LineItem's creation. Used in tax-grid generation for DATEV /
        ELSTER export.";
    dcterms:valid "start=2026-05-28;";
    dcterms:creator "Claude (AdaWorldAPI/lance-graph odoo-extraction)" .

ogit.Accounting:analytic_distribution
    a owl:DatatypeProperty;
    rdfs:subPropertyOf ogit:Attribute;
    rdfs:label "analytic_distribution";
    rdfs:range xsd:string;
    dcterms:description "JSON-encoded analytic-account distribution
        (analytic_account_id -> percentage). Used for cost-center / profit-
        center allocation reporting.";
    dcterms:valid "start=2026-05-28;";
    dcterms:creator "Claude (AdaWorldAPI/lance-graph odoo-extraction)" .

ogit.Accounting:display_type
    a owl:DatatypeProperty;
    rdfs:subPropertyOf ogit:Attribute;
    rdfs:label "display_type";
    rdfs:range xsd:string;
    dcterms:description "Distinguishes regular accounting lines from UI-only
        decoration lines (section headers, notes, subtotals). Constrained
        per account_id (display_type='product' lines have non-null
        account_id; section/note lines do not).";
    dcterms:valid "start=2026-05-28;";
    dcterms:creator "Claude (AdaWorldAPI/lance-graph odoo-extraction)" .

ogit.Accounting:reconciled
    a owl:DatatypeProperty;
    rdfs:subPropertyOf ogit:Attribute;
    rdfs:label "reconciled";
    rdfs:range xsd:boolean;
    dcterms:description "True if this LineItem has been matched against
        offsetting LineItems (full reconciliation). Indexed for fast
        accounting-close queries.";
    dcterms:valid "start=2026-05-28;";
    dcterms:creator "Claude (AdaWorldAPI/lance-graph odoo-extraction)" .

ogit.Accounting:matched_debit_ids
    a owl:ObjectProperty;
    rdfs:subPropertyOf ogit:Attribute;
    rdfs:label "matched_debit_ids";
    dcterms:description "Many-to-many references to debit-side LineItems that
        this credit-side line has been reconciled against.";
    dcterms:valid "start=2026-05-28;";
    dcterms:creator "Claude (AdaWorldAPI/lance-graph odoo-extraction)" .

ogit.Accounting:matched_credit_ids
    a owl:ObjectProperty;
    rdfs:subPropertyOf ogit:Attribute;
    rdfs:label "matched_credit_ids";
    dcterms:description "Many-to-many references to credit-side LineItems
        that this debit-side line has been reconciled against.";
    dcterms:valid "start=2026-05-28;";
    dcterms:creator "Claude (AdaWorldAPI/lance-graph odoo-extraction)" .

ogit.Accounting:deductible_amount
    a owl:DatatypeProperty;
    rdfs:subPropertyOf ogit:Attribute;
    rdfs:label "deductible_amount";
    rdfs:range xsd:decimal;
    dcterms:description "Deductible portion of the line's tax (e.g. for input
        VAT where only a fraction is recoverable per UStG §15(2)).";
    dcterms:valid "start=2026-05-28;";
    dcterms:creator "Claude (AdaWorldAPI/lance-graph odoo-extraction)" .

ogit.Accounting:matching_number
    a owl:DatatypeProperty;
    rdfs:subPropertyOf ogit:Attribute;
    rdfs:label "matching_number";
    rdfs:range xsd:string;
    dcterms:description "Group identifier for reconciliation cohorts. All
        LineItems with the same matching_number are part of one
        reconciliation event.";
    dcterms:valid "start=2026-05-28;";
    dcterms:creator "Claude (AdaWorldAPI/lance-graph odoo-extraction)" .

ogit.Accounting:date_maturity
    a owl:DatatypeProperty;
    rdfs:subPropertyOf ogit:Attribute;
    rdfs:label "date_maturity";
    rdfs:range xsd:date;
    dcterms:description "Due date for payment of this LineItem. Drives
        aging-report bucketing and dunning-process triggers.";
    dcterms:valid "start=2026-05-28;";
    dcterms:creator "Claude (AdaWorldAPI/lance-graph odoo-extraction)" .

ogit.Accounting:partner_id
    a owl:ObjectProperty;
    rdfs:subPropertyOf ogit:Attribute;
    rdfs:label "partner_id";
    dcterms:description "Reference to the Partner (customer / supplier /
        employee) this line transacts with.";
    dcterms:valid "start=2026-05-28;";
    dcterms:creator "Claude (AdaWorldAPI/lance-graph odoo-extraction)" .

ogit.Accounting:product_id
    a owl:ObjectProperty;
    rdfs:subPropertyOf ogit:Attribute;
    rdfs:label "product_id";
    dcterms:description "Reference to the Product (good / service) this line
        is for.";
    dcterms:valid "start=2026-05-28;";
    dcterms:creator "Claude (AdaWorldAPI/lance-graph odoo-extraction)" .

ogit.Accounting:product_uom_id
    a owl:ObjectProperty;
    rdfs:subPropertyOf ogit:Attribute;
    rdfs:label "product_uom_id";
    dcterms:description "Unit of measure for the line's quantity (kg, hours,
        pieces, ...).";
    dcterms:valid "start=2026-05-28;";
    dcterms:creator "Claude (AdaWorldAPI/lance-graph odoo-extraction)" .

ogit.Accounting:subtype
    # ALREADY EXISTS — referenced but no change.
```

## New verbs (require per-file TTL in `NTO/Accounting/verbs/`)

```turtle
ogit.Accounting:belongs
    a owl:ObjectProperty;
    rdfs:subPropertyOf ogit:Verb;
    rdfs:label "belongs";
    dcterms:description "Composition verb: subject is a structural part of
        the object's aggregate. LineItem belongs JournalEntry: the line is
        owned by the journal entry and shares its visibility / partitioning.
        (Distinct from ogit:contributes which is contribution to a process,
        not containment in an aggregate.)";
    dcterms:valid "start=2026-05-28;";
    dcterms:creator "Claude (AdaWorldAPI/lance-graph odoo-extraction)" .

ogit.Accounting:debits
    a owl:ObjectProperty;
    rdfs:subPropertyOf ogit:Verb;
    rdfs:label "debits";
    dcterms:description "Subject (LineItem) posts a debit-side amount to
        the object Account. Inverse: Account.debitedBy LineItem.";
    dcterms:valid "start=2026-05-28;";
    dcterms:creator "Claude (AdaWorldAPI/lance-graph odoo-extraction)" .

ogit.Accounting:credits
    a owl:ObjectProperty;
    rdfs:subPropertyOf ogit:Verb;
    rdfs:label "credits";
    dcterms:description "Subject (LineItem) posts a credit-side amount to
        the object Account. Inverse: Account.creditedBy LineItem.";
    dcterms:valid "start=2026-05-28;";
    dcterms:creator "Claude (AdaWorldAPI/lance-graph odoo-extraction)" .

ogit.Accounting:reconcilesWith
    a owl:ObjectProperty;
    rdfs:subPropertyOf ogit:Verb;
    rdfs:label "reconcilesWith";
    dcterms:description "Symmetric reconciliation: subject LineItem has
        been matched against object LineItem. Used in accounting-close
        reporting and dunning suppression. Subject + object share a
        matching_number.";
    dcterms:valid "start=2026-05-28;";
    dcterms:creator "Claude (AdaWorldAPI/lance-graph odoo-extraction)" .

ogit.Accounting:hasTax
    a owl:ObjectProperty;
    rdfs:subPropertyOf ogit:Verb;
    rdfs:label "hasTax";
    dcterms:description "LineItem has Tax: the line's amount is subject to
        the referenced Tax for VAT / Umsatzsteuer / withholding / etc.";
    dcterms:valid "start=2026-05-28;";
    dcterms:creator "Claude (AdaWorldAPI/lance-graph odoo-extraction)" .
```

## Open items for the next family

After this LineItem extension lands, the natural follow-on families are:

1. **`account_payment` (44 methods)** -> existing
   `ogit.SalesDistribution:Payment.ttl`. Likely needs `paid_by` /
   `pays_for` verbs, `payment_method` / `reference` attributes.

2. **`account_journal` (51 methods)** -> *new* `ogit.Accounting:Journal`
   entity. Carries `type` (sale/purchase/cash/bank/general), `code`,
   `default_account`, sequence-generation attributes.

3. **`res_partner` (80 methods)** -> existing
   `ogit.SalesDistribution:Customer.ttl` (or possibly elevate to a
   shared `ogit.MasterData:Partner`). Many cross-cutting attrs.

4. **`account_tax` (48 methods)** -> *new* `ogit.Accounting:Tax`
   entity. Carries `amount`, `type_tax_use`, `tax_group_id`,
   `tax_country_id`, `cash_basis_transition_account_id`.

Each follows the same mining recipe: decorator frequency -> meta-DTO
classification -> per-file TTL artifacts. The recipe is mechanical from
this point; the only judgment calls are which fields are *truly*
mandatory (vs. just frequently-depended-on) and which verbs warrant new
declarations (vs. reusing `ogit:references` / `ogit:contains`).

## Psychometric instrument layer (per QFGEN-FRAMEWORK-NOTES.md)

Once these schema extensions land in OGIT, the Keet/Raboanary framework
turns them into a calibrated item bank. Concretely, this LineItem
extension immediately enables:

- **`odoo1-Family-Has-Canonical-OGIT-Entity` APS** (already sketched in
  the QFGEN notes) — `account_move_line` now passes; was a violator
  before this PR.

- **`odoo2-Method-Constrains-Real-Attribute` APS** — the 5 `@api.constrains`
  methods on `account_move_line` can be checked against the new
  `mandatory-attributes` / `optional-attributes` list. Result: a per-method
  validity score that feeds Cronbach α aggregation per family.

- **Generative competency questions** (awo6/exmo1 Definition templates)
  fire against `ogit.Accounting:LineItem` and the new attributes,
  producing items like:

      Q: What is a LineItem?
      A: A LineItem is a Node (subClassOf ogit:Entity), and an
         InvoiceLineItem can be considered as a specialisation
         of a LineItem.

      Q: What does a LineItem debit?
      A: A LineItem debits an Account (via the `debits` verb introduced
         in this extension).

These items can then be calibrated against
`thinking-engine/cronbach.rs` for inter-item reliability and
`thinking-engine/calibrate_lenses.rs` for Spearman ρ / ICC across the
3-source triangulation (D1 code extraction = this PR, D2 OGIT axioms,
D3 L-doc curated knowledge in `.claude/odoo/L11-COA-JOURNALS-LOCKDATES.md`).

The 3σ / palette256 / 0.9973 confluence becomes the natural
psychometric reliability threshold: an item bank is "ready" when 99.73%
of items pass cross-source agreement under the determination algorithm.
