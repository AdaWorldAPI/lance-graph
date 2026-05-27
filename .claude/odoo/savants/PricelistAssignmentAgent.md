# Savant: PricelistAssignmentAgent  (id 3 · family 0x64 · lane L8)

**Tuple:** kind=Other(PRICELIST_ASSIGNMENT=1) · inference=Revision · semiring=NarsTruth · style=Analytical
**Feeds Reasoner impl:** `OtherReasoner` (dispatches on `Other(code)`; code `PRICELIST_ASSIGNMENT=1`)

> dispatch: `ReasoningKind::Other(code)` with `code` not in {5,6} -> "domain-specific Other(code)
> reasoner" (`examples/savant_dispatch.rs:34,37`). Revision -> `QueryStrategy::BundleInto` via
> `InferenceType::Revision::default_strategy()`. Style Analytical inherited from 0x64 ProductCatalog.
> **Family deviation:** SAVANTS.md proposed `0x63 ProductCatalog`, but `0x63` (=99) is already
> `ogit:MRORepair` in `data/family_registry.ttl`; lance-graph assigns the next free commercial-cluster
> byte **`0x64`** (=100) for ProductCatalog (`odoo_alignment.rs:47-54`, plan D-ODOO-SAV-1). The
> `contract::savants` tuple carries `family: Some(0x64)`; this doc follows the ratified 0x64, not the
> stale 0x63 in SAVANTS.md.

## What it decides (AXIS-B core)
For a partner with **no explicit pricelist property** and **no country-group match**, decide which
pricelist to assign from the fallback waterfall -- a multi-factor business-policy choice, not a closed
lookup (L8 R15 AXIS-B part). The deterministic head of `_get_partner_pricelist_multi` (feature guard,
explicit `specific_property_product_pricelist`, country-group match) stays AXIS-A; the AXIS-B residual
is the tail: when those miss, choose among {no-geo-restriction pricelist, company `ir.config_parameter`
default, global default, first-available active}. Because partner attributes (country, segment) change
over time, the assignment is a **revised** belief -- re-evaluated against business rules on partner
data change, not frozen at onboarding (L8 R15 tuple rationale). Output is a suggested `pricelist_id`
with NARS `(frequency, confidence)`; woa-rs applies it as the default the user can override.

## Deterministic guard (AXIS-A -- stays in woa-rs)
`_get_partner_pricelist_multi` / `_get_country_pricelist_multi` head is deterministic
(`L8-PRODUCT-UOM-PRICELIST.md:546-579`, R15 AXIS-A part; `product_pricelist.py:L333-384`):
(1) `group_product_pricelist` feature disabled -> empty for all; (2) explicit
`specific_property_product_pricelist` active -> use it; (3) group partners by `country_id`, match
`country_group_ids.country_ids`; (4a) search pricelist with `country_group_ids = False` + active +
company. Steps 1-4a are pure data retrieval. Pricelist structure (currency/company/country-group
scoping, `sequence` priority, R8), the recursion guard (R19), and the rule-application engine
(R10-R14) are all deterministic and stay in woa-rs -- the savant only chooses the *fallback* pricelist
when the deterministic head returns nothing.

## Slot 1 -- Evidence (Arrow EvidenceRef)
Two tables. The *partner assignment context*
`EvidenceRef { table: "res_partner.pricelist_context", schema_fingerprint, rows }`
(one row = the partner needing a pricelist):

| column | dtype | signal |
|---|---|---|
| `partner_id` | `Int64` | the partner the pricelist is scoped to |
| `country_id` | `Int64`/nullable | country axis; drives the (failed) country-group match -> what the fallback must cover |
| `company_id` | `Int64` | scopes candidate pricelists and selects the `ir.config_parameter` key |
| `specific_property_pricelist_id` | `Int64`/nullable | explicit property (AXIS-A consumes it; presence here = no delegation) |
| `customer_rank` | `Int64` | segment/maturity hint feeding the revised belief |
| `property_product_pricelist_param` | `Int64`/nullable | resolved `res.partner.property_product_pricelist_{company_id}` config-param candidate |
| `property_product_pricelist_global` | `Int64`/nullable | resolved global `res.partner.property_product_pricelist` config-param candidate |

The *candidate pricelist corpus* `EvidenceRef { table: "product_pricelist", ... }`
(family 0x64 ProductCatalog; `product.pricelist` -> `schema:PriceSpecification`,
`odoo_alignment.rs:286-295`):

| column | dtype | signal |
|---|---|---|
| `pricelist_id` | `Int64` | rule identity (the suggested output) |
| `currency_id` | `Int64` | required; a pricelist whose currency matches the partner/company is preferred |
| `company_id` | `Int64`/nullable | company scoping (NULL = cross-company) |
| `country_group_ids` | `List<Int64>` | geo scope; **empty** is the first fallback tier (no-geo-restriction) |
| `sequence` | `Int32` | selection priority when several apply (default 16; lower wins, R8) |
| `active` | `Boolean` | soft-delete gate; only active pricelists are candidates |

## Slot 2 -- Odoo field -> signal map                 (cite L-doc file:lines)
- `_get_partner_pricelist_multi` waterfall (feature guard / explicit property / country-group / fallback chain) <- `L8-PRODUCT-UOM-PRICELIST.md:546-560` (R15; `product_pricelist.py:L333-384`).
- AXIS-B fallback tiers (no-geo pricelist, `ir.config_parameter` per-company, global param, first-available) <- `L8-PRODUCT-UOM-PRICELIST.md:553-560` (R15 step 4a-4d).
- pricelist structure: `currency_id` (required), `company_id`, `country_group_ids` (M2M geo), `sequence` (default 16, lower=priority), `active` <- `L8-PRODUCT-UOM-PRICELIST.md:242-258` (R8; `product_pricelist.py:L9-65`).
- `group_product_pricelist` feature flag (whole system off when disabled) <- `L8-PRODUCT-UOM-PRICELIST.md:551-552, 707` (R15 step 1 / open question 3).
- `ir.config_parameter` keys `res.partner.property_product_pricelist_{company_id}` and `res.partner.property_product_pricelist` <- `L8-PRODUCT-UOM-PRICELIST.md:553-560, 709` (R15 / open question 4).
- delegation tuple `(Other("PricelistAssignment"), Revision, NarsTruth, Analytical)` + savant seed + family-deviation note <- `L8-PRODUCT-UOM-PRICELIST.md:581-584` (R15 AXIS-B) and `odoo_alignment.rs:47-54` (0x64 not 0x63).

## Slot 3 -- Property-level alignment
N/A -- class-level pivots only; no `owl:equivalentProperty` defined. (Confirmed: `odoo_alignment.rs`
holds only class-level `owl:equivalentClass` rows; zero property IRIs in the repo.) The relevant
class-level pivot is now realized: `product.pricelist` -> `schema:PriceSpecification` on family
**0x64 ProductCatalog** (`odoo_alignment.rs:286-295`) -- note this corrects L8's ontology rows, which
predate D-ODOO-SAV-1 and still list `product.pricelist` as `None`
(`L8-PRODUCT-UOM-PRICELIST.md:31`). `res.partner` -> `fibo:LegalEntity` is the other class-level pivot
touched. No property crosses the seam at decision time -- the fallback ranks pricelist records against
partner country/company scalars. **N/A.**

## Slot 4 -- AXIS-B decision in evidence terms
Let E = the partner assignment context (slot 1) + the candidate pricelist corpus, entered only after
the AXIS-A head returned nothing (no feature-off, no explicit property, no country-group match).

-> Conclusion C = `AssignPricelist(partner_id, pricelist_id)` emitted with NARS
`(frequency, confidence)` where the fallback tiers are evidence sources fused under NarsTruth (not a
hard first-wins cut):
- **frequency** of a candidate rises with: matching the partner's `country`/`company` scope, an
  empty `country_group_ids` (the intended no-geo fallback tier), agreement with the resolved
  `ir.config_parameter` default (per-company beats global), currency match, and a lower `sequence`.
- **frequency** falls for candidates scoped to a different company or carrying a geo restriction the
  partner does not satisfy.
- **confidence** rises when several tiers agree on the same pricelist (e.g. the no-geo default *is* the
  company config-param value) and falls when the only candidate is the bare "first-available active"
  pick -- that tail pick is a low-confidence guess by construction (L8 R15: "the right pricelist for a
  new partner in an edge case is a business judgment"). Revision keeps the belief updatable as partner
  attributes change. Capped by phi-1.

Discriminating features (ranked): explicit-property absence is the gate; then per-company
`ir.config_parameter` match >> empty-`country_group` (no-geo) default > global `ir.config_parameter`
> currency/company scope agreement > `sequence` > first-available. Revision (vs one-shot deduction) is
chosen because partner country/segment can change and the pricelist should be re-evaluated against the
rules rather than frozen (L8 R15 rationale).

## Parity / GoBD notes
Pricelist assignment is commercial pricing, not GoBD double-entry -- it sets the *price* a Vorgang
line starts from, before tax (L8 R12-R14 engine), so it carries no Festschreibung weight. Suggestion-
only per Iron Rule 7: woa-rs applies the assignment as an overridable default; the user can pick a
different pricelist and a manual `specific_property_product_pricelist` always wins (AXIS-A). The
deterministic head (feature guard + explicit property + country-group) must run first so the savant is
only consulted on the genuine fallback edge case. Family note recorded above: 0x64 ProductCatalog (not
the stale 0x63=MRORepair in SAVANTS.md). NEEDS-INPUT: none for the decision; whether a partner
"segment" beyond `customer_rank`/country exists as evidence is not defined in L8 -- if richer
segmentation is wanted it must be sourced (candidate: L9 partner facets), but the fallback is
decidable from the L8-sourced columns alone.
