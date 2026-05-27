# Savant: PricelistRecommender  (id 23 · family 0x81 · lane L6)

**Tuple:** kind=NextBestAction · inference=Synthesis · semiring=NarsTruth · style=Exploratory
**Feeds Reasoner impl:** `NextBestActionReasoner`   (per the impl-per-ReasoningKind decision)

> dispatch: `ReasoningKind::NextBestAction` -> "induce the action with the highest expected value"
> (`examples/savant_dispatch.rs:32`); the *synthesis* facet (bundle evidence across several candidate
> rules into one recommendation) rides the same impl, selecting `QueryStrategy::BundleAcross` via
> `InferenceType::Synthesis::default_strategy()`. Style Exploratory inherited from 0x81
> SmbFoundryInvoice (the commercial-document negotiation angle is Exploratory per L6 S-3, not the
> family's Direct default).

## What it decides (AXIS-B core)
When **multiple pricelist rules are candidates** for a Vorgang line (several `pricelist.item` rows
survive `_get_applicable_rules_domain` and `_is_applicable_for`), recommend **which rule to apply** --
synthesising across the candidate set rather than blindly taking odoo's deterministic "first in sort
order wins". odoo's engine (`_compute_price_rule`) takes the first applicable rule by
`applied_on ASC, min_quantity DESC, categ_id DESC, id DESC`; that ordering is AXIS-A. The AXIS-B core
is the residual judgment when the sort order is *not* obviously right: e.g. a category rule and a
quantity-break rule both apply, or two rules tie on specificity -- which yields the better commercial
outcome (margin vs competitiveness)? Output is a recommended `pricelist_item_id` (and the price it
implies) with NARS `(frequency, confidence)`; woa-rs applies the deterministic first-match unless it
adopts the recommendation behind its guard (price is then frozen once invoiced, S-3).

## Deterministic guard (AXIS-A -- stays in woa-rs)
Rule candidate selection and first-match application are deterministic:
`_get_applicable_rules_domain` (pricelist + category-ancestor + template/variant + date window,
ordered, `L6-SALE-PURCHASE.md` via L8 R10, `product_pricelist.py:L239-264`), `_is_applicable_for`
(min_quantity + applied_on checks, L8 R11, `product_pricelist_item.py:L526-568`), and
`_compute_price_rule` taking the FIRST applicable rule (L8 R12, `product_pricelist.py:L169-236`). The
price arithmetic per rule (fixed/percentage/formula, margins, UoM/currency conversion, L8 R13/R14) is
fully deterministic. Within sale_order_line, `pricelist_id` is already set on the order and
`_compute_price_unit` is deterministic (`L6-SALE-PURCHASE.md:205-246`, S-3 AXIS-A); the savant is
invoked only for the *which-candidate-rule* ambiguity the first-match sort does not cleanly resolve
(`L6-SALE-PURCHASE.md:246`, S-3 Axis-2).

## Slot 1 -- Evidence (Arrow EvidenceRef)
Two tables. The *pricing request context*
`EvidenceRef { table: "sale_order_line.pricing_context", schema_fingerprint, rows }`
(one row = the line being priced):

| column | dtype | signal |
|---|---|---|
| `order_line_id` | `Int64` | the line the recommendation is scoped to |
| `pricelist_id` | `Int64` | the active pricelist (candidates are drawn from it) |
| `product_id` | `Int64` | variant axis for `_is_applicable_for` |
| `product_tmpl_id` | `Int64` | template axis |
| `product_categ_id` | `Int64` | category axis (ancestor match via `parent_path`) |
| `qty_in_product_uom` | `Float64` | converted order qty (already in product UoM) -- gates `min_quantity` breaks |
| `currency_id` | `Int64` | order currency (rules may convert from another base) |
| `date_order` | `Date32` | validity-window axis (`date_start <= date <= date_end`) |

The *candidate rule corpus* `EvidenceRef { table: "product_pricelist_item", ... }`
(the applicable rules to synthesise across; `product.pricelist.item` ->
`schema:UnitPriceSpecification`, family 0x64, `odoo_alignment.rs:296-305`):

| column | dtype | signal |
|---|---|---|
| `pricelist_item_id` | `Int64` | rule identity (the recommended output) |
| `applied_on` | `Utf8` (`0_product_variant\|1_product\|2_product_category\|3_global`) | specificity tier (primary sort key) |
| `min_quantity` | `Float64` | quantity break (secondary sort, DESC); a higher break that the line meets is "more specific" |
| `compute_price` | `Utf8` (`fixed\|percentage\|formula`) | what price the rule yields |
| `base` | `Utf8` (`list_price\|standard_price\|pricelist`) | base the discount/formula applies to |
| `resulting_price` | `Decimal128` | the price this candidate would produce (computed by the AXIS-A engine per candidate) |
| `categ_id` | `Int64`/nullable | category constraint (tertiary sort, DESC) |
| `date_start` / `date_end` | `Date32`/nullable | validity window |

## Slot 2 -- Odoo field -> signal map                 (cite L-doc file:lines)
- pricelist-rule selection is AXIS-A within the model (pricelist already set) but a "best pricelist / next-action on pricing" recommendation is the delegated tuple <- `L6-SALE-PURCHASE.md:246` (S-3 Axis-2) and `L6-SALE-PURCHASE.md:386-387` (S-5 Axis-2 note, same Exploratory document-basin character).
- `_compute_price_unit` guards (manual-price stick, `qty_invoiced>0` freeze) wrapping the price <- `L6-SALE-PURCHASE.md:205-225` (S-3; `sale_order_line.py:L586-633`).
- candidate fetch `_get_applicable_rules_domain` (ordered by applied_on ASC, min_quantity DESC, categ_id DESC, id DESC) <- `L8-PRODUCT-UOM-PRICELIST.md:312-339` (R10; `product_pricelist.py:L239-264`).
- `_is_applicable_for` per-product applicability (min_quantity gate, applied_on tiers) <- `L8-PRODUCT-UOM-PRICELIST.md:343-370` (R11; `product_pricelist_item.py:L526-568`).
- `_compute_price_rule` first-applicable-rule selection + empty-rule base fallback <- `L8-PRODUCT-UOM-PRICELIST.md:374-411` (R12; `product_pricelist.py:L169-236`).
- per-rule price computation (fixed/percentage/formula, margin clamps) producing `resulting_price` <- `L8-PRODUCT-UOM-PRICELIST.md:415-501` (R13; `product_pricelist_item.py:L570-626`).
- delegation tuple `(NextBestAction, Synthesis, NarsTruth, Exploratory)` <- `L6-SALE-PURCHASE.md:246` (S-3 Axis-2).

## Slot 3 -- Property-level alignment
N/A -- class-level pivots only; no `owl:equivalentProperty` defined. (Confirmed: `odoo_alignment.rs`
holds only class-level `owl:equivalentClass` rows; zero property IRIs in the repo.) Class-level pivots
touched: `product.pricelist.item` -> `schema:UnitPriceSpecification` and `product.pricelist` ->
`schema:PriceSpecification` (family 0x64, `odoo_alignment.rs:286-305`); `sale.order(.line)` is
**unmapped** (`None`; L6 proposes `ubl:Order`/`ubl:OrderLine` -> 0x81 but no row exists yet,
`L6-SALE-PURCHASE.md:761-766`). No property crosses the FIBO/SKR/ZUGFeRD seam at decision time -- the
recommendation ranks pricelist-item records against the line's product/qty/date scalars. **N/A.**

## Slot 4 -- AXIS-B decision in evidence terms
Let E = the pricing request context + the set of *applicable* candidate rules (slot 1), each carrying
its `resulting_price`. Synthesis bundles the candidates across into one recommendation.

-> Conclusion C = `RecommendPricelistRule(order_line_id, pricelist_item_id)` (implying its
`resulting_price`) emitted with NARS `(frequency, confidence)` where:
- candidates are first ordered by the deterministic specificity (applied_on, min_quantity, categ_id,
  id). Synthesis does not merely pick the head: it weighs the *spread* of resulting prices and the
  specificity gaps. NarsTruth fuses agreement -- if the two most specific rules yield the same price,
  that price is strongly supported.
- **frequency** of a recommended rule rises with: higher specificity (`applied_on` variant > product >
  category > global), a `min_quantity` break the line actually meets, and being inside its date window.
- **frequency** is tempered when a more-specific rule yields a price far off the cluster (a possible
  mis-configured rule) -- Synthesis can prefer the better-supported rule over the strictly-first one,
  which is exactly the residual judgment delegated here.
- **confidence** rises when candidates converge (few rules, agreeing prices) and falls when many rules
  with divergent prices all apply (the choice is genuinely ambiguous / negotiation-shaped, matching the
  Exploratory cluster). Capped by phi-1.

Discriminating features (ranked): `applied_on` specificity >> `min_quantity` break met > date-window
fit > `categ_id` ancestor distance > spread of `resulting_price` across candidates. Synthesis (vs
deduction) is chosen because the answer bundles several candidate rules into a recommendation rather
than reading one key -- the "which of several applicable rules" question is a cross-candidate fusion
(L6 S-3 rationale; the deterministic first-match stays the safe default).

## Parity / GoBD notes
Pricelist recommendation sets a Vorgang line's pre-tax price; it is commercial, not GoBD double-entry,
so it carries no Festschreibung weight by itself. But once any quantity is invoiced, `_compute_price_unit`
**freezes** the price (`qty_invoiced > 0` guard, S-3; woa-rs GoBD lock via `src/gobd.rs`,
`L6-SALE-PURCHASE.md:822`) -- the savant must never re-recommend a price on a line that is already
(partly) invoiced. Suggestion-only per Iron Rule 7: woa-rs applies odoo's deterministic first-match by
default and only adopts the synthesised recommendation behind its guard; a manually-edited price
(`technical_price_unit != price_unit`, S-3) always sticks and is never overridden. Multi-currency
candidate prices are converted by the deterministic engine (L8 R14) before the savant compares them.
