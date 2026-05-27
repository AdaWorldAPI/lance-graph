# Savant: FiscalPositionResolver  (id 1 · family 0x80 · lane L9)

**Tuple:** kind=CustomerCategory · inference=Deduction · semiring=NarsTruth · style=Analytical
**Feeds Reasoner impl:** `CustomerCategoryReasoner`   (per the impl-per-ReasoningKind decision)

> dispatch: `ReasoningKind::CustomerCategory` -> "classify against the family codebook (deductive
> lookup)" (`examples/savant_dispatch.rs:29`). Deduction -> `QueryStrategy::CamExact`. Style Analytical
> inherited from 0x80 SmbFoundryCustomer (the fiscal sub-path is Analytical per SAVANTS.md, not the
> family's Empathic default).

## What it decides (AXIS-B core)
Given a partner whose fiscal position is **not** pinned by a manual override and whose country is
known, decide **which `auto_apply=True` fiscal position (tax/account mapping) applies** -- the first
candidate in a priority-ordered list whose five validation predicates (vat_required, zip_range,
state, country, country_group) all pass. The choice is not a single-key lookup: it is a
specificity-ranked search (company-specific positions sort before less-specific; `sequence` breaks
ties) over a variable-length rule set, where each predicate is a partial match (L9 R8 AXIS-B). Output
is a suggested `fiscal_position_id` with NARS `(frequency, confidence)`; woa-rs applies it only as the
auto-detected default, never overriding a manual `property_account_position_id`.

## Deterministic guard (AXIS-A -- stays in woa-rs)
The precedence shell is deterministic (`_get_fiscal_position`, `L9-PARTNER-FISCALPOS.md:211-259`,
R8 AXIS-A part; `partner.py:246-279`): (1) falsy partner -> empty; (2) `intra_eu`/`vat_exclusion`
VAT-prefix computation; (3) delivery-vs-invoicing address selection; (4) **manual override wins** --
`delivery.property_account_position_id or partner.property_account_position_id` returns immediately
(L9 R3, `partner.py:547-550`); (5) no `country_id` -> empty. The savant runs only on the residual:
country present, no manual override, search over `auto_apply` positions. The five predicates
themselves (R8, `partner.py:215-244`) and the zip lexicographic-pad normalisation (R11,
`partner.py:181-206`) are deterministic guards; the savant weights *which rule fires* when several
could match.

## Slot 1 -- Evidence (Arrow EvidenceRef)
Two correlated tables. The *partner classification context*
`EvidenceRef { table: "res_partner.fiscalpos_context", schema_fingerprint, rows }`
(one row = the partner/delivery address being classified):

| column | dtype | signal |
|---|---|---|
| `partner_id` | `Int64` | the partner the fiscal position is scoped to |
| `effective_delivery_id` | `Int64` | resolved delivery-vs-invoicing address (AXIS-A picks it; the predicates read its country/state/zip) |
| `country_id` | `Int64` | the `country` predicate axis (must be present or AXIS-A returns empty) |
| `state_id` | `Int64`/nullable | the `state` predicate axis (`partner.state_id in fpos.state_ids`) |
| `zip` | `Utf8`/nullable | the `zip_range` predicate axis (lexicographic `zip_from <= zip <= zip_to`) |
| `vat` | `Utf8`/nullable | the `vat_required` predicate axis (`_get_vat_required_valid` = `bool(partner.vat)` in community) |
| `company_id` | `Int64` | scopes candidates to this company (and its parents via specificity sort) |

The *candidate corpus* `EvidenceRef { table: "account_fiscal_position", ... }` (the `auto_apply=True`
rules to rank, family `None` -- ontology-unmapped, L9 ontology rows):

| column | dtype | signal |
|---|---|---|
| `fiscal_position_id` | `Int64` | rule identity (the suggested output) |
| `sequence` | `Int32` | tie-break priority within equal company-specificity (lower wins) |
| `company_specificity` | `Int32` | `len(company_id.parent_ids)` -- longer chain sorts first (more specific) |
| `vat_required` | `Boolean` | predicate 1 gate (if true, partner must have VAT) |
| `zip_from` / `zip_to` | `Utf8`/nullable | predicate 2 range bounds (both set or both empty; padded) |
| `state_ids` | `List<Int64>` | predicate 3 allowed states (empty = unconstrained) |
| `country_id` | `Int64`/nullable | predicate 4 required country (NULL = unconstrained) |
| `country_group_id` | `Int64`/nullable | predicate 5 country-group membership + `exclude_state_ids` |

## Slot 2 -- Odoo field -> signal map                 (cite L-doc file:lines)
- `_get_fiscal_position` precedence + intra_eu/vat_exclusion + delivery selection + manual-override short-circuit <- `L9-PARTNER-FISCALPOS.md:211-259` (R8 AXIS-A; `partner.py:246-279`).
- manual override field `property_account_position_id` (company_dependent, always wins) <- `L9-PARTNER-FISCALPOS.md:81-96` (R3; `partner.py:547-550`).
- `_get_first_matching_fpos` specificity sort (company-specific first, then `sequence` asc) <- `L9-PARTNER-FISCALPOS.md:264-267` (R8 AXIS-B; `partner.py:208-213`).
- the five validation predicates (vat_required / zip_range / state / country / country_group, AND-semantics, first match wins) <- `L9-PARTNER-FISCALPOS.md:268-277` (R8; `partner.py:215-244`).
- zip lexicographic-pad normalisation (`_convert_zip_values`, leading-zero pad for digit-only PLZ) <- `L9-PARTNER-FISCALPOS.md:377-404` (R11; `partner.py:181-206`).
- `_get_vat_required_valid` community stub = `bool(partner.vat)` (VIES is Enterprise) <- `L9-PARTNER-FISCALPOS.md:268-273, 417-425` (R8 pred 1 / R13; `partner.py:867-870`).
- delegation tuple `(CustomerCategory, Deduction, NarsTruth, Analytical)` + savant seed line <- `L9-PARTNER-FISCALPOS.md:279-285` (R8 AXIS-B).

## Slot 3 -- Property-level alignment
N/A -- class-level pivots only; no `owl:equivalentProperty` defined. (Confirmed: `odoo_alignment.rs`
holds only class-level `owl:equivalentClass` rows; there are zero property IRIs in the repo.) For the
record, the classes this savant touches do not even have class-level pivots in the seed:
`account.fiscal.position` resolves `None` (ontology-unmapped, needs a Layer-2 alignment axiom -- L9
ontology rows, `L9-PARTNER-FISCALPOS.md:18`), and `res.partner` -> `fibo:LegalEntity` is the only
class-level pivot in play (`odoo_alignment.rs:214-219`). The decision does **not** cross the
FIBO/SKR/ZUGFeRD seam at decision time -- it ranks fiscal-position records against the partner's own
country/VAT/zip scalars.

## Slot 4 -- AXIS-B decision in evidence terms
Let E = the partner classification context row + the `auto_apply` candidate corpus (slot 1), after the
AXIS-A guard has confirmed (no manual override, country present) and chosen `effective_delivery_id`.

-> Conclusion C = `ResolveFiscalPosition(partner_id, fiscal_position_id)` emitted with NARS
`(frequency, confidence)` where:
- candidates are ranked by the deterministic order (company_specificity DESC, sequence ASC); within
  that order the **first** whose five predicates all pass is the deductive winner. NarsTruth fuses the
  predicate matches: a candidate matching on country+state+zip+country_group is stronger evidence than
  a country-only match.
- **frequency** of a given `fiscal_position_id` rises with the number of predicates it satisfies
  beyond the bare minimum (a tightly-scoped DE+PLZ-range+state position beats a bare country match)
  and with its company-specificity rank.
- **confidence** is the NARS weight from how decisively the winner separates from the runner-up: if two
  positions both pass with equal specificity and adjacent `sequence`, confidence drops (the choice is
  brittle to admin re-ordering). A clean single-match keeps confidence high. Capped by phi-1.

Discriminating features (ranked): company_specificity of the matching position >> country predicate >
country_group membership > state predicate > zip_range > vat_required. Deduction here is "the rules
say this position applies"; the *priority ordering* over a mutable rule set is what makes it a
belief-revision classification rather than a frozen lookup (L9 R8 rationale).

## Parity / GoBD notes
Fiscal-position resolution drives `map_tax` / `map_account` (L9 R9/R10) which substitute the tax and
GL accounts on every invoice line -- a wrong position silently mis-maps USt. The savant is
suggestion-only (Iron Rule 7): it never overrides a manual `property_account_position_id` (the
admin's explicit choice is sacrosanct) and woa-rs keeps a deterministic 5-predicate fallback in Rust
for offline/fast-path resolution, the savant adding evidence weighting only when lance-graph is
available (graceful degradation, L9 open question 6). The `intra_eu + vat_exclusion + same_country`
short-circuit (use invoicing not ship-to address) is a B2B-within-DE correctness point that stays in
the AXIS-A guard. NEEDS-INPUT: none for the decision; the `account.fiscal.position` Layer-2 alignment
axiom (to give the candidate corpus a family) is lance-graph follow-on work, not sourceable from L9.
