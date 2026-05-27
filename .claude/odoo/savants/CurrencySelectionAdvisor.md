# Savant: CurrencySelectionAdvisor  (id 9 · family 0x62 · lane L12)

**Tuple:** kind=NextBestAction · inference=Induction · semiring=NarsTruth · style=Analytical
**Feeds Reasoner impl:** `NextBestActionReasoner`   (per the impl-per-ReasoningKind decision)

> dispatch: `ReasoningKind::NextBestAction` -> "induce the action with the highest expected value"
> (`examples/savant_dispatch.rs:31`). Induction -> `QueryStrategy::CamWide`. Style Analytical inherited
> from 0x62 SMBAccounting. (This is the only L12 savant on a named `ReasoningKind` rather than `Other`.)

## What it decides (AXIS-B core)
Decide **which currencies the tenant should enable** (activate on `res.currency`), given the geography of
its partners and transactions. The hard constraints are deterministic (you cannot deactivate a currency
still used as some company's `currency_id`; activating any second currency auto-adds the
`group_multi_currency` toggle — L12 R6); the ambiguous core is the *forward-looking recommendation*: from
the spread of partner countries, invoice/bill currencies seen, and bank-journal currencies, induce which
not-yet-enabled currencies are worth turning on. Output is a ranked set of currencies to enable with NARS
`(frequency, confidence)`; woa-rs surfaces it as a suggestion the tenant confirms (Iron Rule 7), never an
auto-activation.

## Deterministic guard (AXIS-A — stays in woa-rs)
The closed-form rules stay in woa-rs (`res_currency.py:L83-106`, L12 R6): the active-currency count > 1 ⇒
add `group_multi_currency`; **cannot deactivate** a currency still referenced as a company `currency_id`
(hard block). The rounding write-protection (cannot raise rounding / set 0 once `_has_accounting_entries`,
`account/res_currency.py:L26-41`, L12 R7) and the company-tree currency root-delegation
(`_get_company_root_delegated_field_names()=['currency_id']`, branches inherit root currency,
`res_company.py:L341-418`, L12 R9) are deterministic. The savant is invoked only for the *which-to-enable*
recommendation, never for the activate/deactivate guard itself.

## Slot 1 — Evidence (Arrow EvidenceRef)
The geography signal `EvidenceRef { table: "tenant.currency_geography", schema_fingerprint, rows }`
(one row per observed currency / country bucket):

| column | dtype | signal |
|---|---|---|
| `currency_id` | `Int64` | a currency seen in evidence (or a candidate to enable) |
| `currency_code` | `Utf8` | ISO code (e.g. `USD`, `CHF`, `GBP`) — the recommendation unit |
| `is_active` | `Boolean` | whether already enabled (the reasoner recommends among `is_active=false`) |
| `partner_country_count` | `Int64` | how many partners are in the country/zone using this currency — the primary geography signal |
| `invoice_currency_count` | `Int64` | how many invoices/bills already carry this currency (latent demand even if not enabled) |
| `bank_journal_count` | `Int64` | how many bank journals are denominated in this currency |
| `is_company_currency` | `Boolean` | whether some company uses it as `currency_id` (cannot be disabled — AXIS-A hard fact, L12 R6) |
| `last_seen_date` | `Date32`/nullable | recency of the most recent transaction in this currency (recent demand weighs more) |

## Slot 2 — Odoo field → signal map                 (cite L-doc file:lines)
- `group_multi_currency` auto-toggle when active count > 1; cannot deactivate a currency used as a company `currency_id` <- `L12-MULTICOMPANY-CURRENCY.md:40-42` (R6; `res_currency.py:L83-106`).
- rounding write-protection once `_has_accounting_entries` (currency in use by any AML) <- `L12-MULTICOMPANY-CURRENCY.md:44-45` (R7; `account/res_currency.py:L26-41`).
- company-tree currency root-delegation (branches inherit root currency; multi-currency = different ROOT companies) <- `L12-MULTICOMPANY-CURRENCY.md:51-52` (R9; `res_company.py:L96-104, L341-418`).
- `_get_rates` company/root-scoped rate availability (a currency is only useful if rates exist for it) <- `L12-MULTICOMPANY-CURRENCY.md:31-32` (R3; `res_currency.py:L120-139`).
- delegation tuple `name=CurrencySelectionAdvisor family=0x62 reasoning=NextBestAction inference=Induction semiring=NarsTruth style=Analytical — suggest which currencies to enable based on partner/transaction geography` <- `L12-MULTICOMPANY-CURRENCY.md:40-42` (R6 savant seed) and `SAVANTS.md:62`.

## Slot 3 — Property-level alignment
**N/A — class-level pivots only; no `owl:equivalentProperty` defined.** `odoo_alignment.rs` holds only
class-level `owl:equivalentClass` rows and **zero** property IRIs (`odoo_alignment.rs:14, 60-68`).
`res.currency -> fibo:Currency` is an L-doc-proposed class-level row (`L12-MULTICOMPANY-CURRENCY.md:15`)
and is not yet among the realized seed rows in `odoo_alignment.rs` (which seed partner/account/move/
product/uom/hr). The which-currency-to-enable recommendation traverses no property IRI and crosses no
SKR/ZUGFeRD seam — partner country and transaction-currency counts are scalar features, not an ontology
traversal. Never invent IRIs.

## Slot 4 — AXIS-B decision in evidence terms
Let E = the per-currency geography rows (slot 1), restricted to candidates with `is_active = false`.

-> Conclusion C = `RecommendEnableCurrencies({currency_id…})` — a ranked set — emitted with NARS
`(frequency, confidence)` where:
- **frequency** of "enable currency X" rises with: a high `partner_country_count` in X's zone, existing
  `invoice_currency_count`/`bank_journal_count` in X (latent demand the tenant is already transacting),
  and recent `last_seen_date`.
- **frequency** is near-certain (and the recommendation moot) for any currency with `is_company_currency`
  or already `is_active`; those are AXIS-A facts, not recommendations.
- **confidence** is the NARS weight from the volume of corroborating evidence (many partners AND many
  invoices in X ⇒ high; a single stray foreign invoice ⇒ low); a thin history keeps confidence low even
  at high frequency. Capped by phi-1.

Discriminating features (ranked): `partner_country_count` >> `invoice_currency_count` >
`bank_journal_count` > `last_seen_date` recency. Induction here is "tenants with this geography footprint
have historically benefited from enabling these currencies." Enabling the first additional currency also
flips `group_multi_currency` (AXIS-A side-effect, L12 R6) — the savant flags that consequence but the
guard performs it.

## Parity / GoBD notes
Pure configuration advice — no posting, no GoBD Festschreibung interaction. Suggestion-only (Iron Rule 7):
the tenant confirms each activation; the deterministic guards then enforce the irreversibility edges
(cannot disable an in-use currency, L12 R6; cannot retroactively coarsen rounding once entries exist,
L12 R7). The one parity caution: enabling a currency commits the tenant to maintaining its rate table
(L12 R3 `_get_rates` falls back to 1.0 when no rate exists, which would silently mis-convert) — so the
recommendation should pair an enable suggestion with a "rates required" note rather than imply the currency
is immediately safe to transact.
