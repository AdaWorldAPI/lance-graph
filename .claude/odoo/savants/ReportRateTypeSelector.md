# Savant: ReportRateTypeSelector  (id 8 · family 0x62 · lane L12)

**Tuple:** kind=Other(CONSOLIDATION_RATE_POLICY=4) · inference=Deduction · semiring=Boolean · style=Analytical
**Feeds Reasoner impl:** the `Other(4)` reasoner   (CONSOLIDATION_RATE_POLICY)

> dispatch: `ReasoningKind::Other(4)` -> the `Other(code)` default arm "domain-specific Other(code)
> reasoner" (`examples/savant_dispatch.rs:34, 37`; only codes 5|6 take the reconcile arm). Deduction ->
> `QueryStrategy::CamExact`. Semiring `Boolean` — each report line gets a discrete rate-type verdict, not
> a graded belief. Style Analytical inherited from 0x62 SMBAccounting.

## What it decides (AXIS-B core)
For each line of a multi-currency / consolidation report, decide **which rate type converts it**:
`current` (closing/spot rate), `historical` (rate at the transaction date), or `average` (period
time-weighted). The mechanics of *building* each rate column are deterministic SQL (L12 R8); the ambiguous
core is the **accounting-policy choice**: IFRS (IAS 21) and German HGB treat monetary vs non-monetary items
differently — monetary items at closing rate, non-monetary at historical, P&L at average — and the right
type per line depends on the line's nature and the active GAAP. Output is the rate-type verdict per report
line with NARS `(frequency, confidence)`; woa-rs applies it to pick the SQL rate column (Iron Rule 7),
keeping the policy out of hardcoded if/else so it can vary by jurisdiction.

## Deterministic guard (AXIS-A — stays in woa-rs)
The rate-table builders are deterministic SQL and stay in woa-rs: `current` / `historical` / `average`
column construction, the monocurrency fast-path (`VALUES rate=1`), and `average` as a time-weighted `LEAD`
window — `account/res_currency.py:L42-285` (L12 R8). The underlying rate lookup `_get_rates`
(latest WHERE `name<=date` AND `company_id IN (NULL, root_id)`, company-specific > global, fallback 1.0)
and `_convert`/`_get_conversion_rate` are deterministic (L12 R3/R5, `res_currency.py:L120-139, L273-299`).
`use_cta_rates` ⇒ all three columns is an Enterprise consolidation switch (L12 R8). The savant is invoked
only for the *which-rate-type-per-line* policy decision, not the arithmetic.

## Slot 1 — Evidence (Arrow EvidenceRef)
The report line context `EvidenceRef { table: "report_line.rate_policy_context", schema_fingerprint, rows }` (one row per report line):

| column | dtype | signal |
|---|---|---|
| `report_line_id` | `Int64` | the line whose rate type is decided |
| `account_type` | `Utf8` | monetary vs non-monetary nature (e.g. `asset_receivable`/`liability_payable` = monetary ⇒ current; `asset_fixed`/equity = non-monetary ⇒ historical) — the primary IFRS/HGB axis |
| `is_pnl` | `Boolean` | P&L line ⇒ `average` rate (period flows); balance-sheet ⇒ current/historical |
| `transaction_date` | `Date32`/nullable | the date that anchors the `historical` rate (L12 R3 `_get_rates` by date) |
| `period_start` / `period_end` | `Date32` | the window for the `average` time-weighted rate (L12 R8 LEAD window) |
| `currency_id` | `Int64` | source currency; monocurrency fast-path when == company (L12 R8) |
| `company_id` | `Int64` | resolves `root_id` for the rate lookup (branches share root rates, L12 R3/R9) |
| `gaap` | `Utf8` (`ifrs`\|`hgb`\|...) | the active accounting standard — the policy selector that flips monetary/non-monetary treatment |
| `use_cta_rates` | `Boolean` | consolidation switch ⇒ all three rate columns relevant (L12 R8, Enterprise) |

## Slot 2 — Odoo field → signal map                 (cite L-doc file:lines)
- current / historical / average rate-table builders + monocurrency fast-path + time-weighted average (LEAD) + `use_cta_rates` <- `L12-MULTICOMPANY-CURRENCY.md:47-49` (R8; `account/res_currency.py:L42-285`).
- `_get_rates` rate lookup by date + company (latest `name<=date`, company-specific > global, fallback 1.0, uses `root_id`) <- `L12-MULTICOMPANY-CURRENCY.md:31-32` (R3; `res_currency.py:L120-139`).
- `_get_conversion_rate` / `_convert` (`to_rate/from_rate`, zero short-circuit, round to target) <- `L12-MULTICOMPANY-CURRENCY.md:37-39` (R5; `res_currency.py:L273-299`).
- company tree + currency root-delegation (branches inherit root currency + rates) <- `L12-MULTICOMPANY-CURRENCY.md:51-52` (R9; `res_company.py:L96-104, L341-418`).
- delegation tuple `Other("ConsolidationRatePolicy")` + Deduction + Boolean + Analytical (IFRS-vs-HGB policy, delegate so it varies without hardcoded if/else) <- `L12-MULTICOMPANY-CURRENCY.md:47-49` (R8 savant seed) and `savants.rs:69` (`other_kind::CONSOLIDATION_RATE_POLICY=4`).
- Enterprise gap: `account_consolidation` (CTA, intercompany elimination) absent; SQL builders are community but invoked by Enterprise <- `L12-MULTICOMPANY-CURRENCY.md:83-85`.

## Slot 3 — Property-level alignment
**N/A — class-level pivots only; no `owl:equivalentProperty` defined.** `odoo_alignment.rs` holds only
class-level `owl:equivalentClass` rows and **zero** property IRIs (`odoo_alignment.rs:14, 60-68`). The
currency/rate classes map class-level only: `res.currency -> fibo:Currency`, `res.currency.rate ->
fibo:ExchangeRate` (L12 ontology table, `L12-MULTICOMPANY-CURRENCY.md:15-16` — these are L-doc proposals;
the realized seed rows in `odoo_alignment.rs` are partner/account/move/product/uom/hr, currency not yet
seeded). The rate-type-per-line policy decision traverses no property IRI and crosses no SKR/ZUGFeRD seam —
it is an internal GAAP-policy classification over scalar report-line attributes. `account_consolidation`
classes resolve `None` (Enterprise absent, `L12-MULTICOMPANY-CURRENCY.md:21`). Never invent IRIs.

## Slot 4 — AXIS-B decision in evidence terms
Let E = the report-line context row (slot 1).

-> Conclusion C = `RateType(report_line_id) ∈ {Current, Historical, Average}` emitted with NARS
`(frequency, confidence)` where (Boolean semiring ⇒ a discrete winner per line):
- **frequency** for `Current` is high when the line is monetary (`account_type` receivable/payable/cash,
  balance-sheet) under IFRS/HGB closing-rate treatment; for `Historical` when the line is non-monetary
  (fixed assets, equity) anchored at `transaction_date`; for `Average` when `is_pnl` (period flows).
- **confidence** rises when `account_type` + `is_pnl` + `gaap` agree on one rate type and falls in the
  genuinely ambiguous cases the savant exists for (e.g. a non-monetary item carried at fair value, or a
  jurisdiction whose policy diverges from the IFRS default). `use_cta_rates` widens the relevant set to all
  three (consolidation). Capped by phi-1.

Discriminating features (ranked): `account_type` monetary/non-monetary nature >> `is_pnl` flag > `gaap`
standard > `use_cta_rates`. Deduction here is "given the GAAP policy and the line's nature, the rate type
IS determined" — the savant encodes the policy table so it is data, not branches (the explicit L12 R8
rationale "delegate so it varies without hardcoded if/else").

## Parity / GoBD notes
HGB and IFRS differ materially on FX translation (HGB §256a: monetary items at closing rate, the
Imparitätsprinzip caps unrealised gains on long-term items; IFRS/IAS 21 is symmetric) — encoding the policy
as a delegated decision is precisely so a German entity's HGB treatment and a group's IFRS consolidation
can coexist without a code fork. Suggestion-only (Iron Rule 7): the savant picks the rate *type*; woa-rs's
deterministic SQL builders (L12 R8) compute the actual converted amount on the chosen column. Mis-typing a
line (e.g. average where historical is required) mis-states the translation reserve, so the policy verdict
is auditable per line.
