RICHNESS-LANE-OK

# Lane L12 — Multi-company + Multi-currency (K15 Mehrfirma)

## Sources read (file : line-range : depth)
- base/models/res_currency.py : L1-504 : full
- base/models/res_company.py : L1-493 : full
- account/models/res_currency.py : L1-285 : full
- account/models/account_move_line.py : targeted (dual-amount, currency_rate, residual, exchange) : full-region
- account/models/account_move.py : targeted (invoice_currency_rate, _check_company_auto, _check_balanced, exchange) : full-region

## Ontology rows
| odoo class | owl pivot | OGIT family | DOLCE |
|---|---|---|---|
| `res.currency` | fibo:Currency | 0x62 SMBAccounting | Quality |
| `res.currency.rate` | fibo:ExchangeRate | 0x62 SMBAccounting | Quality (time-stamped) |
| `res.company` | fibo:LegalEntity | 0x80 SmbFoundryCustomer | Endurant |
| `account.move` | fibo:Transaction | 0x81 SmbFoundryInvoice | Perdurant |
| `account.move.line` | fibo:JournalEntryLine | 0x61 BillingCore | Perdurant |

`account_consolidation` is Enterprise (absent) → consolidation classes resolve None.

## Rules extracted (18; 14 AXIS-A, 4 AXIS-B/HYBRID)

### R1 — decimal_places from rounding [AXIS-A]
- res_currency.py:L162-168 — if 0<rounding<1: `ceil(log10(1/rounding))` else 0. digits=(12,6).

### R2 — round/is_zero/compare_amounts [AXIS-A]
- res_currency.py:L216-261 — `round` = float_round HALF-UP to multiple of rounding (NOT banker's). `is_zero(a)` = `|a| < rounding*0.5 + eps`. **compare_amounts(a,b) rounds BOTH first, then compares ≠ is_zero(a-b)** (documented asymmetry — replicate exactly). float_round has `rounding_factor*2^-52` epsilon guard → use Decimal (RFC-009).

### R3 — _get_rates: rate lookup by date+company [AXIS-A]
- res_currency.py:L120-139 — latest rate WHERE name<=date AND company_id IN (NULL, **root_id**) ORDER BY company_id DESC (company-specific > global), name DESC LIMIT 1. Fallback: oldest rate. Final fallback: 1.0. **Uses company.root_id, not company.id** — branches share root rates. Constraint: rates only for root companies (no branches). Unique (name, currency_id, company_id).

### R4 — three rate representations [AXIS-A]
- res_currency.py:L342-504 — `rate` (technical, stored: foreign units per 1 base), `company_rate` = rate/last_company_rate (UI), `inverse_company_rate` = 1/company_rate. Write priority: inverse > company > rate (sanitize conflicting). Engine uses only `rate`.

### R5 — _get_conversion_rate / _convert [AXIS-A]
- res_currency.py:L273-299 — same currency → 1; conversion = `to_rate/from_rate` (both relative to base). `_convert`: zero short-circuits to 0.0; else `from_amount * rate`; round to to_currency if round=true.

### R6 — group_multi_currency toggle [HYBRID → SAVANT]
- res_currency.py:L83-106 — active currency count >1 ⇒ add group_multi_currency. Cannot deactivate a currency still used as a company currency_id.
- `SAVANT: name=CurrencySelectionAdvisor family=0x62 reasoning=NextBestAction inference=Induction semiring=NarsTruth style=Analytical — suggest which currencies to enable based on partner/transaction geography.`

### R7 — rounding write-protection [AXIS-A]
- account/res_currency.py:L26-41 — cannot reduce decimal places (raise rounding) or set 0 if `_has_accounting_entries()` (any AML uses this currency). Protects historical amounts from retroactive rounding loss.

### R8 — currency table builders for reporting [AXIS-B → SAVANT]
- account/res_currency.py:L42-285 — current/historical/average rate types; monocurrency fast-path (VALUES rate=1); average = time-weighted (LEAD window). use_cta_rates ⇒ all three (Enterprise consolidation).
- `SAVANT: name=ReportRateTypeSelector family=0x62 reasoning=Other("ConsolidationRatePolicy") inference=Deduction semiring=Boolean style=Analytical — which rate type (current/historical/average) per report line is an IFRS-vs-HGB policy decision; delegate so it varies without hardcoded if/else.`

### R9 — company tree + currency root-delegation [AXIS-A]
- base/res_company.py:L96-104, L341-418 — `_parent_store`; `root_id`; `_get_company_root_delegated_field_names() = ['currency_id']` ⇒ branches always inherit root currency; constraint blocks branch currency ≠ parent. `parent_id` immutable after create. Multi-currency = different ROOT companies, not branches.

### R10 — _accessible_branches [HYBRID → SAVANT]
- base/res_company.py:L429-450 — subset of branches accessible to current user in multi-branch context.
- `SAVANT: name=UserCompanyAccessAdvisor family=0x80 reasoning=CustomerCategory inference=Induction semiring=NarsTruth style=Analytical — branch-access scoping by user role/context.`

### R11 — check_company / check_company_domain_parent_of [AXIS-A]
- account_move.py:L78, L877-881 — `_check_company_auto=True`; related record's company must be in `move.company_id.parent_ids` (ancestor-or-equal, ltree parent_path subtree). Journal drives company (`_compute_company_id`).

### R12 — dual-amount model balance vs amount_currency [AXIS-A]
- account_move_line.py:L59-144 — `balance` (company currency, signed +=debit), `amount_currency` (line currency). When currency==company: equal. debit=max(balance,0)/credit=max(-balance,0) (storno inverts — DE default storno per STORNO_OPTIONAL_COUNTRIES). DB constraint: sign(balance)==sign(amount_currency). `credit*debit=0`.

### R13 — _compute_currency_rate [AXIS-A]
- account_move_line.py:L137-139, L736-749 — invoices: move.invoice_currency_rate or 1.0; non-invoices: _get_conversion_rate(company→line currency, date=move.date).

### R14 — _compute_amount_currency / _inverse [AXIS-A]
- account_move_line.py:L756-762, L1373-1383 — amount_currency = round(balance*rate); inverse balance = round(amount_currency/rate). currency==company short-circuits.

### R15 — invoice_currency_rate snapshot [AXIS-A]
- account_move.py:L515-540, L1115-1141, L2845-2856 — rate company→currency at invoice date, **frozen at post** (manual override needs rate_is_manual flag). Constraint: >0 when currency≠company on invoice.

### R16 — _check_balanced (company decimal_places) [AXIS-A]
- account_move.py:L2754-2793 — SQL `ROUND(SUM(balance), company.currency.decimal_places)=0` per move. **Not hardcoded 2** (JPY=0, KWD=3).

### R17 — _compute_amount_residual (dual-currency) [AXIS-A]
- account_move_line.py:L793-860 — sum partials; residual in company AND foreign currency; `reconciled = is_zero(residual) AND is_zero(residual_currency)` — **both must be zero**. Double-rounding (SQL ROUND + currency.round).

### R18 — exchange gain/loss account selection [AXIS-A core + AXIS-B config → SAVANT]
- account_move.py:L5218-5237, company.py:L135-145 — `sign=compare(open_balance,0)`; >0 → expense_currency_exchange_account_id (loss); <0 → income (gain); posted on currency_exchange_journal_id. Sign-driven (deterministic).
- `SAVANT: name=ExchangeAccountSelector family=0x62 reasoning=Other("ChartAccountMapping") inference=Deduction semiring=Boolean style=Analytical — deterministic sign picks gain/loss; heuristic only for initial SKR account config.`

## Enterprise gaps flagged
- `account_consolidation` (CTA, intercompany elimination, minority interest): absent — fresh build. SQL currency-table builders are community but invoked by Enterprise.
- `account_reports` (multi-company aggregated reports): absent.

## Open questions
1. f64 rate × Decimal amount boundary — define precision rule (accumulate f64 rate, `Decimal::from_f64(...).round_dp(decimal_places)`).
2. Storno mode (DE) debit/credit splitting — implement.
3. `check_company_domain_parent_of` — pre-compute root_id/parent_path for O(1).
4. invoice_currency_rate manual refresh action on draft.
5. Global (company_id NULL) vs company-specific rate fallback — test the ordering.
6. Exchange-diff entries need K11 Festschreibung handling.

## Depth-proof footer
```
Read: base/models/res_currency.py lines=504 depth=full
Read: base/models/res_company.py lines=493 depth=full
Read: account/models/res_currency.py lines=285 depth=full
Read: account/models/account_move_line.py lines=3742 depth=targeted (dual-amount/rate/residual/exchange regions full)
Read: account/models/account_move.py lines=7328 depth=targeted (currency/company/balanced/exchange regions full)
```
