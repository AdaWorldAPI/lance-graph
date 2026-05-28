# Odoo tax: grammar-coded methods

**Scope:** 247 tax-related methods (family or fn or decorator-arg matches `tax|vat|umsatzsteuer|ustva|withholding`).

**Coding axes** (per E-BUSINESS-LOGIC-IS-GRAMMAR-1):

- **T** — transitivity: T (transitive, returns/mutates) | I (intransitive, raises without return)
- **tek** — TEKAMOLO slot: TE (temporal) | KA (causal/regulatory) | MO (modal) | LO (locative) | QU (quantities)
- **men** — mengenmaß: money | percent | rate | count | date | none
- **reg** — regulatory anchor: UStG | HGB | EStG | AO | GoBD | SKR04 | SKR03 | DATEV | ELSTER | Peppol

## Top grammar-coded clusters (n ≥ 3)

| n | T | tek | men | reg |
| ---: | :---: | :---: | --- | --- |
| 39 | T | QU | money | — |
| 31 | T | — | none | — |
| 25 | T | MO | none | — |
| 18 | T | MO | count | — |
| 18 | T | LO | count | — |
| 16 | T | TE | date | — |
| 14 | T | LO | count | UStG |
| 10 | T | MO | money | — |
| 9 | T | LO | none | — |
| 8 | T | MO | count | UStG |
| 7 | I | KA | count | — |
| 7 | T | — | count | — |
| 6 | I | KA | money | — |
| 4 | T | QU | rate | — |
| 4 | T | QU | count | — |
| 3 | T | TE | count | — |
| 3 | I | LO | count | UStG |
| 3 | T | — | money | — |

## Top 50 methods by family

| family | fn | T | tek | men | reg | LOC |
| --- | --- | :---: | :---: | --- | --- | ---: |
| account_account | `_constrains_reconcile` | I | KA | count | — | 8 |
| account_invoice | `_compute_l10n_in_warning` | T | MO | count | — | 87 |
| account_invoice | `_compute_l10n_in_withholding_line_ids` | T | MO | none | — | 8 |
| account_journal | `_compute_accounting_date` | T | TE | date | — | 8 |
| account_move | `_compute_alerts` | T | — | none | — | 14 |
| account_move | `_compute_always_tax_exigible` | T | MO | none | — | 9 |
| account_move | `_compute_always_tax_exigible` | T | MO | none | — | 10 |
| account_move | `_compute_date` | T | TE | date | — | 16 |
| account_move | `_compute_duplicated_ref_ids` | T | — | count | — | 6 |
| account_move_line | `_check_caba_non_caba_shared_tags` | I | MO | count | — | 40 |
| account_move_line | `_check_off_balance` | I | KA | count | — | 10 |
| account_move_line | `_compute_epd_key` | T | QU | count | — | 14 |
| account_move_line | `_compute_epd_needed` | T | QU | money | — | 125 |
| account_move_line | `_compute_is_refund` | T | MO | money | — | 23 |
| account_payment | `_compute_display_withholding` | T | MO | count | — | 18 |
| account_payment | `_compute_l10n_latam_check_warning_msg` | T | MO | count | — | 34 |
| account_payment | `_compute_outstanding_account_id` | T | TE | count | — | 4 |
| account_payment | `_compute_should_withhold_tax` | T | MO | none | — | 5 |
| account_payment | `_compute_withholding_hide_tax_base_account` | T | — | count | — | 8 |
| account_payment_register | `_compute_display_withholding` | T | MO | count | — | 27 |
| account_payment_register | `_compute_l10n_ar_net_amount` | T | QU | money | — | 4 |
| account_payment_register | `_compute_l10n_ar_withholding_ids` | T | TE | date | — | 13 |
| account_payment_register | `_compute_should_withhold_tax` | T | MO | none | — | 5 |
| account_payment_register | `_compute_withholding_hide_tax_base_account` | T | — | count | — | 8 |
| account_payment_register_withholding_line | `_compute_comodel_currency_id` | T | — | none | — | 4 |
| account_payment_register_withholding_line | `_compute_comodel_date` | T | TE | date | — | 4 |
| account_payment_register_withholding_line | `_compute_comodel_payment_type` | T | — | none | — | 4 |
| account_payment_register_withholding_line | `_compute_comodel_percentage_paid_factor` | T | QU | money | — | 20 |
| account_payment_register_withholding_line | `_compute_company_id` | T | — | none | — | 4 |
| account_payment_withholding_line | `_compute_comodel_currency_id` | T | — | none | — | 4 |
| account_payment_withholding_line | `_compute_comodel_date` | T | TE | date | — | 4 |
| account_payment_withholding_line | `_compute_comodel_full_amount` | T | QU | money | — | 4 |
| account_payment_withholding_line | `_compute_comodel_payment_type` | T | — | none | — | 4 |
| account_payment_withholding_line | `_compute_company_id` | T | — | none | — | 4 |
| account_tax | `_check_amount_type` | I | KA | money | — | 6 |
| account_tax | `_check_amount_type_code_formula` | I | MO | money | — | 5 |
| account_tax | `_check_children_scope` | I | KA | money | — | 16 |
| account_tax | `_check_company_consistency` | I | KA | count | — | 10 |
| account_tax | `_check_special_tax_type_constrains` | I | KA | money | — | 7 |
| account_withholding_line | `_compute_account_id` | T | — | count | — | 8 |
| account_withholding_line | `_compute_amount` | T | QU | money | — | 12 |
| account_withholding_line | `_compute_base_amount` | T | QU | percent | — | 12 |
| account_withholding_line | `_compute_original_amounts` | T | QU | money | — | 66 |
| account_withholding_line | `_compute_placeholder_type` | T | MO | none | — | 16 |
| bank_account_verification | `_compute_partner_vat` | T | MO | none | UStG | 6 |
| certificate | `_compute_private_key` | T | MO | none | — | 37 |
| certificate | `_constrains_certificate_key_compatibility` | I | KA | none | — | 28 |
| company | `_compute_account_enabled_tax_country_ids` | T | LO | count | UStG | 12 |
| company | `_compute_company_vat_placeholder` | T | MO | count | UStG | 12 |
| company | `_compute_l10n_in_hsn_code_digit` | T | MO | count | — | 7 |

## Within-subgraph invoke chains (tax → tax)

1 edges among 2 distinct nodes.

**Top chain anchors** (most-invoked tax methods within subgraph):

- `_compute_tax_country_id` (1 callers)

**Sample chain (one walk from a high-anchor node):**

```
  _compute_tax_country_id  <-- called by:
    account_move._validate_taxes_country
```
