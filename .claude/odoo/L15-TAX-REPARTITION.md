RICHNESS-LANE-OK

# Lane L15 — Tax Repartition + Tax Groups + price_include + Cash-Basis

> Deepens K7/L3. L3 covered tax-compute CORE + fiscal-position mapping.
> This lane adds the repartition-line %-split to GL accounts + tax tags,
> ordering subtleties (price_include extraction, include_base_amount,
> is_base_affected), tax-group aggregation for UI/subtotals, and cash-basis
> (CABA) transition mechanics. Almost entirely AXIS-A.

---

## Sources read (file : line-range : depth)

- `/home/user/odoo/addons/account/models/account_tax.py` : L1-5210 : full
- `/home/user/odoo/addons/account/models/account_account_tag.py` : L1-141 : full
- `/home/user/odoo/addons/account/models/account_move.py` : L4080-4148 : full (CABA collection)
- `/home/user/odoo/addons/account/models/company.py` : L131-134 : skim (rounding method field)

---

## Ontology rows

| odoo class | owl pivot | OGIT family (or None) | DOLCE |
|---|---|---|---|
| `account.tax` | fibo:TaxTreatment | 0x62 SMBAccounting | Quality (rate/rule applied to transaction) |
| `account.tax.group` | fibo:TaxCategory | 0x62 SMBAccounting | Abstract (grouping concept) |
| `account.tax.repartition.line` | fibo:PostingRule | 0x62 SMBAccounting | Abstract (split specification) |
| `account.account.tag` | fibo:ReportingTag | None — ontology-unmapped, needs Layer-2 alignment axiom | Abstract |

---

## Rules extracted

### R1 — AccountTaxGroup fields   [AXIS-A]

- **odoo source**: `account_tax.py:L25-68`
- **What it does**: `account.tax.group` defines the grouping bucket for taxes on documents and in the tax-closing entry. Key fields:
  - `tax_payable_account_id` — GL account used as counterpart when the group is a liability to the authorities (Umsatzsteuer-Zahllast).
  - `tax_receivable_account_id` — GL account used when the group is in favour of the company (Vorsteuer-Überhang).
  - `advance_tax_payment_account_id` — downpayment account considered during the tax closing entry (Vorauszahlung).
  - `preceding_subtotal` — optional string label for a subtotal displayed BEFORE this group on the document (e.g. "Subtotal excl. special tax"). When None, the group falls under "Untaxed Amount".
  - `sequence` — order of groups in the document footer; sorted `(sequence, id)`.
  - `country_id` — computed from `company_id.account_fiscal_country_id` or `company_id.country_id`.
- **woa-rs target**: K7 — TaxGroup entity; drives USt-Voranmeldung closing entry accounts + document footer display.
- **Rust sketch**:
  ```rust
  struct TaxGroup {
      id: i64,
      name: String,
      sequence: i32,          // default 10
      company_id: i64,
      tax_payable_account_id: Option<i64>,
      tax_receivable_account_id: Option<i64>,
      advance_tax_payment_account_id: Option<i64>,
      preceding_subtotal: Option<String>,   // None → falls under "Untaxed Amount"
      country_id: i64,
  }
  ```
- **Parity notes**: The `preceding_subtotal` label drives the layered subtotal display in `_get_tax_totals_summary` (L2709-2989). Groups are sorted by `(sequence, id)` at L2813. Multiple groups can share the same `preceding_subtotal` label — they merge into one subtotal section.

---

### R2 — `price_include` compute (override + company default)   [AXIS-A]

- **odoo source**: `account_tax.py:L137-320`
- **What it does**: `price_include` is a computed boolean (not stored). Logic at L302-309:
  ```python
  tax.price_include = (
      tax.price_include_override == 'tax_included'
      or (tax.company_price_include == 'tax_included'
          and not tax.price_include_override)
  )
  ```
  Priority: explicit `price_include_override` field wins; falls back to `company_id.account_price_include`.
  Three-state override: `'tax_included'` | `'tax_excluded'` | `False` (not set → inherit company).
- **woa-rs target**: K7 — every tax lookup must resolve this before calling `_get_tax_details`.
- **Rust sketch**:
  ```rust
  fn price_include(tax: &Tax, company: &Company) -> bool {
      match tax.price_include_override {
          Some(PriceIncludeOverride::TaxIncluded) => true,
          Some(PriceIncludeOverride::TaxExcluded) => false,
          None => company.account_price_include == PriceInclude::TaxIncluded,
      }
  }
  ```
- **Parity notes**: `onchange_price_include` at L717-720 auto-sets `include_base_amount = True` when price_include is enabled — this is a UI hint, not a hard constraint, but worth preserving in woa-rs data model defaults.

---

### R3 — Tax flattening and group-sort (group of taxes)   [AXIS-A]

- **odoo source**: `account_tax.py:L892-971`
- **What it does**: `_flatten_taxes_and_sort_them` (L892-918): iterates `self.sorted(sequence, id)`; if `amount_type == 'group'`, inserts the group's `children_tax_ids` sorted by `(sequence, id)` in place of the parent; records `group_per_tax[child.id] = parent`. Non-group taxes are inserted as-is. Result: flat `sorted_taxes` recordset + `group_per_tax` map.

  `_batch_for_taxes_computation` (L920-971): iterates `sorted_taxes` in **reverse** to group contiguous taxes into batches. A batch stays together if ALL of:
  1. `amount_type` same
  2. `price_include` same (unless `special_mode` active)
  3. `include_base_amount` same
  4. If `include_base_amount=True`: the next-lower tax must NOT have `is_base_affected=True` (a base-affecting tax cannot batch with a subsequent base-affected one without a break)

- **woa-rs target**: K7 — required before any tax amount computation.
- **Rust sketch**:
  ```rust
  fn flatten_and_sort(taxes: &[Tax]) -> (Vec<Tax>, HashMap<i64, Tax>) {
      let mut sorted = vec![];
      let mut group_per_tax: HashMap<i64, Tax> = HashMap::new();
      for tax in taxes.iter().sorted_by_key(|t| (t.sequence, t.id)) {
          if tax.amount_type == AmountType::Group {
              for child in tax.children.iter().sorted_by_key(|c| (c.sequence, c.id)) {
                  group_per_tax.insert(child.id, tax.clone());
                  sorted.push(child.clone());
              }
          } else {
              sorted.push(tax.clone());
          }
      }
      (sorted, group_per_tax)
  }
  ```
- **Parity notes**: Nested groups (group-of-group) are **prohibited** by constraint at L598-613. The `_check_children_scope` constraint also requires child `type_tax_use` ∈ {`none`, parent's use} and child `tax_scope` ∈ {parent's scope, `False`}.

---

### R4 — Three-pass tax amount evaluation in `_get_tax_details`   [AXIS-A]

- **odoo source**: `account_tax.py:L1134-1332`
- **What it does**: The core computation engine. Called by `compute_all` and `_add_tax_details_in_base_line`. Three evaluation passes in order:

  **Pass 1 — Fixed taxes (reverse order, L1253-1254)**:
  ```python
  for tax in reversed(sorted_taxes):
      eval_tax_amount(tax._eval_tax_amount_fixed_amount, tax)
  ```
  `_eval_tax_amount_fixed_amount` (L1079-1092): for `amount_type == 'fixed'`:
  ```python
  sign = -1 if price_unit < 0.0 else 1
  return sign * quantity * self.amount
  ```
  Fixed taxes are evaluated FIRST because their amount may affect the base of subsequent price-included percent taxes.

  **Pass 2 — Price-included taxes (reverse order, L1256-1259)**:
  ```python
  for tax in reversed(sorted_taxes):
      if taxes_data[tax.id]['price_include']:
          eval_tax_amount(tax._eval_tax_amount_price_included, tax)
  ```
  `_eval_tax_amount_price_included` (L1094-1112):
  - For `percent`: `raw_base * (1/(1+sum_pct)) * (self.amount/100)` — the entire batch's percentage is used to extract from the price-included amount.
  - For `division`: `raw_base * self.amount / 100.0` — direct division (no batch-sum normalization).

  **Pass 3 — Price-excluded taxes (forward order, L1261-1264)**:
  ```python
  for tax in sorted_taxes:
      if not taxes_data[tax.id]['price_include']:
          eval_tax_amount(tax._eval_tax_amount_price_excluded, tax)
  ```
  `_eval_tax_amount_price_excluded` (L1114-1132):
  - For `percent`: `raw_base * self.amount / 100.0`
  - For `division`: `raw_base * self.amount / 100.0 / (1 - total_pct)` where `total_pct = sum(batch.amount)/100`.

  **raw_base** for each tax = `quantity * price_unit` (rounded per `round_per_line` if applicable), adjusted by `extra_base_for_tax` accumulated from `_propagate_extra_taxes_base`.

  **Base amount computation** (L1266-1299, reverse order):
  ```python
  base = raw_base + tax_data['extra_base_for_base']
  if tax_data['price_include'] and special_mode in (False, 'total_included'):
      base -= total_tax_amount   # subtract all taxes in this tax's batch
  tax_data['base'] = base
  ```
  `total_tax_amount` = sum of tax amounts in the same batch (including reverse-charge twin amounts).

  **Totals**:
  - `total_excluded = taxes_data_list[0]['base']`
  - `total_included = total_excluded + sum(tax_data['tax_amount'])`
  - If no taxes: `total_included = total_excluded = raw_base`.

- **woa-rs target**: K7 — this IS the tax engine. Must be reproduced exactly.
- **Rust sketch** (key branches):
  ```rust
  fn get_tax_details(taxes: &[Tax], price_unit: Decimal, quantity: Decimal,
                     rounding: RoundingMethod, currency_pd: u32,
                     special_mode: SpecialMode) -> TaxDetailsResult {
      let mut raw_base = price_unit * quantity;
      if rounding == RoundingMethod::RoundPerLine {
          raw_base = round(raw_base, currency_pd);
      }
      // Pass 1: fixed taxes (reverse)
      for tax in taxes.iter().rev() {
          if tax.amount_type == Fixed && !already_computed(tax) {
              let sign = if price_unit < 0 { -1 } else { 1 };
              set_tax_amount(tax, Decimal::from(sign) * quantity * tax.amount);
              propagate_extra_base(tax, &mut taxes_data, special_mode);
          }
      }
      // Pass 2: price-included (reverse)
      for tax in taxes.iter().rev() {
          if taxes_data[tax.id].price_include && !already_computed(tax) {
              let raw = raw_base + taxes_data[tax.id].extra_base_for_tax;
              let amt = eval_price_included(tax, batch, raw);
              set_tax_amount(tax, amt);
              propagate_extra_base(tax, &mut taxes_data, special_mode);
          }
      }
      // Pass 3: price-excluded (forward)
      for tax in taxes.iter() {
          if !taxes_data[tax.id].price_include && !already_computed(tax) {
              let raw = raw_base + taxes_data[tax.id].extra_base_for_tax;
              let amt = eval_price_excluded(tax, batch, raw);
              set_tax_amount(tax, amt);
              propagate_extra_base(tax, &mut taxes_data, special_mode);
          }
      }
      // Base amounts (reverse)
      for tax in taxes.iter().rev() { ... }
  }
  ```
- **Parity notes / gotchas**:
  - `round_per_line`: each `tax_amount` is `float_round(raw, precision_rounding=currency.rounding)` immediately after evaluation (L1179-1180). `round_globally`: amounts stay unrounded until the aggregation pass in `_round_base_lines_tax_details`.
  - `has_negative_factor` (L507-511): a tax with at least one repartition line with `factor < 0` gets a "reverse charge twin" entry with `tax_amount = -original_tax_amount`. Both the positive and negative entries appear in `taxes_data_list`. This is for EU reverse-charge (e.g. +100% / -100% split).
  - `special_mode='total_included'`: ALL taxes treated as price-included regardless of their `price_include` flag. `special_mode='total_excluded'`: ALL treated as price-excluded.
  - `force_price_include` context key maps to `special_mode` at L4918-4923.

---

### R5 — `_propagate_extra_taxes_base`: base-affecting cascade   [AXIS-A]

- **odoo source**: `account_tax.py:L973-1077`
- **What it does**: After each tax's amount is computed, this method updates `extra_base_for_tax` and `extra_base_for_base` on OTHER taxes to reflect the "include_base_amount" cascade. Two symmetric halves:

  **price_include taxes** (L1004-1026):
  - `special_mode in (False, 'total_included')`:
    - If `include_base_amount`: subtract this tax's amount from ALL taxes AFTER it whose `is_base_affected=False` (`extra_base_for_tax`), AND from all taxes BEFORE it (`extra_base_for_base`).
    - Else: subtract from ALL taxes AFTER it (both fields).
    - Also subtract from all taxes BEFORE it (extra_base_for_base only).
  - `special_mode == 'total_excluded'`:
    - If `include_base_amount`: ADD to taxes after it where `is_base_affected=True`.

  **price_excluded taxes** (L1047-1077):
  - `special_mode in (False, 'total_excluded')`:
    - If `include_base_amount`: ADD this tax's amount to taxes AFTER it where `is_base_affected=True`.
  - `special_mode == 'total_included'`:
    - If NOT `include_base_amount`: SUBTRACT from taxes AFTER it.
    - Also subtract from taxes BEFORE it.

  The `get_tax_before()` / `get_tax_after()` helpers (L986-996) stop at the boundary of the current tax's batch.

- **woa-rs target**: K7 — must implement this exactly for compound tax chains (e.g. 19% MwSt + special excise that affects VAT base).
- **Parity notes**: The `is_base_affected` flag is the "consent" side — a tax says "I accept being affected by prior taxes". `include_base_amount` is the "push" side — a tax says "I affect subsequent taxes". Both must be true for the cascade to fire.

---

### R6 — Repartition line %-split to accounts and tags   [AXIS-A]

- **odoo source**: `account_tax.py:L5142-5210` (AccountTaxRepartitionLine class)
- **What it does**: `account.tax.repartition.line` defines how a tax's computed amount is split into GL postings and tax report boxes.

  Key fields:
  - `factor_percent` (float, digits=(16,12), default=100): percentage of the tax amount assigned to this line. Stored with 12 decimal places. `factor = factor_percent / 100.0` (L5192-5194).
  - `repartition_type` ∈ `{'base', 'tax'}`: `'base'` = tags only (no account posting); `'tax'` = actual GL posting.
  - `document_type` ∈ `{'invoice', 'refund'}`: which side of the transaction this line applies to.
  - `account_id`: target GL account for the tax amount split.
  - `tag_ids`: many2many to `account.account.tag` — determines which USt-VA report box gets hit.
  - `use_in_tax_closing` (computed at L5182-5189): `True` when `repartition_type == 'tax'` AND `account_id` is set AND `account_id.internal_group not in ('income', 'expense')`. Marks lines relevant to the periodic VAT closing entry.
  - `sequence` — display/matching order; invoice and refund lines must be in the same sequence order.

  **Constraints** (L561-596):
  - Exactly ONE `'base'` line per document_type.
  - At least ONE `'tax'` line per document_type.
  - Count of invoice lines must equal count of refund lines.
  - Invoice and refund lines must have matching `repartition_type` and `factor_percent` in same order.
  - Sum of positive `factor` values among tax-type lines must equal 1.0 (100%).
  - If any negative factors exist, their sum must equal -1.0 (100% negative, for reverse charge).

  **Default repartition** created on new tax (L490-504):
  ```python
  # invoice side
  {'document_type': 'invoice', 'repartition_type': 'base', 'tag_ids': []}
  {'document_type': 'invoice', 'repartition_type': 'tax', 'tag_ids': []}
  # refund side
  {'document_type': 'refund', 'repartition_type': 'base', 'tag_ids': []}
  {'document_type': 'refund', 'repartition_type': 'tax', 'tag_ids': []}
  ```

- **woa-rs target**: K7 + K8 — the repartition lines determine: (a) which GL accounts tax amounts post to; (b) which USt-VA report boxes are populated.
- **Rust sketch**:
  ```rust
  struct RepartitionLine {
      id: i64,
      tax_id: i64,
      factor_percent: Decimal,    // 12 decimal places
      factor: Decimal,            // = factor_percent / 100
      repartition_type: RepartitionType,   // Base | Tax
      document_type: DocumentType,          // Invoice | Refund
      account_id: Option<i64>,
      tag_ids: Vec<i64>,
      use_in_tax_closing: bool,   // computed: repartition_type==Tax && account && account.internal_group not in (income, expense)
      sequence: i32,
  }

  fn split_tax_amount(
      tax_amount: Decimal,
      tax_reps: &[RepartitionLine],    // filtered to correct document_type, repartition_type=Tax
      currency: &Currency,
      company_currency: &Currency,
  ) -> Vec<RepartitionLineAmount> {
      let mut reps_data: Vec<_> = tax_reps.iter()
          .map(|rep| RepartitionLineAmount {
              rep,
              tax_amount_currency: currency.round(tax_amount * rep.factor * rep_sign),
              tax_amount: company_currency.round(tax_amount_local * rep.factor * rep_sign),
          })
          .collect();
      // Distribute rounding delta on largest-first (L2439-2466)
      distribute_delta_smoothly(&mut reps_data, tax_amount_total, currency);
      reps_data
  }
  ```
- **Parity notes / gotchas**:
  - The rounding delta after per-rep multiplication is distributed via `_distribute_delta_amount_smoothly` (L2439-2466), sorted by largest `|tax_amount_currency|` first (L2439-2441). This ensures the largest slice absorbs any rounding error.
  - `_get_aml_target_tax_account` (L5201-5210): if `tax_exigibility == 'on_payment'` (CABA) AND context NOT `caba_no_transition_account`, returns `cash_basis_transition_account_id` instead of `account_id`. This is the CABA routing mechanism.
  - For reverse-charge taxes (`is_reverse_charge=True`): negative-factor repartition lines are used (L2410-2415), with `tax_rep_sign = -1.0`.

---

### R7 — `_add_accounting_data_to_base_line_tax_details`: tag assignment and grouping key   [AXIS-A]

- **odoo source**: `account_tax.py:L2362-2497`
- **What it does**: After `_get_tax_details` computes amounts, this method enriches each `tax_data` with full accounting information:

  1. **Base-line tags** (L2392-2407): collects `tag_ids` from the `'base'`-type repartition lines of each non-reverse-charge tax (invoice or refund side depending on `is_refund`). Also includes product `account_tag_ids`. Skipped for CABA taxes unless `include_caba_tags=True`.

  2. **Repartition-line amounts** (L2409-2436): for each tax's repartition lines (filtered by `repartition_type='tax'` and correct factor sign):
     ```python
     tax_rep_data['tax_amount_currency'] = currency.round(
         tax_amount_currency * tax_rep.factor * tax_rep_sign
     )
     tax_rep_data['tax_amount'] = company_currency.round(
         tax_data['tax_amount'] * tax_rep.factor * tax_rep_sign
     )
     tax_rep_data['account'] = tax_rep._get_aml_target_tax_account(force_caba_exigibility)
                               or base_line['account_id']
     ```
     If no account on repartition line: falls back to the base line's own account.

  3. **Subsequent tags** (L2468-2496): in reverse order, for each repartition line within a tax, `tax_rep_data['tax_tags']` includes the repartition line's own `tag_ids` PLUS the `base`-repartition tags of any tax that has `is_base_affected=True` and comes after this tax in sequence (only if the subsequent tax has `include_base_amount`). This is the "tag cascade" for compound taxes.

  4. **Grouping key** (L2485-2492): calls `_prepare_base_line_tax_repartition_grouping_key` which produces the key used to merge / de-duplicate tax lines:
     ```python
     {
         'tax_repartition_line_id': tax_rep.id,
         'partner_id': ...,
         'currency_id': ...,
         'group_tax_id': tax_data['group'].id,  # parent group-tax or empty
         'analytic_distribution': ... if tax.analytic or not rep.use_in_tax_closing else False,
         'account_id': tax_rep_data['account'].id or base_line['account_id'],
         'tax_ids': [taxes that this line affects for subsequent base],
         'tax_tag_ids': [merged tags],
     }
     ```

- **woa-rs target**: K7 + K3 (posting) — the grouping key determines which `account.move.line` records are created/updated for taxes.
- **Parity notes**: `analytic_distribution` is cleared on tax lines where `use_in_tax_closing=True` AND `tax.analytic=False` (L2329-2333). This prevents analytic allocations leaking into VAT clearing accounts.

---

### R8 — `_round_base_lines_tax_details`: global vs per-line rounding   [AXIS-A]

- **odoo source**: `account_tax.py:L2178-2288`
- **What it does**: Two rounding methods selectable via `company.tax_calculation_rounding_method` (L131-134 company.py):
  - `'round_per_line'` (default: False — `'round_globally'` is the default): raw amounts already rounded in `_get_tax_details` per tax per line.
  - `'round_globally'` (default): amounts remain raw until this aggregation step.

  **Round-globally flow** (L2286-2288):
  1. `_round_tax_details_tax_amounts` (L1890-1986): groups lines by `(tax, currency, is_refund, is_reverse_charge, price_include, computation_key)`. For each group: rounds the SUM of raw tax amounts; distributes the rounding delta (positive or negative cents) back to individual lines proportionally, largest first. Same for base amounts — with `mode` distinction:
     - `mode='mixed'` (default): price-included taxes use `'included'` mode (round base+tax together then subtract); price-excluded use `'excluded'` (round base and tax independently).
     - `mode='included'`: always `round(base + tax) - tax`.
     - `mode='excluded'`: always `round(base)` and `round(tax)` independently.
  2. `_round_tax_details_base_lines` (L1988-2097): computes `delta_total_excluded{_currency}` — the rounding correction on the base line's `total_excluded`. For price-excluded: `round(sum_raw_total_excluded) - sum_rounded_total_excluded`. For price-included: `round(sum_raw_total_included) - (sum_total_excluded + sum_tax_amount)`.
  3. `_round_tax_details_tax_amounts_from_tax_lines` (L2099-2176): if existing tax lines provided, overrides computed amounts to match the actual posted amounts (for user-edited taxes).

  **Raw rounding** (L2237-2248): copies `currency.round(raw_X)` → `X` for each field.

- **woa-rs target**: K7 — must select rounding mode from company setting. `round_globally` is the standard for Germany/EU.
- **Rust sketch**:
  ```rust
  fn round_base_lines_tax_details(
      base_lines: &mut [BaseLine],
      company: &Company,
      tax_lines: Option<&[TaxLine]>,
  ) {
      // Step 1: raw round
      for bl in base_lines.iter_mut() {
          bl.tax_details.total_excluded = currency.round(bl.tax_details.raw_total_excluded);
          for td in bl.tax_details.taxes_data.iter_mut() {
              td.base_amount = currency.round(td.raw_base_amount);
              td.tax_amount = currency.round(td.raw_tax_amount);
          }
      }
      // Step 2: global delta distribution (round_globally only)
      round_tax_details_tax_amounts(base_lines, company);
      round_tax_details_base_lines(base_lines, company);
      // Step 3: override from existing tax lines (manual amounts)
      if let Some(tl) = tax_lines {
          round_tax_details_tax_amounts_from_tax_lines(base_lines, company, tl);
      }
  }
  ```
- **Parity notes / gotchas**:
  - Delta distribution uses `_distribute_delta_amount_smoothly` (L1836-1888): converts delta to integer "error units" at the currency's precision, distributes proportionally to raw-amount factors, largest first. Remaining 1-unit errors distributed sequentially. This guarantees exact cent-level accuracy.
  - For EDI reporting: use `raw_total_excluded_currency` (unrounded, 6-8 decimal places) NOT `total_excluded_currency` (L2183-2189).

---

### R9 — `compute_all` public API   [AXIS-A]

- **odoo source**: `account_tax.py:L4864-4980`
- **What it does**: The legacy/public entry point used by sale/purchase order lines, wizard computations, etc. Wraps the new engine:
  1. Resolves `special_mode` from context `force_price_include` or `handle_price_include` param (L4918-4923).
  2. Calls `_prepare_base_line_for_taxes_computation(None, ...)` with explicit kwargs.
  3. Calls `_add_tax_details_in_base_line` then `_add_accounting_data_to_base_line_tax_details` with `compute_all_use_raw_base_lines=True` context (uses raw unrounded amounts for repartition split — L4936-4938).
  4. Returns legacy dict:
     ```python
     {
         'base_tags': [...],           # tag ids on the base line
         'taxes': [{                   # one entry PER repartition line (not per tax)
             'id': tax.id,
             'name': ...,
             'amount': tax_rep_data['tax_amount_currency'],  # repartition-line amount
             'base': tax_data['raw_base_amount_currency'],
             'sequence': tax.sequence,
             'account_id': tax_rep_data['account'].id,
             'analytic': tax.analytic,
             'use_in_tax_closing': rep_line.use_in_tax_closing,
             'is_reverse_charge': ...,
             'price_include': tax.price_include,
             'tax_exigibility': tax.tax_exigibility,
             'tax_repartition_line_id': rep_line.id,
             'group': tax_data['group'],   # parent group-tax recordset
             'tag_ids': tax_rep_data['tax_tags'].ids,
             'tax_ids': tax_rep_data['taxes'].ids,
         }],
         'total_excluded': currency.round(raw_total_excluded),
         'total_included': currency.round(raw_total_included),
         'total_void': total_excluded + sum(amounts where account_id is None),
     }
     ```
  - `total_void`: base + all tax amounts WITHOUT an account assigned. Used to compute the taxable amount excluding taxes that don't generate a posting.
  - Rounding of `total_excluded`/`total_included` can be suppressed via context `round_base=False` (L4970).

- **woa-rs target**: K7 — woa-rs can expose a compatibility function with the same signature for places that use the old API. Internally delegate to `_get_tax_details` + `_add_accounting_data_to_base_line_tax_details`.
- **Parity notes**: Note that `taxes` entries are ONE PER REPARTITION LINE, not one per tax. A tax with 2 repartition lines produces 2 entries. The `id` field on each entry is still the tax's id (not the repartition line id) — `tax_repartition_line_id` is the repartition line.

---

### R10 — `_get_tax_totals_summary`: document footer subtotals   [AXIS-A]

- **odoo source**: `account_tax.py:L2709-2989`
- **What it does**: Produces the structured tax totals shown in the document footer (invoice, POS receipt, etc.). Algorithm:
  1. **Global totals** (L2782-2794): sum of all base and tax amounts across all base lines.
  2. **Per-tax-group aggregation** (L2806-2871): groups by `tax_group_id`, sorted by `(group.sequence, group.id)`. For each group:
     - Collects all taxes involved.
     - Computes `display_base_amount`: special-cased for `fixed`-only groups (no base shown) and `division price-included`-only groups (base = total_included).
     - Uses `preceding_subtotal` to assign to a named subtotal section (default: "Untaxed Amount").
  3. **Subtotal accumulation** (L2873-2889): subtotals appear in document order (the `subtotals_order` dict tracks first-encounter order). Each subtotal's `base_amount` = global base + sum of ALL prior subtotals' tax amounts.
  4. **Cash rounding** (L2891-2953): if `cash_rounding` provided, computes delta to reach rounded total; applies via `'add_invoice_line'` (adjusts base) or `'biggest_tax'` (adjusts the largest tax group's amount) strategy.
  5. **Non-deductible** (L2956-2981): for reverse-charge lines marked `special_type='non_deductible'`, their tax amounts are shown separately and subtracted from the tax group totals.

- **woa-rs target**: K7/K8 — drives the invoice tax footer display; also used for VAT return summary.
- **Parity notes**: `same_tax_base` flag (L2954) is True when all tax groups have the same display base amount — controls whether to show base amounts per group (when they differ due to different exemptions).

---

### R11 — Cash-basis (CABA): `tax_exigibility` + transition account   [AXIS-A]

- **odoo source**:
  - `account_tax.py:L164-174` — field definitions
  - `account_tax.py:L247-255` — constraint: CABA transition account must allow reconciliation
  - `account_tax.py:L5201-5210` — `_get_aml_target_tax_account` routing
  - `account_move.py:L4080-4147` — `_collect_tax_cash_basis_values`

- **What it does**:
  - `tax_exigibility` ∈ `{'on_invoice', 'on_payment'}` (default: `'on_invoice'`).
  - `'on_invoice'` (Soll-Besteuerung): tax becomes due immediately when invoice is posted → normal flow, tax posts to the real tax payable/receivable account.
  - `'on_payment'` (Ist-Besteuerung / cash basis): tax becomes due only when payment is received/made.

  **Transition account routing** (`_get_aml_target_tax_account`, L5201-5210):
  ```python
  if tax.tax_exigibility == 'on_payment' and not context.get('caba_no_transition_account'):
      return tax.cash_basis_transition_account_id  # interim account
  else:
      return self.account_id  # final tax account
  ```
  So on invoice posting, CABA taxes post to `cash_basis_transition_account_id` (a temporary clearing account). The transition account MUST be reconcilable (`account.reconcile = True`).

  **CABA collection** (`_collect_tax_cash_basis_values`, account_move.py:L4080-4147):
  When a payment is reconciled against an invoice, this method identifies:
  - Lines where `tax_line_id.tax_exigibility == 'on_payment'` → `caba_treatment = 'tax'`
  - Lines where any tax in `tax_ids` has `tax_exigibility == 'on_payment'` → `caba_treatment = 'base'`
  - `total_balance` / `total_residual` from `asset_receivable`/`liability_payable` lines
  - `is_fully_paid` = company currency residual is zero OR foreign currency residual is zero

  The payment percentage is `total_residual / total_balance` applied to move the amounts from transition account to the real tax account via a new journal entry (`tax_cash_basis_created_move_ids`).

  **CABA tags**: `include_caba_tags=False` by default in `_add_accounting_data_to_base_line_tax_details`. When False: CABA taxes' base-line tags are suppressed (tax not yet exigible, should not appear in the USt-VA box yet). When True (used in CABA reconciliation entry creation): tags are included.

- **woa-rs target**: K7 (Ist-Besteuerung support) + K3 (CABA reconciliation entry). In Germany, Ist-Besteuerung is available for small businesses (§ 20 UStG). Most SMBs use Soll-Besteuerung.
- **Rust sketch**:
  ```rust
  fn get_aml_target_tax_account(
      tax: &Tax, rep: &RepartitionLine, ctx: &Context
  ) -> Option<AccountId> {
      if tax.tax_exigibility == TaxExigibility::OnPayment
          && !ctx.caba_no_transition_account {
          tax.cash_basis_transition_account_id
      } else {
          rep.account_id
      }
  }
  ```
- **Parity notes / gotchas**:
  - The constraint at L247-255: if `tax_exigibility == 'on_payment'` AND `cash_basis_transition_account_id.reconcile == False` → `ValidationError`. Must enforce in woa-rs.
  - `hide_tax_exigibility` field (L163): reads `company_id.tax_exigibility` (a company-level flag). If the company has disabled cash-basis, the field is hidden in UI. woa-rs should respect this: only offer CABA option if the company has it enabled.
  - `company.tax_exigibility = False` means CABA is globally disabled for the company; `'on_payment'` taxes still exist but the feature is hidden.

---

### R12 — `account.account.tag` sign convention for tax report boxes   [AXIS-A]

- **odoo source**: `account_account_tag.py:L1-141`
- **What it does**: Tax tags link tax repartition lines to VAT report boxes. Key fields:
  - `name`: the tag name; for report-linked tags, this matches `account_report_expression.formula` (possibly prefixed with `-`).
  - `applicability` ∈ `{'accounts', 'taxes', 'products'}`: only `'taxes'` tags attach to repartition lines.
  - `balance_negate` (computed, L20, L40-48): True if the formula in `account_report_expression` starts with `-`. Determines whether the tag uses `+balance` or `-balance` when aggregating for the report.
  - `report_expression_id` (computed): links back to the report expression defining which VAT return box this tag feeds.

  **Sign rule** (L51-67): a tag named `"-Kz81"` has `balance_negate=True` → the balance posted to GL lines with this tag is negated when summed into the report box. This is how odoo encodes the +/- convention for USt-VA boxes (Kennzahlen).

  **`_get_tax_tags_domain`** (L88-96): strips leading `-` when searching: `name = formula.lstrip('-')`. The sign is encoded in the tag name's leading character, not in a separate field.

- **woa-rs target**: K8 — the tag → report-expression → Kennzahl mapping drives the USt-Voranmeldung report. Tags with `balance_negate=True` contribute negative balances to their box.
- **Parity notes**: Tags are country-specific (`country_id`). For Germany, tags will have `country_id = DE`. Multi-VAT companies can have tags for foreign countries too (L5178-5180 of account_tax.py). The `_get_related_tax_report_expressions` (L98-108) joins via formula match (`formula = tag.name` or `formula = '-' + tag.name`).

---

### R13 — `_prepare_tax_lines`: GL line generation from repartition data   [AXIS-A]

- **odoo source**: `account_tax.py:L3032-3126`
- **What it does**: Final step — converts `tax_reps_data` (from R7) into the actual `account.move.line` create/update/delete diff:
  1. For each base line: computes the base-line's `amount_currency` / `balance` using `total_excluded + delta_total_excluded` × `sign`.
  2. For each tax_data → for each tax_rep_data: accumulates into `tax_lines_mapping[grouping_key]`:
     - `tax_base_amount += sign * tax_data['base_amount']`
     - `amount_currency += sign * tax_rep_data['tax_amount_currency']`
     - `balance += sign * tax_rep_data['tax_amount']`
  3. Removes zero-amount lines (unless `__keep_zero_line` flag set).
  4. Matches against existing `tax_lines` to produce `tax_lines_to_update` / `tax_lines_to_delete` / `tax_lines_to_add`.

  The `sign` field on base_line (typically +1 for normal invoice, -1 for credit note as viewed from the company's perspective) ensures correct debit/credit.

- **woa-rs target**: K3 + K7 — this is the bridge from tax computation to GL posting.
- **Parity notes**: `tax_lines_to_add` entries already contain all grouping key fields merged with amounts (L3119) — they can be passed directly to `account.move.line.create()`. The `'__keep_zero_line'` hidden key (L3100) prevents pruning of intentional zero-amount tax lines (e.g. 0% exempt lines that still need to appear in the report).

---

### R14 — Validation constraints on repartition lines   [AXIS-A]

- **odoo source**: `account_tax.py:L554-596`
- **What it does**: `_validate_repartition_lines` constraint enforces:
  1. Exactly 1 `'base'` line per document_type (L557-559).
  2. At least 1 `'tax'` line per document_type (L578-580).
  3. Invoice and refund line counts must be equal (L575-576).
  4. Corresponding positions must have matching `repartition_type` AND `factor_percent` (L582-588).
  5. Sum of positive `factor` values among tax lines must == 1.0 (precision 2 decimal places) (L590-593).
  6. If any negative factors: sum must == -1.0 (L594-596).

- **woa-rs target**: K7 — validation in the tax configuration UI/API.
- **Rust sketch**:
  ```rust
  fn validate_repartition_lines(invoice_lines: &[RepLine], refund_lines: &[RepLine]) -> Result<(), ValidationError> {
      ensure!(invoice_lines.iter().filter(|l| l.rep_type == Base).count() == 1)?;
      ensure!(refund_lines.iter().filter(|l| l.rep_type == Base).count() == 1)?;
      ensure!(invoice_lines.iter().any(|l| l.rep_type == Tax))?;
      ensure!(refund_lines.iter().any(|l| l.rep_type == Tax))?;
      ensure!(invoice_lines.len() == refund_lines.len())?;
      for (inv, ref_) in invoice_lines.iter().zip(refund_lines.iter()) {
          ensure!(inv.rep_type == ref_.rep_type && inv.factor_percent == ref_.factor_percent)?;
      }
      let pos_sum: Decimal = invoice_lines.iter().filter(|l| l.factor > 0).map(|l| l.factor).sum();
      ensure!(Decimal::abs(pos_sum - 1.0) < 0.01)?;
      // ... negative factor check
      Ok(())
  }
  ```

---

### R15 — `_adapt_price_unit_to_another_taxes`: fiscal-position price-include adaptation   [AXIS-A]

- **odoo source**: `account_tax.py:L1338-1385`
- **What it does**: Used when a fiscal position maps a price-included tax to a different tax. Adjusts the price unit so the end customer sees the same number.

  Only adapts when ALL taxes in `original_taxes` are price-included (L1362). If any original tax is NOT price-included: returns `price_unit` unchanged.

  Algorithm:
  1. Compute `total_excluded` by calling `_get_tax_details(original_taxes, price_unit, 1.0, round_globally)`.
  2. Re-add the new price-included taxes' amounts: `delta = sum(x['tax_amount'] for x if x['tax'].price_include)` from `_get_tax_details(new_taxes, total_excluded, 1.0, special_mode='total_excluded')`.
  3. Return `total_excluded + delta`.

- **woa-rs target**: K7 — needed when fiscal position changes a price-included tax.
- **Parity notes**: This is the same method mirrored in `account_tax.js`. Precision: uses `round_globally` (no rounding) for intermediate computation to avoid double-rounding.

---

## AXIS-B rules (Savant seeds)

### RS1 — Tax-exigibility mode selection for a company   [AXIS-B / HYBRID]

The determination of whether a company should use `on_invoice` (Soll) vs `on_payment` (Ist) Besteuerung is:
- AXIS-A guard: `tax_exigibility` field must be enabled on company (`company.tax_exigibility = True`).
- AXIS-B core: recommending Ist-Besteuerung eligibility based on revenue thresholds (§ 20 UStG: ≤ 600,000 EUR), industry patterns, accounting setup, etc.

`SAVANT: name=TaxExigibilitySuggestor family=0x62 reasoning=NextBestAction inference=Induction semiring=NarsTruth style=Analytical — heuristic: recommend on_payment vs on_invoice based on company revenue/industry evidence rather than hard-coding threshold`

---

## Enterprise gaps flagged

- `account_reports` module: **absent** (Enterprise). The `account.report` / `account.report.expression` models referenced by `report_expression_id` on tags (L19-20, L40-48 of account_account_tag.py) exist in community but the full report engine is Enterprise. What IS present: the tag linkage mechanism (`tag.name` → `formula` join) and the `balance_negate` computation. The woa-rs K8 engine must implement the USt-VA report fresh but can steal the tag→Kennzahl mapping structure.
- `account_accountant` module: `_predict_specific_tax` referenced at L5036 (invoice predictive tax import) is Enterprise-only. Safe to omit.

---

## Open questions for the Opus porter

1. **Batch boundary edge case**: when `include_base_amount=True` and two consecutive taxes have different `is_base_affected` values within the same sequence, the batch-splitting logic in `_batch_for_taxes_computation` (L948-963) can produce non-obvious batches. Recommend writing a unit test for the [t1: 19% excl include_base; t2: 10% excl is_base_affected=False] case.

2. **Decimal precision on `factor_percent`**: stored with `digits=(16, 12)`. Rust `Decimal` type needs sufficient precision. Recommend `rust_decimal` with `PREC_28` or using `f64` only for the `factor` ratio (since it's used in `round(amount * factor)`).

3. **CABA multi-currency**: `_collect_tax_cash_basis_values` at L4136-4141 returns `None` if multiple currencies are involved. Odoo explicitly does NOT support CABA with mixed currencies on the same move. woa-rs should enforce the same restriction at validation time.

4. **`use_in_tax_closing` semantics for income/expense accounts**: repartition lines posting to income or expense accounts (e.g. non-deductible input VAT directly expensed) have `use_in_tax_closing=False` — they are NOT included in the periodic VAT closing. This is a gotcha for partial-deductibility scenarios.

5. **`total_void` in `compute_all`**: amounts where `account_id` is None contribute to `total_void` (not `total_excluded`). This is relevant for "informational" tax lines that should appear on documents but not generate postings. Confirm if woa-rs needs this distinction.

6. **Tag name sign encoding**: the leading `-` convention (e.g. `"-Kz89"`) is string-level. woa-rs should store tag names verbatim and derive sign at query time via `STARTS_WITH(formula, '-')` as odoo does.

---

## Depth-proof footer

Read: `/home/user/odoo/addons/account/models/account_tax.py` lines=5210 depth=full
Read: `/home/user/odoo/addons/account/models/account_account_tag.py` lines=141 depth=full
Read: `/home/user/odoo/addons/account/models/account_move.py` lines=4080-4148 depth=full (CABA section)
Read: `/home/user/odoo/addons/account/models/company.py` lines=131-134 depth=skim
