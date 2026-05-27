RICHNESS-LANE-OK

# L3-K7-TAX — USt / VAT Computation + Fiscal Position Mapping
**Lane:** K7 TAX  
**Date:** 2026-05-26  
**K-step:** K7 (USt/ELSTER + tax); feeds K8 (USt-VA report line mapping)  
**Target modules (woa-rs):** `src/erp/tax.rs` (to be created), `crates/skr_data/` (data already vendored)

---

## 1. Scope + Odoo Files Read

| File | Lines | Depth |
|---|---|---|
| `/home/user/odoo/addons/account/models/account_tax.py` | 5210 | full |
| `/home/user/odoo/addons/account/models/partner.py` | 1169 | full |
| `/home/user/odoo/addons/l10n_de/data/template/account.tax-de_skr03.csv` | 244 | full |
| `/home/user/odoo/addons/l10n_de/data/template/account.tax-de_skr04.csv` | ~244 | full (first 60 lines shown; structure identical to skr03) |
| `/home/user/odoo/addons/l10n_de/data/template/account.tax.group-de_skr03.csv` | 8 | full |
| `/home/user/odoo/addons/l10n_de/data/template/account.tax.group-de_skr04.csv` | 8 | full |
| `/home/user/odoo/addons/l10n_de/data/template/account.fiscal.position-de_skr03.csv` | 32 | full |
| `/home/user/odoo/addons/l10n_de/data/template/account.fiscal.position-de_skr04.csv` | 32 | full |

---

## 2. Rule Sections

---

### R1 — `_flatten_taxes_and_sort_them` / Group flattening

**File:** `account_tax.py:L892–L918`

**Axis-1 Rich-AST Spec:**

```
FUNCTION _flatten_taxes_and_sort_them(self: Recordset[AccountTax])
  -> (sorted_taxes: Recordset, group_per_tax: Dict[int, AccountTax])

sort_key = (tax.sequence, tax.id or None)   # None sorts before integers (Python None < int)

group_per_tax = {}
sorted_taxes  = empty recordset
for tax in self.sorted(key=sort_key):       # outer pass — group-level sequence
    if tax.amount_type == 'group':
        children = tax.children_tax_ids.sorted(key=sort_key)
        sorted_taxes |= children
        for child in children:
            group_per_tax[child.id] = tax   # maps child → parent group
    else:
        sorted_taxes |= tax                 # non-group added directly
RETURN (sorted_taxes, group_per_tax)
```

Key properties:
- Group taxes (`amount_type == 'group'`) are NEVER in `sorted_taxes` — only their children are.
- Children inherit the group's position via outer sort ordering.
- Example: `[G(seq=2), B([A,D,F]), E(seq=5), C(seq=4)]` → `[A,D,F,C,E,G]` (alphabetic = sequence order).
- `_sql_constraints`: `@api.constrains('children_tax_ids', 'type_tax_use')` prevents nested groups (ValidationError: "Nested group of taxes are not allowed").
- `flatten_taxes_hierarchy()` at L4855–L4856 is just a thin alias: `return self._flatten_taxes_and_sort_them()[0]`.

**Axis classification:** DETERMINISTIC  
**Ontology:** `odoo:account.tax` → see R7 below  
**K-step:** K7  
**woa-rs target:** `src/erp/tax.rs::flatten_and_sort`

---

### R2 — `_batch_for_taxes_computation` — batching into co-computed groups

**File:** `account_tax.py:L920–L971`

**Axis-1 Rich-AST Spec:**

Batches group taxes that must be solved simultaneously (e.g. a batch of price-included percent taxes share a denominator).

```
FUNCTION _batch_for_taxes_computation(
    self,
    special_mode: False | 'total_excluded' | 'total_included' = False,
    filter_tax_function: Optional[Callable] = None,
) -> {
    'batch_per_tax': Dict[int, Recordset],
    'group_per_tax': Dict[int, AccountTax],
    'sorted_taxes': Recordset,
}

(sorted_taxes, group_per_tax) = self._flatten_taxes_and_sort_them()
if filter_tax_function:
    sorted_taxes = sorted_taxes.filtered(filter_tax_function)

batch = empty
is_base_affected = False
# Traverse in REVERSE order to group consecutive same-type taxes
for tax in reversed(sorted_taxes):
    if batch is not empty:
        same_batch = (
            tax.amount_type == batch[0].amount_type
            AND (special_mode OR tax.price_include == batch[0].price_include)
            AND tax.include_base_amount == batch[0].include_base_amount
            AND (
                (tax.include_base_amount AND NOT is_base_affected)
                OR NOT tax.include_base_amount
            )
        )
        if NOT same_batch:
            flush batch → batch_per_tax
            batch = empty
    is_base_affected = tax.is_base_affected
    batch |= tax

flush final batch → batch_per_tax
```

The batch determines the denominator for price-included percent taxes: all taxes in a batch share `total_percentage = sum(tax.amount for tax in batch) / 100.0`.

**Axis classification:** DETERMINISTIC  
**K-step:** K7

---

### R3 — `_get_tax_details` — the core per-line computation engine

**File:** `account_tax.py:L1134–L1332`

**Axis-1 Rich-AST Spec (CRITICAL — reproduce exactly):**

```
FUNCTION _get_tax_details(
    self,
    price_unit: float,
    quantity: float,
    precision_rounding: float = 0.01,   # currency.rounding
    rounding_method: 'round_per_line' | 'round_globally' = 'round_per_line',
    product: Optional[product.product] = None,
    product_uom: Optional[uom.uom] = None,
    special_mode: False | 'total_excluded' | 'total_included' = False,
    manual_tax_amounts: Optional[Dict] = None,
    filter_tax_function: Optional[Callable] = None,
) -> {
    'total_excluded': float,
    'total_included': float,
    'taxes_data': List[Dict],   # one per leaf tax in sorted_taxes
}
```

**Step 1 — Setup:**
```
batching = self._batch_for_taxes_computation(special_mode, filter_tax_function)
sorted_taxes = batching['sorted_taxes']

# Initialize per-tax data dict
for tax in sorted_taxes:
    if tax.has_negative_factor:   # reverse-charge marker
        price_include = False     # always treat as price-excluded
    elif special_mode == 'total_included':
        price_include = True
    elif special_mode == 'total_excluded':
        price_include = False
    else:
        price_include = tax.price_include

    taxes_data[tax.id] = {
        'tax': tax,
        'price_include': price_include,
        'group': batching['group_per_tax'].get(tax.id),
        'batch': batching['batch_per_tax'][tax.id],
        'extra_base_for_tax': 0.0,
        'extra_base_for_base': 0.0,
    }
    if tax.has_negative_factor:
        reverse_charge_taxes_data[tax.id] = {..., 'is_reverse_charge': True}
```

**Step 2 — raw_base:**
```
raw_base = quantity * price_unit
if rounding_method == 'round_per_line':
    raw_base = float_round(raw_base, precision_rounding=precision_rounding)
```

**Step 3 — Evaluation order (three passes):**

Pass A: FIXED taxes in REVERSE order (so they can affect price-included bases that follow):
```
for tax in reversed(sorted_taxes):
    eval _eval_tax_amount_fixed_amount(tax, batch, raw_base + extra_base_for_tax, ctx)
    # fixed: sign = -1 if price_unit < 0 else 1
    #        result = sign * quantity * tax.amount
    # → records tax_amount; if rounding_method == 'round_per_line': float_round(...)
    # → calls _propagate_extra_taxes_base to update extra_base for other taxes
```

Pass B: PRICE-INCLUDED taxes in REVERSE order:
```
for tax in reversed(sorted_taxes):
    if taxes_data[tax.id]['price_include'] and 'tax_amount' not in taxes_data[tax.id]:
        eval _eval_tax_amount_price_included(tax, batch, raw_base + extra_base_for_tax, ctx)
```

  `_eval_tax_amount_price_included` (L1094–L1112):
  ```
  if amount_type == 'percent':
      total_percentage = sum(t.amount for t in batch) / 100.0
      # If all taxes in batch share denominator:
      to_price_excluded_factor = 1 / (1 + total_percentage)  # 0.0 if total_percentage == -1
      return raw_base * to_price_excluded_factor * self.amount / 100.0

  if amount_type == 'division':
      return raw_base * self.amount / 100.0
      # NOTE: division price-included is raw_base * rate directly (NO denominator)
  ```

Pass C: PRICE-EXCLUDED taxes in NORMAL order:
```
for tax in sorted_taxes:
    if not taxes_data[tax.id]['price_include'] and 'tax_amount' not in taxes_data[tax.id]:
        eval _eval_tax_amount_price_excluded(tax, batch, raw_base + extra_base_for_tax, ctx)
```

  `_eval_tax_amount_price_excluded` (L1114–L1132):
  ```
  if amount_type == 'percent':
      return raw_base * self.amount / 100.0

  if amount_type == 'division':
      total_percentage = sum(t.amount for t in batch) / 100.0
      incl_base_multiplicator = 1.0 if total_percentage == 1.0 else 1 - total_percentage
      return raw_base * self.amount / 100.0 / incl_base_multiplicator
      # WARNING: division price-excluded divides by (1 - sum_rates). If rates sum to 1.0,
      # multiplicator is forced to 1.0 to avoid division by zero.
  ```

**Step 4 — Base amounts (reverse pass):**
```
subsequent_taxes = empty
for tax in reversed(sorted_taxes):
    total_tax_amount = sum(taxes_data[t.id]['tax_amount'] for t in batch)
    + reverse_charge amounts if has_negative_factor

    base = raw_base + tax_data['extra_base_for_base']
    if price_include AND special_mode in (False, 'total_included'):
        base -= total_tax_amount    # CRITICAL: removes total batch tax from base for price-included
    tax_data['base'] = base

    if tax.include_base_amount:
        tax_data['taxes'] |= subsequent_taxes   # subsequent taxes affected by this one
    if tax.is_base_affected:
        subsequent_taxes |= tax
```

**Step 5 — rounding if round_per_line:**
```
if rounding_method == 'round_per_line':
    each tax_amount = float_round(tax_amount, precision_rounding=precision_rounding)
    # Applied INSIDE add_tax_amount_to_results at step 3 time (L1179-1180)
```

**Step 6 — Totals:**
```
total_excluded = taxes_data_list[0]['base']     # base of FIRST tax in list
tax_amount = sum(td['tax_amount'] for td in taxes_data_list)
total_included = total_excluded + tax_amount

# Edge case: no taxes
if not taxes_data_list:
    total_included = total_excluded = raw_base
```

**Axis classification:** DETERMINISTIC  
**K-step:** K7  
**woa-rs target:** `src/erp/tax.rs::get_tax_details`

---

### R4 — `_propagate_extra_taxes_base` — cross-tax base propagation

**File:** `account_tax.py:L973–L1077`

**Axis-1 Rich-AST Spec:**

This is called after each tax_amount is computed to update `extra_base_for_tax` and `extra_base_for_base` for other taxes in the sequence.

```
FUNCTION _propagate_extra_taxes_base(tax, taxes_data, special_mode):
    # get_tax_before(): yields taxes that appear BEFORE tax in sorted_taxes
    # get_tax_after():  yields taxes that appear AFTER  tax in sorted_taxes

    add_extra_base(other_tax, sign):
        tax_amount = taxes_data[tax.id]['tax_amount']
        if 'tax_amount' NOT in taxes_data[other_tax.id]:
            taxes_data[other_tax.id]['extra_base_for_tax'] += sign * tax_amount
        taxes_data[other_tax.id]['extra_base_for_base'] += sign * tax_amount
        # NOTE: extra_base_for_base is ALWAYS updated (affects base display)
        #       extra_base_for_tax  is only updated if other_tax NOT yet computed

    if tax.price_include:
        if special_mode in (False, 'total_included'):
            if tax.include_base_amount:
                for other_tax in get_tax_after():
                    if NOT other_tax.is_base_affected:
                        add_extra_base(other_tax, -1)
            else:
                for other_tax in get_tax_after():
                    add_extra_base(other_tax, -1)
            for other_tax in get_tax_before():
                add_extra_base(other_tax, -1)
        else:  # special_mode == 'total_excluded'
            if tax.include_base_amount:
                for other_tax in get_tax_after():
                    if other_tax.is_base_affected:
                        add_extra_base(other_tax, +1)

    elif NOT tax.price_include:
        if special_mode in (False, 'total_excluded'):
            if tax.include_base_amount:
                for other_tax in get_tax_after():
                    if other_tax.is_base_affected:
                        add_extra_base(other_tax, +1)
        else:  # special_mode == 'total_included'
            if NOT tax.include_base_amount:
                for other_tax in get_tax_after():
                    add_extra_base(other_tax, -1)
            for other_tax in get_tax_before():
                add_extra_base(other_tax, -1)
```

**Axis classification:** DETERMINISTIC  
**K-step:** K7

---

### R5 — `_add_tax_details_in_base_line` — the outer driver: discount, rate, rounding

**File:** `account_tax.py:L1733–L1805`

**Axis-1 Rich-AST Spec:**

```
FUNCTION _add_tax_details_in_base_line(base_line, company, rounding_method=None):
    rounding_method = rounding_method or company.tax_calculation_rounding_method
    # company.tax_calculation_rounding_method is 'round_per_line' (default) or 'round_globally'

    price_unit_after_discount = base_line['price_unit'] * (1 - base_line['discount'] / 100.0)

    taxes_computation = base_line['tax_ids']._get_tax_details(
        price_unit=price_unit_after_discount,
        quantity=base_line['quantity'],
        precision_rounding=base_line['currency_id'].rounding,
        rounding_method=rounding_method,
        product=base_line['product_id'],
        product_uom=base_line['product_uom_id'],
        special_mode=base_line['special_mode'],
        filter_tax_function=base_line['filter_tax_function'],
    )

    # Non-deductible (reverse charge): strip is_reverse_charge entries from total
    if base_line['special_type'] == 'non_deductible':
        for tax_data in taxes_data:
            if tax_data.get('is_reverse_charge'):
                taxes_computation['total_included'] -= tax_data['tax_amount']
                # remove from list

    rate = base_line['rate']   # FX rate (1.0 for domestic EUR)

    tax_details = {
        'raw_total_excluded_currency': taxes_computation['total_excluded'],
        'raw_total_excluded':          taxes_computation['total_excluded'] / rate (or 0.0),
        'raw_total_included_currency': taxes_computation['total_included'],
        'raw_total_included':          taxes_computation['total_included'] / rate (or 0.0),
        'taxes_data': [],
    }

    if rounding_method == 'round_per_line':
        tax_details['raw_total_excluded'] = company.currency_id.round(tax_details['raw_total_excluded'])
        tax_details['raw_total_included'] = company.currency_id.round(tax_details['raw_total_included'])

    for tax_data in taxes_computation['taxes_data']:
        tax_amount  = tax_data['tax_amount'] / rate (or 0.0)
        base_amount = tax_data['base_amount'] / rate (or 0.0)
        if rounding_method == 'round_per_line':
            tax_amount  = company.currency_id.round(tax_amount)
            base_amount = company.currency_id.round(base_amount)
        tax_details['taxes_data'].append({
            **tax_data,
            'raw_tax_amount_currency':  tax_data['tax_amount'],
            'raw_tax_amount':           tax_amount,
            'raw_base_amount_currency': tax_data['base_amount'],
            'raw_base_amount':          base_amount,
        })
```

**Two-currency pattern:** every amount exists in both `_currency` (transaction currency) and without suffix (company/local currency). The conversion is `/ rate`. `rate = 1.0` for EUR-only setups (Stefan's case).

**Axis classification:** DETERMINISTIC  
**K-step:** K7

---

### R6 — `_round_base_lines_tax_details` + `_round_tax_details_tax_amounts` — global rounding

**File:** `account_tax.py:L2179–L2288` and `L1890–L1987`

**Axis-1 Rich-AST Spec:**

Global rounding is needed only when `rounding_method == 'round_globally'`. The pattern:

1. **Raw rounding** (L2237–2248): copy `raw_*` → rounded fields using `currency.round(...)`.
2. **Apply manual_tax_amounts** (L2250–2273): override individual tax/base amounts from stored overrides.
3. **Compute total_included + delta_total_excluded** (L2275–2288): `total_included = total_excluded + sum(tax_amounts)`, `delta_total_excluded = 0.0` (then adjusted in step 4).
4. **`_round_tax_details_tax_amounts`** (L1890–1987): Aggregates `raw_total_tax_amount` per tax across all lines, rounds it globally, distributes the delta back to individual lines using `_distribute_delta_amount_smoothly` (proportional distribution, largest-first to minimize leftover).
5. **`_round_tax_details_base_lines`** (L1988–2097): Adjusts `delta_total_excluded` so that globally `round(sum(raw_total_excluded)) == sum(total_excluded + delta)`.

```
_distribute_delta_amount_smoothly(precision_digits, delta_amount, target_factors):
    # Converts delta to integer units of precision, distributes proportionally
    # Remainder distributed one unit at a time to factors sorted by largest weight first
    precision_rounding = 10^(-precision_digits)
    nb_of_errors = round(abs(delta / precision_rounding))
    # ... sorted proportional distribution
```

**Rounding mode for base amounts** (`mode` parameter):
- `'mixed'` (default): uses `'included'` logic if ALL non-zero taxes are price-included, else `'excluded'`.
- `'excluded'`: round base independently from tax.
- `'included'`: round `base + tax` together, then derive base as `round(total) - tax`.

**Axis classification:** DETERMINISTIC  
**K-step:** K7

---

### R7 — `_add_accounting_data_to_base_line_tax_details` — repartition lines

**File:** `account_tax.py:L2362–L2496`

**Axis-1 Rich-AST Spec:**

This is the bridge from computed tax amounts to actual accounting journal entries. Called AFTER rounding.

```
FUNCTION _add_accounting_data_to_base_line_tax_details(base_line, company, include_caba_tags=False):
    is_refund = base_line['is_refund']
    repartition_lines_field = 'refund_repartition_line_ids' if is_refund else 'invoice_repartition_line_ids'

    # Tags on BASE line (for USt-VA line mapping)
    base_line['tax_tag_ids'] = product.account_tag_ids (if product) union
        for each tax_data (not is_reverse_charge, on_invoice exigibility):
            tax[repartition_lines_field].filtered(repartition_type=='base').tag_ids

    for tax_data in taxes_data:
        # REVERSE CHARGE handling (has_negative_factor):
        if is_reverse_charge:
            tax_reps = filter(repartition_type=='tax' AND factor < 0.0)
            tax_rep_sign = -1.0
        else:
            tax_reps = filter(repartition_type=='tax' AND factor >= 0.0)
            tax_rep_sign = 1.0

        for tax_rep in tax_reps:
            tax_rep_data = {
                'tax_rep': tax_rep,
                'tax_amount_currency': currency.round(tax_amount_currency * tax_rep.factor * tax_rep_sign),
                'tax_amount':          company_currency.round(tax_amount * tax_rep.factor * tax_rep_sign),
                'account': tax_rep._get_aml_target_tax_account() or base_line['account_id'],
                    # _get_aml_target_tax_account: returns cash_basis_transition_account if on_payment
                    #                              else tax_rep.account_id
            }

        # Distribute rounding delta across repartition lines:
        # sorted by (-abs(tax_amount_currency), -abs(tax_amount)) — largest first
        # _distribute_delta_amount_smoothly fills remainder

        # Tags on TAX repartition line:
        tax_rep_data['tax_tags'] = product_tags union
            (if on_invoice): tax_rep.tag_ids
        tax_rep_data['taxes'] = tax_data['taxes']   # for include_base_amount chains
```

**Refund sign handling** (L1504–L1540, `_turn_base_line_is_refund_flag_off`):
- When `is_refund=True`, the `repartition_lines_field` switches to `'refund_repartition_line_ids'`.
- The sign convention for refunds is carried by `base_line['sign']` (typically -1 for credit notes, +1 for invoices).
- In `_prepare_tax_lines` (L3033–L3126): `amount_currency += sign * tax_rep_data['tax_amount_currency']`.
- To negate a refund programmatically: `_turn_base_line_is_refund_flag_off` negates quantity and all amounts in tax_details.

**Axis classification:** DETERMINISTIC  
**K-step:** K7 (produces K3 journal entry data)

---

### R8 — `compute_all` — legacy public API

**File:** `account_tax.py:L4864–L4980`

**Axis-1 Rich-AST Spec:**

```
FUNCTION compute_all(
    self,
    price_unit, currency=None, quantity=1.0, product=None, partner=None,
    is_refund=False, handle_price_include=True, include_caba_tags=False,
    rounding_method=None,
) -> {
    'base_tags': List[int],
    'taxes': List[Dict],    # one per repartition line (NOT per tax!)
    'total_excluded': float,
    'total_included': float,
    'total_void':  float,   # total of reps with no account_id
}
```

**special_mode resolution:**
```
if 'force_price_include' in context:
    special_mode = 'total_included' if context['force_price_include'] else 'total_excluded'
elif not handle_price_include:
    special_mode = 'total_excluded'    # ignores all price_include flags
else:
    special_mode = False               # normal: respect each tax's price_include
```

**Output construction:**
```
company = self[0].company_id._accessible_branches()[:1] or self[0].company_id
currency = currency or company.currency_id

base_line = _prepare_base_line_for_taxes_computation(None, partner_id=partner, ...)
_add_tax_details_in_base_line(base_line, company, rounding_method=rounding_method)
_add_accounting_data_to_base_line_tax_details(base_line, company, include_caba_tags)
    # NOTE: context 'compute_all_use_raw_base_lines'=True → uses raw_tax_amount_currency
    #       instead of rounded tax_amount_currency for repartition computation

total_excluded = raw_total_excluded_currency   # NOTE: RAW, not rounded (yet)
total_included = raw_total_included_currency

for tax_data in tax_details['taxes_data']:
    for tax_rep_data in tax_data['tax_reps_data']:
        taxes.append({
            'id':                     tax.id,
            'name':                   tax.name (localized via partner.lang if partner),
            'amount':                 tax_rep_data['tax_amount_currency'],
            'base':                   tax_data['raw_base_amount_currency'],
            'sequence':               tax.sequence,
            'account_id':             tax_rep_data['account'].id,
            'analytic':               tax.analytic,
            'use_in_tax_closing':     rep_line.use_in_tax_closing,
            'is_reverse_charge':      tax_data['is_reverse_charge'],
            'price_include':          tax.price_include,
            'tax_exigibility':        tax.tax_exigibility,
            'tax_repartition_line_id': rep_line.id,
            'group':                  tax_data['group'],
            'tag_ids':                tax_rep_data['tax_tags'].ids,  # USt-VA tags
            'tax_ids':                tax_rep_data['taxes'].ids,
        })
        if NOT rep_line.account_id:
            total_void += tax_rep_data['tax_amount_currency']

if context.get('round_base', True):    # default True
    total_excluded = currency.round(total_excluded)
    total_included = currency.round(total_included)
```

**Critical edge cases:**
- Empty self (`not self`): returns all-zero result with `company = self.env.company`.
- `_fix_tax_included_price` (L4994–L5003): subtracts tax from price if product has price-included taxes NOT applicable to the line.
- `_fix_tax_included_price_company` (L5005–L5011): same but filters by `company_id` first.

**Axis classification:** DETERMINISTIC  
**K-step:** K7

---

### R9 — `AccountTaxRepartitionLine` model

**File:** `account_tax.py:L5142–L5210`

**Axis-1 Rich-AST Spec:**

Fields that matter for computation:
```
factor_percent: Float(digits=(16,12))   # e.g. 100.0 or -100.0
factor:         Computed = factor_percent / 100.0   # e.g. 1.0 or -1.0
repartition_type: 'base' | 'tax'
document_type:    'invoice' | 'refund'
account_id:       Many2one account.account (can be empty → no posting)
tag_ids:          Many2many account.account.tag (USt-VA grid tags)
use_in_tax_closing: Computed Boolean
    = repartition_type == 'tax'
      AND account_id
      AND account_id.internal_group NOT IN ('income', 'expense')
```

`_get_aml_target_tax_account(force_caba_exigibility=False)`:
```
if NOT force_caba_exigibility AND tax.tax_exigibility == 'on_payment'
   AND NOT context.get('caba_no_transition_account'):
    return tax.cash_basis_transition_account_id   # Cash Basis transition account
else:
    return self.account_id
```

**Negative factor / reverse charge pattern (§13b UStG):**
- `has_negative_factor = True` when any invoice repartition line has `factor < 0.0`.
- In §13b taxes (e.g. `tax_ust_19_13b_ausland_ohne_vst_skr03`): repartition has one line +100% (Vorsteuer account) and one line -100% (USt liability). The negative line is treated as `is_reverse_charge=True` in computation.
- `_add_accounting_data_to_base_line_tax_details` splits them: positive → normal tax rep, negative → reverse_charge rep.

**_sql_constraints** on repartition:
```
@api.constrains('invoice_repartition_line_ids', ...)
_validate_repartition_lines:
    - exactly ONE base line per document type
    - at least ONE tax repartition line per document type
    - invoice and refund must have SAME NUMBER of lines
    - same percentages in same order
    - sum of positive factors == 1.0 (100%)
    - if negative factors exist: sum of negative factors == -1.0 (-100%)
```

**Axis classification:** DETERMINISTIC  
**K-step:** K7, K3

---

### R10 — `AccountTaxGroup` model

**File:** `account_tax.py:L25–L68`

**Axis-1 Rich-AST Spec:**

```
Fields:
    name:                       Char (translatable)
    sequence:                   Integer default=10
    company_id:                 Many2one res.company (required)
    tax_payable_account_id:     Many2one account.account  → Verbindlichkeiten USt (e.g. 1776)
    tax_receivable_account_id:  Many2one account.account  → Forderungen VSt (e.g. 1545)
    advance_tax_payment_account_id: Many2one account.account → Vorauszahlung USt (e.g. 1780)
    country_id:                 Computed from company_id.account_fiscal_country_id or company_id.country_id
    preceding_subtotal:         Char (optional label before this group in invoice subtotal display)

_compute_country_id @depends('company_id'):
    group.country_id = company.account_fiscal_country_id or company.country_id
```

Tax groups in l10n_de (both SKR03 and SKR04):
- `tax_group_0`:   VAT 0%
- `tax_group_7`:   VAT 7% (Ermäßigter Steuersatz)
- `tax_group_55`:  VAT 5.5% (Land-/Forstwirtschaft)
- `tax_group_107`: VAT 10.7% (Land-/Forstwirtschaft)
- `tax_group_x`:   VAT x% (variable rate)
- `tax_group_19`:  VAT 19% (Regelsteuersatz)

Group accounts (SKR03): receivable=1545, payable=1797, advance=1780.
Group accounts (SKR04): receivable=1421, payable=3860, advance=3820.

**Axis classification:** DETERMINISTIC  
**K-step:** K7, K8

---

### R11 — `AccountFiscalPosition.map_tax` + `map_account`

**File:** `partner.py:L154–L166`

**Axis-1 Rich-AST Spec:**

```
FUNCTION map_tax(self: FiscalPosition, taxes: Recordset[AccountTax]) -> Recordset[AccountTax]:
    if not self:
        return taxes    # no fiscal position → identity

    if not self.tax_ids and taxes.fiscal_position_ids:
        return env['account.tax']   # empty FP with any tax linked removes all taxes
        # CRITICAL EDGE CASE: FPs used by tax units with no explicit tax_ids → removes all taxes

    # tax_map is a Binary field computed from:
    #   tax_map[src_tax.id] = [dest_tax.id, ...]
    # where dest_tax is in self.tax_ids and src_tax in dest_tax.original_tax_ids

    return env['account.tax'].browse(unique(
        tax_id
        for tax in taxes
        for tax_id in (self.tax_map or {}).get(tax.id, [tax.id])
        # If tax.id NOT in tax_map → identity mapping (keep original)
        # If tax.id IS in tax_map → replace with all dest tax IDs
    ))
```

```
FUNCTION map_account(self: FiscalPosition, account: AccountAccount) -> AccountAccount:
    return env['account.account'].browse(
        (self.account_map or {}).get(account.id, account.id)
    )
    # account_map: {src_account_id: dest_account_id}
    # built from account_fiscal_position_account (many2one pairs)
    # identity mapping if no match
```

**Ontology of tax_map:**
```
@depends('tax_ids')
_compute_tax_map:
    for dest_tax in self.tax_ids:
        for src_tax in dest_tax.original_tax_ids:
            tax_map[src_tax.id].append(dest_tax.id)
    # IMPORTANT: mapping is MANY-to-MANY — one src can map to multiple destinations
    # (all get inserted via unique() to deduplicate)
```

**Axis classification:** DETERMINISTIC (pure lookup table)  
**Ontology:** `odoo:account.fiscal.position` → FLAG — UNRESOLVED (see §3)  
**K-step:** K7  
**woa-rs target:** `src/erp/tax.rs::map_tax`, `src/erp/tax.rs::map_account`

---

### R12 — `AccountFiscalPosition._get_fiscal_position` — auto-apply logic

**File:** `partner.py:L246–L279`

**Axis-1 Rich-AST Spec:**

```
FUNCTION _get_fiscal_position(partner, delivery=None) -> FiscalPosition | empty:
    if not partner:
        return empty

    company = self.env.company

    # EU intra-community detection
    intra_eu = False
    vat_exclusion = False
    if company.vat and partner.vat:
        eu_country_codes = set of EU country codes (from base.europe ref)
        intra_eu = company.vat[:2] in eu_country_codes AND partner.vat[:2] in eu_country_codes
        vat_exclusion = company.vat[:2] == partner.vat[:2]

    # If same-country EU VAT or no delivery → use invoicing address
    if not delivery or (intra_eu AND vat_exclusion AND partner.country_id == company.country_id):
        delivery = partner

    # STEP 1: Manual override always wins
    manual = delivery.property_account_position_id or partner.property_account_position_id
    if manual:
        return manual    # early return

    # STEP 2: No country → no auto-apply
    if not partner.country_id:
        return empty

    # STEP 3: Search all auto_apply positions for this company, ordered by sequence
    all_auto_apply = self.search(
        _check_company_domain(company) + [('auto_apply', '=', True)]
        # ORDER BY sequence (model default)
    )
    return all_auto_apply._get_first_matching_fpos(delivery)
```

```
FUNCTION _get_first_matching_fpos(self: Recordset[FP], partner) -> FP | empty:
    # Sort: company-specific first (more parent_ids = deeper hierarchy),
    #       then by sequence ascending
    sorted_fpos = self.sorted(key=lambda f: (-len(f.company_id.parent_ids), f.sequence))

    for fpos in sorted_fpos:
        if ALL validation functions pass:
            return fpos
    return empty
```

```
_get_fpos_validation_functions(partner) returns list of lambdas:
    1. VAT required: not fpos.vat_required OR partner has valid VAT
    2. ZIP range:    not (fpos.zip_from AND fpos.zip_to) OR zip_from <= partner.zip <= zip_to
    3. State:        not fpos.state_ids OR partner.state_id in fpos.state_ids
    4. Country:      not fpos.country_id OR partner.country_id == fpos.country_id
    5. Country group: not fpos.country_group_id OR
                       (partner.country_id in group.country_ids
                        AND (not partner.state_id OR state not in group.exclude_state_ids))
    # ALL five must pass (short-circuit AND)
```

**Rule order for l10n_de (SKR03/SKR04):**
```
sequence 10: fiscal_position_domestic        (auto_apply=True,  country=DE)
sequence 20: fiscal_position_eu_vat_id       (auto_apply=True,  country_group=EU, vat_required=True)
sequence 40: fiscal_position_eu_no_id        (auto_apply=False) ← manual only
sequence 50: fiscal_position_non_eu          (auto_apply=False) ← manual only
sequence 60: fiscal_position_non_eu_service  (auto_apply=False) ← manual only
sequence 30: fiscal_position_eu_vat_id_service (auto_apply=False) ← manual only
```

Only two German fiscal positions auto-apply: Domestic (DE country) and EU with VAT ID (EU country group + VAT required). All others must be manually assigned on the partner.

**Axis classification:** DETERMINISTIC (ordered rule table, closed-form).  
The selection is an ordered priority list, not scoring — each rule is a conjunction of exact checks. NO heuristic.  
Rule order: company-specific depth first, then sequence ascending.  
The `intra_eu + vat_exclusion` shortcut to use invoicing address instead of delivery is deterministic (exact string prefix comparison on VAT numbers).  
**K-step:** K7  
**woa-rs target:** `src/erp/tax.rs::get_fiscal_position`

---

### R13 — `AccountTax` field computations and constraints

**File:** `account_tax.py:L71–L240`

**Axis-1 Rich-AST Spec:**

Key fields for computation:
```
amount_type:  'group' | 'fixed' | 'percent' | 'division'
    - group:    delegates to children_tax_ids
    - fixed:    tax = sign * quantity * amount  (sign from price_unit sign)
    - percent:  tax = base * amount / 100
    - division: tax = base * amount/100 / (1 - total_rate/100)  [price-excl]
                      or base * amount / 100                     [price-incl]
amount:       Float(digits=(16,4))   — e.g. 19.0 for 19%, or fixed amount in EUR

price_include: Computed, NOT stored
    @depends('price_include_override')
    = (price_include_override == 'tax_included')
      OR (company_price_include == 'tax_included' AND NOT price_include_override)
    # company_price_include is the company's default; override wins

include_base_amount: Boolean default=False
    # If True: this tax's amount is added to the base for subsequent taxes
is_base_affected: Boolean default=True
    # If True: taxes before this one (with include_base_amount) affect this tax's base

tax_exigibility: 'on_invoice' (default) | 'on_payment'
    # on_payment → Cash Basis → uses cash_basis_transition_account_id

sequence: Integer default=1 — processing order (lower = earlier)
```

`@api.constrains('company_id', 'name', 'type_tax_use', 'tax_scope', 'country_id')` `_constrains_name`:
- Tax names must be unique within (company hierarchy, name, type_tax_use, tax_scope, country_id).
- Enforced via `split_every(100, ...)` batches.

`@api.constrains('tax_group_id')` `validate_tax_group_id`:
- tax_group.country_id must match tax.country_id.

`@api.constrains('tax_exigibility', 'cash_basis_transition_account_id')`:
- If `on_payment`: cash_basis_transition_account must allow reconciliation.

**l10n_de_datev_code field:** `account_tax.l10n_de_datev_code` — present in CSV as e.g. `"3"` (tax_ust_19_skr03) or `"9"` (tax_vst_19_skr03). This is the DATEV Steuerschlüssel. Only set for domestic taxes; EU/export taxes have empty string.

**Axis classification:** DETERMINISTIC  
**K-step:** K7

---

### R14 — `_adapt_price_unit_to_another_taxes` — fiscal position price_unit adjustment

**File:** `account_tax.py:L1338–L1385`

**Axis-1 Rich-AST Spec:**

When a fiscal position maps a price-included tax to another tax, the price_unit must be adjusted:

```
FUNCTION _adapt_price_unit_to_another_taxes(price_unit, product, original_taxes, new_taxes, product_uom=None):
    # Only adapt if ALL original taxes are price-included
    if original_taxes == new_taxes or False in original_taxes.mapped('price_include'):
        return price_unit   # no-op

    # Find price without any tax (total_excluded)
    computation = original_taxes._get_tax_details(
        price_unit, 1.0, rounding_method='round_globally', ...
    )
    price_unit = computation['total_excluded']  # strip original taxes

    # Find new price_unit with new price-included taxes added back
    computation = new_taxes._get_tax_details(
        price_unit, 1.0, rounding_method='round_globally',
        special_mode='total_excluded',   # treat given price as tax-excluded
    )
    delta = sum(x['tax_amount'] for x in computation['taxes_data'] if x['tax'].price_include)
    return price_unit + delta   # add only price-included portions of new taxes
```

**Axis classification:** DETERMINISTIC  
**K-step:** K7

---

## 3. l10n_de Tax Code Catalogue

### 3a. Domestic taxes (SKR03, active, standard types)

| ID | Name (DE) | Rate | type_tax_use | DATEV code | Tags (invoice base / tax) | USt-VA line |
|---|---|---|---|---|---|---|
| tax_ust_19_skr03 | 19% USt | 19% | sale | 3 | 81_BASE / 81_TAX | Kz 81 |
| tax_ust_7_skr03 | 7% USt | 7% | sale | 2 | 86_BASE / 86_TAX | Kz 86 |
| tax_vst_19_skr03 | 19% VSt | 19% | purchase | 9 | (none) / 66 | Kz 66 |
| tax_vst_7_skr03 | 7% VSt | 7% | purchase | 8 | (none) / 66 | Kz 66 |

### 3b. EU intra-community (active, §4 Abs 1b / §89)

| ID | Scenario | Rate | Direction | Tags | VA line |
|---|---|---|---|---|---|
| tax_eu_19_purchase_skr03 | Innergem. Erwerb 19% | 19% | purchase | 89_BASE / 89_TAX(-100%) + 61 | Kz 89+61 |
| tax_eu_7_purchase_skr03  | Innergem. Erwerb 7%  | 7%  | purchase | 93_BASE / 93_TAX(-100%) + 61 | Kz 93+61 |
| tax_eu_sale_skr03 | Steuerfreie innergem. Lieferung | 0% | sale | 41 / (none) | Kz 41 |

### 3c. §13b Reverse Charge (Steuerschuldnerschaft des Leistungsempfängers)

| ID | Scenario | Rate | Tags base / tax reps |
|---|---|---|---|
| tax_ust_19_13b_ausland_ohne_vst_skr03 | Ausländ. Werklieferungen (ohne VSt) | 19% | 84 / +67\|85 (1785) + -100% (1787) |
| tax_ust_19_13b_eu_ohne_vst_skr03 | Sonst. EU-Leistungen (ohne VSt) | 19% | 46 / +67 (1785) + 47-100% (1787) |
| tax_ust_vst_19_purchase_13b_bau_skr03 | §13b Bauleistung Empfänger (19%/19%) | 19% | 60rc\|84 / +67\|85 (1577) + -100% (1787) |
| tax_ust_free_bau_skr03 | §13b Bauleistung Erbringer (0%) | 0% | 60 / (none) |

### 3d. Drittland (Non-EU export/import)

| ID | Scenario | Rate | Tags |
|---|---|---|---|
| tax_export_skr03 | Steuerfreie Ausfuhr §4 Nr. 1a | 0% sale | 43 |
| tax_import_19_and_payable_skr03 | Einfuhrumsatzsteuer 19% | 19% purchase | (none) / 62 (1588) + -100% (1788) |

### 3e. Fiscal positions (l10n_de SKR03 / SKR04)

| ID | Name | auto_apply | Trigger | Effect |
|---|---|---|---|---|
| fiscal_position_domestic_skr03 | Geschäftspartner Inland | Yes | country=DE | Identity (no remapping) |
| fiscal_position_eu_vat_id_partner_skr03 | EU mit USt-ID | Yes | EU country_group + vat_required | Remaps: USt-19 → EU-sale; VSt-19 → EU-purchase; account remaps (e.g. 8400→8125) |
| fiscal_position_eu_no_id_partner_skr03 | EU ohne USt-ID | No | manual | Different EU accounts, no VAT requirement |
| fiscal_position_eu_vat_id_partner_service_skr03 | EU Dienstleister | No | manual | Service-specific tax remaps |
| fiscal_position_non_eu_partner_skr03 | Drittland | No | manual | Export tax remaps; account remaps (8400→8120) |
| fiscal_position_non_eu_partner_service_skr03 | Drittland Dienstleister | No | manual | Service-specific non-EU remaps |

Tax remapping mechanism: `original_tax_ids` on the **destination** tax points to domestic taxes it replaces. The `tax_map` computed field inverts this.

---

## 4. Enterprise Gap / USt-Voranmeldung (K8 bridge)

**ENTERPRISE GAP FLAG:**

The `l10n_de_tax_statement` module (which provides the USt-Voranmeldung / Umsatzsteuer-Voranmeldung ELSTER XML export) is **NOT present** in the community clone at `/home/user/odoo/addons/`. 

Evidence: `find /home/user/odoo/addons -name "*.py" -path "*l10n_de*"` returns only `l10n_de` (community). No `l10n_de_reports`, no `l10n_de_tax_statement`, no `account_reports`.

What IS available in community:
- Tax **tag_ids** on repartition lines — these are the USt-VA grid codes (Kz 81, Kz 86, Kz 41, Kz 66, Kz 89, etc.). The tags are referenced by ID strings like `89_BASE`, `81_TAX`, etc. in the CSV.
- The `account.account.tag` records with `applicability='taxes'` encode the line number for each Kennziffer (Kz).
- `compute_all` returns `tag_ids` in each tax entry — this IS the K8 bridge.

**What the Enterprise module adds (not available here):**
- ELSTER XML serialization of the USt-Voranmeldung form.
- Pre-filled line aggregation summing tagged amounts to Kz-numbers.
- Electronic submission wrapper.

**Community alternative for K8:** The tag_ids returned by `compute_all` (and stored on `account.move.line.tax_tag_ids`) can be aggregated by grouping move lines by tag to produce Kz totals. This is the structural approach — the engine for generating the Voranmeldung must be built fresh in woa-rs, using the tag→Kz mapping derived from the l10n_de data.

**Note to K8 porter (lane L4):** The tag string IDs in the CSV (e.g. `81_BASE`, `81_TAX`, `89_BASE`, `89_TAX`) correspond directly to USt-VA Kennziffern. The `_BASE` suffix tags the base amount line; `_TAX` tags the tax amount line. Aggregate `account.move.line` where `tax_tag_ids` contains tag X to get Kz X total.

---

## 5. Ontology Mapping

### 5a. `odoo:account.tax`

**Current status in `crates/skr_data/src/odoo_alignment.rs`:** FILE DOES NOT EXIST in woa-rs (confirmed: `find /home/user/woa-rs -name "odoo_alignment.rs"` returns empty). The briefing references `lance-graph-callcenter/src/odoo_alignment.rs` which is not in this repo.

**FLAG — UNRESOLVED:** `odoo:account.tax` has no alignment row.

**Proposed mapping:**
```
odoo:account.tax
  → owl:equivalentClass fibo-fbc-fi-fi:Tax
    (FIBO: Financial Industry Business Ontology — Financial Instruments)
  → OGIT family: SMBAccounting / BillingCore (whichever covers tax records)
  → DOLCE marker: Quality
    (suffix `.tax` → Quality per briefing rule)
```

**Proposed alignment row:**
```rust
("account.tax", Some(OgitFamily::SmBAccounting), DolceMarker::Quality,
 "fibo-fbc-fi-fi:Tax")
```

### 5b. `odoo:account.tax.group`

**FLAG — UNRESOLVED**

**Proposed mapping:**
```
odoo:account.tax.group
  → owl:equivalentClass fibo-fbc-fi-fi:TaxCategory (or ubl:TaxCategory)
  → OGIT family: SMBAccounting
  → DOLCE marker: Quality (.group → grouped Quality)
```

### 5c. `odoo:account.tax.repartition.line`

**FLAG — UNRESOLVED**

**Proposed mapping:**
```
odoo:account.tax.repartition.line
  → owl:equivalentClass fibo-be-le-lp:TaxDistributionLine (or ubl:TaxSubtotal)
  → OGIT family: BillingCore
  → DOLCE marker: Perdurant (.line → Perdurant per briefing rule)
```

### 5d. `odoo:account.fiscal.position`

**FLAG — UNRESOLVED**

**Proposed mapping:**
```
odoo:account.fiscal.position
  → owl:equivalentClass fibo-fbc-pas-caa:TaxJurisdiction
    (FIBO: Regulatory Compliance, Tax Jurisdiction concept)
    Alternatively: ubl:TaxScheme or schema:TaxType
  → OGIT family: SMBAccounting (regulatory/compliance sub-family)
  → DOLCE marker: Abstract (.rule/.template → Abstract per briefing rule)
```

**Rationale:** A fiscal position is a mapping rule / tax regime, not a transaction event → Abstract.

### 5e. `odoo:account.fiscal.position.account`

**FLAG — UNRESOLVED**

**Proposed mapping:**
```
odoo:account.fiscal.position.account
  → owl:equivalentClass fibo-fbc-pas-caa:AccountMapping (construct)
  → OGIT family: SMBAccounting
  → DOLCE marker: Abstract (it's a mapping rule)
```

---

## 6. woa-rs Calibration

From `grep` output (Step 3):
- `/home/user/woa-rs/src/url.rs:L98`: `TAX_RESERVE_TOGGLE = "/api/tax-reserve/toggle"` — there is a tax reserve toggle endpoint.
- `/home/user/woa-rs/src/models/maintenance_contract.rs:L33`: `/// db.Float default=19.0 — f64 per POLICY §1 (tax rate snapshot)` — tax rate stored as f64 on maintenance_contract.
- No `src/erp/` directory exists yet.
- `crates/skr_data/` exists but contains only account/chart-of-accounts data (`Konto`, `KontoTyp`, `SkrRahmen`). No tax records, no fiscal positions, no repartition data.

**Gap assessment:**
- woa-rs has zero implementation of K7 tax computation logic.
- The SKR CSV tax data is in `/home/user/odoo/addons/l10n_de/data/template/` and not yet vendored into `crates/skr_data/`.
- The tax rate is hardcoded as `f64 = 19.0` in maintenance contracts — no dynamic tax lookup.
- No fiscal position logic exists.

**Recommendation for porter:** Create `src/erp/tax.rs` with the full deterministic algorithm from R1–R8. The l10n_de CSVs can be statically compiled into `crates/skr_data/` similarly to the existing `skr03.rs`/`skr04.rs` pattern (static arrays of structs).

---

## 7. Porter's Checklist — Non-Obvious Gotchas

1. **Batch denominator sharing (R2):** Price-included percent taxes in the SAME batch divide by a SHARED denominator `1 + sum(rates)`. Two 10% price-included taxes on the same line → each gets `base / 1.2 * 0.1`, NOT `base / 1.1 * 0.1` independently. The batch determines what goes in the denominator.

2. **Three-pass evaluation order (R3):** Fixed → price-included (reverse) → price-excluded (forward). This order is not arbitrary — fixed taxes must run first because their amount feeds `extra_base_for_tax` for price-included batches that follow.

3. **`extra_base_for_tax` vs `extra_base_for_base` (R4):** `extra_base_for_tax` only updates if the target tax has NOT yet been computed (guards against double-counting). `extra_base_for_base` ALWAYS updates (affects the displayed base amount even after computation).

4. **Price-included base formula (R3, Step 4):** `base = raw_base + extra_base_for_base - total_batch_tax_amount` only when `price_include AND special_mode in (False, 'total_included')`. Otherwise `base = raw_base + extra_base_for_base`.

5. **`total_excluded` = first tax's base, not `raw_base` (R3, Step 6):** When there are price-included taxes, `total_excluded` (the net base) is taken from `taxes_data_list[0]['base']`, which already has the tax stripped. This is the correct price-net-of-tax.

6. **Reverse-charge split (§13b) (R9):** A tax with `has_negative_factor=True` generates TWO entries in `taxes_data`: one normal (positive factor reps) and one `is_reverse_charge=True` (negative factor reps). The `compute_all` function includes both. The non-deductible filter strips the reverse-charge entry. Porter must handle both branches.

7. **`round_per_line` rounds DURING computation (R3 + R5):** `float_round(tax_amount, precision_rounding=...)` is applied inside `add_tax_amount_to_results` at computation time. For `round_globally`, rounding is deferred to `_round_base_lines_tax_details`.

8. **`compute_all` uses RAW amounts for repartition (R8):** The context key `compute_all_use_raw_base_lines=True` causes `_add_accounting_data_to_base_line_tax_details` to use `raw_tax_amount_currency` instead of the rounded `tax_amount_currency` when computing per-repartition-line amounts. This avoids double-rounding.

9. **`map_tax` edge case (R11):** If fiscal position has no `tax_ids` (empty) but the input taxes have `fiscal_position_ids` set, ALL taxes are removed. This is used by "tax units" to create zones without taxation.

10. **Fiscal position auto-apply: only two auto-apply in DE (R12):** Domestic (sequence 10) and EU-with-VAT-ID (sequence 20) are the only positions with `auto_apply=True`. All others require manual assignment on the partner's `property_account_position_id`.

11. **ZIP range comparison is string-based (R12):** `zip_from <= partner.zip <= zip_to` is a string comparison (alphabetic), not numeric. The `_convert_zip_values` method right-pads numeric zips with leading zeros to `max_length` to make string comparison behave numerically.

12. **`_get_first_matching_fpos` sorts by company depth THEN sequence (R12):** A fiscal position defined on a child company wins over one on the parent company even if it has a higher sequence number. The key is `(-len(company_id.parent_ids), sequence)`.

13. **USt-Voranmeldung line structure is Enterprise (§4):** The tag IDs (e.g. `81_BASE`, `81_TAX`) encode Kz numbers structurally, but the ELSTER XML serialization and the report engine that aggregates them is in `l10n_de_reports` (Enterprise). Community clone only has the tags and the raw `account.move.line.tax_tag_ids` storage.

14. **`division` amount_type (R3):** Price-excluded division: tax = `base * rate / (1 - total_rate)`. If `total_rate == 1.0` (100%), the divisor is forced to 1.0 to avoid division by zero. This is a legal edge case (100% division tax). Price-included division: tax = `base * rate` (no division by denominator — the price IS the included total).

15. **`include_base_amount` vs `is_base_affected` (R4):** These are paired flags. `t1.include_base_amount=True` means "my tax amount affects the base of subsequent taxes". `t2.is_base_affected=True` means "I accept being affected by preceding include_base_amount taxes". Both must be true for the effect to apply.

16. **`price_include` is COMPUTED, not stored directly (R13):** It derives from `price_include_override` (per-tax) and `company_price_include` (company-wide default). A porter must replicate this two-level override logic, not just read a stored boolean.

---

## 8. Axis-2 Classification Summary

All rules in this lane are **DETERMINISTIC**. No rule in K7 requires NARS delegation:

- Tax amount computation is pure arithmetic (closed-form formulas).
- Fiscal position selection is an ordered rule table (no scoring, no fuzzy matching).
- Repartition is table lookup + proportional delta distribution (deterministic rounding algorithm).
- Tax group assignment is a priority search (country + company filter, first match wins).

The only operationally "interesting" piece is the `_get_first_matching_fpos` partner-match traversal, but it is fully deterministic (ordered predicates, no weights). Classification: **DETERMINISTIC**, port to Rust directly.

---

## Read Depth Proof

```
Read: /home/user/odoo/addons/account/models/account_tax.py  lines=5210  depth=full
Read: /home/user/odoo/addons/account/models/partner.py       lines=1169  depth=full
Read: /home/user/odoo/addons/l10n_de/data/template/account.tax-de_skr03.csv             lines=244  depth=full
Read: /home/user/odoo/addons/l10n_de/data/template/account.tax-de_skr04.csv             lines=~244 depth=full (structural read, identical schema)
Read: /home/user/odoo/addons/l10n_de/data/template/account.tax.group-de_skr03.csv       lines=8    depth=full
Read: /home/user/odoo/addons/l10n_de/data/template/account.tax.group-de_skr04.csv       lines=8    depth=full
Read: /home/user/odoo/addons/l10n_de/data/template/account.fiscal.position-de_skr03.csv lines=32   depth=full
Read: /home/user/odoo/addons/l10n_de/data/template/account.fiscal.position-de_skr04.csv lines=32   depth=full
Read: /home/user/woa-rs/.claude/board/odoo-richness/BRIEFING.md                          lines=124  depth=thorough
Read: /home/user/woa-rs/crates/skr_data/src/lib.rs                                       lines=84   depth=full
```
