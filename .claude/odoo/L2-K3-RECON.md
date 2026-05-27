RICHNESS-LANE-OK

# L2-K3-RECON — Odoo Richness Export: K3 Reconciliation (Offene-Posten Matching)

**Lane:** L2 · **K-step:** K3 (Double-entry reconciliation / Offene-Posten Abgleich)
**Date:** 2026-05-26
**Status:** Draft — read-only analysis, NO Rust written, NO cargo run

---

## 1. Scope and Odoo Files Read

| File | Lines | Depth |
|---|---|---|
| `account/models/account_move_line.py` | 3742 | full (reconciliation region + residual computes: L240–295, L793–861, L2104–2948, L3100–3107, L3361–3391) |
| `account/models/account_partial_reconcile.py` | 706 | full |
| `account/models/account_full_reconcile.py` | 46 | full |

Calibration grep: `/home/user/woa-rs/src/` + `/home/user/woa-rs/crates/` — results examined, no existing reconciliation engine found (see §3 below).

---

## 2. Per-Rule Sections

---

### Rule R-1: Residual Amount Computation (`_compute_amount_residual`)

**Odoo source:** `account_move_line.py:L793–861`

#### Axis-1 Rich-AST Spec

**`@api.depends`:** `'debit', 'credit', 'amount_currency', 'account_id', 'currency_id', 'company_id', 'matched_debit_ids', 'matched_credit_ids'`

**Eligibility filter (L800):**
```
need_residual_lines = self.filtered(
    lambda x: x.account_id.reconcile
              or x.account_id.account_type in ('asset_cash', 'liability_credit_card')
)
```
Lines on non-reconcilable accounts that are NOT cash/credit-card → `amount_residual = 0.0`, `amount_residual_currency = 0.0`, `reconciled = False` immediately.

**SQL aggregation (L811–835) — two UNION ALL queries:**
```sql
SELECT part.debit_move_id AS line_id, 'debit' AS flag,
       COALESCE(SUM(part.amount), 0.0) AS amount,
       ROUND(SUM(part.debit_amount_currency), curr.decimal_places) AS amount_currency
FROM account_partial_reconcile part
JOIN res_currency curr ON curr.id = part.debit_currency_id
WHERE part.debit_move_id IN %s
GROUP BY part.debit_move_id, curr.decimal_places

UNION ALL

SELECT part.credit_move_id, 'credit',
       COALESCE(SUM(part.amount), 0.0),
       ROUND(SUM(part.credit_amount_currency), curr.decimal_places)
FROM account_partial_reconcile part
JOIN res_currency curr ON curr.id = part.credit_currency_id
WHERE part.credit_move_id IN %s
GROUP BY part.credit_move_id, curr.decimal_places
```
Result map: `{(line_id, 'debit'|'credit'): (amount, amount_currency)}`.

**Residual arithmetic (L845–860) — per eligible line:**
```
comp_curr = line.company_currency_id or self.env.company.currency_id
foreign_curr = line.currency_id or comp_curr

(debit_amount, debit_amount_currency) = amounts_map.get((line._origin.id, 'debit'), (0.0, 0.0))
(credit_amount, credit_amount_currency) = amounts_map.get((line._origin.id, 'credit'), (0.0, 0.0))

line.amount_residual = comp_curr.round(line.balance - debit_amount + credit_amount)
line.amount_residual_currency = foreign_curr.round(
    line.amount_currency - debit_amount_currency + credit_amount_currency
)
line.reconciled = (
    comp_curr.is_zero(line.amount_residual)
    and foreign_curr.is_zero(line.amount_residual_currency)
)
```

**Key invariants:**
- `balance = debit - credit` (odoo field, always the raw sign-bearing company-currency amount)
- Debit partials reduce the residual of the debit line (subtracted).
- Credit partials reduce the residual of the credit line (added, since credit_amount is negative in balance terms — but the SQL COALESCE always returns positive `amount`; the formula uses `+ credit_amount` because `balance` for a credit line is negative, so adding a positive credit_amount moves residual toward zero).
- `reconciled = True` only when BOTH company-currency residual AND foreign-currency residual are zero (critical for multi-currency lines).
- `_origin` is used to exclude ORM `NewId` virtual records from the SQL lookup.
- Flush of `account.partial.reconcile` and `res.currency.decimal_places` is forced before the SQL (L807–808).

**SQL index backing:** `_journal_id_neg_amnt_residual_idx` at L487 — `(journal_id) WHERE amount_residual < 0`.

**AXIS CLASSIFICATION:** DETERMINISTIC (arithmetic identity, closed-form rounding). Port directly.

**Ontology:**
`odoo:account.move.line` → `fibo:JournalEntryLine` → OGIT family `SMBInvoice` (0x81), slot `SLOT_JOURNAL_LINE` (0x06) → DOLCE Perdurant (curated override, `odoo_alignment.rs:L139–143`). **RESOLVED.**

**K-step:** K3
**woa-rs target:** `src/erp/reconcile.rs` — `fn compute_residual(line: &ErpOpenItemAR, partials: &[PartialReconcile]) -> ResidualAmounts`

---

### Rule R-2: `reconcile()` Public Entry Point

**Odoo source:** `account_move_line.py:L3100–3102`

```python
def reconcile(self):
    """ Reconcile the current move lines all together. """
    return self._reconcile_plan([self])
```

Trivial wrapper: passes `self` (the recordset of AMLs to reconcile) as a single-element plan list to `_reconcile_plan`.

**AXIS CLASSIFICATION:** DETERMINISTIC (delegation only).

**woa-rs target:** `src/erp/reconcile.rs` — `pub fn reconcile(lines: &[AmlRef]) -> ReconcileResult`

---

### Rule R-3: `remove_move_reconcile()` (Undo Reconciliation)

**Odoo source:** `account_move_line.py:L3104–3106`

```python
def remove_move_reconcile(self):
    """ Undo a reconciliation """
    (self.matched_debit_ids + self.matched_credit_ids).unlink()
```

The `unlink()` on `account.partial.reconcile` cascades (see R-9 below) to: reverse/unlink exchange-diff and CABA moves, unlink the full reconcile, reset matching numbers, update payment states.

**AXIS CLASSIFICATION:** DETERMINISTIC (cascade orchestration).

**woa-rs target:** `src/erp/reconcile.rs` — `pub fn remove_reconcile(lines: &[AmlRef]) -> UnreconcileResult`

---

### Rule R-4: `_reconcile_plan()` — Outer Orchestrator

**Odoo source:** `account_move_line.py:L2755–2777`

**Control flow:**
1. `plan_list, all_amls = self._optimize_reconciliation_plan(reconciliation_plan)` — validate, sort, split by currency, check eligibility.
2. Open `all_amls.move_id._check_balanced(move_container)` context manager.
3. Open `all_amls.move_id._sync_dynamic_lines(move_container)` context manager.
4. Call `self._reconcile_plan_with_sync(plan_list, all_amls)`.

**Critical guard:** the two context managers ensure the parent moves stay balanced and dynamic lines (e.g. tax lines) are kept in sync during the reconciliation mutation.

**AXIS CLASSIFICATION:** DETERMINISTIC (orchestration/guard).

**woa-rs target:** `src/erp/reconcile.rs`

---

### Rule R-5: `_optimize_reconciliation_plan()` — Sort, Split, Validate

**Odoo source:** `account_move_line.py:L2647–2739`

**Purpose:** Convert an arbitrary list of AML recordsets (the plan) into a validated, sorted, currency-split tree.

**Sort key (L2678–2684):**
```
(date_maturity or date, currency_id, amount_currency, balance)
```
Reduced mode (context `reduced_line_sorting=True`) drops `amount_currency` and `balance`: `(date_maturity or date, currency_id)`.

**Currency split (L2691–2700):** If a plan node contains lines in more than one currency, it is split into sub-nodes, one per currency. Lines with the same currency are grouped together.

**Eligibility validation per node (`_check_amls_exigibility_for_reconciliation`, L2609–2644):**
- Raises `UserError` if any line is already `reconciled` (unless it has a partial matching number starting with `'P'` — those are partially reconciled and can still be included).
- Raises `UserError` if any line's parent move is `cancelled`.
- Raises `UserError` if lines are from more than one account.
- Raises `UserError` if lines span more than one root company.
- Raises `UserError` if the account does not have `reconcile=True` AND is not `asset_cash`/`liability_credit_card`.

**AXIS CLASSIFICATION:** Sorting/splitting by date+currency: DETERMINISTIC. The *selection* of which open items to propose reconciling (candidate matching): see R-Axis2 tag below.

**Axis-2 component note:** `_optimize_reconciliation_plan` only processes lines the caller explicitly passes — it does NOT select candidate lines. The candidate selection heuristic belongs to `account.reconcile.model` (lane L5). Everything in R-5 is DETERMINISTIC given the input set.

**woa-rs target:** `src/erp/reconcile.rs`

---

### Rule R-6: `_reconcile_plan_with_sync()` — Core Execution Engine

**Odoo source:** `account_move_line.py:L2779–2947`

This is the central algorithm. Full control-flow:

#### Step 1 — Prefetch & Pre-hook (L2784–2803)
Force ORM cache population for `move_id`, `matched_debit_ids`, `matched_credit_ids` on all involved AMLs. Build `aml_values_map`:
```
{ aml: { 'aml': aml, 'amount_residual': aml.amount_residual,
          'amount_residual_currency': aml.amount_residual_currency,
          'parent_state': aml.parent_state } }
```
Call `_reconcile_pre_hook()` → records which invoices are `not_paid` vs `in_payment` (for post-hook state transition).

#### Step 2 — Prepare Partials (L2806–2820)
For each plan node, call `_prepare_reconciliation_plan(plan, aml_values_map)` (see R-7). Collect:
- `partials_values_list`: list of dicts for `account.partial.reconcile.create`.
- `exchange_diff_values_list`: list of exchange-diff move-vals dicts.

#### Step 3 — Create Partials (L2823–2831)
```python
partials = self.env['account.partial.reconcile'].create(partials_values_list)
```
Batch-create all partials in one ORM call. If context `add_caba_vals=True`: call `partials._set_draft_caba_move_vals()`.

#### Step 4 — Create Exchange Difference Moves (L2834–2849)
```python
exchange_moves = self._create_exchange_difference_moves(exchange_diff_values_list)
```
Then link each exchange_move to its partial via `partial.exchange_move_id = exchange_move` (iterating over the Cartesian product, using `used_exchange_moves` and `used_partials` sets to avoid double-linking).

#### Step 5 — Cash-Basis Tax Entries (L2852–2860)
```python
def is_cash_basis_needed(amls):
    return any(amls.company_id.mapped('tax_exigibility')) \
        and amls.account_id.account_type in ('asset_receivable', 'liability_payable')
```
If NOT `move_reverse_cancel` context AND NOT `no_cash_basis` context AND `is_cash_basis_needed`:
```python
plan['partials'].with_context(no_exchange_difference_no_recursive=False)
    ._create_tax_cash_basis_moves()
plan['partials']._set_draft_caba_move_vals()
```

#### Step 6 — Full Reconcile Detection (L2862–2923)

**`is_line_reconciled` predicate (L2865–2875):**
```
if aml.reconciled: return True
if not aml.matched_debit_ids and not aml.matched_credit_ids: return False
# Exchange difference case: balance=0 but amount_currency != 0
if has_multiple_currencies:
    return aml.company_currency_id.is_zero(aml.amount_residual)
else:
    return aml.currency_id.is_zero(aml.amount_residual_currency)
```
Note: `has_multiple_currencies` is True when `len(involved_amls.currency_id) > 1`.

**Full batch discovery (L2877–2899):**
```
number2lines = all_amls._reconciled_by_number()
for plan in plan_list:
    for aml in plan['amls']:
        if 'full_batch_index' already assigned: skip
        involved_amls = plan['amls']._filter_reconciled_by_number(number2lines)
        has_multiple_currencies = len(involved_amls.currency_id) > 1
        is_fully_reconciled = all(
            is_line_reconciled(involved_aml, has_multiple_currencies)
            for involved_aml in involved_amls
        )
        full_batches.append({
            'amls': involved_amls,
            'is_fully_reconciled': is_fully_reconciled,
        })
        # tag each involved aml with full_batch_index
```

**Full reconcile creation (L2912–2924):**
```python
full_reconcile_values_list = []
for full_batch in full_batches:
    if full_batch['is_fully_reconciled']:
        full_reconcile_values_list.append({
            'partial_reconcile_ids': [Command.link(p.id) for p in involved_partials],
            'reconciled_line_ids': [Command.link(aml.id) for aml in amls],
        })
self.env['account.full.reconcile'].create(full_reconcile_values_list)
```
Uses `Command.link` (NOT `Command.set`) to avoid triggering `unlink` which forces a flush.

#### Step 7 — CABA Rounding Auto-reconciliation (L2927–2945)
For any `caba_lines_to_reconcile` in a full_batch (populated by the CABA move creation path), reconcile the cash-basis transition-account rounding lines with the exchange-move counterparts.

#### Step 8 — Post-hook (L2947)
`all_amls._reconcile_post_hook(pre_hook_data)` → triggers `_invoice_paid_hook()` on invoices that transitioned to `paid`/`in_payment` state.

**AXIS CLASSIFICATION:** DETERMINISTIC (all arithmetic is closed-form). Port entirely to Rust.

**K-step:** K3
**woa-rs target:** `src/erp/reconcile.rs`

---

### Rule R-7: `_prepare_reconciliation_amls()` — Debit/Credit Pairing Loop

**Odoo source:** `account_move_line.py:L2503–2567`

**Algorithm:**
1. Partition `values_list` into two iterators: `debit_values_list` (lines with `balance > 0` or `amount_currency > 0`) and `credit_values_list` (lines with `balance < 0` or `amount_currency < 0`).
2. Iterative pairing loop:
   - Advance `debit_values` from iterator if exhausted.
   - Advance `credit_values` from iterator if exhausted.
   - Break if either iterator is exhausted.
   - Call `_prepare_reconciliation_single_partial(debit_values, credit_values)` (see R-8).
   - If `results['debit_values']` is None → debit AML is fully consumed → fetch next debit.
   - If `results['credit_values']` is None → credit AML is fully consumed → fetch next credit.
3. Track `fully_reconciled_aml_ids` (set of AML ids that are now zero-residual).

**Key note on ordering:** The **order of lines in `values_list` determines the matching order**. This is set by `_optimize_reconciliation_plan`'s sort key: `(date_maturity or date, currency_id, amount_currency, balance)`. Older maturity first; within same maturity, by currency then amount. This is FIFO / date-of-maturity matching.

**Multi-partner within a plan (L2585–2586, in `_prepare_reconciliation_plan`):**
```python
if len(remaining_amls.mapped('partner_id')) > 1:
    remaining_amls = remaining_amls.sorted(lambda aml: (aml.partner_id and aml.partner_id.id) or False)
```
When a plan node contains multiple partners, lines are additionally sorted by partner before the debit/credit pairing.

**AXIS CLASSIFICATION:** DETERMINISTIC (the pairing loop is a deterministic iterator, not a heuristic). The ORDER in which lines are passed determines the match — that ordering is determined upstream. Port to Rust.

**woa-rs target:** `src/erp/reconcile.rs`

---

### Rule R-8: `_prepare_reconciliation_single_partial()` — Single Pair Amount Math

**Odoo source:** `account_move_line.py:L2194–2500`

This is the mathematical heart. Full Axis-1 detail:

#### Step 1 — Determine Reconciliation Currency (L2239–2248)

Priority:
1. If `debit_currency != company_currency` AND both debit and credit have residual in `debit_currency` → `recon_currency = debit_currency`.
2. Else if `credit_currency != company_currency` AND both have residual in `credit_currency` → `recon_currency = credit_currency`.
3. Else → `recon_currency = company_currency`.

#### Step 2 — Residual Availability (`_prepare_move_line_residual_amounts`, L2116–2191)

For each side, builds `available_residual_per_currency: Dict[Currency, {residual, rate}]`:
- If `remaining_amount != 0`: adds company_currency entry with `rate=1`.
- If `currency != company_currency` and `remaining_amount_curr != 0`: adds foreign_currency entry with `rate = abs(amount_currency / balance)` (accounting rate).
- Special case for receivable/payable lines: if the line is in company currency but the counterpart is in a foreign currency, converts using `get_odoo_rate` (which checks context `forced_rate_from_register_payment`, then falls back to the payment line's accounting rate, then to the odoo FX table rate on the invoice date or AML date).

#### Step 3 — Exchange-Line Mode Detection (L2273–2279)

`exchange_line_mode = True` when:
- `recon_currency == company_currency`
- `debit_currency == credit_currency`
- At least one side has no residual in the foreign currency.

This handles reconciling exchange-difference lines with their counterparts.

#### Step 4 — Full/Partial Match Detection (L2282–2285)

```
compare_amounts = recon_currency.compare_amounts(recon_debit_amount, recon_credit_amount)
min_recon_amount = min(recon_debit_amount, recon_credit_amount)
debit_fully_matched = (compare_amounts <= 0)   # debit amount <= credit amount
credit_fully_matched = (compare_amounts >= 0)  # credit amount <= debit amount
```

#### Step 5 — Compute Partial Amounts (L2301–2396)

**Case A: `recon_currency == company_currency`**
```
partial_amount = min_recon_amount   # in company currency
if debit_rate:
    partial_debit_amount_currency = debit_currency.round(debit_rate * min_recon_amount)
    # clamp to remaining
    partial_debit_amount_currency = min(partial_debit_amount_currency, remaining_debit_amount_curr)
else:
    partial_debit_amount_currency = 0.0
# same for credit side (using -remaining_credit_amount_curr as upper bound)
```

**Case B: `recon_currency != company_currency` (foreign-currency reconciliation)**

Range-based anti-exchange-diff logic (L2335–2384):
```
partial_debit_amount_range = get_amount_range_after_rate(
    currency_from=debit_currency, currency_to=company_currency,
    amount=min_recon_amount, rate=(1/debit_rate) if debit_rate else 0.0
)
# range = [round((amount - half_rounding) * rate),
#          round(amount * rate),
#          round((amount + half_rounding) * rate)]
partial_debit_amount = min(range[1], remaining_debit_amount)
# same for credit

# Anti-exchange-diff optimization: if both ranges overlap, set
# partial_amount = min(remaining_debit_amount, -remaining_credit_amount)
# to avoid creating a spurious exchange-diff move.
if (debit_amount_in_credit_range and credit_amount_in_debit_range):
    partial_amount = min(remaining_debit_amount, -remaining_credit_amount)
    partial_debit_amount = partial_amount
    partial_credit_amount = partial_amount
```

Foreign-currency amount for each side:
```
if debit_currency == company_currency:
    partial_debit_amount_currency = partial_amount
else:
    partial_debit_amount_currency = min_recon_amount   # the foreign amount
# same for credit
```

#### Step 6 — Exchange Difference Computation (L2400–2467)

Triggered if NOT `no_exchange_difference` AND NOT `no_exchange_difference_no_recursive` context.

**`recon_currency == company_currency` sub-case:**
- If debit fully matched AND residual currency != 0 after partial: exchange_diff on `debit_exchange_amount = remaining_debit_amount_curr - partial_debit_amount_currency`.
- If credit fully matched: exchange_diff on `credit_exchange_amount = remaining_credit_amount_curr + partial_credit_amount_currency`.
- Exchange diff amounts stored as `{'amount_residual_currency': X}`.

**`recon_currency != company_currency` sub-case:**
- If debit fully matched: `debit_exchange_amount = remaining_debit_amount - partial_amount` → stored as `{'amount_residual': X}`.
- If debit NOT fully matched: `debit_exchange_amount = partial_debit_amount - partial_amount` → only if > 0.
- Symmetric for credit.
- Exchange diff amounts stored as `{'amount_residual': X}`.

Exchange diff values passed to `exchange_lines_to_fix._prepare_exchange_difference_move_vals(amounts_list, exchange_date=max(debit_date, credit_date))`.

#### Step 7 — Update Running Residuals (L2472–2499)

```
remaining_debit_amount -= partial_amount
remaining_credit_amount += partial_amount
remaining_debit_amount_curr -= partial_debit_amount_currency
remaining_credit_amount_curr += partial_credit_amount_currency

# Propagate back to debit_values/credit_values dicts
debit_values['amount_residual'] = remaining_debit_amount
debit_values['amount_residual_currency'] = remaining_debit_amount_curr
credit_values['amount_residual'] = remaining_credit_amount
credit_values['amount_residual_currency'] = remaining_credit_amount_curr

# Mark as None (fully consumed) if both residuals are zero
if debit_currency.is_zero(debit_values['amount_residual_currency'])
   and company_currency.is_zero(debit_values['amount_residual']):
    res['debit_values'] = None
# same for credit
```

#### Result dict structure:
```
{
    'debit_values': debit_values | None,
    'credit_values': credit_values | None,
    'partial_values': {
        'amount': partial_amount,           # always positive, company currency
        'debit_amount_currency': ...,       # always positive, debit's foreign currency
        'credit_amount_currency': ...,      # always positive, credit's foreign currency
        'debit_move_id': debit_aml.id,
        'credit_move_id': credit_aml.id,
    },
    'exchange_values': { ... } | absent,   # only if exchange diff needed
}
```

**AXIS CLASSIFICATION:** DETERMINISTIC (pure arithmetic with rounding; all cases are closed-form). The `get_amount_range_after_rate` range-overlap check is deterministic arithmetic, not a heuristic.

**Rounding protocol:** `currency.round(...)` throughout — uses `res.currency.decimal_places` stored field, NOT Python's `Decimal`. All intermediate amounts are plain Python `float`; rounding is applied at each persistence boundary.

**K-step:** K3
**woa-rs target:** `src/erp/reconcile.rs`

---

### Rule R-9: `AccountPartialReconcile` — Structure and Key Fields

**Odoo source:** `account_partial_reconcile.py:L9–67`

**Fields:**
| Field | Type | Semantics |
|---|---|---|
| `debit_move_id` | Many2one AML | The debit-side journal item (required, indexed) |
| `credit_move_id` | Many2one AML | The credit-side journal item (required, indexed) |
| `full_reconcile_id` | Many2one AccountFullReconcile | Set when reconciliation is complete; `btree_not_null` index |
| `exchange_move_id` | Many2one account.move | The exchange-difference move created for this partial |
| `amount` | Monetary (company_currency) | Always positive; the reconciled amount in company currency |
| `debit_amount_currency` | Monetary (debit_currency_id) | Always positive; reconciled amount in debit's foreign currency |
| `credit_amount_currency` | Monetary (credit_currency_id) | Always positive; reconciled amount in credit's foreign currency |
| `max_date` | Date | `max(debit.date, credit.date)` — used for aging reports |
| `company_id` | Many2one | Precomputed: invoice side if any, else credit side |
| `draft_caba_move_vals` | Json | Snapshot of CABA values at time of partial creation (for re-reconciliation after invoice posting) |
| `debit_currency_id` | Many2one, related, stored | = `debit_move_id.currency_id` |
| `credit_currency_id` | Many2one, related, stored | = `credit_move_id.currency_id` |

**`company_id` compute (L91–98):**
```python
if partial.debit_move_id.move_id.is_invoice(True):
    partial.company_id = partial.debit_move_id.company_id
else:
    partial.company_id = partial.credit_move_id.company_id
```
The invoice side wins for exchange-diff and CABA entry creation.

**Constraint (L73–77):** `_check_required_computed_currencies` — both `debit_currency_id` and `credit_currency_id` must be set; raises `ValidationError` otherwise.

**`create` hook (L149–153):**
```python
def create(self, vals_list):
    partials = super().create(vals_list)
    partials._get_to_update_payments(from_state='in_process').state = 'paid'
    self._update_matching_number(partials.debit_move_id + partials.credit_move_id)
    return partials
```
On creation: any fully-matched payments transition `in_process → paid`; matching numbers are updated.

**`unlink` cascade (L104–146):**
1. Collect payments to reset (`paid → in_process`).
2. Collect CABA moves to reverse (`account.move` where `tax_cash_basis_rec_id` in partial ids).
3. Collect exchange-diff moves to reverse (`self.exchange_move_id`).
4. Collect full reconcile to unlink (`self.full_reconcile_id`).
5. `super().unlink()` — delete the partials.
6. `full_to_unlink.unlink()` — cascade delete the full reconcile.
7. Reverse (or unlink if draft) CABA and exchange-diff moves: posted moves get `_reverse_moves`; draft moves get `unlink()`.
8. `_update_matching_number(all_reconciled)` — refresh matching numbers.
9. `to_update_payments.state = 'in_process'` — reset payment state.

**AXIS CLASSIFICATION:** DETERMINISTIC (field structure and cascade). Port directly.

**Ontology:**
`odoo:account.partial.reconcile` → **UNRESOLVED** in `odoo_alignment.rs` (not in `ODOO_ALIGNMENTS` slice). See §4 FLAG below.

**K-step:** K3
**woa-rs target:** `src/erp/reconcile.rs` (struct `PartialReconcile`)

---

### Rule R-10: `AccountFullReconcile` — Structure and Creation

**Odoo source:** `account_full_reconcile.py:L1–45`

**Fields:**
| Field | Type | Semantics |
|---|---|---|
| `partial_reconcile_ids` | One2many to AccountPartialReconcile | All partials that together achieve full reconciliation |
| `reconciled_line_ids` | One2many to AML | All AMLs participating in the full reconcile |

**`create` override (L13–45):**
Bypasses the ORM's M2M write (which would trigger unlink+flush) by using raw SQL `execute_values`:
```sql
UPDATE account_move_line line
   SET full_reconcile_id = source.full_id
  FROM (VALUES %s) AS source(full_id, line_ids)
 WHERE line.id = ANY(source.line_ids)
```
And similarly for `account_partial_reconcile`:
```sql
UPDATE account_partial_reconcile partial
   SET full_reconcile_id = source.full_id
  FROM (VALUES %s) AS source(full_id, partial_ids)
 WHERE partial.id = ANY(source.partial_ids)
```
Then invalidates recordset caches for the affected fields. Uses `tracking_disable=True` context to suppress chatter.

After creation: calls `self.env['account.partial.reconcile']._update_matching_number(fulls.reconciled_line_ids)` — converts matching numbers from `'P<partial_id>'` format to `'<full_reconcile_id>'` (string of the integer ID, without prefix).

**AXIS CLASSIFICATION:** DETERMINISTIC.

**Ontology:**
`odoo:account.full.reconcile` → **UNRESOLVED** in `odoo_alignment.rs`. See §4 FLAG below.

**K-step:** K3
**woa-rs target:** `src/erp/reconcile.rs` (struct `FullReconcile`)

---

### Rule R-11: Matching Number Protocol

**Odoo source:** `account_partial_reconcile.py:L171–215`

The `matching_number` field on AML (L284–290 in account_move_line.py) is a Char with btree index. Its value encoding:
- `None`/`False` — unreconciled
- `'I<anything>'` — temporary import marker (from `_reconcile_marked`)
- `'P<partial_id>'` — partially reconciled (partial_id = the smallest partial id in the graph)
- `'<full_reconcile_id>'` (decimal integer as string) — fully reconciled

**Graph-merge algorithm (`_update_matching_number`, L171–215):**
```
number2lines: Dict[partial_id, List[aml_id]]
line2number: Dict[aml_id, partial_id]

for partial in all_partials.sorted('id'):       # sorted ascending by partial id
    debit_min_id = line2number.get(partial.debit_move_id.id)
    credit_min_id = line2number.get(partial.credit_move_id.id)

    if both assigned:
        if they differ: merge into the smaller number
            for each line in number2lines[max]: reassign to min
    elif only debit assigned:
        add credit to debit's graph (number2lines[debit_min_id].append(credit.id))
    elif only credit assigned:
        add debit to credit's graph
    else:
        create new graph node: number2lines[partial.id] = [debit.id, credit.id]
```
Result: a union-find structure keyed by the smallest partial_id in each connected component.

**SQL update (L204–212):**
```sql
UPDATE account_move_line l
   SET matching_number = CASE
           WHEN l.full_reconcile_id IS NOT NULL THEN l.full_reconcile_id::text
           ELSE 'P' || source.number
       END
  FROM (VALUES %s) AS source(number, ids)
 WHERE l.id = ANY(source.ids)
```
Lines not in any graph component get `matching_number = False`.

**AXIS CLASSIFICATION:** DETERMINISTIC (union-find graph coloring). Port to Rust.

**K-step:** K3
**woa-rs target:** `src/erp/reconcile.rs` — `fn update_matching_numbers(partials: &[PartialReconcile])`

---

### Rule R-12: Exchange Difference Move Creation

**Odoo source:** `account_move_line.py:L2957–3098`

**`_prepare_exchange_difference_move_vals` (L2957–3043):**

When a line is fully matched in foreign currency but has a residual in company currency (due to exchange rate changes), an exchange-diff journal entry is created.

**Account selection (L2952–2955):**
```python
def _get_exchange_account(self, company, amount):
    if amount > 0.0:
        return company.expense_currency_exchange_account_id   # loss
    return company.income_currency_exchange_account_id         # gain
```

**Line vals for each amount dict:**
If `'amount_residual'` key:
```
amount_residual = amounts['amount_residual']
amount_residual_currency = 0.0 if line.currency_id != company_currency else amount_residual
```
If `'amount_residual_currency'` key:
```
amount_residual = 0.0
amount_residual_currency = amounts['amount_residual_currency']
```

Two lines per exchange diff:
1. Line on AML's original account (mirrors the residual, with `reconciled_lines_ids=[line]` linkage):
   - `debit = -amount_residual if amount_residual < 0 else 0`
   - `credit = amount_residual if amount_residual > 0 else 0`
   - `amount_currency = -amount_residual_currency`
2. Counterpart on exchange gain/loss account:
   - `debit = amount_residual if amount_residual > 0 else 0`
   - `credit = -amount_residual if amount_residual < 0 else 0`
   - `amount_currency = amount_residual_currency`

**Date:** `max(aml.date, journal.accounting_date)` where `accounting_date` is the journal's next valid accounting date given the exchange_date (respects lock dates).

**`_create_exchange_difference_moves` (L3046–3098):**
- Early return if empty list (prevents infinite recursion).
- Validates exchange journal + gain/loss accounts are configured.
- Creates moves with `no_exchange_difference=True` context (prevents recursion).
- Posts moves where both parent AMLs are in `posted` state.

**AXIS CLASSIFICATION:** DETERMINISTIC (accounting identity — exchange diff is the arithmetic residual that must balance the books).

**Config requirements (portal-level, enforced at runtime):**
- `company.currency_exchange_journal_id` must be set.
- `company.expense_currency_exchange_account_id` must be set.
- `company.income_currency_exchange_account_id` must be set.

**K-step:** K3
**woa-rs target:** `src/erp/reconcile.rs` — `fn create_exchange_difference_move(lines: &[ExchangeDiffSpec]) -> Move`

---

### Rule R-13: Tax Cash-Basis Collection and Move Creation

**Odoo source:** `account_partial_reconcile.py:L221–686`

#### `_collect_tax_cash_basis_values` (L221–334)

For each partial, for each of the two sides (debit/credit):
1. Call `move._collect_tax_cash_basis_values()` on the parent move (defined in account_move.py — not read in this lane).
2. Skip if no CABA values.
3. Check `company.tax_cash_basis_journal_id` is set (raises `UserError` if not).
4. Compute `partial_amount` and `partial_amount_currency` based on which side matches.
5. **Rate computation:**
   - If both sides are invoices: use source line's own rate (`rate_amount = source_line.balance`, `rate_amount_currency = source_line.amount_currency`, `payment_date = move.date`).
   - Otherwise: use counterpart's rate (`payment_date = counterpart_line.date`).
   - If source and counterpart have different foreign currencies: use `res.currency._get_conversion_rate(company_currency, source_currency, company, payment_date)`.
   - Else if `rate_amount` is non-zero: `payment_rate = rate_amount_currency / rate_amount`.
   - Else: `payment_rate = 0.0`.
6. Compute `percentage`:
   - If move's currency == company currency: `percentage = partial_amount / move_values['total_balance']`.
   - Else: `percentage = partial_amount_currency / move_values['total_amount_currency']`.
7. Append `{'partial', 'percentage', 'payment_rate', 'both_move_posted', 'counterpart_move'}` to `move_values['partials']`.

**CABA Move Creation (`_create_tax_cash_basis_moves`, L506–686):**

For each move in `tax_cash_basis_values_per_move.values()`:
- For each partial in `move_values['partials']`:
  - Determine move date: `max(partial.max_date, company_lock_date)` or `today` if past lock date.
  - For each `to_process_lines` item (type `'tax'` or `'base'`):
    - `amount_currency = line.currency_id.round(line.amount_currency * partial_values['percentage'])`
    - **Rounding fix on last partial (L553–565):** If `caba_treatment == 'tax'` AND (move is fully paid OR remaining residual < computed amount) AND this is the last partial → use `amount_residual_per_tax_line[line.id]` instead (ensures tax lines sum exactly to original).
    - `balance = amount_currency / payment_rate` (or 0 if rate is zero).
  - Group lines by `grouping_key` (currency, partner, account, tax_ids, repartition_line_id, analytic_distribution).
  - Create two lines per group: the CABA line + its counterpart.
  - If `both_move_posted`: add to `moves_to_create_and_post`, else `moves_to_create_in_draft`.

**Trigger condition (from R-6 Step 5):**
`is_cash_basis_needed` = `any(company.tax_exigibility)` AND `account_type in ('asset_receivable', 'liability_payable')`.

**AXIS CLASSIFICATION:** DETERMINISTIC (percentage-based proportional allocation is arithmetic). The `_collect_tax_cash_basis_values` aggregation is deterministic given the partial amounts. The `_create_tax_cash_basis_moves` is arithmetic. Port to Rust.

**K-step:** K3 + K7 (touches tax reporting)
**woa-rs target:** `src/erp/reconcile.rs` + `src/erp/tax_cash_basis.rs`

---

### Rule R-14: `_reconcile_pre_hook` / `_reconcile_post_hook`

**Odoo source:** `account_move_line.py:L2741–2752`

**Pre-hook:**
```python
def _reconcile_pre_hook(self):
    invoices = self.move_id.filtered(lambda move: move.is_invoice(include_receipts=True))
    return {
        'not_paid_invoices': invoices.filtered(lambda inv: inv.payment_state not in ('paid', 'in_payment')),
        'in_payment_invoices': invoices.filtered(lambda inv: inv.payment_state == 'in_payment'),
    }
```

**Post-hook:**
```python
def _reconcile_post_hook(self, data):
    (
        data['not_paid_invoices'].filtered(lambda inv: inv.payment_state in ('paid', 'in_payment'))
        + data['in_payment_invoices'].filtered(lambda inv: inv.payment_state == 'paid')
    )._invoice_paid_hook()
```
Calls `_invoice_paid_hook()` on invoices whose payment state changed. This typically sends confirmation emails, updates subscription states, etc.

**AXIS CLASSIFICATION:** DETERMINISTIC (state transition detection). The downstream `_invoice_paid_hook` behavior may contain business notifications (Axis-2 relevant for mail/email delegation to lance-graph via `MailIntent` reasoning kind) — but the hook trigger condition itself is deterministic.

**K-step:** K3
**woa-rs target:** `src/erp/reconcile.rs`

---

### Rule R-15 (Axis-2): Reconcile Candidate Matching / Suggestion

**Odoo reference:** `account.reconcile.model` (lane L5; NOT read in this lane per scope). Also: the UI "auto-reconcile" feature in the bank statement reconciliation wizard.

**Classification: HEURISTIC — Axis 2.**

The *selection* of which open items to propose as reconciliation candidates — given a bank transaction or incoming payment — is not computable from a closed-form rule. Odoo uses `account.reconcile.model` with configurable matching rules (line label regex, partner, amount window, date tolerance) and a scoring system. This is a multi-factor ranking/scoring task.

**NARS Delegation Contract Tuple:**
```
(
  ReasoningKind::Other("ReconcileMatch"),   // no exact variant; propose this name
  InferenceType::Induction,                 // "lines like X tend to match Y"
  SemiringChoice::NarsTruth,               // belief + confidence, not Boolean
  ThinkingStyle cluster: Analytical,        // inherited from SMBAccounting (0x62) family
)
```

**Inheritance chain:**
`odoo:account.reconcile.model` → `fibo:MatchingRule` (proposed pivot, not in ODOO_ALIGNMENTS) → OGIT family `SMBAccounting` (0x62, chart-of-accounts / ledger basin) → ThinkingStyle cluster: **Analytical** (Critical sub-cluster, inherited from BillingCore/SMBAccounting posting-logic).

**Contract call shape:**
```rust
reasoner.reason(ReasoningContext {
    namespace: "erp.k3.reconcile_match",
    kind: ReasoningKind::Other("ReconcileMatch"),
    evidence: vec![
        // open_item amounts, dates, partner, account
        // incoming payment amount, date, reference text
    ],
    budget: Budget::default(),
})
```

**What NOT to hand-code:** any scoring/ranking of open items by likelihood of matching, any fuzzy reference-number matching, any ML-based counterpart suggestion. These are Axis-2.

**What IS deterministic (Axis-1, still R-8):** once the human or heuristic has decided which AML pairs to reconcile, the arithmetic in R-8 is fully deterministic.

**K-step:** K3
**woa-rs target:** `src/contracts/reconcile_match.rs` (contract surface only; no brain logic in woa-rs)

---

## 3. woa-rs Calibration — Current Gap

Grep of `reconcile|offene.?post|residual|skonto|bezahlt` in `/home/user/woa-rs/src/` and `/home/user/woa-rs/crates/`:

**Findings:**
- `src/models/erp/k3_debitors.rs` — `ErpOpenItemAR` model exists with fields: `original_betrag`, `offen_betrag` (both `Decimal(15,2)`), `status` (`'offen'|'teilbezahlt'|'bezahlt'|'mahn1'|...`), `mahnstufe` (`i16`), `skonto_prozent` (`Decimal(5,2)`), `skonto_tage` (`i16`).
- `src/models/erp/k4_creditors.rs` — `ErpOpenItemAP` analogous for creditor side.
- `src/models/erp/k5_bank.rs` — `MatchedOpenItemArId`, `MatchedOpenItemApId` FK fields on bank transaction entity.
- `src/url.rs` — `/mahnwesen/{wid}/bezahlt` route exists.
- No `reconcile.rs`, no `PartialReconcile` struct, no `FullReconcile` struct, no `amount_residual` field anywhere.

**Gap assessment:** woa-rs has the open-item data model skeleton (`offen_betrag`, status machine) but has **NO reconciliation engine**. The following are entirely absent:
- Residual computation from partial records.
- Partial reconcile record creation.
- Full reconcile record + matching-number management.
- Exchange-difference move creation.
- Tax cash-basis move triggering.
- The `bezahlt` flag in `WorkOrder` is a simple boolean; there is no link to a formal reconciliation record.

**woa-rs target module:** `src/erp/reconcile.rs` (new file).

---

## 4. Enterprise/Unresolved Flags

### FLAG-1: `odoo:account.partial.reconcile` — UNRESOLVED in `odoo_alignment.rs`

`resolve_odoo("odoo:account.partial.reconcile")` → `None`.

**Proposed alignment:**
```
odoo:account.partial.reconcile
    → owl pivot: fibo:SettlementObligation (or fibo:Obligation)
    → OGIT family: SMBAccounting (0x62)
    → DOLCE: Perdurant (it is an event — the act of partial settlement)
```
**Justification:** A partial reconcile is a financial settlement event linking two AMLs. FIBO's `Obligation`/`Settlement` cluster is the nearest OWL class. It is NOT a new family — it inherits `SMBAccounting` (0x62) because it is a sub-event of the ledger posting process.

**Proposed `ODOO_ALIGNMENTS` row** (must be inserted in lexicographic order — before `account.account.skr03`):
```rust
OdooAlignment {
    odoo_class: "odoo:account.partial.reconcile",
    owl: OwlIdentity::new(FAM_SMB_ACCOUNTING, SLOT_PARTIAL_RECONCILE),  // new slot e.g. 0x07
    owl_pivot_label: "fibo:SettlementObligation",
    dolce: DolceMarker::Perdurant,
},
```

### FLAG-2: `odoo:account.full.reconcile` — UNRESOLVED in `odoo_alignment.rs`

`resolve_odoo("odoo:account.full.reconcile")` → `None`.

**Proposed alignment:**
```
odoo:account.full.reconcile
    → owl pivot: fibo:FullSettlement
    → OGIT family: SMBAccounting (0x62)
    → DOLCE: Perdurant (it is the completed settlement event)
```
**Proposed `ODOO_ALIGNMENTS` row:**
```rust
OdooAlignment {
    odoo_class: "odoo:account.full.reconcile",
    owl: OwlIdentity::new(FAM_SMB_ACCOUNTING, SLOT_FULL_RECONCILE),  // new slot e.g. 0x08
    owl_pivot_label: "fibo:FullSettlement",
    dolce: DolceMarker::Perdurant,
},
```

Both proposed alignments require adding new within-family slots to `SMBAccounting` (0x62). They do NOT require a new family (Option B compliant). The alignment rows must be added in lexicographic order in `ODOO_ALIGNMENTS`.

### FLAG-3: Enterprise boundary — NOT applicable for K3

Reconciliation in community odoo is fully implemented. No Enterprise gap for this lane.

### FLAG-4: `account.reconcile.model` — lane L5, not read in this lane

The candidate-matching model is Axis-2 and is intentionally deferred to lane L5. Its absence does not block the Axis-1 arithmetic port.

---

## 5. Complete Ontology Mapping Table

| Odoo class | OWL pivot | OGIT family | DOLCE | Status in odoo_alignment.rs |
|---|---|---|---|---|
| `odoo:account.move.line` | `fibo:JournalEntryLine` | SMBInvoice (0x81), slot 0x06 | Perdurant | **RESOLVED** (L139–143) |
| `odoo:account.move` | `fibo:Transaction` | SMBInvoice (0x81), slot 0x03 | Perdurant | **RESOLVED** (L133–137) |
| `odoo:account.account` | `fibo:Account` | SMBAccounting (0x62), slot 0x04 | Endurant | **RESOLVED** (L127–131) |
| `odoo:account.partial.reconcile` | `fibo:SettlementObligation` (proposed) | SMBAccounting (0x62), slot 0x07 (proposed) | Perdurant | **UNRESOLVED — FLAG-1** |
| `odoo:account.full.reconcile` | `fibo:FullSettlement` (proposed) | SMBAccounting (0x62), slot 0x08 (proposed) | Perdurant | **UNRESOLVED — FLAG-2** |

---

## 6. Porter's Checklist — Non-Obvious Gotchas

1. **`reconciled = True` requires BOTH residuals to be zero.** In multi-currency scenarios, a line with `amount_residual = 0` but `amount_residual_currency != 0` is NOT reconciled. This is the primary source of subtle bugs in simple ports.

2. **Residual formula direction.** `amount_residual = balance - debit_matched + credit_matched`. Credit partials ADD to the residual (because the line's balance is negative, so adding the positive credit-matched amount moves toward zero). Do not accidentally double-negate.

3. **The debit/credit pairing is ORDER-DEPENDENT.** The sort key is `(date_maturity or date, currency_id, amount_currency, balance)`. FIFO by maturity date. The Rust implementation MUST sort identically before the pairing loop.

4. **`min_recon_amount` is in `recon_currency`, not company currency.** When `recon_currency != company_currency`, the amounts from both sides are in foreign currency before taking the min.

5. **Anti-exchange-diff range overlap check (L2376–2384).** This is a rounding-tolerance optimization: if the two converted amounts fall within each other's rounding bands, treat them as equal and use `min(remaining_debit, -remaining_credit)` to avoid generating a spurious exchange-diff move. This is easy to miss and causes spurious exchange-diff entries if omitted.

6. **Exchange-diff context flags:** `no_exchange_difference` suppresses ALL exchange-diff creation. `no_exchange_difference_no_recursive` (used internally when creating CABA moves) prevents recursive exchange-diff creation from the CABA reconciliation. Both must be threaded through the Rust call chain.

7. **The `_update_matching_number` union-find iterates partials sorted by `id` ascending.** The smallest partial id in a connected component becomes the component's label. Any Rust port must replicate this stable sort to produce identical matching numbers.

8. **`AccountFullReconcile.create` uses raw SQL (`execute_values`).** The ORM `Command.set` on M2M would trigger an implicit unlink that forces a flush and breaks batch creation. In Rust/sea-orm, use a bulk `UPDATE` rather than setting the FK via the relation.

9. **CABA rounding correction on last partial (L553–565).** For the last partial of a fully-paid invoice, tax lines use the accumulated residual (`amount_residual_per_tax_line[line.id]`) rather than `percentage * amount_currency`. This ensures tax lines sum exactly to the original amounts and avoids penny-rounding discrepancies in the tax report.

10. **`draft_caba_move_vals` (Json field).** Set on the partial when `add_caba_vals` context is True (during CABA reconciliation). This JSON snapshot is used to detect whether re-reconciliation is needed when the invoice transitions from draft to posted. The porter must preserve this field and its write path.

11. **Payment state side-effect on partial create/unlink.** `_get_to_update_payments` checks if the matched payment is `in_process` (→ `paid` on create) or `paid` (→ `in_process` on unlink). This is a state machine on `account.payment` that must be co-ported.

12. **`exchange_line_mode`** (L2273–2279): when both sides share the same foreign currency but at least one has no amount in that currency, the mode suppresses rate computation for the opposite side. Without this, the exchange-diff lines themselves would generate more exchange-diffs recursively.

13. **`company_id` on partial prefers the invoice side** (L94–98). When creating exchange-diff and CABA entries, the company is taken from the invoice AML, not the payment AML. This matters for multi-company setups.

14. **`matching_number` format.** A fully-reconciled line has `matching_number = str(full_reconcile_id.id)` (pure integer string, no prefix). A partially reconciled line has `matching_number = 'P' + str(min_partial_id)`. An import-pending line has `matching_number = 'I<anything>'`. These are semantically distinct and must not be confused in queries.

---

## 7. Depth Proof

Read: `/home/user/odoo/addons/account/models/account_move_line.py` lines=3742 depth=full (reconciliation region: L240–295, L793–861, L2104–2948, L3100–3107, L3361–3391)
Read: `/home/user/odoo/addons/account/models/account_partial_reconcile.py` lines=706 depth=full
Read: `/home/user/odoo/addons/account/models/account_full_reconcile.py` lines=46 depth=full
Read: `/home/user/lance-graph/crates/lance-graph-callcenter/src/odoo_alignment.rs` lines=523 depth=full
Read: `/home/user/woa-rs/.claude/board/odoo-richness/BRIEFING.md` lines=124 depth=full
