RICHNESS-LANE-OK

# L1 — K3 Double-Entry POSTING (Odoo Richness Export)

**Lane:** L1-K3-POST  
**K-step:** K3 double-entry posting · K11 Festschreibung (hash chain)  
**woa-rs target module suggestion:** `src/erp/posting.rs` (orchestration facade);
hash chain logic can live in `src/erp/festschreibung.rs` (K11).

---

## 1. Scope and Files Read

| File | Lines | Depth |
|---|---|---|
| `/home/user/odoo/addons/account/models/account_move.py` | 7328 | full (multi-chunk, entire posting, sequence, hash, reversal, state-transition regions) |
| `/home/user/odoo/addons/account/models/account_move_line.py` | 3742 | full (balance/debit/credit computes, amount_currency, constraints, tax generation) |
| `/home/user/odoo/addons/account/models/sequence_mixin.py` | 512 | full |
| `/home/user/odoo/addons/l10n_de/models/account_move.py` | 19 | full |

---

## 2. Per-Rule Sections

---

### Rule K3-1 — `_post()` main posting flow

**Odoo file:Lrange:** `account_move.py:L5504-L5736`

#### Axis-1 Rich-AST Spec

**Signature:** `def _post(self, soft=True) -> RecordSet[account.move]`  
`self` is a recordset (batch) of moves to post.

**Step 0 — Access guard (L5518-L5519):**
```python
if not self.env.su and not self.env.user.has_group('account.group_account_invoice'):
    raise AccessError(...)
```
Early return if caller is not superuser and not in the invoice group. Rust: check user permissions before entering `post()`.

**Step 1 — Context patch (L5522):**
```python
self = self.with_context(skip_is_manually_modified=True)
```
Prevents downstream `is_manually_modified` recomputation from firing during post. Rust: thread a context flag through the service call, not the entity itself.

**Step 2 — Invoice-specific pre-validations (L5526-L5618), collected into `validation_msgs: set`:**

All validation messages are collected (not raised individually). A single `UserError` is raised at the end if the set is non-empty (L5616-L5618). Each check below is skipped for non-invoice (`entry`-type) moves unless noted.

| Check | Condition | Error | Notes |
|---|---|---|---|
| Quick-edit total mismatch | `quick_edit_mode` AND `quick_edit_total_amount` AND `currency.compare_amounts(total_amount, quick_edit_total_amount) != 0` | "current total / expected total" | Uses `currency_id.compare_amounts()` — integer compare after rounding to `decimal_places` |
| Archived bank account | `partner_bank_id` AND NOT `partner_bank_id.active` | "archived bank account" | |
| Untrusted inbound bank account | `partner_bank_id` AND `is_inbound()` AND NOT `allow_out_payment` | If superuser/public/portal: silently `partner_bank_id = False`; else if `_user_can_trust()`: `RedirectWarning`; else: `UserError` | Three-branch trust dispatch |
| Negative total | `float_compare(amount_total, 0.0, precision_rounding=currency_id.rounding) < 0` | "create a credit note instead" | Uses odoo `float_compare` with currency rounding precision |
| Missing partner (sale) | `not partner_id` AND `is_sale_document()` | "Customer field required" | |
| Missing partner (purchase) | `not partner_id` AND `is_purchase_document()` | "Vendor field required" | |
| Missing invoice_date (sale) | `not invoice_date` AND `is_sale_document(include_receipts=True)` | Auto-sets `invoice_date = today`; if currency rate is manual, protects it via `protecting()` context manager | Auto-fill pattern, not error |
| Missing invoice_date (purchase) | `not invoice_date` AND `is_purchase_document(include_receipts=True)` | "Bill/Refund date is required" | |

**Step 3 — Move-level validations (L5585-L5614), also collected into `validation_msgs`:**

| Check | Condition | Error |
|---|---|---|
| Line-level account/journal check | Per line (excluding sections/notes): calls `line_ids._check_constrains_account_id_journal_id()` | UserError from that method |
| State guard | `state in ['posted', 'cancel']` | "entry must be in draft" |
| Empty lines | No lines with display_type not in `('line_section', 'line_subsection', 'line_note')` | "Even magicians can't post nothing!" |
| Future-date auto-post config | `not soft AND move.auto_post != 'no' AND move.date > today` | "configured to be auto-posted on {date}" |
| Archived journal | NOT `journal_id.active` | "archived journal" |
| Inactive currency | `display_inactive_currency_warning` | "inactive currency" |
| Archived account | Any line's `account_id.active = False` (unless `skip_account_deprecation_check` in context) | "archived account" |
| Company mismatch | Any `account_id.company_ids` does not intersect `move.company_id.parent_ids` | "accounts from a different company" |

**Step 4 — Analytic account guard (L5620-L5624):**
```python
if inactive_analytic_ids := self.line_ids.sudo().with_context(active_test=False).distribution_analytic_account_ids.filtered(lambda a: not a.active):
    raise UserError(...)
```
Raises immediately (not accumulated into `validation_msgs`). Note: uses `sudo()` + `active_test=False` to see archived analytic accounts.

**Step 5 — Soft mode: future-date deferral (L5626-L5635):**
```python
if soft:
    future_moves = self.filtered(lambda move: move.date > today)
    for move in future_moves:
        if move.auto_post == 'no':
            move.auto_post = 'at_date'
        # post message log
    to_post = self - future_moves
else:
    to_post = self
```
`soft=True` is the default. Future-dated moves are NOT posted but their `auto_post` is flipped to `'at_date'` so a cron will post them later. Hard mode (`soft=False`) posts everything including future dates (used internally in `action_post()`).

**Step 6 — Lock-date date adjustment (L5637-L5641):**
```python
for move in to_post:
    affects_tax_report = move._affect_tax_report()
    lock_dates = move._get_violated_lock_dates(move.date, affects_tax_report)
    if lock_dates:
        move.date = move._get_accounting_date(...)
```
If the move's date falls in a locked period, the date is AUTOMATICALLY ADVANCED to the next open accounting date. This is NOT an error; it is a silent date mutation. Porter note: this is heuristic (see Axis-2 tagging below if you want to delegate it; but it can also be reproduced deterministically by ordering lock dates).

**Step 7 — Analytic lines batch creation (L5644):**
```python
to_post.line_ids._create_analytic_lines()
```
Batch for performance; cache invalidation is reduced.

**Step 8 — Recurring entries copy (L5647-L5648):**
```python
if not self.env.context.get('skip_recurring_copy'):
    to_post.filtered(lambda m: m.auto_post not in ('no', 'at_date'))._copy_recurring_entries()
```
After posting, if `auto_post` is `'monthly'`/`'quarterly'`/`'yearly'`, a copy is created for the next period with the date advanced by the appropriate delta (`relativedelta(months=1/3/12)`). The copy preserves `auto_post`, `auto_post_until`, `auto_post_origin_id`, `invoice_user_id`. Due date is adjusted if `invoice_payment_term_id` is unset.

**Step 9 — Partner sync (L5650-L5658):**
Forces all `partner_id` fields on move lines to match the move's `commercial_partner_id` (guard against OCR races).

**Step 10 — Cash-basis reconciliation setup (L5661-L5690):**
Identifies draft reversal moves that need to be posted alongside their counterpart. Collects exchange-difference moves. Handles CABA (cash-basis accounting) partial reconcile invalidation when the draft move changed since it was last reconciled.

**Step 11 — State transition (L5692-L5695):**
```python
to_post.write({'state': 'posted', 'posted_before': True})
```
`posted_before` is set to `True` permanently and never reset. This is used for sequence-name enforcement: a move that was once posted and has a name cannot have its name reset even if it returns to draft.

**Step 12 — Non-deductible lines naming (L5701-L5708):**
Appends `"<move_number> - private part"` to non-deductible lines' names after the sequence number is known.

**Step 13 — Reversal reconciliation (L5710-L5711):**
```python
draft_reverse_moves.reversed_entry_id._reconcile_reversed_moves(draft_reverse_moves, ...)
to_post.line_ids._reconcile_marked()
```

**Step 14 — Partner rank update (L5713-L5729):**
Increments `customer_rank`/`supplier_rank` counters on partners.

**Step 15 — Zero-amount invoice hook (L5731-L5734):**
If total is zero for an invoice, fires `_invoice_paid_hook()`.

**Returns:** `to_post` — the recordset of actually-posted moves.

**Axis classification:** DETERMINISTIC.  
K-step: K3.  
woa-rs target: `src/erp/posting.rs::post_moves()`.

**Ontology mapping:**  
`odoo:account.move` → `fibo:Transaction` → OGIT family `SmbFoundryInvoice` (0x81) → DOLCE Perdurant.

---

### Rule K3-2 — `action_post()` public entrypoint

**Odoo file:Lrange:** `account_move.py:L6073-L6094`

#### Axis-1 Rich-AST Spec

```python
def action_post(self):
    if (
        not self.env.context.get('disable_abnormal_invoice_detection', True)
        and self.filtered(lambda m: m.abnormal_amount_warning or m.abnormal_date_warning)
    ):
        # open validate.account.move wizard
        return {...}
    if self:
        self._post(soft=False)
    if autopost_bills_wizard := self._show_autopost_bills_wizard():
        return autopost_bills_wizard
    return False
```

Key: `action_post` calls `_post(soft=False)`. The `disable_abnormal_invoice_detection` context key defaults to `True` in this codebase (the comment says "Disabled by default to avoid breaking automated action flow"), meaning the wizard is **not** shown by default. Only when an external caller sets it to `False` does it appear.

`_show_autopost_bills_wizard()` (L5838-L5875): shows a wizard to enable automatic posting for a vendor if 3 or more consecutive bills from the same partner were not manually modified. This is:
- **Axis-2 HEURISTIC**: counting "unmodified consecutive bills" is a heuristic recommendation.
- Contract tuple: `(PostingAnomaly, Induction, NarsTruth, Analytical)`.
- Inherited ThinkingStyle cluster: **Analytical** (OGIT `SmbFoundryInvoice` → posting-anomaly check).

**Axis classification:** Mostly DETERMINISTIC; `_show_autopost_bills_wizard` sub-call is HEURISTIC (Axis-2).  
K-step: K3.

---

### Rule K3-3 — `_check_balanced()` and `_get_unbalanced_moves()` — balance invariant

**Odoo file:Lrange:** `account_move.py:L2755-L2794`

#### Axis-1 Rich-AST Spec

`_check_balanced` is a `@contextmanager`; it wraps create/write operations. It calls `_get_unbalanced_moves()` which executes a raw SQL query:

```sql
SELECT line.move_id,
       ROUND(SUM(line.debit), currency.decimal_places) debit,
       ROUND(SUM(line.credit), currency.decimal_places) credit
  FROM account_move_line line
  JOIN account_move move ON move.id = line.move_id
  JOIN res_company company ON company.id = move.company_id
  JOIN res_currency currency ON currency.id = company.currency_id
 WHERE line.move_id IN %s
GROUP BY line.move_id, currency.decimal_places
HAVING ROUND(SUM(line.balance), currency.decimal_places) != 0
```

**Critical semantics:**
- Rounding is applied **per company currency's `decimal_places`** (not per move's foreign currency).
- The rounding is `ROUND(SUM(...), decimal_places)` — SQL-level rounding on the aggregated sum, not per-line rounding.
- `balance = debit - credit` for normal lines; `balance = credit - debit` for storno lines (see K3-6 below).
- The query runs on committed DB state (flush is called first: L2782).
- Guard: `with self._disable_recursion(container, 'check_move_validity', default=True, target=False)` prevents recursive invocation.

**Error handling:** If exactly 1 unbalanced move: generic "The entry is not balanced." If multiple: lists each move name.

**What "balanced" means in odoo:**
`SUM(balance) == 0` after rounding to company currency decimal places. This is the double-entry invariant: sum of debits == sum of credits. Debit = `balance > 0`, Credit = `balance < 0` (for non-storno lines).

**Axis classification:** DETERMINISTIC.  
K-step: K3.  
woa-rs target: `src/erp/posting.rs::check_balanced()`.

**Ontology mapping:**  
`odoo:account.move` → `fibo:Transaction` → OGIT `SmbFoundryInvoice` (0x81) → DOLCE Perdurant.

---

### Rule K3-4 — `_compute_balance`, `_compute_debit_credit`, `_compute_amount_currency` — line-level amounts

**Odoo file:Lrange:** `account_move_line.py:L708-L762`

#### Axis-1 Rich-AST Spec

**`_compute_balance` (L708-L724):**
```python
def _compute_balance(self):
    for line in self:
        if line.display_type in ('line_section', 'line_subsection', 'line_note'):
            line.balance = False  # structural lines have no balance
        elif not line.move_id.is_invoice(include_receipts=True):
            # journal entry (not invoice): auto-balance to zero by computing
            # the negative sum of all other lines
            active_line_ids = [lid for lid in self.env.context.get('line_ids', []) if isinstance(lid, int)]
            existing_lines = self.env['account.move.line'].browse(active_line_ids)
            outdated_lines = line.move_id.line_ids._origin
            new_lines = line.move_id.line_ids - line
            line.balance = -sum((existing_lines - outdated_lines + new_lines).mapped('balance'))
        else:
            line.balance = 0  # for invoices, balance is computed elsewhere (from price_unit etc.)
```

Key: on journal entries (`entry` type), the LAST line to be assigned has its balance auto-computed as the negative of all others. This is the "balancing line" UX convenience. For invoices, `balance` is driven by `price_unit`/`quantity`/`discount` via the invoice compute chain.

**`_compute_debit_credit` (L727-L734):**
```python
@api.depends('balance')
def _compute_debit_credit(self):
    for line in self:
        if not line.is_storno:
            line.debit = line.balance if line.balance > 0.0 else 0.0
            line.credit = -line.balance if line.balance < 0.0 else 0.0
        else:
            # STORNO: flip debit/credit sign presentation
            line.debit = line.balance if line.balance < 0.0 else 0.0
            line.credit = -line.balance if line.balance > 0.0 else 0.0
```

This is a pure derived field: `debit` and `credit` are always non-negative; their signs are determined by whether `balance` is positive or negative. The storno flag flips the visual presentation (negative debit / positive credit for a storno cancellation entry).

**`_compute_amount_currency` (L757-L762):**
```python
@api.depends('currency_rate', 'balance')
def _compute_amount_currency(self):
    for line in self:
        if line.amount_currency is False:
            line.amount_currency = line.currency_id.round(line.balance * line.currency_rate)
        if line.currency_id == line.company_id.currency_id and not line.move_id.is_invoice(True):
            line.amount_currency = line.balance
```

`amount_currency` is the balance expressed in the line's foreign currency. When the line currency matches the company currency (no FX), `amount_currency == balance`. Otherwise: `currency_id.round(balance * currency_rate)`. The `currency_id.round()` method uses the currency's `decimal_places` (e.g. 2 for EUR).

**`_compute_currency_rate` (L736-L749):**
```python
@api.depends('currency_id', 'company_id', 'move_id.invoice_currency_rate', 'move_id.date')
def _compute_currency_rate(self):
    for line in self:
        if line.move_id.is_invoice(include_receipts=True):
            line.currency_rate = line.move_id.invoice_currency_rate or 1.0
        elif line.currency_id:
            line.currency_rate = self.env['res.currency']._get_conversion_rate(
                from_currency=line.company_currency_id,
                to_currency=line.currency_id,
                company=line.company_id,
                date=line.move_id.invoice_date or line.move_id.date or fields.Date.context_today(line),
            )
        else:
            line.currency_rate = 1
```

For invoices: uses the header-level `invoice_currency_rate` (locked at time of invoice creation). For journal entries: looks up the live exchange rate at the move date.

**`depends` chains (summarised):**
- `balance` ← computed from `price_unit`, `quantity`, `discount`, `tax_ids`, `currency_id` (invoice lines) or auto-balance (journal entry last line)
- `debit`, `credit` ← `balance`, `is_storno`
- `amount_currency` ← `currency_rate`, `balance`
- `currency_rate` ← `currency_id`, `company_id`, `move_id.invoice_currency_rate`, `move_id.date`

**Axis classification:** DETERMINISTIC.  
K-step: K3.  
woa-rs target: `src/erp/posting.rs` (line amount computation helpers).

**Ontology mapping:**  
`odoo:account.move.line` → `fibo:JournalEntryLine` → OGIT family `SmbFoundryInvoice` (0x81), slot `SLOT_JOURNAL_LINE` (0x06) → DOLCE Perdurant (the alignment row explicitly sets Perdurant; note: the test suite documents a known divergence from suffix heuristic which would give Endurant — curated row wins).

---

### Rule K3-5 — Line-level constraints (`_sql_constraints` and `@api.constrains`)

**Odoo file:Lrange:** `account_move_line.py:L1438-L1578`

#### Axis-1 Rich-AST Spec

**`_check_constrains_account_id_journal_id` (L1438-L1454):**  
Not a decorator — called explicitly from `_post()` (step 2 above) rather than on every write:
- Skip lines with `display_type in ('line_section', 'line_subsection', 'line_note')`.
- If account is archived (not active) AND NOT `is_imported` AND NOT `skip_account_deprecation_check`: raise.
- If `account.currency_id` is set AND it is neither the company currency nor the line's own `currency_id`: raise (forces secondary currency consistency).
- If account is the journal's `default_account_id` or `suspense_account_id`: skip remaining checks (these are always valid).

**`_check_off_balance` (L1456-L1465, `@api.constrains`):**
- If any line's account type is `'off_balance'`, ALL lines in the same move MUST have `account_type == 'off_balance'` — otherwise raise.
- Off-balance lines CANNOT have `tax_ids` or `tax_line_id`.
- Off-balance lines CANNOT be reconciled.

**`_check_payable_receivable` (L1467-L1480, `@api.constrains`):**
- On sale documents: `liability_payable` account is forbidden; `payment_term` display_type XOR `asset_receivable` account type (if one is present, both must be).
- On purchase documents: `asset_receivable` account is forbidden; `payment_term` display_type XOR `liability_payable` account type.

**`_check_caba_non_caba_shared_tags` (L1512-L1551, `@api.constrains`):**
Prevents cash-basis (on_payment) taxes and non-cash-basis taxes from sharing repartition tags on the same line. Raises `ValidationError` if `caba_base_tags & non_caba_base_tags` is non-empty (or for tax-affects-tax scenarios).

**`_constrains_matching_number` (L1553-L1570, `@api.constrains`):**
Matching number format: `^((P?\d+)|(I.+))$`. Invariants:
- `'I'` prefix → temporary (import) number; cannot be in `matched_debit_ids` or `matched_credit_ids`.
- `'P'` prefix → partial reconciliation; MUST have partials; must NOT have `full_reconcile_id`.
- Numeric-only → full reconciliation; MUST have `full_reconcile_id`; number MUST equal `str(full_reconcile_id.id)`.

**`_constrains_deductible_amount` (L1572-L1578, `@api.constrains`):**
Non-purchase documents must have `deductible_amount == 100`. Value must be in `[0, 100]`.

**Axis classification:** DETERMINISTIC.  
K-step: K3.  
woa-rs target: `src/erp/posting.rs` (validation functions called from `post_moves()`).

---

### Rule K3-6 — Storno (Gegenbuchung) and `_reverse_moves()`

**Odoo file:Lrange:** `account_move.py:L5430-L5474`

#### Axis-1 Rich-AST Spec

```python
TYPE_REVERSE_MAP = {
    'entry': 'entry',
    'out_invoice': 'out_refund',
    'out_refund': 'out_invoice',
    'in_invoice': 'in_refund',
    'in_refund': 'in_invoice',
    'out_receipt': 'out_refund',
    'in_receipt': 'in_refund',
}
```

**`_reverse_moves(default_values_list=None, cancel=False)` (L5430-L5474):**

```
for each (move, default_values):
    default_values['move_type'] = TYPE_REVERSE_MAP[move.move_type]
    default_values['reversed_entry_id'] = move.id
    default_values['partner_id'] = move.partner_id.id
    reverse_move = move.copy(default_values)   # full copy with context flags

# After creating all reverse copies, flip balance/amount_currency on journal-entry lines:
reverse_moves.write({
    'line_ids': [
        Command.update(line.id, {
            'balance': -line.balance,
            'amount_currency': -line.amount_currency,
            # if company.account_storno: also flip is_storno flag
            **({'is_storno': not line.is_storno} if line.company_id.account_storno else {})
        })
        for line in reverse_moves.line_ids
        if line.move_id.move_type == 'entry' or line.display_type == 'cogs'
    ]
})

if cancel:
    reverse_moves._post(soft=False)   # immediately post the reversal
```

**Key semantics:**
1. A reverse move is a **copy** of the original, not a mutation. This is the GoBD Storno pattern: two audit rows exist in the ledger — the original and the reversal.
2. Balance and amount_currency are **negated** on `entry`-type move lines and `cogs`-type lines. Invoice-type lines are NOT negated here because their amounts come from the copied `price_unit` etc. which are themselves negated by the invoice copy machinery.
3. `is_storno` flag: if `company.account_storno` is enabled, the flag is toggled (for German/Austrian storno-debit presentation — a storno entry shows the credit as a negative debit rather than a positive credit, keeping the visual ledger cleaner). See `_compute_debit_credit` (K3-4): storno lines flip debit/credit sign.
4. `reversed_entry_id`: FK on the reverse move pointing back to the original. The original move gets `reversal_move_id` pointing forward.
5. `cancel=True`: the reversal is immediately posted (creates the two-row audit trail in one operation). Used by `button_cancel` → `_unlink_or_reverse()` (L5486-L5502).

**GoBD mapping:** Storno = zwei Buchungszeilen (Original + Stornobuchung). Never deletes the original row. `button_cancel` path for posted+hashed moves: `_unlink_or_reverse()` picks `_reverse_moves(cancel=True)` when `_can_be_unlinked()` returns False (i.e., when the move has an `inalterable_hash`).

**`_can_be_unlinked()` (L5476-L5481):**
```python
def _can_be_unlinked(self):
    lock_date = self.company_id._get_user_fiscal_lock_date(self.journal_id)
    posted_caba_entry = ...
    posted_exchange_diff_entry = ...
    return not self.inalterable_hash and self.date > lock_date and not posted_caba_entry and not posted_exchange_diff_entry
```
A move with `inalterable_hash` can NEVER be unlinked — it MUST be reversed (GoBD Festschreibung). This is the K11 boundary.

**Axis classification:** DETERMINISTIC.  
K-step: K3 (reversal creation) + K11 (hash guard on unlinking).  
woa-rs target: `src/erp/posting.rs::reverse_moves()`.

---

### Rule K3-7 — `button_draft()` / `button_cancel()` — state transitions

**Odoo file:Lrange:** `account_move.py:L6162-L6253`

#### Axis-1 Rich-AST Spec

**`button_draft()` (L6162-L6174):**
```
Guard: state must be 'cancel' or 'posted'
Guard: not need_cancel_request (e-invoice government lock)
_check_draftable():
    - not an exchange difference move (cannot draft)
    - not a CABA (cash-basis) entry (cannot draft)
    - not inalterable_hash (CANNOT reset to draft a locked entry)
Actions on success:
    - unlink all analytic_line_ids (with skip_analytic_sync=True)
    - state = 'draft'
    - sending_data = False
    - _detach_attachments() (detaches invoice PDF with timestamp)
```

**Critical:** `inalterable_hash` check in `_check_draftable()` (L6229-L6230):
```python
if move.inalterable_hash:
    raise UserError(_('You cannot reset to draft a locked journal entry.'))
```
A hashed/festgeschrieben entry is permanently immutable. This is the K11 boundary again.

**`button_cancel()` (L6241-L6253):**
```
1. Filter moves in state 'posted' → call button_draft() on them first
2. Guard: state must be 'draft' after step 1
3. line_ids.remove_move_reconcile()   # unreconcile all
4. payment_ids.state = "canceled"
5. write({'auto_post': 'no', 'state': 'cancel'})
```

Note: `button_cancel` does NOT call `_reverse_moves`. It directly sets state to `cancel`. This is different from the GoBD storno pattern — `button_cancel` is only safe on un-hashed moves. Hashed moves go through `_unlink_or_reverse()` → `_reverse_moves(cancel=True)`.

**State machine (complete):**
```
draft ──[_post()]──► posted ──[button_draft()]──► draft
                  │
                  └──[button_cancel()]──► (draft) ──► cancel
                  │
                  └──[_reverse_moves(cancel=True)]──► posted (reversal created AND posted)
```

**Axis classification:** DETERMINISTIC.  
K-step: K3 (state machine) + K11 (hash guard).  
woa-rs target: `src/erp/posting.rs`.

---

### Rule K3-8 — Sequence / Belegnummer assignment

**Odoo file:Lrange:**  
- `account_move.py:L938-L964` (`_compute_name`)  
- `account_move.py:L4157-L4267` (`_get_last_sequence_domain`, `_get_starting_sequence`)  
- `sequence_mixin.py:L269-L473` (full mixin: `_get_last_sequence`, `_set_next_sequence`, `_locked_increment`, `_deduce_sequence_number_reset`)

#### Axis-1 Rich-AST Spec

**Sequence format families (sequence_mixin.py:L41-L45):**
| Family | Regex pattern | Reset |
|---|---|---|
| `year_range_month` | `^<prefix><year><sep><year_end><sep2><month><sep3><seq><suffix>$` | per fiscal-year-month |
| `monthly` | `^<prefix><year><sep><month><sep2><seq><suffix>$` | per calendar month |
| `year_range` | `^<prefix><year><sep><year_end><sep><seq><suffix>$` | per fiscal year range |
| `yearly` | `^<prefix><year><sep><seq><suffix>$` | per calendar year |
| `fixed` | `^<prefix><seq><suffix>$` | never |

Priority order for `_deduce_sequence_number_reset`: year_range_month → monthly → year_range → yearly → fixed (first match wins).

**Starting sequence format (account_move.py:L4229-L4267):**
- Sales/bank/cash/credit journals: `"<CODE>/<year_part>/<00000>"` (annual, 5-digit seq)
- Other (purchase/general) journals: `"<CODE>/<year_part>/<MM>/<0000>"` (monthly, 4-digit seq)
- Staggered fiscal year (not ending Dec 31): year_part is `"<YY>-<YY>"` (e.g. `"23-24"`), seq length 4.
- Refund sequence prefix: `"R"` prepended to starting sequence.
- Payment sequence prefix: `"P"` prepended to starting sequence.
- Self-billing: `"<CODE><partner_id_padded>/<year_part>/<MM>/<0000>"`.

**`_get_last_sequence_domain` override (account_move.py:L4157-L4227):**
Filters to the same journal. If NOT relaxed:
1. Finds reference move (most recent in same period + journal + type-family).
2. Deduces reset periodicity from reference name via `_deduce_sequence_number_reset`.
3. Computes `date_start`/`date_end` for the period via `_get_sequence_date_range`.
4. Applies `anti_regex` to exclude format-crossing contamination (e.g. monthly regex matching yearly names).
5. Filters by refund/payment/self-billing discriminators.

**`_get_last_sequence` (sequence_mixin.py:L269-L310):** Runs:
```sql
SELECT name FROM account_move
{where_string}
AND sequence_prefix = (SELECT sequence_prefix FROM account_move {where_string} ORDER BY id DESC LIMIT 1)
ORDER BY sequence_number DESC LIMIT 1
```
Returns the highest sequence number in the current prefix+period window.

**`_locked_increment` (sequence_mixin.py:L355-L423) — gap-prevention core:**
```
1. Check transaction-local cache: if cache[cache_key] exists, increment in memory.
2. Otherwise: open a SAVEPOINT.
3. Loop: seq += 1; attempt UPDATE account_move SET name = <next_seq> WHERE id = <self.id>
4. If UniqueViolation or ExclusionViolation: rollback to SAVEPOINT, retry.
5. On success: store cache[cache_key] = seq; return sequence string.
```
The UNIQUE constraint on `(journal_id, sequence_prefix, sequence_number)` is the gap-prevention mechanism. No two moves can have the same sequence in the same journal+prefix. The lock is implicit in the B-tree index entry (PostgreSQL `_bt_doinsert` exclusive lock).

**`_compute_name` (account_move.py:L938-L954):**
```python
self = self.sorted(lambda m: (m.date, m.ref or '', m._origin.id))
for move in self:
    if move.state == 'cancel': continue
    move_has_name = move.name and move.name != '/'
    if not move.posted_before and not move._sequence_matches_date():
        move.name = False; continue   # reset if date-sequence mismatch, first time only
    if move.date and not move_has_name and move.state != 'draft':
        move._set_next_sequence()
self._inverse_name()
```
Key: `posted_before=True` freezes the name. If a move was posted, its name is never reset by `_compute_name` even if the date changes.

**Axis classification:** DETERMINISTIC (the sequence assignment algorithm itself).  
One Axis-2 sub-case: if `relaxed=True` is used (fallback to a different period's format), the format detection is heuristic — but the fallback is deterministic once the format is found.  
K-step: K3 (Belegnummer), also affects K11 (sequence gaps detected by `_get_chains_to_hash`).  
woa-rs target: `src/erp/posting.rs` (sequence assignment), or a dedicated `src/erp/sequence.rs`.

---

### Rule K3-9 — `_get_computed_taxes()` and `_sync_tax_lines()` — tax line generation

**Odoo file:Lrange:**  
- `account_move_line.py:L944-L973` (`_get_computed_taxes`)  
- `account_move.py:L3258-L3464` (`_sync_tax_lines`)

#### Axis-1 Rich-AST Spec

**`_get_computed_taxes()` (aml:L944-L973):**  
Priority chain for tax resolution on a line:
1. Sale document: `product.taxes_id` filtered to company → fallback to `account_id.tax_ids` (type='sale').
2. Purchase document: `product.supplier_taxes_id` filtered to company → fallback to `account_id.tax_ids` (type='purchase').
3. Other (journal entry with `account_default_taxes` context): all `account_id.tax_ids`.
4. Otherwise: `account_id.tax_ids` unless `skip_computed_taxes` context or `is_entry()`.
5. Always filter by company: `_filter_taxes_by_company(company_id)`.
6. If `fiscal_position_id`: map taxes via `fiscal_position.map_tax(tax_ids)`.

**`_sync_tax_lines()` (am:L3258-L3464):**  
Contextmanager called from `create`/`write` hooks. Tracks changes to base lines and tax lines. Decision logic:

```
If currency or move_type changed on an invoice: round_from_tax_lines = False
Elif a base line with tax_ids was removed: round_from_tax_lines = any_field_has_changed(tax_lines)
Elif a base line was modified:
    round_from_tax_lines = (
        all changed lines have no tax_ids  # no impact
        OR (tax line list changed OR any tax line field is manually protected)
    )
    If round_from_tax_lines and any balance/amount_currency provided: skip (manual override)
Elif currency_rate changed: round_from_tax_lines = 'reapply_currency_rate'
Else: skip (no change)
```

Then:
```python
base_lines_values, tax_lines_values = move._get_rounded_base_and_tax_lines(round_from_tax_lines=...)
AccountTax._add_accounting_data_in_base_lines_tax_details(base_lines_values, company, ...)
tax_results = AccountTax._prepare_tax_lines(base_lines_values, company, tax_lines=tax_lines_values)
```
Results: `tax_lines_to_add`, `tax_lines_to_delete`, `tax_lines_to_update`, `base_lines_to_update`.

Tax line `display_type` is set to `'tax'` on create. Non-deductible tax lines (`display_type='non_deductible_tax'`) are computed separately from partial-deductibility amount.

**Axis classification:** DETERMINISTIC for core tax arithmetic. The choice of which tax to apply (fiscal position mapping, product tax priority) has a mild heuristic component — but the odoo code resolves it deterministically via priority chain above.  
K-step: K7 (tax) — but K3 because tax lines are part of the balanced double-entry posting.  
woa-rs target: `src/erp/posting.rs` + `src/erp/tax.rs` (K7 lane).

---

### Rule K3-10 / K11 — Inalterability hash chain (`_get_integrity_hash_fields`, `_calculate_hashes`, `_hash_moves`, `_get_chains_to_hash`)

**Odoo file:Lrange:**  
- `account_move.py:L4548-L4558` (`_get_integrity_hash_fields`)  
- `account_move.py:L4727-L4759` (`_calculate_hashes`)  
- `account_move.py:L4581-L4593` (`_hash_moves`)  
- `account_move.py:L4683-L4725` (`_get_chains_to_hash`)  
- `account_move_line.py:L3352-L3359` (`line._get_integrity_hash_fields`)  
- `MAX_HASH_VERSION = 4` (am:L46)

#### Axis-1 Rich-AST Spec

**Hash versions:**
| Version | Move fields | Line fields added |
|---|---|---|
| 1 | `date, journal_id, company_id` | `debit, credit, account_id, partner_id` |
| 2, 3, 4 | `name, date, journal_id, company_id` | `name, debit, credit, account_id, partner_id` |

Current production version: `MAX_HASH_VERSION = 4`.

**`_calculate_hashes(previous_hash)` (L4727-L4759):**
```python
for move in self:  # self must be sorted by sequence_number (caller's responsibility)
    # Strip version prefix from previous_hash if present
    if previous_hash and previous_hash.startswith("$"):
        previous_hash = previous_hash.split("$")[2]
    
    values = {}
    for fname in move._get_integrity_hash_fields():
        values[fname] = _getattrstring(move, fname)
    
    for line in move.line_ids:
        for fname in line._get_integrity_hash_fields():
            k = 'line_%d_%s' % (line.id, fname)
            values[k] = _getattrstring(line, fname)
    
    current_record = dumps(values, sort_keys=True, ensure_ascii=True, indent=None, separators=(',', ':'))
    hash_string = sha256((previous_hash + current_record).encode('utf-8')).hexdigest()
    move2hash[move] = f"${hash_version}${hash_string}" if hash_version >= 4 else hash_string
    previous_hash = move2hash[move]  # chain: each move's hash feeds the next
```

**`_getattrstring` serialization:**
- `many2one` field → `field_value.id` (integer ID as string)
- `monetary` field (hash_version >= 3) → `float_repr(value, currency.decimal_places)` (e.g. `"1234.56"`)
- Other fields → `str(field_value)`
- JSON: `dumps(values, sort_keys=True, ensure_ascii=True, indent=None, separators=(',', ':'))`
  → compact, keys alphabetically sorted, ASCII-only, no whitespace.

**Chain structure:**  
Moves are chained per `(journal_id, sequence_prefix)`. Each chain starts from `previous_hash = ''` (or the last hashed move's `inalterable_hash` stripped of version prefix). The chain is linear — move N's hash includes move N-1's hash as input.

**`_get_chains_to_hash` (L4683-L4725):**  
Groups moves by `journal_id` → then by `sequence_prefix`. For each chain:
1. Finds the last hashed move (`inalterable_hash IS NOT NULL`) in the sequence.
2. Searches for unhashed moves with `sequence_number > last_hashed.sequence_number`.
3. Raises `UserError` if: gap detected in sequence numbers; all entries already hashed; any bank statement line is unreconciled.

**`_hash_moves` (L4581-L4593):** Calls `_get_chains_to_hash`, then for each chain calls `_calculate_hashes(previous_hash)`, writes `inalterable_hash` on each move, posts a chatter message.

**`button_hash` (L6232-L6233):** User-visible "lock" button: `self._hash_moves(force_hash=True)`.

**`_can_be_unlinked` gate (L5476-L5481):**
```python
return not self.inalterable_hash and ...
```
`inalterable_hash` set → permanently immutable → must be reversed, not deleted.

**GoBD mapping:**
- K11 Festschreibung = `inalterable_hash` set on a move.
- woa-rs already has `after_hash` column on `ErpJournal` (K2) and `erp_gobd_festschreibung` on tenants. The odoo hash chain logic is richer: it chains across the entire sequence prefix, includes line-level fields, and uses a version prefix `$4$<sha256>`.
- woa-rs `_shared::chain_hash` + `_shared::serialize_for_hash` are the existing analogues. They need to be extended to: (a) include line-level fields in the hash input, (b) chain across the sequence prefix (not just per-journal), (c) support version prefix.

**Axis classification:** DETERMINISTIC (K11). The chain topology detection (which moves to include) is deterministic SQL. The hash itself is SHA-256 (deterministic).  
K-step: **K11 Festschreibung**.  
woa-rs target: `src/erp/festschreibung.rs`.

---

### Rule K3-11 — German override (`l10n_de`) — `_post` extension

**Odoo file:Lrange:** `l10n_de/models/account_move.py:L1-L19`

#### Axis-1 Rich-AST Spec

```python
def _compute_show_delivery_date(self):
    super()._compute_show_delivery_date()
    for move in self:
        if move.country_code == 'DE':
            move.show_delivery_date = move.is_sale_document()

def _post(self, soft=True):
    for move in self:
        if move.country_code == 'DE' and move.is_sale_document() and not move.delivery_date:
            move.delivery_date = move.invoice_date or fields.Date.context_today(self)
    return super()._post(soft)
```

The German override:
1. Shows the delivery date field on sale documents (for GoBD §14 UStG Lieferdatum).
2. Auto-fills `delivery_date` to `invoice_date` (or today) before posting if it is missing on German sale documents.

**Axis classification:** DETERMINISTIC (date auto-fill rule).  
K-step: K3 (pre-post date logic), also K7 (§14 UStG Lieferdatum is relevant for VAT reporting).  
woa-rs target: German locale hook in `src/erp/posting.rs` (or a `src/erp/l10n_de.rs`).

---

### Rule K3-12 — `_compute_name` and `posted_before` — name freeze

**Odoo file:Lrange:** `account_move.py:L938-L964`

#### Axis-1 Rich-AST Spec (supplementary to K3-8)

Critical invariant from `_compute_name`:
```python
if not move.posted_before and not move._sequence_matches_date():
    move.name = False; continue
```

`posted_before` is set to `True` in `_post()` and NEVER reset (not even by `button_draft()`). Therefore:
- A move that has ever been posted keeps its sequence name forever.
- `_sequence_matches_date()` checks that the name's year/month matches the move's date; if not, the name is reset ONLY if `posted_before == False`.
- `action_switch_move_type()` (L5953-L5977): raises `ValidationError` if `posted_before and name` — you cannot change the type of a posted (or previously-posted) document.

**Axis classification:** DETERMINISTIC.  
K-step: K3.

---

## 3. Enterprise / Unresolved Flags

**Enterprise gap (community only):**  
- `account_asset` (K12 Anlagen): the hash fields and sequence mixin are present in community account_move but asset depreciation postings are Enterprise. The structure (account.move + account.move.line) is shared.
- `account_reports` (K8 BWA/SuSa/GuV etc.): entirely Enterprise. Community exposes only `account.move.line` aggregation.
- The `account.lock_date` / `tax_lock_date` field references in `_check_balanced` and `_post` are community — only the hard-lock date enforcement (GoBD fiscal year lock) is present.

**No unresolved ontology classes for this lane:**  
Both `odoo:account.move` and `odoo:account.move.line` have confirmed rows in `ODOO_ALIGNMENTS` (lance-graph-callcenter `odoo_alignment.rs:L132-L143`):
```
odoo:account.move     → fibo:Transaction      → SmbFoundryInvoice (0x81) → DOLCE Perdurant
odoo:account.move.line → fibo:JournalEntryLine → SmbFoundryInvoice (0x81) → DOLCE Perdurant
```
(Note: the `odoo_alignment.rs` row for `account.move.line` explicitly sets Perdurant, overriding the Endurant that the suffix heuristic would yield. The test at line ~456 documents this curated override.)

**woa-rs gap analysis:**  
Grep of `/home/user/woa-rs/src/` and `/home/user/woa-rs/crates/` for posting/sequence terms found:
- `erp_gobd_festschreibung` on `tenants` table + `models/tenant.rs` — K11 stub EXISTS.
- `after_hash String(64)` on `ErpJournal` (k2_journal.rs) — K11 hash column EXISTS but documented as "engine's job" (Sprint-3).
- `buchungsnummer: i64` on `ErpJournal` — sequence number exists, gap-free UNIQUE constraint documented.
- `belegnummer: String` on `ErpOpenItemAR` and `ErpSupplierInvoice` — Belegnummer string fields exist.
- **MISSING:** No `_post`-equivalent service layer yet (`src/erp/posting.rs` does not exist).
- **MISSING:** No balance-check (`SUM(debit) == SUM(credit)`) implementation.
- **MISSING:** No sequence mixin logic (the gap-prevention locking pattern via savepoints).
- **MISSING:** No hash-chain implementation at the move-level (only column declared; `_shared::chain_hash` exists but is not wired).
- **MISSING:** No storno / `_reverse_moves` equivalent (comments reference the Gegenbuchung pattern but no service exists).

**woa-rs is significantly thinner than odoo in K3.** The schema layer is partially present; the service layer is absent.

---

## 4. Porter's Checklist — Non-Obvious Gotchas

1. **`validation_msgs` is a `set`, not a list.** Duplicate error messages are silently deduplicated. The error raised is `"\n".join(sorted(validation_msgs))` — order is non-deterministic. Rust: use a `BTreeSet<String>` to get stable ordering.

2. **`soft=True` default means future-dated moves are NOT posted but flipped to `auto_post='at_date'`.** The cron `_autopost_draft_entries` picks them up later. Rust: implement the cron trigger or expose it as a separate `autopost_scheduled_entries()` method.

3. **Balance check runs on DB-flushed data, not ORM cache.** The SQL `HAVING ROUND(SUM(balance), decimal_places) != 0` is authoritative. Do not rely on in-memory balance sums for the constraint — flush first.

4. **Company currency vs line currency:** Balance check rounds to COMPANY currency `decimal_places`, not the invoice's foreign currency. A EUR company posting a USD invoice: the USD amounts are converted to EUR first (via `balance` column which is always in company currency), then the rounding is applied to company-currency decimal places (2 for EUR).

5. **`_locked_increment` uses a SAVEPOINT loop.** This is not a simple SELECT MAX(sequence_number) + 1. It uses an `UPDATE + catch UniqueViolation + retry` pattern to guarantee gap-free sequences under concurrent load. Rust: implement this with a PostgreSQL `UPDATE ... RETURNING` inside a transaction savepoint, or use a dedicated sequence table with `SELECT ... FOR UPDATE`.

6. **`posted_before` is permanent.** Once set to `True` in `_post()`, it never goes back to `False` — not even `button_draft()` resets it. This means `_compute_name` will never reset a previously-posted move's name based on date mismatch. Rust: store `posted_before` as a non-nullable boolean column with a migration that seeds existing rows as `false`.

7. **Storno flag (`is_storno`) flips debit/credit PRESENTATION only.** The underlying `balance` is still positive for a storno-debit. The visual debit becomes `balance` (instead of `max(balance, 0)`) when `is_storno=True`. Porter must implement this in any ledger display layer.

8. **Hash chain is per `(journal_id, sequence_prefix)` pair, NOT per journal alone.** A journal with both normal invoices (prefix `INV/2024/`) and refunds (prefix `RINV/2024/`) has TWO independent hash chains. Each chain is locked separately.

9. **`_getattrstring` for `monetary` fields uses `float_repr(value, currency.decimal_places)` in hash_version >= 3.** `float_repr` in odoo formats to exactly N decimal places (e.g. `"1234.56"`). Use the same representation in Rust: `format!("{:.prec$}", value, prec = decimal_places)` — NOT `Decimal::to_string()` which may omit trailing zeros.

10. **`_calculate_hashes` strips version prefix before chaining:** `if previous_hash.startswith("$"): previous_hash = previous_hash.split("$")[2]`. The raw sha256 hex string is used as input, not the full `$4$<hex>` formatted value. But the stored value IS the formatted `$4$<hex>`. Rust: strip version prefix before computing next hash; store with prefix.

11. **German delivery date auto-fill happens BEFORE `super()._post()`.** The l10n_de hook mutates `delivery_date` on the ORM object before the base post flow runs. For DE-locale moves, `delivery_date` must be populated before the hash is computed (if the hash includes `delivery_date` — currently it does not, but worth checking future hash field additions).

12. **`_sync_tax_lines` is a contextmanager called on `create`/`write`**, not on `_post`. Tax lines are kept up-to-date continuously in draft state. By the time `_post()` runs, tax lines should already be correct. `_post` itself does NOT re-sync taxes — it only validates.

13. **Reconciliation is triggered during `_post` (L5710-L5711).** If a move has a `reversed_entry_id` that is already posted, they are reconciled together during posting. This means the Rust posting service must call reconciliation logic as a post-posting step, not before.

14. **`sequence_override_regex` on journal:** The journal can override the regex used for sequence format detection. If a journal has `sequence_override_regex` set, it takes priority over the mixin's built-in regexes (account_move.py:L84-L101). Rust: check `journal.sequence_override_regex` before applying the default regex table.

---

## 5. Axis-2 Delegation Tags (Heuristic Rules)

### Axis-2.A — Auto-post bills wizard

**Rule:** `_show_autopost_bills_wizard()` — recommends enabling automatic bill posting after observing 3+ consecutive unmodified bills from the same partner.

**Contract tuple:**  
`(PostingAnomaly, Induction, NarsTruth, Analytical)`
- `ReasoningKind::PostingAnomaly` — closest match (anomaly = detecting a pattern in posting behaviour)
- `InferenceType::Induction` — "things-like-X" reasoning (pattern from past bills)
- `SemiringChoice::NarsTruth` — truth accumulation over multiple observations (3 bills)
- `ThinkingStyle cluster: Analytical` — inherited from `SmbFoundryInvoice` family (billing/posting domain → Analytical/Critical per briefing)

**Reasoning surface call:** `Reasoner::reason(ReasoningContext { namespace: "posting.autopost_recommendation", kind: PostingAnomaly, evidence: [partner_id, nb_unmodified_bills], budget: ... })`

### Axis-2.B — Lock-date auto-advance

**Rule:** `move.date = move._get_accounting_date(...)` when the date falls in a locked period — automatically advances the date to the next open period.

**Classification borderline:** The odoo implementation is deterministic (find the next date after the lock date). However, multi-factor lock date handling (tax lock vs fiscal lock vs hard lock) has judgment aspects. For woa-rs, implement as DETERMINISTIC (advance to `lock_date + 1 day`).

**If delegated:**  
`(PostingAnomaly, Abduction, NarsTruth, Analytical)`
- `Abduction`: "why is this date wrong / what period should this go to"

---

## 6. Ontology Mapping Summary

| Odoo class | OWL/FIBO pivot | OGIT family | DOLCE |
|---|---|---|---|
| `odoo:account.move` | `fibo:Transaction` | `SmbFoundryInvoice` (0x81), slot TRANSACTION (0x03) | Perdurant |
| `odoo:account.move.line` | `fibo:JournalEntryLine` | `SmbFoundryInvoice` (0x81), slot JOURNAL_LINE (0x06) | Perdurant (curated override; suffix heuristic gives Endurant) |

Both resolved in `/home/user/lance-graph/crates/lance-graph-callcenter/src/odoo_alignment.rs:L132-L143`. No new alignment rows needed for this lane.

**Inherited ThinkingStyle cluster (Axis-2):** `Analytical` — from `SmbFoundryInvoice` family (billing/accounting domain). Posting-anomaly checks inherit Analytical/Critical per the briefing's family table.

---

## 7. K-Step Cross-Reference

| Rule | K-step |
|---|---|
| K3-1 `_post()` | K3 |
| K3-2 `action_post()` | K3 |
| K3-3 `_check_balanced()` | K3 |
| K3-4 balance/debit/credit/amount_currency computes | K3 |
| K3-5 line constraints | K3 |
| K3-6 `_reverse_moves()` / Storno | K3 + K11 |
| K3-7 `button_draft`/`button_cancel` | K3 + K11 |
| K3-8 Sequence / Belegnummer | K3 |
| K3-9 Tax line generation | K3 + K7 |
| K3-10/K11 Hash chain / Festschreibung | **K11** |
| K3-11 German l10n_de override | K3 + K7 (UStG date) |
| K3-12 `posted_before` / name freeze | K3 |

---

Read: /home/user/odoo/addons/account/models/account_move.py lines=7328 depth=full  
Read: /home/user/odoo/addons/account/models/account_move_line.py lines=3742 depth=full  
Read: /home/user/odoo/addons/account/models/sequence_mixin.py lines=512 depth=full  
Read: /home/user/odoo/addons/l10n_de/models/account_move.py lines=19 depth=full
