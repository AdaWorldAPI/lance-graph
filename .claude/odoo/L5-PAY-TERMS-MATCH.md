RICHNESS-LANE-OK

# L5 — Payments · Payment Terms · Reconcile-Model Matching Rules

**Lane:** L5 — Payments + Payment Terms + Reconcile-model matching  
**Analyst:** Claude Sonnet 4.6, read-only pass, 2026-05-26  
**K-steps covered:** K3 (double-entry posting), K5 (bank matching), Mahnwesen-precursor

---

## 1. Scope and files read

| File | Lines | Depth |
|---|---|---|
| `addons/account/models/account_payment.py` | 1247 | full |
| `addons/account/models/account_payment_term.py` | 368 | full |
| `addons/account/models/account_reconcile_model.py` | 201 | full |

woa-rs calibration targets examined:

- `src/erp/engine/bank_match.rs` (303 LoC, existing implementation)
- `src/models/erp/k3_debitors.rs` (debtors + Mahnwesen model layer)
- `src/models/erp/k4_creditors.rs` (creditors + payment run model layer)
- `src/contracts/erp/k3_debitors.rs` (contract DTOs)

---

## 2. Rule Sections

---

### RULE P1 — Payment move-line generation: `_prepare_move_lines_per_type` + `_generate_journal_entry`

**File:** `account_payment.py:L311-L408, L1069-L1101`

#### Axis-1 Rich-AST Spec

**Entry point chain:**
`action_post (L1126)` sets `state='in_process'` (or `'paid'` for cash accounts, L1141).
`write(vals)` override (L947) detects `state in ('in_process','paid')`. If `not pay.move_id`:
calls `_generate_journal_entry()` then posts the move with `action_post()`.

**`_generate_journal_entry` (L1069-L1079):**

```
need_move = self.filtered(lambda p: not p.move_id and p.outstanding_account_id)
assert len(self)==1 or (not write_off_line_vals and not force_balance and not line_ids)
move_vals = [pay._generate_move_vals(...) for pay in need_move]
moves = env['account.move'].create(move_vals)
for pay, move in zip(need_move, moves):
    pay.write({'move_id': move.id, 'state': 'in_process'})
```

**`_generate_move_vals` (L1081-L1101):** Returns:

```python
{
    'move_type': 'entry',           # ALWAYS 'entry', not invoice type
    'ref': self.memo,
    'date': self.date,
    'journal_id': self.journal_id.id,
    'company_id': self.company_id.id,
    'partner_id': self.partner_id.id,
    'currency_id': self.currency_id.id,
    'partner_bank_id': self.partner_bank_id.id,
    'line_ids': line_ids or [Command.create(lv) for lv in self._prepare_move_line_default_vals(...)],
    'origin_payment_id': self.id,
}
```

**`_prepare_move_lines_per_type` (L311-L391) - core journal-line factory:**

1. Guard: if `not outstanding_account_id` raise UserError (L323-L326).
2. Label: `line_name = ''.join(x[1] for x in _get_aml_default_display_name_list())`. Built from `payment_method_line_id.name` + optional `": " + memo` (L329).
3. Withholding lines: `_prepare_move_withholding_lines({})` returns `[]` in community (L283-L285). Enterprise-only hook.
4. Mutual exclusion: if both `withholding_lines` and `write_off_lines` non-empty, `write_off_lines` is silently cleared (L342-L346).
5. Liquidity amount:
   - `inbound`: `liquidity_amount_currency = +self.amount`
   - `outbound`: `liquidity_amount_currency = -self.amount`
   - other: `0.0`
6. Balance (company-currency):
   - If `force_balance` set AND no `write_off_line_vals`: `sign = 1 if liq_amt > 0 else -1; liquidity_balance = sign * abs(force_balance)` (L358-L360)
   - Otherwise: `liquidity_balance = currency._convert(liq_amount_currency, company_currency, company, date)` (L362-L367)
7. Withholding subtraction from liquidity (L368-L369):
   `liquidity_amount_currency -= withholding_amount_currency`
   `liquidity_balance -= withholding_balance`
8. Counterpart (always residual, closed-form double-entry):
   `counterpart_amount_currency = -liquidity_amount_currency - write_off_amount_currency - withholding_amount_currency`
   `counterpart_balance = -liquidity_balance - write_off_balance - withholding_balance`

**Line dictionaries (L287-L309):** Both liquidity and counterpart lines share:

```python
{
    'name': line_name,
    'date_maturity': self.date,
    'partner_id': self.partner_id.id,
    'account_id': <outstanding_account_id or destination_account_id>,
    'currency_id': self.currency_id.id,
    'balance': <computed>,
    'amount_currency': <computed>,
}
```

**`_seek_for_lines` helper (L214-L242) - dispatcher for existing move lines:**

Iterates `move_id.line_ids` and classifies into three buckets:

- liquidity_lines: `line.account_id in _get_valid_liquidity_accounts()` (L227)
  Valid liquidity = `journal.default_account_id | payment_method_line.payment_account_id | inbound_method_lines.payment_account_id | outbound_method_lines.payment_account_id | outstanding_account_id` (L244-L252)
- counterpart_lines: `line.account_id.account_type in ['asset_receivable','liability_payable']` or `== company.transfer_account_id` (L229)
- writeoff_lines: everything else

Edge case (L236-L240): if exactly one writeoff line exists and liquidity or counterpart bucket is empty, the writeoff line is promoted into the missing slot.

**`_synchronize_to_moves` (L995-L1060) - Payment to Move sync:**

Trigger fields (L1063-L1067): `date, amount, payment_type, partner_type, payment_reference, currency_id, partner_id, destination_account_id, partner_bank_id, journal_id`.

Flow:
1. Skip if `move_id.state == 'posted'` (L1004) - posted entries are immutable.
2. Preserve existing write-off amounts (L1012-L1022).
3. `zip_longest` over liquidity_lines vs new vals: Command.update / .create / .delete (L1027-L1033).
4. Counterpart: single line, always update-or-create (L1035-L1040).
5. Write-off lines: delete old (`(2, id, 0)` = ORM unlink), create new (L1042-L1045).
6. Write to move: `date, partner_id, currency_id, partner_bank_id, line_ids` always;
   `journal_id + name='/'` only if `journal_id` changed (L1048-L1060).
7. Context `skip_invoice_sync=True` prevents recursion.

**`_compute_destination_account_id` (L625-L647):**

- `partner_type='customer'`: `partner.property_account_receivable_id` (or search `asset_receivable`).
- `partner_type='supplier'`: `partner.property_account_payable_id` (or search `liability_payable`).

**`_compute_outstanding_account_id` (L621-L623):**
`outstanding_account_id = payment_method_line_id.payment_account_id`

**`_compute_currency_id` (L616-L618):**
`currency = journal.currency_id or journal.company_id.currency_id`

**State machine (L36-L48, L1126-L1163):**

```
draft       --[action_post]--> in_process   (general)
draft       --[action_post]--> paid         (asset_cash journal, L1141)
in_process  --[_compute_state: liq_residual==0 or no reconcile flag]--> paid
in_process  --[all linked invoices paid]--> paid
any         --[action_cancel]--> canceled   (draft moves unlinked, posted moves button_cancel)
any         --[action_reject]--> rejected
canceled/in_process --[action_draft]--> draft
```

**`_compute_state` (L453-L467):**

```python
if state in ('paid','in_process') and move_id:
    liquidity = _seek_for_lines()[0]
    state = ('paid'
        if company_currency.is_zero(sum(liquidity.amount_residual))
           or not any(liquidity.account_id.reconcile)
        else 'in_process')
if state=='in_process' and all linked invoices paid:
    state = 'paid'
```

Critical: `or not any(liquidity.account_id.reconcile)` means if the liquidity account has `reconcile=False`, payment goes to 'paid' even with non-zero residual (direct bank account path, no statement needed).

**`_compute_reconciliation_status` (L469-L497):**

```
is_matched   = liquidity lines' residual == 0
             OR journal uses its own default_account_id directly (L490-L493)
is_reconciled = counterpart+writeoff lines with account.reconcile==True have residual == 0
```

Currency selector (L488): `amount_residual` if pay.currency == company.currency; else `amount_residual_currency`.

**Constraints (L863-L882):**

- `@api.constrains('payment_method_line_id')`: must not be null; must match journal (L863-L872).
- `@api.constrains('state', 'move_id')`: posted payment with outstanding_account_id must have move (L874-L882).
- SQL: `CHECK(amount >= 0.0)` (L199-L202).

**Axis classification:** DETERMINISTIC
**K-step:** K3 (double-entry posting), K5 (outstanding account / bank matching via `is_matched`)
**woa-rs target:** `src/erp/engine/payment.rs` (new), `src/models/erp/k5_bank.rs` (outstanding account linkage)

**Ontology mapping:**
`odoo:account.payment` -- FLAG: UNRESOLVED in odoo_alignment.rs --
Proposed: `odoo:account.payment` -> `fibo-FBC-PAS-FPAS:Payment` -> OGIT family SMBAccounting/BillingCore -> DOLCE Perdurant (a payment is an event with date + state transitions; .payment suffix -> Perdurant by analogy with .move).

---

### RULE P2 — Payment term computation: `_compute_terms`

**File:** `account_payment_term.py:L171-L256`

#### Axis-1 Rich-AST Spec

**Signature:**

```python
def _compute_terms(self, date_ref, currency, company, tax_amount, tax_amount_currency,
                   sign, untaxed_amount, untaxed_amount_currency, cash_rounding=None) -> dict
```

**Output structure:**

```python
{
    'total_amount': float,              # company-currency total
    'discount_percentage': float,       # 0.0 if no early_discount
    'discount_date': date | False,      # date_ref + relativedelta(days=discount_days)
    'discount_balance': float,          # company-currency amount after Skonto
    'discount_amount_currency': float,  # move-currency amount after Skonto
    'line_ids': [
        {
            'date': date,              # due date
            'company_amount': float,   # balance in company currency
            'foreign_amount': float,   # balance in move currency
        }, ...
    ]
}
```

**Step 1 - totals (L187-L190):**

```python
total_amount = tax_amount + untaxed_amount
total_amount_currency = tax_amount_currency + untaxed_amount_currency
rate = abs(total_amount_currency / total_amount) if total_amount else 0.0
```

The exchange rate is embedded from move amounts, NOT from a live currency lookup.
This means any fixed invoice rate is preserved through all term calculations.

**Step 2 - early discount (Skonto) computation (L200-L214):**

Guard: `early_discount == True`. Constraint ensures only single-line 100% terms can have early_discount.

Three computation modes (controlled by `early_pay_discount_computation`):

| Mode | Formula for `discount_balance` (company currency) | Country default |
|---|---|---|
| `'included'` | `company_currency.round(total_amount * (1 - pct))` | DE, AT, CH, most |
| `'excluded'` | `company_currency.round(total_amount - untaxed_amount * pct)` | NL |
| `'mixed'`    | same formula as 'excluded' | BE |

Where `pct = self.discount_percentage / 100.0`.

Parallel computation for `discount_amount_currency` using `currency.round(...)`.

Cash-rounding adjustment for discount (L210-L214):

```python
if cash_rounding:
    diff = cash_rounding.compute_difference(currency, discount_amount_currency)
    if not currency.is_zero(diff):
        discount_amount_currency += diff
        discount_balance = company_currency.round(discount_amount_currency / rate) if rate else 0.0
```

The company-currency discount balance is re-derived from the cash-rounded move-currency value.

**Step 3 - term lines loop (L219-L254):**

```
residual_amount = total_amount
residual_amount_currency = total_amount_currency

for i, line in enumerate(self.line_ids):  # ordered by id (insertion order)
    on_balance_line = (i == len(self.line_ids) - 1)  # LAST line always = residual
```

Per-line computation:

- Balance line (last): `company_amount = residual_amount; foreign_amount = residual_amount_currency`
- Fixed line: `company_amount = sign * company_currency.round(line.value_amount / rate) if rate else 0.0`
  `foreign_amount = sign * currency.round(line.value_amount)`
- Percent line: `company_amount = company_currency.round(total_amount * (line.value_amount / 100.0))`
  `foreign_amount = currency.round(total_amount_currency * (line.value_amount / 100.0))`

Cash rounding for non-balance lines (L242-L250): same diff pattern; company_amount re-derived via rate.

Residual tracking: `residual_amount -= term_vals['company_amount']` after each non-balance line. The last line absorbs all rounding differences.

**Rounding:** `company_currency.round(...)` delegates to `res.currency` precision. `float_round` is used ONLY in `_check_lines` constraint, not in `_compute_terms`.

**`_check_lines` constraint (L156-L169):**

- Sum of percent lines must equal 100 (using `float_round` with `Payment Terms` decimal precision).
- If `early_discount=True` and `len(line_ids) > 1`: ValidationError.
- `discount_percentage` must be > 0.0.
- `discount_days` must be > 0.

**Due-date computation `_get_due_date` (L310-L327) on AccountPaymentTermLine:**

| delay_type | Formula |
|---|---|
| `'days_after'` | `date_ref + relativedelta(days=nb_days)` |
| `'days_after_end_of_month'` | `end_of(date_ref, 'month') + relativedelta(days=nb_days)` |
| `'days_after_end_of_next_month'` | `end_of(date_ref + relativedelta(months=1), 'month') + relativedelta(days=nb_days)` |
| `'days_end_of_month_on_the'` | see below |

For `'days_end_of_month_on_the'` (L318-L326):
- `days_next_month` is a string field (size=2), parsed with `int()`. `ValueError` -> default to 1.
- If `days_next_month == 0` after int(): returns `end_of(date_ref + relativedelta(days=nb_days), 'month')`.
- Otherwise: `date_ref + relativedelta(days=nb_days) + relativedelta(months=1, day=days_next_month)`.

**`_compute_discount_computation` (L82-L90):** Country-specific defaults at payment term creation:

- `'BE'` -> `'mixed'`
- `'NL'` -> `'excluded'`
- everything else -> `'included'`

**`_get_amount_due_after_discount` (L61-L79):** Simplified display version:

```python
if early_pay_discount_computation in ('excluded', 'mixed'):
    discount_amount_currency = (total - untaxed) * pct  # tax-only portion
else:
    discount_amount_currency = total * pct               # full total
amount_due = currency.round(total - discount_amount_currency)
# optional cash_rounding adjustment follows same pattern
```

**`_compute_value_amount` (L359-L367):** Auto-fills percent value_amount so all lines sum to 100.

**`_compute_days` (L350-L356):** Auto-fills `nb_days` for new lines to `last_line.nb_days + 30`.
Only fires when `not line.nb_days and len(payment_id.line_ids) > 1`.

**Axis classification:** DETERMINISTIC
**K-step:** K3 (payment terms feed invoice due dates -> Mahnwesen escalation timing + Skonto)
**woa-rs target:** `src/erp/engine/payment_term.rs` (new module - no current equivalent found).
The flat fields `ErpDebtor.skonto_prozent / skonto_tage` (k3_debitors.rs) are snapshots; this module is the structured source.

**Ontology mapping:**
`odoo:account.payment.term` -- FLAG: UNRESOLVED in odoo_alignment.rs --
Proposed: `odoo:account.payment.term` -> `fibo-FBC-PAS-FPAS:PaymentObligationTerms` (preferred) or `ubl:PaymentTerms` -> OGIT family SMBAccounting/BillingCore -> DOLCE Abstract (.term suffix -> Abstract by briefing rule).

`odoo:account.payment.term.line` -> sub-component of PaymentObligationTerms, no independent alignment row. DOLCE Abstract.

---

### RULE P3 — Early Payment Discount (Skonto) mode selection

**File:** `account_payment_term.py:L41-L46, L82-L90`

#### Axis-1 Rich-AST Spec

`early_pay_discount_computation` is a stored computed Selection, re-evaluated when `company_id` changes.
Three modes affect only the Skonto base (whether VAT is discounted or not):

| Mode | German accounting concept | Discount base |
|---|---|---|
| `'included'` | Skonto auf Bruttobetrag | `total_amount * pct` |
| `'excluded'` | Skonto nur auf Nettobetrag (VAT stays full) | `untaxed_amount * pct` |
| `'mixed'`    | Skonto auf Nettobetrag (BE) | same as 'excluded' |

For Germany (DE): always `'included'` - Skonto reduces both net and VAT proportionally.
This is the canonical "14 Tage netto, 2% Skonto bei 8 Tagen" case.

The `early_discount` boolean is a user toggle. Defaults: `discount_percentage=2.0`, `discount_days=10`.

Constraint: `early_discount=True` is only valid with single-line 100% terms (L163-L165).

**Axis classification:** DETERMINISTIC
**K-step:** K3 + K7 (Skonto affects VAT base in 'excluded'/'mixed' modes)
**woa-rs target:** `src/erp/engine/payment_term.rs`, enum `SkontoMode { Included, Excluded, Mixed }`.

---

### RULE P4 — Payment reconciliation status and Mahnwesen integration

**File:** `account_payment.py:L453-L497`

#### Axis-1 Rich-AST Spec

Two distinct boolean flags serve different purposes:

**`is_matched` (bank reconciliation, K5):**
True when liquidity lines' residual is zero, OR when the journal uses its own default_account_id directly (direct bank path without statement lines, L490-L493).

**`is_reconciled` (invoice clearance, K3):**
True when counterpart+writeoff lines that have `account_id.reconcile=True` all have zero residual.

These are INDEPENDENT. A payment can be `is_matched=True, is_reconciled=False` (bank acknowledged but invoice not yet cleared). Mahnwesen MUST check `is_reconciled`, not `is_matched`, before escalating dunning.

Currency selector (L488): `amount_residual` when pay.currency == company.currency; else `amount_residual_currency`.

**@api.depends (L469):** `move_id.line_ids.amount_residual`, `amount_residual_currency`, `account_id`, `state` - all four must be fresh.

**Axis classification:** DETERMINISTIC
**K-step:** K3 (open-item clearance), K5 (bank match)
**woa-rs target:** feeds `src/erp/engine/debitor.rs` Mahnwesen logic (already exists). The `ErpOpenItemAR.offen_betrag` open-item balance in woa-rs is the equivalent of `amount_residual`.

---

### RULE P5 — Reconcile-model matching rules (NARS-HEAVY, Axis-2)

**File:** `account_reconcile_model.py:L91-L200` (model), `L8-L89` (line sub-model)

**THIS RULE IS NARS-HEAVY.** The reconcile.model matching is a textbook Axis-2 heuristic: multi-dimensional filter-and-score over incoming bank statement lines against a set of configured rules. The outcome is not deterministic from the data alone — it depends on which models exist, their sequence order, and how multiple evidence dimensions combine.

#### Axis-1 Rich-AST Spec

**Match dimensions defined on `AccountReconcileModel`:**

**Dimension 1 - Journal filter (`match_journal_ids`, L123-L126):**
Many2many of `account.journal`. Model only applies when the statement line's journal is in this set.
Empty set = applies to all journals. Hard in/out filter, no score contribution.

**Dimension 2 - Amount filter (`match_amount` + `match_amount_min/max`, L127-L134):**
Selection: `lower | greater | between`. Compared against statement line amount.
Hard filter: if set, model only fires for amounts in range.

**Dimension 3 - Label/communication filter (`match_label` + `match_label_param`, L135-L143):**
Selection: `contains | not_contains | match_regex`.
- `contains`: case-insensitive substring of statement label/narration/transaction details.
- `not_contains`: negation of contains.
- `match_regex`: compiled Python regex against the same fields.
Validated at save via `_check_match_label_param` (L149-L156). This is the primary textual evidence dimension.

**Dimension 4 - Partner filter (`match_partner_ids`, L144-L145):**
Many2many of `res.partner`. Hard filter on statement line's partner.

**Trigger (`trigger`, L107-L108):**
`manual` (proposed, user confirms) vs `auto_reconcile` (applied automatically).
Controls enforcement level, not matching.

**`can_be_proposed` computed field (L158-L161):**

```python
can_be_proposed = (not mapped_partner_id
                   and (match_label or match_amount or match_partner_ids
                        or trigger == 'auto_reconcile'))
```

Depends on: `mapped_partner_id, match_label, match_amount, match_partner_ids, trigger`.

**`mapped_partner_id` computed field (L163-L167):**

```python
is_partner_mapping = (match_label
                      and len(line_ids) == 1
                      and line_ids[0].partner_id
                      and not line_ids[0].account_id)
mapped_partner_id = line_ids[0].partner_id if is_partner_mapping else False
```

Depends on: `match_label, line_ids.partner_id, line_ids.account_id`.
A model with exactly one line having partner but no account = partner-mapping rule (not a reconcile candidate).

**Model sequencing:** `_order = 'sequence, id'` (L94). Odoo applies models in sequence order, takes the first match. This greedy first-match is what makes the community implementation heuristic-shaped even if individual dimensions are deterministic.

**Write-off line sub-model (`AccountReconcileModelLine`, L8-L89):**

When a model matches, it generates journal entries via `line_ids`. Each line:

- `account_id`: contra-account for write-off/categorisation.
- `amount_type` (L25-L34): `fixed | percentage | percentage_st_line | regex`.
  - `fixed`: hard-coded value (validated non-zero, L78-L79).
  - `percentage`: percentage of matched open item's balance (validated non-zero, L82-L83).
  - `percentage_st_line`: percentage of statement line's own amount (validated non-zero, L80-L81).
  - `regex` (L36-L51): extracts amount from statement label. First capturing group = integer part, optional second = decimal part (two-digit). Regex validated at save (L84-L88). Default pattern: `([\d,]+)`.
- `tax_ids`: taxes on the write-off amount.
- `partner_id` + `label`: for the generated line.
- `amount` (L36): `float` computed from `amount_string` via `float(amount_string)` with fallback 0 on ValueError (L68-L73).

**`action_reconcile_stat` (L175-L188):**
SQL: `SELECT ARRAY_AGG(DISTINCT move_id) FROM account_move_line WHERE reconcile_model_id = %s`.
Returns journal entries that were created by this model. Not a matching method, just a stat.

---

#### Axis-2 Classification: HEURISTIC -> NARS delegation

**NARS contract tuple:**

```
ReasoningKind:   Other("BankStatementMatch")
InferenceType:   Induction (primary) with Abduction fallback
SemiringChoice:  NarsTruth
ThinkingStyle:   Analytical (INHERITED from SMBAccounting/BillingCore OGIT family)
```

**Justification - `ReasoningKind: Other("BankStatementMatch")`:**

None of the four named kinds fit precisely:
- `CustomerCategory`: partner classification. Not this.
- `PostingAnomaly`: anomaly detection on existing postings. Closest alternative (unmatched statement lines are anomalous open items), but the primary operation is identification not anomaly detection.
- `NextBestAction`: forward-looking recommendation. Not this.
- `InvoiceCompleteness`: document completeness. Not this.

Bank statement matching is identification of what a statement line IS from its properties. Use `Other` with proposed name `BankStatementMatch`. Flag: propose adding to the enum in `lance_graph_contract`.

**Justification - `InferenceType: Induction` (primary) with `Abduction` fallback:**

Induction: "Things with label containing 'RG-2024-0078' and amount 1234.56 are usually payment for invoice RG-2024-0078." This is pattern generalisation - the model learned (from user configuration) that these evidence properties co-occur with a particular open-item type. The reconcile model configuration IS the inductive knowledge base.

Abduction fallback: when no single model matches with high confidence (ambiguous label, multiple open items with matching amount), the reasoner should switch to abduction - "what is the most likely explanation for this statement line?" - generating hypotheses from open items and evaluating them against evidence. This handles the multi-candidate ranking case that odoo's greedy first-match cannot.

**Justification - `SemiringChoice: NarsTruth`:**

Each match dimension contributes independent evidence. Odoo community applies all dimensions as hard AND-filters, but the underlying problem structure is multi-dimensional evidential: a label match raises frequency, a partner match raises confidence, an amount match in range raises both. NarsTruth (frequency f, confidence c) allows evidence fusion across dimensions. A high-frequency (many similar past matches) high-confidence (multiple independent dimensions agree) match gets the highest truth value. Boolean would collapse all nuance; HammingMin would be too harsh on partial matches; NarsTruth is the right semiring for graded multi-dimensional evidence fusion.

**Justification - `ThinkingStyle: Analytical` (inherited from SMBAccounting/BillingCore):**

`account.reconcile.model` maps to the SMBAccounting/BillingCore OGIT family (bank statement + open-item reconciliation is core accounting infrastructure). The briefing states SMBAccounting/BillingCore posting-anomaly checks inherit Analytical/Critical. Statement matching is fundamentally Analytical: decompose the statement line into evidence dimensions, compare against known candidates, select by rule-based pattern. Not Creative (no novel generation), not Empathic (no partner intent modelling), not Exploratory (domain is fixed). Analytical is correct.

**ReasoningContext shape for woa-rs:**

```rust
ReasoningContext {
    namespace: "erp.bank_statement.match",
    kind: ReasoningKind::Other(/* BankStatementMatch id */),
    evidence: &[
        // statement line properties
        ("st_line.amount",        "1234.56"),
        ("st_line.currency",      "EUR"),
        ("st_line.label",         "SVWZ+RG-2024-0078 Mustermann GmbH"),
        ("st_line.partner_iban",  "DE89370400440532013000"),
        ("st_line.date",          "2026-05-15"),
        ("st_line.journal_id",    "42"),
        // candidate open items (from ErpOpenItemAR / ErpOpenItemAP)
        ("candidate[0].id",         "881"),
        ("candidate[0].amount",     "1234.56"),
        ("candidate[0].belegnummer","RG-2024-0078"),
        ("candidate[0].partner",    "Mustermann GmbH"),
        // reconcile model rule propositions (from DB)
        ("model[0].seq",            "10"),
        ("model[0].label_match",    "contains:RG-"),
        ("model[0].amount_range",   "between:1000:2000"),
        // ... additional candidates and models
    ],
    budget: Budget { steps: 50, confidence_threshold: 0.75 },
}
// Returns: Conclusion { match_id: Option<i64>, confidence: f32, write_off_lines: Vec<...> }
```

woa-rs then applies the match (updating `match_status`, `match_confidence`, FK) using the existing `bank_match.rs` engine infrastructure.

**Mapping to existing woa-rs `bank_match.rs`:**

`src/erp/engine/bank_match.rs` (Sprint-3b) already implements a deterministic scoring engine with hard confidence levels (100/80/60/40/0), mirroring `../WoA/woa/erp/engine/bank_match.py`. This covers the community equivalent of `trigger='manual'` with fixed match criteria.

The odoo reconcile-model adds on top of this:
- Configurable label regex/contains patterns (richer than `extract_belegnummern`).
- Partner-mapping rules (absent from bank_match.rs).
- Write-off line generation with amount extraction from label regex (absent).
- Auto-reconcile trigger (bank_match.rs only proposes, never auto-posts).

Gap: woa-rs `bank_match.rs` covers approximately 40% of odoo's reconcile-model surface. The NARS delegation handles the configurable-pattern matching and write-off line selection; the deterministic hard match (belegnummer, IBAN, exact amount) stays in Rust.

**K-step:** K3 (open-item clearance), K5 (bank statement matching)
**woa-rs target:** `src/erp/engine/bank_match.rs` (extend existing) + `src/erp/engine/reconcile_match.rs` (new, NARS-delegating layer).

**Ontology mapping:**
`odoo:account.reconcile.model` -- FLAG: UNRESOLVED in odoo_alignment.rs --
Proposed: `odoo:account.reconcile.model` -> `sh:NodeShape` / `sh:rule` pattern (a reconcile model is a declarative rule shape over statement line properties; when the shape is satisfied, a particular accounting action fires) -> OGIT family SMBAccounting -> DOLCE Abstract (.model suffix -> Abstract by briefing rule).
Note: No FIBO class covers reconciliation rules (FIBO focuses on instruments and obligations). SHACL sh:rule is the closest W3C standard. The `can_be_proposed` / `trigger` fields parallel `sh:condition` / `sh:action` patterns.

`odoo:account.reconcile.model.line` -> sub-component of sh:NodeShape rule, no independent alignment row. DOLCE Abstract.

---

## 3. Enterprise / Unresolved Flags

### Enterprise Boundary

**E1 - `_prepare_move_withholding_lines` (account_payment.py:L283-L285):**
Community returns `[]`. Enterprise modules override to generate WHT lines. Hook is called at L337; always `[]` in community. Porter: implement hook with empty default; override is future extension point.

**E2 - `_valid_payment_states` (account_payment.py:L254-L258):**
Enterprise: returns `['in_process', 'paid']`. Community: `['in_process']`.
`account.move._get_invoice_in_payment_state()` returns `'paid'` in Enterprise, `'in_payment'` in community.
Affects when a payment is considered "valid" for open-item matching.

**E3 - Auto-reconcile trigger:**
`trigger='auto_reconcile'` field stored in community, but the auto-application engine is in Enterprise bank statement widget. Community stores the flag but does not execute it. Flag: woa-rs can store and read `trigger` but auto-application requires NARS delegation (RULE P5).

### UNRESOLVED Ontology Alignments

| odoo class | Status | Proposed owl pivot | Proposed OGIT family | DOLCE |
|---|---|---|---|---|
| `account.payment` | UNRESOLVED | `fibo-FBC-PAS-FPAS:Payment` | SMBAccounting/BillingCore | Perdurant |
| `account.payment.term` | UNRESOLVED | `fibo-FBC-PAS-FPAS:PaymentObligationTerms` or `ubl:PaymentTerms` | SMBAccounting/BillingCore | Abstract |
| `account.reconcile.model` | UNRESOLVED | `sh:NodeShape` / `sh:rule` | SMBAccounting | Abstract |

All three are confirmed absent: no `odoo_alignment.rs` file exists in woa-rs (the alignment infrastructure is in `lance-graph`, not yet mirrored into the woa-rs crate tree).

---

## 4. Ontology Mapping Lines (summary)

```
odoo:account.payment
  -> owl:equivalentClass fibo-FBC-PAS-FPAS:Payment
  -> OGIT family SMBAccounting/BillingCore
  -> DOLCE Perdurant (payment is a temporal event with date + state transitions)
  [UNRESOLVED - FLAG: needs alignment row in odoo_alignment.rs]

odoo:account.payment.term
  -> owl:equivalentClass fibo-FBC-PAS-FPAS:PaymentObligationTerms
     alt: ubl:PaymentTerms
  -> OGIT family SMBAccounting/BillingCore
  -> DOLCE Abstract (.term suffix -> Abstract by briefing rule)
  [UNRESOLVED - FLAG: needs alignment row]

odoo:account.payment.term.line
  -> sub-component of PaymentObligationTerms
  -> DOLCE Abstract
  [no independent row needed]

odoo:account.reconcile.model
  -> owl:equivalentClass sh:NodeShape with sh:rule action pattern
  -> OGIT family SMBAccounting
  -> DOLCE Abstract (.model suffix -> Abstract by briefing rule)
  [UNRESOLVED - FLAG: needs alignment row]

odoo:account.reconcile.model.line
  -> sub-component of sh:NodeShape
  -> DOLCE Abstract
  [no independent row needed]
```

---

## 5. K-step Map

| Rule | K-step | Description |
|---|---|---|
| P1 (payment move gen) | K3 | Double-entry journal entries for payment |
| P1 (outstanding account) | K5 | Bank / outstanding receipts-payments linkage |
| P2 (payment term compute) | K3 | Due-date splits feed invoice aging |
| P2 (Skonto) | K3+K7 | Skonto reduces tax base in excluded/mixed modes |
| P3 (discount mode) | K3+K7 | Country-specific Skonto-on-VAT behaviour |
| P4 (reconciliation status) | K3+K5 | Open-item clearance + bank match status |
| P5 (reconcile model) | K3+K5 | Bank statement to open item matching + write-off generation |

Mahnwesen (K3) dependency: escalation timing reads `ErpOpenItemAR.skonto_bis`
(computed from `zahlungsziel_tage + skonto_tage`, mirroring `discount_date = date_ref + relativedelta(days=discount_days)`)
and checks the open-item residual (equivalent of `is_reconciled`) before promoting to the next dunning stage.
The K-adjacent graph: payment terms (K3) -> due dates -> open items (K3) -> Mahnwesen (K3) -> bank matching (K5) -> final clearance (K3).

---

## 6. Porter's Checklist - Non-obvious Gotchas

1. **Last line is always the balance, regardless of type.** `_compute_terms` treats `i == len(line_ids) - 1` as balance line. Even a `'percent'`-typed last line gets `residual_amount`, not a percentage. This is the primary rounding-absorption mechanism. Porter must use index-based guard, not type-based.

2. **Exchange rate is embedded, not live.** `rate = abs(total_amount_currency / total_amount)`. A live FX lookup gives a different result. All rounding in the term computation uses this embedded rate. Do NOT substitute a currency-table rate in the port.

3. **Skonto country-mode is company-level, not per-term.** `early_pay_discount_computation` is recomputed from `company_id.country_code`. A German company always gets `'included'`. This cannot be overridden per-term in community. The flat `ErpDebtor.skonto_prozent` in woa-rs does not encode this mode; the mode must be derived from company at runtime.

4. **`amount_residual` vs `amount_residual_currency` selector** in `_compute_reconciliation_status` (L488): if pay.currency == company.currency use `amount_residual`; else use `amount_residual_currency`. Mixing produces wrong `is_reconciled` results. This matters for EUR-company paying a USD invoice.

5. **`not any(liquidity.account_id.reconcile)` in `_compute_state`**: if the liquidity account has `reconcile=False`, payment goes to 'paid' with non-zero residual. This is the direct bank account path. woa-rs Mahnwesen must not treat this path as unpaid.

6. **Write-off lines are silently dropped when withholding lines exist** (L342-L346). Community always has empty withholding lines; Enterprise may not. The port's hook must signal this conflict rather than silently discarding.

7. **`_synchronize_to_moves` skips posted moves** (L1004): changes to a payment after posting do NOT update the journal entry. Only draft moves are synchronised. Do not attempt to sync posted moves.

8. **Journal name reset on journal change** (L1056-L1059): `'name': '/'` resets the sequence number when `journal_id` changes on a draft payment. The target journal will re-sequence it. Only reachable on draft moves (see gotcha 7).

9. **`days_end_of_month_on_the` with `days_next_month=0`** (L323-L324): returns end of the month, not day 0 of next month. String-to-int with `except ValueError: days_next_month = 1` means invalid strings silently become day 1 of the following month.

10. **Reconcile model sequence is the greedy tie-breaker.** Odoo takes the first model in sequence order that matches. The NARS reasoner should receive all candidate models as evidence and return the highest truth-value match, which may differ from the first-in-sequence match. This is an intentional improvement over greedy matching for ambiguous lines.

11. **`can_be_proposed` excludes partner-mapping models** (L161: `not model.mapped_partner_id`). A model whose sole purpose is partner assignment (one line with partner, no account) is a lookup table, not a reconcile candidate. woa-rs should treat these as a separate `PartnerMapping` evidence source.

12. **Amount field on reconcile model line is a stored compute of `amount_string`** (L67-L73): `float(amount_string)` with fallback 0 on ValueError. The `amount_string` is the authoritative field; `amount` is a cached float. For regex-type lines, `amount_string` IS the regex, and `amount` = 0 always.

---

## 7. woa-rs Gap Summary

| Capability | odoo has | woa-rs has | Gap |
|---|---|---|---|
| Payment journal entry (K3) | Full (P1) | No `engine/payment.rs` | New: `engine/payment.rs` |
| Payment terms / due-date splits | Full (P2) | Flat fields (zahlungsziel_tage, skonto_tage) | New: `engine/payment_term.rs` |
| Skonto computation modes | Full (P2, P3) | `skonto_prozent` field only | Enrich: add SkontoMode enum + computation |
| Bank statement matching | Partial (P5) | `engine/bank_match.rs` (confidence 100/80/60/40) | Extend: label regex, partner mapping, write-off gen |
| NARS delegation for match | Odoo: heuristic | None | New: `engine/reconcile_match.rs` + NARS contract call |
| Ontology alignment rows | Needed (3 classes) | No odoo_alignment.rs in woa-rs | Action: add when skr_data grows |

---

Read: /home/user/odoo/addons/account/models/account_payment.py lines=1247 depth=full
Read: /home/user/odoo/addons/account/models/account_payment_term.py lines=368 depth=full
Read: /home/user/odoo/addons/account/models/account_reconcile_model.py lines=201 depth=full
