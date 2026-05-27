RICHNESS-LANE-OK

# Lane L11 — Chart of Accounts + Journals + Lock Dates + Sequences

## Sources read (file : line-range : depth)

- odoo/addons/account/models/account_account.py : L1-1642 : full
- odoo/addons/account/models/account_account_tag.py : L1-140 : full
- odoo/addons/account/models/account_journal.py : L1-1300 : full
- odoo/addons/account/models/company.py : L50-749 (lock-date section) : full
- odoo/addons/account/models/sequence_mixin.py : L1-511 : full
- odoo/addons/account/models/account_move.py : L2796-2813, L3780-3974, L5476-5481, L6570-6634 : targeted full

## Ontology rows

| odoo class | owl pivot | OGIT family (or None) | DOLCE |
|---|---|---|---|
| `account.account` | fibo:Account | 0x62 SMBAccounting | Endurant |
| `account.group` | fibo:AccountingGroup (inferred) | 0x62 SMBAccounting | Endurant |
| `account.account.tag` | schema:Thing / fibo:Annotation | None — needs Layer-2 axiom | Abstract |
| `account.journal` | fibo:Journal | 0x62 SMBAccounting | Endurant |
| `res.company` (lock-date ext) | fibo:LegalEntity | 0x62 SMBAccounting | Endurant |
| `sequence.mixin` | fibo:SequenceIdentifier (inferred) | None — abstract mixin, inherits host model | Abstract |

## Rules extracted

### R1 — account_type taxonomy (19-value enum) [AXIS-A]
- **odoo**: account_account.py:L44-70
- 19 closed enum values: asset_receivable, asset_cash, asset_current, asset_non_current, asset_prepayments, asset_fixed, liability_payable, liability_credit_card, liability_current, liability_non_current, equity, equity_unaffected, income, income_other, expense, expense_other, expense_depreciation, expense_direct_cost, off_balance. `internal_group` = prefix before first `_` (`asset|liability|equity|income|expense|off`).
- **target**: `crates/skr_data` `AccountType` enum + `internal_group()`.
- **parity**: `equity_unaffected` = "Current Year Earnings" (Gewinn/Verlust laufendes Jahr) — does NOT carry forward.

### R2 — include_initial_balance [AXIS-A]
- **odoo**: account_account.py:L638-646
- True iff internal_group ∉ {income, expense} AND type ≠ equity_unaffected. Drives Bilanz (carry-forward) vs GuV (reset each year). `fn include_initial_balance(t: AccountType) -> bool`.

### R3 — reconcile flag + constraints [AXIS-A]
- **odoo**: account_account.py:L27-31, L187-194, L664-673, L963-989
- reconcile auto-computed: income/expense/equity → false; receivable/payable → true (forced); cash/credit_card/off_balance → false. Constraint: receivable/payable MUST be reconcilable; off_balance cannot reconcile or carry taxes. Toggle true→false blocked if partial reconciliations exist. Toggle true: SQL sets `reconciled = (debit=0 AND credit=0 AND amount_currency=0)`, `amount_residual = debit-credit`.

### R4 — account code format + uniqueness [AXIS-A]
- **odoo**: account_account.py:L14-16, L310-316, L466-544, L1079-1129
- Regex `^[A-Za-z0-9.]+$`. `code` is **company-dependent** (JSONB `code_store`, keyed by root company id). `_search_new_account_code`: increment trailing digit group, fall back to `.copy{n}`. Uniqueness checked across parent/child company hierarchy (not just same company). SKR03/04 = 4-digit numeric.

### R5 — account_group hierarchy via code_prefix [AXIS-A]
- **odoo**: account_account.py:L1497-1642
- group has equal-length `code_prefix_start`/`code_prefix_end` (DB constraint). Account ∈ group iff `prefix_start <= LEFT(code, len) <= prefix_end`. Parent = longest fitting prefix (SQL `DISTINCT ON (child) ORDER BY char_length(parent.prefix_start) DESC`). Same-length groups cannot overlap. Drives BWA/SuSa groupings (K8).

### R6 — account.account.tag structure [AXIS-A]
- **odoo**: account_account_tag.py:L1-141
- Scoped by `applicability ∈ {accounts, taxes, products}` + optional `country_id`; unique `(name, applicability, country_id)`. Tax tags drive report buckets; tag name starting `-` ⇒ `balance_negate` (computed via LEFT JOIN to account_report_expression — Enterprise, derived only). Master operating/financing/investing tags undeletable.

### R7 — journal type taxonomy [AXIS-A]
- **odoo**: account_journal.py:L106-119
- 6 types: sale, purchase, cash, bank, credit, general. Drives default account domain, suspense/payment lines (liquidity), refund_sequence (sale/purchase default true), payment_sequence (bank/cash/credit default true).

### R8 — journal default account assignment [AXIS-A]
- **odoo**: account_journal.py:L926-1021
- bank/cash/credit journal without default_account_id → auto-create account; prefix from `company.bank_account_code_prefix`/`cash_account_code_prefix`; type asset_cash (bank/cash) or liability_credit_card (credit); digit count inferred from chart (fallback 6).

### R9 — journal code auto-gen [AXIS-A]
- **odoo**: account_journal.py:L883-903
- default prefixes INV/BILL/CSH/BNK/CCD/MISC; try N=1..99 until unique; max 5 chars.

### R10 — restrict_mode_hash_table (GoBD) [AXIS-A]
- **odoo**: account_journal.py:L145-146, L794-801
- Boolean. Once any entry hashed (`inalterable_hash != False`), cannot set back to false (UserError). This is the woa-rs K11 Festschreibung anchor flag; hash itself in account_move (L1, RFC-011).

### R11 — lock-date taxonomy (5 types) [AXIS-A]
- **odoo**: company.py:L59-69, L78-114
- fiscalyear_lock_date (global), tax_lock_date (entries with taxes; auto-set on tax-closing post), sale_lock_date, purchase_lock_date, hard_lock_date (irreversible). `SOFT_LOCK_DATE_FIELDS` = first four. `user_hard_lock_date` = max hard lock across parent hierarchy.

### R12 — _get_user_lock_date (soft lock + exception) [AXIS-A]
- **odoo**: company.py:L597-630
- Walks `parent_ids` (sudo). ignore_exceptions=true → max(company[field]). Else: per parent with lock set, find active `account.lock_exception` (state=active, user∈{None,current}, field < company[field]); effective = max(soft, exception[field] or min). `account.lock_exception` = community model enabling per-user temporary soft-lock override.

### R13 — _get_violated_soft_lock_date [AXIS-A]
- **odoo**: company.py:L646-663
- regular = user_lock_date(ignore_exceptions=true). If date > regular → None. Else with_exc = user_lock_date(false); if date > with_exc → None (exception covers), else Some(with_exc). Two-pass for fast short-circuit.

### R14 — _get_lock_date_violations (multi-lock sweep) [AXIS-A]
- **odoo**: company.py:L665-700
- Given accounting_date + flags(fiscalyear, sale, purchase, tax, hard) → Vec<(date, field)>. Hard: `user_hard_lock_date >= date` ⇒ violated.

### R15 — _get_violated_lock_dates (move-level, journal-aware) [AXIS-A]
- **odoo**: company.py:L713-729; account_move.py:L6609-6616
- fiscalyear=true always; sale=journal.type==sale; purchase=journal.type==purchase; tax=has_tax; hard=true. Sorted ascending. **Non-obvious**: general journal NOT subject to sale/purchase locks.

### R16 — _check_fiscal_lock_dates (write/post guard) [AXIS-A]
- **odoo**: account_move.py:L2796-2813, L3905-3958
- On write (date/state change on posted) and `_post()`. Calls violations with tax=false (tax checked separately). Skipped if `context.bypass_lock_check is BYPASS_LOCK_CHECK` (object identity sentinel, not truthiness).

### R17 — _get_user_fiscal_lock_date (copy/unlink guard) [AXIS-A]
- **odoo**: company.py:L632-644; account_move.py:L5476-5481, L3789-3791
- Single effective fiscal lock for a journal = max(fiscalyear, hard, +sale/purchase by type). Used by `_can_be_unlinked` (date > lock) and `copy_data` (bump copy date to lock+1day).

### R18 — _validate_locks (write-time guard on lock changes) [AXIS-A]
- **odoo**: company.py:L542-595
- hard_lock_date: cannot unset, cannot decrease (UserError). Setting hard/fiscal: RedirectWarning if draft entries ≤ date, or unreconciled bank statement lines ≤ max(fiscalyear, hard).

### R19 — _get_accounting_date (date bump) [AXIS-A]
- **odoo**: account_move.py:L6570-6607
- Locked period ⇒ base = last_violation + 1day. Sale + number_reset month/year ⇒ last day of month/year capped at today. Non-sale similar by reset family. Couples sequence format ↔ lock semantics.

### R20 — _deduce_sequence_number_reset (5 format families) [AXIS-A]
- **odoo**: sequence_mixin.py:L193-225
- Try regexes in order: year_range_monthly → monthly → year_range → yearly → fixed. Guard: if year_end & year both present, require year_end == (year+1) mod 10^len.

### R21 — _get_sequence_format_param [AXIS-A]
- **odoo**: sequence_mixin.py:L312-353
- Extract prefix1/year/prefix2/month/prefix3/seq/suffix/year_end + lengths; build format string `{prefix1}{year:0Nd}...{seq:0Md}{suffix}`. Edge: empty seq with prefix+suffix ⇒ treat suffix as prefix.

### R22 — _get_last_sequence (alphabetical-max gotcha) [AXIS-A]
- **odoo**: sequence_mixin.py:L269-310
- Finds prior seq by **greatest alphabetical** sequence_field, restricted to latest sequence_prefix, ORDER BY sequence_number DESC LIMIT 1. **Gotcha**: renaming INV→FACT re-uses numbers (INV > FACT). Prefix comparison is string-ordering, not numeric.

### R23 — _locked_increment (gap prevention via DB lock) [AXIS-A]
- **odoo**: sequence_mixin.py:L355-424
- Atomic increment via PG B-tree index lock (`UPDATE ... WHERE id`), savepoint retry on Unique/ExclusionViolation. In-tx cache keyed `(format_with_seq0, index_value)` avoids further savepoints. The gap-prevention mechanism (sea-orm needs raw SQL).

### R24 — _set_next_sequence / _get_next_sequence_format [AXIS-A]
- **odoo**: sequence_mixin.py:L425-473
- last = _get_last_sequence() (relaxed/starting fallback); new period ⇒ reset seq=0, set year/month from date; locked_increment; set field.

### R25 — _is_end_of_seq_chain (gap detection) [AXIS-A core + AXIS-B anomaly] [HYBRID]
- **odoo**: sequence_mixin.py:L487-511
- Group by (format, values\seq); contiguity `max-min == len-1` AND highest is last in DB. Deletion/reversal safety guard. Detecting *existing* anomalous gaps (deleted posted entries — GoBD violation) is heuristic.
- `SAVANT: name=SequenceGapAnomalyDetector family=0x62 reasoning=PostingAnomaly inference=Abduction semiring=NarsTruth style=Analytical — gaps in journal sequences may indicate deleted posted entries (GoBD breach); abduce over sequence_prefix+number distribution beyond the creation-time contiguity guard.`

### R26 — sequence_prefix / sequence_number stored fields [AXIS-A]
- **odoo**: sequence_mixin.py:L47-48, L183-191
- `_compute_split_sequence` strips fixed-regex part → prefix + trailing int; the index columns making _get_last_sequence efficient.

### R27 — _constrains_date_sequence [AXIS-A]
- **odoo**: sequence_mixin.py:L156-181
- Validate sequence-embedded year/month matches record date. Bypass via `ir.config_parameter sequence.mixin.constraint_start_date` (default 1970-01-01) for historical imports.

### R28 — _sequence_matches_date [AXIS-A]
- **odoo**: sequence_mixin.py:L138-154
- Validate year/month in sequence vs date range from reset type.

### R29 — copy date bump [AXIS-A]
- **odoo**: account_move.py:L3789-3791
- On copy/reversal: if source date ≤ user fiscal lock, set copy date = lock + 1 day.

## Enterprise gaps flagged
- `account_asset` (K12 depreciation): absent. Only base `asset_fixed`/`expense_depreciation` types present; engine fresh.
- `account_reports` (K8 BWA/SuSa engine): absent. Tag→report_expression link references Enterprise `account_report_expression`; spec tag structure only.
- `account.lock_exception`: present in community (referenced company.py:L612), not deep-read here.

## Open questions for the Opus porter
1. `account.lock_exception` field shape — per-type columns matching lock fields? (domain at L618 implies identical column names).
2. `BYPASS_LOCK_CHECK` sentinel — implement as typed context marker (object identity, not bool).
3. `_sequence_index` on account.move (likely journal_id) — confirm before building the uniqueness index.
4. 2-digit year sequences (`INV/25/00001`) — handle for historical imports.
5. Multi-company `user_hard_lock_date` walks full `parent_ids` — needs cached parent_path/root traversal.
6. `is_self_billing` journals (per-partner sequences) — interacts with L6 invoice flow.

## Depth-proof footer
```
Read: odoo/addons/account/models/account_account.py lines=1642 depth=full
Read: odoo/addons/account/models/account_account_tag.py lines=140 depth=full
Read: odoo/addons/account/models/account_journal.py lines=1300 depth=full
Read: odoo/addons/account/models/company.py lines=1148 depth=full (lock-date section L50-749)
Read: odoo/addons/account/models/sequence_mixin.py lines=511 depth=full
Read: odoo/addons/account/models/account_move.py lines=7328 depth=targeted (L2796-2813, L3780-3974, L5476-5481, L6570-6634)
```
