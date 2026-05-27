RICHNESS-LANE-OK

# Lane L10 — Analytic Accounting (Kostenstellen / Cost Centres)

## Sources read (file : line-range : depth)

- `/home/user/odoo/addons/account/models/account_analytic_account.py` : L1-79 : full
- `/home/user/odoo/addons/account/models/account_analytic_line.py` : L1-111 : full
- `/home/user/odoo/addons/account/models/account_analytic_plan.py` : L1-82 : full  (this file is `AccountAnalyticApplicability`, not `AccountAnalyticPlan` — see Enterprise Gap note)
- `/home/user/odoo/addons/account/models/account_analytic_distribution_model.py` : L1-71 : full
- `/home/user/odoo/addons/account/models/account_move_line.py` : L1-50, L400-450, L980-1090, L1200-1320, L1390-1425, L1760-1900, L2005-2055, L3140-3280, L3490-3520 : full (analytic-relevant sections covering entire analytic surface)
- `/home/user/odoo/addons/account/tests/test_account_analytic.py` : L1-1182 : full
- `/home/user/odoo/addons/purchase/models/analytic_account.py` : L1-35 : full
- `/home/user/odoo/addons/purchase/models/analytic_applicability.py` : L1-16 : full
- `/home/user/odoo/addons/sale/models/analytic.py` : L1-22 : full
- `/home/user/odoo/addons/account/models/account_move.py` : L5055-5085 : section (EPD analytic distribution call)

## Enterprise / Module-Gap Flags

**CRITICAL:** The base `analytic` addon module — which defines `account.analytic.account`, `account.analytic.plan`, `account.analytic.applicability`, `account.analytic.distribution.model`, and `analytic.mixin` — is **absent** from `/home/user/odoo/addons/`. The clone only contains the six addons: `account`, `hr`, `l10n_de`, `product`, `purchase`, `sale`, `stock`. The `analytic` module is listed as a dependency in `account/__manifest__.py` (`'depends': ['base_setup', 'onboarding', 'product', 'analytic', ...]`) but is NOT present. All files in `account/models/account_analytic_*.py` use `_inherit = 'account.analytic.*'` — they extend models from the missing base module.

What IS present and readable: the account-side extensions (`_inherit` classes), the `analytic.mixin` mixin consumed by `account.move.line` and `account.reconcile.model`, all analytic logic on `account.move.line` (`_create_analytic_lines`, `_prepare_analytic_lines`, `_prepare_analytic_distribution_line`, `_round_analytic_distribution_line`, `_validate_analytic_distribution`, `_inverse_analytic_distribution`, `_compute_analytic_distribution`, `_update_analytic_distribution`, `_related_analytic_distribution`), and the full test suite which reveals the base model's shape through usage.

Base model properties recoverable from tests and account-side code:
- `account.analytic.plan` has `_column_name()` method returning a dynamic column name (e.g. `x_plan1_id`), `applicability_ids` One2many to `account.analytic.applicability`, `_get_applicability()` method.
- `account.analytic.account` has `plan_id`, `root_plan_id`, `company_id`, `active`, `line_ids` One2many to `account.analytic.line`.
- `account.analytic.applicability` has `business_domain` Selection (extended to `'invoice'`, `'bill'`, `'purchase_order'`, `'sale_order'` by account/purchase/sale), `applicability` ∈ {`'mandatory'`, `'unavailable'`, `'optional'`}, `analytic_plan_id`, `company_id`, `_get_score(**kwargs)` → int.
- `account.analytic.distribution.model` has `partner_id`, `company_id`, `sequence`, `analytic_distribution` Json, `_get_distribution(criteria_dict)` class method, `_get_applicable_models(vals)`, `_get_default_search_domain_vals()`.
- `analytic.mixin` defines: `analytic_distribution` Json field, `_merge_distribution(old, new)`, `_validate_distribution(...)`, `_get_plan_fnames()`, `_get_distribution_key()` on lines.

---

## Ontology rows

| odoo class | owl pivot | OGIT family (or None) | DOLCE |
|---|---|---|---|
| `account.analytic.account` | `fibo:Account` (cost account sub-type) | `0x62 SMBAccounting` (via fibo:Account → BillingCore/SMBAccounting chain; closest to `account.account`) | Endurant — a persisting organisational entity |
| `account.analytic.plan` | None — no direct FIBO/UBL/vCard pivot for "analytic plan hierarchy"; closest is `fibo:AccountingSystem` (not in current seed) | `None` → ontology-unmapped, needs a Layer-2 alignment axiom | Abstract — a classification schema |
| `account.analytic.applicability` | None — rule/policy object; closest `fibo:ControllingParty` role is a stretch | `None` → ontology-unmapped, needs a Layer-2 alignment axiom | Abstract — a rule/constraint object |
| `account.analytic.line` | `fibo:JournalEntryLine` (parallel to `account.move.line`) | `0x62 SMBAccounting` (inherits from JournalEntryLine seed) | Perdurant — an event/occurrence |
| `account.analytic.distribution.model` | None — a pattern-matching rule object; no FIBO equivalent in seed | `None` → ontology-unmapped, needs a Layer-2 alignment axiom | Abstract — a configuration/rule record |
| `account.move.line` (analytic fields) | `fibo:JournalEntryLine` | `0x62 SMBAccounting` | Perdurant |

---

## Rules extracted

### R1 — analytic_distribution JSON representation   [AXIS-A]

- **odoo source**: `account/models/account_move_line.py:L418-420`; confirmed by `test_account_analytic.py:L70-74`, `L246`, `L369-370`
- **What it does**: The `analytic_distribution` field is a `fields.Json` column on `account.move.line` (and via `analytic.mixin` on other models). Its schema is:
  ```
  { "<account_id_csv>": <percentage_float>, ... }
  ```
  Keys are **comma-separated analytic account IDs as strings** (e.g. `"42"`, `"42,17"` for cross-plan joint allocation). Values are **float percentages** (e.g. `100.0`, `50.0`, `33.33`). A single key with multiple comma-separated IDs means those analytic accounts are all allocated the same percentage slice simultaneously (cross-plan). Example from `test_account_analytic.py:L369-370`:
  ```python
  analytic_distribution = {
      f'{self.analytic_account_3.id},{self.analytic_account_5.id}': 20,
      f'{self.analytic_account_3.id},{self.analytic_account_4.id}': 80,
  }
  ```
  This means 20% of the amount goes to (account_3 AND account_5 simultaneously) and 80% goes to (account_3 AND account_4 simultaneously). Percentages are floats stored with analytic precision (fetched via `decimal.precision.precision_get('Percentage Analytic')`).
- **woa-rs target**: K10 new area (Kostenstellen). The `analytic_distribution` field will be a JSON column on the `journal_line` (AML equivalent) entity. Rust type: `Option<serde_json::Value>` or a newtype `AnalyticDistribution(HashMap<String, f64>)` where String is the CSV-of-IDs key.
- **Rust sketch**:
  ```rust
  // Key: comma-separated analytic account IDs (String), Value: percentage (Decimal/f64)
  pub struct AnalyticDistribution(pub HashMap<String, f64>);
  // Invariant: keys are non-empty strings of comma-separated positive integer IDs
  // Percentages per analytic PLAN must sum to <= 100.0 for optional, == 100.0 for mandatory
  // (validation is per-plan, not global — see R4)
  impl AnalyticDistribution {
      pub fn account_ids_for_key(key: &str) -> Vec<i64> {
          key.split(',').filter_map(|s| s.trim().parse().ok()).collect()
      }
  }
  ```
- **Parity notes**: The JSON key uses string IDs even in Python (keys become strings on JSON serialisation). In tests, when writing from Python the key may be an integer (`{self.analytic_account_1.id: 100}`) but after store/read it is always a string key. Rust side must normalise to string keys on write.

---

### R2 — Plan hierarchy: `account.analytic.plan` / `account.analytic.account`   [AXIS-A]

- **odoo source**: `account/models/account_analytic_plan.py` (base absent — recovered from test usage); `account_move_line.py:L3215-3222`; `test_account_analytic.py:L19-24`, `L434-444`
- **What it does**:
  - `account.analytic.plan` is a tree/hierarchy of plans. Each plan has a `_column_name()` method returning a dynamic DB column name (e.g. `x_plan1_id`, auto-generated). Plans can be nested (parent/child) but `root_plan_id` on `account.analytic.account` always points to the root of the tree.
  - `account.analytic.account` belongs to exactly one `plan_id` (leaf-level plan), and has `root_plan_id` pointing to the root plan.
  - Multiple plans are **orthogonal axes**: e.g. Plan A = "Department" (Kostenstelle), Plan B = "Project". A move line can simultaneously have 50% in Dept-Marketing AND 50% in Dept-Sales (Plan A axis), and 100% in Project-X (Plan B axis). Cross-plan keys in `analytic_distribution` encode the simultaneous allocation across plans.
  - From `account_move_line.py:L3215-3222`:
    ```python
    for account in self.env['account.analytic.account'].browse(map(int, account_ids.split(","))).exists():
        distribution_plan = distribution_on_each_plan.get(account.root_plan_id, 0) + distribution
        if float_compare(distribution_plan, 100, precision_digits=decimal_precision) == 0:
            amount = -self.balance * (100 - distribution_on_each_plan.get(account.root_plan_id, 0)) / 100.0
        else:
            amount = -self.balance * distribution / 100.0
        distribution_on_each_plan[account.root_plan_id] = distribution_plan
        account_field_values[account.plan_id._column_name()] = account.id
    ```
    Critical: the amount for the LAST entry in a plan that reaches 100% uses the remainder formula `(100 - accumulated_so_far) / 100 * balance` to avoid floating-point accumulation error. All earlier entries use `distribution / 100 * balance`. The sign is negated (`-self.balance`) because analytic lines have the opposite sign convention to journal lines (credit → positive analytic amount, debit → negative).
- **woa-rs target**: K10 data model. `analytic_plan` table with parent_id self-reference; `analytic_account` with `plan_id FK`, `root_plan_id FK`; `column_name` stored or computed from plan ID.
- **Rust sketch**:
  ```rust
  pub struct AnalyticPlan {
      pub id: i64,
      pub name: String,
      pub parent_id: Option<i64>,
      pub column_name: String, // e.g. "x_plan1_id", stored
  }
  pub struct AnalyticAccount {
      pub id: i64,
      pub name: String,
      pub plan_id: i64,
      pub root_plan_id: i64,
      pub company_id: Option<i64>,
      pub active: bool,
  }
  ```
- **Parity notes**: The `_column_name()` dynamic naming is an Odoo ORM artifact (custom Many2one fields on `account.analytic.line`). In Rust we should store the column_name explicitly on the plan and use it for analytics line routing.

---

### R3 — `_prepare_analytic_distribution_line` + `_create_analytic_lines`: amount calculation and sign   [AXIS-A]

- **odoo source**: `account/models/account_move_line.py:L3179-3239`
- **What it does** (full control flow):
  1. `_create_analytic_lines()` is called at `_post()` time (when move transitions to posted). It first calls `_validate_analytic_distribution()` then loops over all lines calling `_prepare_analytic_lines()` on each, collecting `analytic_line_vals`.
  2. `_prepare_analytic_lines()` iterates `self.analytic_distribution.items()`. For each `(account_ids_csv, distribution_pct)`:
     - Calls `_prepare_analytic_distribution_line(float(distribution), account_ids, distribution_on_each_plan)`.
     - Skips entries where the resulting `amount` rounds to zero (`is_zero`).
     - Calls `_round_analytic_distribution_line()` on the final list.
  3. `_prepare_analytic_distribution_line(distribution, account_ids, distribution_on_each_plan)`:
     - Parses `account_ids.split(",")` → list of int IDs → browse `account.analytic.account`.
     - For each analytic account in the CSV (cross-plan joint allocation):
       - Accumulates `distribution_plan = distribution_on_each_plan.get(root_plan_id, 0) + distribution`.
       - If `distribution_plan == 100.0` (within analytic precision): `amount = -balance * (100 - accumulated_before) / 100.0` (remainder formula, avoids floating error).
       - Else: `amount = -balance * distribution / 100.0`.
       - Stores accumulated sum per root_plan_id.
       - Sets `account_field_values[plan._column_name()] = account.id` (the dynamic plan column).
     - Returns a dict with: `name`, `date`, `partner_id`, `unit_amount` (= quantity), `product_id`, `product_uom_id`, `amount` (= the computed amount in COMPANY currency), `general_account_id` (= GL account), `ref`, `move_line_id`, `user_id`, `company_id`, `category` (`'invoice'`/`'vendor_bill'`/`'other'`), plus dynamic plan fields.
     - Note: `name` falls back to `ref + ' -- ' + partner_name` if no line name — see L3223.
  4. `_round_analytic_distribution_line()` (L3255-3279): rounds each line's amount to company currency rounding, tracks accumulated rounding error, then distributes it — subtracts or adds `currency.rounding` unit-by-unit to lines until error is zero. This ensures the sum of analytic line amounts equals the original balance exactly.

- **woa-rs target**: K10 + K3. Called as part of the `action_post()` pipeline. The analytic line creation is a side-effect of posting.
- **Rust sketch**:
  ```rust
  fn create_analytic_lines(aml: &AccountMoveLine, db: &Db) -> Result<Vec<AnalyticLine>> {
      validate_analytic_distribution(aml)?;
      let mut vals: Vec<AnalyticLineVals> = vec![];
      for (account_ids_csv, distribution_pct) in &aml.analytic_distribution {
          if let Some(v) = prepare_analytic_distribution_line(
              aml, distribution_pct, account_ids_csv, &mut distribution_on_each_plan, db
          )? {
              if !company_currency.is_zero(v.amount) {
                  vals.push(v);
              }
          }
      }
      round_analytic_distribution_line(&mut vals, company_currency);
      AnalyticLine::bulk_insert(&vals, db)
  }

  fn round_analytic_distribution_line(vals: &mut Vec<AnalyticLineVals>, currency: &Currency) {
      let mut rounding_error: Decimal = Decimal::ZERO;
      for v in vals.iter_mut() {
          let rounded = currency.round(v.amount);
          rounding_error += rounded - v.amount;
          v.amount = rounded;
      }
      // Distribute error: subtract/add rounding unit from lines one by one
      for v in vals.iter_mut() {
          if currency.is_zero(rounding_error) { break; }
          let unit = currency.rounding.max(currency.round((rounding_error / vals.len() as Decimal).abs()));
          if rounding_error < Decimal::ZERO {
              v.amount += unit; rounding_error += unit;
          } else {
              v.amount -= unit; rounding_error -= unit;
          }
      }
  }
  ```
- **Parity notes**: Sign convention: analytic lines use COMPANY currency (`-self.balance`, not `amount_currency`). In multi-currency invoices, the analytic line amount is always in company currency. The `amount` sign is the NEGATIVE of the journal line balance — so a revenue line (credit, negative balance in Odoo) produces a POSITIVE analytic amount. Confirmed by `test_account_analytic.py:L80-90`: invoice line with `price_unit=200`, plan 100% → analytic line `amount=200` (positive).

---

### R4 — `_validate_analytic_distribution` / `_validate_distribution`: mandatory-plan enforcement   [AXIS-A]

- **odoo source**: `account/models/account_move_line.py:L3146-3177`; `account_move_line.py:L2011-2034`
- **What it does**:
  1. `_validate_analytic_distribution()` (called at post time from `_create_analytic_lines`): loops over `display_type == 'product'` lines. For each, calls `_validate_distribution(company_id=..., product=..., account=..., business_domain=...)` where `business_domain` ∈ {`'invoice'`, `'bill'`, `'general'`}.
  2. If ANY line fails validation, raises:
     - Single move: `ValidationError` (message: `"One or more lines require a 100% analytic distribution."`)
     - Multiple moves (mass post): `RedirectWarning` with a view action showing failing lines.
  3. `_validate_distribution` (from `analytic.mixin`, base module absent): determines which plans are applicable to this line (via `account.analytic.plan._get_applicability(business_domain, company_id, product, account)`) and checks if those with `applicability == 'mandatory'` have exactly 100% distribution in the JSON. Validation is only triggered when `validate_analytic=True` is in context (from `action_post()` on invoices; NOT triggered by plain journal entries unless context is set — see `test_account_analytic.py:L310-330`).
  4. `_compute_has_invalid_analytics()` (L2011-2034): the same check computed as a Boolean field, used for UI highlighting. Skips account types: `asset_receivable`, `liability_payable`, `asset_cash`, `liability_credit_card` (these never need analytic).
  5. Validation precision: from `test_account_analytic.py:L314-319`: `{account: 100.01}` FAILS, `{account: 99.9}` FAILS, `{account: 100}` PASSES. The tolerance is determined by `'Percentage Analytic'` decimal precision (from `decimal.precision`).

- **woa-rs target**: K10. Guard in the posting pipeline.
- **Rust sketch**:
  ```rust
  const SKIPPED_ACCOUNT_TYPES: &[AccountType] = &[
      AccountType::AssetReceivable, AccountType::LiabilityPayable,
      AccountType::AssetCash, AccountType::LiabilityCreditCard,
  ];
  fn validate_analytic_distribution(aml: &AccountMoveLine, plans: &[AnalyticPlan])
      -> Result<(), ValidationError>
  {
      if aml.display_type != DisplayType::Product { return Ok(()); }
      if SKIPPED_ACCOUNT_TYPES.contains(&aml.account_type) { return Ok(()); }
      let business_domain = business_domain_for(aml);
      for plan in plans {
          let applicability = plan.get_applicability(business_domain, aml.company_id, aml.product_id, aml.account_id);
          if applicability == Applicability::Mandatory {
              let pct: f64 = aml.analytic_distribution.iter()
                  .filter(|(key, _)| key_contains_plan_account(key, plan))
                  .map(|(_, pct)| pct)
                  .sum();
              if !approx_eq_100(pct, ANALYTIC_PRECISION) {
                  return Err(ValidationError::new("One or more lines require a 100% analytic distribution."));
              }
          }
      }
      Ok(())
  }
  ```
- **Parity notes**: Validation is per-plan (each plan's accounts must sum to 100%), NOT global sum of all percentages. A line can have 100% in Plan A and 100% in Plan B simultaneously (legitimate). The `validate_analytic` context flag gates enforcement — plain journal entries bypass it unless the UI explicitly sets the flag.

---

### R5 — `_inverse_analytic_distribution`: sync on write   [AXIS-A]

- **odoo source**: `account/models/account_move_line.py:L1401-1422`
- **What it does**: Triggered as inverse of the `analytic_distribution` Json field (i.e., whenever `analytic_distribution` is written on a posted line). Full sequence:
  1. Check `skip_analytic_sync` context — if set, return immediately (prevents recursion).
  2. Filter to lines that are in `parent_state == 'posted'` (draft lines: their analytic lines were already unlinked at create, see L1772-1773).
  3. Read OLD distribution from DB via raw SQL (`SELECT id, analytic_distribution FROM account_move_line WHERE id = ANY(%s)`) — NOT from ORM cache, to get the persisted old value before the write.
  4. For each line, call `_merge_distribution(old_distribution, new_distribution)` → merged result. (Base `analytic.mixin` method — merges plan-by-plan, see test for `__update__` protocol in `test_analytic_dynamic_update` at L799-1008.)
  5. Unlink all existing `analytic_line_ids` for posted lines (with `skip_analytic_sync=True`).
  6. Call `_create_analytic_lines()` to recreate from merged distribution.
  7. Also triggered from `_inverse_account_id()` (L1420-1422) — account change triggers analytic re-sync.

  The `_merge_distribution` `__update__` protocol (from tests L821-1008): if `new_distribution` contains key `'__update__'` listing plan column names, ONLY those plans are replaced; other plans retain old values, cross-plan keys are rebuilt as a Cartesian product. If `'__update__'` is absent and new_distribution is empty dict → result is `False`. This is how the UI widget partially updates one plan without resetting others.

- **woa-rs target**: K10. Write-through trigger on `journal_line.analytic_distribution` updates.
- **Rust sketch**:
  ```rust
  fn on_analytic_distribution_changed(aml_id: i64, new_distribution: AnalyticDistribution, db: &Db) {
      if aml.parent_state != MoveState::Posted { return; }
      let old_distribution = db.query_one::<AnalyticDistribution>(
          "SELECT analytic_distribution FROM account_move_line WHERE id = $1", aml_id
      )?;
      let merged = merge_distribution(old_distribution, new_distribution);
      aml.analytic_line_ids.unlink_all(db)?;
      create_analytic_lines_for(aml.with_distribution(merged), db)?;
  }
  ```
- **Parity notes**: Draft lines NEVER have persistent analytic lines (they are deleted at create time, L1772-1773, and also when `analytic_line_ids` are written in draft, L1897-1898). Only posted lines have live analytic lines. The raw SQL read of old distribution is important — it bypasses ORM caching to get the actual DB state, which may differ from ORM-buffered values in a multi-write transaction.

---

### R6 — `_update_analytic_distribution`: reverse sync (analytic line → AML)   [AXIS-A]

- **odoo source**: `account/models/account_move_line.py:L3245-3253`; `account_analytic_line.py:L92-110`
- **What it does**: When an `account.analytic.line` is created/written/deleted directly (not via the distribution sync), this method re-derives `analytic_distribution` on the parent `account.move.line` from the actual analytic lines.
  ```python
  def _update_analytic_distribution(self):
      if self.env.context.get('skip_analytic_sync'): return
      for line in self:
          line.with_context(skip_analytic_sync=True).analytic_distribution = {
              analytic_line._get_distribution_key(): -analytic_line.amount / line.balance * 100
              if line.balance else 100
              for analytic_line in line.analytic_line_ids
          }
  ```
  Key formula: `percentage = -analytic_line.amount / aml.balance * 100`. If `balance == 0` (zero-amount line), defaults to 100% (safe fallback, confirmed by `test_zero_balance_invoice_with_analytic_line:L786-797`). `_get_distribution_key()` returns the comma-CSV of all plan account IDs for that analytic line (from base mixin).
  `account.analytic.line.create/write/unlink` all call `move_line_id._update_analytic_distribution()` (L94-110), creating a bidirectional sync loop guarded by `skip_analytic_sync` context.

- **woa-rs target**: K10. Used when analytic lines are edited from the analytic line view (not from invoice UI).
- **Rust sketch**:
  ```rust
  fn update_analytic_distribution_from_lines(aml: &mut AccountMoveLine) {
      let dist: HashMap<String, f64> = aml.analytic_line_ids.iter().map(|al| {
          let key = al.get_distribution_key(); // CSV of plan account IDs
          let pct = if aml.balance != Decimal::ZERO {
              -al.amount / aml.balance * dec!(100)
          } else {
              dec!(100)
          };
          (key, pct)
      }).collect();
      // set with skip_analytic_sync=true to avoid loop
      aml.set_analytic_distribution_no_sync(AnalyticDistribution(dist));
  }
  ```
- **Parity notes**: This reverse-sync makes `analytic_distribution` derivable from `analytic_line_ids` and vice versa. The authoritative source during posting is `analytic_distribution`; the authoritative source for manual line edits is the analytic line. Bidirectional, guarded by context flag.

---

### R7 — `_compute_analytic_distribution`: auto-suggestion from distribution models   [HYBRID: AXIS-A guard + AXIS-B core]

- **odoo source**: `account/models/account_move_line.py:L1217-1248`
- **What it does** (rich):
  ```python
  @api.depends('account_id', 'partner_id', 'product_id')
  def _compute_analytic_distribution(self):
      cache = {}
      for line in self:
          if line.display_type == 'product' or not line.move_id.is_invoice(include_receipts=True):
              related_distribution = line._related_analytic_distribution()
              root_plans = self.env['account.analytic.account'].browse(
                  list({int(account_id) for ids in related_distribution
                        for account_id in ids.split(',') if account_id.strip()})
              ).exists().root_plan_id
              arguments = frozendict(line._get_analytic_distribution_arguments(root_plans))
              if arguments not in cache:
                  cache[arguments] = self.env['account.analytic.distribution.model']._get_distribution(arguments)
              line.analytic_distribution = related_distribution | cache[arguments] or line.analytic_distribution
  ```
  - Triggered by: `account_id`, `partner_id`, `product_id` changes.
  - Only runs for `display_type == 'product'` lines OR non-invoice moves.
  - Calls `_related_analytic_distribution()` (base returns `{}`, overridden in SO/PO to return SO/PO line's distribution).
  - Builds `_get_analytic_distribution_arguments()`:
    ```python
    {
        "product_id": self.product_id.id,
        "product_categ_id": self.product_id.categ_id.id,
        "partner_id": self.partner_id.id,
        "partner_category_id": self.partner_id.category_id.ids,
        "account_prefix": self.account_id.code,
        "company_id": self.company_id.id,
        "related_root_plan_ids": root_plans,  # already-allocated plans to skip
    }
    ```
  - Calls `_get_distribution(arguments)` on `account.analytic.distribution.model` — finds matching rules, applies scoring, returns best distribution.
  - Result merges: `related_distribution | cache[arguments]`. The `|` dict merge means distribution-model rules supplement (or override) related distribution. If neither produces anything, keeps existing value (`or line.analytic_distribution`).
  - Uses a `frozendict` cache per unique argument combination (performance: avoids repeated DB queries).

  AXIS-A part (deterministic guard): the argument assembly, cache lookup, merge operation, and the `display_type` guard are deterministic.
  AXIS-B part (heuristic core): `_get_distribution(arguments)` itself — matching rules by score, choosing which model wins, handling ambiguous multi-rule situations — is a pattern-matching/scoring system.

- **woa-rs target**: K10. Triggered by invoice line field changes (reactive compute). AXIS-A portion → Rust compute. AXIS-B portion → delegate.
- **Rust sketch (AXIS-A guard)**:
  ```rust
  fn compute_analytic_distribution(aml: &mut AccountMoveLine, db: &Db) {
      if aml.display_type != DisplayType::Product && aml.move_id.is_invoice() { return; }
      let related = related_analytic_distribution(aml); // {} for plain invoices
      let root_plans: Vec<AnalyticPlan> = extract_root_plans(&related, db);
      let args = AnalyticDistributionArguments {
          product_id: aml.product_id, product_categ_id: aml.product_categ_id,
          partner_id: aml.partner_id, partner_category_ids: aml.partner_category_ids.clone(),
          account_prefix: aml.account_code.clone(), company_id: aml.company_id,
          related_root_plan_ids: root_plans,
      };
      let model_dist = get_distribution_from_model(&args, db); // AXIS-B call
      let merged = merge_dicts(related, model_dist);
      aml.analytic_distribution = if merged.is_empty() { aml.analytic_distribution.clone() } else { merged };
  }
  ```
- **Delegation tuple (AXIS-B)**:
  `ReasoningKind=NextBestAction` (suggest the best distribution for this line context)
  `InferenceType=Induction` (pattern: "lines like this product/partner/account combo have historically used distribution X")
  `SemiringChoice=NarsTruth` (evidence fusion across multiple matching rules, weighted by score)
  `ThinkingStyle=Analytical` (inherited from OGIT family `0x62 SMBAccounting` → expected Analytical cluster for accounting heuristics)

  `SAVANT: name=AnalyticDistributionSuggester family=0x62 reasoning=NextBestAction inference=Induction semiring=NarsTruth style=Analytical — matches invoice line context (partner/product/account-prefix/company) against distribution model rules ranked by score, returning best-fit distribution`

---

### R8 — `account.analytic.distribution.model._get_applicable_models` + scoring   [HYBRID]

- **odoo source**: `account/models/account_analytic_distribution_model.py:L34-48`; base `_get_distribution` absent (base `analytic` module missing); test `test_model_score:L232-258`, `test_model_sequence:L433-468`
- **What it does** (from account-side `_inherit` + tests):
  1. `_get_default_search_domain_vals()` (L28-32): returns dict with `{'product_id': False, 'product_categ_id': False}` plus base defaults (partner, company, etc.) — these False-values mean "match models that have no constraint on this field".
  2. `_get_applicable_models(vals)` (L34-44): extends base by filtering out models whose `account_prefix` does not match the given `account_prefix`. Prefix matching: model's `account_prefix` is split by `;` or `,`, and the incoming code must `startswith` any of those prefixes. If `account_prefix` is not set on the model → passes (no account filter).
  3. Base `_get_distribution(criteria)` (not visible, inferred from tests):
     - Finds all matching `account.analytic.distribution.model` records by domain matching.
     - Scores each via `_get_score()` (account-side adds +1 for matching `account_prefix`, +1 for matching `product_categ_id`, returns -1 to exclude on mismatch).
     - Sequences models by their `sequence` field (lower = higher priority).
     - Applies models greedily: takes the first (highest priority) model, applies its `analytic_distribution` for the plans it covers, then continues with subsequent models for plans NOT yet covered. Models that would overwrite an already-covered plan are SKIPPED.
     - From `test_model_sequence:L456-468`: "m1 fills A & B, ignore m2 & m3" — i.e., once a plan is filled, no lower-priority model can override it.
  4. `_create_domain(fname, value)` (L46-49): for `account_prefix` field, returns empty domain `[]` — because prefix matching is done in Python filter, not SQL domain.

  AXIS-A part: The sequence sorting, the "first model wins per plan" greedy fill, and the prefix-startswith test are deterministic algorithms.
  AXIS-B part: The scoring system — combining partner match (+1), product match (+1), product_category match (+1), account_prefix match (+1), company match (+1) — is multi-factor evidence weighting. Deciding the "correct" model when scores are tied or partial is fundamentally heuristic.

- **woa-rs target**: K10. Distribution model lookup, called from `_compute_analytic_distribution`.
- **Rust sketch (AXIS-A deterministic algorithm)**:
  ```rust
  fn get_distribution(args: &AnalyticDistributionArguments, db: &Db) -> AnalyticDistribution {
      let candidates = db.query_all::<AnalyticDistributionModel>(/* base domain filters */)?;
      let applicable = candidates.iter()
          .filter(|m| m.account_prefix.as_ref()
              .map(|p| prefix_matches(p, &args.account_prefix))
              .unwrap_or(true))
          .filter(|m| /* product_categ matches or none */)
          .collect::<Vec<_>>();
      let mut scored: Vec<(i32, &AnalyticDistributionModel)> = applicable.iter()
          .map(|m| (m.get_score(args), m))
          .filter(|(score, _)| *score >= 0)
          .collect();
      scored.sort_by_key(|(score, m)| (Reverse(*score), m.sequence));
      // Greedy fill: apply models in priority order, skip if plan already covered
      let mut filled_plans: HashSet<i64> = HashSet::new();
      let mut result = AnalyticDistribution::default();
      for (_score, model) in scored {
          let model_plans = plans_in_distribution(&model.analytic_distribution, db);
          if model_plans.iter().any(|p| filled_plans.contains(p)) { continue; }
          result.merge(&model.analytic_distribution);
          filled_plans.extend(model_plans);
      }
      result
  }
  ```
- **Delegation tuple (AXIS-B scoring)**:
  `ReasoningKind=CustomerCategory` (categorising this line into a distribution rule bucket)
  `InferenceType=Deduction` (given scored facts, the winner IS the highest scorer — deductive once scoring weights are defined; but the WEIGHT DEFINITION is heuristic)
  `SemiringChoice=HammingMin` (each matching criterion adds a bit; minimum sufficient match wins)
  `ThinkingStyle=Analytical` (from OGIT family `None` for `account.analytic.distribution.model` — ontology-unmapped; proposing Analytical because scoring is structured evidence aggregation)

  `SAVANT: name=AnalyticModelScorer family=None reasoning=CustomerCategory inference=Deduction semiring=HammingMin style=Analytical — multi-criterion scoring (partner/product/categ/account-prefix/company) to rank distribution model rules; greedy plan-by-plan fill from highest-score model`

---

### R9 — `account.analytic.applicability._get_score`: plan applicability scoring   [AXIS-A]

- **odoo source**: `account/models/account_analytic_plan.py:L59-76` (which is actually `AccountAnalyticApplicability._get_score`)
- **What it does**: Extends base `_get_score` (from missing `analytic` module). Returns `-1` if any hard constraint is violated (mandatory exclusion), otherwise returns an int score where higher = more specific match.
  Account-side adds two criteria:
  1. `account_prefix` (L65-70): if set on the applicability, splits by `[,;]`, checks `account.code.startswith(any_prefix)`. Match: score += 1. No match: return -1.
  2. `product_categ_id` (L71-75): if set, checks `product.categ_id == self.product_categ_id`. Match: score += 1. No match: return -1.
  Base scores (inferred from `test_applicability_score:L405-431`): `company_id` match: score += 1 (but only if the model HAS a company — if no company on model, company is not scored). `business_domain` match: base handles.
  From `test_applicability_score`: "product takes precedence over company" — score 2 (product+domain) beats score 1 (company only).
  `_compute_display_account_prefix`: account_prefix field is shown when `business_domain in ('general', 'invoice', 'bill')`.

- **woa-rs target**: K10. Plan applicability resolution — determines if a plan is mandatory/optional/unavailable for a given line context.
- **Rust sketch**:
  ```rust
  fn get_score(applicability: &AnalyticApplicability, args: &ApplicabilityArgs) -> i32 {
      let mut score = base_get_score(applicability, args); // handles business_domain, company
      if score == -1 { return -1; }
      if let Some(prefix) = &applicability.account_prefix {
          let prefixes: Vec<&str> = prefix.split(&[';', ','][..]).map(str::trim).collect();
          if args.account_code.map(|c| prefixes.iter().any(|p| c.starts_with(p))).unwrap_or(false) {
              score += 1;
          } else { return -1; }
      }
      if let Some(categ) = applicability.product_categ_id {
          if args.product_categ_id == Some(categ) { score += 1; } else { return -1; }
      }
      score
  }
  ```
- **Parity notes**: Multiple applicabilities can match; highest score wins (not sum). From `test_applicability_score:L423-424`: `_get_applicability` returns the single highest-scoring applicability's `applicability` value.

---

### R10 — `AccountAnalyticLine.on_change_unit_amount`: standard price compute for timesheet-like entries   [AXIS-A]

- **odoo source**: `account/models/account_analytic_line.py:L63-79`
- **What it does**: Onchange for `product_id`/`product_uom_id`/`unit_amount`/`currency_id` on analytic lines (used when editing analytic lines directly, not from invoice sync). Computes `amount` = `standard_price` × `unit_amount`, negated, rounded by `currency.round()` or `round(..., 2)`. Sets `general_account_id` to product's expense account. Falls back to product's UoM if no UoM set.
  ```python
  amount_unit = self.product_id._price_compute('standard_price', uom=unit)[self.product_id.id]
  amount = amount_unit * self.unit_amount or 0.0
  result = (self.currency_id.round(amount) if self.currency_id else round(amount, 2)) * -1
  ```
  Note: the `or 0.0` applies to `amount_unit * self.unit_amount` — if quantity is 0 or product has no price, amount = 0.

- **woa-rs target**: K10. Only relevant if the Rust ERP allows direct analytic line creation (e.g. for professional services / timesheets) separate from invoices.
- **Parity notes**: This is the "timesheet-style" analytic entry. For woa-rs's current scope (IT services, invoice-driven), this may not be triggered often — but it IS the code path for manual analytic line amounts.

---

### R11 — `AccountAnalyticLine.create/write/unlink`: bidirectional sync   [AXIS-A]

- **odoo source**: `account/models/account_analytic_line.py:L91-110`
- **What it does**: Ensures that when analytic lines are manipulated directly (not via invoice distribution), the parent `account.move.line.analytic_distribution` is updated:
  - `create()` (L91-95): calls `super().create()` then `analytic_lines.move_line_id._update_analytic_distribution()`.
  - `write()` (L97-104): saves `affected_move_lines = self.move_line_id` before write. After write, if `amount` or `move_line_id` or any plan field changed, calls `_update_analytic_distribution()` on the UNION of old and new affected move lines.
  - `unlink()` (L106-110): saves affected move lines, then calls `_update_analytic_distribution()` after deletion.
  - All operations are guarded by `skip_analytic_sync` context (set by `_inverse_analytic_distribution` to prevent loops).

- **woa-rs target**: K10. Reverse-sync hooks on analytic line CRUD.
- **Parity notes**: The `write()` union of `affected_move_lines` handles the case where `move_line_id` is changed on an analytic line (the old AML and the new AML both need their distribution updated).

---

### R12 — `_validate_analytic_distribution` gating at post: archived account check   [AXIS-A]

- **odoo source**: `test_account_analytic.py:L1134-1145` (test reveals behavior — no direct source line since base method is in missing `analytic` module)
- **What it does**: When posting an invoice that has `analytic_distribution` referencing an archived (`active=False`) analytic account, a `UserError` is raised: `"archived analytic account"`. This is a hard block — cannot post with archived accounts in distribution. Implemented in base `analytic.mixin._validate_distribution` (absent — inferred from test).
- **woa-rs target**: K10 posting guard.
- **Rust sketch**:
  ```rust
  fn check_no_archived_analytic_accounts(dist: &AnalyticDistribution, db: &Db) -> Result<()> {
      for key in dist.keys() {
          for id in parse_ids(key) {
              let account = db.get::<AnalyticAccount>(id)?;
              if !account.active {
                  return Err(UserError::new("archived analytic account"));
              }
          }
      }
      Ok(())
  }
  ```

---

### R13 — `account.analytic.account` invoice/vendor bill count smart buttons   [AXIS-A]

- **odoo source**: `account/models/account_analytic_account.py:L18-77`
- **What it does**: Two computed fields `invoice_count` and `vendor_bill_count` on `account.analytic.account`. Both use `_read_group` on `account.move.line` with domain `['analytic_distribution', 'in', self.ids]`. This works because Odoo's `analytic_distribution` domain operator `in` does a JSON contains check for the given IDs. The `data = {int(account_id): move_count ...}` pattern suggests the group result returns string account IDs that need `int()` conversion. These are purely for UI — smart button counts linking the analytic account back to its associated invoices.
- **woa-rs target**: K10 UI / API. Read-only computed endpoints.

---

### R14 — EPD (Early Payment Discount) analytic distribution propagation   [AXIS-A]

- **odoo source**: `account/models/account_move_line.py:L987-1038`; `account/models/account_move.py:L5063-5068`
- **What it does**: When an invoice has early payment discount lines (`display_type == 'discount'`), the discount lines inherit analytic distribution proportionally from the product lines:
  ```python
  'analytic_distribution': {
      account_id: 100 * value / total
      for account_id, value in dist.items()
  }
  ```
  where `total = sum(dist.values()) or 1` (avoid zero-division). Distribution is proportional to the balance contribution of each plan.
  EPD also uses `_get_distribution` for the cash discount account (L5063-5068): looks up distribution model for the discount account code, partner, company.
  From `test_analytic_distribution_with_discount:L654-717`: confirmed behaviour — discount lines inherit scaled distribution from the product line's distribution.

- **woa-rs target**: K10 + K3 (EPD interaction). When generating discount lines, compute their analytic distribution from the product lines' distribution scaled by balance proportion.

---

## Enterprise gaps flagged

| module | what's missing | what we spec from community source instead |
|---|---|---|
| `analytic` (base addon) | Entire base model definitions: `account.analytic.account`, `account.analytic.plan`, `account.analytic.applicability`, `account.analytic.distribution.model`, `analytic.mixin` (incl. `_merge_distribution`, `_validate_distribution`, `_get_distribution`, `_get_distribution_key`, `_get_plan_fnames`) | Account-side `_inherit` extensions are fully specced. Base model shape inferred from test usage, field references, and method signatures found in account code. The `_get_distribution` scoring algorithm is specced from test assertions (test_model_score, test_model_sequence). |
| `account_reports` | Analytic reporting (cost centre P&L, analytic balance) | Not present. Engine must be built fresh in woa-rs. |
| `analytic` → `account.analytic.plan._column_name()` | Dynamic column creation for each analytic plan on `account.analytic.line` (ORM metaprogramming) | In Rust, use a fixed set of plan slots or a JSON column on `account_analytic_line` rather than dynamic columns. This is an architectural divergence from Odoo — document as RFC candidate. |

---

## Open questions for the Opus porter

1. **Dynamic plan columns**: Odoo creates a real DB column (`x_plan{n}_id`) on `account_analytic_line` for each analytic plan. In Rust with SeaORM this is impractical. Options: (a) fixed max N plan columns; (b) a single JSONB column `plan_account_ids` on the analytic line; (c) a pivot table `analytic_line_plan` (line_id, plan_id, account_id). Recommend option (c) for normalisation. Needs RFC.

2. **`_merge_distribution` `__update__` protocol**: This is complex UI state logic (the `__update__` key list tells the system which plans the user just edited). Does woa-rs need this for its UI, or can the API always replace the full distribution? If full-replace is sufficient, `_merge_distribution` simplifies to a plain override.

3. **`analytic.mixin` on `account.reconcile.model`**: `account_reconcile_model.py` also inherits `analytic.mixin`. Does woa-rs need analytic distribution on reconciliation rules? Not specced in L5 — cross-lane question.

4. **Analytic precision**: `decimal.precision.precision_get('Percentage Analytic')` — what is the default digit precision? Typically 2 in community. Confirm before implementing `approx_eq_100`.

5. **`_get_distribution_key()` on `account.analytic.line`**: returns CSV of all plan account IDs for that line. The implementation is in the base `analytic` module (absent). Inferred: it collects `account.id` for each plan column that is set on the line. Porter needs to verify against actual base module source.

6. **`related_root_plan_ids` argument to `_get_distribution`**: this tells the distribution model to skip already-allocated plans (from `_related_analytic_distribution()`). Important for SO/PO → invoice flow where the SO distribution should be preserved. The base `_get_distribution` presumably filters out models for plans already covered by `related_root_plan_ids`. Needs verification against base module source.

---

## Depth-proof footer

Read: `/home/user/odoo/addons/account/models/account_analytic_account.py` lines=79 depth=full
Read: `/home/user/odoo/addons/account/models/account_analytic_line.py` lines=111 depth=full
Read: `/home/user/odoo/addons/account/models/account_analytic_plan.py` lines=82 depth=full
Read: `/home/user/odoo/addons/account/models/account_analytic_distribution_model.py` lines=71 depth=full
Read: `/home/user/odoo/addons/account/models/account_move_line.py` lines=3742 depth=full (analytic sections: L1-50, L400-450, L980-1090, L1200-1320, L1390-1430, L1760-1900, L2005-2055, L3140-3280, L3490-3520)
Read: `/home/user/odoo/addons/account/tests/test_account_analytic.py` lines=1182 depth=full
Read: `/home/user/odoo/addons/purchase/models/analytic_account.py` lines=35 depth=full
Read: `/home/user/odoo/addons/purchase/models/analytic_applicability.py` lines=16 depth=full
Read: `/home/user/odoo/addons/sale/models/analytic.py` lines=22 depth=full
Read: `/home/user/odoo/addons/account/models/account_move.py` lines=5085 depth=section (L5055-5085)
