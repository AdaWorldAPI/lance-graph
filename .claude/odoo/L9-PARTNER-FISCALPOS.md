RICHNESS-LANE-OK

# Lane L9 — Partner Accounting Properties + Fiscal-Position Assignment

## Sources read (file : line-range : depth)

- `/home/user/odoo/addons/account/models/partner.py` : L1–1170 : full
- `/home/user/woa-rs/.claude/odoo/BRIEFING.md` : L1–166 : full
- `/home/user/woa-rs/.claude/odoo/BRIEFING-GAP.md` : L1–82 : full

---

## Ontology rows

| odoo class | owl pivot | OGIT family (or None) | DOLCE |
|---|---|---|---|
| `res.partner` (account extension) | `vcard:Individual` / `fibo:LegalEntity` (company=True) | `0x80 SmbFoundryCustomer` | Endurant (persistent object) |
| `account.fiscal.position` | `fibo:TaxJurisdiction` (closest; no direct fibo term for fiscal mapping) | `None` — ontology-unmapped, needs Layer-2 alignment axiom | Endurant (named configuration object) |
| `account.fiscal.position.account` | `fibo:AccountMapping` (proposed pivot) | `None` — ontology-unmapped | Endurant |
| `account.payment.term` (referenced) | `fibo:PaymentTerms` | `0x61 BillingCore` | Endurant |
| `account.account` (receivable/payable props) | `fibo:Account` | `0x62 SMBAccounting` | Endurant |

**DOLCE notes:**
- `res.partner` is clearly Endurant: it persists through time and has properties that can change (rank, payment term, fiscal position).
- `account.fiscal.position` is also Endurant: a named configuration record that maps tax A → tax B; it does not "happen" (Perdurant) — it is consulted.
- The `_get_fiscal_position()` **resolution event** is a Perdurant (a happening/process), but the *result* (a fiscal position record) is Endurant.

---

## Rules extracted

### R1 — Per-partner property accounts (receivable / payable)   [AXIS-A]

- **odoo source**: `partner.py:537–546`
- **What it does**: Two `company_dependent` Many2one fields on `res.partner`:
  - `property_account_receivable_id` → restricted to `account_type = 'asset_receivable'`
  - `property_account_payable_id` → restricted to `account_type = 'liability_payable'`
  Both are `company_dependent=True` (stored in `ir.property` keyed by `(partner_id, company_id)`), `check_company=True`, and `ondelete='restrict'` (cannot delete an account that is a partner default). The domain constraint means you cannot assign a non-AR/AP account here.
  These fields are **inherited** by child partners through `commercial_partner_id` rollup (see R5).
- **woa-rs target**: K3 double-entry posting; data foundation for partner → AR/AP account lookup when building journal entries.
- **Rust sketch**:
  ```rust
  // In Partner entity (sea-orm):
  // property_account_receivable_id: Option<AccountId>  -- company-scoped
  // property_account_payable_id:    Option<AccountId>  -- company-scoped
  // Lookup at invoice creation:
  fn get_receivable_account(partner: &CommercialPartner, company_id: CompanyId) -> Result<AccountId, WoaError> {
      partner.property_account_receivable_id
          .ok_or(WoaError::MissingPartnerAccount("receivable"))
  }
  // Constraint: must verify account.account_type == 'asset_receivable' at write time
  // ondelete=restrict: FK with RESTRICT in DB migration
  ```
  Storage note: Odoo uses `ir.property` (EAV), not a direct column. In woa-rs, model as nullable foreign-key columns scoped to company (or use a separate `partner_accounting_props` table per company if multi-company K15 needed).
- **Parity notes / gotchas**: The `company_dependent` EAV storage means each company can see a *different* account for the same partner. woa-rs must decide: flat columns (single-company) or company-keyed rows. K15 (Mehrfirma) is currently missing; plan for the multi-company shape now to avoid a schema migration.

---

### R2 — Per-partner payment terms (customer + supplier)   [AXIS-A]

- **odoo source**: `partner.py:551–558`
- **What it does**: Two `company_dependent` Many2one fields:
  - `property_payment_term_id` → `account.payment.term` — used when issuing sales invoices TO this partner
  - `property_supplier_payment_term_id` → `account.payment.term` — used when receiving vendor bills FROM this partner
  Both are `check_company=True`. The customer term drives invoice due-date computation; the supplier term drives vendor bill due-date computation.
- **woa-rs target**: K3 invoice due-date computation (already partially in L5); this lane provides the *lookup source*.
- **Rust sketch**:
  ```rust
  fn get_customer_payment_term(partner: &CommercialPartner, company_id: CompanyId) -> Option<PaymentTermId> {
      partner.property_payment_term_id
  }
  fn get_supplier_payment_term(partner: &CommercialPartner, company_id: CompanyId) -> Option<PaymentTermId> {
      partner.property_supplier_payment_term_id
  }
  // If None, fall back to journal/company default (outside this lane)
  ```
- **Parity notes / gotchas**: When `None`, odoo falls back silently (no term → due immediately, or journal default). Document that `None` = immediate payment, not an error.

---

### R3 — Manual fiscal-position field on partner   [AXIS-A]

- **odoo source**: `partner.py:547–550`
- **What it does**: `property_account_position_id` — a `company_dependent` Many2one to `account.fiscal.position`. When set, this **always wins** over auto-detection (see R8). The field is `check_company=True`, meaning only fiscal positions belonging to the same company (or a parent company via `check_company_domain_parent_of`) are valid.
- **woa-rs target**: K7 tax compute — input to fiscal-position resolution (R8).
- **Rust sketch**:
  ```rust
  // In Partner entity:
  // property_account_position_id: Option<FiscalPositionId>  -- company-scoped
  //
  // At resolution time:
  fn get_manual_fiscal_position(partner: &Partner, company_id: CompanyId) -> Option<FiscalPositionId> {
      partner.property_account_position_id
  }
  ```
- **Parity notes**: This is the "override" escape hatch. Any auto-detection logic (R8) is bypassed entirely if this is set.

---

### R4 — customer_rank / supplier_rank counters   [AXIS-A]

- **odoo source**: `partner.py:600–601` (field declarations) + `partner.py:800–833` (_increase_rank method) + `partner.py:773–784` (create hook)
- **What it does**:
  - `customer_rank`: Integer (default=0, copy=False). Incremented each time the partner appears on a confirmed sale/invoice. Used to order partners in customer search mode.
  - `supplier_rank`: Integer (default=0, copy=False). Incremented each time the partner appears on a confirmed purchase/vendor bill.
  - **create hook** (L773–784): if `res_partner_search_mode == 'customer'` in context and `customer_rank` not explicitly set, initialises to 1 (and vice versa for supplier).
  - **`_increase_rank(field, n=1)`** (L800–833): The increment is NOT immediate. If the partner already has rank > 0, the increment is **deferred to a post-commit hook** to avoid serialization errors (high-concurrency environment). If rank is currently 0, it increments immediately (so `customer_rank > 0` filtering works right away). The post-commit hook catches `psycopg2.errors.OperationalError` silently (just logs debug).
  - **`_order` property** (L356–363): When `res_partner_search_mode == 'customer'`, partners are ordered by `customer_rank DESC` prepended to the normal order. Same for supplier.
- **woa-rs target**: Data foundation; partner search UX (L9); also determines which partners appear as "customers" vs "vendors" in filtered views.
- **Rust sketch**:
  ```rust
  // Fields on partner table:
  // customer_rank: i32 DEFAULT 0
  // supplier_rank: i32 DEFAULT 0
  //
  // On invoice confirm / sale confirm → call increase_rank("customer_rank", 1)
  // Strategy: for woa-rs (single-tenant, low concurrency), immediate increment is safe.
  // For high-concurrency: use a background task with retry.
  async fn increase_rank(pool: &DbPool, partner_id: PartnerId, field: RankField, n: i32) -> Result<(), WoaError> {
      sqlx::query!("UPDATE res_partner SET customer_rank = customer_rank + ? WHERE id = ?", n, partner_id)
          .execute(pool).await?;
      Ok(())
  }
  // On create in customer search context: set customer_rank = 1 if not supplied
  ```
- **Parity notes / gotchas**: The deferred-post-commit strategy in odoo is a PostgreSQL-specific serialization-error mitigation. MySQL (woa-rs's DB) does not have the same serialization error pattern, so immediate increment is fine. However, note the odoo bug-in-the-open: if the post-commit fails with `OperationalError`, the rank silently stays un-incremented. woa-rs should log a warning rather than silent swallow.

---

### R5 — commercial_partner_id rollup for accounting fields   [AXIS-A]

- **odoo source**: `partner.py:702–710` (_find_accounting_partner + _commercial_fields)
- **What it does**:
  - `_find_accounting_partner(partner)` (L702–704): Returns `partner.commercial_partner_id`. This means all accounting entries for a contact (child partner) are posted to the *commercial* (company-level) partner, not the individual contact. Invoice lines reference the child; the AR/AP account lookup uses the commercial parent.
  - `_commercial_fields()` (L706–710): Extends the base list with `['property_account_payable_id', 'property_account_receivable_id', 'property_account_position_id', 'property_payment_term_id', 'property_supplier_payment_term_id', 'credit_limit']`. This causes odoo's partner hierarchy logic to **sync these fields from parent to children** when a partner is set as a child of a company.
- **woa-rs target**: K3 journal entry partner assignment; partner data model.
- **Rust sketch**:
  ```rust
  fn find_accounting_partner(partner: &Partner) -> PartnerId {
      partner.commercial_partner_id.unwrap_or(partner.id)
  }
  // commercial_partner_id: if partner.parent_id is set AND partner.is_company=False,
  //   commercial_partner_id = parent.commercial_partner_id (recursive up to root company)
  // otherwise commercial_partner_id = self
  //
  // When syncing commercial fields:
  // If partner gets a new parent, the five fields above are copied from parent → child.
  ```
- **Parity notes / gotchas**:
  - The `write()` hook at L749–771 enforces: if you change `parent_id` on a partner that already has accounting move lines, it **re-points all move lines** to the new `commercial_partner_id`. This is a significant operation — it uses `bypass_lock_check` to write through GoBD lock. In woa-rs (K11 Festschreibung), this bypass must be replicated deliberately.
  - VAT guard at L757–758: if the child partner has a different VAT than the new parent, raising `UserError` — you cannot reparent if VAT differs.

---

### R6 — credit / debit computed balances   [AXIS-A]

- **odoo source**: `partner.py:365–449` (_credit_debit_get, _asset_difference_search, _credit_search, _debit_search)
- **What it does**:
  - `credit` (Total Receivable): Sum of `amount_residual` on posted move lines where `account_type = 'asset_receivable'` and `reconciled IS NOT TRUE`, for all partners in the set, filtered to current company's root subtree.
  - `debit` (Total Payable): Same but `account_type = 'liability_payable'`, sign negated (`-val`).
  - SQL is executed directly (not ORM) for performance; uses `account_move_line`'s search query infrastructure (`_search` → `from_clause` + `where_clause`).
  - Both fields are `Monetary` with `search` functions (`_credit_search`, `_debit_search`) allowing filtering partners by outstanding balance using operators `<`, `=`, `>`, `>=`, `<=`.
  - The search (`_asset_difference_search`) uses raw SQL with `SPLIT_PART(line_company.parent_path, '/', 1)::int = company.root_id.id` — PostgreSQL-specific!
- **woa-rs target**: K3 partner balance views (dashboard, partner list).
- **Rust sketch**:
  ```rust
  // As a computed/derived value (not stored), fetch on demand:
  async fn get_partner_credit(pool: &DbPool, partner_id: PartnerId, company_root_id: CompanyId) -> Decimal {
      // SELECT SUM(aml.amount_residual) FROM account_move_line aml
      //   JOIN account_account aa ON aml.account_id = aa.id
      //   JOIN account_move am ON aml.move_id = am.id
      //   WHERE aa.account_type = 'asset_receivable'
      //     AND aml.partner_id = partner_id
      //     AND aml.reconciled = FALSE
      //     AND am.state = 'posted'
      //     AND aml.company_id IN (SELECT id FROM res_company WHERE root_id = company_root_id)
      // MySQL note: no SPLIT_PART — use JOIN to res_company WHERE root_company_id = ?
  }
  ```
- **Parity notes / gotchas**: The odoo SQL uses `SPLIT_PART(parent_path, '/', 1)::int` — a PostgreSQL extension. MySQL equivalent is a subquery on the company tree. woa-rs (single-company initially) can simplify to `company_id = ?`.

---

### R7 — Days Sales Outstanding (DSO) compute   [AXIS-A]

- **odoo source**: `partner.py:472–490`
- **What it does**: `days_sales_outstanding = (credit / total_invoiced_tax_included) * days_since_oldest_invoice`. The formula:
  1. Find all posted sale-type invoices for `commercial_partner_id` in current company.
  2. `oldest_invoice_date` = min(`invoice_date`) across those invoices.
  3. `total_invoiced_tax_included` = sum(`amount_total_signed`).
  4. `days_since_oldest_invoice` = today − oldest_invoice_date (in days).
  5. If `total_invoiced_tax_included == 0` → DSO = 0 (no division).
  6. `credit` is the AR balance (R6).
- **woa-rs target**: K3 partner dashboard / AR aging.
- **Rust sketch**:
  ```rust
  fn compute_dso(credit: Decimal, total_invoiced: Decimal, oldest_invoice_date: NaiveDate, today: NaiveDate) -> Decimal {
      if total_invoiced.is_zero() { return Decimal::ZERO; }
      let days = (today - oldest_invoice_date).num_days();
      (credit / total_invoiced) * Decimal::from(days)
  }
  ```
- **Parity notes**: Uses `amount_total_signed` (includes tax), not net. DSO = 0 when no invoices exist (not an error).

---

### R8 — Fiscal-position auto-resolution: `_get_fiscal_position`   [HYBRID: AXIS-A guard + AXIS-B core]

- **odoo source**: `partner.py:246–279` (_get_fiscal_position) + `partner.py:208–244` (_get_first_matching_fpos, _get_fpos_validation_functions)

#### AXIS-A guard — deterministic precedence

**What it does (guard layer)**:
1. If `partner` is falsy → return empty (no fiscal position).
2. Compute `intra_eu`: both company and partner have VAT, both VAT prefixes are in the EU country codes set.
3. Compute `vat_exclusion`: both VAT prefixes are identical (same country).
4. **Delivery address selection**: if no `delivery` arg, OR if (`intra_eu AND vat_exclusion AND partner.country_id == company.country_id`), then `delivery = partner`. Otherwise keep separate delivery address.
5. **Manual override wins**: check `delivery.property_account_position_id` (company-scoped), then `partner.property_account_position_id`. If either is set → return it immediately. No auto-detection.
6. If `partner.country_id` is empty → return empty (can't match country-based rules).
7. Search all `auto_apply=True` fiscal positions for current company.
8. Call `_get_first_matching_fpos(delivery)` to find first match.

**Rust sketch (guard)**:
```rust
fn get_fiscal_position(
    partner: &Partner,
    delivery: Option<&Partner>,
    company: &Company,
    eu_country_codes: &HashSet<&str>,
    auto_apply_positions: &[FiscalPosition],
) -> Option<FiscalPositionId> {
    // Step 1: partner must exist
    // Step 2-3: intra_eu / vat_exclusion
    let intra_eu = company.vat.as_ref().zip(partner.vat.as_ref()).map(|(cv, pv)| {
        eu_country_codes.contains(&cv[..2]) && eu_country_codes.contains(&pv[..2])
    }).unwrap_or(false);
    let vat_exclusion = company.vat.as_ref().zip(partner.vat.as_ref())
        .map(|(cv, pv)| cv[..2] == pv[..2]).unwrap_or(false);

    // Step 4: delivery selection
    let effective_delivery = if delivery.is_none()
        || (intra_eu && vat_exclusion && partner.country_id == Some(company.country_id)) {
        partner
    } else {
        delivery.unwrap()
    };

    // Step 5: manual override
    if let Some(fp) = effective_delivery.property_account_position_id
                       .or(partner.property_account_position_id) {
        return Some(fp);
    }

    // Step 6: no country → no match
    if partner.country_id.is_none() { return None; }

    // Step 7-8: delegate to matching (AXIS-B)
    get_first_matching_fpos(effective_delivery, auto_apply_positions)
}
```

#### AXIS-B core — heuristic matching

**What it does (matching layer)**: `_get_first_matching_fpos(delivery)` (L208–213):
- Sort all candidate fiscal positions: **company-specific first** (longer `parent_ids` chain = more specific company → goes first), then by `sequence` ascending.
- For each fpos in sorted order, run ALL 5 validation functions; return the first fpos where all pass.

**5 validation predicates** (L215–244):
1. `vat_required`: `not fpos.vat_required OR partner._get_vat_required_valid(company)`. The base `_get_vat_required_valid` simply returns `bool(partner.vat)` — hook for VIES in Enterprise.
2. `zip_range`: `not (fpos.zip_from AND fpos.zip_to) OR (partner.zip AND fpos.zip_from <= partner.zip <= fpos.zip_to)`. Lexicographic comparison (zip stored as Char, padded to equal length with leading zeros for digit-only zips via `_convert_zip_values`).
3. `state`: `not fpos.state_ids OR partner.state_id in fpos.state_ids`.
4. `country`: `not fpos.country_id OR partner.country_id == fpos.country_id`.
5. `country_group`: `not fpos.country_group_id OR (partner.country_id in group.country_ids AND (not partner.state_id OR partner.state_id not in group.exclude_state_ids))`.

All 5 must pass (AND semantics). First match wins (ORDER BY company-specificity DESC, sequence ASC).

**Why AXIS-B**: The matching is not "look up a single exact key" — it is a priority-ordered search through a variable-length list of rules, each with multi-dimensional predicates (VAT, zip range, state, country, country group). The *choice* of which rule fires is evidence-weighted (more-specific company overrides less-specific; sequence ordering is an admin-configured priority). In a live system, adding new fiscal positions changes which rule fires for all partners. This is exactly the kind of multi-factor, priority-ranked, belief-revision pattern that NARS/lance-graph handles better than brittle Rust if/else.

- **Delegation tuple**:
  - `ReasoningKind = CustomerCategory` (classifying a partner into a tax treatment category)
  - `InferenceType = Deduction` (the rules are explicit lookup — but the *priority ordering* introduces induction over the rule set)
  - `SemiringChoice = NarsTruth` (evidence fusion: each predicate is a partial match; the "best" fiscal position wins by highest-specificity evidence)
  - `ThinkingStyle = Analytical` (inherited from `0x80 SmbFoundryCustomer` family; resolving a customer's tax category is an analytical classification task)

`SAVANT: name=FiscalPositionResolver family=0x80 reasoning=CustomerCategory inference=Deduction semiring=NarsTruth style=Analytical — multi-predicate priority-ranked fiscal position matching is a belief-revision classification over the partner's country/VAT/zip evidence, not a single-key lookup; delegate to lance-graph so new fpos rules do not require Rust recompilation.`

- **Parity notes / gotchas**:
  - Zip comparison is **lexicographic** (string `<=`), not numeric, because German PLZ can have leading zeros (e.g., `01067`). The `_convert_zip_values` method pads digit-only zips with leading zeros to equal length before storing. woa-rs must replicate this padding on write.
  - The `intra_eu + vat_exclusion + same_country` short-circuit forces `delivery = partner` (use invoicing address, not ship-to address). This matters for B2B within Germany: a DE company shipping to a DE customer uses the invoicing address's fiscal position, not the delivery address.
  - The company-specificity sort (`-len(f.company_id.parent_ids)`) is a multi-company concern — in single-company woa-rs, this is a no-op but must be preserved for K15.

---

### R9 — `map_tax`: fiscal-position tax remapping   [AXIS-A]

- **odoo source**: `partner.py:154–163`
- **What it does**:
  ```python
  def map_tax(self, taxes):
      if not self:
          return taxes  # no fiscal position → taxes unchanged
      if not self.tax_ids and taxes.fiscal_position_ids:
          return self.env['account.tax']  # empty fpos with fpos-aware taxes → remove all taxes
      return self.env['account.tax'].browse(unique(
          tax_id
          for tax in taxes
          for tax_id in (self.tax_map or {}).get(tax.id, [tax.id])
      ))
  ```
  - `tax_map` is a Binary computed field (dict): `{src_tax_id: [dest_tax_id, ...]}` built from the M2M `tax_ids` via `original_tax_ids` back-relation (L98–105).
  - For each input tax: look up in `tax_map`; if found, replace with the mapped list; if not found, keep original (identity mapping).
  - `unique()` deduplicates the output (preserving order).
  - Special case: empty fiscal position (`not self.tax_ids`) + tax has `fiscal_position_ids` → **removes the tax entirely**. This handles "OSS/distance-selling" fiscal positions that nullify specific taxes.
- **woa-rs target**: K7 tax compute — called for every invoice line when a fiscal position is active.
- **Rust sketch**:
  ```rust
  fn map_tax(
      fiscal_pos: Option<&FiscalPosition>,
      taxes: &[TaxId],
      tax_map: &HashMap<TaxId, Vec<TaxId>>,
      tax_has_fpos: &HashSet<TaxId>, // taxes that have fiscal_position_ids set
  ) -> Vec<TaxId> {
      let Some(fp) = fiscal_pos else { return taxes.to_vec(); };
      if fp.tax_ids.is_empty() {
          // empty fpos: remove taxes that are fpos-aware
          return taxes.iter()
              .filter(|t| !tax_has_fpos.contains(t))
              .cloned().collect();
      }
      // Normal mapping: src → [dest...] or identity
      let mut result = Vec::new();
      let mut seen = HashSet::new();
      for &tax in taxes {
          let mapped = tax_map.get(&tax).map(|v| v.as_slice()).unwrap_or(&[tax]);
          for &dest in mapped {
              if seen.insert(dest) { result.push(dest); }
          }
      }
      result
  }
  ```
- **Parity notes**:
  - `unique()` in odoo preserves insertion order (Python `dict` semantics since 3.7). Rust `HashSet` does not preserve order; the sketch above preserves insertion order via manual `seen` set.
  - The "empty fpos removes all fpos-aware taxes" branch is subtle and easy to miss — it requires knowing which taxes have `fiscal_position_ids` set (a back-reference query).

---

### R10 — `map_account`: fiscal-position account remapping   [AXIS-A]

- **odoo source**: `partner.py:165–166`
- **What it does**:
  ```python
  def map_account(self, account):
      return self.env['account.account'].browse(
          (self.account_map or {}).get(account.id, account.id)
      )
  ```
  `account_map` (Binary computed, L107–110): `{src_account_id: dest_account_id}` from `account_ids` One2many. Simple dict lookup; if no mapping exists for this account, return the account unchanged (identity).
- **woa-rs target**: K3 journal entry posting — account substitution per fiscal position.
- **Rust sketch**:
  ```rust
  fn map_account(
      fiscal_pos: Option<&FiscalPosition>,
      account_id: AccountId,
      account_map: &HashMap<AccountId, AccountId>,
  ) -> AccountId {
      fiscal_pos
          .and_then(|_| account_map.get(&account_id))
          .copied()
          .unwrap_or(account_id)
  }
  ```
- **Parity notes**: Simpler than `map_tax` — one-to-one mapping only (no list). No "empty fpos" special case. The `account_src_dest_uniq` constraint (L320–323) ensures no duplicate src→dest pairs per fiscal position.

---

### R11 — Zip value normalisation on write   [AXIS-A]

- **odoo source**: `partner.py:181–206` (_convert_zip_values, create, write overrides)
- **What it does**: Before storing `zip_from`/`zip_to`, if both are present and both are digit-only, pad both to `max(len(zip_from), len(zip_to))` with leading zeros. E.g., `('1000', '99999')` → `('01000', '99999')`. This ensures lexicographic comparison `zip_from <= partner.zip <= zip_to` works correctly for numeric German PLZ.
  - Validation constraint (L112–116): `zip_from` and `zip_to` must either both be empty or both be set, AND `zip_from <= zip_to`.
- **woa-rs target**: fiscal-position table migration + write handler.
- **Rust sketch**:
  ```rust
  fn convert_zip_values(zip_from: &str, zip_to: &str) -> (String, String) {
      if zip_from.is_empty() || zip_to.is_empty() { return (zip_from.to_string(), zip_to.to_string()); }
      let max_len = zip_from.len().max(zip_to.len());
      let from = if zip_from.chars().all(|c| c.is_ascii_digit()) {
          format!("{:0>width$}", zip_from, width = max_len)
      } else { zip_from.to_string() };
      let to = if zip_to.chars().all(|c| c.is_ascii_digit()) {
          format!("{:0>width$}", zip_to, width = max_len)
      } else { zip_to.to_string() };
      (from, to)
  }
  // Constraint check before write:
  fn validate_zip_range(zip_from: &str, zip_to: &str) -> Result<(), WoaError> {
      match (zip_from.is_empty(), zip_to.is_empty()) {
          (true, true) => Ok(()),
          (false, false) if zip_from <= zip_to => Ok(()),
          _ => Err(WoaError::Validation("Invalid zip range")),
      }
  }
  ```

---

### R12 — `is_domestic` computed field   [AXIS-A]

- **odoo source**: `partner.py:74–77`
- **What it does**: `is_domestic = (self == self.company_id.domestic_fiscal_position_id)`. A Boolean stored field on `account.fiscal.position`. True only for the fiscal position designated as the company's domestic/default position. Used in UI to flag the home-country fiscal position.
- **woa-rs target**: fiscal-position entity, company config.
- **Rust sketch**: Boolean column on fiscal position; OR look it up dynamically by comparing `company.domestic_fiscal_position_id == self.id`.

---

### R13 — VAT format check hooks   [AXIS-A / partial AXIS-B]

- **odoo source**: `partner.py:841–870` (_check_vat, _run_vat_checks, _get_vat_required_valid)
- **What it does**:
  - `_run_vat_checks(country, vat, partner_name, validation)` (L848–865): In community, this is a **stub** — returns `(vat, country_code)` unchanged. Real VAT validation is in `base_vat` module (which this file imports from for `_ref_vat`). Community code cannot validate VAT syntax; Enterprise/`base_vat` does.
  - `_get_vat_required_valid(company)` (L867–870): stub — returns `bool(partner.vat)`. Enterprise extends this with VIES lookup.
  - `_check_vat(validation)` calls `_run_vat_checks` and may reformat the VAT.
- **woa-rs target**: Partner VAT field validation.
- **Parity notes / gotchas**: The base community `_run_vat_checks` is essentially a no-op. Real validation requires `base_vat` which IS present in the community clone (imported at L14). The actual validation logic lives in `base_vat/models/res_partner.py` — that file should be read separately if L9 needs full VAT validation coverage. Flag as **Enterprise gap partially** (VIES check is Enterprise; format check is community via `base_vat`).

---

### R14 — autopost_bills field   [AXIS-A with AXIS-B annotation]

- **odoo source**: `partner.py:602–608`
- **What it does**: Selection field `autopost_bills ∈ {always, ask, never}` (default: `ask`). Controls whether vendor bills from this partner are auto-posted:
  - `always`: auto-post every time.
  - `ask`: prompt after 3 validations without edits.
  - `never`: never auto-post.
- **woa-rs target**: K3 bill posting flow.
- **Note**: The `ask` logic (counting validations without edits) is likely implemented in the bill posting flow, not here. This field is the *configuration*, not the counter logic.
- **Rust sketch**: Enum field on Partner; consulted at bill validation time.

---

### R15 — credit_limit + use_partner_credit_limit   [AXIS-A guard + AXIS-B risk signal]

- **odoo source**: `partner.py:515–524` (fields) + `partner.py:670–683` (compute/inverse)
- **What it does**:
  - `credit_limit: Float` (company_dependent). Per-partner credit limit. Falls back to company-level default.
  - `use_partner_credit_limit: Boolean` (computed): True if partner's credit_limit differs from the company-level fallback.
  - `_compute_use_partner_credit_limit`: compares `partner.credit_limit` against `_fields['credit_limit'].get_company_dependent_fallback(self)`.
  - `_inverse_use_partner_credit_limit`: if toggled off, resets partner's credit_limit to the company default.
  - `show_credit_limit` (L681–683): True if `company.account_use_credit_limit` is enabled.
- **woa-rs target**: K3 / accounts-receivable risk check.
- **Parity notes**: Credit limit enforcement logic (blocking an invoice if limit exceeded) is in the sale/invoice flow, not here. This lane captures only the field definitions and compute logic.

---

### R16 — `trust` field (debtor quality signal)   [AXIS-B annotation]

- **odoo source**: `partner.py:566`
- **What it does**: `trust ∈ {good, normal, bad}` — company_dependent. Used as a qualitative risk signal for this partner as a debtor. No automatic computation; manually set by accounting staff.
- **woa-rs target**: K3 AR risk / dunning escalation (Mahnwesen).
- **Delegation tuple**: This field is an input to dunning escalation heuristics. The *assignment* of trust is a human judgment; the *use* of trust in escalation decisions is AXIS-B.
  - `ReasoningKind = CustomerCategory`
  - `InferenceType = Induction` (pattern over payment history to suggest trust level)
  - `SemiringChoice = NarsTruth`
  - `ThinkingStyle = Analytical` (from `0x80 SmbFoundryCustomer`)

`SAVANT: name=PartnerTrustAdvisor family=0x80 reasoning=CustomerCategory inference=Induction semiring=NarsTruth style=Analytical — the 'trust' rating should be inferred from payment history patterns rather than maintained manually; lance-graph can update beliefs on each payment event via Revision inference.`

---

### R17 — EDI format fields   [AXIS-A]

- **odoo source**: `partner.py:576–597` (fields) + `partner.py:651–667` (compute/inverse)
- **What it does**: `invoice_edi_format` (computed, stored via `invoice_edi_format_store`) — determines the eInvoice format (ZUGFeRD, XRechnung, etc.) for this partner. Logic:
  - Reads from `commercial_partner_id.invoice_edi_format_store`.
  - If store = `'none'` → `invoice_edi_format = False`.
  - If store is empty → falls back to `_get_suggested_invoice_edi_format()` (stub, returns False in base; overridden in l10n modules).
  - Inverse: if set to the suggested value → clear store (let suggestion win); if cleared → store `'none'`; otherwise store the value.
- **woa-rs target**: K9 DATEV / eInvoice export (X-Rechnung).
- **Parity notes**: `_get_suggested_invoice_edi_format()` is a hook — in `l10n_de`, it likely returns `'xrechnung'` for German partners. Must read `l10n_de` extension to get the DE-specific suggestion.

---

### R18 — Partner deletion guard   [AXIS-A]

- **odoo source**: `partner.py:786–798` (_unlink_if_partner_in_account_move)
- **What it does**: `@api.ondelete(at_uninstall=False)` — prevents deletion of any partner that appears on any `account.move` in draft or posted state. Raises `UserError`. Note: applies to ALL states (not just posted) to prevent orphaning draft invoices.
- **woa-rs target**: DELETE endpoint for partners.
- **Rust sketch**:
  ```rust
  async fn check_partner_deletable(pool: &DbPool, partner_id: PartnerId) -> Result<(), WoaError> {
      let count = sqlx::query_scalar!(
          "SELECT COUNT(*) FROM account_move WHERE partner_id = ? AND state IN ('draft', 'posted')",
          partner_id
      ).fetch_one(pool).await?;
      if count > 0 {
          return Err(WoaError::Validation("Partner cannot be deleted: used in accounting"));
      }
      Ok(())
  }
  ```

---

### R19 — VAT-on-reparent guard   [AXIS-A]

- **odoo source**: `partner.py:749–770`
- **What it does**: When `parent_id` changes on a partner that has accounting move lines:
  1. If the new parent has a different VAT than the partner → raise `UserError` (cannot reparent if VAT differs, as this would corrupt the legal entity mapping on existing entries).
  2. If reparenting succeeds, update `partner_id` on ALL existing move lines for this partner to `partner.commercial_partner_id` (bypassing GoBD lock).
  3. Also update `commercial_partner_id` on account.move records that are "entirely" for this partner.
- **woa-rs target**: Partner write endpoint; K11 GoBD considerations.
- **Parity notes / gotchas**: The bypass_lock_check is explicit — it passes `BYPASS_LOCK_CHECK` sentinel. In woa-rs's K11 implementation, this bypass path must be replicated (and audited/logged, since GoBD requires traceability of such changes).

---

### R20 — `_compute_tax_map` and `_compute_account_map`   [AXIS-A]

- **odoo source**: `partner.py:98–110`
- **What it does**: These computed Binary fields pre-build lookup dictionaries at fiscal-position load time:
  - `tax_map`: `{src_tax_id: [dest_tax_id, ...]}` — built by iterating `tax_ids` (the M2M of destination taxes) and accessing `dest_tax.original_tax_ids` (the back-relation from dest to src). Multiple src taxes can map to the same dest.
  - `account_map`: `{src_account_id: dest_account_id}` — simple dict from `account_ids` lines.
  Both are stored as Binary (serialised dict) — effectively a denormalised cache for fast runtime lookup.
- **woa-rs target**: Cache these at fiscal-position load time; avoid re-querying on every invoice line.
- **Rust sketch**:
  ```rust
  struct FiscalPosition {
      id: FiscalPositionId,
      tax_map: HashMap<TaxId, Vec<TaxId>>,    // pre-built
      account_map: HashMap<AccountId, AccountId>, // pre-built
      // ... other fields
  }
  // Build on load:
  fn build_tax_map(tax_mappings: &[TaxMapping]) -> HashMap<TaxId, Vec<TaxId>> {
      let mut map: HashMap<TaxId, Vec<TaxId>> = HashMap::new();
      for tm in tax_mappings {
          map.entry(tm.src_tax_id).or_default().push(tm.dest_tax_id);
      }
      map
  }
  ```

---

## Enterprise gaps flagged

| Module | What's missing | What we spec from community |
|---|---|---|
| `base_vat` (community, present) | VIES real-time VAT validation | Base `_run_vat_checks` is a stub; format validation in `base_vat` is present but `_get_vat_required_valid` is stub (no VIES). woa-rs: implement basic DE/EU format check; VIES lookup is a future K7 enhancement. |
| `account_reports` (Enterprise, absent) | Partner ledger reports, AR aging reports with fiscal-position breakdown | Not present. Reports built fresh on woa-rs side. |
| `account_asset` (Enterprise, absent) | No direct dependency in this lane | N/A |
| `sale` (community, present but separate) | `credit_to_invoice` field is "TO OVERRIDE in Sales" (L412) — always returns False in base account | woa-rs: when porting sale module, must override this compute. Flag as pending. |
| VIES / VAT validation | `_get_vat_required_valid` hook (L867–870) is community stub returning `bool(partner.vat)` | Enterprise extends with live VIES API call. woa-rs: implement as optional external service call; default = syntactic check only. |
| `l10n_de` EDI suggestion | `_get_suggested_invoice_edi_format()` stub (L697–700) | The DE-specific suggestion (XRechnung) is in `l10n_de` — read that file separately for K9. |

---

## Open questions for the Opus porter

1. **Multi-company storage shape for `property_*` fields**: Odoo uses `ir.property` (EAV table) for `company_dependent` fields — one row per `(partner, company)`. woa-rs currently has no multi-company (K15). Should we model these as nullable FK columns on `res_partner` (single-company, simple) and plan a schema migration for K15, or pre-build a `partner_accounting_props(partner_id, company_id, ...)` table now?

2. **`commercial_partner_id` cascade on move lines**: The write hook (R19) updates move lines when reparenting — this bypasses GoBD lock. In woa-rs (K11 Festschreibung), do we allow this bypass? If yes, must it be logged in the audit trail? Stefan's GoBD obligations apply.

3. **Zip range matching semantics**: German PLZ `01067` (Dresden) — if zip_from and zip_to are both digit-only, odoo pads them to equal length for lexicographic comparison. What about mixed (alphanumeric) postal codes? The code only pads when `isdigit()` — alphanumeric zips (UK, Canada) are compared as-is. Is woa-rs targeting DE-only (all numeric) or international?

4. **`autopost_bills` counter**: The `ask` branch triggers after "3 validations without edits" — where is this counter? Not in this file. Likely in `account.move` posting logic. Needs a cross-reference with L1 (account.move state machine).

5. **`trust` field vs dunning**: The `trust` value (R16) is used in dunning (Mahnwesen). Where exactly? The dunning model (Mahnwesen) is in woa-rs scope (woa-rs has partial Mahnwesen from Round 9). Should the trust field be in the initial DB schema?

6. **`_get_first_matching_fpos` and the Savant boundary**: Should woa-rs implement the deterministic 5-predicate matching in Rust AS WELL as having the Savant, or should the Savant fully replace it? Recommendation: implement deterministic fallback in Rust (for offline/fast-path); Savant adds evidence weighting when lance-graph is available (graceful degradation).

7. **`account_map` / `tax_map` Binary fields**: Odoo serialises these as Python dicts in a Binary column. In woa-rs: use `JSONB` (if Postgres) or `JSON` (MySQL) column, or compute from normalized join tables on each request? Given fiscal positions change rarely, an in-process cache (built at startup, invalidated on write) is sufficient.

---

## Depth-proof footer

Read: `/home/user/odoo/addons/account/models/partner.py` lines=1170 depth=full
Read: `/home/user/woa-rs/.claude/odoo/BRIEFING.md` lines=166 depth=full
Read: `/home/user/woa-rs/.claude/odoo/BRIEFING-GAP.md` lines=82 depth=full
