RICHNESS-LANE-OK

# Lane L8 — Product + UoM + Pricelist + Costing

> Generated 2026-05-26. Read-only harvest lane; no cargo, no src edits, no git.

## Sources read (file : line-range : depth)

- `/home/user/odoo/addons/product/models/product_template.py` : L1-1598 : full
- `/home/user/odoo/addons/product/models/product_product.py` : L1-1197 : full
- `/home/user/odoo/addons/product/models/product_category.py` : L1-69 : full
- `/home/user/odoo/addons/product/models/product_pricelist.py` : L1-415 : full
- `/home/user/odoo/addons/product/models/product_pricelist_item.py` : L1-684 : full
- `/home/user/odoo/addons/product/models/uom_uom.py` : L1-30 : full (product extension only — stub)
- `/home/user/odoo/addons/account/models/uom_uom.py` : L1-59 : full (account extension — UNECE codes)
- `/home/user/odoo/addons/account/models/product.py` : L1-523 : full (account extension on product.category + product.template)
- `/home/user/odoo/addons/stock/models/product.py` : L1274-1389 : full (stock extension on product.category + uom.uom)
- `uom/models/uom_uom.py` : NOT PRESENT in this clone — core `uom` module absent; logic reconstructed from call-sites + WebFetch of raw.githubusercontent.com/odoo/odoo/17.0/addons/uom/models/uom_uom.py

---

## Ontology rows

| odoo class | owl pivot | OGIT family (or None) | DOLCE |
|---|---|---|---|
| `product.template` | `schema:Product` | None (ontology-unmapped; needs Layer-2 alignment axiom under a new `0x63 ProductCatalog` family or mapped as Endurant under `0x62 SMBAccounting`) | Endurant (persistent physical/service thing) |
| `product.product` | `schema:Product` (variant) | None (same gap; sub-type of template via `_inherits`) | Endurant |
| `product.category` | `schema:Category` / `skos:Concept` | None (unmapped) | Abstract |
| `uom.uom` | `schema:QuantitativeValue` / `qudt:Unit` | None (unmapped) | Abstract |
| `uom.category` | `schema:Enumeration` | None (unmapped) | Abstract |
| `product.pricelist` | `schema:PriceSpecification` | None (unmapped) | Abstract |
| `product.pricelist.item` | `schema:UnitPriceSpecification` | None (unmapped) | Abstract |

All seven odoo classes resolve to `None` from `resolve_odoo_to_family()`. They are ontology-unmapped and need Layer-2 alignment axioms before Savant delegation can be wired up. Proposed mapping: `product.*` → `schema:Product` → new OGIT family `0x63 ProductCatalog` (Analytical style, matching the catalog/pricing domain).

---

## Rules extracted

---

### R1 — Product type enum + purchase_ok gate  [AXIS-A]

- **odoo source**: `product_template.py:54-65`, `product_template.py:117`
- **What it does**: `type` ∈ `{consu, service, combo}`. `purchase_ok` is stored+recomputable; base implementation is a no-op (`pass`) — extended by `purchase` module. `sale_ok` defaults True. `combo` type forbids `attribute_line_ids` (constraint at L490-493) and requires at least one `combo_ids` entry (L490-492). When type changes to non-combo, `combo_ids` is cleared (L604-606).
- **woa-rs target**: Data model for `product` entity (K-step: data foundation). Maps to a `ProductKind` enum in Rust.
- **Rust sketch**:
  ```rust
  pub enum ProductKind { Goods, Service, Combo }
  pub struct Product {
      kind: ProductKind,
      sale_ok: bool,
      purchase_ok: bool,  // overridden by purchase module
      combo_ids: Vec<ComboId>,  // non-empty iff kind==Combo
  }
  // Constraint: kind==Combo => combo_ids.len() >= 1
  // Constraint: kind==Combo => attribute_lines.is_empty()
  ```
- **Parity notes**: `purchase_ok` is a `compute` with `store=True, readonly=False` — effectively a stored field with a no-op compute in base. Only relevant when purchase module is present.

---

### R2 — lst_price / list_price / price_extra triad  [AXIS-A]

- **odoo source**: `product_product.py:25-35`, `product_product.py:319-335`, `product_product.py:308-316`
- **What it does**: `lst_price` (on `product.product`) = `list_price` (on template) + `price_extra` (sum of variant attribute PTAVs) + optional UoM conversion. Context key `uom` (int id) triggers conversion via `uom_id._compute_price(list_price, to_uom)`. Setting `lst_price` via `_set_product_lst_price` reverses: `value = convert(lst_price) - price_extra → write({'list_price': value})`.
- **woa-rs target**: Pricing layer (Vorgang line price). A product variant's sale price in a given UoM.
- **Rust sketch**:
  ```rust
  fn lst_price(product: &ProductVariant, to_uom: Option<&Uom>) -> Decimal {
      let base = match to_uom {
          Some(u) if u != &product.uom_id =>
              uom_compute_price(product.list_price, &product.uom_id, u),
          _ => product.list_price,
      };
      base + product.price_extra   // price_extra = sum of ptav.price_extra
  }
  ```
- **Parity notes**: `price_extra` is computed from `product_template_attribute_value_ids.mapped('price_extra')` — sum of per-variant attribute surcharges. No rounding applied at this layer; rounding deferred to pricelist or currency.

---

### R3 — standard_price (cost) — company-dependent, variant-level  [AXIS-A]

- **odoo source**: `product_product.py:62-68`, `product_template.py:100-107`, `product_template.py:310-321`
- **What it does**: `standard_price` lives on `product.product` (variant) with `company_dependent=True` — each company stores its own cost per variant. On the template it is a delegated compute (`_compute_template_field_from_variant_field`) that reads from the single variant if exactly one variant exists, else returns `False`/`0.0`. Constraint: `standard_price >= 0` (validated by `_onchange_standard_price` at `product_product.py:399-402`).
- **woa-rs target**: Cost field for FIFO/AVCO/Standard costing (K3 valuation bridge; also used in pricelist formula base `standard_price`).
- **Rust sketch**:
  ```rust
  // company_dependent: stored as (company_id, product_id) → Decimal in ir.property or JSON column
  fn standard_price(variant: &ProductVariant, company: &Company) -> Decimal {
      // fetched per-company; default 0.0
      assert!(price >= Decimal::ZERO);  // constraint
      price
  }
  ```
- **Parity notes**: `standard_price` fetch is elevated to `sudo()` when `price_type == 'standard_price'` in `_price_compute` — cost is accessible to all users through pricelist computation even if restricted directly. Note `cost_currency_id` may differ from `currency_id` when company currency differs.

---

### R4 — currency_id / cost_currency_id compute  [AXIS-A]

- **odoo source**: `product_template.py:88-92`, `product_template.py:257-267`
- **What it does**: `currency_id` = `company_id.currency_id OR main_company.currency_id`. `cost_currency_id` = same logic but depends on `env.company` (context-sensitive). Both are computed, not stored. `cost_currency_id` depends on `depends_context('company')`.
- **woa-rs target**: Currency resolution for pricelist computations.
- **Rust sketch**:
  ```rust
  fn currency_id(product: &ProductTemplate, env: &Env) -> CurrencyId {
      product.company_id
          .and_then(|c| c.currency_id)
          .unwrap_or(env.main_company().currency_id)
  }
  ```
- **Parity notes**: `cost_currency_id` uses `env.company` (the active company from context), not `product.company_id`. These can diverge in multi-company setups.

---

### R5 — UoM core model: factor, rounding, _compute_quantity, _compute_price  [AXIS-A]

- **odoo source**: `uom/models/uom_uom.py` (NOT in this clone — reconstructed from call-sites + GitHub 17.0 raw fetch)
- **What it does**:
  - Fields: `name`, `category_id` (Many2one `uom.category`), `factor` (Float, non-zero SQL constraint), `factor_inv` (computed = 1/factor), `rounding` (Float, positive SQL constraint), `uom_type` ∈ `{bigger, reference, smaller}`, `active`.
  - SQL constraints: `factor != 0`, `rounding > 0`, reference unit has `factor = 1.0`.
  - Each category has exactly one reference unit (`uom_type == 'reference'`).
  - **`_compute_quantity(qty, to_unit, round=True, rounding_method='UP', raise_if_failure=True)`**:
    1. Validate `self.category_id == to_unit.category_id` (else raise or return qty as-is if `raise_if_failure=False`).
    2. `result = (qty / self.factor) * to_unit.factor`  — convert through reference unit.
    3. If `round=True`: apply `float_round(result, precision_rounding=to_unit.rounding, rounding_method=rounding_method)`.
    4. Return `result`.
  - **`_compute_price(price, to_unit)`**:
    1. If `self == to_unit` or `not price`: return `price`.
    2. Validate categories match.
    3. `return (price * self.factor) / to_unit.factor`
    4. No rounding applied (prices carry full float precision until currency rounds).
  - Usage confirmed in `_adjust_uom_quantities` (stock): `rounding_method='HALF-UP'`.
- **woa-rs target**: Core UoM conversion engine (data foundation; used in every pricelist + stock move + invoice line).
- **Rust sketch**:
  ```rust
  pub struct Uom {
      pub id: UomId,
      pub category_id: UomCategoryId,
      pub factor: f64,      // ratio vs reference; reference unit has factor=1.0
      pub rounding: f64,    // precision multiplier (e.g. 0.01, 1.0, 0.001)
      pub uom_type: UomType,  // Bigger | Reference | Smaller
  }

  pub fn compute_quantity(
      from: &Uom, qty: f64, to: &Uom,
      round: bool, rounding_method: RoundingMethod,
      raise_if_failure: bool,
  ) -> Result<f64, UomError> {
      if from.category_id != to.category_id {
          if raise_if_failure { return Err(UomError::CategoryMismatch); }
          return Ok(qty);
      }
      let result = (qty / from.factor) * to.factor;
      if round {
          Ok(float_round(result, to.rounding, rounding_method))
      } else {
          Ok(result)
      }
  }

  pub fn compute_price(from: &Uom, price: f64, to: &Uom) -> f64 {
      if from.id == to.id || price == 0.0 { return price; }
      // category check implied; porter must add guard
      (price * from.factor) / to.factor
  }
  ```
- **Parity notes**:
  - `factor` is the ratio compared to the reference unit. Bigger UoMs have `factor < 1` (e.g. dozen: factor=1/12 ≈ 0.0833), smaller UoMs have `factor > 1`.
  - Wait — the formula `(qty / self.factor) * to_unit.factor` means: convert self→reference by dividing by self.factor, then reference→to_unit by multiplying by to_unit.factor. This is consistent if `reference.factor = 1.0`.
  - `_compute_price` inverts: `price * self.factor / to.factor` — price per from-unit → price per to-unit.
  - `rounding_method` defaults to `'UP'` in `_compute_quantity` (ceiling toward positive infinity), but `'HALF-UP'` is used in stock procurement (`_adjust_uom_quantities`). The porter must NOT silently normalise to HALF-UP everywhere.
  - Stock extension (`stock/models/product.py:L1344-1375`) blocks factor/relative_factor/relative_uom_id changes when open stock moves or non-zero quants exist — this is a hard integrity guard.
  - UNECE codes (EDI/e-invoice): `account/models/uom_uom.py` maps 26 standard UoM xml_ids to UNECE Rec-20 codes (C62, KGM, HUR, etc.) via `_get_unece_code()`. Default fallback: `'C62'` (unit).

---

### R6 — product.category hierarchy + account properties  [AXIS-A]

- **odoo source**: `product_category.py:L1-69` (base), `account/models/product.py:L13-29` (account extension)
- **What it does**:
  - Base: `parent_id` (Many2one, cascade), `parent_path` (materialized), `_check_category_recursion` (DFS cycle guard).
  - `complete_name` = recursive `parent.complete_name / name` breadcrumb.
  - Account extension adds two `company_dependent` Many2one fields:
    - `property_account_income_categ_id` → `account.account` (income account for customer invoices)
    - `property_account_expense_categ_id` → `account.account` (expense/COGS account for vendor bills)
  - Both use domain excluding `asset_receivable`, `liability_payable`, `asset_cash`, `liability_credit_card`, `off_balance`.
  - Stock extension adds: `route_ids`, `removal_strategy_id`, `putaway_rule_ids`, `packaging_reserve_method`.
- **woa-rs target**: Product category table (data foundation). Account properties feed K3 GL posting.
- **Rust sketch**:
  ```rust
  pub struct ProductCategory {
      pub id: CategoryId,
      pub name: String,
      pub parent_id: Option<CategoryId>,
      pub parent_path: String,   // materialized closure path "1/4/7/"
      // per-company, nullable:
      pub property_account_income_categ_id: Option<AccountId>,
      pub property_account_expense_categ_id: Option<AccountId>,
      // stock:
      pub removal_strategy_id: Option<RemovalStrategyId>,
  }
  // Constraint: no cycles in parent_id chain (_has_cycle check)
  ```
- **Parity notes**: `company_dependent` fields are stored in `ir.property` (or modern JSON column) keyed by `(model, field, company_id, res_id)`. In woa-rs these should be modelled as per-company overrides, not flat columns.

---

### R7 — Account resolution waterfall for product income/expense accounts  [AXIS-A]

- **odoo source**: `account/models/product.py:L67-97`
- **What it does**: `_get_product_accounts()` returns `{'income': ..., 'expense': ...}` following priority chain:
  1. `product.property_account_income_id` (product-level, company-dependent)
  2. Walk `categ_id → parent_id → ...` until account found (`_get_category_account`)
  3. `company.income_account_id` (company default)
  Same for expense. Then `get_product_accounts(fiscal_pos)` maps through fiscal position if provided.
- **woa-rs target**: GL account resolution when posting invoice lines (K3).
- **Rust sketch**:
  ```rust
  fn get_income_account(product: &ProductTemplate, company: &Company) -> AccountId {
      product.property_account_income_id
          .or_else(|| walk_category_account(&product.categ_id, "income"))
          .unwrap_or(company.income_account_id)
  }
  fn walk_category_account(categ: &ProductCategory, kind: &str) -> Option<AccountId> {
      let mut c = Some(categ);
      while let Some(cat) = c {
          let acc = if kind == "income" { cat.property_account_income_categ_id }
                    else { cat.property_account_expense_categ_id };
          if acc.is_some() { return acc; }
          c = cat.parent_id.as_ref();
      }
      None
  }
  ```
- **Parity notes**: The walk is unbounded — follows full parent chain. In Rust, guard against degenerate deep trees. Fiscal position mapping (`map_account`) is an additional substitution layer (handled in L3).

---

### R8 — Pricelist structure: currency, company, country-group scoping  [AXIS-A]

- **odoo source**: `product_pricelist.py:L9-65`
- **What it does**: `product.pricelist` has: `currency_id` (required, defaults to company currency), `company_id` (optional scoping), `country_group_ids` (M2M for geo-scoping), `item_ids` (the rules). `sequence` (int, default 16) determines selection priority when multiple pricelists apply. `active` boolean for soft-delete.
- **woa-rs target**: Pricelist table (Vorgang pricing).
- **Rust sketch**:
  ```rust
  pub struct Pricelist {
      pub id: PricelistId,
      pub name: String,
      pub currency_id: CurrencyId,
      pub company_id: Option<CompanyId>,
      pub country_group_ids: Vec<CountryGroupId>,
      pub sequence: i32,    // default 16; lower = higher priority
      pub active: bool,
  }
  ```
- **Parity notes**: Deleting a pricelist that is used as `base_pricelist_id` in another pricelist's rules is blocked (`_unlink_except_used_as_rule_base`). Recursion prevention via DFS is enforced on save.

---

### R9 — Pricelist item fields: applied_on, min_quantity, date validity  [AXIS-A]

- **odoo source**: `product_pricelist_item.py:L51-153`
- **What it does**:
  - `applied_on` ∈ `{3_global, 2_product_category, 1_product, 0_product_variant}` — scoping level.
  - Sort order: `applied_on ASC, min_quantity DESC, categ_id DESC, id DESC` — more specific rules sort first; higher min_quantity sorts first within same specificity.
  - `min_quantity` (Float, digits='Product Unit', default=0) — expressed in product's default UoM.
  - `date_start` / `date_end` (Datetime) — validity window; constraint: `date_start < date_end` if both set.
  - `compute_price` ∈ `{percentage, formula, fixed}`.
  - `base` ∈ `{list_price, standard_price, pricelist}`.
  - For `fixed`: `fixed_price` (Float).
  - For `percentage`: `percent_price` (Float, the discount %).
  - For `formula`: `price_discount` (Float, %), `price_round` (Float, rounding step), `price_surcharge` (Float, additive), `price_min_margin` (Float), `price_max_margin` (Float).
  - `price_markup` = `-price_discount` (computed inverse, used when `base == standard_price`).
- **woa-rs target**: Pricelist rule table.
- **Rust sketch**:
  ```rust
  pub enum AppliedOn { Global, ProductCategory, Product, ProductVariant }
  pub enum ComputePrice { Fixed, Percentage, Formula }
  pub enum RuleBase { ListPrice, StandardPrice, Pricelist }

  pub struct PricelistItem {
      pub pricelist_id: PricelistId,
      pub applied_on: AppliedOn,
      pub categ_id: Option<CategoryId>,
      pub product_tmpl_id: Option<ProductTemplateId>,
      pub product_id: Option<ProductVariantId>,
      pub min_quantity: f64,         // in product default UoM
      pub date_start: Option<DateTime<Utc>>,
      pub date_end: Option<DateTime<Utc>>,
      pub compute_price: ComputePrice,
      pub base: RuleBase,
      pub base_pricelist_id: Option<PricelistId>,  // only when base==Pricelist
      pub fixed_price: f64,
      pub percent_price: f64,
      pub price_discount: f64,
      pub price_round: f64,
      pub price_surcharge: f64,
      pub price_min_margin: f64,
      pub price_max_margin: f64,
  }
  // Constraint: min_margin <= max_margin (if both non-zero)
  // Constraint: date_start < date_end (if both set)
  // Constraint: base==Pricelist => base_pricelist_id must be set
  // Constraint: no pricelist graph cycles (DFS enforced on write)
  ```

---

### R10 — _get_applicable_rules_domain: rule filtering  [AXIS-A]

- **odoo source**: `product_pricelist.py:L239-264`
- **What it does**: Fetches ALL rules for a pricelist that could potentially apply to a set of products on a given date. Domain combines:
  1. `pricelist_id == self.id`
  2. `categ_id IS NULL OR categ_id parent_of product.categ_id` (category hierarchy)
  3. `product_tmpl_id IS NULL OR product_tmpl_id IN [template ids]`
  4. `product_id IS NULL OR product_id IN [variant ids]`
  5. `date_start IS NULL OR date_start <= date`
  6. `date_end IS NULL OR date_end >= date`
  Result is ordered by `applied_on ASC, min_quantity DESC, categ_id DESC, id DESC` (from `_order`).
- **woa-rs target**: Rule candidate selection query in pricing engine.
- **Rust sketch**:
  ```rust
  fn get_applicable_rules(
      pl: &Pricelist, products: &[Product], date: DateTime<Utc>
  ) -> Vec<PricelistItem> {
      // SQL query equivalent — filter pricelist items by:
      // pricelist_id = pl.id
      // AND (categ_id IS NULL OR categ_id is ancestor-of product.categ_id)
      // AND (tmpl_id IS NULL OR tmpl_id IN template_ids)
      // AND (variant_id IS NULL OR variant_id IN variant_ids)
      // AND (date_start IS NULL OR date_start <= date)
      // AND (date_end IS NULL OR date_end >= date)
      // ORDER BY applied_on ASC, min_quantity DESC, categ_id DESC, id DESC
  }
  ```
- **Parity notes**: `categ_id parent_of` uses Odoo's `parent_path` materialized closure. The Porter must replicate this via a prefix-check: `product.categ.parent_path.starts_with(rule.categ.parent_path)`.

---

### R11 — _is_applicable_for: rule applicability check  [AXIS-A]

- **odoo source**: `product_pricelist_item.py:L526-568`
- **What it does**: Fine-grained per-product applicability after candidate fetch:
  1. `min_quantity > 0 AND qty_in_product_uom < min_quantity` → False.
  2. `applied_on == 2_product_category`: check `product.categ_id == rule.categ_id OR product.categ.parent_path.startswith(rule.categ.parent_path)`.
  3. `applied_on == 1_product` (on template): `product.id == rule.product_tmpl_id`.
  4. `applied_on == 0_product_variant` (on template): only if template has exactly one variant AND that variant == rule.product_id.
  5. `applied_on == 1_product` (on variant): `product.product_tmpl_id == rule.product_tmpl_id`.
  6. `applied_on == 0_product_variant` (on variant): `product.id == rule.product_id`.
- **woa-rs target**: Inner loop of pricing engine.
- **Rust sketch**:
  ```rust
  fn is_applicable_for(rule: &PricelistItem, product: &Product, qty_in_product_uom: f64) -> bool {
      if rule.min_quantity > 0.0 && qty_in_product_uom < rule.min_quantity {
          return false;
      }
      match rule.applied_on {
          AppliedOn::Global => true,
          AppliedOn::ProductCategory => {
              product.categ.parent_path.starts_with(&rule.categ.parent_path)
          }
          AppliedOn::Product => product.product_tmpl_id == rule.product_tmpl_id.unwrap(),
          AppliedOn::ProductVariant => product.id == rule.product_id.unwrap(),
      }
  }
  ```
- **Parity notes**: `qty_in_product_uom` is already converted to product default UoM before this check (done in `_compute_price_rule`). The `min_quantity` field on items is always in product default UoM — do not apply UoM conversion a second time.

---

### R12 — _compute_price_rule: main pricing engine loop  [AXIS-A]

- **odoo source**: `product_pricelist.py:L169-236`
- **What it does**: Core method — mono-pricelist, multi-product. For each product:
  1. Determine `target_uom` (passed `uom` arg or product's own `uom_id`).
  2. If `target_uom != product_uom`: convert quantity to product UoM via `target_uom._compute_quantity(quantity, product_uom, raise_if_failure=False)` — for `min_quantity` comparison.
  3. Iterate `rules` (pre-fetched, sorted) in order; take FIRST rule where `_is_applicable_for` returns True.
  4. If no rule matches, `suitable_rule` = empty recordset.
  5. Call `suitable_rule._compute_price(product, quantity, target_uom, date, currency)`.
  6. Return `{product_id: (price, rule_id)}`.
- **woa-rs target**: Central pricing function called from sale order line, invoice line.
- **Rust sketch**:
  ```rust
  fn compute_price_rule(
      pl: &Pricelist, products: &[Product],
      quantity: f64, uom: Option<&Uom>, date: DateTime<Utc>, currency: &Currency,
  ) -> HashMap<ProductId, (f64, Option<PricelistItemId>)> {
      let rules = get_applicable_rules(pl, products, date);
      let mut results = HashMap::new();
      for product in products {
          let product_uom = &product.uom_id;
          let target_uom = uom.unwrap_or(product_uom);
          let qty_in_product_uom = if target_uom != product_uom {
              compute_quantity(target_uom, quantity, product_uom, false, RoundingMethod::Up, false)
                  .unwrap_or(quantity)
          } else { quantity };
          let suitable_rule = rules.iter()
              .find(|r| is_applicable_for(r, product, qty_in_product_uom));
          let price = match suitable_rule {
              Some(rule) => compute_price_from_rule(rule, product, quantity, target_uom, date, currency),
              None => compute_base_price_no_rule(product, target_uom, date, currency),
          };
          results.insert(product.id, (price, suitable_rule.map(|r| r.id)));
      }
      results
  }
  ```
- **Parity notes**: When no rule matches, `suitable_rule._compute_price` is called on an EMPTY recordset. The `else` branch at `product_pricelist_item.py:L623-624` handles this: `price = self._compute_base_price(...)` where `self` is empty, which uses `list_price` as base. This is a subtle fallback — the porter must handle the empty-rule case by returning the base list/cost price in the target UoM/currency.

---

### R13 — _compute_price (item): fixed / percentage / formula branches  [AXIS-A]

- **odoo source**: `product_pricelist_item.py:L570-626`
- **What it does**: Three-way branch on `compute_price`:

  **A. `fixed`**:
  ```
  price = uom_convert(self.fixed_price, from=product_uom, to=target_uom)
  ```
  Uses `product_uom._compute_price(fixed_price, target_uom)` to convert the stored fixed price from product UoM to requested UoM.

  **B. `percentage`**:
  ```
  base_price = _compute_base_price(...)
  price = base_price - (base_price * (percent_price / 100))
  # i.e. percent_price=20 means 20% discount → price = 80% of base
  # percent_price can be negative (markup)
  price = price or 0.0   # coerce -0.0 to 0.0
  ```

  **C. `formula`** (most complex):
  ```
  base_price = _compute_base_price(...)
  price_limit = base_price   # saved for margin clamps
  discount = price_discount if base != standard_price else -price_markup
  price = base_price - (base_price * (discount / 100))
  if price_round:
      price = float_round(price, precision_rounding=price_round)
  if price_surcharge:
      price += uom_convert(price_surcharge, product_uom, target_uom)
  if price_min_margin:
      price = max(price, price_limit + uom_convert(price_min_margin, product_uom, target_uom))
  if price_max_margin:
      price = min(price, price_limit + uom_convert(price_max_margin, product_uom, target_uom))
  ```
  Note: `price_markup = -price_discount` (they are inverses); when `base == standard_price` the discount field label changes to "Markup" but the formula uses `-price_markup` so the arithmetic is the same sign convention.

- **woa-rs target**: Pricing computation kernel.
- **Rust sketch**:
  ```rust
  fn compute_price_from_rule(
      rule: &PricelistItem, product: &Product, qty: f64,
      target_uom: &Uom, date: DateTime<Utc>, currency: &Currency,
  ) -> f64 {
      let product_uom = &product.uom_id;
      let uom_cvt = |p: f64| uom_compute_price(product_uom, p, target_uom);

      match rule.compute_price {
          ComputePrice::Fixed => uom_cvt(rule.fixed_price),
          ComputePrice::Percentage => {
              let base = compute_base_price(rule, product, qty, target_uom, date, currency);
              let price = base - base * (rule.percent_price / 100.0);
              if price == -0.0 { 0.0 } else { price }
          }
          ComputePrice::Formula => {
              let base = compute_base_price(rule, product, qty, target_uom, date, currency);
              let price_limit = base;
              let discount = if rule.base == RuleBase::StandardPrice {
                  -rule.price_markup
              } else {
                  rule.price_discount
              };
              let mut price = base - base * (discount / 100.0);
              if rule.price_round > 0.0 {
                  price = float_round(price, rule.price_round, RoundingMethod::HalfUp);
                  // NOTE: odoo uses float_round with precision_rounding=price_round
                  // default rounding_method for float_round is 'HALF-UP'
              }
              if rule.price_surcharge != 0.0 {
                  price += uom_cvt(rule.price_surcharge);
              }
              if rule.price_min_margin != 0.0 {
                  price = price.max(price_limit + uom_cvt(rule.price_min_margin));
              }
              if rule.price_max_margin != 0.0 {
                  price = price.min(price_limit + uom_cvt(rule.price_max_margin));
              }
              price
          }
      }
  }
  ```
- **Parity notes**:
  - `price_surcharge` and margin fields are UoM-converted (they are stored in product UoM units).
  - `price_round` in `float_round` is `precision_rounding` (step size, e.g. 0.05 means round to nearest 5 cents), NOT decimal digits. To get 9.99-style prices: `price_round=10.0, price_surcharge=-0.01`.
  - The margin clamp: `price_min_margin` is added to `base_price` (not to zero) — it is a minimum MARGIN over base, not a minimum absolute price.
  - When `base == standard_price`: the label is "Markup" and `price_discount` stores the negative of the markup (`price_markup = -price_discount`). But the formula uses `discount = -price_markup = price_discount` effectively. Symmetric.

---

### R14 — _compute_base_price: base resolution + currency conversion  [AXIS-A]

- **odoo source**: `product_pricelist_item.py:L628-659`
- **What it does**: Resolves base price for percentage/formula rules:
  1. `base == 'pricelist'`: recursively call `base_pricelist_id._get_product_price(...)` using `base_pricelist_id.currency_id` as src currency.
  2. `base == 'standard_price'`: fetch `product._price_compute('standard_price', uom, date)` using `product.cost_currency_id` as src.
  3. `base == 'list_price'`: fetch `product._price_compute('list_price', uom, date)` using `product.currency_id` as src.
  4. If `src_currency != currency` (the target pricelist currency): convert via `src_currency._convert(price, currency, company, date, round=False)` — no rounding at this step.
- **woa-rs target**: Base price resolver in pricing kernel.
- **Rust sketch**:
  ```rust
  fn compute_base_price(
      rule: &PricelistItem, product: &Product, qty: f64,
      target_uom: &Uom, date: DateTime<Utc>, currency: &Currency,
  ) -> f64 {
      let (price, src_currency) = match rule.base {
          RuleBase::Pricelist => {
              let pl = rule.base_pricelist_id.unwrap();
              let p = pl.get_product_price(product, qty, Some(&pl.currency), Some(target_uom), date);
              (p, &pl.currency)
          }
          RuleBase::StandardPrice => {
              let prices = product.price_compute("standard_price", Some(target_uom), None, None, date);
              (prices[product.id], &product.cost_currency)
          }
          RuleBase::ListPrice => {
              let prices = product.price_compute("list_price", Some(target_uom), None, None, date);
              (prices[product.id], &product.currency)
          }
      };
      if src_currency != currency {
          currency_convert(price, src_currency, currency, date, /*round=*/false)
      } else {
          price
      }
  }
  ```
- **Parity notes**: Pricelist chaining (`base == 'pricelist'`) fetches the base pricelist price using `base_pricelist_id.currency_id` explicitly — it does NOT pass the outer pricelist's currency to the inner call. Currency conversion happens AFTER the inner call returns. A recursion guard (cycle detection) exists at the constraint level but not at runtime — cycles can't exist if `_check_pricelist_recursion` is enforced.

---

### R15 — Partner pricelist assignment: country-group → fallback waterfall  [AXIS-B / HYBRID]

- **odoo source**: `product_pricelist.py:L333-384`
- **What it does**: `_get_partner_pricelist_multi(partner_ids)` — determines which pricelist applies to each partner:
  1. If `group_product_pricelist` feature disabled → return empty pricelist for all.
  2. For each partner: check `specific_property_product_pricelist` (explicit property set on partner form). If active → use it.
  3. Remaining partners: group by `country_id`. For each country, find pricelist with matching `country_group_ids.country_ids`.
  4. Fallback waterfall (`_get_country_pricelist_multi`):
     a. Search pricelist with `country_group_ids = False` (no geo restriction) + active + company filter.
     b. `ir.config_parameter` `res.partner.property_product_pricelist_{company_id}`.
     c. `ir.config_parameter` `res.partner.property_product_pricelist` (global default).
     d. Any active pricelist (`search(pl_domain, limit=1)`).
  - **AXIS-A part**: The lookup steps 1-4a are deterministic lookups — pure data retrieval.
  - **AXIS-B part**: When no explicit property and no country match exists, the fallback is multi-factor: country group config, company config param, global param, first-available. This is heuristic assignment, not a closed formula. The "right" pricelist for a new partner in an edge case is a business judgment.

- **woa-rs target**: Partner onboarding / sale order price selection.
- **Rust sketch (AXIS-A part)**:
  ```rust
  fn resolve_partner_pricelist(partner: &Partner, company: &Company) -> Option<PricelistId> {
      // 1. Feature guard
      if !feature_enabled("group_product_pricelist") { return None; }
      // 2. Explicit property
      if let Some(pl) = partner.specific_pricelist_id.filter(|pl| pl.active) {
          return Some(pl.id);
      }
      // 3. Country group match
      if let Some(country) = &partner.country_id {
          if let Some(pl) = find_pricelist_for_country(country, company) {
              return Some(pl.id);
          }
      }
      // 4. Fallback chain (heuristic — see SAVANT seed)
      find_fallback_pricelist(company)
  }
  ```
- **Delegation tuple (AXIS-B — fallback resolution)**:
  `ReasoningKind=Other("PricelistAssignment")` `InferenceType=Deduction` (it's a lookup chain, but the ordering of fallbacks is a policy choice) — actually `InferenceType=Revision` (when partner data changes — country, segment — the pricelist should be re-evaluated against business rules). `SemiringChoice=NarsTruth` (evidence from country + company config + segment). `ThinkingStyle=Analytical` (inherited from expected `0x63 ProductCatalog` family — if unmapped, proposed Analytical as the catalog assignment domain is rule-based not creative).

`SAVANT: name=PricelistAssignmentAgent family=None(needs_0x63_ProductCatalog) reasoning=Other("PricelistAssignment") inference=Revision semiring=NarsTruth style=Analytical — fallback pricelist chain (no country match, no explicit property) requires business-policy judgment not deterministic lookup`

---

### R16 — _price_compute (template + variant): UoM + currency conversion gate  [AXIS-A]

- **odoo source**: `product_template.py:L737-768`, `product_product.py:L1101-1131`
- **What it does**: Returns `{product_id: float}` for a set of products. On template:
  1. Fetch raw price from field (`list_price` or `standard_price`).
  2. For `list_price`: add `_get_attributes_extra_price()` (context key `current_attributes_price_extra` sum).
  3. For `standard_price`: fallback to first variant's price if template price is 0.
  4. If `uom` arg: convert via `template.uom_id._compute_price(price, uom)`.
  5. If `currency` arg: convert via `price_currency._convert(price, currency, company, date)`.
  On variant: same but adds `no_variant_attributes_price_extra` from context key `no_variant_attributes_price_extra`.
- **woa-rs target**: Price extraction utility called by pricelist base resolution.
- **Parity notes**: `_get_attributes_extra_price` reads from `env.context` (template version) or sums `ptav.price_extra` (variant version). The context key `current_attributes_price_extra` is a tuple of floats injected by the configurator when a partial combination is being priced. The porter must handle this context injection pattern.

---

### R17 — cost_method / property_valuation: MISSING (Enterprise gap)  [AXIS-A — Enterprise gap]

- **odoo source**: NOT PRESENT in community clone. Expected location: `stock_account` module or `stock/models/product.py` (not found). Fields `cost_method` ∈ `{standard, average, fifo}` and `property_valuation` ∈ `{manual_periodic, real_time}` live in `stock_account` which requires Enterprise or `account+stock` merged module.
- **woa-rs target**: K3 inventory valuation + K13 costing base.
- **Spec from structure**: Based on odoo documentation and the `_run_fifo` / `_run_average` references in L13's scope:
  - `cost_method` on `product.category` (company-dependent).
  - `standard`: cost is `standard_price`, fixed until manually changed.
  - `average` (AVCO): `standard_price` updated on each receipt: `new_cost = (qty_on_hand * old_cost + qty_received * unit_cost) / (qty_on_hand + qty_received)`.
  - `fifo`: cost pulled from receipt layers (oldest first); `standard_price` reflects last layer cost.
  - `property_valuation`: `manual_periodic` (periodic inventory; no automatic GL on move) vs `real_time` (perpetual; GL entry on every stock move).
- **Enterprise gaps flagged**: See section below.

---

### R18 — Variant creation / combination matrix  [AXIS-A]

- **odoo source**: `product_template.py:L770-868`
- **What it does**: `_create_variant_ids()` — after attribute line changes, recomputes the Cartesian product of attribute values. For each possible combination:
  - If variant exists → activate.
  - If variant doesn't exist → create.
  - Variants no longer in any possible combination → unlink or archive (`_unlink_or_archive`).
  - Dynamic attributes (`create_variant='dynamic'`) skip full matrix creation; variants created on demand.
  - Hard limit: `ir.config_parameter product.dynamic_variant_limit` (default 1000) variants per template.
- **woa-rs target**: Product variant management.
- **Parity notes**: The combination-index dedup key is `combination_indices = ",".join(sorted ptav ids)` stored on `product.product`. Unique index enforced: `(product_tmpl_id, combination_indices) WHERE active IS TRUE`.

---

### R19 — Pricelist recursion guard (DFS cycle detection)  [AXIS-A]

- **odoo source**: `product_pricelist_item.py:L321-353`
- **What it does**: On create/write of pricelist items with `base='pricelist'`, DFS traversal starting from `base_pricelist_id` following chains of `base='pricelist'` items. If `pricelist_id` (the owning pricelist) appears in the traversal path → `ValidationError`.
- **woa-rs target**: Integrity constraint on pricelist write.
- **Rust sketch**: Topological sort or DFS on write; reject if cycle detected. O(V+E) where V=pricelists, E=pricelist-based rules.

---

### R20 — UoM change guard on posted invoices  [AXIS-A]

- **odoo source**: `account/models/product.py:L130-149`
- **What it does**: `@api.constrains('uom_id')` on `product.template` — runs a SQL query checking if any `account_move_line` in `posted` state references this product with a DIFFERENT UoM. If found → `ValidationError`. Prevents silent unit-of-measure drift on locked accounting documents.
- **woa-rs target**: Constraint when updating product UoM after invoices posted.
- **Parity notes**: This is a cross-model constraint touching `account_move_line`. The porter must implement a validation hook on product UoM write that checks invoice lines.

---

### R21 — UoM factor immutability guard (stock moves / quants)  [AXIS-A]

- **odoo source**: `stock/models/product.py:L1344-1375`
- **What it does**: `write()` override on `uom.uom` — blocks changes to `factor`, `relative_factor`, `relative_uom_id` if:
  - Any `stock.move` in non-terminal state (`not in {cancel, done}`) uses this UoM.
  - Any `stock.move.line` in non-terminal state uses this UoM.
  - Any `stock.quant` with `quantity != 0` uses a product whose template's `uom_id` is this UoM.
- **woa-rs target**: UoM integrity constraint in inventory.
- **Parity notes**: This is a write-time guard, not a DB constraint. Must be implemented as a pre-write validator in woa-rs.

---

### R22 — Contextual price / pricelist resolution (template)  [AXIS-A]

- **odoo source**: `product_template.py:L1534-1550`
- **What it does**: `_get_contextual_price(product)` reads from `env.context`:
  - `pricelist` (int id) → resolves pricelist.
  - `quantity` (float, default 1.0).
  - `uom` (int id, optional).
  - `date` (optional).
  Then calls `pricelist._get_product_price(product, quantity, uom, date)`.
  `_get_contextual_pricelist()` simply reads `env.context.get('pricelist')`.
- **woa-rs target**: UI/API entry point for contextual price display.
- **Parity notes**: In woa-rs this context pattern should be replaced with explicit function parameters rather than implicit context bag — cleaner for the Rust type system.

---

### R23 — `_get_tax_included_unit_price`: UoM + fiscal-pos + currency normalisation  [AXIS-A / HYBRID]

- **odoo source**: `account/models/product.py:L222-293`
- **What it does**: Helper to get price unit for invoicing, combining:
  1. Resolve product price (`lst_price` for sale, `standard_price` for purchase).
  2. Apply UoM conversion if `product_uom != product.uom_id`.
  3. Apply fiscal position tax adaptation (`_adapt_price_unit_to_another_taxes`) if fiscal_pos given — adjusts price for included-tax differences.
  4. Apply currency conversion (no rounding at this step).
- **AXIS-A part**: Steps 1-2 and 4 are deterministic.
- **AXIS-B part**: Step 3 (fiscal position application) — choosing the right fiscal position is heuristic (handled in L3). But once chosen, adapting the price is deterministic math.
- **woa-rs target**: Invoice line unit price computation (K3 + K7).

---

## Enterprise gaps flagged

| Module | What's missing | What we spec from community data/structure |
|---|---|---|
| `stock_account` | `cost_method` ∈ `{standard, average, fifo}` on `product.category`; `property_valuation` ∈ `{manual_periodic, real_time}`; `_run_fifo`, `_run_average` logic; SVL (stock valuation layer) creation on stock moves | R17 above: fields documented from public Odoo docs; AVCO/FIFO engine built fresh in woa-rs (L13 lane handles SVL) |
| `product_margin` | Margin computation on sale order lines | Not present; woa-rs can compute from `(price - standard_price) / price` inline |
| `account_accountant` | `_predict_specific_product` (ML-based product prediction on invoice import) — referenced at `account/models/product.py:L357-371` | Flag: AXIS-B candidate if EDI import is implemented; skip for now |
| `uom` (core module) | `uom_uom.py` core model with `_compute_quantity`, `_compute_price`, `factor`, `rounding` | Reconstructed in R5 from call-sites + GitHub raw fetch; porter should verify against actual source |

---

## Open questions for the Opus porter

1. **UoM factor direction**: The formula `(qty / from.factor) * to.factor` implies reference unit has `factor=1.0` and bigger UoMs have `factor < 1` (e.g., dozen ≈ 0.0833). Verify against actual `uom` module data seeds before implementing. The alternative reading (`factor` = how many base units in this UoM, so dozen=12) would invert the formula.

2. **`company_dependent` fields**: Odoo stores these via `ir.property` (or a new JSON-based mechanism in v17). In woa-rs, model as a separate `product_company_properties` junction table keyed by `(company_id, product_id)` with typed columns, rather than a generic key-value store.

3. **Pricelist feature flag**: `group_product_pricelist` disables the entire pricelist system when off. woa-rs should support a `pricing_mode: Simple | Pricelist` enum at the company level.

4. **`ir.config_parameter` for pricelist defaults**: The partner pricelist fallback uses `ir.config_parameter` keys `res.partner.property_product_pricelist_{company_id}` and `res.partner.property_product_pricelist`. Map these to a typed `CompanySettings` struct field in woa-rs.

5. **`price_round` rounding method**: `float_round(price, precision_rounding=price_round)` — what rounding method does Odoo's `float_round` default to when only `precision_rounding` is given? From the `_compute_rule_tip` code this appears to be `HALF-UP` (Python `round()` semantics). Verify in odoo source `tools/float_utils.py`.

6. **Combo product pricing**: `type='combo'` disables taxes and supplier taxes (`_onchange_type` in account extension). Pricing of combo products (sum of item prices with discounts) is not detailed in the community source — likely handled in `pos` or `sale` module extensions.

7. **`no_variant_attributes_price_extra`**: Context-injected extra for no-variant attributes (attributes that affect price but don't create variants). The configurator injects this; the pricelist engine does NOT include it by default. Clarify whether woa-rs needs to handle this for the product configurator flow.

8. **`_compute_price_before_discount`** (`product_pricelist_item.py:L661-684`): Used when the pricelist is configured to "show discount to customer" — walks the pricelist chain to find the lowest rule whose pricelist shows discount, then returns the base price at that level (so the "before discount" price shown on the order reflects the underlying list price). This is a display feature; the porter should flag it as needed only if the UX shows strikethrough pricing.

---

## Depth-proof footer

```
Read: /home/user/woa-rs/.claude/odoo/BRIEFING.md lines=166 depth=full
Read: /home/user/woa-rs/.claude/odoo/BRIEFING-GAP.md lines=82 depth=full
Read: /home/user/odoo/addons/product/models/product_template.py lines=1598 depth=full (4 chunks)
Read: /home/user/odoo/addons/product/models/product_product.py lines=1197 depth=full (3 chunks)
Read: /home/user/odoo/addons/product/models/product_category.py lines=69 depth=full
Read: /home/user/odoo/addons/product/models/product_pricelist.py lines=415 depth=full
Read: /home/user/odoo/addons/product/models/product_pricelist_item.py lines=684 depth=full
Read: /home/user/odoo/addons/product/models/uom_uom.py lines=30 depth=full (product extension stub only)
Read: /home/user/odoo/addons/account/models/uom_uom.py lines=59 depth=full (account extension)
Read: /home/user/odoo/addons/account/models/product.py lines=523 depth=full
Read: /home/user/odoo/addons/stock/models/product.py lines=1389 depth=partial (L1274-1389 for ProductCategory+UomUom sections; remainder is stock product/template which is L7 territory)
WebFetch: https://raw.githubusercontent.com/odoo/odoo/17.0/addons/uom/models/uom_uom.py depth=full (uom core module absent from clone; reconstructed via WebFetch)
```
