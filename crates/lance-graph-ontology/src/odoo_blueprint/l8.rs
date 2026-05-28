//! Lane L8 (PRODUCT-UOM-PRICELIST) — typed Odoo entity declarations.
//!
//! Source: `.claude/odoo/L8-PRODUCT-UOM-PRICELIST.md`.
//!
//! Entities: `product.category`, `uom.uom`, `product.template`,
//! `product.product`, `product.pricelist`, `product.pricelist.item`.
//!
//! ## L6 overlap note
//! `product.pricelist` and `product.pricelist.item` may overlap with L6
//! (partner/pricing configuration). L8 is declared authoritative for the
//! pricelist data model; L6 covers the partner-assignment heuristics
//! (`_get_partner_pricelist_multi` AXIS-B dispatch → `PricelistAssignmentAgent`).
//!
//! ## Ontology gap
//! All seven L8 classes resolve to `None` from `resolve_odoo_to_family()`.
//! Proposed mapping: `product.*` → `schema:Product` → new OGIT family
//! `0x63 ProductCatalog` (Analytical style). See L8-PRODUCT-UOM-PRICELIST.md §
//! "Ontology rows".

use super::{
    OdooConfidence, OdooConstraint, OdooConstraintKind, OdooDecorator, OdooDecoratorKind,
    OdooEntity, OdooField, OdooFieldKind, OdooMethod, OdooMethodKind, OdooProvenance,
    OdooReturnKind, OdooSemanticRole, OdooSourceRef,
};

// ─── product.category ─────────────────────────────────────────────────────────

pub const PRODUCT_CATEGORY: OdooEntity = OdooEntity {
    model_name: "product.category",
    description: "Hierarchical product category; carries company-dependent income/expense \
                  GL account properties and forms the ancestor chain used in pricelist \
                  rule applicability checks (parent_path prefix match).",
    fields: &[
        OdooField {
            name: "name",
            kind: OdooFieldKind::Char,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Identity,
        },
        // Materialized breadcrumb: "All / Consumable / Electronic" — recursive concat.
        OdooField {
            name: "complete_name",
            kind: OdooFieldKind::Computed,
            target: None,
            required: false,
            computed: Some("_compute_complete_name"),
            depends: &["name", "parent_id.complete_name"],
            semantic_role: OdooSemanticRole::Identity,
        },
        // Cascade delete; DFS cycle guard via _check_category_recursion.
        OdooField {
            name: "parent_id",
            kind: OdooFieldKind::Many2one,
            target: Some("product.category"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        // Materialized closure path e.g. "1/4/7/" — used in pricelist categ ancestor check.
        OdooField {
            name: "parent_path",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "child_id",
            kind: OdooFieldKind::One2many,
            target: Some("product.category"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        // company_dependent Many2one → account.account; income account for customer invoices.
        // Account resolution waterfall (R7): product → categ walk → company default.
        OdooField {
            name: "property_account_income_categ_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.account"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Tax,
        },
        // company_dependent; expense/COGS account for vendor bills.
        OdooField {
            name: "property_account_expense_categ_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.account"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Tax,
        },
    ],
    methods: &[
        OdooMethod {
            name: "_check_category_recursion",
            kind: OdooMethodKind::Constrain,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        // Account resolution waterfall: product property → categ walk → company default.
        // Called when posting invoice lines (K3). L8 captures the data-model side;
        // fiscal position mapping is handled in L3.
        OdooMethod {
            name: "_get_product_accounts",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiConstrains,
            targets: &["parent_id"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["name", "parent_id.complete_name"],
        },
    ],
    state_machine: None,
    constraints: &[
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "No cycles in parent_id chain (DFS _check_category_recursion)",
            source_method: Some("_check_category_recursion"),
        },
    ],
    provenance: OdooProvenance {
        l_doc: "L8-PRODUCT-UOM-PRICELIST.md",
        l_doc_lines: (180, 238),
        odoo_source: &[
            OdooSourceRef {
                path: "addons/product/models/product_category.py",
                line_range: (1, 69),
            },
            OdooSourceRef {
                path: "addons/account/models/product.py",
                line_range: (13, 29),
            },
        ],
        confidence: OdooConfidence::Curated,
    },
};

// ─── uom.uom ──────────────────────────────────────────────────────────────────

pub const UOM_UOM: OdooEntity = OdooEntity {
    model_name: "uom.uom",
    description: "Unit of Measure: factor-based conversion engine. Each UoM belongs to a \
                  category with exactly one reference unit (factor=1.0). \
                  _compute_quantity and _compute_price are the core conversion primitives \
                  called throughout pricing, stock, and invoicing.",
    fields: &[
        OdooField {
            name: "name",
            kind: OdooFieldKind::Char,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Identity,
        },
        OdooField {
            name: "category_id",
            kind: OdooFieldKind::Many2one,
            target: Some("uom.category"),
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        // Conversion ratio vs reference unit. Reference unit has factor=1.0.
        // Bigger UoMs: factor<1 (dozen≈0.0833); smaller: factor>1.
        // SQL constraint: factor != 0.
        OdooField {
            name: "factor",
            kind: OdooFieldKind::Float,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Quantity,
        },
        // Computed = 1/factor; stored for convenience.
        OdooField {
            name: "factor_inv",
            kind: OdooFieldKind::Computed,
            target: None,
            required: false,
            computed: Some("_compute_factor_inv"),
            depends: &["factor"],
            semantic_role: OdooSemanticRole::Quantity,
        },
        // Precision step (e.g. 0.01, 1.0, 0.001). SQL constraint: rounding > 0.
        OdooField {
            name: "rounding",
            kind: OdooFieldKind::Float,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Quantity,
        },
        // bigger | reference | smaller. Each category has exactly one 'reference'.
        OdooField {
            name: "uom_type",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "active",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Status,
        },
    ],
    methods: &[
        // Core quantity conversion (R5):
        //   result = (qty / self.factor) * to_unit.factor
        //   If round=True: float_round(result, to_unit.rounding, rounding_method)
        //   rounding_method defaults 'UP'; stock uses 'HALF-UP'.
        //   Raises UomError::CategoryMismatch if categories differ (raise_if_failure=True).
        OdooMethod {
            name: "_compute_quantity",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Number,
            triggers: &[],
        },
        // Price conversion (R5): price * self.factor / to_unit.factor.
        // No rounding; prices carry full float until currency rounds.
        // Called from lst_price compute (R2) and pricelist fixed-price (R13).
        OdooMethod {
            name: "_compute_price",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Number,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_factor_inv",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        // Stock guard (R21): blocks factor/relative_factor/relative_uom_id changes
        // when open stock.move or non-zero stock.quant uses this UoM.
        OdooMethod {
            name: "write",
            kind: OdooMethodKind::Override,
            return_kind: OdooReturnKind::Boolean,
            triggers: &[],
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["factor"],
        },
    ],
    state_machine: None,
    constraints: &[
        OdooConstraint {
            kind: OdooConstraintKind::Sql,
            condition: "factor != 0",
            source_method: None,
        },
        OdooConstraint {
            kind: OdooConstraintKind::Sql,
            condition: "rounding > 0",
            source_method: None,
        },
        // Enforced at data-seed level, not via Python constraint in community source.
        OdooConstraint {
            kind: OdooConstraintKind::Domain,
            condition: "Each uom.category has exactly one uom_type='reference' with factor=1.0",
            source_method: None,
        },
        // Write-time guard — not a DB constraint.
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "Cannot change factor/relative_factor/relative_uom_id when open \
                        stock.move or non-zero stock.quant references this UoM",
            source_method: Some("write"),
        },
    ],
    provenance: OdooProvenance {
        l_doc: "L8-PRODUCT-UOM-PRICELIST.md",
        l_doc_lines: (119, 177),
        odoo_source: &[
            OdooSourceRef {
                // Core uom module absent from clone; reconstructed via GitHub WebFetch.
                path: "addons/uom/models/uom_uom.py",
                line_range: (1, 200),
            },
            OdooSourceRef {
                path: "addons/stock/models/product.py",
                line_range: (1344, 1375),
            },
        ],
        confidence: OdooConfidence::Curated,
    },
};

// ─── product.template ─────────────────────────────────────────────────────────

pub const PRODUCT_TEMPLATE: OdooEntity = OdooEntity {
    model_name: "product.template",
    description: "Product template: the canonical catalog record. Drives variant generation \
                  via attribute_line_ids (Cartesian product). Holds list_price, standard_price \
                  (company-dependent), currency_id, and the UoM. The pricing chain \
                  (_price_compute, _get_contextual_price) resolves UoM and currency before \
                  the pricelist engine applies rules.",
    fields: &[
        OdooField {
            name: "name",
            kind: OdooFieldKind::Char,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Identity,
        },
        // Enum: consu | service | combo.
        // combo type: combo_ids required, attribute_line_ids forbidden.
        OdooField {
            name: "type",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "active",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Status,
        },
        // Base sale price per default UoM; + price_extra for variants = lst_price.
        OdooField {
            name: "list_price",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Money,
        },
        // Cost per variant; company_dependent. Used as pricelist base 'standard_price'.
        // constraint: standard_price >= 0.
        OdooField {
            name: "standard_price",
            kind: OdooFieldKind::Computed,
            target: None,
            required: false,
            computed: Some("_compute_template_field_from_variant_field"),
            depends: &["product_variant_ids", "product_variant_ids.standard_price"],
            semantic_role: OdooSemanticRole::Money,
        },
        // = company_id.currency_id OR main_company.currency_id (computed, not stored).
        OdooField {
            name: "currency_id",
            kind: OdooFieldKind::Computed,
            target: Some("res.currency"),
            required: false,
            computed: Some("_compute_currency_id"),
            depends: &["company_id"],
            semantic_role: OdooSemanticRole::Money,
        },
        // = env.company.currency_id (context-sensitive, depends_context('company')).
        OdooField {
            name: "cost_currency_id",
            kind: OdooFieldKind::Computed,
            target: Some("res.currency"),
            required: false,
            computed: Some("_compute_cost_currency_id"),
            depends: &[],
            semantic_role: OdooSemanticRole::Money,
        },
        OdooField {
            name: "uom_id",
            kind: OdooFieldKind::Many2one,
            target: Some("uom.uom"),
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "uom_po_id",
            kind: OdooFieldKind::Many2one,
            target: Some("uom.uom"),
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "categ_id",
            kind: OdooFieldKind::Many2one,
            target: Some("product.category"),
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "company_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.company"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "sale_ok",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
        // stored+recomputable; base impl is no-op; overridden by purchase module.
        OdooField {
            name: "purchase_ok",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
        // Drives variant matrix (Cartesian product of attribute values).
        OdooField {
            name: "attribute_line_ids",
            kind: OdooFieldKind::One2many,
            target: Some("product.template.attribute.line"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "product_variant_ids",
            kind: OdooFieldKind::One2many,
            target: Some("product.product"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        // Non-empty only when type='combo'.
        OdooField {
            name: "combo_ids",
            kind: OdooFieldKind::Many2many,
            target: Some("product.combo"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        // Optional product-level income account override (company_dependent).
        OdooField {
            name: "property_account_income_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.account"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Tax,
        },
        OdooField {
            name: "property_account_expense_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.account"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Tax,
        },
    ],
    methods: &[
        // Returns {product_id: float} for a set of products.
        // Steps: raw price → attributes_extra_price → UoM convert → currency convert.
        // Called by pricelist base resolution (R14, R16).
        OdooMethod {
            name: "_price_compute",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        // Reads env.context pricelist/quantity/uom/date; calls pricelist._get_product_price.
        // Entry point for UI contextual price display (R22).
        OdooMethod {
            name: "_get_contextual_price",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Number,
            triggers: &[],
        },
        OdooMethod {
            name: "_get_contextual_pricelist",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Record,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_currency_id",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_cost_currency_id",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_template_field_from_variant_field",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        // (R18) Recomputes Cartesian product of attribute values after attribute line changes.
        // Activates existing variants, creates missing, archives/unlinks removed.
        // Dynamic attributes (create_variant='dynamic') skip full matrix.
        // Limit: ir.config_parameter product.dynamic_variant_limit (default 1000).
        OdooMethod {
            name: "_create_variant_ids",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        // Account resolution waterfall (R7): product property → categ walk → company default.
        OdooMethod {
            name: "_get_product_accounts",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        // (R23) Invoice line unit price: resolves lst_price/standard_price → UoM convert
        // → fiscal-pos tax adaptation → currency convert (no rounding).
        OdooMethod {
            name: "_get_tax_included_unit_price",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Number,
            triggers: &[],
        },
        // type='combo' → combo_ids required; attribute_line_ids forbidden (R1).
        OdooMethod {
            name: "_check_combo_constraints",
            kind: OdooMethodKind::Constrain,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        // (R20) Cross-model: blocks uom_id change if posted account_move_line exists
        // with a different UoM for this product.
        OdooMethod {
            name: "_check_uom_id",
            kind: OdooMethodKind::Constrain,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["company_id"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["product_variant_ids", "product_variant_ids.standard_price"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiConstrains,
            targets: &["uom_id"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiConstrains,
            targets: &["type", "attribute_line_ids", "combo_ids"],
        },
    ],
    state_machine: None,
    constraints: &[
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "type='combo' requires combo_ids.len() >= 1 and attribute_line_ids.is_empty()",
            source_method: Some("_check_combo_constraints"),
        },
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "Cannot change uom_id when a posted account_move_line with a different \
                        UoM references this product",
            source_method: Some("_check_uom_id"),
        },
    ],
    provenance: OdooProvenance {
        l_doc: "L8-PRODUCT-UOM-PRICELIST.md",
        l_doc_lines: (42, 115),
        odoo_source: &[
            OdooSourceRef {
                path: "addons/product/models/product_template.py",
                line_range: (1, 1598),
            },
            OdooSourceRef {
                path: "addons/account/models/product.py",
                line_range: (67, 293),
            },
        ],
        confidence: OdooConfidence::Curated,
    },
};

// ─── product.product ──────────────────────────────────────────────────────────

pub const PRODUCT_PRODUCT: OdooEntity = OdooEntity {
    model_name: "product.product",
    description: "Product variant: _inherits product.template. Adds variant-specific pricing \
                  (lst_price = list_price + price_extra + optional UoM conversion), \
                  company-dependent standard_price (cost), and the combination_indices \
                  dedup key that enforces uniqueness within the variant matrix.",
    fields: &[
        OdooField {
            name: "product_tmpl_id",
            kind: OdooFieldKind::Many2one,
            target: Some("product.template"),
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        // Variant sale price = template.list_price + price_extra + optional UoM convert (R2).
        // Setting lst_price reverses via _set_product_lst_price.
        OdooField {
            name: "lst_price",
            kind: OdooFieldKind::Computed,
            target: None,
            required: false,
            computed: Some("_compute_product_lst_price"),
            depends: &["list_price", "price_extra"],
            semantic_role: OdooSemanticRole::Money,
        },
        // Sum of product_template_attribute_value.price_extra for this variant's PTAVs.
        OdooField {
            name: "price_extra",
            kind: OdooFieldKind::Computed,
            target: None,
            required: false,
            computed: Some("_compute_product_price_extra"),
            depends: &["product_template_attribute_value_ids.price_extra"],
            semantic_role: OdooSemanticRole::Money,
        },
        // company_dependent=True: each company stores its own cost per variant.
        // constraint: standard_price >= 0.
        OdooField {
            name: "standard_price",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Money,
        },
        // Combination dedup key: ",".join(sorted ptav ids).
        // Unique index: (product_tmpl_id, combination_indices) WHERE active IS TRUE.
        OdooField {
            name: "combination_indices",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Identity,
        },
        OdooField {
            name: "product_template_attribute_value_ids",
            kind: OdooFieldKind::Many2many,
            target: Some("product.template.attribute.value"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "active",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Status,
        },
    ],
    methods: &[
        // lst_price = list_price + price_extra; context key 'uom' triggers
        // uom_id._compute_price(list_price, to_uom) before adding price_extra.
        OdooMethod {
            name: "_compute_product_lst_price",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        // Inverse of _compute_product_lst_price:
        // value = convert(lst_price) - price_extra → write({'list_price': value}).
        OdooMethod {
            name: "_set_product_lst_price",
            kind: OdooMethodKind::Inverse,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_product_price_extra",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        // Returns {product_id: float}; variant adds no_variant_attributes_price_extra
        // from context key on top of template._price_compute result (R16).
        OdooMethod {
            name: "_price_compute",
            kind: OdooMethodKind::Override,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        // standard_price >= 0 guard (R3).
        OdooMethod {
            name: "_onchange_standard_price",
            kind: OdooMethodKind::Onchange,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["list_price", "price_extra"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["product_template_attribute_value_ids.price_extra"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiOnchange,
            targets: &["standard_price"],
        },
    ],
    state_machine: None,
    constraints: &[
        OdooConstraint {
            kind: OdooConstraintKind::Sql,
            condition: "UNIQUE(product_tmpl_id, combination_indices) WHERE active IS TRUE \
                        — variant dedup within template",
            source_method: None,
        },
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "standard_price >= 0 (validated via _onchange_standard_price)",
            source_method: Some("_onchange_standard_price"),
        },
    ],
    provenance: OdooProvenance {
        l_doc: "L8-PRODUCT-UOM-PRICELIST.md",
        l_doc_lines: (62, 115),
        odoo_source: &[OdooSourceRef {
            path: "addons/product/models/product_product.py",
            line_range: (1, 1197),
        }],
        confidence: OdooConfidence::Curated,
    },
};

// ─── product.pricelist ────────────────────────────────────────────────────────
// NOTE: L6 overlap. L8 is authoritative for the pricelist data model.
// L6 covers the partner-assignment heuristics
// (_get_partner_pricelist_multi AXIS-B → PricelistAssignmentAgent).

pub const PRODUCT_PRICELIST: OdooEntity = OdooEntity {
    model_name: "product.pricelist",
    description: "Pricelist: currency + company + country-group scoping + ordered rule set. \
                  _compute_price_rule is the core multi-product pricing engine (R12). \
                  Gated by group_product_pricelist feature flag. \
                  Authoritative lane: L8 (data model + engine); \
                  L6 covers partner-assignment heuristics (PricelistAssignmentAgent).",
    fields: &[
        OdooField {
            name: "name",
            kind: OdooFieldKind::Char,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Identity,
        },
        OdooField {
            name: "currency_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.currency"),
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Money,
        },
        OdooField {
            name: "company_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.company"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "country_group_ids",
            kind: OdooFieldKind::Many2many,
            target: Some("res.country.group"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "item_ids",
            kind: OdooFieldKind::One2many,
            target: Some("product.pricelist.item"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Document,
        },
        // Lower sequence = higher priority when multiple pricelists qualify.
        OdooField {
            name: "sequence",
            kind: OdooFieldKind::Integer,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "active",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Status,
        },
    ],
    methods: &[
        // Core pricing engine (R12): mono-pricelist, multi-product.
        // Returns {product_id: (price, rule_id)}.
        // 1. Determine target_uom; convert qty to product UoM for min_qty check.
        // 2. Pre-fetch applicable rules via _get_applicable_rules_domain.
        // 3. For each product: first matching rule → _compute_price (item); else list_price fallback.
        OdooMethod {
            name: "_compute_price_rule",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        // Public entry point: wraps _compute_price_rule, returns scalar price.
        OdooMethod {
            name: "_get_product_price",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Number,
            triggers: &[],
        },
        // Rule candidate fetch (R10): filters by pricelist_id, product set, date validity,
        // categ ancestry. ORDER BY applied_on ASC, min_quantity DESC, categ_id DESC, id DESC.
        OdooMethod {
            name: "_get_applicable_rules_domain",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        // Partner pricelist assignment (R15, AXIS-B — PricelistAssignmentAgent):
        // explicit property → country-group match → fallback waterfall
        // (no-geo-restriction pricelist → ir.config_parameter → first active).
        // AXIS-B: fallback ordering is a business-policy judgment, not a closed formula.
        // See L6 for full SAVANT delegation spec.
        OdooMethod {
            name: "_get_partner_pricelist_multi",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        // Blocks delete when this pricelist is used as base_pricelist_id in another rule.
        OdooMethod {
            name: "_unlink_except_used_as_rule_base",
            kind: OdooMethodKind::Override,
            return_kind: OdooReturnKind::Boolean,
            triggers: &[],
        },
    ],
    decorators: &[],
    state_machine: None,
    constraints: &[
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "Cannot delete a pricelist used as base_pricelist_id in another rule",
            source_method: Some("_unlink_except_used_as_rule_base"),
        },
    ],
    provenance: OdooProvenance {
        l_doc: "L8-PRODUCT-UOM-PRICELIST.md",
        l_doc_lines: (242, 412),
        odoo_source: &[OdooSourceRef {
            path: "addons/product/models/product_pricelist.py",
            line_range: (1, 415),
        }],
        confidence: OdooConfidence::Curated,
    },
};

// ─── product.pricelist.item ────────────────────────────────────────────────────

pub const PRODUCT_PRICELIST_ITEM: OdooEntity = OdooEntity {
    model_name: "product.pricelist.item",
    description: "Pricelist rule: scoped by applied_on (global/category/product/variant), \
                  min_quantity (product default UoM), date validity window, and \
                  compute_price strategy (fixed/percentage/formula). \
                  _is_applicable_for is the inner-loop applicability check; \
                  _compute_price executes the three-way pricing branch (R13). \
                  DFS cycle guard (R19) prevents recursive pricelist chaining.",
    fields: &[
        OdooField {
            name: "pricelist_id",
            kind: OdooFieldKind::Many2one,
            target: Some("product.pricelist"),
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        // 3_global | 2_product_category | 1_product | 0_product_variant
        OdooField {
            name: "applied_on",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "categ_id",
            kind: OdooFieldKind::Many2one,
            target: Some("product.category"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "product_tmpl_id",
            kind: OdooFieldKind::Many2one,
            target: Some("product.template"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "product_id",
            kind: OdooFieldKind::Many2one,
            target: Some("product.product"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        // In product default UoM. min_quantity comparison done AFTER UoM conversion.
        OdooField {
            name: "min_quantity",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "date_start",
            kind: OdooFieldKind::Datetime,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Date,
        },
        OdooField {
            name: "date_end",
            kind: OdooFieldKind::Datetime,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Date,
        },
        // fixed | percentage | formula
        OdooField {
            name: "compute_price",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
        // list_price | standard_price | pricelist
        OdooField {
            name: "base",
            kind: OdooFieldKind::Selection,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
        // Only set when base='pricelist'; DFS cycle check on write.
        OdooField {
            name: "base_pricelist_id",
            kind: OdooFieldKind::Many2one,
            target: Some("product.pricelist"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        // compute_price='fixed': stored in product UoM; UoM-converted on retrieval.
        OdooField {
            name: "fixed_price",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Money,
        },
        // compute_price='percentage': discount %; negative = markup.
        OdooField {
            name: "percent_price",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Money,
        },
        // Formula: discount % on base price; when base=standard_price label shows as Markup.
        OdooField {
            name: "price_discount",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Money,
        },
        // price_markup = -price_discount (computed inverse for standard_price display).
        OdooField {
            name: "price_markup",
            kind: OdooFieldKind::Computed,
            target: None,
            required: false,
            computed: Some("_compute_price_markup"),
            depends: &["price_discount"],
            semantic_role: OdooSemanticRole::Money,
        },
        // Rounding step (precision_rounding, e.g. 0.05 = nearest 5 cents), not decimal digits.
        OdooField {
            name: "price_round",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Money,
        },
        // Additive surcharge in product UoM; UoM-converted before adding to formula price.
        OdooField {
            name: "price_surcharge",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Money,
        },
        // Minimum margin over base price (not absolute floor): price >= base + price_min_margin.
        OdooField {
            name: "price_min_margin",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Money,
        },
        OdooField {
            name: "price_max_margin",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Money,
        },
    ],
    methods: &[
        // Fine-grained applicability check (R11) — called in the inner loop of
        // _compute_price_rule after candidate fetch. Checks min_quantity,
        // applied_on scope (categ parent_path prefix, tmpl/variant id match).
        OdooMethod {
            name: "_is_applicable_for",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Boolean,
            triggers: &[],
        },
        // Three-way branch (R13):
        //   fixed   → uom_compute_price(fixed_price, product_uom, target_uom)
        //   percentage → base - base*(percent_price/100)
        //   formula → base*(1-discount/100) → round → surcharge → margin clamp
        // Empty recordset branch (no matching rule): falls back to list_price base.
        OdooMethod {
            name: "_compute_price",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Number,
            triggers: &[],
        },
        // Base price resolver (R14):
        //   pricelist → recursive base_pricelist._get_product_price
        //   standard_price → product._price_compute('standard_price') + cost_currency_id
        //   list_price → product._price_compute('list_price') + currency_id
        // Then currency_convert(src → pricelist.currency, round=False).
        OdooMethod {
            name: "_compute_base_price",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Number,
            triggers: &[],
        },
        // DFS cycle detection on write when base='pricelist' (R19):
        // traverses base_pricelist_id chains; raises ValidationError if cycle found.
        OdooMethod {
            name: "_check_pricelist_recursion",
            kind: OdooMethodKind::Constrain,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_price_markup",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiConstrains,
            targets: &["base_pricelist_id", "pricelist_id"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["price_discount"],
        },
    ],
    state_machine: None,
    constraints: &[
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "date_start < date_end (when both set)",
            source_method: None,
        },
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "price_min_margin <= price_max_margin (when both non-zero)",
            source_method: None,
        },
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "base='pricelist' requires base_pricelist_id; no pricelist graph cycles \
                        (DFS enforced on write)",
            source_method: Some("_check_pricelist_recursion"),
        },
    ],
    provenance: OdooProvenance {
        l_doc: "L8-PRODUCT-UOM-PRICELIST.md",
        l_doc_lines: (263, 660),
        odoo_source: &[OdooSourceRef {
            path: "addons/product/models/product_pricelist_item.py",
            line_range: (1, 684),
        }],
        confidence: OdooConfidence::Curated,
    },
};

// ─── ENTITIES slice ────────────────────────────────────────────────────────────

pub const ENTITIES: &[OdooEntity] = &[
    PRODUCT_CATEGORY,
    UOM_UOM,
    PRODUCT_TEMPLATE,
    PRODUCT_PRODUCT,
    PRODUCT_PRICELIST,
    PRODUCT_PRICELIST_ITEM,
];

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::odoo_blueprint::{OdooConfidence, OdooFieldKind};

    #[test]
    fn l8_entity_count() {
        assert_eq!(ENTITIES.len(), 6);
    }

    #[test]
    fn product_category_has_parent_path() {
        let f = PRODUCT_CATEGORY
            .fields
            .iter()
            .find(|f| f.name == "parent_path")
            .expect("parent_path field");
        assert_eq!(f.kind, OdooFieldKind::Char);
    }

    #[test]
    fn uom_uom_sql_constraints() {
        let sql: Vec<_> = UOM_UOM
            .constraints
            .iter()
            .filter(|c| c.kind == crate::odoo_blueprint::OdooConstraintKind::Sql)
            .collect();
        assert_eq!(sql.len(), 2, "factor!=0 and rounding>0");
    }

    #[test]
    fn product_template_curated() {
        assert_eq!(
            PRODUCT_TEMPLATE.provenance.confidence,
            OdooConfidence::Curated
        );
        assert_eq!(
            PRODUCT_TEMPLATE.provenance.l_doc,
            "L8-PRODUCT-UOM-PRICELIST.md"
        );
    }

    #[test]
    fn product_product_lst_price_computed() {
        let lst = PRODUCT_PRODUCT
            .fields
            .iter()
            .find(|f| f.name == "lst_price")
            .expect("lst_price field");
        assert_eq!(lst.kind, OdooFieldKind::Computed);
        assert_eq!(lst.computed, Some("_compute_product_lst_price"));
    }

    #[test]
    fn pricelist_item_has_applicable_check() {
        let m = PRODUCT_PRICELIST_ITEM
            .methods
            .iter()
            .find(|m| m.name == "_is_applicable_for")
            .expect("_is_applicable_for");
        assert_eq!(m.return_kind, OdooReturnKind::Boolean);
    }

    #[test]
    fn all_entities_have_l8_ldoc() {
        for e in ENTITIES {
            assert_eq!(
                e.provenance.l_doc,
                "L8-PRODUCT-UOM-PRICELIST.md",
                "{} has wrong l_doc",
                e.model_name
            );
            assert_eq!(
                e.provenance.confidence,
                OdooConfidence::Curated,
                "{} not Curated",
                e.model_name
            );
        }
    }
}
