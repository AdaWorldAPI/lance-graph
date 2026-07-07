//! Lane L8 (PRODUCT-UOM-PRICELIST) — typed Odoo entity declarations.
//!
//! Source: `.claude/odoo/L8-PRODUCT-UOM-PRICELIST.md`.
//!
//! Entities: `product.category`, `uom.uom`, `product.template`,
//! `product.product`, `product.pricelist`, `product.pricelist.item`.
//!
//! **L6 overlap**: L8 is authoritative for the pricelist data model.
//! L6 covers partner-assignment heuristics (`_get_partner_pricelist_multi`
//! AXIS-B → `PricelistAssignmentAgent`).
//!
//! **Ontology gap**: all seven classes → `None` from `resolve_odoo_to_family()`.
//! Proposed: `product.*` → `schema:Product` → new OGIT family `0x63 ProductCatalog`.

use super::{
    OdooConfidence, OdooConstraint, OdooConstraintKind, OdooDecorator, OdooDecoratorKind,
    OdooEntity, OdooEntityKind, OdooField, OdooFieldKind, OdooMethod, OdooMethodKind,
    OdooProvenance, OdooReturnKind, OdooSemanticRole, OdooSourceRef,
};

// helper macro to cut boilerplate on simple field declarations
macro_rules! f {
    ($name:expr, $kind:expr, $target:expr, $req:expr, $comp:expr, $deps:expr, $role:expr) => {
        OdooField {
            name: $name,
            kind: $kind,
            target: $target,
            required: $req,
            computed: $comp,
            depends: $deps,
            semantic_role: $role,
        }
    };
}

macro_rules! m {
    ($name:expr, $kind:expr, $ret:expr) => {
        OdooMethod { name: $name, kind: $kind, return_kind: $ret, triggers: &[] }
    };
}

// ─── product.category ─────────────────────────────────────────────────────────

pub const PRODUCT_CATEGORY: OdooEntity = OdooEntity {
    model_name: "product.category",
    kind: OdooEntityKind::Model,
    description: "Hierarchical product category; parent_path materialized closure used in \
                  pricelist categ ancestor check; company-dependent income/expense GL accounts \
                  (account resolution waterfall R7).",
    fields: &[
        f!("name", OdooFieldKind::Char, None, true, None, &[], OdooSemanticRole::Identity),
        // Recursive breadcrumb "All / Consumable / Electronic".
        f!("complete_name", OdooFieldKind::Computed, None, false,
           Some("_compute_complete_name"), &["name", "parent_id.complete_name"],
           OdooSemanticRole::Identity),
        // Cascade delete; DFS cycle guard _check_category_recursion.
        f!("parent_id", OdooFieldKind::Many2one, Some("product.category"), false,
           None, &[], OdooSemanticRole::Reference),
        // Materialized closure "1/4/7/" — pricelist rule categ prefix check.
        f!("parent_path", OdooFieldKind::Char, None, false, None, &[],
           OdooSemanticRole::Reference),
        f!("child_id", OdooFieldKind::One2many, Some("product.category"), false,
           None, &[], OdooSemanticRole::Reference),
        // company_dependent; income account for customer invoices (R6/R7).
        f!("property_account_income_categ_id", OdooFieldKind::Many2one,
           Some("account.account"), false, None, &[], OdooSemanticRole::Tax),
        // company_dependent; expense/COGS account for vendor bills.
        f!("property_account_expense_categ_id", OdooFieldKind::Many2one,
           Some("account.account"), false, None, &[], OdooSemanticRole::Tax),
    ],
    methods: &[
        m!("_check_category_recursion", OdooMethodKind::Constrain, OdooReturnKind::Unit),
        // Account resolution waterfall (R7): product prop → categ walk → company default.
        m!("_get_product_accounts", OdooMethodKind::Helper, OdooReturnKind::Dict),
    ],
    decorators: &[
        OdooDecorator { kind: OdooDecoratorKind::ApiConstrains, targets: &["parent_id"] },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["name", "parent_id.complete_name"],
        },
    ],
    state_machine: None,
    constraints: &[OdooConstraint {
        kind: OdooConstraintKind::Python,
        condition: "No cycles in parent_id chain (DFS _check_category_recursion)",
        source_method: Some("_check_category_recursion"),
    }],
    provenance: OdooProvenance {
        l_doc: "L8-PRODUCT-UOM-PRICELIST.md",
        l_doc_lines: (180, 238),
        odoo_source: &[
            OdooSourceRef { path: "addons/product/models/product_category.py", line_range: (1, 69) },
            OdooSourceRef { path: "addons/account/models/product.py", line_range: (13, 29) },
        ],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── uom.uom ──────────────────────────────────────────────────────────────────

pub const UOM_UOM: OdooEntity = OdooEntity {
    model_name: "uom.uom",
    kind: OdooEntityKind::Model,
    description: "Unit of Measure: factor-based conversion. Reference unit factor=1.0. \
                  _compute_quantity: (qty/from.factor)*to.factor, then float_round. \
                  _compute_price: price*from.factor/to.factor (no rounding). \
                  SQL constraints: factor!=0, rounding>0. \
                  Stock guard: blocks factor change when open moves/quants (R21).",
    fields: &[
        f!("name", OdooFieldKind::Char, None, true, None, &[], OdooSemanticRole::Identity),
        f!("category_id", OdooFieldKind::Many2one, Some("uom.category"), true,
           None, &[], OdooSemanticRole::Reference),
        // Ratio vs reference. Bigger UoM: factor<1 (dozen≈0.0833); smaller: factor>1.
        f!("factor", OdooFieldKind::Float, None, true, None, &[], OdooSemanticRole::Quantity),
        f!("factor_inv", OdooFieldKind::Computed, None, false,
           Some("_compute_factor_inv"), &["factor"], OdooSemanticRole::Quantity),
        // Precision step e.g. 0.01, 1.0. SQL constraint: rounding>0.
        f!("rounding", OdooFieldKind::Float, None, true, None, &[], OdooSemanticRole::Quantity),
        // bigger | reference | smaller; each category has exactly one 'reference'.
        f!("uom_type", OdooFieldKind::Selection, None, true, None, &[], OdooSemanticRole::Policy),
        f!("active", OdooFieldKind::Boolean, None, true, None, &[], OdooSemanticRole::Status),
    ],
    methods: &[
        // Core qty conversion (R5); rounding_method defaults 'UP', stock uses 'HALF-UP'.
        m!("_compute_quantity", OdooMethodKind::Helper, OdooReturnKind::Number),
        // Price conversion (R5): no rounding; called from lst_price + pricelist fixed (R2, R13).
        m!("_compute_price", OdooMethodKind::Helper, OdooReturnKind::Number),
        m!("_compute_factor_inv", OdooMethodKind::Compute, OdooReturnKind::Unit),
        // Stock guard (R21): blocks factor write when open stock.move/quant uses this UoM.
        m!("write", OdooMethodKind::Override, OdooReturnKind::Boolean),
    ],
    decorators: &[
        OdooDecorator { kind: OdooDecoratorKind::ApiDepends, targets: &["factor"] },
    ],
    state_machine: None,
    constraints: &[
        OdooConstraint { kind: OdooConstraintKind::Sql, condition: "factor != 0",
            source_method: None },
        OdooConstraint { kind: OdooConstraintKind::Sql, condition: "rounding > 0",
            source_method: None },
        // Each category has exactly one uom_type='reference' with factor=1.0 (data-seed level).
        OdooConstraint { kind: OdooConstraintKind::Domain,
            condition: "Each uom.category has exactly one reference unit (factor=1.0)",
            source_method: None },
        OdooConstraint { kind: OdooConstraintKind::Python,
            condition: "Cannot change factor when open stock.move or non-zero stock.quant \
                        references this UoM",
            source_method: Some("write") },
    ],
    provenance: OdooProvenance {
        l_doc: "L8-PRODUCT-UOM-PRICELIST.md",
        l_doc_lines: (119, 177),
        odoo_source: &[
            // Core uom module absent from clone; reconstructed via GitHub WebFetch (R5).
            OdooSourceRef { path: "addons/uom/models/uom_uom.py", line_range: (1, 200) },
            OdooSourceRef { path: "addons/stock/models/product.py", line_range: (1344, 1375) },
        ],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── product.template ─────────────────────────────────────────────────────────

pub const PRODUCT_TEMPLATE: OdooEntity = OdooEntity {
    model_name: "product.template",
    kind: OdooEntityKind::Model,
    description: "Product template: canonical catalog record. Drives variant matrix via \
                  attribute_line_ids (Cartesian product, _create_variant_ids, R18). \
                  Holds list_price, company-dependent standard_price, currency_id. \
                  _price_compute + _get_contextual_price are the pricing entry points (R16, R22).",
    fields: &[
        f!("name", OdooFieldKind::Char, None, true, None, &[], OdooSemanticRole::Identity),
        // consu | service | combo. combo: combo_ids required, attribute_line_ids forbidden.
        f!("type", OdooFieldKind::Selection, None, true, None, &[], OdooSemanticRole::Policy),
        f!("active", OdooFieldKind::Boolean, None, true, None, &[], OdooSemanticRole::Status),
        // Base sale price per default UoM; variant lst_price adds price_extra (R2).
        f!("list_price", OdooFieldKind::Float, None, false, None, &[], OdooSemanticRole::Money),
        // company_dependent; delegated compute from single variant when one variant exists (R3).
        f!("standard_price", OdooFieldKind::Computed, None, false,
           Some("_compute_template_field_from_variant_field"),
           &["product_variant_ids", "product_variant_ids.standard_price"],
           OdooSemanticRole::Money),
        // company_id.currency_id OR main_company.currency_id (R4).
        f!("currency_id", OdooFieldKind::Computed, Some("res.currency"), false,
           Some("_compute_currency_id"), &["company_id"], OdooSemanticRole::Money),
        // env.company.currency_id; depends_context('company') (R4).
        f!("cost_currency_id", OdooFieldKind::Computed, Some("res.currency"), false,
           Some("_compute_cost_currency_id"), &[], OdooSemanticRole::Money),
        f!("uom_id", OdooFieldKind::Many2one, Some("uom.uom"), true,
           None, &[], OdooSemanticRole::Reference),
        f!("uom_po_id", OdooFieldKind::Many2one, Some("uom.uom"), true,
           None, &[], OdooSemanticRole::Reference),
        f!("categ_id", OdooFieldKind::Many2one, Some("product.category"), true,
           None, &[], OdooSemanticRole::Reference),
        f!("company_id", OdooFieldKind::Many2one, Some("res.company"), false,
           None, &[], OdooSemanticRole::Reference),
        f!("sale_ok", OdooFieldKind::Boolean, None, true, None, &[], OdooSemanticRole::Policy),
        // stored+recomputable; no-op base; overridden by purchase module.
        f!("purchase_ok", OdooFieldKind::Boolean, None, true, None, &[], OdooSemanticRole::Policy),
        // Drives variant matrix Cartesian product.
        f!("attribute_line_ids", OdooFieldKind::One2many,
           Some("product.template.attribute.line"), false, None, &[], OdooSemanticRole::Reference),
        f!("product_variant_ids", OdooFieldKind::One2many, Some("product.product"), false,
           None, &[], OdooSemanticRole::Reference),
        // Non-empty only when type='combo'.
        f!("combo_ids", OdooFieldKind::Many2many, Some("product.combo"), false,
           None, &[], OdooSemanticRole::Reference),
        // company_dependent product-level income account override (R7).
        f!("property_account_income_id", OdooFieldKind::Many2one, Some("account.account"),
           false, None, &[], OdooSemanticRole::Tax),
        f!("property_account_expense_id", OdooFieldKind::Many2one, Some("account.account"),
           false, None, &[], OdooSemanticRole::Tax),
    ],
    methods: &[
        // Returns {product_id: float}: raw price → attr extra → UoM → currency (R16).
        m!("_price_compute", OdooMethodKind::Helper, OdooReturnKind::Dict),
        // Reads context pricelist/qty/uom/date; calls pricelist._get_product_price (R22).
        m!("_get_contextual_price", OdooMethodKind::Helper, OdooReturnKind::Number),
        m!("_get_contextual_pricelist", OdooMethodKind::Helper, OdooReturnKind::Record),
        m!("_compute_currency_id", OdooMethodKind::Compute, OdooReturnKind::Unit),
        m!("_compute_cost_currency_id", OdooMethodKind::Compute, OdooReturnKind::Unit),
        m!("_compute_template_field_from_variant_field", OdooMethodKind::Compute,
           OdooReturnKind::Unit),
        // Recomputes Cartesian product after attribute line changes (R18).
        // Limit: ir.config_parameter product.dynamic_variant_limit (default 1000).
        m!("_create_variant_ids", OdooMethodKind::Helper, OdooReturnKind::Unit),
        // Account resolution: product prop → categ walk → company default (R7).
        m!("_get_product_accounts", OdooMethodKind::Helper, OdooReturnKind::Dict),
        // Invoice line unit price: lst_price → UoM → fiscal-pos tax adapt → currency (R23).
        m!("_get_tax_included_unit_price", OdooMethodKind::Helper, OdooReturnKind::Number),
        // type='combo': combo_ids >= 1 and attribute_line_ids empty (R1).
        m!("_check_combo_constraints", OdooMethodKind::Constrain, OdooReturnKind::Unit),
        // Cross-model: blocks uom_id change if posted account_move_line has different UoM (R20).
        m!("_check_uom_id", OdooMethodKind::Constrain, OdooReturnKind::Unit),
    ],
    decorators: &[
        OdooDecorator { kind: OdooDecoratorKind::ApiDepends, targets: &["company_id"] },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["product_variant_ids", "product_variant_ids.standard_price"],
        },
        OdooDecorator { kind: OdooDecoratorKind::ApiConstrains, targets: &["uom_id"] },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiConstrains,
            targets: &["type", "attribute_line_ids", "combo_ids"],
        },
    ],
    state_machine: None,
    constraints: &[
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "type='combo': combo_ids.len()>=1, attribute_line_ids.is_empty()",
            source_method: Some("_check_combo_constraints"),
        },
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "Cannot change uom_id when posted account_move_line with different UoM \
                        references this product (R20)",
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
            OdooSourceRef { path: "addons/account/models/product.py", line_range: (67, 293) },
        ],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── product.product ──────────────────────────────────────────────────────────

pub const PRODUCT_PRODUCT: OdooEntity = OdooEntity {
    model_name: "product.product",
    kind: OdooEntityKind::Model,
    description: "Product variant: _inherits product.template. lst_price = list_price + \
                  price_extra ± UoM conversion (R2). company_dependent standard_price ≥ 0 \
                  (R3). combination_indices unique index within template (R18).",
    fields: &[
        f!("product_tmpl_id", OdooFieldKind::Many2one, Some("product.template"), true,
           None, &[], OdooSemanticRole::Reference),
        // list_price + price_extra; context 'uom' triggers uom._compute_price (R2).
        f!("lst_price", OdooFieldKind::Computed, None, false,
           Some("_compute_product_lst_price"), &["list_price", "price_extra"],
           OdooSemanticRole::Money),
        // Sum of product_template_attribute_value.price_extra for this variant's PTAVs.
        f!("price_extra", OdooFieldKind::Computed, None, false,
           Some("_compute_product_price_extra"),
           &["product_template_attribute_value_ids.price_extra"],
           OdooSemanticRole::Money),
        // company_dependent; constraint: standard_price >= 0 (R3).
        f!("standard_price", OdooFieldKind::Float, None, false, None, &[],
           OdooSemanticRole::Money),
        // Dedup key: ",".join(sorted ptav ids). Unique index: (tmpl_id, combo_indices) active.
        f!("combination_indices", OdooFieldKind::Char, None, false, None, &[],
           OdooSemanticRole::Identity),
        f!("product_template_attribute_value_ids", OdooFieldKind::Many2many,
           Some("product.template.attribute.value"), false, None, &[],
           OdooSemanticRole::Reference),
        f!("active", OdooFieldKind::Boolean, None, true, None, &[], OdooSemanticRole::Status),
    ],
    methods: &[
        m!("_compute_product_lst_price", OdooMethodKind::Compute, OdooReturnKind::Unit),
        // Inverse: value = convert(lst_price) - price_extra → write({'list_price': value}).
        m!("_set_product_lst_price", OdooMethodKind::Inverse, OdooReturnKind::Unit),
        m!("_compute_product_price_extra", OdooMethodKind::Compute, OdooReturnKind::Unit),
        // Adds no_variant_attributes_price_extra from context key (R16).
        m!("_price_compute", OdooMethodKind::Override, OdooReturnKind::Dict),
        m!("_onchange_standard_price", OdooMethodKind::Onchange, OdooReturnKind::Unit),
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
        OdooDecorator { kind: OdooDecoratorKind::ApiOnchange, targets: &["standard_price"] },
    ],
    state_machine: None,
    constraints: &[
        OdooConstraint {
            kind: OdooConstraintKind::Sql,
            condition: "UNIQUE(product_tmpl_id, combination_indices) WHERE active IS TRUE",
            source_method: None,
        },
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "standard_price >= 0",
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
        regulation_iri: &[],
    },
};

// ─── product.pricelist ────────────────────────────────────────────────────────
// L6 overlap: L8 owns the data model + engine; L6 owns partner-assignment heuristics.

pub const PRODUCT_PRICELIST: OdooEntity = OdooEntity {
    model_name: "product.pricelist",
    kind: OdooEntityKind::Model,
    description: "Pricelist: currency + company + country-group scoping + ordered rule set. \
                  _compute_price_rule is the core engine (R12). \
                  Gated by group_product_pricelist feature flag. \
                  L6 overlap: partner assignment (_get_partner_pricelist_multi AXIS-B \
                  → PricelistAssignmentAgent) is L6 territory.",
    fields: &[
        f!("name", OdooFieldKind::Char, None, true, None, &[], OdooSemanticRole::Identity),
        f!("currency_id", OdooFieldKind::Many2one, Some("res.currency"), true,
           None, &[], OdooSemanticRole::Money),
        f!("company_id", OdooFieldKind::Many2one, Some("res.company"), false,
           None, &[], OdooSemanticRole::Reference),
        f!("country_group_ids", OdooFieldKind::Many2many, Some("res.country.group"), false,
           None, &[], OdooSemanticRole::Reference),
        f!("item_ids", OdooFieldKind::One2many, Some("product.pricelist.item"), false,
           None, &[], OdooSemanticRole::Document),
        // Lower sequence = higher priority when multiple pricelists qualify.
        f!("sequence", OdooFieldKind::Integer, None, false, None, &[], OdooSemanticRole::Policy),
        f!("active", OdooFieldKind::Boolean, None, true, None, &[], OdooSemanticRole::Status),
    ],
    methods: &[
        // Core engine (R12): {product_id: (price, rule_id)}. Pre-fetches applicable rules,
        // iterates per product, first match → _compute_price; no match → list_price fallback.
        m!("_compute_price_rule", OdooMethodKind::Helper, OdooReturnKind::Dict),
        m!("_get_product_price", OdooMethodKind::Helper, OdooReturnKind::Number),
        // Rule candidate fetch (R10): filter by pricelist_id, products, date, categ ancestry.
        // ORDER BY applied_on ASC, min_quantity DESC, categ_id DESC, id DESC.
        m!("_get_applicable_rules_domain", OdooMethodKind::Helper, OdooReturnKind::Dict),
        // Partner assignment (R15, AXIS-B): explicit property → country-group → fallback
        // waterfall. Fallback ordering is business-policy judgment → PricelistAssignmentAgent.
        m!("_get_partner_pricelist_multi", OdooMethodKind::Helper, OdooReturnKind::Dict),
        m!("_unlink_except_used_as_rule_base", OdooMethodKind::Override, OdooReturnKind::Boolean),
    ],
    decorators: &[],
    state_machine: None,
    constraints: &[OdooConstraint {
        kind: OdooConstraintKind::Python,
        condition: "Cannot delete a pricelist used as base_pricelist_id in another rule",
        source_method: Some("_unlink_except_used_as_rule_base"),
    }],
    provenance: OdooProvenance {
        l_doc: "L8-PRODUCT-UOM-PRICELIST.md",
        l_doc_lines: (242, 412),
        odoo_source: &[OdooSourceRef {
            path: "addons/product/models/product_pricelist.py",
            line_range: (1, 415),
        }],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── product.pricelist.item ────────────────────────────────────────────────────

pub const PRODUCT_PRICELIST_ITEM: OdooEntity = OdooEntity {
    model_name: "product.pricelist.item",
    kind: OdooEntityKind::Model,
    description: "Pricelist rule: applied_on scope, min_quantity (product UoM), date validity. \
                  Three compute_price strategies: fixed/percentage/formula (R13). \
                  Base resolvers: list_price/standard_price/pricelist (R14). \
                  DFS cycle guard prevents recursive pricelist chaining (R19).",
    fields: &[
        f!("pricelist_id", OdooFieldKind::Many2one, Some("product.pricelist"), true,
           None, &[], OdooSemanticRole::Reference),
        // 3_global | 2_product_category | 1_product | 0_product_variant.
        f!("applied_on", OdooFieldKind::Selection, None, true, None, &[], OdooSemanticRole::Policy),
        f!("categ_id", OdooFieldKind::Many2one, Some("product.category"), false,
           None, &[], OdooSemanticRole::Reference),
        f!("product_tmpl_id", OdooFieldKind::Many2one, Some("product.template"), false,
           None, &[], OdooSemanticRole::Reference),
        f!("product_id", OdooFieldKind::Many2one, Some("product.product"), false,
           None, &[], OdooSemanticRole::Reference),
        // In product default UoM; qty already converted before _is_applicable_for check.
        f!("min_quantity", OdooFieldKind::Float, None, false, None, &[], OdooSemanticRole::Quantity),
        f!("date_start", OdooFieldKind::Datetime, None, false, None, &[], OdooSemanticRole::Date),
        f!("date_end", OdooFieldKind::Datetime, None, false, None, &[], OdooSemanticRole::Date),
        // fixed | percentage | formula.
        f!("compute_price", OdooFieldKind::Selection, None, true, None, &[], OdooSemanticRole::Policy),
        // list_price | standard_price | pricelist.
        f!("base", OdooFieldKind::Selection, None, false, None, &[], OdooSemanticRole::Policy),
        // Only when base='pricelist'; DFS cycle check on write (R19).
        f!("base_pricelist_id", OdooFieldKind::Many2one, Some("product.pricelist"), false,
           None, &[], OdooSemanticRole::Reference),
        // fixed: stored in product UoM; UoM-converted on retrieval (R13-A).
        f!("fixed_price", OdooFieldKind::Float, None, false, None, &[], OdooSemanticRole::Money),
        // percentage: discount %; negative = markup (R13-B).
        f!("percent_price", OdooFieldKind::Float, None, false, None, &[], OdooSemanticRole::Money),
        // formula: discount % on base; base=standard_price → label shows "Markup" (R13-C).
        f!("price_discount", OdooFieldKind::Float, None, false, None, &[], OdooSemanticRole::Money),
        // price_markup = -price_discount (computed inverse for standard_price display).
        f!("price_markup", OdooFieldKind::Computed, None, false,
           Some("_compute_price_markup"), &["price_discount"], OdooSemanticRole::Money),
        // Rounding step (precision_rounding e.g. 0.05 = nearest 5c), NOT decimal digits.
        f!("price_round", OdooFieldKind::Float, None, false, None, &[], OdooSemanticRole::Money),
        // Additive surcharge in product UoM; UoM-converted before adding to formula price.
        f!("price_surcharge", OdooFieldKind::Float, None, false, None, &[], OdooSemanticRole::Money),
        // Min margin over base (not absolute floor): price >= base + price_min_margin.
        f!("price_min_margin", OdooFieldKind::Float, None, false, None, &[], OdooSemanticRole::Money),
        f!("price_max_margin", OdooFieldKind::Float, None, false, None, &[], OdooSemanticRole::Money),
    ],
    methods: &[
        // Inner-loop applicability check (R11): min_quantity, categ parent_path prefix,
        // tmpl/variant id match. Called after candidate fetch in _compute_price_rule.
        m!("_is_applicable_for", OdooMethodKind::Helper, OdooReturnKind::Boolean),
        // Three-way branch (R13): fixed→uom_cvt; percentage→base*(1-p/100);
        // formula→discount→round→surcharge→margin clamp. Empty self → list_price fallback.
        m!("_compute_price", OdooMethodKind::Helper, OdooReturnKind::Number),
        // Base resolver (R14): pricelist→recursive call; standard_price→cost_currency;
        // list_price→currency. Then currency_convert(src→pl.currency, round=False).
        m!("_compute_base_price", OdooMethodKind::Helper, OdooReturnKind::Number),
        // DFS cycle detection on write when base='pricelist' (R19).
        m!("_check_pricelist_recursion", OdooMethodKind::Constrain, OdooReturnKind::Unit),
        m!("_compute_price_markup", OdooMethodKind::Compute, OdooReturnKind::Unit),
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiConstrains,
            targets: &["base_pricelist_id", "pricelist_id"],
        },
        OdooDecorator { kind: OdooDecoratorKind::ApiDepends, targets: &["price_discount"] },
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
                        (DFS on write, R19)",
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
        regulation_iri: &[],
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
    use crate::odoo_blueprint::{OdooConfidence, OdooConstraintKind, OdooFieldKind};

    #[test]
    fn l8_entity_count() {
        assert_eq!(ENTITIES.len(), 6);
    }

    #[test]
    fn product_category_has_parent_path() {
        let f = PRODUCT_CATEGORY.fields.iter().find(|f| f.name == "parent_path").unwrap();
        assert_eq!(f.kind, OdooFieldKind::Char);
    }

    #[test]
    fn uom_uom_sql_constraints() {
        let sql: Vec<_> = UOM_UOM.constraints.iter()
            .filter(|c| c.kind == OdooConstraintKind::Sql)
            .collect();
        assert_eq!(sql.len(), 2, "factor!=0 and rounding>0");
    }

    #[test]
    fn product_template_curated() {
        assert_eq!(PRODUCT_TEMPLATE.provenance.confidence, OdooConfidence::Curated);
        assert_eq!(PRODUCT_TEMPLATE.provenance.l_doc, "L8-PRODUCT-UOM-PRICELIST.md");
    }

    #[test]
    fn product_product_lst_price_computed() {
        let lst = PRODUCT_PRODUCT.fields.iter().find(|f| f.name == "lst_price").unwrap();
        assert_eq!(lst.kind, OdooFieldKind::Computed);
        assert_eq!(lst.computed, Some("_compute_product_lst_price"));
    }

    #[test]
    fn pricelist_item_has_applicable_check() {
        let m = PRODUCT_PRICELIST_ITEM.methods.iter()
            .find(|m| m.name == "_is_applicable_for")
            .unwrap();
        assert_eq!(m.return_kind, OdooReturnKind::Boolean);
    }

    #[test]
    fn all_entities_have_l8_ldoc() {
        for e in ENTITIES {
            assert_eq!(e.provenance.l_doc, "L8-PRODUCT-UOM-PRICELIST.md",
                       "{} has wrong l_doc", e.model_name);
            assert_eq!(e.provenance.confidence, OdooConfidence::Curated,
                       "{} not Curated", e.model_name);
        }
    }
}
