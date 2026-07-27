//! Lane L13 (STOCK-VALUATION-PROCUREMENT) — typed Odoo entity declarations.
//!
//! Source: `.claude/odoo/L13-STOCK-VALUATION-PROCUREMENT.md`.
//!
//! Entities covered (5):
//!   - `stock.valuation.layer`    — SVL interface contract (FIFO/AVCO/standard; absent from
//!                                  community clone, specced as fresh woa-rs build)
//!   - `stock.warehouse.orderpoint` — min/max reorder rule; drives ReorderTimingAdvisor +
//!                                    ReplenishmentReportAdvisor savants
//!   - `stock.rule`               — procurement-side priority-resolution (EXTENDS L7 basics);
//!                                  drives ProcurementRuleSelector + RouteTiebreaker savants
//!   - `stock.lot`                — lot/serial master; uniqueness constraint, name generation
//!   - `res.company` (valuation)  — anglo-saxon vs continental GL config
//!
//! L7 overlap note:
//!   `stock.rule` basic shape (action / group_propagation / delay / partner_address_id /
//!   location_src_id / location_dest_id / picking_type_id / route_id) was captured in L7.
//!   This L13 entity focuses on the PROCUREMENT-PRIORITY fields (_get_rule route walk,
//!   _run_pull, _run_push, _get_stock_move_values) and the two tiebreak savants.
//!   L7's `stock.move.procure_method` routing (_action_confirm R7 branch) is the consumer.
//!
//! NARS dispatch surfaces:
//!   - `ProcurementRuleSelector` → stock.rule._get_rule (location-hierarchy walk + route priority)
//!   - `RouteTiebreaker`         → stock.rule equal-sequence tiebreak (R16)
//!   - `ReorderTimingAdvisor`    → stock.warehouse.orderpoint._compute_deadline_date (R9)
//!   - `ReplenishmentReportAdvisor` → stock.warehouse.orderpoint._get_orderpoint_action (R15)

use super::{
    OdooConfidence, OdooConstraint, OdooConstraintKind, OdooDecorator, OdooDecoratorKind,
    OdooEntity, OdooEntityKind, OdooField, OdooFieldKind, OdooMethod, OdooMethodKind,
    OdooProvenance, OdooReturnKind, OdooSemanticRole, OdooSourceRef,
};

// ─── stock.valuation.layer ────────────────────────────────────────────────────

/// SVL interface contract (built fresh in woa-rs; absent from community clone).
///
/// GL bridge: receipt Dr stock_input Cr inventory_valuation;
///            delivery Dr inventory_valuation Cr stock_output.
/// Anglo-saxon delivery: Dr COGS_interim Cr inventory_valuation (deferred to invoice).
/// Continental delivery (DE default): Dr stock_output Cr inventory_valuation.
/// Cost methods: standard (=standard_price) | average (running avg, Decimal HALF_UP) | fifo
/// (oldest layer; vacuum on later bill price).
const STOCK_VALUATION_LAYER: OdooEntity = OdooEntity {
    model_name: "stock.valuation.layer",
    kind: OdooEntityKind::Model,
    description: "One accounting-valuation record created per done stock.move; carries signed \
                  quantity + unit_cost + value (qty × unit_cost, rounded at currency precision); \
                  FIFO: remaining_qty/value tracks open layer balance; absent from community clone \
                  — interface contract specced for fresh woa-rs engine (R1-R2, DOLCE Perdurant).",
    fields: &[
        OdooField {
            name: "product_id",
            kind: OdooFieldKind::Many2one,
            target: Some("product.product"),
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "company_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.company"),
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "stock_move_id",
            kind: OdooFieldKind::Many2one,
            target: Some("stock.move"),
            required: false,
            computed: None,
            depends: &[],
            // The done move that triggered this SVL entry.
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "quantity",
            kind: OdooFieldKind::Float,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // Signed: positive=receipt, negative=delivery/correction.
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "unit_cost",
            kind: OdooFieldKind::Float,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // standard_price (standard) | running avg (avco) | oldest-layer cost (fifo).
            semantic_role: OdooSemanticRole::Money,
        },
        OdooField {
            name: "value",
            kind: OdooFieldKind::Monetary,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // quantity × unit_cost, rounded at currency precision (Decimal HALF_UP).
            semantic_role: OdooSemanticRole::Money,
        },
        OdooField {
            name: "remaining_qty",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // FIFO only: open balance on this layer (decremented by outgoing moves).
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "remaining_value",
            kind: OdooFieldKind::Monetary,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // FIFO: remaining_qty × unit_cost; vacuum on later bill-price reconciliation.
            semantic_role: OdooSemanticRole::Money,
        },
        OdooField {
            name: "description",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Document,
        },
    ],
    methods: &[
        OdooMethod {
            name: "_run_fifo",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        OdooMethod {
            name: "_run_average_price",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        OdooMethod {
            name: "_create_account_move_vals",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
    ],
    decorators: &[],
    state_machine: None,
    constraints: &[OdooConstraint {
        kind: OdooConstraintKind::Python,
        // value must equal quantity × unit_cost at currency rounding precision.
        condition: "value = round(quantity × unit_cost, currency.decimal_places) — Decimal HALF_UP",
        source_method: None,
    }],
    provenance: OdooProvenance {
        l_doc: "L13-STOCK-VALUATION-PROCUREMENT.md",
        // Specced from SVL interface contract block + R1/R2 (lines 33-36, 85-91).
        l_doc_lines: (33, 91),
        odoo_source: &[OdooSourceRef {
            // Absent from community clone — fresh woa-rs build required.
            path: "addons/stock_account/models/stock_valuation_layer.py",
            line_range: (1, 1),
        }],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── stock.warehouse.orderpoint ───────────────────────────────────────────────

pub const STOCK_WAREHOUSE_ORDERPOINT: OdooEntity = OdooEntity {
    model_name: "stock.warehouse.orderpoint",
    kind: OdooEntityKind::Model,
    description: "Min/max reorder rule for one product at one location; drives scheduler batch \
                  (_procure_orderpoint_confirm R11 / _run_scheduler_tasks R12); \
                  qty_to_order = max(min,max) − (virtual_available + qty_in_progress) rounded UP \
                  to replenishment_uom multiple (R8); deadline = first day below min within lead \
                  horizon (R9 — ReorderTimingAdvisor NARS surface); replenishment report = \
                  _get_orderpoint_action (R15 — ReplenishmentReportAdvisor NARS surface); \
                  OWL pivot fibo:Obligation (DOLCE Perdurant).",
    fields: &[
        OdooField {
            name: "name",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: Some("_compute_name"),
            depends: &["product_id", "location_id"],
            semantic_role: OdooSemanticRole::Identity,
        },
        OdooField {
            name: "product_id",
            kind: OdooFieldKind::Many2one,
            target: Some("product.product"),
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "location_id",
            kind: OdooFieldKind::Many2one,
            target: Some("stock.location"),
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "company_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.company"),
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "product_min_qty",
            kind: OdooFieldKind::Float,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // Trigger threshold: procure when qty_forecast < product_min_qty.
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "product_max_qty",
            kind: OdooFieldKind::Float,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // Target level: qty_to_order = max(min,max) − current virtual.
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "qty_to_order",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: Some("_compute_qty_to_order"),
            depends: &[
                "product_id",
                "location_id",
                "product_min_qty",
                "product_max_qty",
            ],
            // Rounded UP to replenishment_uom multiple; overridden by qty_to_order_manual.
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "qty_to_order_manual",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Manual override; cleared when trigger=auto fires.
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "qty_forecast",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: Some("_compute_qty_forecast"),
            depends: &["product_id", "location_id"],
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "replenishment_uom_id",
            kind: OdooFieldKind::Many2one,
            target: Some("uom.uom"),
            required: false,
            computed: None,
            depends: &[],
            // UoM multiple to round UP qty_to_order.
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "lead_days",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: Some("_compute_lead_days"),
            // sum of rule delays + horizon_time; used by ReorderTimingAdvisor.
            depends: &["route_id", "product_id"],
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "route_id",
            kind: OdooFieldKind::Many2one,
            target: Some("stock.route"),
            required: false,
            computed: None,
            depends: &[],
            // Procurement route for this orderpoint; narrows rule selection.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "trigger",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // 'automatic' (scheduler) | 'manual' (user-initiated replenishment).
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "deadline_date",
            kind: OdooFieldKind::Date,
            target: None,
            required: false,
            computed: Some("_compute_deadline_date"),
            // First day qty_forecast < product_min_qty within horizon; ReorderTimingAdvisor surface.
            depends: &["product_id", "location_id", "product_min_qty", "lead_days"],
            semantic_role: OdooSemanticRole::Date,
        },
    ],
    methods: &[
        OdooMethod {
            name: "_compute_qty_to_order",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_deadline_date",
            kind: OdooMethodKind::Compute,
            // R9: simulate daily net flow; first day below min = deadline = day − lead_days.
            // Savant surface: ReorderTimingAdvisor (evidence-weighted horizon_days/lead_days).
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_get_qty_to_order",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Number,
            triggers: &[],
        },
        OdooMethod {
            name: "_get_orderpoint_action",
            kind: OdooMethodKind::Action,
            // R15: virtual_available across replenishment locs; forecast<0 → lead-day re-read;
            // create/update manual orderpoints. Savant: ReplenishmentReportAdvisor.
            return_kind: OdooReturnKind::Action,
            triggers: &[],
        },
        OdooMethod {
            name: "_procure_orderpoint_confirm",
            kind: OdooMethodKind::Action,
            // R11: batch-1000 scheduler; date = lead_horizon noon → UTC; savepoint per rule.run.
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_quantity_in_progress",
            kind: OdooMethodKind::Helper,
            // Extended by purchase_stock to count open RFQs (open question Q5).
            return_kind: OdooReturnKind::Number,
            triggers: &[],
        },
        OdooMethod {
            name: "get_horizon_days",
            kind: OdooMethodKind::Helper,
            // R10: context.global_horizon_days else company.horizon_days (default 365).
            return_kind: OdooReturnKind::Number,
            triggers: &[],
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &[
                "product_id",
                "location_id",
                "product_min_qty",
                "product_max_qty",
            ],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["product_id", "location_id", "product_min_qty", "lead_days"],
        },
    ],
    state_machine: None,
    constraints: &[OdooConstraint {
        kind: OdooConstraintKind::Python,
        condition: "product_min_qty <= product_max_qty",
        source_method: None,
    }],
    provenance: OdooProvenance {
        l_doc: "L13-STOCK-VALUATION-PROCUREMENT.md",
        l_doc_lines: (55, 79),
        odoo_source: &[OdooSourceRef {
            path: "stock/models/stock_orderpoint.py",
            line_range: (1, 817),
        }],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── stock.rule (procurement-priority extension) ──────────────────────────────

pub const STOCK_RULE: OdooEntity = OdooEntity {
    model_name: "stock.rule",
    kind: OdooEntityKind::Model,
    description: "Procurement rule mapping (dest, route) → action (pull/push/transparent); \
                  L7 captured the basic shape; L13 focuses on procurement-priority resolution: \
                  _get_rule builds location-ancestor chain, _search_rule_for_warehouses orders by \
                  (route_seq, seq), first-match wins (R3 — ProcurementRuleSelector NARS surface); \
                  equal-sequence tiebreak is arbitrary Python-sort-stability (R16 — RouteTiebreaker \
                  NARS surface); _run_pull (R4) / _run_push (R6) / _get_stock_move_values (R5); \
                  OWL pivot fibo:Agreement (DOLCE Abstract). [L7 overlap: basic fields]",
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
            name: "action",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // 'pull'|'push'|'pull_push'|'manufacture'|'buy'.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "sequence",
            kind: OdooFieldKind::Integer,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // ORDER BY (route.sequence, rule.sequence); equal-sequence = RouteTiebreaker surface.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "route_id",
            kind: OdooFieldKind::Many2one,
            target: Some("stock.route"),
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "location_src_id",
            kind: OdooFieldKind::Many2one,
            target: Some("stock.location"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "location_dest_id",
            kind: OdooFieldKind::Many2one,
            target: Some("stock.location"),
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "delay",
            kind: OdooFieldKind::Integer,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Days to subtract from date_planned (_get_stock_move_values R5).
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "partner_address_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.partner"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "picking_type_id",
            kind: OdooFieldKind::Many2one,
            target: Some("stock.picking.type"),
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "propagate_cancel",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "location_dest_from_rule",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // True: use rule.location_dest_id; false: use picking_type default (_get_stock_move_values R5).
            semantic_role: OdooSemanticRole::Policy,
        },
    ],
    methods: &[
        OdooMethod {
            name: "_get_rule",
            kind: OdooMethodKind::Helper,
            // R3: build location ancestor chain → _search_rule_for_warehouses grouped by
            // (dest, warehouse, route) ORDER BY (route_seq, seq); first match wins.
            // SAVANT: ProcurementRuleSelector (NextBestAction, Induction, NarsTruth, Analytical).
            return_kind: OdooReturnKind::Record,
            triggers: &[],
        },
        OdooMethod {
            name: "_search_rule_for_warehouses",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Recordset,
            triggers: &[],
        },
        OdooMethod {
            name: "_run_pull",
            kind: OdooMethodKind::Action,
            // R4: validate location_src; sort positive-qty first; mts_else_mto→make_to_stock;
            // build move vals; group by company; create (sudo, with_company) + _action_confirm.
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_run_push",
            kind: OdooMethodKind::Action,
            // R6: transparent (modify dest in-place, recurse) vs manual (copy move, new_date = move.date + delay).
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_get_stock_move_values",
            kind: OdooMethodKind::Helper,
            // R5: date = date_planned - rule.delay; partner = rule.partner_address_id or values;
            // location_dest conditional on location_dest_from_rule; to_refund if qty<0.
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        OdooMethod {
            name: "_run_scheduler_tasks",
            kind: OdooMethodKind::Cron,
            // R12: compute qty_to_order → deadline → procure_confirm → assign (batch 1000).
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
    ],
    decorators: &[],
    state_machine: None,
    constraints: &[
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            // _run_pull: validate location_src exists when action=pull.
            condition: "action=pull requires location_src_id to be set",
            source_method: Some("_run_pull"),
        },
    ],
    provenance: OdooProvenance {
        l_doc: "L13-STOCK-VALUATION-PROCUREMENT.md",
        l_doc_lines: (38, 82),
        odoo_source: &[OdooSourceRef {
            path: "stock/models/stock_rule.py",
            line_range: (1, 747),
        }],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── stock.lot ────────────────────────────────────────────────────────────────

pub const STOCK_LOT: OdooEntity = OdooEntity {
    model_name: "stock.lot",
    kind: OdooEntityKind::Model,
    description: "Lot/serial number master; uniqueness per (product_id, company_id, name) \
                  with cross-company NULL check (R13); name auto-generation by incrementing \
                  last numeric segment with zero-fill (R14 — generate_lot_names); \
                  schema:ProductModel + GS1 pivot; DOLCE Endurant.",
    fields: &[
        OdooField {
            name: "name",
            kind: OdooFieldKind::Char,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // Natural key within (product_id, company_id). Auto-generated via lot_sequence_id.
            semantic_role: OdooSemanticRole::Identity,
        },
        OdooField {
            name: "product_id",
            kind: OdooFieldKind::Many2one,
            target: Some("product.product"),
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
            // NULL = shared cross-company; uniqueness spans all companies when NULL.
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "lot_sequence_id",
            kind: OdooFieldKind::Many2one,
            target: Some("ir.sequence"),
            required: false,
            computed: None,
            depends: &[],
            // Per-product DB sequence; next_by_id() drives generate_lot_names (Q6).
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "product_qty",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: Some("_compute_product_qty"),
            depends: &["quant_ids.quantity"],
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "expiration_date",
            kind: OdooFieldKind::Datetime,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // FEFO removal strategy reads this via removal_date on quant.
            semantic_role: OdooSemanticRole::Date,
        },
    ],
    methods: &[
        OdooMethod {
            name: "generate_lot_names",
            kind: OdooMethodKind::ApiModel,
            // R14: find last numeric segment (regex \d+), increment count, zfill to original width;
            // preserve prefix/suffix.
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        OdooMethod {
            name: "_get_next_serial",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Recordset,
            triggers: &[],
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiModel,
            targets: &[],
        },
    ],
    state_machine: None,
    constraints: &[
        OdooConstraint {
            kind: OdooConstraintKind::Sql,
            // Partial unique index (company_id IS NOT NULL case); app checks NULL-company case.
            condition: "UNIQUE(product_id, company_id, name) — DB partial unique; app cross-company NULL check",
            source_method: None,
        },
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            // R13: company_id NULL lots checked cross-company in Python (stock_lot.py:L103-126).
            condition: "when company_id IS NULL: name must be unique across all companies for this product",
            source_method: Some("_check_unique"),
        },
    ],
    provenance: OdooProvenance {
        l_doc: "L13-STOCK-VALUATION-PROCUREMENT.md",
        l_doc_lines: (71, 75),
        odoo_source: &[OdooSourceRef {
            path: "stock/models/stock_lot.py",
            line_range: (1, 431),
        }],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── res.company (valuation config) ───────────────────────────────────────────

const RES_COMPANY_VALUATION: OdooEntity = OdooEntity {
    model_name: "res.company",
    kind: OdooEntityKind::Model,
    description: "Company-level accounting/valuation configuration relevant to stock; \
                  anglo_saxon_accounting (R2): AngloSaxon = COGS deferred to invoice via \
                  expense_account_id + price_difference_account_id; Continental (DE default, \
                  GoBD-correct) = COGS at delivery; horizon_days (R10) = replenishment scheduler \
                  look-ahead (default 365 days); OWL pivot fibo:LegalEntity config \
                  (0x62 SMBAccounting family, DOLCE Abstract).",
    fields: &[
        OdooField {
            name: "anglo_saxon_accounting",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // True: COGS deferred to invoice; False (default): COGS at delivery (continental/GoBD).
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "expense_account_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.account"),
            required: false,
            computed: None,
            depends: &[],
            // Anglo-saxon: COGS interim account (Dr on delivery, reversed at invoice).
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "price_difference_account_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.account"),
            required: false,
            computed: None,
            depends: &[],
            // Anglo-saxon: absorbs std-vs-bill delta at invoice reconciliation.
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "horizon_days",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // R10: replenishment scheduler look-ahead (default 365). get_horizon_days reads
            // context.global_horizon_days first, else this field.
            semantic_role: OdooSemanticRole::Policy,
        },
    ],
    methods: &[],
    decorators: &[],
    state_machine: None,
    constraints: &[],
    provenance: OdooProvenance {
        l_doc: "L13-STOCK-VALUATION-PROCUREMENT.md",
        l_doc_lines: (35, 37),
        odoo_source: &[
            OdooSourceRef {
                path: "stock/models/res_company.py",
                line_range: (44, 47),
            },
            OdooSourceRef {
                path: "account/models/company.py",
                line_range: (146, 311),
            },
        ],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── ENTITIES slice ───────────────────────────────────────────────────────────

/// Entities documented in lane L13 (stock valuation + procurement rules
/// + reorder timing + replenishment + route tiebreak).
///
/// 5 entities:
///   [0] `stock.valuation.layer`      — SVL interface contract (FIFO/AVCO/standard, fresh woa-rs)
///   [1] `stock.warehouse.orderpoint` — min/max reorder, ReorderTimingAdvisor, ReplenishmentReport
///   [2] `stock.rule`                 — procurement priority walk, ProcurementRuleSelector,
///                                      RouteTiebreaker [L7 overlap: basic fields]
///   [3] `stock.lot`                  — lot/serial master, uniqueness, name generation
///   [4] `res.company` (valuation)    — anglo-saxon vs continental GL, horizon_days
///
/// Skipped:
///   - `procurement.group` — thin grouping model; no bespoke fields beyond name/partner;
///     referenced by move_ids as FK only.
///   - `stock.replenishment.option` — absent from community clone; wizard model for the
///     replenishment report UI (no field spec in L13 prose).
///   - `product.product` (standard_price only) — L13 R1 covers standard_price as a
///     company-dependent Float; full product.product lives in product-lane scope;
///     captured here via stock.valuation.layer.unit_cost semantics instead.
pub const ENTITIES: &[OdooEntity] = &[
    STOCK_VALUATION_LAYER,
    STOCK_WAREHOUSE_ORDERPOINT,
    STOCK_RULE,
    STOCK_LOT,
    RES_COMPANY_VALUATION,
];

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::odoo_blueprint::{OdooConfidence, OdooFieldKind, OdooSemanticRole};

    #[test]
    fn entities_slice_has_expected_count() {
        assert_eq!(ENTITIES.len(), 5);
    }

    #[test]
    fn all_entities_are_curated() {
        for entity in ENTITIES {
            assert_eq!(
                entity.provenance.confidence,
                OdooConfidence::Curated,
                "entity {} must be Curated",
                entity.model_name
            );
        }
    }

    #[test]
    fn all_entities_cite_l13_doc() {
        for entity in ENTITIES {
            assert_eq!(
                entity.provenance.l_doc, "L13-STOCK-VALUATION-PROCUREMENT.md",
                "entity {} must cite L13 doc",
                entity.model_name
            );
        }
    }

    #[test]
    fn svl_has_money_fields() {
        let unit_cost = STOCK_VALUATION_LAYER
            .fields
            .iter()
            .find(|f| f.name == "unit_cost")
            .expect("unit_cost must be on stock.valuation.layer");
        assert_eq!(unit_cost.semantic_role, OdooSemanticRole::Money);

        let value = STOCK_VALUATION_LAYER
            .fields
            .iter()
            .find(|f| f.name == "value")
            .expect("value must be on stock.valuation.layer");
        assert_eq!(value.kind, OdooFieldKind::Monetary);

        let remaining = STOCK_VALUATION_LAYER
            .fields
            .iter()
            .find(|f| f.name == "remaining_qty")
            .expect("remaining_qty must be on stock.valuation.layer");
        assert_eq!(remaining.semantic_role, OdooSemanticRole::Quantity);
    }

    #[test]
    fn svl_has_no_state_machine() {
        assert!(STOCK_VALUATION_LAYER.state_machine.is_none());
    }

    #[test]
    fn orderpoint_deadline_is_computed() {
        let f = STOCK_WAREHOUSE_ORDERPOINT
            .fields
            .iter()
            .find(|f| f.name == "deadline_date")
            .expect("deadline_date must be on stock.warehouse.orderpoint");
        assert_eq!(f.computed, Some("_compute_deadline_date"));
        assert_eq!(f.semantic_role, OdooSemanticRole::Date);
    }

    #[test]
    fn orderpoint_has_replenishment_methods() {
        let names: Vec<&str> = STOCK_WAREHOUSE_ORDERPOINT
            .methods
            .iter()
            .map(|m| m.name)
            .collect();
        assert!(names.contains(&"_get_orderpoint_action"));
        assert!(names.contains(&"_procure_orderpoint_confirm"));
        assert!(names.contains(&"_quantity_in_progress"));
        assert!(names.contains(&"get_horizon_days"));
    }

    #[test]
    fn orderpoint_trigger_is_policy() {
        let f = STOCK_WAREHOUSE_ORDERPOINT
            .fields
            .iter()
            .find(|f| f.name == "trigger")
            .expect("trigger must be on orderpoint");
        assert_eq!(f.semantic_role, OdooSemanticRole::Policy);
    }

    #[test]
    fn stock_rule_sequence_is_policy() {
        let f = STOCK_RULE
            .fields
            .iter()
            .find(|f| f.name == "sequence")
            .expect("sequence must be on stock.rule");
        assert_eq!(f.semantic_role, OdooSemanticRole::Policy);
    }

    #[test]
    fn stock_rule_has_procurement_methods() {
        let names: Vec<&str> = STOCK_RULE.methods.iter().map(|m| m.name).collect();
        assert!(names.contains(&"_get_rule"));
        assert!(names.contains(&"_run_pull"));
        assert!(names.contains(&"_run_push"));
        assert!(names.contains(&"_get_stock_move_values"));
        assert!(names.contains(&"_run_scheduler_tasks"));
    }

    #[test]
    fn stock_lot_has_uniqueness_constraint() {
        assert!(!STOCK_LOT.constraints.is_empty());
        let sql_c = STOCK_LOT
            .constraints
            .iter()
            .find(|c| c.kind == OdooConstraintKind::Sql);
        assert!(
            sql_c.is_some(),
            "stock.lot must have a SQL unique constraint"
        );
    }

    #[test]
    fn stock_lot_name_is_identity() {
        let f = STOCK_LOT
            .fields
            .iter()
            .find(|f| f.name == "name")
            .expect("name must be on stock.lot");
        assert_eq!(f.semantic_role, OdooSemanticRole::Identity);
        assert_eq!(f.kind, OdooFieldKind::Char);
    }

    #[test]
    fn res_company_valuation_has_anglo_saxon_policy() {
        let f = RES_COMPANY_VALUATION
            .fields
            .iter()
            .find(|f| f.name == "anglo_saxon_accounting")
            .expect("anglo_saxon_accounting must be on res.company valuation entity");
        assert_eq!(f.semantic_role, OdooSemanticRole::Policy);
        assert_eq!(f.kind, OdooFieldKind::Boolean);
    }

    #[test]
    fn res_company_valuation_has_horizon_days() {
        let f = RES_COMPANY_VALUATION
            .fields
            .iter()
            .find(|f| f.name == "horizon_days")
            .expect("horizon_days must be on res.company valuation entity");
        assert_eq!(f.kind, OdooFieldKind::Float);
        assert_eq!(f.semantic_role, OdooSemanticRole::Policy);
    }

    #[test]
    fn res_company_valuation_has_multi_source_refs() {
        assert_eq!(
            RES_COMPANY_VALUATION.provenance.odoo_source.len(),
            2,
            "res.company valuation spans stock/res_company.py + account/company.py"
        );
    }
}
