//! Lane L7 (STOCK) — typed Odoo entity declarations.
//!
//! Source: `.claude/odoo/L7-STOCK.md`.
//!
//! Entities covered (6):
//!   - `stock.move`      (R1 — state machine, _action_confirm/_action_assign/_action_done)
//!   - `stock.move.line` (R1 sub-model — quant-update driver)
//!   - `stock.quant`     (R2, R3, R5, R6 — availability + removal strategy)
//!   - `stock.picking`   (R7 — picking state machine + backorder judge)
//!   - `stock.location`  (R3 — removal strategy dispatch + bypass flag)
//!   - `stock.warehouse` (FLAG-4 — operational config; alignment needs authoring)
//!
//! NARS dispatch surfaces:
//!   - `RemovalStrategySelector` → stock.quant._gather + stock.location.removal_strategy_id
//!   - `MoveAssignmentPrioritizer` → stock.picking.action_assign sort policy
//!   - `BackorderJudge` → stock.picking._create_backorder / _check_backorder

use super::{
    OdooConfidence, OdooConstraint, OdooConstraintKind, OdooDecorator, OdooDecoratorKind,
    OdooEntity, OdooEntityKind, OdooField, OdooFieldKind, OdooMethod, OdooMethodKind,
    OdooProvenance, OdooReturnKind, OdooSemanticRole, OdooSourceRef, OdooState,
    OdooStateMachine, OdooStateSemantic, OdooTransition,
};

// ─── stock.move state machine ─────────────────────────────────────────────────

const STOCK_MOVE_STATE_MACHINE: OdooStateMachine = OdooStateMachine {
    state_field: "state",
    states: &[
        OdooState { name: "draft", semantic: OdooStateSemantic::Draft },
        OdooState { name: "waiting", semantic: OdooStateSemantic::InProgress },
        OdooState { name: "confirmed", semantic: OdooStateSemantic::InProgress },
        OdooState { name: "partially_available", semantic: OdooStateSemantic::InProgress },
        OdooState { name: "assigned", semantic: OdooStateSemantic::Active },
        OdooState { name: "done", semantic: OdooStateSemantic::Completed },
        OdooState { name: "cancel", semantic: OdooStateSemantic::Cancelled },
    ],
    transitions: &[
        OdooTransition { from: "draft", to: "waiting", trigger: "_action_confirm", guards: &[] },
        OdooTransition { from: "draft", to: "confirmed", trigger: "_action_confirm", guards: &[] },
        OdooTransition { from: "confirmed", to: "partially_available", trigger: "_action_assign", guards: &[] },
        OdooTransition { from: "confirmed", to: "assigned", trigger: "_action_assign", guards: &[] },
        OdooTransition { from: "waiting", to: "assigned", trigger: "_action_assign", guards: &[] },
        OdooTransition { from: "partially_available", to: "assigned", trigger: "_action_assign", guards: &[] },
        OdooTransition { from: "assigned", to: "done", trigger: "_action_done", guards: &[] },
        OdooTransition { from: "partially_available", to: "done", trigger: "_action_done", guards: &[] },
        OdooTransition { from: "confirmed", to: "cancel", trigger: "_action_cancel", guards: &[] },
        OdooTransition { from: "waiting", to: "cancel", trigger: "_action_cancel", guards: &[] },
        OdooTransition { from: "assigned", to: "cancel", trigger: "_action_cancel", guards: &[] },
        OdooTransition { from: "partially_available", to: "cancel", trigger: "_action_cancel", guards: &[] },
    ],
};

// ─── stock.move ───────────────────────────────────────────────────────────────

const STOCK_MOVE: OdooEntity = OdooEntity {
    model_name: "stock.move",
    kind: OdooEntityKind::Model,
    description: "One product movement between two stock locations; state machine \
                  (draft→confirmed→assigned→done/cancel); drives quant reservation \
                  via _action_assign; quant mutation via move_line_ids._action_done \
                  (K10, DOLCE Perdurant, FLAG-1). _recompute_state re-derives state \
                  live from quantity vs product_uom_qty — NOT a simple FSM.",
    fields: &[
        OdooField { name: "name", kind: OdooFieldKind::Char, target: None, required: true, computed: None, depends: &[], semantic_role: OdooSemanticRole::Identity },
        OdooField { name: "state", kind: OdooFieldKind::Selection, target: None, required: false, computed: Some("_recompute_state"), depends: &["quantity", "product_uom_qty", "move_orig_ids.state"], semantic_role: OdooSemanticRole::Status },
        OdooField { name: "product_id", kind: OdooFieldKind::Many2one, target: Some("product.product"), required: true, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        // Demand qty in move UoM. Always write product_uom_qty, not product_qty.
        OdooField { name: "product_uom_qty", kind: OdooFieldKind::Float, target: None, required: true, computed: None, depends: &[], semantic_role: OdooSemanticRole::Quantity },
        // READ-ONLY computed — writes raise UserError. product_uom._compute_quantity(HALF-UP).
        OdooField { name: "product_qty", kind: OdooFieldKind::Float, target: None, required: false, computed: Some("_compute_product_qty"), depends: &["product_uom_qty", "product_uom", "product_id.uom_id"], semantic_role: OdooSemanticRole::Quantity },
        OdooField { name: "quantity", kind: OdooFieldKind::Float, target: None, required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Quantity },
        OdooField { name: "product_uom", kind: OdooFieldKind::Many2one, target: Some("uom.uom"), required: true, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "location_id", kind: OdooFieldKind::Many2one, target: Some("stock.location"), required: true, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "location_dest_id", kind: OdooFieldKind::Many2one, target: Some("stock.location"), required: true, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "picking_id", kind: OdooFieldKind::Many2one, target: Some("stock.picking"), required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "move_line_ids", kind: OdooFieldKind::One2many, target: Some("stock.move.line"), required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "move_orig_ids", kind: OdooFieldKind::Many2many, target: Some("stock.move"), required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "move_dest_ids", kind: OdooFieldKind::Many2many, target: Some("stock.move"), required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        // 'make_to_stock'|'make_to_order'|'mts_else_mto'. Branch selector in _action_confirm/_action_assign.
        OdooField { name: "procure_method", kind: OdooFieldKind::Selection, target: None, required: true, computed: None, depends: &[], semantic_role: OdooSemanticRole::Policy },
        // 'at_confirm'|'manual'|'by_date'. Gate for _should_assign_at_confirm.
        OdooField { name: "reservation_method", kind: OdooFieldKind::Selection, target: None, required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Policy },
        OdooField { name: "reservation_date", kind: OdooFieldKind::Date, target: None, required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Date },
        // '0' normal | '1' urgent. action_assign sorts by -int(priority) first.
        OdooField { name: "priority", kind: OdooFieldKind::Selection, target: None, required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Policy },
        OdooField { name: "date_deadline", kind: OdooFieldKind::Datetime, target: None, required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Date },
        // True: cancel propagates to move_dest_ids when ALL siblings cancelled.
        OdooField { name: "propagate_cancel", kind: OdooFieldKind::Boolean, target: None, required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Policy },
        OdooField { name: "picked", kind: OdooFieldKind::Boolean, target: None, required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Status },
        OdooField { name: "price_unit", kind: OdooFieldKind::Float, target: None, required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Money },
        // min(product_qty, quant.available_qty) or product_qty when state=done.
        OdooField { name: "availability", kind: OdooFieldKind::Float, target: None, required: false, computed: Some("_compute_product_availability"), depends: &["state", "product_id", "product_uom_qty", "location_id"], semantic_role: OdooSemanticRole::Quantity },
        OdooField { name: "rule_id", kind: OdooFieldKind::Many2one, target: Some("stock.rule"), required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
    ],
    methods: &[
        OdooMethod { name: "_action_confirm", kind: OdooMethodKind::Action, return_kind: OdooReturnKind::Self_, triggers: &["waiting", "confirmed"] },
        OdooMethod { name: "_action_assign", kind: OdooMethodKind::Action, return_kind: OdooReturnKind::Unit, triggers: &["partially_available", "assigned"] },
        OdooMethod { name: "_action_done", kind: OdooMethodKind::Action, return_kind: OdooReturnKind::Recordset, triggers: &["done"] },
        OdooMethod { name: "_action_cancel", kind: OdooMethodKind::Action, return_kind: OdooReturnKind::Self_, triggers: &["cancel"] },
        OdooMethod { name: "_recompute_state", kind: OdooMethodKind::Compute, return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_compute_product_qty", kind: OdooMethodKind::Compute, return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_compute_product_availability", kind: OdooMethodKind::Compute, return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_create_backorder", kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Self_, triggers: &[] },
        OdooMethod { name: "_split", kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Record, triggers: &[] },
        OdooMethod { name: "_get_available_quantity", kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Number, triggers: &[] },
        OdooMethod { name: "_update_reserved_quantity", kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_should_bypass_reservation", kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Boolean, triggers: &[] },
        OdooMethod { name: "_trigger_assign", kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_push_apply", kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Unit, triggers: &[] },
    ],
    decorators: &[
        OdooDecorator { kind: OdooDecoratorKind::ApiDepends, targets: &["product_uom_qty", "product_uom", "product_id.uom_id"] },
        OdooDecorator { kind: OdooDecoratorKind::ApiDepends, targets: &["quantity", "product_uom_qty", "move_orig_ids.state"] },
    ],
    state_machine: Some(&STOCK_MOVE_STATE_MACHINE),
    constraints: &[
        OdooConstraint { kind: OdooConstraintKind::Python, condition: "product_qty is read-only; writing raises UserError — always write product_uom_qty", source_method: None },
        OdooConstraint { kind: OdooConstraintKind::Python, condition: "cancel blocked when state=done and location_dest_usage != 'inventory'", source_method: Some("_action_cancel") },
    ],
    provenance: OdooProvenance {
        l_doc: "L7-STOCK.md",
        l_doc_lines: (31, 108),
        odoo_source: &[OdooSourceRef { path: "addons/stock/models/stock_move.py", line_range: (107, 2503) }],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── stock.move.line ──────────────────────────────────────────────────────────

const STOCK_MOVE_LINE: OdooEntity = OdooEntity {
    model_name: "stock.move.line",
    kind: OdooEntityKind::Model,
    description: "One lot/package/owner reservation or done-qty record within a stock move; \
                  _action_done drives quant mutation via stock.quant._update_available_quantity; \
                  serial products get one move line per unit (K10, DOLCE Perdurant).",
    fields: &[
        OdooField { name: "move_id", kind: OdooFieldKind::Many2one, target: Some("stock.move"), required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "picking_id", kind: OdooFieldKind::Many2one, target: Some("stock.picking"), required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "product_id", kind: OdooFieldKind::Many2one, target: Some("product.product"), required: true, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "location_id", kind: OdooFieldKind::Many2one, target: Some("stock.location"), required: true, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "location_dest_id", kind: OdooFieldKind::Many2one, target: Some("stock.location"), required: true, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "lot_id", kind: OdooFieldKind::Many2one, target: Some("stock.lot"), required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "package_id", kind: OdooFieldKind::Many2one, target: Some("stock.quant.package"), required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "owner_id", kind: OdooFieldKind::Many2one, target: Some("res.partner"), required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "quantity", kind: OdooFieldKind::Float, target: None, required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Quantity },
        OdooField { name: "reserved_qty", kind: OdooFieldKind::Float, target: None, required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Quantity },
        OdooField { name: "picked", kind: OdooFieldKind::Boolean, target: None, required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Status },
    ],
    methods: &[
        OdooMethod { name: "_action_done", kind: OdooMethodKind::Action, return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_apply_putaway_strategy", kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Unit, triggers: &[] },
    ],
    decorators: &[],
    state_machine: None,
    constraints: &[OdooConstraint {
        kind: OdooConstraintKind::Python,
        condition: "serial-tracked products: exactly one move line per unit (quantity = 1)",
        source_method: None,
    }],
    provenance: OdooProvenance {
        l_doc: "L7-STOCK.md",
        l_doc_lines: (77, 108),
        odoo_source: &[OdooSourceRef { path: "addons/stock/models/stock_move.py", line_range: (1763, 1820) }],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── stock.quant ──────────────────────────────────────────────────────────────

const STOCK_QUANT: OdooEntity = OdooEntity {
    model_name: "stock.quant",
    kind: OdooEntityKind::Model,
    description: "Persistent stock record: qty of one product at one location with \
                  specific lot/package/owner; reserved_quantity >= 0 invariant; \
                  _gather = RemovalStrategySelector dispatch surface (FIFO/LIFO/FEFO/ \
                  closest/least_packages, Axis-2 NARS, XorBundle); \
                  in_date = min(incoming_dates) — FIFO invariant at quant level \
                  (K10, DOLCE Endurant, FLAG-3).",
    fields: &[
        OdooField { name: "product_id", kind: OdooFieldKind::Many2one, target: Some("product.product"), required: true, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "location_id", kind: OdooFieldKind::Many2one, target: Some("stock.location"), required: true, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "lot_id", kind: OdooFieldKind::Many2one, target: Some("stock.lot"), required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "package_id", kind: OdooFieldKind::Many2one, target: Some("stock.quant.package"), required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "owner_id", kind: OdooFieldKind::Many2one, target: Some("res.partner"), required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "quantity", kind: OdooFieldKind::Float, target: None, required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Quantity },
        OdooField {
            name: "reserved_quantity",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // max(0, reserved + delta) — never negative (hard floor in _update_available_quantity).
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "in_date",
            kind: OdooFieldKind::Datetime,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // min(incoming_dates) after merge — FIFO invariant preserved at quant level.
            semantic_role: OdooSemanticRole::Date,
        },
        OdooField {
            name: "removal_date",
            kind: OdooFieldKind::Datetime,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // FEFO: domain filter removal_date >= threshold OR NULL in _get_gather_domain.
            semantic_role: OdooSemanticRole::Date,
        },
        OdooField { name: "inventory_quantity", kind: OdooFieldKind::Float, target: None, required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Quantity },
        OdooField { name: "inventory_quantity_set", kind: OdooFieldKind::Boolean, target: None, required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Status },
    ],
    methods: &[
        OdooMethod { name: "_get_available_quantity", kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Number, triggers: &[] },
        OdooMethod { name: "_gather", kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Recordset, triggers: &[] },
        OdooMethod { name: "_get_removal_strategy", kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Dict, triggers: &[] },
        OdooMethod { name: "_get_gather_domain", kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Dict, triggers: &[] },
        OdooMethod { name: "_get_reserve_quantity", kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Dict, triggers: &[] },
        OdooMethod { name: "_update_available_quantity", kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Dict, triggers: &[] },
        OdooMethod { name: "_apply_inventory", kind: OdooMethodKind::Action, return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_merge_quants", kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_unlink_zero_quants", kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_run_least_packages_removal_strategy_astar", kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Recordset, triggers: &[] },
    ],
    decorators: &[OdooDecorator { kind: OdooDecoratorKind::ApiAutovacuum, targets: &[] }],
    state_machine: None,
    constraints: &[
        OdooConstraint { kind: OdooConstraintKind::Sql, condition: "UNIQUE(product_id, location_id, lot_id, package_id, owner_id)", source_method: None },
        OdooConstraint { kind: OdooConstraintKind::Python, condition: "reserved_quantity >= 0 (max(0, reserved + delta) in _update_available_quantity)", source_method: Some("_update_available_quantity") },
        OdooConstraint { kind: OdooConstraintKind::Python, condition: "serial tracking: fractional reserve qty clamped to 0 (_get_reserve_quantity)", source_method: Some("_get_reserve_quantity") },
    ],
    provenance: OdooProvenance {
        l_doc: "L7-STOCK.md",
        l_doc_lines: (110, 415),
        odoo_source: &[OdooSourceRef { path: "addons/stock/models/stock_quant.py", line_range: (617, 1138) }],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── stock.picking state machine ──────────────────────────────────────────────

const STOCK_PICKING_STATE_MACHINE: OdooStateMachine = OdooStateMachine {
    state_field: "state",
    states: &[
        OdooState { name: "draft", semantic: OdooStateSemantic::Draft },
        OdooState { name: "waiting", semantic: OdooStateSemantic::InProgress },
        OdooState { name: "confirmed", semantic: OdooStateSemantic::InProgress },
        // partially_available move → 'assigned' picking when move_type='direct' (_compute_state L858).
        OdooState { name: "assigned", semantic: OdooStateSemantic::Active },
        OdooState { name: "done", semantic: OdooStateSemantic::Completed },
        OdooState { name: "cancel", semantic: OdooStateSemantic::Cancelled },
    ],
    transitions: &[
        OdooTransition { from: "draft", to: "confirmed", trigger: "action_confirm", guards: &[] },
        OdooTransition { from: "confirmed", to: "assigned", trigger: "action_assign", guards: &[] },
        OdooTransition { from: "assigned", to: "done", trigger: "button_validate", guards: &["_sanity_check"] },
        OdooTransition { from: "confirmed", to: "done", trigger: "button_validate", guards: &["_sanity_check"] },
    ],
};

// ─── stock.picking ────────────────────────────────────────────────────────────

const STOCK_PICKING: OdooEntity = OdooEntity {
    model_name: "stock.picking",
    kind: OdooEntityKind::Model,
    description: "Group of stock moves for one logistics operation (receipt/delivery/internal); \
                  state COMPUTED from move states (not stored); \
                  BackorderJudge dispatch surface on button_validate; \
                  MoveAssignmentPrioritizer on action_assign sort \
                  (-priority, not date_deadline, date_deadline, date, id) \
                  (K10, DOLCE Perdurant, FLAG-2).",
    fields: &[
        OdooField { name: "name", kind: OdooFieldKind::Char, target: None, required: false, computed: Some("_compute_name"), depends: &[], semantic_role: OdooSemanticRole::Identity },
        OdooField {
            name: "state",
            kind: OdooFieldKind::Selection,
            target: None,
            required: false,
            computed: Some("_compute_state"),
            // Aggregated from move states — NOT stored independently.
            depends: &["move_ids.state", "move_ids.is_locked", "move_ids.quantity"],
            semantic_role: OdooSemanticRole::Status,
        },
        OdooField { name: "move_ids", kind: OdooFieldKind::One2many, target: Some("stock.move"), required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "move_line_ids", kind: OdooFieldKind::One2many, target: Some("stock.move.line"), required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField {
            name: "picking_type_id",
            kind: OdooFieldKind::Many2one,
            target: Some("stock.picking.type"),
            required: true,
            computed: None,
            depends: &[],
            // Carries create_backorder ('ask'|'always'|'never') + reservation_method policy.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField { name: "location_id", kind: OdooFieldKind::Many2one, target: Some("stock.location"), required: true, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "location_dest_id", kind: OdooFieldKind::Many2one, target: Some("stock.location"), required: true, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "partner_id", kind: OdooFieldKind::Many2one, target: Some("res.partner"), required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField {
            name: "move_type",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // 'direct' (partial ok) | 'one' (all-at-once). Controls _get_relevant_state_among_moves.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField { name: "backorder_id", kind: OdooFieldKind::Many2one, target: Some("stock.picking"), required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "user_id", kind: OdooFieldKind::Many2one, target: Some("res.users"), required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "priority", kind: OdooFieldKind::Selection, target: None, required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Policy },
        OdooField { name: "date_deadline", kind: OdooFieldKind::Datetime, target: None, required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Date },
    ],
    methods: &[
        OdooMethod { name: "_compute_state", kind: OdooMethodKind::Compute, return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "action_confirm", kind: OdooMethodKind::Action, return_kind: OdooReturnKind::Unit, triggers: &["confirmed"] },
        OdooMethod { name: "action_assign", kind: OdooMethodKind::Action, return_kind: OdooReturnKind::Unit, triggers: &["assigned"] },
        OdooMethod { name: "button_validate", kind: OdooMethodKind::Action, return_kind: OdooReturnKind::Action, triggers: &["done"] },
        OdooMethod { name: "_create_backorder", kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Recordset, triggers: &[] },
        OdooMethod { name: "_check_backorder", kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Recordset, triggers: &[] },
        OdooMethod { name: "_sanity_check", kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Boolean, triggers: &[] },
        OdooMethod { name: "_action_done", kind: OdooMethodKind::Action, return_kind: OdooReturnKind::Unit, triggers: &[] },
    ],
    decorators: &[OdooDecorator {
        kind: OdooDecoratorKind::ApiDepends,
        targets: &["move_ids.state", "move_ids.is_locked", "move_ids.quantity"],
    }],
    state_machine: Some(&STOCK_PICKING_STATE_MACHINE),
    constraints: &[OdooConstraint {
        kind: OdooConstraintKind::Python,
        condition: "button_validate blocked when no moves with quantity > 0; lots required per picking_type",
        source_method: Some("_sanity_check"),
    }],
    provenance: OdooProvenance {
        l_doc: "L7-STOCK.md",
        l_doc_lines: (419, 500),
        odoo_source: &[OdooSourceRef { path: "addons/stock/models/stock_picking.py", line_range: (575, 1602) }],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── stock.location ───────────────────────────────────────────────────────────

const STOCK_LOCATION: OdooEntity = OdooEntity {
    model_name: "stock.location",
    kind: OdooEntityKind::Model,
    description: "Node in the stock location hierarchy (physical or virtual); \
                  removal_strategy_id = RemovalStrategySelector dispatch surface \
                  (product.categ wins over location walk, default 'fifo'); \
                  should_bypass_reservation() short-circuits quant ops for \
                  virtual/supplier/customer locations (K10, DOLCE Endurant).",
    fields: &[
        OdooField { name: "name", kind: OdooFieldKind::Char, target: None, required: true, computed: None, depends: &[], semantic_role: OdooSemanticRole::Identity },
        OdooField {
            name: "complete_name",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: Some("_compute_complete_name"),
            // Used in 'closest' removal strategy sort: sorted by complete_name ASC.
            depends: &["name", "location_id.complete_name"],
            semantic_role: OdooSemanticRole::Identity,
        },
        OdooField { name: "location_id", kind: OdooFieldKind::Many2one, target: Some("stock.location"), required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField {
            name: "usage",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // 'internal'|'customer'|'supplier'|'inventory'|'production'|'transit'|'view'.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "removal_strategy_id",
            kind: OdooFieldKind::Many2one,
            target: Some("product.removal"),
            required: false,
            computed: None,
            depends: &[],
            // RemovalStrategySelector dispatch. product.categ wins; walk hierarchy; default fifo.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField { name: "active", kind: OdooFieldKind::Boolean, target: None, required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Status },
        OdooField { name: "last_inventory_date", kind: OdooFieldKind::Date, target: None, required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Audit },
    ],
    methods: &[
        OdooMethod { name: "should_bypass_reservation", kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Boolean, triggers: &[] },
        OdooMethod { name: "_compute_complete_name", kind: OdooMethodKind::Compute, return_kind: OdooReturnKind::Unit, triggers: &[] },
    ],
    decorators: &[OdooDecorator { kind: OdooDecoratorKind::ApiDepends, targets: &["name", "location_id.complete_name"] }],
    state_machine: None,
    constraints: &[OdooConstraint {
        kind: OdooConstraintKind::Domain,
        condition: "child_of domain in _get_gather_domain (non-strict) spans all sub-locations",
        source_method: None,
    }],
    provenance: OdooProvenance {
        l_doc: "L7-STOCK.md",
        l_doc_lines: (155, 207),
        odoo_source: &[OdooSourceRef { path: "addons/stock/models/stock_quant.py", line_range: (617, 791) }],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── stock.warehouse ──────────────────────────────────────────────────────────

const STOCK_WAREHOUSE: OdooEntity = OdooEntity {
    model_name: "stock.warehouse",
    kind: OdooEntityKind::Model,
    description: "Physical warehouse site with operational config (picking types, routes, \
                  multi-step delivery/reception policy); alignment UNRESOLVED — \
                  proposed gs1:Location (K10, DOLCE Endurant, FLAG-4). \
                  TODO: alignment row needed before OGIT hydration.",
    fields: &[
        OdooField { name: "name", kind: OdooFieldKind::Char, target: None, required: true, computed: None, depends: &[], semantic_role: OdooSemanticRole::Identity },
        OdooField { name: "code", kind: OdooFieldKind::Char, target: None, required: true, computed: None, depends: &[], semantic_role: OdooSemanticRole::Identity },
        OdooField { name: "partner_id", kind: OdooFieldKind::Many2one, target: Some("res.partner"), required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Address },
        OdooField { name: "lot_stock_id", kind: OdooFieldKind::Many2one, target: Some("stock.location"), required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField {
            name: "reception_steps",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // 'one_step'|'two_steps'|'three_steps'.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "delivery_steps",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // 'ship_only'|'pick_ship'|'pick_pack_ship'.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField { name: "route_ids", kind: OdooFieldKind::Many2many, target: Some("stock.route"), required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "company_id", kind: OdooFieldKind::Many2one, target: Some("res.company"), required: true, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
    ],
    methods: &[
        OdooMethod { name: "_get_picking_type_create_edit", kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Dict, triggers: &[] },
        OdooMethod { name: "_update_reception_delivery", kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Unit, triggers: &[] },
    ],
    decorators: &[],
    state_machine: None,
    constraints: &[OdooConstraint {
        kind: OdooConstraintKind::Sql,
        condition: "UNIQUE(code, company_id) — warehouse code unique per company",
        source_method: None,
    }],
    provenance: OdooProvenance {
        l_doc: "L7-STOCK.md",
        l_doc_lines: (608, 644),
        odoo_source: &[OdooSourceRef {
            // TODO: FLAG-4 — alignment row needed (proposed gs1:Location).
            path: "addons/stock/models/stock_warehouse.py",
            line_range: (1, 200),
        }],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── ENTITIES slice ───────────────────────────────────────────────────────────

/// Entities documented in lane L7 (stock moves + quants + reservations
/// + removal strategy + backorder).
///
/// 6 entities:
///   [0] `stock.move`      — state machine (draft→confirmed→assigned→done), reservation loop
///   [1] `stock.move.line` — quant-update driver, lot/serial tracking
///   [2] `stock.quant`     — availability arithmetic, removal strategy, inventory adjustment
///   [3] `stock.picking`   — picking state machine, backorder judge, assignment sort
///   [4] `stock.location`  — removal strategy dispatch, bypass reservation
///   [5] `stock.warehouse` — operational config (FLAG-4: needs alignment authoring)
///
/// Skipped (not primary L7 entities):
///   - `stock.rule`         — procurement rules; separate lane concern.
///   - `stock.lot`          — lot/serial master; referenced as FK only.
///   - `stock.picking.type` — config model; captured inline via picking_type_id.
///   - `product.removal`    — strategy config; captured via removal_strategy_id.
pub const ENTITIES: &[OdooEntity] = &[
    STOCK_MOVE,
    STOCK_MOVE_LINE,
    STOCK_QUANT,
    STOCK_PICKING,
    STOCK_LOCATION,
    STOCK_WAREHOUSE,
];

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::odoo_blueprint::{OdooConfidence, OdooFieldKind, OdooSemanticRole};

    #[test]
    fn entities_slice_has_expected_count() {
        assert_eq!(ENTITIES.len(), 6);
    }

    #[test]
    fn stock_move_has_state_machine() {
        let sm = STOCK_MOVE.state_machine.expect("stock.move must have a state machine");
        assert_eq!(sm.state_field, "state");
        assert_eq!(sm.states.len(), 7);
        assert!(!sm.transitions.is_empty());
    }

    #[test]
    fn stock_move_state_machine_includes_partially_available() {
        let sm = STOCK_MOVE.state_machine.unwrap();
        let names: Vec<&str> = sm.states.iter().map(|s| s.name).collect();
        assert!(names.contains(&"partially_available"));
        assert!(names.contains(&"waiting"));
        assert!(names.contains(&"assigned"));
    }

    #[test]
    fn stock_move_product_qty_is_computed() {
        let f = STOCK_MOVE.fields.iter().find(|f| f.name == "product_qty").unwrap();
        assert_eq!(f.computed, Some("_compute_product_qty"));
        assert_eq!(f.kind, OdooFieldKind::Float);
    }
    #[test]
    fn stock_move_reservation_policy_fields_present() {
        let ns: Vec<&str> = STOCK_MOVE.fields.iter().map(|f| f.name).collect();
        assert!(ns.contains(&"reservation_method") && ns.contains(&"reservation_date")
            && ns.contains(&"procure_method") && ns.contains(&"priority"));
    }
    #[test]
    fn stock_quant_reservation_invariant_captured() {
        let f = STOCK_QUANT.fields.iter().find(|f| f.name == "reserved_quantity").unwrap();
        assert_eq!(f.semantic_role, OdooSemanticRole::Quantity);
        assert!(!STOCK_QUANT.constraints.is_empty());
    }
    #[test]
    fn stock_quant_has_removal_strategy_methods() {
        let ns: Vec<&str> = STOCK_QUANT.methods.iter().map(|m| m.name).collect();
        assert!(ns.contains(&"_gather") && ns.contains(&"_get_removal_strategy")
            && ns.contains(&"_get_reserve_quantity") && ns.contains(&"_update_available_quantity"));
    }
    #[test]
    fn stock_picking_state_is_computed() {
        let f = STOCK_PICKING.fields.iter().find(|f| f.name == "state").unwrap();
        assert_eq!(f.computed, Some("_compute_state"));
    }
    #[test]
    fn stock_picking_has_state_machine() {
        let sm = STOCK_PICKING.state_machine.expect("stock.picking must have a state machine");
        assert_eq!(sm.state_field, "state");
        assert_eq!(sm.states.len(), 6);
    }
    #[test]
    fn stock_picking_move_type_is_policy() {
        let f = STOCK_PICKING.fields.iter().find(|f| f.name == "move_type").unwrap();
        assert_eq!(f.semantic_role, OdooSemanticRole::Policy);
    }
    #[test]
    fn stock_location_removal_strategy_is_policy() {
        let f = STOCK_LOCATION.fields.iter().find(|f| f.name == "removal_strategy_id").unwrap();
        assert_eq!(f.kind, OdooFieldKind::Many2one);
        assert_eq!(f.semantic_role, OdooSemanticRole::Policy);
    }
    #[test]
    fn stock_location_has_bypass_method() {
        assert!(STOCK_LOCATION.methods.iter().any(|m| m.name == "should_bypass_reservation"));
    }
    #[test]
    fn stock_warehouse_has_steps_policy_fields() {
        let ns: Vec<&str> = STOCK_WAREHOUSE.fields.iter().map(|f| f.name).collect();
        assert!(ns.contains(&"reception_steps") && ns.contains(&"delivery_steps"));
    }
    #[test]
    fn all_entities_are_curated() {
        for e in ENTITIES {
            assert_eq!(e.provenance.confidence, OdooConfidence::Curated, "{} must be Curated", e.model_name);
        }
    }
    #[test]
    fn all_entities_cite_l7_doc() {
        for e in ENTITIES {
            assert_eq!(e.provenance.l_doc, "L7-STOCK.md", "{} must cite L7 doc", e.model_name);
        }
    }
}
