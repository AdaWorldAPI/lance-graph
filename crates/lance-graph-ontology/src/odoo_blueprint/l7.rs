//! Lane L7 (STOCK) — typed Odoo entity declarations.
//!
//! Source: `.claude/odoo/L7-STOCK.md`.
//!
//! Entities covered (6):
//!   - `stock.move`       (R1 — state machine, `_action_confirm`/`_action_assign`/`_action_done`)
//!   - `stock.move.line`  (R1 sub-model — actual quant update driver)
//!   - `stock.quant`      (R2, R3, R5, R6 — availability arithmetic + removal strategy)
//!   - `stock.picking`    (R7 — picking state machine + backorder judge)
//!   - `stock.location`   (R3 — removal strategy dispatch + bypass reservation flag)
//!   - `stock.warehouse`  (FLAG-4 — operational configuration, routes)
//!
//! NARS dispatch surfaces captured:
//!   - `RemovalStrategySelector` → `stock.quant._gather` + `stock.location.removal_strategy_id`
//!   - `MoveAssignmentPrioritizer` → `stock.picking.action_assign` sort policy
//!   - `BackorderJudge` → `stock.picking._create_backorder` / `_check_backorder`

use super::{
    OdooConfidence, OdooConstraint, OdooConstraintKind, OdooDecorator, OdooDecoratorKind,
    OdooEntity, OdooField, OdooFieldKind, OdooMethod, OdooMethodKind, OdooProvenance,
    OdooReturnKind, OdooSemanticRole, OdooSourceRef, OdooState, OdooStateMachine,
    OdooStateSemantic, OdooTransition,
};

// ─── stock.move state machine ─────────────────────────────────────────────────
//
// Source: stock_move.py L107–120 (state field), L1546–1641 (_action_confirm),
//         L1901–2043 (_action_assign), L2101–2169 (_action_done),
//         L2044–2084 (_action_cancel), L2268–2288 (_recompute_state).
//
// Note: `partially_available` is an intermediate state — `_recompute_state`
// re-derives state live from `quantity` vs `product_uom_qty` at any time.

const STOCK_MOVE_STATE_MACHINE: OdooStateMachine = OdooStateMachine {
    state_field: "state",
    states: &[
        OdooState { name: "draft", semantic: OdooStateSemantic::Draft },
        // Waiting: upstream move not done OR make_to_order with no procurement.
        OdooState { name: "waiting", semantic: OdooStateSemantic::InProgress },
        // Confirmed: waiting for reservable stock.
        OdooState { name: "confirmed", semantic: OdooStateSemantic::InProgress },
        // Partially reserved — `_recompute_state` keeps this live.
        OdooState { name: "partially_available", semantic: OdooStateSemantic::InProgress },
        // Fully reserved — all product_uom_qty covered by quants.
        OdooState { name: "assigned", semantic: OdooStateSemantic::Active },
        OdooState { name: "done", semantic: OdooStateSemantic::Completed },
        OdooState { name: "cancel", semantic: OdooStateSemantic::Cancelled },
    ],
    transitions: &[
        OdooTransition {
            from: "draft",
            to: "waiting",
            trigger: "_action_confirm",
            guards: &[],
            // Fires when move_orig_ids non-empty OR procure_method='make_to_order'.
        },
        OdooTransition {
            from: "draft",
            to: "confirmed",
            trigger: "_action_confirm",
            guards: &[],
            // Fires for make_to_stock moves without upstream dependency.
        },
        OdooTransition {
            from: "confirmed",
            to: "partially_available",
            trigger: "_action_assign",
            guards: &[],
        },
        OdooTransition {
            from: "confirmed",
            to: "assigned",
            trigger: "_action_assign",
            guards: &[],
        },
        OdooTransition {
            from: "waiting",
            to: "assigned",
            trigger: "_action_assign",
            guards: &[],
        },
        OdooTransition {
            from: "partially_available",
            to: "assigned",
            trigger: "_action_assign",
            guards: &[],
        },
        OdooTransition {
            from: "assigned",
            to: "done",
            trigger: "_action_done",
            guards: &[],
        },
        OdooTransition {
            from: "partially_available",
            to: "done",
            trigger: "_action_done",
            guards: &[],
        },
        OdooTransition {
            // Any non-done state → cancel (except done with location_dest_usage=inventory guard).
            from: "confirmed",
            to: "cancel",
            trigger: "_action_cancel",
            guards: &[],
        },
        OdooTransition {
            from: "waiting",
            to: "cancel",
            trigger: "_action_cancel",
            guards: &[],
        },
        OdooTransition {
            from: "assigned",
            to: "cancel",
            trigger: "_action_cancel",
            guards: &[],
        },
        OdooTransition {
            from: "partially_available",
            to: "cancel",
            trigger: "_action_cancel",
            guards: &[],
        },
    ],
};

// ─── stock.move ───────────────────────────────────────────────────────────────

const STOCK_MOVE: OdooEntity = OdooEntity {
    model_name: "stock.move",
    description: "A single product movement event between two stock locations; \
                  carries a state machine (draft→confirmed→assigned→done/cancel); \
                  drives quant reservation via _action_assign and quant mutation \
                  via move_line_ids._action_done (K10, DOLCE Perdurant, FLAG-1).",
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
            name: "state",
            kind: OdooFieldKind::Selection,
            target: None,
            required: false,
            computed: Some("_recompute_state"),
            // copy=False, index=True, readonly=True; re-derived from quantity vs product_uom_qty.
            depends: &["quantity", "product_uom_qty", "move_orig_ids.state"],
            semantic_role: OdooSemanticRole::Status,
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
            name: "product_uom_qty",
            kind: OdooFieldKind::Float,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // Demand quantity in move UoM.  Writing product_qty raises UserError.
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "product_qty",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: Some("_compute_product_qty"),
            // product_uom._compute_quantity(product_uom_qty, product_id.uom_id, HALF-UP).
            // READ-ONLY: writes raise UserError — always write product_uom_qty instead.
            depends: &["product_uom_qty", "product_uom", "product_id.uom_id"],
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "quantity",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Actual done/reserved quantity; triggers _recompute_state when written.
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "product_uom",
            kind: OdooFieldKind::Many2one,
            target: Some("uom.uom"),
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
            name: "location_dest_id",
            kind: OdooFieldKind::Many2one,
            target: Some("stock.location"),
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "picking_id",
            kind: OdooFieldKind::Many2one,
            target: Some("stock.picking"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "move_line_ids",
            kind: OdooFieldKind::One2many,
            target: Some("stock.move.line"),
            required: false,
            computed: None,
            depends: &[],
            // Move lines drive the actual quant update in _action_done.
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "move_orig_ids",
            kind: OdooFieldKind::Many2many,
            target: Some("stock.move"),
            required: false,
            computed: None,
            depends: &[],
            // Upstream moves; non-empty → state='waiting' on confirm (chained MTO).
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "move_dest_ids",
            kind: OdooFieldKind::Many2many,
            target: Some("stock.move"),
            required: false,
            computed: None,
            depends: &[],
            // Downstream moves triggered after _action_done.
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "procure_method",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // 'make_to_stock' | 'make_to_order' | 'mts_else_mto'.
            // Drives branch selection in _action_confirm and _action_assign.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "reservation_method",
            kind: OdooFieldKind::Selection,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // 'at_confirm' | 'manual' | 'by_date'. Gate for _should_assign_at_confirm.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "reservation_date",
            kind: OdooFieldKind::Date,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Set to today() when reservation_method='at_confirm'. Used in _trigger_assign domain.
            semantic_role: OdooSemanticRole::Date,
        },
        OdooField {
            name: "priority",
            kind: OdooFieldKind::Selection,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // '0' normal | '1' urgent.  Picking.action_assign sorts by -int(priority) first.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "date_deadline",
            kind: OdooFieldKind::Datetime,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Used in action_assign sort: moves WITH deadlines before moves WITHOUT.
            semantic_role: OdooSemanticRole::Date,
        },
        OdooField {
            name: "propagate_cancel",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // When True: if ALL sibling moves cancel, propagate cancel to move_dest_ids.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "picked",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // False on backorder moves (reset in _create_backorder at picking level).
            semantic_role: OdooSemanticRole::Status,
        },
        OdooField {
            name: "price_unit",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Community valuation hook (L129); full AVCO/standard-cost in stock_account module.
            semantic_role: OdooSemanticRole::Money,
        },
        OdooField {
            name: "availability",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: Some("_compute_product_availability"),
            depends: &["state", "product_id", "product_uom_qty", "location_id"],
            // min(product_qty, quant.available_quantity) or product_qty when state=done.
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "rule_id",
            kind: OdooFieldKind::Many2one,
            target: Some("stock.rule"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
    ],
    methods: &[
        OdooMethod {
            name: "_action_confirm",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Self_,
            triggers: &["waiting", "confirmed"],
            // Entry guard: skip if state != 'draft'. Sets reservation_date on at_confirm.
        },
        OdooMethod {
            name: "_action_assign",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Unit,
            triggers: &["partially_available", "assigned"],
            // Reservation loop: bypass / MTS / MTO branches.
            // MoveAssignmentPrioritizer dispatch surface (Axis-2, NextBestAction).
        },
        OdooMethod {
            name: "_action_done",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Recordset,
            triggers: &["done"],
            // Delegates actual quant update to move_line_ids._action_done().
        },
        OdooMethod {
            name: "_action_cancel",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Self_,
            triggers: &["cancel"],
            // Calls _do_unreserve(), propagates cancel if propagate_cancel=True and all siblings cancelled.
        },
        OdooMethod {
            name: "_recompute_state",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
            // Re-derives state from quantity vs product_uom_qty — NOT a simple FSM trigger.
        },
        OdooMethod {
            name: "_compute_product_qty",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_product_availability",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_create_backorder",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Self_,
            triggers: &[],
            // Calls _split for each move where quantity < product_uom_qty.
            // BackorderJudge dispatch surface.
        },
        OdooMethod {
            name: "_split",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Record,
            triggers: &[],
            // Creates copy of move with backorder qty; uses float_round at Product Unit precision.
        },
        OdooMethod {
            name: "_get_available_quantity",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Number,
            triggers: &[],
            // Wrapper: short-circuits to product_qty when location bypasses reservation.
        },
        OdooMethod {
            name: "_update_reserved_quantity",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_should_bypass_reservation",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Boolean,
            triggers: &[],
            // True when location.should_bypass_reservation() OR not product_id.is_storable.
        },
        OdooMethod {
            name: "_trigger_assign",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
            // Auto-reservation trigger: searches confirmed/partially_available MTS moves
            // with reservation_date <= today, sorts by priority/date, calls _action_assign.
        },
        OdooMethod {
            name: "_push_apply",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["product_uom_qty", "product_uom", "product_id.uom_id"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["quantity", "product_uom_qty", "move_orig_ids.state"],
        },
    ],
    state_machine: Some(&STOCK_MOVE_STATE_MACHINE),
    constraints: &[
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "product_qty field is read-only; writing it raises UserError — \
                        always write product_uom_qty instead",
            source_method: None,
        },
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "cancel blocked when state=done and location_dest_usage != 'inventory'",
            source_method: Some("_action_cancel"),
        },
    ],
    provenance: OdooProvenance {
        l_doc: "L7-STOCK.md",
        l_doc_lines: (31, 108),
        odoo_source: &[OdooSourceRef {
            path: "addons/stock/models/stock_move.py",
            line_range: (107, 2503),
        }],
        confidence: OdooConfidence::Curated,
    },
};

// ─── stock.move.line ──────────────────────────────────────────────────────────

const STOCK_MOVE_LINE: OdooEntity = OdooEntity {
    model_name: "stock.move.line",
    description: "One lot/package/owner-specific reservation or done-qty record within \
                  a stock move; _action_done drives the actual quant mutation via \
                  stock.quant._update_available_quantity (K10, DOLCE Perdurant).",
    fields: &[
        OdooField {
            name: "move_id",
            kind: OdooFieldKind::Many2one,
            target: Some("stock.move"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "picking_id",
            kind: OdooFieldKind::Many2one,
            target: Some("stock.picking"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
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
            name: "location_dest_id",
            kind: OdooFieldKind::Many2one,
            target: Some("stock.location"),
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "lot_id",
            kind: OdooFieldKind::Many2one,
            target: Some("stock.lot"),
            required: false,
            computed: None,
            depends: &[],
            // Lot/serial tracking key; serial products get one move line per unit.
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "package_id",
            kind: OdooFieldKind::Many2one,
            target: Some("stock.quant.package"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "owner_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.partner"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "quantity",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Qty done on this line. Drives quant update in _action_done.
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "reserved_qty",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "picked",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // False = unpicked lines unlinked before backorder creation.
            semantic_role: OdooSemanticRole::Status,
        },
    ],
    methods: &[
        OdooMethod {
            name: "_action_done",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
            // Calls stock.quant._update_available_quantity for each line.
        },
        OdooMethod {
            name: "_apply_putaway_strategy",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
            // Applied on moves_to_redirect after _action_assign via bypass branch.
        },
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
        odoo_source: &[OdooSourceRef {
            path: "addons/stock/models/stock_move.py",
            line_range: (1763, 1820),
        }],
        confidence: OdooConfidence::Curated,
    },
};

// ─── stock.quant ──────────────────────────────────────────────────────────────

const STOCK_QUANT: OdooEntity = OdooEntity {
    model_name: "stock.quant",
    description: "Persistent stock record: quantity of one product at one location \
                  with a specific lot/package/owner combination; tracks reserved_quantity \
                  separately (reserved >= 0 invariant); removal strategy + availability \
                  arithmetic live here (K10, DOLCE Endurant, FLAG-3).",
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
            name: "location_id",
            kind: OdooFieldKind::Many2one,
            target: Some("stock.location"),
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "lot_id",
            kind: OdooFieldKind::Many2one,
            target: Some("stock.lot"),
            required: false,
            computed: None,
            depends: &[],
            // Lot-specific quants float to top of _gather result (tie-breaker).
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "package_id",
            kind: OdooFieldKind::Many2one,
            target: Some("stock.quant.package"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "owner_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.partner"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "quantity",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // On-hand quantity. Can be negative (deficit quant) after concurrent write.
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "reserved_quantity",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // max(0, reserved + delta) invariant — never goes negative.
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "in_date",
            kind: OdooFieldKind::Datetime,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // min(incoming_dates) after merge — FIFO invariant at quant level.
            semantic_role: OdooSemanticRole::Date,
        },
        OdooField {
            name: "removal_date",
            kind: OdooFieldKind::Datetime,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // FEFO: domain filter `removal_date >= threshold OR NULL` in _get_gather_domain.
            semantic_role: OdooSemanticRole::Date,
        },
        OdooField {
            name: "inventory_quantity",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Counted quantity for inventory adjustment (_apply_inventory).
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "inventory_quantity_set",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Status,
        },
    ],
    methods: &[
        OdooMethod {
            name: "_get_available_quantity",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Number,
            triggers: &[],
            // available = quantity - reserved_quantity, floored at 0 if allow_negative=False.
            // Lot-grouped for tracked products; strict mode filters by exact lot_id.
        },
        OdooMethod {
            name: "_gather",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Recordset,
            triggers: &[],
            // RemovalStrategySelector dispatch surface (Axis-2 NARS, NextBestAction, XorBundle).
            // Strategy: fifo|lifo|fefo|closest|least_packages; final sort: lot quants first.
        },
        OdooMethod {
            name: "_get_removal_strategy",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
            // Priority: product.categ_id.removal_strategy_id > walk location hierarchy > 'fifo'.
        },
        OdooMethod {
            name: "_get_gather_domain",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
            // Builds search domain; adds removal_date filter for FEFO (with_expiration context).
        },
        OdooMethod {
            name: "_get_reserve_quantity",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
            // Returns [(quant, qty_to_reserve), ...] without mutating.
            // Full-packaging floor, UoM DOWN+HALF-UP, serial guard (no partial serial).
        },
        OdooMethod {
            name: "_update_available_quantity",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
            // Pessimistic lock (try_lock_for_update). in_date = min(incoming_dates).
            // reserved_quantity = max(0, reserved + delta) — hard floor.
        },
        OdooMethod {
            name: "_apply_inventory",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
            // Converts inventory_quantity diff to a stock move and calls _action_done.
        },
        OdooMethod {
            name: "_merge_quants",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
            // Raw SQL cleanup for duplicate quants from concurrent transactions.
        },
        OdooMethod {
            name: "_unlink_zero_quants",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
            // Deletes quants where quantity≈0 AND reserved=0 AND inventory=0 AND no user.
        },
        OdooMethod {
            name: "_run_least_packages_removal_strategy_astar",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Recordset,
            triggers: &[],
            // A* over package combinations (FLAG-6: MemoryError guard, performance boundary).
        },
    ],
    decorators: &[OdooDecorator {
        kind: OdooDecoratorKind::ApiAutovacuum,
        targets: &[],
        // _unlink_zero_quants scheduled as autovacuum cron.
    }],
    state_machine: None,
    constraints: &[
        OdooConstraint {
            kind: OdooConstraintKind::Sql,
            condition: "UNIQUE(product_id, location_id, lot_id, package_id, owner_id) \
                        — one quant record per (product, location, lot, package, owner) tuple",
            source_method: None,
        },
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "reserved_quantity >= 0 (max(0, reserved + delta) invariant in \
                        _update_available_quantity)",
            source_method: Some("_update_available_quantity"),
        },
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "serial-tracked products cannot have fractional reserved qty \
                        (_get_reserve_quantity: if tracking=serial and qty != int(qty) → qty=0)",
            source_method: Some("_get_reserve_quantity"),
        },
    ],
    provenance: OdooProvenance {
        l_doc: "L7-STOCK.md",
        l_doc_lines: (110, 415),
        odoo_source: &[OdooSourceRef {
            path: "addons/stock/models/stock_quant.py",
            line_range: (617, 1138),
        }],
        confidence: OdooConfidence::Curated,
    },
};

// ─── stock.picking ────────────────────────────────────────────────────────────

const STOCK_PICKING_STATE_MACHINE: OdooStateMachine = OdooStateMachine {
    state_field: "state",
    states: &[
        OdooState { name: "draft", semantic: OdooStateSemantic::Draft },
        OdooState { name: "waiting", semantic: OdooStateSemantic::InProgress },
        OdooState { name: "confirmed", semantic: OdooStateSemantic::InProgress },
        // Picking 'assigned' = at least ready to ship (partially_available move → assigned picking
        // when move_type='direct'). See _compute_state L858–862.
        OdooState { name: "assigned", semantic: OdooStateSemantic::Active },
        OdooState { name: "done", semantic: OdooStateSemantic::Completed },
        OdooState { name: "cancel", semantic: OdooStateSemantic::Cancelled },
    ],
    transitions: &[
        OdooTransition {
            from: "draft",
            to: "confirmed",
            trigger: "action_confirm",
            guards: &[],
        },
        OdooTransition {
            from: "confirmed",
            to: "assigned",
            trigger: "action_assign",
            guards: &[],
        },
        OdooTransition {
            from: "assigned",
            to: "done",
            trigger: "button_validate",
            guards: &["_sanity_check"],
        },
        OdooTransition {
            from: "confirmed",
            to: "done",
            trigger: "button_validate",
            guards: &["_sanity_check"],
        },
    ],
};

const STOCK_PICKING: OdooEntity = OdooEntity {
    model_name: "stock.picking",
    description: "A group of stock moves constituting one logistics operation \
                  (receipt/delivery/internal transfer); state COMPUTED from move states; \
                  BackorderJudge dispatch surface on button_validate; \
                  MoveAssignmentPrioritizer surfaces on action_assign sort \
                  (K10, DOLCE Perdurant, FLAG-2).",
    fields: &[
        OdooField {
            name: "name",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: Some("_compute_name"),
            depends: &[],
            semantic_role: OdooSemanticRole::Identity,
        },
        OdooField {
            name: "state",
            kind: OdooFieldKind::Selection,
            target: None,
            required: false,
            computed: Some("_compute_state"),
            // Aggregated from move states — NOT stored independently.
            depends: &[
                "move_ids.state",
                "move_ids.is_locked",
                "move_ids.quantity",
            ],
            semantic_role: OdooSemanticRole::Status,
        },
        OdooField {
            name: "move_ids",
            kind: OdooFieldKind::One2many,
            target: Some("stock.move"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "move_line_ids",
            kind: OdooFieldKind::One2many,
            target: Some("stock.move.line"),
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
            // Carries create_backorder ('ask'|'always'|'never') and reservation_method policy.
            semantic_role: OdooSemanticRole::Policy,
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
            name: "location_dest_id",
            kind: OdooFieldKind::Many2one,
            target: Some("stock.location"),
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "partner_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.partner"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "move_type",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // 'direct' (as-soon-as-possible, partial ok) | 'one' (all-at-once).
            // Controls _get_relevant_state_among_moves: 'one'=worst move, 'direct'=best move.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "backorder_id",
            kind: OdooFieldKind::Many2one,
            target: Some("stock.picking"),
            required: false,
            computed: None,
            depends: &[],
            // Set on backorder picking to link it to the origin picking.
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "user_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.users"),
            required: false,
            computed: None,
            depends: &[],
            // Reset to False on backorder creation (must be reassigned).
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "priority",
            kind: OdooFieldKind::Selection,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "date_deadline",
            kind: OdooFieldKind::Datetime,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Date,
        },
    ],
    methods: &[
        OdooMethod {
            name: "_compute_state",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
            // Aggregates from move states; partially_available → 'assigned' for move_type='direct'.
        },
        OdooMethod {
            name: "action_confirm",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Unit,
            triggers: &["confirmed"],
        },
        OdooMethod {
            name: "action_assign",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Unit,
            triggers: &["assigned"],
            // Sorts moves: (-priority, not date_deadline, date_deadline, date, id).
            // MoveAssignmentPrioritizer dispatch surface (Axis-2, NextBestAction).
        },
        OdooMethod {
            name: "button_validate",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Action,
            triggers: &["done"],
            // May show backorder wizard when picking_type.create_backorder='ask'.
            // BackorderJudge dispatch surface (Axis-2, NextBestAction, Abduction).
        },
        OdooMethod {
            name: "_create_backorder",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Recordset,
            triggers: &[],
            // Creates copy picking with unfinished moves. Resets user_id=False, picked=False.
            // If reservation_method='at_confirm': immediately calls action_assign on backorder.
        },
        OdooMethod {
            name: "_check_backorder",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Recordset,
            triggers: &[],
            // Returns pickings needing backorder (only where create_backorder='ask').
        },
        OdooMethod {
            name: "_sanity_check",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Boolean,
            triggers: &[],
            // Lots required check, no empty moves.
        },
        OdooMethod {
            name: "_action_done",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
    ],
    decorators: &[OdooDecorator {
        kind: OdooDecoratorKind::ApiDepends,
        targets: &["move_ids.state", "move_ids.is_locked", "move_ids.quantity"],
    }],
    state_machine: Some(&STOCK_PICKING_STATE_MACHINE),
    constraints: &[OdooConstraint {
        kind: OdooConstraintKind::Python,
        condition: "button_validate blocked when no moves with quantity > 0 \
                    (sanity_check: lots required per picking_type configuration)",
        source_method: Some("_sanity_check"),
    }],
    provenance: OdooProvenance {
        l_doc: "L7-STOCK.md",
        l_doc_lines: (419, 500),
        odoo_source: &[OdooSourceRef {
            path: "addons/stock/models/stock_picking.py",
            line_range: (575, 1602),
        }],
        confidence: OdooConfidence::Curated,
    },
};

// ─── stock.location ───────────────────────────────────────────────────────────

const STOCK_LOCATION: OdooEntity = OdooEntity {
    model_name: "stock.location",
    description: "A node in the stock location hierarchy (physical or virtual); \
                  carries removal_strategy_id (FIFO/FEFO/LIFO/closest/least_packages) \
                  as the RemovalStrategySelector dispatch surface; \
                  should_bypass_reservation() short-circuits quant operations for \
                  virtual/supplier/customer locations (K10, DOLCE Endurant).",
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
            name: "complete_name",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: Some("_compute_complete_name"),
            depends: &["name", "location_id.complete_name"],
            // Used in 'closest' removal strategy sort: sorted by complete_name ASC.
            semantic_role: OdooSemanticRole::Identity,
        },
        OdooField {
            name: "location_id",
            kind: OdooFieldKind::Many2one,
            target: Some("stock.location"),
            required: false,
            computed: None,
            depends: &[],
            // Parent location; _get_removal_strategy walks this hierarchy.
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "usage",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // 'internal'|'customer'|'supplier'|'inventory'|'production'|'transit'|'view'.
            // Non-internal usages typically bypass reservation.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "removal_strategy_id",
            kind: OdooFieldKind::Many2one,
            target: Some("product.removal"),
            required: false,
            computed: None,
            depends: &[],
            // RemovalStrategySelector dispatch surface.
            // Priority: product.categ_id.removal_strategy_id wins over location hierarchy.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "active",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Status,
        },
        OdooField {
            name: "last_inventory_date",
            kind: OdooFieldKind::Date,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Updated by _apply_inventory after adjustment.
            semantic_role: OdooSemanticRole::Audit,
        },
    ],
    methods: &[
        OdooMethod {
            name: "should_bypass_reservation",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Boolean,
            triggers: &[],
            // True for virtual/supplier/customer locations.
            // Short-circuits _action_assign to avoid touching quants.
        },
        OdooMethod {
            name: "_compute_complete_name",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
    ],
    decorators: &[OdooDecorator {
        kind: OdooDecoratorKind::ApiDepends,
        targets: &["name", "location_id.complete_name"],
    }],
    state_machine: None,
    constraints: &[OdooConstraint {
        kind: OdooConstraintKind::Domain,
        condition: "child_of domain used in _get_gather_domain (non-strict mode) — \
                    quant search spans all sub-locations",
        source_method: None,
    }],
    provenance: OdooProvenance {
        l_doc: "L7-STOCK.md",
        l_doc_lines: (155, 207),
        odoo_source: &[OdooSourceRef {
            path: "addons/stock/models/stock_quant.py",
            line_range: (617, 791),
        }],
        confidence: OdooConfidence::Curated,
    },
};

// ─── stock.warehouse ──────────────────────────────────────────────────────────

const STOCK_WAREHOUSE: OdooEntity = OdooEntity {
    model_name: "stock.warehouse",
    description: "A physical warehouse site carrying operational configuration \
                  (picking types, routes, multi-step delivery/reception policy); \
                  alignment UNRESOLVED — proposed gs1:Location (K10, DOLCE Endurant, FLAG-4). \
                  TODO: alignment row needed before OGIT hydration.",
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
            name: "code",
            kind: OdooFieldKind::Char,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // Short code (e.g. 'WH') used in picking sequence names.
            semantic_role: OdooSemanticRole::Identity,
        },
        OdooField {
            name: "partner_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.partner"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Address,
        },
        OdooField {
            name: "lot_stock_id",
            kind: OdooFieldKind::Many2one,
            target: Some("stock.location"),
            required: false,
            computed: None,
            depends: &[],
            // Default internal stock location for this warehouse.
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "reception_steps",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // 'one_step' | 'two_steps' | 'three_steps'. Determines route configuration.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "delivery_steps",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // 'ship_only' | 'pick_ship' | 'pick_pack_ship'.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "route_ids",
            kind: OdooFieldKind::Many2many,
            target: Some("stock.route"),
            required: false,
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
    ],
    methods: &[
        OdooMethod {
            name: "_get_picking_type_create_edit",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        OdooMethod {
            name: "_update_reception_delivery",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
            // Reconfigures routes/picking-types when reception_steps or delivery_steps changes.
        },
    ],
    decorators: &[],
    state_machine: None,
    constraints: &[OdooConstraint {
        kind: OdooConstraintKind::Sql,
        condition: "UNIQUE(code, company_id) — warehouse code must be unique per company",
        source_method: None,
    }],
    provenance: OdooProvenance {
        l_doc: "L7-STOCK.md",
        l_doc_lines: (608, 644),
        odoo_source: &[OdooSourceRef {
            // TODO: stock.warehouse alignment row needed (FLAG-4 — proposed gs1:Location).
            path: "addons/stock/models/stock_warehouse.py",
            line_range: (1, 200),
        }],
        confidence: OdooConfidence::Curated,
    },
};

// ─── ENTITIES slice ───────────────────────────────────────────────────────────

/// Entities documented in lane L7 (stock moves + quants + reservations
/// + removal strategy + backorder).
///
/// 6 entities total:
///   [0] `stock.move`       — state machine (draft→confirmed→assigned→done), reservation loop
///   [1] `stock.move.line`  — quant-update driver, lot/serial tracking
///   [2] `stock.quant`      — availability arithmetic, removal strategy, inventory adjustment
///   [3] `stock.picking`    — picking state machine, backorder judge, assignment sort
///   [4] `stock.location`   — removal strategy dispatch, bypass reservation
///   [5] `stock.warehouse`  — operational config (FLAG-4: needs alignment authoring)
///
/// Entities NOT included (skipped):
///   - `stock.rule` — route/procurement rules; a separate procurement-lane concern.
///   - `stock.lot`  — lot/serial master; referenced as FK but not a primary L7 entity.
///   - `stock.picking.type` — configuration model; documented inline via picking_type_id policy.
///   - `product.removal` — removal strategy configuration; referenced via removal_strategy_id.
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
        let sm = STOCK_MOVE
            .state_machine
            .expect("stock.move must have a state machine");
        assert_eq!(sm.state_field, "state");
        // 7 states: draft, waiting, confirmed, partially_available, assigned, done, cancel.
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
        let f = STOCK_MOVE
            .fields
            .iter()
            .find(|f| f.name == "product_qty")
            .expect("product_qty must be present");
        assert_eq!(f.computed, Some("_compute_product_qty"));
        assert_eq!(f.kind, OdooFieldKind::Float);
    }

    #[test]
    fn stock_move_reservation_policy_fields_present() {
        let names: Vec<&str> = STOCK_MOVE.fields.iter().map(|f| f.name).collect();
        assert!(names.contains(&"reservation_method"));
        assert!(names.contains(&"reservation_date"));
        assert!(names.contains(&"procure_method"));
        assert!(names.contains(&"priority"));
    }

    #[test]
    fn stock_quant_reservation_invariant_captured() {
        let f = STOCK_QUANT
            .fields
            .iter()
            .find(|f| f.name == "reserved_quantity")
            .expect("reserved_quantity must be present");
        assert_eq!(f.semantic_role, OdooSemanticRole::Quantity);
        // max(0, reserved + delta) invariant documented in constraints.
        assert!(!STOCK_QUANT.constraints.is_empty());
    }

    #[test]
    fn stock_quant_has_removal_strategy_methods() {
        let method_names: Vec<&str> =
            STOCK_QUANT.methods.iter().map(|m| m.name).collect();
        assert!(method_names.contains(&"_gather"));
        assert!(method_names.contains(&"_get_removal_strategy"));
        assert!(method_names.contains(&"_get_reserve_quantity"));
        assert!(method_names.contains(&"_update_available_quantity"));
    }

    #[test]
    fn stock_picking_state_is_computed() {
        let f = STOCK_PICKING
            .fields
            .iter()
            .find(|f| f.name == "state")
            .expect("state must be present on stock.picking");
        assert_eq!(f.computed, Some("_compute_state"));
    }

    #[test]
    fn stock_picking_has_state_machine() {
        let sm = STOCK_PICKING
            .state_machine
            .expect("stock.picking must have a state machine");
        assert_eq!(sm.state_field, "state");
        // 6 states: draft, waiting, confirmed, assigned, done, cancel.
        assert_eq!(sm.states.len(), 6);
    }

    #[test]
    fn stock_picking_move_type_is_policy() {
        let f = STOCK_PICKING
            .fields
            .iter()
            .find(|f| f.name == "move_type")
            .expect("move_type must be present");
        assert_eq!(f.semantic_role, OdooSemanticRole::Policy);
    }

    #[test]
    fn stock_location_removal_strategy_is_policy() {
        let f = STOCK_LOCATION
            .fields
            .iter()
            .find(|f| f.name == "removal_strategy_id")
            .expect("removal_strategy_id must be on stock.location");
        assert_eq!(f.kind, OdooFieldKind::Many2one);
        assert_eq!(f.semantic_role, OdooSemanticRole::Policy);
    }

    #[test]
    fn stock_location_has_bypass_method() {
        let found = STOCK_LOCATION
            .methods
            .iter()
            .any(|m| m.name == "should_bypass_reservation");
        assert!(found, "should_bypass_reservation must be documented on stock.location");
    }

    #[test]
    fn stock_warehouse_has_steps_policy_fields() {
        let names: Vec<&str> = STOCK_WAREHOUSE.fields.iter().map(|f| f.name).collect();
        assert!(names.contains(&"reception_steps"));
        assert!(names.contains(&"delivery_steps"));
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
    fn all_entities_cite_l7_doc() {
        for entity in ENTITIES {
            assert_eq!(
                entity.provenance.l_doc,
                "L7-STOCK.md",
                "entity {} must cite L7 doc",
                entity.model_name
            );
        }
    }
}
