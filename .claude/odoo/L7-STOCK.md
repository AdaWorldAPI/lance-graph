RICHNESS-LANE-OK

# L7 — Inventory: Stock Moves, Picking, Quant Valuation + Reservation

**Lane:** L7-STOCK  
**Date:** 2026-05-26  
**Author:** Claude Sonnet 4.6 (read-only analysis lane)  
**K-step:** K10 (Lager/Inventur) — net-new ERP inventory richness  
**Sentinel:** RICHNESS-LANE-OK (first line)

---

## 1. Scope + Files Read

| File | Lines | Depth |
|---|---|---|
| `/home/user/odoo/addons/stock/models/stock_move.py` | 2683 | full |
| `/home/user/odoo/addons/stock/models/stock_quant.py` | 1563 | full |
| `/home/user/odoo/addons/stock/models/stock_picking.py` | 2149 | full |

**woa-rs calibration grep:** `grep -rn "stock\|lager\|inventory\|bestand\|material\|quant\|reserv" /home/user/woa-rs/src/ 2>/dev/null | head`

**Finding:** woa-rs has `src/models/erp/k10_inventory.rs` — a simple WoA-app Lager model (warehouse/inventory/stock_movement/serial). It is a flat movement ledger with no state machine, no reservation concept, no multi-step picking, no quant-level tracking, no removal strategies. The Odoo inventory subsystem is **entirely net-new richness** relative to woa-rs. No collision, pure addition.

---

## 2. Rule Sections

---

### R1 — Stock Move State Machine

**File:** `stock_move.py:107–120` (state field), `L1546–1641` (`_action_confirm`), `L1901–2043` (`_action_assign`), `L2101–2169` (`_action_done`), `L2044–2084` (`_action_cancel`), `L2268–2288` (`_recompute_state`)

#### Axis-1: Rich-AST Spec

**State Enumeration** (`stock_move.py:L107–120`):
```
draft          → New (not confirmed)
waiting        → Waiting Another Move (upstream move not done)
confirmed      → Waiting (confirmed but product not reservable yet)
partially_available → Some qty reserved
assigned       → Fully reserved (available)
done           → Transfer completed
cancel         → Cancelled
```
Default on create: `'draft'`. Field: `copy=False, index=True, readonly=True`.

**Transition Rules** (`_action_confirm`, `L1546–1641`):
- Entry guard: `move.state != 'draft'` → skip.
- If `move.move_orig_ids` is non-empty → `waiting` (has upstream dependency).
- Else if `procure_method == 'make_to_order'` → `waiting` + optionally create procurement.
- Else if `rule_id.procure_method == 'mts_else_mto'` → `confirmed` + optionally create procurement.
- Else → `confirmed`.
- After state write: if `reservation_method == 'at_confirm'`, set `reservation_date = today`.
- After merge: immediately call `_action_assign` on `confirmed`/`partially_available` moves that `_should_bypass_reservation()` or `_should_assign_at_confirm()`.
- Negative qty moves (`product_uom_qty < 0`) are treated as returns: locations swapped, `product_uom_qty *= -1`, `picking_type_id = return_picking_type_id`.

**`_should_bypass_reservation`** (`L1825–1828`): bypasses if `location.should_bypass_reservation()` OR `not product_id.is_storable`. Storable = consumable/service get no quants.

**`_should_assign_at_confirm`** (`L1830–1831`): True when bypass OR `reservation_method == 'at_confirm'` OR `reservation_date <= today`.

**`_recompute_state`** (`L2268–2288`): Re-derives state from `quantity` vs `product_uom_qty`:
- If `quantity >= product_uom_qty` → `assigned`.
- If `0 < quantity < product_uom_qty` → `partially_available`.
- If `procure_method == 'make_to_order'` and no `move_orig_ids` → `waiting`.
- Else → `confirmed`.
- Skip if state is `cancel`, `done`, or `draft` with zero quantity.

**`_action_cancel`** (`L2044–2084`):
- Guard: cannot cancel `done` moves where `location_dest_usage != 'inventory'`.
- Calls `_do_unreserve()` before cancelling.
- Sets state `cancel`.
- If `propagate_cancel=True` and ALL siblings are cancelled → also cancel `move_dest_ids` (chain propagation).
- Detaches `move_orig_ids` from cancelled moves (clears, sets `procure_method='make_to_stock'`).

**`_action_done`** (`L2101–2169`):
1. Draft moves are first confirmed (merge=False).
2. Moves with `picked=False` and `quantity <= 0` → cancelled (unless `is_inventory`).
3. Unpicked move lines unlinked.
4. `_create_backorder()` called for partial fulfilment (unless `cancel_backorder=True`).
5. `move_line_ids.sorted()._action_done()` — move lines drive the actual quant update.
6. state → `'done'`, `date = now()`.
7. `move_dest_ids._action_assign()` called (downstream moves get available quants).
8. Push rules applied via `_push_apply()`.
9. Picking `_create_backorder()` called if `picking` exists.

**`_create_backorder` (move-level)** (`L2174–2190`): For each move where `quantity < product_uom_qty` (using general Product Unit decimal precision, NOT UoM rounding), compute `qty_split = product_uom_qty - quantity`, call `_split(qty_split)`, create new move, confirm without merge and without creating procurement.

**`_split`** (`L2216–2260`): Creates a copy of the move with `product_uom_qty = qty` (the backorder portion). Updates original move's `product_uom_qty = max(0, original_qty - qty)`. Uses `float_round` at Product Unit precision. UoM conversion: first tries to round-trip the quantity through move UoM; if round-trip fails (fractional UoM), falls back to product default UoM.

#### Rounding / UoM / Float Handling
- `float_compare`, `float_is_zero`, `float_round` from `odoo.tools.float_utils` — NOT Python's built-in float comparison.
- Product quantity computed in `product_id.uom_id` (the product's default UoM), not the move's `product_uom`. Conversion via `product_uom._compute_quantity(..., rounding_method='HALF-UP')`.
- Backorder qty uses `precision_get('Product Unit')` (the named decimal precision, not UoM rounding).
- `product_qty` field is computed (`_compute_product_qty`): `product_uom._compute_quantity(product_uom_qty, product_id.uom_id, rounding_method='HALF-UP')`. Setting `product_qty` raises `UserError` — must write `product_uom_qty` instead.

#### Axis Classification
**DETERMINISTIC** — Axis-1. The state machine transitions are closed-form rules with no heuristic weighting. Port directly.

**Ontology:**
`odoo:stock.move` → UNRESOLVED (FLAG — see Section 3)  
Proposed: `odoo:stock.move` → `gs1:LogisticEvent` → OGIT family needs authoring (see Section 3)  
DOLCE: **Perdurant** (`.move` suffix classifier, temporal event)  
K-step: **K10** (Lager/Inventur)  
woa-rs target: `src/erp/inventory/stock_move.rs`

---

### R2 — Availability Arithmetic: `_get_available_quantity`

**File:** `stock_quant.py:L793–832` (quant-level), `stock_move.py:L1846–1850` (move wrapper)

#### Axis-1: Rich-AST Spec

**Move-level wrapper** (`stock_move.py:L1846–1850`):
```python
def _get_available_quantity(self, location_id, lot_id=None, package_id=None, owner_id=None, strict=False, allow_negative=False):
    if location_id.should_bypass_reservation():
        return self.product_qty
    return self.env['stock.quant']._get_available_quantity(self.product_id, location_id, ...)
```
If location bypasses reservation (e.g. virtual/supplier/customer locations), return the full `product_qty` unconditionally.

**Quant-level** (`stock_quant.py:L793–832`):
```
available_quantity = quantity - reserved_quantity
```
For untracked products (`tracking == 'none'`):
- `sum(quants.quantity) - sum(quants.reserved_quantity)`.
- If `allow_negative=False`: floor at 0.0 using `product_id.uom_id.compare(available_quantity, 0.0) >= 0`.

For tracked products (lot/serial):
- Group by lot_id: `{lot_id: 0.0, ..., 'untracked': 0.0}`.
- Sum `quantity - reserved_quantity` per lot bucket.
- If `strict=True` and `lot_id` specified: skip quants without lot.
- If `allow_negative=False`: only sum buckets with `compare(qty, 0) > 0`.

**`_compute_product_availability`** (`stock_move.py:L489–500`): Field `availability` on move:
- If `state == 'done'`: `availability = product_qty`.
- Else: `min(product_qty, quant._get_available_quantity(product_id, location_id))`.

#### Axis Classification
**DETERMINISTIC** — Axis-1. Pure arithmetic: `quantity - reserved_quantity`, min-capped, lot-grouped. No heuristics.

**Ontology:**
`odoo:stock.quant` → UNRESOLVED (FLAG — see Section 3)  
Proposed: → `gs1:LogisticEvent` (instance of stock at location) OR needs new alignment row (see Section 3)  
DOLCE: **Endurant** (persistent stock record)  
K-step: **K10**  
woa-rs target: `src/erp/inventory/quant.rs`

---

### R3 — Quant `_gather`: Candidate Selection (Removal Strategy)

**File:** `stock_quant.py:L617–791`

#### Axis-1: Rich-AST Spec

**`_get_removal_strategy`** (`L617–628`):
Priority order:
1. `product_id.categ_id.removal_strategy_id.method` (product category wins).
2. Walk `location_id` up the location hierarchy (`loc = loc.location_id` loop) until `loc.removal_strategy_id` found.
3. Default: `'fifo'`.

**Available strategies:**
- `'fifo'`: order `in_date ASC, id ASC` (oldest incoming date first).
- `'lifo'`: order `in_date DESC, id DESC` (newest first).
- `'fefo'`: not in `_get_removal_strategy_order` (returns `False` for `'closest'`); FEFO is handled via `with_expiration` context key in `_get_gather_domain` — filters quants where `removal_date >= expiration_threshold` OR `removal_date IS NULL`, then sorted by lot expiry date. Note: `_get_removal_strategy_order` raises `UserError` for unknown strategies.
- `'least_packages'`: order `in_date ASC, id ASC` (same as FIFO), but domain is computed by A* search first (see below).
- `'closest'`: no SQL order (returns `False`); sorted in Python: `res.sorted(lambda q: (q.location_id.complete_name, -q.id))` — nearest location alphabetically, most recent id.

**`_get_gather_domain`** (`L750–769`): Builds the search domain:
- Always: `product_id = product_id.id`.
- Non-strict: `lot_id IN [lot_id, False]`, `package_id = package_id`, `owner_id = owner_id`, `location_id CHILD_OF location_id.id`.
- Strict: `lot_id IN [False, lot_id]`, exact `package_id`, exact `owner_id`, exact `location_id = location_id.id` (no child_of).
- If context has `with_expiration`: adds `removal_date >= threshold OR removal_date IS NULL`.

**`_gather`** (`L771–791`):
```
removal_strategy = _get_removal_strategy(product_id, location_id)
domain = _get_gather_domain(...)
if removal_strategy == 'least_packages' and qty:
    domain = _run_least_packages_removal_strategy_astar(domain, qty)
order = _get_removal_strategy_order(removal_strategy)
# cache bypass for strict non-least-packages
if quants_cache and strict and removal_strategy != 'least_packages':
    res = from cache
else:
    res = self.search(domain, order=order)
if removal_strategy == 'closest':
    res = res.sorted(lambda q: (complete_name, -id))
return res.sorted(lambda q: not q.lot_id)  # ← quants WITH lot_id float to top
```

The final `.sorted(lambda q: not q.lot_id)` means lot-specific quants are preferred over lot-less quants when selecting for reservation. This is a tie-breaker that promotes specificity.

**`_run_least_packages_removal_strategy_astar`** (`L630–738`): An A* search over package combinations to find the minimum number of packages that satisfy the requested quantity. Uses a priority queue with heuristic `len(taken_packages) + remaining_qty / package_size[next]`. Falls back to `best_leaf` (partial or overselecting) on `MemoryError`.

#### Axis-2: HEURISTIC (Axis-2 NARS Candidate — STRONG)

This is the primary NARS candidate for this lane. The allocation strategy over candidate quants is a **next-best-action** selection problem:
- Given demand qty + candidate quants (each with `in_date`, `lot_id`, `removal_date`, `package_id`, `location`).
- The system selects WHICH quants to bind to the reservation, in what order, up to the demand.
- FIFO/FEFO/LIFO/closest/least_packages are competing heuristics with different optimality criteria.
- The choice of strategy itself (product category or location hierarchy) is a policy, not a closed-form derivation.
- Partial reservation (some quants, not enough) is a natural intermediate result — the Reasoner must know when to stop.

**Contract Tuple:**
```
ReasoningKind:       NextBestAction
InferenceType:       Induction (for FIFO/FEFO pattern recognition over historical lot dates)
                     + Abduction (for "why can't this be fulfilled" — missing quants, wrong location)
SemiringChoice:      XorBundle  (multi-quant binding: we select a set of quants that jointly satisfy demand;
                                 XOR semantics apply because each unit of demand is satisfied by exactly one quant)
ThinkingStyleCluster: Exploratory  (breadth over candidate quants, scanning all options before committing)
```

**ThinkingStyle inheritance:** `stock.quant` maps toward logistic stock/goods tracking. In OGIT, this aligns with the `Equipment/Resource` or `Logistics` family (if it existed). The removal strategy is a resource allocation problem — Exploratory cluster is justified because the Reasoner needs to scan the candidate space (all available quants with their dates/lots/locations) before selecting the optimal bundle.

**Evidence the Reasoner receives:**
```
namespace:  "stock.inventory.reservation"
kind:       NextBestAction
evidence: {
  demand_qty: f64,
  product_id: str,
  location_id: str,
  strategy_hint: "fifo" | "fefo" | "lifo" | "closest" | "least_packages",
  candidate_quants: [
    { quant_id, location, lot_id, in_date, removal_date, quantity, reserved_quantity, available_qty }
    ...  // pre-filtered to available (quantity - reserved_quantity > 0)
  ],
  strict: bool,  // exact lot/package/owner match required?
  packaging_uom_id: Option<str>,  // affects full-packaging rounding
}
budget:     default
```

**Reasoner output (Conclusion):** Ordered list of `(quant_id, qty_to_reserve)` pairs summing to `min(demand, total_available)`. Includes explanation field for Abduction path ("insufficient stock at location X", "all lots reserved", etc.).

**Justification from `_gather` ordering logic:** The fact that Odoo encodes 5 different strategies (FIFO/LIFO/FEFO/closest/least_packages) with a product-category-then-location-hierarchy fallback, plus the A* search for least_packages, confirms this is a heuristic multi-criterion optimization — not a closed-form rule. The NARS Reasoner can encode each strategy as an inductive pattern and select the optimal assignment.

**Ontology:** (see quant entry above, Section 3)  
K-step: **K10**  
woa-rs target: `src/erp/inventory/reservation.rs` (stub that delegates to lance-graph reasoning contract)

---

### R4 — `_action_assign`: Reservation / Assignment

**File:** `stock_move.py:L1901–2043`

#### Axis-1: Rich-AST Spec

The full reservation loop. Called on moves in `['confirmed', 'waiting', 'partially_available']` state (unless `force_qty` specified).

**Pre-loop setup:**
- `reserved_availability = {move: move.quantity}` — snapshot of already-reserved quantities (avoids cache invalidation mid-loop).
- `roundings = {move: move.product_id.uom_id.rounding}`.
- MTO moves: `quants_cache` pre-built via `_get_quants_by_products_locations`.

**Per-move logic:**
```
missing_reserved_uom_quantity = product_uom_qty - reserved_availability[move]
if missing <= 0:
    → assigned_moves_ids.add(move.id); continue
missing_reserved_quantity = convert to product uom (HALF-UP)
```

**Branch A: Bypass reservation** (`_should_bypass_reservation()` = True):
- Location is virtual/supplier/customer OR product not storable.
- If `move.move_orig_ids`: pull from `_get_available_move_lines(...)` — look at what upstream done moves brought, subtract what sibling moves already took (in-loop accounting of partially committed quantities).
- If serial tracking: create one move line per unit.
- Otherwise: update existing compatible move line or create new one.
- Result: `assigned_moves_ids.add(move.id)` + `moves_to_redirect.add(move.id)`.

**Branch B: Normal MTS (no upstream moves)**:
- If `procure_method == 'make_to_order'`: skip.
- Call `move._update_reserved_quantity(need, move.location_id, strict=False)` → calls `_get_reserve_quantity` on quant.
- If `taken_quantity == 0`: continue (no stock).
- If `taken_quantity == need`: `assigned_moves_ids.add`.
- Else: `partially_available_moves_ids.add`.

**Branch C: MTO / chained moves** (has `move_orig_ids`, not bypass):
- `_get_available_move_lines(assigned_moves_ids, partially_available_moves_ids)` — cross-reference what upstream delivered.
- For each `(location, lot, package, owner) → qty` bucket:
  - Compute `need = product_qty - sum(existing_mls) - sum(taken_quantities)`.
  - Call `_update_reserved_quantity_vals(min(qty, need), ..., strict=True)`.
  - Accumulate `taken_quantities`.
- If any taken: check if `need - taken_quantity ≈ 0` → assigned, else partially_available.

**Post-loop:**
```python
self.env['stock.move.line'].create(move_line_vals_list)  # batch create
StockMove.browse(partially_available_moves_ids).write({'state': 'partially_available'})
StockMove.browse(assigned_moves_ids).write({'state': 'assigned'})
StockMove.browse(moves_to_redirect).move_line_ids._apply_putaway_strategy()
```

**`_update_reserved_quantity`** (`L1763–1773`): Calls `_get_reserve_quantity` on quant, creates move lines via `_prepare_move_line_vals`.

**`_update_reserved_quantity_vals`** (`L1775–1820`): Deduplicates quants with same `(location, lot, package, owner)` key (groups them). Updates existing move lines where UoM round-trip is exact; creates new move lines otherwise. Serial products: one move line per unit.

**`_get_available_move_lines`** (`L1893–1899`): Available = what upstream done moves brought (`_get_available_move_lines_in`) MINUS what siblings already reserved (`_get_available_move_lines_out`). Siblings include moves already processed in this same `_action_assign` call (the `assigned_moves_ids`/`partially_available_moves_ids` sets are passed in to prevent double-counting).

**`action_assign` (picking level)** (`stock_picking.py:L1195–1208`): Sorts moves by `(-priority, not date_deadline, date_deadline, date, id)` before calling `_action_assign()`. This means:
- High-priority moves (priority='1', Urgent) reserved first.
- Among equal priority: moves WITH deadlines before moves WITHOUT.
- Among equal priority+deadline: by deadline date ascending.
- Tie-break: by date, then by id.

#### Axis-2: HEURISTIC (Axis-2 NARS Candidate — STRONG)

The `action_assign` sorting is a policy (priority-first, deadline-aware), and the decision of whether a partially-reserved move should block or proceed is a judgment call (see `move_type` = 'one' vs 'direct' in picking). The inner `_action_assign` loop is deterministic given a fixed quant set, but the quant selection (via `_gather`) is heuristic (see R3).

For the sorting policy specifically:
```
ReasoningKind:       NextBestAction
InferenceType:       Induction (which move to satisfy first)
SemiringChoice:      NarsTruth  (confidence-weighted priority scoring)
ThinkingStyleCluster: Exploratory (scanning candidate moves and their urgency signals)
```

**Ontology:** (same as stock.move — see Section 3)  
K-step: **K10**  
woa-rs target: `src/erp/inventory/reservation.rs`

---

### R5 — `_get_reserve_quantity`: Quant-Level Reservation Arithmetic

**File:** `stock_quant.py:L834–914`

#### Axis-1: Rich-AST Spec

Returns `[(quant, qty_to_reserve), ...]` without mutating anything.

**Full flow:**
1. `quants = _gather(...)` — sorted by removal strategy.
2. `available_quantity = quants._get_available_quantity(...)` — total available across all gathered quants.
3. Full-packaging check: if `packaging_uom_id` context and `categ.packaging_reserve_method == 'full'`:  
   `available_quantity = packaging_uom._check_qty(min(quantity, available_quantity), product_uom, 'DOWN')`.  
   This floors to whole-package multiples.
4. `quantity = min(quantity, available_quantity)` — cap at what's available.
5. UoM conversion (non-strict + different UoM):  
   `quantity_move_uom = product_uom._compute_quantity(quantity, uom_id, rounding_method='DOWN')`  
   `quantity = uom_id._compute_quantity(quantity_move_uom, product_uom_id, rounding_method='HALF-UP')`  
   (DOWN then HALF-UP guarantees never-over-reserve).
6. Serial guard: if `tracking == 'serial'` and `quantity != int(quantity)` → quantity = 0 (cannot partially reserve a serial).
7. Negative quantity path (unreservation): drain `reserved_quantity` from quants.
8. Positive reservation:
   - Pre-compute `negative_reserved_quantity` dict: quants where `quantity - reserved_quantity < 0` (deficit quants).
   - For each quant in strategy-order:
     - `max_quantity = quant.quantity - quant.reserved_quantity`.
     - Skip if `<= 0`.
     - Offset by any negative-reserved at same `(location, lot, package, owner)`.
     - `reserve_qty = min(max_quantity, remaining_demand)`.
     - Append `(quant, reserve_qty)`.
     - Decrement remaining demand.
     - Break when demand = 0 or available = 0.

#### Axis Classification
**DETERMINISTIC** (the arithmetic once quants are gathered) + **HEURISTIC** (the quant selection via `_gather` — see R3).

The arithmetic itself is Axis-1; the quant selection delegates to R3.

**Ontology:** (same as stock.quant — Section 3)  
K-step: **K10**  
woa-rs target: `src/erp/inventory/quant.rs`

---

### R6 — `_update_available_quantity`: Quant Mutation (Done Move)

**File:** `stock_quant.py:L1037–1105`

#### Axis-1: Rich-AST Spec

Called when a move line is done — the actual stock update.

```python
def _update_available_quantity(product_id, location_id, quantity=False, reserved_quantity=False,
                                lot_id=None, package_id=None, owner_id=None, in_date=None):
```

**`in_date` handling:**
- Gather existing quants (strict = True for exact match).
- If `lot_id` and `quantity > 0`: keep only lot-specific quants.
- If `lot_id` and `quantity <= 0`: keep quants where `quantity > 0` OR `lot_id` set (avoid removing from negative-no-lot quants).
- `incoming_dates = [quant.in_date for quant in quants where quantity > 0]`.
- If `in_date` param provided: append it.
- `in_date = min(incoming_dates)` (oldest date is canonical for the quant group — FIFO semantics preserved even after updates).
- If no dates: `in_date = now()`.

**Write or Create:**
- If existing quant found: `try_lock_for_update(limit=1)` (pessimistic locking, one quant at a time).
  - `quantity += quantity_delta`, `reserved_quantity = max(0, reserved + reserved_qty_delta)`.
  - `in_date = min(incoming_dates)`.
- If no quant: create new quant with all key fields.
- Returns `(available_quantity, in_date)`.

**Concurrency:** `_merge_quants()` handles duplicate quants that arise from concurrent transactions creating the same (product, location, lot, package, owner) combination. Uses raw SQL `WITH dupes AS (SELECT min(id) ...) UPDATE ... DELETE ...`.

**`_unlink_zero_quants`** (`L1122–1138`): Raw SQL query deletes quants where `round(quantity, 2*product_uom_precision) = 0` AND `reserved_quantity = 0` AND `inventory_quantity = 0` AND `user_id IS NULL`. Uses `max(6, uom_precision * 2)` decimal places to avoid rounding artifacts.

#### Axis Classification
**DETERMINISTIC** — Axis-1. Pure accounting: add/subtract, lock, write. No heuristics.

**Rounding note:** `reserved_quantity = max(0, reserved + delta)` — reserved can never go negative (floor at 0). This is an invariant.

**Ontology:** (stock.quant — Section 3)  
K-step: **K10**  
woa-rs target: `src/erp/inventory/quant.rs`

---

### R7 — Picking State Machine

**File:** `stock_picking.py:L575–863` (state field + `_compute_state`), `L1186–1208` (action_confirm/action_assign), `L1255–1280` (`_action_done`), `L1577–1602` (`_create_backorder`)

#### Axis-1: Rich-AST Spec

**Picking state** (`L575–589`) is COMPUTED from move states (not stored independently):
```
draft          → Any move is draft
waiting        → Waiting for another operation
confirmed      → Waiting for availability
assigned       → Ready
done           → Done
cancel         → Cancelled
```

**`_compute_state`** (`L816–863`): Aggregates per picking:
```
any_draft              → 'draft'
all_cancel             → 'cancel'
all_cancel_done:
  all_done_are_scrapped AND any_cancel_and_not_scrapped → 'cancel'
  else → 'done'
else (active moves):
  if location.should_bypass_reservation() AND all make_to_stock → 'assigned'
  else: relevant_move_state = _get_relevant_state_among_moves()
    if relevant_move_state == 'partially_available':
      → 'assigned'  (NOTE: partial is treated as assigned at picking level for 'as soon as possible' policy)
    else:
      → relevant_move_state
```

**`_get_relevant_state_among_moves`** (`stock_move.py:L1320–1360`): Priority sort map `{assigned:4, waiting:3, partially_available:2, confirmed:1}`. For `move_type == 'one'` (deliver all at once): uses the MOST important (highest priority) move's state. Otherwise: uses the LEAST important move's state.

**`action_confirm`** (`L1186–1193`):
1. Call `_action_confirm` on all draft moves.
2. Trigger scheduler on non-draft, non-cancel, non-done moves.

**`action_assign`** (`L1195–1208`):
1. If draft: first call `action_confirm`.
2. Sort moves: `(-int(priority), not bool(date_deadline), date_deadline, date, id)`.
3. Guard: if no moves to assign → `UserError('Nothing to check the availability for.')`.
4. Call `moves._action_assign()`.

**`button_validate`** (`L1398–1458`):
1. Filter out already-done pickings.
2. Auto-confirm draft pickings.
3. For draft pickings: if move has no quantity done but has demand → set `quantity = product_uom_qty`.
4. `_sanity_check()` (lots required, no empty moves).
5. `_pre_action_done_hook()` — may show backorder wizard.
6. `_action_done()` with `cancel_backorder=True/False` based on `picking_type.create_backorder` setting.
7. Auto-print reception report if configured.

**`_create_backorder` (picking level)** (`L1577–1602`):
1. Find `moves_to_backorder = picking._get_moves_to_backorder()` (state not in done/cancel).
2. Recompute states.
3. Create a copy of the picking (`_create_backorder_picking`) with empty moves: `copy({'name': '/', 'move_ids': [], 'move_line_ids': [], 'backorder_id': picking.id})`.
4. Move the `moves_to_backorder` to the new picking: `moves_to_backorder.write({'picking_id': backorder.id, 'picked': False})`.
5. Also move their `move_line_ids` to the new picking.
6. Set `backorder_picking.user_id = False` (no responsible, must be reassigned).
7. Post chatter message on original picking: "Backorder X created."
8. If `reservation_method == 'at_confirm'`: immediately `action_assign()` on the backorder.

**`_check_backorder`** (`L1533–1546`): Returns pickings needing backorder (only where `create_backorder == 'ask'`). A picking needs a backorder if any move (not cancelled) has `product_uom_qty > 0 AND not picked` OR `_get_picked_quantity() < product_uom_qty` (using Product Unit decimal precision).

#### Axis-2: HEURISTIC — Backorder decision

The backorder creation decision involves judgment:
- `picking_type.create_backorder` = 'ask' | 'always' | 'never' is a policy.
- The wizard presented to the user (`_action_generate_backorder_wizard`) is an interactive judgment.
- Whether a partial fulfilment constitutes a valid delivery (`move_type = 'direct'` vs `'one'`) is a business judgment.

```
ReasoningKind:       NextBestAction  (should we create backorder or cancel remainder?)
InferenceType:       Abduction  (why is this partially fulfilled? insufficient stock? fulfil-as-possible?)
SemiringChoice:      NarsTruth  (confidence in whether partial delivery is acceptable)
ThinkingStyleCluster: Exploratory
```

**Ontology:** (stock.picking — Section 3)  
K-step: **K10**  
woa-rs target: `src/erp/inventory/picking.rs`

---

### R8 — Inventory Adjustment via Quant

**File:** `stock_quant.py:L996–1035` (`_apply_inventory`), `L1253–1292` (`_get_inventory_move_values`)

#### Axis-1: Rich-AST Spec

**`_apply_inventory`** (`L996–1035`): Converts `inventory_quantity` (counted) to a stock move:
1. Set `inventory_quantity_set = True`.
2. For each quant: check `inventory_diff_quantity = inventory_quantity - quantity`.
3. If `diff > 0`: create move from `inventory_location → quant.location_id` (stock gain).
4. If `diff < 0`: create move from `quant.location_id → inventory_location` (stock loss).
5. `inventory_location = product.property_stock_inventory` or company default.
6. Create all moves and call `moves._action_done()` (with `inventory_mode=False` context).
7. Trigger auto-assign on downstream moves.
8. Update `location.last_inventory_date`.
9. Call `action_clear_inventory_quantity()` (reset counted).

**`_get_inventory_move_values`** (`L1253–1292`): Builds move dict with `is_inventory=True`, `picked=True`, `state='confirmed'`. Includes one move line inline. Lot, package, owner, restricted_partner preserved.

**Skip condition:** If context has `from_inverse_qty` AND `diff == 0`: skip (no-op inventory write).

#### Axis Classification
**DETERMINISTIC** — Axis-1. Mechanical: diff → move direction → done. No heuristics.

**Ontology:** (stock.quant)  
K-step: **K10**  
woa-rs target: `src/erp/inventory/inventory_adjustment.rs`

---

### R9 — `_trigger_assign` + Auto-Reservation Flow

**File:** `stock_move.py:L2482–2503`

#### Axis-1: Rich-AST Spec

```python
def _trigger_assign(self):
    if not self or config_param('stock.picking_no_auto_reserve'):
        return
    product_domains = OR([('product_id', '=', m.product_id.id), ('location_id', 'parent_of', m.location_dest_id.id)] for m in self)
    static_domain = [
        ('state', 'in', ['confirmed', 'partially_available']),
        ('procure_method', '=', 'make_to_stock'),
        ('reservation_date', '<=', today) OR ('picking_type_id.reservation_method', '=', 'at_confirm')
    ]
    moves_to_reserve = StockMove.search(static_domain & product_domains, order='priority desc, date asc, id asc')
    moves_to_reserve = moves_to_reserve.sorted(key=lambda m: any(r in self.reference_ids.ids for r in m.reference_ids.ids), reverse=True)
    moves_to_reserve._action_assign()
```

The secondary sort (reference_ids intersection) prioritizes moves that share a reference with the triggering move — this is a soft heuristic to prefer related document moves.

#### Axis Classification
**DETERMINISTIC** for the domain/filter. **HEURISTIC** for the reference_ids secondary sort (preference for related documents is a policy choice).

K-step: **K10**  
woa-rs target: `src/erp/inventory/reservation.rs`

---

## 3. Enterprise/Unresolved Flags

### FLAG-1: stock.move — UNRESOLVED in odoo_alignment

`stock.move` currently returns `None` from `resolve_odoo_to_family` in any alignment table.

**Proposed mapping:**
```
odoo:stock.move → owl:equivalentClass → gs1:LogisticEvent (GS1 Digital Link ontology)
              → OGIT family: needs alignment authoring
              → DOLCE: Perdurant (temporal event — a movement of goods in time)
```

GS1 has `gs1:LogisticEvent` for "an event in the logistics process" which fits stock moves well. However no OGIT family currently covers GS1 logistics events. **Options:**
1. Map to existing OGIT `SoftwareApplicationComponent` (wrong semantic fit).
2. Map to `Organization/OrganizationUnit` (wrong).
3. **Recommendation: Declare needs new alignment row.** Do NOT invent a CAM family. Flag for alignment authoring in a dedicated pass. Interim: use `Other(7)` as the ReasoningKind tag + `"StockMove"` as the proposed name.

### FLAG-2: stock.picking — UNRESOLVED in odoo_alignment

`stock.picking` currently returns `None`.

**Proposed mapping:**
```
odoo:stock.picking → owl:equivalentClass → ubl:DespatchAdvice (UBL 2.1 — "a document sent by a Supplier to a Customer")
                 → OGIT family: needs alignment authoring
                 → DOLCE: Perdurant (temporal process — coordinating a transfer of goods)
```

`ubl:DespatchAdvice` is the closest UBL class (covers both inbound and outbound logistics). The picking aggregate (collection of moves + header) maps to a despatch advice header. **Recommendation: needs alignment authoring.** Interim: `Other(8)` + `"StockPicking"`.

### FLAG-3: stock.quant — UNRESOLVED in odoo_alignment

`stock.quant` currently returns `None`.

**Proposed mapping:**
```
odoo:stock.quant → owl:equivalentClass → gs1:QuantityElement (GS1 — quantity at a location)
               → OGIT family: needs alignment authoring
               → DOLCE: Endurant (persistent entity — stock quantity at rest at a location)
```

A quant is a persistent record of how much of a product exists at a specific location with specific lot/package/owner characteristics. `gs1:QuantityElement` or alternatively `gs1:PhysicalInventoryReportInventoryType` (GS1 SCB). **Recommendation: needs alignment authoring.** Interim: `Other(9)` + `"StockQuant"`.

### FLAG-4: stock.warehouse — UNRESOLVED in odoo_alignment

`stock.warehouse` currently returns `None`.

**Proposed mapping:**
```
odoo:stock.warehouse → owl:equivalentClass → gs1:Location (GS1 — "a physical location")
                    → OGIT family: needs alignment authoring (closest existing: SoftwareApplication? No.)
                    → DOLCE: Endurant (persistent physical entity)
```

`gs1:Location` (or `vcard:ADR` for the address component) covers the physical site. The warehouse entity also carries operational configuration (picking types, routes). **Recommendation: needs alignment authoring.** Interim: `Other(10)` + `"StockWarehouse"`.

### FLAG-5: FEFO strategy — not fully community

The `removal_strategy = 'fefo'` path uses `with_expiration` context and `removal_date` on quants. The community code supports it in `_get_gather_domain` (`L767–768`) but the full FEFO UI configuration (product expiration dates, auto-application) may require enterprise modules (`stock_enterprise` / `product_expiry`). The community `_get_removal_strategy_order` does NOT include 'fefo' — it falls through to `UserError`. FEFO is handled purely by domain filtering (expiry date filter + lot-date sort), not a named order. Porter must handle this carefully.

### FLAG-6: Least-packages A* — performance boundary

The `_run_least_packages_removal_strategy_astar` has a `MemoryError` guard and uses a Python `heapq` priority queue. For large warehouse inventories (many packages), this can be slow or fail. The `best_leaf` fallback uses a partial/overselecting package combination. This is an explicit approximation — Axis-2 by design, but the A* itself is deterministic once the package list is known. In woa-rs, the least-packages strategy should delegate to the NARS Reasoner (Axis-2, `XorBundle` semiring over package combinations).

### No Enterprise boundary hits in this lane

The stock module in community is complete for the core functionality. `account_valuation` (landed costs, AVCO/FIFO costing methods) is partially in community and partially enterprise. The basic `price_unit` field on stock moves (L129) covers the community valuation hook. Full standard-cost/AVCO valuation (`_generate_valuation_lines`) is in `stock_account` (separate community module, not examined in this lane).

---

## 4. Ontology Mapping Summary

| Odoo class | owl pivot | OGIT family | DOLCE | Status |
|---|---|---|---|---|
| `stock.move` | `gs1:LogisticEvent` | **needs authoring** | Perdurant | UNRESOLVED — FLAG-1 |
| `stock.picking` | `ubl:DespatchAdvice` | **needs authoring** | Perdurant | UNRESOLVED — FLAG-2 |
| `stock.quant` | `gs1:QuantityElement` | **needs authoring** | Endurant | UNRESOLVED — FLAG-3 |
| `stock.warehouse` | `gs1:Location` | **needs authoring** | Endurant | UNRESOLVED — FLAG-4 |

All four classes need new alignment rows. Do NOT invent CAM families. Use `Other(7–10)` + proposed names as interim contract tags.

---

## 5. woa-rs Integration

### K-step
**K10 (Lager/Inventur)** — net-new. No K3/K7/K8/K9/K11/K12/K13/K15 overlap. Odoo inventory richness is entirely additive to woa-rs's existing `k10_inventory.rs` model.

### Existing woa-rs K10 baseline
`src/models/erp/k10_inventory.rs` (432 lines) contains:
- `warehouse::Model` — Lager, flat.
- `inventory::Model` — Bestand snapshot, no state machine.
- `stock_movement::Model` — Immutable ledger row (GoBD-compliant), bewegungsart enum as string.
- `serial::Model` — Seriennummern + MHD.

**Critical gap vs Odoo:** woa-rs has no:
- Move state machine (draft/confirmed/assigned/done).
- Reservation concept (reserved_quantity).
- Multi-step picking (picking type, backorder).
- Removal strategy (FIFO/FEFO/LIFO).
- Quant-level reservation tracking.

### Suggested new module: `src/erp/inventory/`

```
src/erp/inventory/
├── mod.rs              — module root
├── stock_move.rs       — state machine (Axis-1 deterministic port)
├── quant.rs            — available_quantity arithmetic, _update_available_quantity (Axis-1)
├── reservation.rs      — _action_assign stub + NARS delegation contract (Axis-2)
├── picking.rs          — picking state machine + _create_backorder (Axis-1 + Axis-2 judgment)
└── inventory_adjustment.rs — _apply_inventory, diff → move (Axis-1)
```

**Reservation delegates to lance-graph reasoning:**
```rust
// src/erp/inventory/reservation.rs
pub async fn assign_quants(
    context: ReasoningContext<StockReservationEvidence>,
    client: &dyn ReasonerContract,
) -> Result<Vec<(QuantId, Decimal)>, WoaError> {
    let conclusion = client.reason(context).await?;
    // parse (quant_id, qty) pairs from conclusion
}
```

The `ReasonerContract` trait comes from `lance-graph-contract` (BBB-allowed). The brain (NARS planner) is NOT in the customer binary.

---

## 6. Porter's Checklist — Non-Obvious Gotchas

1. **`product_qty` is read-only.** Writing `product_qty` raises `UserError`. Always write `product_uom_qty`. The field `product_qty` is a computed field converting `product_uom_qty` to the product's default UoM via `HALF-UP` rounding. Porter must replicate this computed field, NOT store product_qty directly.

2. **State is NOT a simple enum transition.** `_recompute_state` re-derives from `quantity` vs `product_uom_qty` at any time. After any write to `product_uom_qty` or `quantity`, state must be recomputed. This is a live invariant, not a simple FSM.

3. **Backorder uses Product Unit precision, not UoM rounding.** `float_compare(move.quantity, move.product_uom_qty, precision_digits=rounding)` where `rounding = precision_get('Product Unit')` (a named decimal precision in Odoo settings, typically 2). This is NOT the same as `product_uom.rounding` (which is a float factor like 0.001 for UoM with 3 decimals).

4. **`in_date` on quants uses `min()` of incoming dates.** When multiple quants at the same location are merged, `in_date = min(all_incoming_dates)` — oldest date wins. This is the FIFO invariant preserved at the quant level.

5. **`reserved_quantity = max(0, reserved + delta)` — can never go negative.** This is a hard floor. If a reservation is released for more than was reserved, it clamps at 0. Porter must replicate this clamp.

6. **`_action_assign` runs in a single loop with in-memory accounting.** The `assigned_moves_ids` and `partially_available_moves_ids` sets are passed into `_get_available_move_lines_out` to account for sibling moves that were processed in the same batch call. This prevents double-reservation within a single `_action_assign` invocation. Porter must replicate this in-loop accounting.

7. **Picking state 'partially_available' → picking state 'assigned'.** At the picking level, `partially_available` maps to `assigned` when `move_type == 'direct'` (as soon as possible). This is counter-intuitive — a picking with only partially fulfilled moves can still be `assigned` if the shipping policy allows partial delivery. See `_compute_state:L858–862`.

8. **`_gather` final sort: lot quants float to top.** After strategy ordering, `_gather` applies `.sorted(lambda q: not q.lot_id)` — quants WITH lots sort before quants WITHOUT lots. This ensures lot-tracked quants are consumed before untracked ones. Porter must replicate this secondary sort.

9. **`_should_bypass_reservation` short-circuits for non-storable products.** Services and consumables (`not product_id.is_storable`) never touch quants. Porter must gate all quant operations on `is_storable`.

10. **`propagate_cancel` chain.** Cancel propagation is conditional: ALL siblings must be cancelled before the destination move is cancelled. This prevents cancelling a shared-upstream move that another branch still needs. Porter must implement the sibling-state check.

11. **Concurrency via `try_lock_for_update`.** `_update_available_quantity` locks the first quant row for update (pessimistic locking). In Rust/sea-orm, use `SELECT ... FOR UPDATE` via raw SQL or sea-orm's `lock()` method. The `_merge_quants` cleanup must also be implemented as a scheduled task.

12. **`_create_backorder` at picking level resets `user_id = False` and `picked = False` on moves.** The backorder is unassigned (no responsible user). If `reservation_method == 'at_confirm'`, the backorder is immediately re-assigned (action_assign called). Porter must replicate this post-creation auto-assign.

---

## 7. NARS Candidates Summary

| Rule | Kind | InferenceType | Semiring | ThinkingStyle | Strength |
|---|---|---|---|---|---|
| R3 `_gather` removal strategy | NextBestAction | Induction + Abduction | XorBundle | Exploratory | **STRONG** |
| R4 `action_assign` move priority sort | NextBestAction | Induction | NarsTruth | Exploratory | STRONG |
| R7 backorder decision | NextBestAction | Abduction | NarsTruth | Exploratory | MODERATE |
| R9 reference_ids secondary sort | NextBestAction | Induction | NarsTruth | Exploratory | WEAK (deterministic fallback acceptable) |

---

## Depth Proof

Read: `/home/user/odoo/addons/stock/models/stock_move.py` lines=2683 depth=full  
Read: `/home/user/odoo/addons/stock/models/stock_quant.py` lines=1563 depth=full  
Read: `/home/user/odoo/addons/stock/models/stock_picking.py` lines=2149 depth=full  
Read: `/home/user/woa-rs/.claude/board/odoo-richness/BRIEFING.md` lines=124 depth=full  
Read: `/home/user/woa-rs/src/models/erp/k10_inventory.rs` lines=432 depth=full  
woa-rs calibration: grep confirms inventory is thin/absent at the Odoo-richness level (no state machine, no reservation, no quant model). Richness is net-new.
