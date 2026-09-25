# 2026-09-25 — `Lowering::Tern3`: the six-view window holds six planes and materialises them once

**Status:** MEASURED · DONE (production lowering) · OPEN (why Keep varies more than Count; the inlined-ceiling gap from the Tern2 entry is unchanged)
**Extends:** `entries/2026-09-25-tern2-two-level-lowering.md` from 4–5 to exactly six planes.

## The idea
The masked views are HELD rather than evaluated op by op. A Boolean chain over six resident planes is read once as a 64-bit truth table. It is then materialised in ONE pass per chunk, as a tree of three ternlog tables. That is the `ceil((6-1)/2) = 3` floor, because each three-input table absorbs exactly two binary combinations. Before this change, six planes fell to the tiled path, which spends one pass per op.

## What landed (`crates/lance-graph-mask-risc`)
- `Program::fused_tern3`, tried after `fused_tern2` and claiming exactly six distinct planes, recognises the chain through `chain_table6`. `decompose6` then tries three split shapes in order:
  - **balanced** `h(g1(a,b,c), g2(d,e,f))`: the 8×8 chart (rows are the `A` assignments, columns the `B` assignments) has at most 2 row classes and at most 2 column classes, and the two inner tables are independent;
  - `H(t, p, q)`, where `t` is a Tern2 split of the other four planes, embedded in five positions with a don't-care leaf;
  - `H(t, w)`, where `t` is a Tern2 split of the other five planes.
- `chain_table5` is now a view of `chain_table6`: with leaf 5 unused, the low half of the 64-bit table IS the 32-bit table. One interpreter replaces two.
- Execution (`run_fused_tern3`):
  - `t0` and `t1` go into two on-stack `TILE_WORDS` chunks.
  - The root is folded into `Count`/`Any` or written into `Out::Mask`.
  - No scratch slot is used and nothing is allocated.
  - Edge words follow Tern2's rules.
  - `Keep` without `Out::Mask` still runs tiled.
- The can-stay-silent case is six-input majority (at least 4 of 6), which has no three-table tree of any shape.

## Numbers (`examples/tern3_probe.rs`, n = 2^20, `target-cpu=native`, median of 41, 3 runs)
Every arm is asserted equal to the tiled result. The tiled path keeps intermediates in tile-sized scratch, so memory traffic is the six plane reads either way. The saving is PASSES per tile, 5 against 3, so the prediction was 5/3 = 1.67×.

| chain | passes | Count tiled → Tern3 (µs) | × | Keep × |
|---|---|---|---|---|
| `a&b&c&d&e&f` | 5 : 3 | 17.8–21.0 → 10.4–12.1 | 1.71–1.74 | 1.75–1.78 |
| `(a&b&c)^(d\|e\|f)` | 5 : 3 | 18.9–22.2 → 10.5–12.2 | 1.80–1.82 | 1.35–2.07 |
| `((((a&b)\|c)^d)&e)\|f` | 5 : 3 | 18.8–21.8 → 10.3–12.0 | 1.82 | 1.50–1.51 |

Count lands at or just above the pass-count prediction. Keep also writes the full output bitmap and varies more from run to run; that spread is not yet explained.

## Tests and disable runs
- **Unit (`ir.rs`):**
  - every plan any shape returns rebuilds its own function bit for bit (fixed tables plus 20,000 structured random tables);
  - each shape is pinned by a table only it claims;
  - majority declines.
- **`tests/tern3.rs`:**
  - random six-plane chains match the tiled twin over Count/Any/Keep and every awkward extent (942 held, 42 declined);
  - each shape matches the row-at-a-time oracle end to end;
  - `Keep` without `Out::Mask` stays tiled.
- **Disable runs,** each red, then restored green:
  - the executor reading `t1` for `t0`;
  - the balanced shape removed;
  - the balanced root table corrupted;
  - Keep's tail clear dropped.
