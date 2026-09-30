# A Boolean mask is the support of a frontier, never its bag (2026-09-30)

**Status:** MEASURED + TEST-PINNED. Plan: `.claude/plans/cypher-mask-multiplicity-contract-v1.md` (ratified v3).

**Finding.** `cypher-mask-lowering-v1` lowered `count(*)` to the popcount of the final mask. After a hop, Cypher's bag semantics count one row per binding (path), not one per node.

| query on KNOWS = {1→2, 1→3, 2→3, 3→4, 4→5} | DataFusion | mask chain |
|---|---|---|
| `(a)->(b)->(c) RETURN count(*)` | 4 | 3 |
| `count(DISTINCT c.id)` | 3 | 3 |
| `*1..2` from id 1, `count(*)` | 4 | 3 |
| `count(DISTINCT b)` over the 2-hop | 3 | forward `dst₁` = 4 |

The last row is the second defect: a forward hop chain is the exact support of the TERMINAL variable only.

**Divergence recorded OPEN.** On KNOWS = {1→2, 2→1, 2→2}, DataFusion's 2-hop `count(*)` is **5**, the walk count. Cypher's relationship uniqueness gives 4, the trail count. DataFusion is in grace, so this is not fixed.

**Code.** `LogicalOperator::consumer_semantics()` (`logical_plan.rs`) returns one of five carrier kinds: `TerminalSet`, `EarlierSet`, `TerminalCount`, `EarlierCount`, `Bindings`. It is not `Serialize`, and a guard test enforces that.

The G2 disables each went red-then-green:
- force sensitivity to false;
- force the focus to the terminal;
- drop the cross-variable filter;
- let `Join` through.

**Census (W0-b), debug 0.** 310 classified.

| | before | after |
|---|---|---|
| Full | 117 | 70 |
| Split | 193 | 240 |

New reasons, by query count:

| reason | queries |
|---|---|
| Bindings | 90 |
| TerminalCount | 35 |
| EarlierCount | 9 |
| EarlierSet | 7 |

The value-DISTINCT T-12 reason moved 7 → 6.

After the chain-linearity fix (Codex P2 on #1305), the census reads 313 queries. The three new classifier-test queries join the corpus as `Bindings`. Full is unchanged at 70, so no existing query was reclassified.

**OPEN.**
- `lance-graph-as-the-modelgraph-v1.md` §15 still quotes the pre-contract Full fraction (37.3 %). It needs a footnote.
- No count-lane operator exists. `Weighted` lowering is gated on the exactness conditions in contract §3.2 R4.

**Amendment (same day, before merge).** Contract §7.
- v1 semantics is WALK: DataFusion, Ladybug (`PathSemantic::WALK` default, no `r1 <> r2` in `rewriteMatchPattern`) and SQL joins all compute it. D-CMM-5 is reclassified from "DataFusion divergence" to "TRAIL is a separate mode".
- Walks and trails also diverge on an acyclic graph when the pattern changes direction: `(a)->(b)<-(c)` on {1→2} gives `count(DISTINCT c)` 1 as a walk, 0 as a trail.
- MEASURED carrier sufficiency (`python3 .claude/tools/carrier_sufficiency.py`): node support answers only Exists/Support under WALK; per-node counts add Count/CountBy of the current and later nodes; the last hop's edge population adds the previous node; nothing per-node or per-edge answers a TRAIL question two hops on (two parallel self-loops plus 1→0: 3-hop trail count 0 vs 2 with equal per-edge trail counts).
- The mask-RISC survival rule stays; §7.4 states the condition under which a later PR may relax it.

