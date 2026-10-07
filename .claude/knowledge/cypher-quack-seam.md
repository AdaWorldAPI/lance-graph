# Cypher → Quack seam — where Cypher meaning stops and execution starts

> **READ BY:** `cypher-lowering-warden`, `query-stage-profiler`,
> `fold-carrier-scientist`, and any session that lowers Cypher (or any graph
> query language) onto Quack / mask-risc / R2IL.
> **MANDATORY BEFORE:** editing `crates/lance-graph-cypher-quack`, adding a
> Cypher fast path, or routing any Cypher query away from DataFusion.
> **Status:** CURRENT-CONTRACT for the experimental crate (TEST-PINNED by
> `tests/differential.rs`); MEASURED numbers in
> `.claude/research/cypher-engine-autopsy.md`.

## 1. The seam

Lower from the **bound `LogicalOperator`**, never from the raw AST. The
upstream parser + `SemanticAnalyzer` + `LogicalPlanner` already bind names
against `GraphConfig` and substitute `$params`; lowering from the AST would
bind a second time. Quack's `bind::Draft` is too narrow to be the seam
(eq/ne, count/rows, one table).

```
text → parse → semantic → LogicalOperator → demand::classify → lower_plan(Binding)
     → quack::Query → quack::lower → mask-risc Program → execute over borrowed lanes
```

## 2. Demand before lowering (the #1305 law)

`demand::classify` returns `TerminalSet | EarlierSet | TerminalCount |
EarlierCount | Bindings`. It decides **which carrier is sufficient** before
any op is emitted:

- one-hop `count(*)` → edge-table row count (one edge row is one walk);
- one-hop `count(DISTINCT target)` → `CountDistinctOrderedU32` over a
  dst-ordered edge table;
- `Bindings` → refuse (needs tuples, not a scalar).

**Standing oracles (never delete):**
- DAG `{1→2,1→3,2→3,3→4,4→5}`: 2-hop `count(*)` = **4** paths, **3** end nodes.
- cycle `{1→2,2→1,2→2}`: 2-hop walks = **5**, trails = 4 (DataFusion and Quack
  are both WALK).
- parallel edges `0→1 ×2, 2→2, 2→1`: one-hop walks = **4**, distinct targets = 2.

A mask is support. Never answer a walk count with a mask.

## 3. Refuse, never fall back

Every unlowered shape returns a typed `Refusal` carrying the construct. A
refusal is the answer. There is no silent fallback to DataFusion inside the
seam; a caller that wants one must make that routing decision explicitly and
visibly.

Current gaps (each is a primitive gap, not a seam bug):
- **two or more hops** — needs a foreign-value sum (2-hop count = Σ in·out, a
  factorized fold rule; legal for count/sum, illegal for count-distinct across
  factors);
- **WHERE with a hop** — ordered compare through an fk (only `EqU32Via`
  exists); a node-table `Keep` fed to the edge `Semijoin` is the forbidden
  two-program shape;
- **variable length** — hop chain over a computed frontier.

## 4. Layout facts are bind facts

`EdgeTable::dst_ordered` decides whether `count(DISTINCT b)` is lowerable at
all. Sort order, uniqueness and run guarantees are V3 representation: they
belong to the binding and invalidate it when they change. They are never
execution statistics.

## 5. Inherited DataFusion divergences (pinned, not ours to hide)

- `count(n)` / `count(DISTINCT b)` fail to plan — `<var>__id` is hard-coded.
- `sum()` over no rows is NULL; openCypher says 0. Quack returns 0.
- A DataFusion physical plan cannot be executed twice (RepartitionExec).

## 6. Prepared parameters

Quack copies literals into `Pred` with no value-dependent folding, so a
parameter slot is the op index of a sentinel literal: patch and execute.
Measured exact for 1004 values (patched == fresh lowering). If a lowering ever
folds on a value, the slot must move up into `Query`.
