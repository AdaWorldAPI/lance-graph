# Cypher query-engine autopsy — upstream vs AdaWorld fork

> **READ BY:** anyone proposing a Cypher parser, a Cypher→Quack seam, a
> prepared-query cache, or a claim about Cypher latency in this repo.
> **Status:** MEASURED (2026-10-07), except where a row says otherwise.
> **Specimens:** U = upstream `lance-format/lance-graph` (pinned control,
> DataFusion 50.3 / lance 1.0 / arrow 56); A = this fork (DataFusion 54.1 /
> lance 12 / arrow 58). Both run the SAME Cypher frontend code (parser,
> semantic analyzer, logical planner, DataFusion planner are byte-identical
> or differ only by additive `pub mod` lines and the DF54 API move).
> **Instruments (committed):** `crates/lance-graph-benches/examples/cypher_stage_probe/`
> (`common.rs` is compiled verbatim by both hosts), `crates/lance-graph-cypher-quack`
> (the experimental seam + `tests/differential.rs`), `.claude/tools/carrier_sufficiency.py --min`.

## A. Architecture (both specimens)

```
text ──nom parser──▶ owned AST ──SemanticAnalyzer(GraphConfig, $params)──▶ bound AST
     ──LogicalPlanner──▶ LogicalOperator ──DataFusionPlanner──▶ DF LogicalPlan
     ──(per call) SessionContext + InMemoryCatalog + MemTable registration
     ──optimizer + physical planning──▶ ExecutionPlan ──collect──▶ RecordBatch
```

Fork-only, experimental, routed nowhere:

```
bound LogicalOperator ──demand::classify──▶ Demand
                      ──lower_plan(Binding)──▶ quack::Query ──quack::lower──▶ mask-risc Program
                      ──execute_into(borrowed lanes)──▶ scalar
```

## B. Regression matrix

| Capability | U | A | Class |
|---|---|---|---|
| Q0–Q10 + var-length (#1305 DAG 4/3, cycle 5) answers at 2/5/100/10K/1M rows | ✓ | ✓ identical | A PARITY |
| Upstream's 12 integration test files run against the fork (241 tests) | ✓ | ✓ 241/241 | A PARITY |
| `count(n)` / `count(DISTINCT b)` on a node variable | ✗ plan error `b__id` | ✗ same | D FRONTIER (inherited: `datafusion_planner/expression.rs` hard-codes `<var>__id`) |
| `sum()` over no rows | NULL | NULL | D FRONTIER (inherited SQL semantics; openCypher says 0) |
| Re-executing one DataFusion physical plan | panics `partition not used yet` | same | D FRONTIER (RepartitionExec is one-shot) |
| `graph_execution` bench measured only `.next()` | defect | repaired (#1375) | C FORK ADVANCE |
| Delta reader | present | removed (deliberate, dependency ruling) | B REGRESSION, deliberate |
| `default-features = false` build of `lance-graph` | — | ✗ does not compile (ndarray used outside `ndarray-hpc`) | B REGRESSION (the documented fallback mode is broken) |
| `CsrIndex` used by Cypher execution | no | no | potential only |
| Python README "Build a Knowledge Graph from Text" | runs | should run unchanged (all Python on the path byte-identical; binding signatures unchanged) | A PARITY by source; NOT RUN (pyarrow/pylance/maturin absent) |
| Quack execution of bound Cypher | — | count/sum/min/max + WHERE; one-hop count(*); one-hop count(DISTINCT target) | C FORK ADVANCE (experimental) |
| Prepared parameters | none (re-plans) | Quack Program patch: 1004/1004 patched == fresh lowering | C FORK ADVANCE (probe) |

## C. Timing (median, µs, quiet machine, release, debug=0)

Cold DataFusion path, below 10K rows: **650–5 000 µs total**, of which physical
planning 340–3 700, MemTable registration 140–420, DF logical planning
40–370, parse 6–30 (in-process cold), bind ≲20, logical plan ≲20.

Parser micro-bench (identical in U and A — same code):

| query | parse ns | allocs | bytes | alloc-only control ns |
|---|---|---|---|---|
| Q0 | 1106 | 17 | 2246 | 542 |
| Q1 | 1476 | 25 | 3217 | 857 |
| Q4 | 1268 | 27 | 2535 | 793 |
| 296-char | 7868 | 114 | 7351 | 4306 |

Allocation is ~half of parse time; parse is **< 0.1 %** of a cold query.

1M rows, execution only, A vs U (DF 54 vs DF 50, same Cypher code):
Q3 28 vs 52 ms · Q4 64 vs 99 · Q6 49 vs 70 · Q10 74 vs 124 · QV 12 vs 33 ·
Q8 ~23 vs 40–43 · **slower:** Q0 0.49–0.62 vs 0.36–0.42 · Q7 15.9 vs 13.2–14.1.

Quack arm (hand-lowered, 1M rows, exec): Q0 3.3 µs · Q1 0.20 ms · Q2 0.52 ms ·
Q3 2.4 ms · Q4 18.9 ms (two folds + host dot) · Q5 0.90 ms · Q6 6.5 ms ·
Q9 0.25 ms (`Any` does not short-circuit).

Prepared point lookup (Q5): Quack patch+exec **10.4 µs @10K / 862 µs @1M**
vs DataFusion re-bind+plan+exec **1117 µs / 1898 µs**.

## D. Repair list

- **P0** — none found.
- **P1** — Delta reader removed (deliberate; restore only by its own PR).
- **P2** — `<var>__id` hard-code (inherited); `sum` over empty = NULL
  (inherited); DF physical plan not re-executable (inherited, D-class);
  `default-features = false` does not build (fork).
- **P3** — DF 54 slower on Q0/Q7 (upstream-engine regression, not ours);
  mask-risc `Any` scans fully.
- **P4** — allocation-light AST (≤ ~0.5 µs per query to win).

## E. Primitive gaps (each named by the consumer that needs it)

| Gap | Consumer | Truly absent? |
|---|---|---|
| foreign-value sum (Σ over a hop of a per-node count) | 2-hop `count(*)`, `*1..2` | yes in Quack; DF does it as a join |
| ordered compare through an fk | `WHERE a.age > x` + hop | yes — only `EqU32Via` exists |
| hop chain over a computed frontier | var-length, 2-hop DISTINCT | forbidden shape today (Keep → Semijoin) |
| `Any` short-circuit | `EXISTS` / `LIMIT 1` | executor choice, not a new op |

No primitive is created by this autopsy.

## F. Parser ruling — **KEEP**

Measured parse is < 0.1 % of a cold query and identical in both specimens. A
fast parser buys nothing until planning is cached. An allocation-light AST is P4.

## G. Execution ruling

The seam is **bound `LogicalOperator` → `quack::Query`** with a fork-owned
`Binding`, gated by the #1305 demand classifier, refusing with a typed reason
rather than falling back. Lowering from the raw AST would duplicate binding;
Quack's `bind::Draft` is too narrow (eq/ne, count/rows, one table). The next
slice is prepared programs (one op = one parameter slot), which the probe
already shows is exact.

## H. The README mini-program as a compatibility specimen

Traced from source (not run: no pyarrow/pylance/maturin locally).

- `graph.yaml` → `KnowledgeGraphConfig.load_graph_config` (re-reads YAML each
  call) → binding `GraphConfig` builder (clones per `with_*`) → Rust
  `GraphConfig` (labels lowercased).
- `LanceGraphStore`: one `<root>/<Label>.lance` dataset per table, case kept;
  `upsert_table` rewrites the WHOLE table per call (scan → `to_pylist` → dedupe
  → `from_pylist` → overwrite), minting a version each time.
- `LanceKnowledgeGraph.run()` caches nothing: new `CypherQuery`,
  `node_labels()`/`relationship_types()` in QUERY case (`MATCH (e:entity)` looks
  for `entity.lance`), full scan of each referenced table (no projection or
  limit pushdown), Arrow C-stream import (zero-copy for one chunk), new
  `SessionContext` + catalog + MemTable, full plan, collect, export.
- `CypherEngine` builds config + catalog + `SessionContext` once; each
  `execute` still re-parses/re-plans/re-optimises, skipping only the scan,
  import and context build. It is a frozen snapshot (later upserts invisible)
  and does **not** lowercase field names while `run()` does — mixed-case
  columns can behave differently under the two APIs (both specimens).
- The heuristic extractor never yields relationships, so the README program
  never writes a RELATIONSHIP dataset.

**Can `graph.yaml` be the one schema source?** Not as parsed today: only
`nodes.*.id_field` and `relationships.*.{source,target}` are read and every
other key is dropped. A numeric binder needs ordered properties with lane
kinds, id type, endpoint labels and declared ordering. Rust already carries
untyped `NodeMapping.property_fields` / `RelationshipMapping.type_field`. The
natural extension is an ordered `properties: [{name, type}]` list (list order =
ordinal) plus `source_label` / `target_label` / `ordered_by`, compiled once into
both `GraphConfig` and the cypher-quack `Binding` (see `D-BIND-BUNDLE-0.md` §1).
q2's `quarto-yaml` (spans on every node, key order kept) is reusable as the
parse layer; `quarto-yaml-validation` validates structure but stops at the first
error and has no integer type; `quarto-config` drags pandoc types and is not
reusable. No JS runtime (q2 removed deno_core/rusty_v8 for exactly the reason
that nothing executable needed it).

**Stage split to measure when a Python toolchain exists** (deliberately not
merged into one number): query construction/parse · `node_labels()` discovery ·
Lance scan to pyarrow · C-stream import · catalog/context build · bind+plan ·
execute · `to_pylist`. `run()` vs `CypherEngine.execute` differ exactly in
scan + import + context build.
