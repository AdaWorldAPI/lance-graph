# cypher-mask-multiplicity-contract-v1 — a mask is the SUPPORT of a frontier, never its multiplicity

> **Status:** RATIFIED v3 (5+3 council, Phase 5). Corrects `cypher-mask-lowering-v1.md`.
> No mint. No DataFusion extension. Debug-0 builds only.
> **D-ids:** D-CMM-0 (plan corrections), D-CMM-1 (classifier), D-CMM-2 (DataFusion pins),
> D-CMM-3 (census consumer), D-CMM-4 (count lane, queued), D-CMM-5 (walk/trail divergence, open),
> D-CMM-6 (modelgraph footnote, queued) — rows in `STATUS_BOARD.md` under `D-CMM`.

## §0 The defect, in one fixture

KNOWS = {1→2, 1→3, 2→3, 3→4, 4→5} — the edges of `create_knows_dataset` in `crates/lance-graph/tests/test_explain_output.rs`.

`MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) RETURN count(*)`

- Bag semantics: paths 1→2→3, 1→3→4, 2→3→4, 3→4→5 ⇒ **4**. DataFusion returns 4 (G1a, measured).
- `cypher-mask-lowering-v1` R-6 chain + T-1 popcount: hop₁ dst = {2,3,4,5}, hop₂ dst = {3,4,5} ⇒ **3**.
- `RETURN count(DISTINCT c)` ⇒ **3**. After a hop, the plan's lowering answers the DISTINCT-endpoint question and calls it `count(*)`. Over a single population, T-1 is exact.
- `RETURN count(DISTINCT b)` ⇒ **3** ({2,3,4}), but R-6's forward `dst₁` = {2,3,4,5} ⇒ 4. A forward mask is the exact support of the TERMINAL variable only.

On this acyclic fixture, walk count = trail count. The two diverge only on cycles (G1b).

`-[:KNOWS*1..2]->` from `a.id = 1`: DataFusion returns 4 rows, targets Bob, Charlie, Charlie, David
(`test_datafusion_pipeline.rs:2226-2258`); distinct endpoints = 3.

The same defect reaches `RETURN c` and `RETURN c.name` after a hop (T-3/T-4): Cypher returns one
row per binding, so Charlie appears twice above; a mask holds Charlie once.

The plan already graces `collect` and ORDER as multiplicity (`INTEGRATION_PLANS.md:212`, `w0b_corpus_census.rs:230`). It does not recognise that `count`/`sum`/`avg`/`RETURN` after a hop are multiplicity-dependent. No entry found in `.claude/plans`, `.claude/board`, or `.claude/knowledge` states that (search: multiplicity | bag semantics | walk.*trail).

## §1 Frozen decisions

- F1. DataFusion is in grace: maintained, not extended (`CLAUDE.md` § ⊘ OPERATOR RULING 2026-09-05).
- F2. Mask programs are a lowering target; nothing is minted (`cypher-mask-lowering-v1.md` §5.2).
- F3. One-population, tail and transpose laws hold (`cypher-mask-lowering-v1.md` §5.3).
- F4. Absence is a zero fallback, not a NULL (`mask-risc/src/lib.rs:44-47`).
- F5. Builds use `CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_INCREMENTAL=0`.

## §2 Input inventory (measured by the savants, file:line read)

| site | claim | status under §0 |
|---|---|---|
| plan T-1/T-2 (`:359-360`) | `count(*)`/`count(n)` = popcount | exact with no hop; wrong after a hop |
| plan T-3/T-4 (`:361-362`) | `RETURN n` / `n.prop` = the mask / mask+lane | exact with no hop; after a hop a node repeats once per binding |
| plan T-5/T-7 (`:363,365`) | `sum` / `avg` = masked sum / popcount | wrong after a hop (weighted by path count) |
| plan T-6 (`:364`) | `min`/`max` | exact after a hop for the terminal variable only |
| plan T-11/T-12 (`:369-370`) | `DISTINCT n` free; `count(DISTINCT …)` grace | T-11 right; T-12 must split: DISTINCT over a node variable = support ([G]), over a value = [GRACE] |
| plan R-6 (`:339`) | 2-hop = `dst₁` becomes `src₂` | exact support of the TERMINAL variable only |
| plan R-7/R-8 (`:340-341`) | var-length = delta-frontier fixpoint | mechanism `[G]` for terminal support/reachability; carries no multiplicity |
| plan §4.1 (`:413-414`), §4.6 (`:489`), §11 (`:961, 969-971`) | "`count(*)` is a popcount"; "how many rows? yes" | hold for one population only |
| plan §7 F-F2 (`:830`), F-S2 (`:842`), W0-c (`:756-762`) | SET equality is the oracle | cannot see multiplicity; the §0 fixture passes it while returning 3 |
| `INTEGRATION_PLANS.md:207-209` | "35 [G] · 9 [H] · 9 [GRACE]" | already stale against the plan's own 31/13/9 |
| `examples/w0b_corpus_census.rs:209-245, 323-426` | a hop + `count`/`sum`/`avg`/`RETURN c` is lowerable; `count(DISTINCT n)` is grace | wrong in both directions; the Full fraction is biased with unknown sign |
| `logical_plan.rs:94-97, 549-560, 570-575` | aggregates live in `Project.projections` as `AggregateFunction{distinct}`; `RETURN DISTINCT` wraps `Distinct` | classifier needs no AST |
| `datafusion_planner/builder/expand_ops.rs:147-204, 281` | var-length unrolls and UNIONs; fresh rel alias per hop; no edge-inequality predicate in `:140-290` (read; no whole-crate negative claimed) | DataFusion counts WALKS: G1b measures 5 against Cypher's 4 |

## §3 Resolution

### §3.1 PR 0 — correct the plan (doc only; narrow, never delete)

A mask answers *which distinct nodes the TERMINAL variable can take*. An earlier variable's support needs a backward (semi-join) pass over the transpose, which R-6 does not contain.

- **"single population"** replaces "no hop" everywhere: no `Expand`, `VariableLengthExpand`, `Join`, or `Unwind`.
- T-1/T-2/T-3/T-4/T-5/T-7 each split:
  - a single-population row keeps its grade;
  - a post-hop row is `[GRACE]` in v1. That is a routing decision, not a verdict: the row becomes `[H]` under §3.2 rule 5.
- The post-hop GRACE applies to the non-DISTINCT forms only:
  - `RETURN DISTINCT c` over the terminal variable stays T-11 `[G]`;
  - T-9 `EXISTS` is unaffected.
- T-6 post-hop stays `[G]` for the terminal variable only.
- T-12 splits three ways:
  - DISTINCT over the terminal node variable is `[G]`;
  - over a non-terminal variable, `[GRACE]` in v1;
  - over a value, `[GRACE]`.
- R-6/R-7/R-8: the mechanism stays `[G]` for support and reachability of the terminal variable. The rows gain "carries no multiplicity".
- §4.6 is narrowed, not deleted:
  - "how many rows?" becomes "how many distinct nodes of the terminal variable? — yes";
  - a new row reads "how many bindings? — no in v1".
- §4.1's sentence is narrowed to "`count(*)` over a single population is a popcount; after a hop it is not".
- §7 F-F2, F-S2 and W0-c each gain a multiplicity arm that **records** the divergence on the §0 fixture and routes the query to GRACE. It is never a pass condition on the mask path.
- §11's row count and tally are recomputed row by row. `INTEGRATION_PLANS.md` is append-only, so its stale "35/9/9" gets a dated correction line, not an edit.

### §3.2 PR A — the contract, and its one real consumer

The vocabulary is the house one: *factorized* (Kuzu; `graphrag-industry-comparison.md:41`, `LATEST_STATE.md:3557`) and the plus-times semiring (GraphBLAS). `Frontier` is taken (`planner/src/nars/tactics.rs:151`), so the carrier is named `BindingFrontier`.

```
BindingFrontier { support: Mask,                 // Boolean semiring
                  mult:    Option<count lane>,   // plus-times, u64, checked
                  binding: Option<CSR slices> }  // parent identity, factorized; native-side only
```

`ConsumerSemantics` names the **carrier** a query's consumer needs. It does not say the query lowers:

| kind | carrier | examples (after a hop over a→b→c) |
|---|---|---|
| `TerminalSet` | support of the terminal variable | `RETURN DISTINCT c`, `count(DISTINCT c)`, `min(c.age)`; every single-population query |
| `EarlierSet` | support of ONE earlier variable (backward semi-join) | `count(DISTINCT b)`, `min(a.age)` |
| `TerminalCount` | support + mult keyed on the terminal | `count(*)`, `sum(c.age)`, `RETURN c`, `RETURN c, count(*)` |
| `EarlierCount` | support + mult keyed on ONE earlier variable (backward count) | `RETURN a, count(*)`, `RETURN a`, `sum(a.age)` |
| `Bindings` | identity of ≥2 variables | `RETURN a, c`, `WHERE a.age = c.age`, nested `Project` (`WITH`), `Join`, `Unwind` |

Classification, over the top `Project` (through `Sort`/`Offset`/`Limit`/`Distinct` wrappers):

1. `Join`, `Unwind`, a nested `Project`, a non-`Project` top, or any unknown shape ⇒ `Bindings`. The classifier fails closed.
2. No hop (single population) ⇒ `TerminalSet`: one row is one node.
3. With a hop:
   - a `Filter` whose predicate references ≥2 variables ⇒ `Bindings`;
   - otherwise let V = the variables the projections reference (`*` excluded). |V| ≥ 2 ⇒ `Bindings`;
   - V = ∅ ⇒ the focus is the terminal; otherwise the focus is the single member of V;
   - *multiplicity-sensitive* ⇔ some non-DISTINCT `count`/`sum`/`avg`/`collect`, or no aggregate and no `Distinct` wrapper;
   - focus × sensitivity picks one of the four remaining kinds.
4. Orthogonal to the carrier, and judged separately by the plan §4 rows: `ORDER BY`, `SKIP`/`LIMIT`, value-`DISTINCT`, `collect` as a sequence, value-keyed grouping.

Rules for a future lowering (none is built here):

- **R1.** In v1 only `TerminalSet` may lower fully. For a non-`TerminalSet` consumer over a mask-lowerable prefix, the outcome is `Split` or `Grace`. The kind is recorded so `Split` stays reachable.
- **R2.** `binding` reuses CSR slices (`AdjacencyBatch`, `planner/src/adjacency/batch.rs:37-57`, `!Clone`). It never becomes an owned per-binding vector, and it never crosses to Java.
- **R3.** Unfold, when coded, is `materialize_bindings`, O(output) in its doc. `mask-risc/src/exec.rs:329` "the ONE materialiser" then gets a scope note.
- **R4.** "Exact" means mult equals the Cypher trail count:
  - fixed k-hop over one directed type is exact iff no closed walk of length ≤ k−1 exists;
  - any undirected hop is `[GRACE]`;
  - var-length is exact only on a proven-acyclic relation, or within girth;
  - mult comes from edge multiplicity, never from a deduplicated adjacency;
  - overflow refuses, never wraps.
  - Matching DataFusion's count is a separate, OPEN property (§4).
- **R5.** A `WITH` that aggregates or is `DISTINCT` resets multiplicity to 1; a plain `WITH` carries it. v1 classifies any nested `Project` as `Bindings`.

PR A ships:

- **The enum and the method.** `enum ConsumerSemantics` derives `Debug, Clone, Copy, PartialEq, Eq`, with no serde. A guard test proves `!Serialize` via inherent-vs-trait method resolution, and has a can-fire arm on `LogicalOperator`, which is `Serialize` (`logical_plan.rs:18`). The method is `LogicalOperator::consumer_semantics(&self)`, pure. The crate is edition 2021: no let-chains.
- **Its consumer.** `w0b_corpus_census.rs`'s `classify_plan` calls it. A non-`TerminalSet` query with a hop gets a grace reason naming its kind. The census's T-12 arm splits: `DISTINCT` over a node variable is not grace; over a value it is.
- **The differential tests (G1)** in the existing `tests/test_datafusion_varlength_complex.rs`. This crate keeps top-level test files and has no `tests/integration/`, so a new binary is not warranted.
- **No `BindingFrontier` type, no operator, no `Cargo.toml` change.**

## §4 Non-goals

- No DataFusion extension node, no change to `expand_batch`, no trail enforcement in DataFusion.
  DataFusion counts WALKS: measured 5 on G1b, where Cypher's trail count is 4. The divergence is **recorded OPEN** (F1).
- No counting op in mask-risc's IR. No `BindingFrontier` type in code; only the `ConsumerSemantics` enum ships.
- No OPTIONAL MATCH. No change to the dense-rowid address model.

## §5 Pre-registered gates

- **G1a (DAG, measured green):**
  - 2-hop `count(*)` = 4; `count(DISTINCT c.id)` = 3;
  - `*1..2` from id 1: `count(*)` = 4; `count(DISTINCT b.id)` = 3.
- **G1b (cycle, divergence pin, measured green):** KNOWS = {1→2, 2→1, 2→2}. 2-hop `count(*)` asserts `== 5`, the walk count. The Cypher trail count is 4. It is a record of DataFusion's behaviour, not a correctness claim.
- **G2 (classifier, unit tests in `logical_plan.rs`):** at least one case per kind, including:
  - `count(DISTINCT b)` ⇒ `EarlierSet` (the §2 defect);
  - `RETURN c` ⇒ `TerminalCount` vs `RETURN DISTINCT c` ⇒ `TerminalSet`;
  - `RETURN a, count(*)` ⇒ `EarlierCount`; `RETURN c, count(*)` ⇒ `TerminalCount`;
  - `WHERE a.age = c.age` ⇒ `Bindings`;
  - `WITH` ⇒ `Bindings`.
- **G2 disables (red-then-green each):**
  - force the multiplicity-sensitivity to `false` ⇒ the Count arms fail;
  - force the focus to the terminal ⇒ the Earlier arms fail;
  - drop the cross-variable Filter check ⇒ the `Bindings` arm fails.
- **G3 (can stay silent):** single-population `count(*)` ⇒ `TerminalSet`. Paired arm: two disconnected scans (`Join`) with `count(*)` ⇒ `Bindings`.
- **G4 (census moves):** debug 0, before and after. Pre-registered:
  - the new multiplicity/earlier reason fires on > 0 queries;
  - the T-12 node-variable rescue is reported as its own count.
  - Recorded in the board entry.
- **G5:** debug 0 for `cargo test -p lance-graph --lib logical_plan`, the G1 tests, and `cargo clippy -p lance-graph --all-targets -- -D warnings`; `cargo fmt --check`. `Cargo.toml` unchanged. No model identifier in the diff or in commit messages.
- **G6:** the PR 0 edits cite §0. The three tallies agree. Board writes use `Edit` (never an open-for-write that reads the same file), are checked with `wc -l`, and `supersession_index.py` is regenerated LAST.

## §6 Change ledger v1 → v2

- `Frontier` → `BindingFrontier` (name collision). #163 ListArray citation dropped (unverified).
- Three consumer kinds → four (`Grouped` added; `CountBindings` → `Weighted`, and it now covers `sum`/`avg`/`RETURN <var>`).
- Rule 5 generalised: girth, directed-only, edge multiplicity.
- Cyclic fixture G1b added (the DAG cannot tell walks from trails).
- G2 gets paired disables. The classifier becomes a method, and is wired into the census as its consumer (answers AP6 dead surface).
- PR 0 scope widened: T-3..T-7, T-12, §4.1, §11, the wave oracles, and the `INTEGRATION_PLANS` tally.
- DataFusion's walk semantics recorded as an OPEN divergence.
- v2 → v3 (reviewers):
  - The enum is re-cut by carrier into five kinds. `Grouped` is split because the terminal and earlier variables need different carriers.
  - `SetOnly` is restricted to the terminal variable; the overclaim-auditor's `count(DISTINCT b)` = 3 vs 4 finding is now in §0.
  - "No hop" becomes "single population" (a `Join` has no hop).
  - `WITH` reset is limited to aggregating or DISTINCT `WITH`.
  - `Split` stays reachable (R1).
  - G1b is a strict `== 5` divergence pin; G2 has three binding disables.
  - A `!Serialize` guard is added; `INTEGRATION_PLANS` gets a correction line, not an edit.
  - The model identifier was removed from the Phase-0 commit trailer.
- Retained from `cypher-mask-lowering-v1` unchanged:
  - the Boolean core (§3.1-§3.3);
  - R-1 to R-5 and R-9;
  - T-8, T-9 and T-10;
  - the §5 placement ruling and §5.3 laws.
