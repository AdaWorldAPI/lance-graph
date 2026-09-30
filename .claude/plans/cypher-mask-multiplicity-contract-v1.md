# cypher-mask-multiplicity-contract-v1 — a mask is the SUPPORT of a frontier, never its multiplicity

> **Status:** SPEC v1 (5+3 council, Phase 0). Corrects `cypher-mask-lowering-v1.md`.
> No mint. No DataFusion extension. Debug-0 builds only.

## §0 The defect, in one fixture

`crates/lance-graph/tests/test_explain_output.rs:32-47` KNOWS = {1→2, 1→3, 2→3, 3→4, 4→5}.

`MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) RETURN count(*)`

- Cypher bag semantics: paths 1→2→3, 1→3→4, 2→3→4, 3→4→5 ⇒ **4**.
- `cypher-mask-lowering-v1` R-6 chain + T-1 popcount: hop₁ dst = {2,3,4,5}, hop₂ dst = {3,4,5} ⇒ **3**.
- `RETURN count(DISTINCT c)` ⇒ **3**. The plan's lowering answers the DISTINCT question and labels it `count(*)`.

The same fixture, `-[:KNOWS*1..2]->` from `a.id = 1`: paths 1→2, 1→3, 1→2→3, 1→3→4 ⇒ 4; distinct endpoints {2,3,4} ⇒ 3.

## §1 Frozen decisions (not re-litigated)

- F1. DataFusion is in grace: maintained, not extended (`CLAUDE.md` § ⊘ OPERATOR RULING 2026-09-05).
- F2. Mask programs are a lowering target; nothing is minted (`cypher-mask-lowering-v1.md` §5.2).
- F3. One-population, tail and transpose laws hold (`cypher-mask-lowering-v1.md` §5.3).
- F4. Absence is a zero fallback, not a NULL (`mask-risc/src/lib.rs:44-47`).
- F5. Builds use `CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_INCREMENTAL=0`.

## §2 Input inventory

| site | claim it makes | status under §0 |
|---|---|---|
| plan T-1 (`:359`) | `count(*)` = popcount of the final mask | wrong whenever the pattern has a hop |
| plan T-2 (`:360`) | `count(n)` = `count(*)` because no NULL | right about NULL; inherits T-1's hop defect |
| plan R-6 (`:339`) | fixed k-hop = `dst₁` becomes `src₂` | right for the SUPPORT; drops path count |
| plan R-7/R-8 (`:340-341`) | var-length = delta-frontier fixpoint | computes reachability; Cypher `*` enumerates trails |
| plan T-11 (`:369`) | `DISTINCT n` is free | right only when n is the last frontier variable |
| plan §4.1 (`:398-415`) | multiplicity is `[GRACE]` | right, but T-1/R-6 contradict it silently |
| plan §4.6 (`:484-497`) | "how many rows? yes (popcount)" | inverted: popcount counts distinct nodes, not binding rows |
| `logical_plan.rs:19-122` | `Expand`, `VariableLengthExpand`, `Project`, `Distinct`, `Limit` | the consumer is visible in the plan tree |
| `logical_plan.rs:72-91` | DF var-length unrolls and unions | the reference answer |
| `mask-risc/src/ir.rs:265-295` | `Terminal::{Count,Any,All,MaskedSum…}` | all set-semantics terminals; no weighted count |
| `csr_index.rs` (#160) | CSR with stable per-source order | the factorized carrier |
| planner `adjacency/batch.rs` | `AdjacencyBatch` | factorized frontier prior art |

## §3 Resolution

### §3.1 PR 0 — correct the plan (doc only)

Rewrite T-1, T-2, R-6, R-7, R-8, T-11 and §4.6 so each names its consumer:

- A mask answers **which distinct nodes** the last variable can take. Nothing else.
- `count(*)` / `count(n)` after a hop, `sum` over a hop, var-length `count(*)`, `LIMIT` without `ORDER BY` over a bag, and any `RETURN` of two or more variables are `[GRACE]` in v1.
- `count(DISTINCT c)`, `RETURN DISTINCT c`, `EXISTS`, and a hop feeding only a `WHERE` on c stay `[G]`.
- §4.6 gets a row: "how many BINDINGS? — no (popcount counts nodes)".

### §3.2 PR A — the contract, stated before any code

```
Frontier  { support: Mask,                  // Boolean semiring
            mult:    Option<CountLane>,     // counting semiring, u64, checked
            binding: Option<Offsets> }      // parent identity (factorized)

ConsumerSemantics = SetOnly        // needs support
                  | CountBindings  // needs support + mult
                  | Bindings       // needs support + binding ⇒ Unfold
```

Rules:

1. A lowering carries the cheapest `Frontier` its consumer permits.
2. `SetOnly` may use today's mask chain. `CountBindings` and `Bindings` may not.
3. **Unfold** is the only road from a factorized frontier to rows, and it is a named materializer (`materialize…`, O(output) in its doc).
4. Var-length under `CountBindings` stays `[GRACE]`: a counting semiring counts WALKS, Cypher counts TRAILS (relationship uniqueness). They differ on any cycle.
5. Fixed k-hop under `CountBindings` over one relationship type is exact only when no walk reuses an edge. For k=2 that means no self-loop; otherwise `[GRACE]`.
6. `mult` overflow is a refusal, never a wrap.

PR A ships **no** new operator. It ships:

- the classification as a pure function over `LogicalOperator` (`SetOnly` / `CountBindings` / `Bindings`), and
- the proof fixture as a differential test that pins the DataFusion answers (4 / 3 / 4 / 3) and asserts each query's classification.

`CountLane` execution (a CSR SpMV hop) is a later PR, gated on a consumer.

## §4 Non-goals

- No DataFusion extension node, no change to `expand_batch`.
- No counting op in mask-risc's IR in this council.
- No OPTIONAL MATCH (the grammar has none).
- No change to the dense-rowid address model.

## §5 Pre-registered gates

- G1. The fixture test asserts `count(*)`=4, `count(DISTINCT c)`=3, var-length `count(*)`=4, `count(DISTINCT b)`=3 through the existing `CypherQuery` path.
- G2. The classifier returns `SetOnly` for the DISTINCT forms and `CountBindings` for the `count(*)` forms. Disable-verified: force it to `SetOnly` and G2 fails.
- G3. Can-stay-silent: a single-variable `MATCH (n) WHERE … RETURN count(*)` classifies `SetOnly` (no hop, popcount is exact).
- G4. `cargo test -p lance-graph --test <fixture>` and `cargo clippy -p lance-graph -- -D warnings` green, debug 0.
- G5. PR 0's edited rows each cite §0.

## §6 Savant questions

**prior-art** — Q1 Does any EPIPHANIES/knowledge entry already state "mask = support, multiplicity lost"? Q2 Does `AdjacencyBatch` or #163's ListArray already define a factorized frontier we should name instead of `Frontier`? Q3 Is there an existing Unfold/flatten name?

**iron-rule** — Q1 Does `mult: CountLane` violate I-VSA-IDENTITIES or the mask-native no-population rule? Q2 Is a named `materialize…` Unfold consistent with the materialization exception? Q3 Does a classifier over `LogicalOperator` create a second evaluator (plan OQ-10)?

**runtime-archaeologist** — Q1 Does the DF path really return 4 for the 2-hop `count(*)` and 4 for `*1..2` from 1? Q2 Does DF enforce relationship uniqueness in var-length? Q3 Do `Project`/`Distinct`/aggregate nodes expose enough in `LogicalOperator` to classify without the AST? Q4 Where does `count(*)` live in the logical plan?

**cascade-impact** — Q1 Which plan rows and waves (§7 F-H*, F-F*) change under PR 0? Q2 Which other plans cite T-1/R-6/R-7 as settled? Q3 Board files owed.

**creative-explorer** — Q1 Is there a case where the support mask plus a per-node count lane is exact for var-length? Q2 Is ConsumerSemantics really three values? Q3 What does `WITH b, count(*) AS k` (group by an earlier variable) need?
