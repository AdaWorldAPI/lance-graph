# cypher-kanban-ast-unification-v1 — Cypher IS the kanban-board AST; the GUID-keyed substrate IS the graph; four subsystems collapse to one

> **Status:** PLAN (pre-5+3-council). Captures the unification from
> `E-GUID-IS-THE-GRAPH` + `E-CYPHER-IS-THE-KANBAN-AST` before it dilutes,
> and sequences the q2-rewire wiring. **datafusion stays the default query
> engine; the SurrealQL lowering rides the `lite-unified` gate (#540,
> default-OFF) — process, not switch.**

## The thesis (one AST, four sides)

A kanban board is a graph. The GUID is the key. Therefore Cypher — the AST for
graph patterns — is the board's native query/mutation language, and the planner
already shares **one IR/AST** across all four sides:

| Side | Surface | Already shipped |
|---|---|---|
| **Query language** | Cypher / Gremlin / SPARQL / GQL → one IR | planner `strategy/{cypher,gremlin,sparql,gql}_parse.rs` |
| **Adapter / egress** | SurrealQL AST (`DEFINE`/traversal) | `ExecTarget::SurrealQl` + `ogar-adapter-surrealql`; odoo ontology traversal = existence proof |
| **Planner layer** | thinking-styles + MUL over the IR | planner `thinking/` (12 styles, NARS, sigma chain), `mul/` |
| **Board** | kanban phases the cards move through | `kanban::{KanbanColumn, ExecTarget}` |

Board ⇄ graph mapping (no new types):
- **card → GUID node** (`NodeGuid` key; `classid→ClassView`, `local_key`=family++identity)
- **column / phase → state** (a `Column` node via `:IN` edge, or a `classid`/property)
- **move → edge rewrite** (`(c)-[:IN]->(Doing)` ⇒ `(c)-[:IN]->(Done)`)
- **dependency / blocks → edge**; **WIP limit → a Cypher `count` guard**; **swimlane → basin/label**
- **`KanbanColumn::can_transition_to` → the graph schema** (which edges are legal)

The `KanbanColumn` cognitive cycle is the *mailbox* instantiation; an odoo project
board / woa work-order board / q2 case board are *domain* instantiations —
**same AST, same substrate, different `classid` + edge schema.**

## What's already true (no work)

- **The substrate is the graph** (`E-GUID-IS-THE-GRAPH`): node = `NodeGuid` key,
  edge = `EdgeBlock` slot (byte → neighbor `local_key`) or `CausalEdge64`,
  traversal = prefix-route + slot-deref, zero value decode.
- **Cypher parses** (core nom parser, 44 tests: node + relationship patterns,
  var-length `*1..2`, WHERE/RETURN/etc).
- **The polyglot front-ends → one IR**; **thinking-styles + MUL** plan over it;
  **`ExecTarget::SurrealQl`** is the egress.
- **odoo ontology traversal already runs through the SurrealQL AST adapter** —
  the existence proof that Cypher/ontology-over-SurrealQL works.

## The gap (what this plan wires) — additive, layout-preserving

### Inc 0 — `Backend::MailboxSoa` router variant
`graph_router::Backend` gains a `MailboxSoa` variant whose scan:
- resolves **key → row** via `NodeGuid::{from_guid_prefix, local_key}` (no guid
  value-column — the key IS the address),
- follows edges by **`EdgeBlock` slot deref** (byte → neighbor `local_key`) and/or
  `MailboxSoaView::edges_raw` (`CausalEdge64`),
- a Cypher `MATCH (n:Label)` lowers to a **classid prefix route**; `(a)-[r]->(b)`
  to an **edge-slot deref** — both zero-value-decode.
Additive to the existing 3-backend router (DataFusion / Blasgraph / Palette); no
`NodeRow`/stride/`ENVELOPE_LAYOUT_VERSION` change.

### Inc 1 — Cypher → SurrealQL lowering behind `lite-unified` (default-OFF)
The board's mutations/queries lower to SurrealQL (`ExecTarget::SurrealQl`) **only
under `lite-unified`**; datafusion stays the default. This is the egress side of
the AST, gated per the #540 process-not-switch invariant.

### Inc 2 — kanban-board-as-Cypher over the GUID substrate
Express board ops as Cypher patterns dispatched through the planner with
thinking-styles/MUL: a move = a Cypher edge-rewrite → IR → style/MUL plan →
`ExecTarget`; a query (cards-in-column / blocked / WIP-count) = a Cypher `MATCH`.
The transition-legality (`KanbanColumn::can_transition_to`) is the schema guard.

### Inc 3 — q2 consumer wiring (the rewire)
q2 (Palantir-Gotham/neo4j-aspiring) cases = a **domain `classid` + edge schema**
over the same substrate + AST. q2 depends on lance-graph as the hot path (NOT a
substitute — `.claude/rules/architectural-compliance.md`); its board/case
traversal is Cypher over `Backend::MailboxSoa`.

## Falsification gates

- **F1 (router):** a Cypher `MATCH (n:Label)` over `Backend::MailboxSoa` returns
  the same node set as the DataFusion backend on the same data (backend parity).
- **F2 (zero-decode):** a 1-hop traversal touches only key + `EdgeBlock` bytes,
  never the 480 B value slab (assert via a value-access counter / borrow check).
- **F3 (board legality):** an illegal kanban move (`!can_transition_to`) is
  rejected at plan time, not after the edge rewrite.
- **F4 (gate isolation):** with `lite-unified` OFF, behaviour is byte-identical to
  today (datafusion path); the SurrealQL lowering is unreachable.
- **F5 (no new layer):** the unification adds ONE router variant + ONE lowering
  path behind a flag — no new node/edge type, no second object model, no
  kanban-as-traversal type.

## Scope guards (truth-architect)

- "Cypher is the kanban AST" = Cypher is the **surface** that lowers to the shared
  IR/SurrealQL AST — NOT a claim the nom parser emits SurrealQL today (it lowers
  to DataFusion; SurrealQL lowering is Inc 1 behind `lite-unified`).
- **datafusion is NOT deprecated** (#540). The router gains a backend; it does not
  lose one.
- No measured-perf claim without a bench (F1 is a correctness/parity gate, not a
  speed gate).
- The mailbox-`KanbanColumn` lifecycle and a domain board are the SAME structure;
  do NOT fork a second kanban type per domain — vary `classid` + edge schema.

## Cross-refs
`E-GUID-IS-THE-GRAPH`, `E-CYPHER-IS-THE-KANBAN-AST`, `E-AR-DO-WIRING`
(ontology→DO consumers), `lite-unified-surrealql-lance-v1` (#540 gate),
`canonical_node::{NodeGuid, EdgeBlock, NodeRow}`, `soa_view::MailboxSoaView`,
`kanban::{KanbanColumn, ExecTarget}`, planner `thinking/` +
`strategy/cypher_parse.rs`, `graph_router::Backend`,
`.claude/rules/architectural-compliance.md` (q2 must consume, not substitute).
