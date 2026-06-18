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
| **Adapter / egress** | SurrealQL AST (`DEFINE`/traversal) | `ExecTarget::SurrealQl` enum tag only (no lowering yet; `ogar-adapter-surrealql` is NOT a crate — nearest is `surreal_container::SurrealStore`, a `BLOCKED(C)` stub); odoo path = CONJECTURE, see verdict §5 |
| **Planner layer** | thinking-styles + MUL over the IR | planner `thinking/` (12 styles, NARS, sigma chain), `mul/` |
| **Board** | kanban phases the cards move through | `kanban::{KanbanColumn, ExecTarget}` |

Board ⇄ graph mapping (no new types):
- **card → GUID node** (`NodeGuid` key; `classid→ClassView`, `local_key`=family++identity)
- **column / phase → state** (a `Column` node via `:IN` edge, or a `classid`/property)
- **move → edge rewrite** (`(c)-[:IN]->(Doing)` ⇒ `(c)-[:IN]->(Done)`)
- **dependency / blocks → edge**; **WIP limit → a Cypher `count` guard**; **swimlane → basin/label**
- **legal moves → the graph schema:** mailbox-lifecycle legality stays with
  `KanbanColumn::can_transition_to`; domain-board legality resolves via
  `classid → ClassView` (NOT `KanbanColumn` — see verdict §4a)

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
- **odoo ontology work suggests the Cypher/ontology-over-SurrealQL path** — but
  this is **CONJECTURE, not a proof** (verdict §5): what ships is a tagged
  `const ActionDef`(`ExecTarget::SurrealQl`) + `classify_odoo` classification, with
  NO executor arm and NO Cypher→SurrealQL lowering in source yet. Treat as a
  convergent-but-unwired signal, not a running traversal.

## The gap (what this plan wires) — additive, layout-preserving

### Inc 0 — `Backend::MailboxSoa` router variant
`graph_router::Backend` gains a `MailboxSoa` variant whose scan:
- resolves **key → row** via `NiblePath::from_guid_prefix(&guid)` + `NodeGuid::local_key`
  + `MailboxSoaView::row_for_local_key` (no guid value-column — the key IS the address;
  `from_guid_prefix` is on `NiblePath`, not `NodeGuid` — see verdict §2),
- follows edges via the **`classid`-resolved** representation (verdict §4b) —
  `EdgeBlock` slot deref (adjacency, byte → neighbor `local_key`) **XOR**
  `MailboxSoaView::edges_raw` (`CausalEdge64`, causal), selected by the class's
  `EdgeCodecFlavor`/`ReadMode`, never guessed by availability,
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
thinking-styles/MUL: a move = a Cypher edge-rewrite **routed through the DO arm**
(`ActionInvocation` commit gate: def-match → RBAC → state-guard → MUL — NOT a raw
`MATCH…SET`, see verdict §4d); a query (cards-in-column / blocked / WIP-count) =
a Cypher `MATCH`. Legality is `classid → ClassView`-resolved for a domain board;
`KanbanColumn::can_transition_to` governs only the mailbox cognitive cycle.

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
- **F3 (board legality, all guard classes — was F6 in the verdict):** an illegal
  domain-board move (rejected by the `classid → ClassView` schema) **and** any
  DO-arm guard failure (RBAC / state-guard / MUL) are rejected at plan time, not
  after the edge rewrite. (Mailbox-cycle legality is the separate
  `KanbanColumn::can_transition_to` check.)
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

---

## 5+3 Council verdict (2026-06-18) — revisions applied

8/8 reported. Core thesis **SOUND** (convergence: `StepDomain::Kanban` exists; all 4 polyglot parsers already return one `PlanInput`/`QueryFeatures` IR — 3 of 4 sides are genuinely one IR). Revisions:

1. **Headline demoted** (dilution-sentinel): "four sides of one AST" → **"one IR, four *relationships*"** (surface=Cypher / egress=SurrealQL / planner-layer=styles / mutated=board).
2. **CATCH-CRITICAL fixed first** (baton-auditor): `MailboxSoaView` had no key→row resolver — added `row_for_local_key(local_key) -> Option<usize>` default-`None` (deferred binding). `from_guid_prefix` is on **`NiblePath`** (`hhtl.rs:262`), not `NodeGuid` — Inc 0 routes via `NiblePath::from_guid_prefix(&guid)` + `NodeGuid::local_key` + `MailboxSoaView::row_for_local_key`.
3. **Resequenced** (integration-lead): **first shippable = Inc 0 + F1/F2 ALONE**, on the DataFusion-default `ExecTarget::Native` path — zero SurrealQL, zero `lite-unified`, zero q2 coupling. **Inc 1 (Cypher→SurrealQL) is a *dependency-on* `lite-unified-v1`'s OQ-LU-2, NOT a duplicate deliverable** (the two plans were claiming the same lowering).
4. **Three boundaries to pin before Inc 0 lands** (ripple-architect):
   - **(b) edge-representation is `classid`-resolved, not query-guessed:** a relationship-type binds to `EdgeBlock` (adjacency) XOR `CausalEdge64` (causal) via the class's `EdgeCodecFlavor`/`ReadMode` — the router must not pick by availability.
   - **(a) domain-board transition schema ≠ `KanbanColumn`:** a domain board's legal moves resolve via `classid → ClassView`, NOT `KanbanColumn::can_transition_to` (that encodes the *mailbox* Rubicon DAG only). Same graph shape, different transition algebra.
   - **(d) board mutation routes through the DO arm:** a move = `ActionInvocation` through the commit gate (def-match→RBAC→state-guard→MUL), NOT a raw `MATCH…SET` edge-rewrite — otherwise WIP/permission/MUL guards are bypassed. Add F6: an illegal move fails at plan time for ALL guard classes, not just Rubicon legality.
5. **Status downgrades** (truth-architect + archaeologist): the "odoo existence proof" is a tagged `const` + `classify_odoo`, NOT a running traversal → CONJECTURE; "zero-value-decode" (F2) is design-intent until the value-access counter exists; `ogar-adapter-surrealql` is not a crate (it's `surreal_container::SurrealStore`, a `BLOCKED(C)` stub).

**Net first increment (this PR):** the `row_for_local_key` deferred-binding contract method (the named dropped baton) — additive, zero-dep, testable now. `Backend::MailboxSoa` (Inc 0 proper) follows once the three boundaries above are pinned.
