# Business-Thinking-OGIT Triangle (append-only knowledge)

> **Status:** D-PARITY-V2-2 (palantir-parity-cascade-v2 §"Business Logic ↔ Thinking-style ↔ OGIT").
> **Authored:** 2026-05-07.
> **READ BY:** crewai-rust (agent dispatch), n8n-rs (workflow routing),
> lance-graph-planner (strategy selection), business-logic
> module authors when proposing new operations.
> **Append-only:** new operations append rows; rows already published
> are never edited. Confidence/correction notes append below the table
> as dated entries.
>
> **Iron rule of this doc:** every row cites a real `*.ttl` under
> `/home/user/OGIT/NTO/<Namespace>/verbs/` AND a real
> `lance_graph_contract::thinking::ThinkingStyle` variant. If a row
> appears that fails either check, it is invalid and must be reverted
> in a follow-up dated entry.

## The triangle (recap from the v2 plan)

```
                 Thinking style
                 (lance_graph_contract::thinking::ThinkingStyle, 36 variants)
                       ▲
                       │  dispatches
                       │
   Business operation ──┼── OGIT verb
   (industry case)     │  (OGIT/NTO/<Namespace>/verbs/<verb>.ttl)
                       │
                       │  describes
                       ▼
                 OGIT entities
                 (OGIT/NTO/<Namespace>/entities/*.ttl, the
                  ogit:from / ogit:to of the verb)
```

Each row carries `(operation_name, thinking_style, ogit_verb,
ogit_entities[])`. The ThinkingStyle is the closest cluster member
from the 36 styles — picked by reading
`crates/lance-graph-contract/src/thinking.rs` and matching the verb's
`dcterms:description` to one of the six clusters
(Analytical / Creative / Empathic / Direct / Exploratory / Meta) and
then to the specific style inside that cluster.

## Routing table — WorkOrder namespace (12 rows, v2 D-PARITY-V2-9 first batch)

The 12 verbs below are the WorkOrder NTO verbs in OGIT (per
`/home/user/OGIT/NTO/WorkOrder/verbs/*.ttl`). Each row maps a
business operation to its dispatched ThinkingStyle and the OGIT
subject/object entities the verb connects.

| # | Operation (business) | ThinkingStyle | Cluster | OGIT verb (TTL) | OGIT entities (`from` → `to`) |
|---|---|---|---|---|---|
| 1 | Issue order | `Pragmatic` | Direct | `OGIT/NTO/WorkOrder/verbs/Issued.ttl` | `ogit.WorkOrder:Customer` → `ogit.WorkOrder:Order` |
| 2 | Assign user to order | `Methodical` | Analytical | `OGIT/NTO/WorkOrder/verbs/Assigned.ttl` | `ogit.WorkOrder:User` → `ogit.WorkOrder:Order` |
| 3 | Add line item | `Systematic` | Analytical | `OGIT/NTO/WorkOrder/verbs/HasPosition.ttl` | `ogit.WorkOrder:Order` → `ogit.WorkOrder:Position` |
| 4 | Record work activity | `Investigative` | Exploratory | `OGIT/NTO/WorkOrder/verbs/HasActivity.ttl` | `ogit.WorkOrder:Order` → `ogit.WorkOrder:Activity` |
| 5 | Attach picture | `Precise` | Analytical | `OGIT/NTO/WorkOrder/verbs/HasPicture.ttl` | `ogit.WorkOrder:Order` → `ogit.WorkOrder:Picture` |
| 6 | Audit history | `Reflective` | Meta | `OGIT/NTO/WorkOrder/verbs/HasHistory.ttl` | `ogit.WorkOrder:Order` → `ogit.WorkOrder:HistoryEntry` |
| 7 | Reference catalogue article | `Analytical` | Analytical | `OGIT/NTO/WorkOrder/verbs/RefersToArticle.ttl` | `ogit.WorkOrder:Position` → `ogit.WorkOrder:Article` |
| 8 | Authenticate portal user | `Critical` | Analytical | `OGIT/NTO/WorkOrder/verbs/AccessesPortal.ttl` | `ogit.WorkOrder:CustomerPortalUser` → `ogit.WorkOrder:Customer` |
| 9 | Custody password vault | `Sovereign` | Meta | `OGIT/NTO/WorkOrder/verbs/OwnsPasswords.ttl` | `ogit.WorkOrder:Customer` → `ogit.WorkOrder:PasswordEntry` |
| 10 | Log billable time | `Concise` | Direct | `OGIT/NTO/WorkOrder/verbs/LogsTime.ttl` | `ogit.WorkOrder:User` → `ogit.WorkOrder:TimeSheet` |
| 11 | Record vehicle trip | `Logical` | Analytical | `OGIT/NTO/WorkOrder/verbs/Drives.ttl` | `ogit.WorkOrder:User` → `ogit.WorkOrder:LogbookEntry` |
| 12 | Partition by tenant | `Frank` | Direct | `OGIT/NTO/WorkOrder/verbs/BelongsToTenant.ttl` | many entity classes → `ogit.WorkOrder:Tenant` (mandatory partition) |

### Per-row rationale

1. **Issue order** — `Pragmatic`: Customer issuing an Order is a
   transactional commit. The Direct cluster's `Pragmatic` variant is
   the cluster member that emphasises "act on what is, not what
   could be"; it dispatches the convergent path that commits the
   transaction with minimal deliberation.
2. **Assign user to order** — `Methodical`: allocating a user to an
   Order is a structured scheduling step. The Analytical cluster's
   `Methodical` variant captures "step-by-step with explicit
   preconditions" — exactly resource assignment shape.
3. **Add line item** — `Systematic`: composing Positions onto an
   Order is decomposition of a whole into parts. `Systematic` (also
   Analytical) is the cluster member that builds wholes from parts.
4. **Record work activity** — `Investigative`: Activities log what
   happened during work execution; the Exploratory cluster's
   `Investigative` variant is "uncovering and recording observed
   facts" — matches activity logging better than Methodical's
   prescriptive shape.
5. **Attach picture** — `Precise`: pictures are visual evidence;
   `Precise` (Analytical) emphasises lossless capture, which is the
   correct disposition for photo documentation.
6. **Audit history** — `Reflective`: HistoryEntry is metacognition
   over the Order's own past — a reading-back of state transitions.
   The Meta cluster's `Reflective` variant fires when the system
   reasons about its own previous decisions.
7. **Reference catalogue article** — `Analytical`: catalog lookups
   are pure joins. The Analytical cluster's namesake `Analytical`
   variant dispatches the standard join cost model.
8. **Authenticate portal user** — `Critical`: every portal login is
   a gating decision; `Critical` (Analytical) is the cluster member
   that fires on yes/no security predicates and surfaces failure
   immediately.
9. **Custody password vault** — `Sovereign`: vault custody is a
   final-authority operation — there is no further appeal once the
   Customer accepts custody. `Sovereign` (Meta) is the cluster
   member that owns the buck-stops-here disposition.
10. **Log billable time** — `Concise`: time entry is a one-shot
    commit with no narrative; the Direct cluster's `Concise` variant
    emphasises minimal payload. Distinct from `Pragmatic` (Issue
    order) by virtue of being information-thin rather than
    action-thin.
11. **Record vehicle trip** — `Logical`: a Fahrtenbuch (logbook)
    entry is a structured sequence (start → route → end) with
    business-vs-private flag. `Logical` (Analytical) is the
    sequential-reasoning cluster member that models "this leads to
    that" routing.
12. **Partition by tenant** — `Frank`: every multi-tenant entity
    must declare its tenant. `Frank` (Direct) is the cluster member
    that asserts assignment without deliberation — exactly the
    shape of mandatory partitioning at row creation.

## Confidence / corrections (append below)

- **2026-05-07 (initial publication):** rows 1-12 are the closest
  cluster picks given the 36-variant taxonomy in
  `lance_graph_contract::thinking`. None has been validated against
  real dispatch traces; treat them as **CONJECTURE** until a
  reasoning probe confirms the cluster choice survives a real
  workload. The cluster (column 4) is the more durable claim; the
  specific variant within the cluster (column 3) is the more
  fragile one.

## Cross-references

- `.claude/plans/palantir-parity-cascade-v2.md` §"Business Logic ↔
  Thinking-style ↔ OGIT (the third triangle)" — the architectural
  framing this doc fills in.
- `.claude/plans/ogit-cascade-supabase-callcenter-v1.md` Pillar 1 —
  OGIT as the universal SPO-G lingua franca; this doc is the
  routing table on top.
- `crates/lance-graph-contract/src/thinking.rs` — the canonical 36
  ThinkingStyle variants in 6 clusters (Analytical 0..5,
  Creative 6..11, Empathic 12..17, Direct 18..23, Exploratory
  24..29, Meta 30..35).
- `OGIT/NTO/WorkOrder/verbs/*.ttl` — the 12 verb TTL files cited
  per row.
- `crewai-rust` agent dispatch + `n8n-rs` workflow routing — the
  consumers that read this routing table.

## Out of scope (this batch)

- **Healthcare / SMB / CallCenter namespaces** — populate in v3
  (after the 12-row WorkOrder pattern is validated).
- **Verb-on-verb composition** (e.g. "Issue order then Assign
  user") — currently covered by `UnifiedStep.depends_on` per
  D-PARITY-V2-6 (LF-12 Pipeline DAG), not by this routing table.
- **Field modulation per operation** — `FieldModulation` already
  travels with `ThinkingStyle`; per-operation overrides would be a
  separate column in this table only if dispatch traces show the
  default modulation is wrong.
