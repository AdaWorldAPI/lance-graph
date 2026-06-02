# Orchestration boundary — the loop is infra; lance-graph = contract + resolver (v1, 2026-06-02)

## READ BY: truth-architect, integration-lead, anyone touching the cognitive/orchestration loop

> **STATUS: DRAFT — pending full-surface council (5-research + 3-brutal) verification.**
> Born from a *reverted duplication* (`d5f5aa6`): this session built a kanban-wire + a
> mailbox-as-actor that already exist as infra. This doc fixes the boundary so it doesn't recur.
> Every attribution below is **to be verified by the council against the actual code across
> surreal / ractor / Lance** — not asserted from lance-graph alone (that scope error caused the
> duplication).

## Thesis

The orchestration loop — *resolved understanding → committed action* (version → kanban → actor) —
is **infrastructure already standing across surreal + ractor + Lance + the contract.**
**lance-graph builds none of it.** lance-graph owns exactly two things: the **contract** (the shared
types the others consume) and the **NARS resolver** (`route_against`) — the one cognitive operation
none of the infra does.

## The boundary (owners) — council verifies each attribution + live/scaffold

| layer | owner | grounding (to verify) | status |
|---|---|---|---|
| store (versioned data + MVCC clock) | **Lance** | `surrealdb/core/src/kvs/lance/` | ? |
| transparent view over Lance | **surreal `kvs/lance`** (surrealkv/kvs over the Lance backend) | `kv-lance = [dep:lance, lancedb, arrow-*]` (`core/Cargo.toml:27`) | ? |
| trigger + kanban | **SurrealDB** LIVE → `Notification` on the version | `surrealdb/types/src/notification.rs`, `LIVE_QUERY` | ? |
| actor + mailbox + lifecycle | **ractor** | the ractor fork = the `Actor`/mailbox/supervision framework | ? |
| shared types (SOT) | **`lance-graph-contract`** | surreal-core deps it (`lance-graph` feature, `Cargo.toml:155`) | ? |
| the NARS resolver | **lance-graph** (`route_against`, causal-edge) | `causal-edge/src/syllogism.rs` (6d2b121) | live (6 tests) |

## The transparent view (the key elegance)

surrealkv/kvs over `kvs/lance` (the `kv-lance` Lance backend) = **SurrealDB *is* a transparent view
of Lance** — same data, no separate cold store, **no sync/transcode**. This obviates the callcenter
"parallelbetrieb reconciler" bandaid (R4 of the prior council). The trigger is free: LIVE on the
transparent view fires on the Lance version change → the kanban update. *"Can't get cheaper AND
better-timed than that."*

## The lesson (why this doc exists)

This session built — then reverted (`d5f5aa6`) — a `rubicon_transition` kanban-wire + a
`MailboxSoA.phase`/`advance` actor-step. Both **duplicated infra**: the kanban transition is
SurrealDB's (LIVE), the actor lifecycle is ractor's, and the store is one transparent Lance view.
The cause: a 5+3 *build*-council **scoped to lance-graph's crates only** → it rigorously found "the
wire is absent *here*" and I mis-read that as "build it here." **A rigorous council inside the wrong
fence gives false confidence.**

> **RULE:** verify the FULL surface (surreal + ractor + Lance) before scoping a build.
> **"Absent in crate X" ≠ "build it in crate X."** The thing may already live next door.

## Open questions (the council answers these)

1. Verify each owner-attribution above against the real code (surreal / ractor / Lance), with
   honest **live-vs-scaffold** marking. Which layers are wired, which are scaffold/feature-gated?
2. Does lance-graph own anything **beyond** the contract + `route_against` (the codecs? the planner?
   deepnsm? the resolver feeding surreal's view) — or is that the complete cognitive contribution?
3. Is the loop **fully assembled** (store → view → trigger → actor), or are there real gaps — and
   whose are they (surreal / ractor), confirmed **not** lance-graph's to fill?
