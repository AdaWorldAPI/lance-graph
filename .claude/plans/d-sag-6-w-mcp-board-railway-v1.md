# D-SAG-6-W: MCP read surface, /board UI, Railway plan (W2-W4) v1

- **Date:** 2026-10-27
- **Status:** PROPOSAL / docs only. Covers W2-W4 of D-SAG-6-W (sibling doc `d-sag-6-w-dogfood-mcp-workbench-v1.md`). Deployment is NOT authorized by this document.
- **Depends on:** W1 lab crate `crates/sag6-workbench-probe`. Every function named below exists there and is covered by `tests/falsifiers.rs` (15 tests). Fields marked **W2** do not exist in W1 and need the sync job's richer snapshots.
- **Sources:** design-notes facts only (Linear read 2026-10-27; lance-graph main `aa569bc`).

## 0. Scope and non-goals

| In scope | Out of scope |
|---|---|
| Seven read-only query tools over a snapshot of project-graph records | Executing methods, ActionDefs or any write path |
| A plain server-rendered `/board` over the same seven functions | Any client-side graph engine |
| A deployment proposal for an isolated Railway service | Touching `adarail_mcp` or its secrets |

Rules that hold for every tool:

- No tool executes methods, reads the filesystem, modifies code, or mutates issues (GitHub or Linear).
- MCP tool annotations on all seven: `readOnlyHint=true`, `destructiveHint=false`, `idempotentHint=true`, `openWorldHint=false`.
- Inputs are bounded: ids are strings of at most 128 bytes; every list has a `limit` (default 25, max 100); unknown fields are rejected.
- Answer kinds (mirrors dir-sim `RoleEvidence`; never a bare bool):

| Kind | Meaning |
|---|---|
| WITNESSED | A record in the sealed frame supports the answer; the witness is returned. |
| NOT_HELD | Absence, returned ONLY when coverage of the queried source set is complete. |
| UNDETERMINED | Coverage is incomplete; the unread sources are listed in `unread[]`. |
| REFUSED | The operation is not admitted (reason and issue ids returned). |
| UNSUPPORTED | The operation needs a capability that does not exist (for example an unminted predicate binding). |

- Multi-hop traversal is REFUSED, citing EXO-140 (`Filter::Semijoin` unclassified under ADMISSION B; operator decision) and ISS-GATHER-HOP-1616 (`OutgoingHop` quarantined behind `quarantined-gather-hop`).
- Method binding is refused for every predicate. There is no OGAR GitHub port and no Linear port; PullRequest and SourceSymbol have no OGAR concept. Every `Predicate` has `class_id: None`; no ClassId is invented.
- An identity that no snapshot mentions is never NOT_HELD: absence from what was read is not evidence that it does not exist upstream.

### 0.1 Common envelope

```json
{
  "answer": "WITNESSED | NOT_HELD | UNDETERMINED | REFUSED | UNSUPPORTED",
  "frame": "string (opaque server handle of the seal the answer was computed under)",
  "complete": "bool (WITNESSED only: the edge set was read in full)",
  "witnesses": [ { "id": { "system": "linear|github|claude_session", "key": "string", "revision": "string|null" },
                   "snapshot": { "system": "string", "object": "string", "revision": "string", "fetched_at": "string" },
                   "locator": "https://...", "grade": "observed_source | ... | hypothesis" } ],
  "unread": [ { "subject": "GlobalId", "predicate": "string", "direction": "forward|inverse" } ],
  "refusal": { "reason": "frame:<ForeignFrame|SourceChanged|MapChanged|CoordinatesChanged|FrameResealed> | multi_hop_not_admitted | predicate_unbound | executor", "cites": ["string"] }
}
```

Each field is a direct rendering of W1's `Answer` / `Witness` / `Reason`. **Frames on the wire.** A W1 `LocalRef` carries a `Frame` that compares by identity of issue, so it cannot be serialized as a number. The W2 server keeps the `LocalRef`s it issued and hands out an opaque `frame` handle; a request naming a handle that is not the current seal is REFUSED with the `FrameStore` reason. A client never sees or sends an ordinal.

## 1. MCP tool specification (read-only)

| Tool | W1 function | Input (all strings ≤ 128 bytes, unknown fields rejected) | Result | Refusal / unknown cases |
|---|---|---|---|---|
| `resolve_identity` | `Workbench::resolve_identity` | `system`, `key`, `revision` (required for revision-scoped keys: symbols, debt rows) | `{ global_id, frame }` | Not in any read snapshot → UNDETERMINED (never NOT_HELD). Revision-scoped key without `revision` → REFUSED. |
| `query_relationships` | `Workbench::query_relationships` | `subject`, `predicate` (one of `blocks`, `related_to`, `mentions_symbol`, `tracked_by_debt_row`, `cites_pr`, `authored_in_session`), `direction` (`forward`/`inverse`), `hops` (const 1), `limit` ≤ 100 | `witnesses[]`, `complete` | `hops > 1` → REFUSED `multi_hop_not_admitted` citing EXO-140 + ISS-GATHER-HOP-1616. Empty + complete → NOT_HELD. Empty + incomplete → UNDETERMINED with `unread`. Stale frame → REFUSED. |
| `trace_provenance` | `Workbench::trace_provenance` | `id` | every edge touching `id`: `witnesses[]` with snapshot, locator, grade | No edge → UNDETERMINED. `DurableRef` / `ThoughtCommitId` lookups → UNSUPPORTED (types do not exist; EXO-148, D-RST-1). |
| `inspect_predicate` | `Workbench::inspect_predicate`, `Workbench::prepare_method` | `key`, `bind_method` (bool) | `{ key, symmetric, inverse, class_id: null, convergence }`; e.g. `blocks` → `convergence: "ogar_vocab::class_ids::PROJECT_RELATION"` (a candidate, not a binding) | `bind_method=true` → REFUSED `predicate_unbound`, for every predicate and every classid named. Unknown key → UNSUPPORTED. |
| `explain_blocker` | `Workbench::explain_blocker` | `id` | `{ blocked_by: Answer, debt_rows: Answer, transitive: Answer }` | `transitive` is always REFUSED `multi_hop_not_admitted`. **W2:** a blocker's status and labels (e.g. "CC: Operator Decision") once the sync job stores them; W1 records carry no labels. |
| `compare_snapshots` | `Workbench::compare_snapshots` | `a`, `b` (retained snapshot ids) | `{ added: Record[], removed: Record[] }` | Either snapshot not retained → UNDETERMINED. This is a set difference of records; it does not claim temporal-version semantics and is not `QueryReference::at`. **W2:** field-level diffs need stored payloads. |
| `list_unverified_claims` | `Workbench::list_unverified_claims` | `limit` ≤ 100, `cursor` | records graded `hypothesis` | **W2:** additional reasons (no oracle cross-check, stale source, detached result per #1792's 1,032 vs 31) need the sync job; W1 lists grade only. |

### 1.1 Worked example: `explain_blocker` on EXO-111

This is what W1 returns on its fixture (`tests/falsifiers.rs::multi_hop_is_refused_not_emulated` asserts the three parts), rendered in the envelope:

```json
{
  "blocked_by": {
    "answer": "WITNESSED", "complete": true,
    "witnesses": [ { "id": { "system": "linear", "key": "EXO-112", "revision": null },
                     "snapshot": { "system": "linear", "object": "EXO-112",
                                   "revision": "2026-10-23T08:48:48.110Z", "fetched_at": "2026-10-27" },
                     "locator": "https://linear.app/exo-red/issue/EXO-112", "grade": "observed_source" } ]
  },
  "debt_rows": { "answer": "UNDETERMINED",
                 "unread": [ { "subject": "linear:EXO-111", "predicate": "tracked_by_debt_row", "direction": "forward" } ] },
  "transitive": { "answer": "REFUSED",
                  "refusal": { "reason": "multi_hop_not_admitted", "cites": ["EXO-140", "ISS-GATHER-HOP-1616"] } }
}
```

`blocked_by` is complete because EXO-111's relation list was read in full, and it equals Linear's own `blockedBy` (`[EXO-112]`). `debt_rows` is UNDETERMINED, not NOT_HELD: nobody checked whether a board row tracks EXO-111. What blocks EXO-112 would be a second hop, so it is refused rather than answered. (That EXO-112 is an operator decision is true in Linear, but is a W2 field: W1 does not store labels.)

## 2. /board UI plan (minimal)

Principle: one handler table, two transports. Each of the seven functions is registered once; MCP (streamable HTTP) and plain GET JSON for `/board` both call the same handler with the same JSON contract. The frontend contains no business logic and no client-side graph engine. Pages are server-rendered HTML (askama, matching the a2ui-rs / medcare precedent). Every page shows its `source_links` and the snapshot revision and `frame_id` in a footer.

| View | Route | Backing function(s) |
|---|---|---|
| Overview and entry | `GET /board` | `list_unverified_claims` (summary), snapshot header |
| Subject neighborhood | `GET /board/subject/{id}` | `resolve_identity`, `query_relationships` |
| Predicate / ClassId inspection | `GET /board/predicate/{key}` | `inspect_predicate` |
| Technical-debt blockers and ownership | `GET /board/blockers` | `explain_blocker` over the open issue set |
| Provenance trace | `GET /board/trace/{id}` | `trace_provenance` |
| Snapshot comparison | `GET /board/compare?a=&b=` | `compare_snapshots` |
| Unverified / stale queue | `GET /board/unverified` | `list_unverified_claims` |

Notes:

- REFUSED, UNSUPPORTED and UNDETERMINED answers render as labelled states with their reason; they are never rendered as empty results.
- Query parameters are validated against the same input schemas as section 1.
- Pages are GET only; there are no forms and no write endpoints.

## 3. Railway deployment plan (W4, NOT authorized here)

### 3.1 Existing deployment facts

- `mcp.exo.red` is the Railway service `adarail-production` (adarail_mcp): Python 3.11 FastAPI, SSE transport, write-capable tools (`ingest`, `post`, `message`, `upload_image`), roughly thirty secret environment names including `GITHUB_TOKEN` and a writable Upstash credential.
- Its two Railway config files disagree. `railway.toml`: builder `DOCKERFILE`, healthcheck `/health`, timeout 100 s, restart ON_FAILURE (max 3). `railway.json`: builder `NIXPACKS`, healthcheck `/health`, timeout 10 s, same restart policy. Which one Railway honors for that service has not been verified here; the workbench service must not inherit either and should declare its own builder explicitly.
- Precedent for the server side: `surrealdb/surrealdb/mcp/src/service.rs` builds a `StreamableHttpService` (rmcp 1.x) with a `LocalSessionManager` and stateful mode, behind an auth middleware.

### 3.2 Topology

```
                 Railway workspace (existing)
 +---------------------------+     +-----------------------------+
 | sag6-workbench-sync (cron)|     | sag6-workbench (query)      |
 | read-only GitHub + Linear |     | no upstream tokens          |
 | writes immutable snapshots|---->| mounts volume READ-ONLY     |
 +---------------------------+ vol +-----------------------------+
                                          |  /mcp  /board  /healthz  /readyz
                                    workbench.mcp.exo.red (proposed)
```

- A new, isolated Railway service named `sag6-workbench`, built from a Dockerfile in lance-graph. It is not co-hosted with adarail_mcp and shares no environment variables with it.
- Two processes: (a) the query service, read-only, holding no upstream tokens; (b) the sync job (separate service or Railway cron) holding read-only GitHub and Linear credentials and writing immutable snapshot files to a Railway volume. The query service mounts that volume read-only.
- Optional cache: a Redis instance, cache only, never canonical, rebuildable from snapshots. Whether to have one at all is an operator decision.

### 3.3 Domain

Bare `mcp.exo.red` is occupied. Proposed: `workbench.mcp.exo.red`. Serving under `mcp.exo.red/workbench` would require changing adarail_mcp and is not recommended. Operator decision.

### 3.4 Environment variables (names only)

| Name | Process | Purpose |
|---|---|---|
| `WORKBENCH_BEARER_TOKENS` | query | Accepted bearer tokens, stored hashed |
| `WORKBENCH_SNAPSHOT_DIR` | query, sync | Snapshot directory on the volume |
| `PORT` | query | Listen port injected by Railway |
| `RUST_LOG` | both | Log filter |
| `GITHUB_READONLY_TOKEN` | sync only | Fine-grained token: `contents:read`, `pull_requests:read`, `issues:read` on named repositories |
| `LINEAR_READONLY_API_KEY` | sync only | Read-only Linear API key |
| `WORKBENCH_SYNC_REPOS` | sync only | Named repositories to sync |
| `WORKBENCH_SYNC_LINEAR_ISSUES` | sync only | Named Linear issues or filters to sync |
| `WORKBENCH_REDIS_URL` | query, optional | Cache only |

No variable is shared with adarail_mcp. No secret value appears in this repository, the image, or the snapshots.

### 3.5 Endpoints

| Path | Behaviour |
|---|---|
| `POST/GET /mcp` | MCP streamable HTTP via rmcp `StreamableHttpService`; bearer required |
| `GET /healthz` | Liveness only; returns no data |
| `GET /readyz` | Ready only when a snapshot is loaded, the frame is sealed, and snapshot age is under a configured threshold |
| `GET /board/*` | Section 2 pages; bearer required |

### 3.6 Authentication

Bearer token required on `/mcp` and `/board`; scope is read only; per-token rate limit; no anonymous access; no CORS wildcard; TLS terminated by the Railway edge. `/healthz` and `/readyz` expose no snapshot content.

### 3.7 Source sync

- Each sync writes one append-only, immutable snapshot (raw upstream JSON as fetched, plus revision and `fetched_at`) to a temporary path and publishes it by atomic rename.
- On a new snapshot the query service re-seals a new frame; earlier frames then refuse as stale (FrameStore semantics). Replaying the same snapshot is idempotent.
- Snapshot files are the canonical record. Redis, if present, only caches derived reads.

### 3.8 Resource budget (PROPOSED, to be measured)

Small instance; memory bounded in proportion to snapshot size; volume sized for retained snapshots. No measured numbers exist yet; W1 measurement must set them before any request is made to the operator.

### 3.9 Rollback and recovery

- Redeploy the previous image. Snapshots on the volume are untouched by a redeploy.
- Loss of Redis is harmless (cache only).
- Deleting the service removes nothing upstream: the service holds no write credentials and the sync tokens are read-only.

### 3.10 Unresolved prerequisites (operator)

1. Domain (`workbench.mcp.exo.red` or other).
2. Railway workspace, and who owns the new service.
3. Issuance of the read-only GitHub and Linear credentials.
4. Whether a Redis cache is used at all.
5. Linear workspace consent for the read-only API key.

### 3.11 Later and optional

Ghidra may later become a source-analysis adapter feeding symbol records. It is not a prerequisite for W2-W4.
