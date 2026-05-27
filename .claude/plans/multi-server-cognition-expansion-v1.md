# Multi-Server Cognition — Legacy-Stack Displacement + Expansion Readiness (v1)

> **Status:** PROPOSAL — rationale + expansion-readiness. NOT committed deliverables.
> §2 (displacement) is current-state reasoning; §3 (multi-server) is a design
> hypothesis that follows from the invariants but is **unbuilt**; §4 (determinism)
> is a real, **unproven** prerequisite.
> **Purpose:** Kill the "for multiple servers we need JanusGraph / Cassandra /
> Zitadel" argument, and record the conditional path to multi-server HA/scale-out
> that does **not** require them.
> **Confidence:** HIGH vs JanusGraph + Cassandra; vs Zitadel — displaced by **Ory**
> (Kratos+Hydra) for authN, §5; HIGH that the log-replicated-SoA shape is
> *correct*; LOW that it is *proven* (gated on §4).

---

## 1. The argument being killed

"For multiple servers / HA / scale-out we should adopt a proven distributed
system" → reaches for **JanusGraph** (distributed graph), **Cassandra**
(wide-column), or **Zitadel** (identity, on Cockroach-Raft). Each drags a
heavyweight external service + its own consistency model + a JVM/Go runtime into
a stack whose entire premise is a **single Rust binary** with an in-binary,
zero-copy, **no-serialization** cognitive substrate.

## 2. Component-by-component displacement

| Legacy | Reached for | Covered by | Verdict |
|---|---|---|---|
| **JanusGraph** | distributed graph = Gremlin + Cassandra/HBase + ES/Solr | SurrealDB graph + lance-graph SoA/SIMD + Cypher/Gremlin/SPARQL front-ends on a Raft KV | **Displaced.** Replaces a JVM 3-system assembly with one Rust stack + CP consistency. Its only edge is proven billion-edge cluster scale — not this regime. |
| **Cassandra** | scale-out wide-column store | SurrealDB-on-TiKV (Raft, CP, transactional) | **Displaced on the right axis.** A belief/memory substrate wants CP (read-your-writes, no last-write-wins clobber), not Cassandra's AP/eventual. Cassandra wins only at extreme multi-DC write-availability under partition — not needed. |
| **Zitadel** | identity / OIDC IdP (runs on Cockroach-Raft) | authZ — RBAC `Policy` + `TenantId` Chinese-wall + JWT verify + merkle audit — **already in-stack**. authN/IdP — **Ory** (decided, §5) | **Displaced by Ory, not by Raft.** Raft was the wrong axis: the value is IdP *behavior*, not storage Raft. **Ory Kratos+Hydra** (composable, headless) fills authN; binary stays verification-only. See §5. |
| **TiKV** | — | (it IS the Raft engine — the displacement *mechanism*, not a thing displaced) | Role = the replicated **log** tier (§3). It is "under lance," not "instead of lance" — see §3. |
| **Databend** | — | — | Orthogonal. Cold analytics edge only (export-fed; its Raft replicates only its own metadata). Not in the multi-server path. |

## 3. The multi-server expansion architecture

"Multiple servers" is a replication seam, and a replication seam is a
serialization boundary — which the no-serialization invariant forbids in the
**hot** path. So the seam goes where serialization is already legal (the
durable/egress tier), **never** on the hot SoA.

**Replicate the log, not the SoA:**

- Raft replicates the **episodic / belief-delta log** — append-only, ordered,
  content-addressed; each entry = one committed belief delta.
- Each server keeps its **own local zero-copy SoA**, rebuilt by *applying* the
  log. SoA = the Raft **state machine**; the AriGraph **Witness episodic arc =
  the Raft log**; the Markov belief-chain = the state machine it drives.

This dissolves the apparent "TiKV **or** lance" choice at the KVS layer — a false binary:

- **TiKV/Raft replicates the *log*** (committed belief-deltas; cross-node consensus).
- **lance materializes the *state* locally** per node (zero-copy SoA, rebuilt from applied entries).
- → TiKV-the-log **under** lance-the-state. Both present, different roles.

**Maps onto the existing `surrealdb/core/src/kvs/lance` engine:** the **WAL is the
replication unit** (Raft-replicate the WAL); `memtable` + the lance dataset are
the per-node materialized state followers rebuild. The structure is already
shaped for this.

**Rubikon = the append point.** `commit_gate.rs` is where a candidate belief
becomes committed → that is the Raft append. The 550 ms candidate cloud stays
**node-local, uncommitted, un-replicated** (no consensus cost); only committed
belief-deltas pay the Raft round-trip. Consensus load is bounded to **real
commits**, not every thought-spark. (Libet/Heckhausen admission control = the
gate; the gate = the consensus boundary.)

## 4. Hard prerequisite — deterministic apply (the unproven part)

Same log → same SoA on every node **only if** applying a belief-delta is
byte-deterministic across machines. NARS truth revision is f32 math, and
floating-point **non-associativity** can make two nodes diverge from an
identical log. So multi-server convergence **depends on** the determinism
discipline the repo already prizes — `thinking-engine::reencode_safety` (x256
byte-determinism) + the D-SDR-26 determinism rule. This is the **precondition**,
not a side-quest.

- Log entries are the **binary `canonical_bytes` / `CausalEdge64`** form — never
  JSON. Keeps the no-serialization invariant intact: hot path zero-copy,
  replication log in canonical binary.
- **CONJECTURE (NOT PROVEN):** the full NARS/truth apply path is byte-deterministic
  across architectures. Per the `bf16-hhtl-terrain` probe-queue discipline, the
  next deliverable here is the **determinism probe**, not more synthesis.

## 5. The one genuinely separate decision — authN — **DECIDED: Ory**

The storage of identity isn't the gap (it's just another tenant-partitioned
table); the gap is IdP *behavior* (issue tokens, login, MFA, SSO/OIDC). This was
always **orthogonal to Raft**, and the decision is **Ory** — the composable,
headless option, deliberately *not* Zitadel's all-in-one:

- **Ory Hydra** = OAuth2 / OIDC server → token **issuance** (the piece the Raft
  stack structurally lacks).
- **Ory Kratos** = identity lifecycle → login, registration, MFA, recovery, user store.
- **Ory Keto** = **not adopted** — authZ stays in-stack (`Policy` RBAC + `TenantId`
  Chinese-wall + merkle audit); Keto's Zanzibar model would duplicate it.

**Integration:** Ory runs the IdP (its own Postgres/Cockroach persister); the
binary stays a **verification-only consumer** — `auth.rs` already verifies the
JWTs Hydra issues. authN therefore lives entirely **at the request boundary,
never in the belief-delta hot loop**, and touches **none** of the §3 consensus
layer. The legacy "Zitadel-or-build-it" fork collapses: Ory (Kratos+Hydra) fills
authN, the in-stack authZ is untouched.

**Open (low-risk default = standalone):** Ory doesn't natively persist to
SurrealDB, so its store stays standalone Postgres/Cockroach rather than riding
the shared replicated log. Revisit only if a SurrealDB persister is ever worth writing.

## 6. Phasing (only if/when multi-server is actually needed)

Single-node (Stefan-style) needs **none** of this — local lance, no Raft, no
TiKV. The expansion is a readiness path, sequenced:

1. **Determinism proof** (§4) — byte-deterministic NARS apply probe. Gate for everything below.
2. **WAL-as-Raft-log** — promote the `kvs/lance` WAL to a Raft-replicated log; followers apply to local lance + memtable.
3. **Commit-gate = append point** — wire `commit_gate.rs` to the Raft append; pre-Rubikon stays node-local.
4. **Follower-read semantics** — local SoA may lag the leader; route CP reads to leader or wait-for-apply.
5. **authN decision** (§5) — independent; only if external auth is needed.

## Cross-references

- `surrealdb/core/src/kvs/lance/{wal.rs, memtable.rs, flusher.rs, commit_gate.rs, schema.rs}` — the LSM-on-lance engine this rides (schema is opaque-KV today; typed columns deferred).
- `.claude/plans/cognitive-substrate-convergence-v1.md` — D-CSV-6/7 (WitnessCorpus + MailboxSoA W-slot); the belief-delta log is the episodic arc those build.
- `.claude/board/TECH_DEBT.md` — TD-ARIGRAPH-EPISODIC-FIDELITY-1 (the episodic arc / Witness), TD-JSON-SERIALIZATION-SITES-1 (canonical_bytes vs JSON on the log).
- `thinking-engine::reencode_safety` + D-SDR-26 — the determinism prerequisite (§4).
- `CLAUDE.md` — AGI-as-glove (the SoA *is* the substrate), I-SUBSTRATE-MARKOV (the Markov state machine), the no-serialization invariant.

---

*Authored 2026-05-27. PROPOSAL — §2 displacement is current-state; §3 multi-server expansion is gated on the §4 determinism probe.*
