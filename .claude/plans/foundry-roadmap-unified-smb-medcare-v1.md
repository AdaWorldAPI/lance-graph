# Foundry Roadmap — Unified SMB + MedCare Consumer Plan v1

> **Status:** Active
> **Author:** main thread (Opus 4.7 1M), session 2026-04-28
> **Supersedes:** `foundry-consumer-parity-v1.md` (PR #276)
> **Mirrored to:** `lance-graph`, `smb-office-rs`, `medcare-rs`
> **Cross-ref:** PR #276 (UNKNOWN resolutions); `smb-office-rs/docs/foundry-parity-checklist.md`; `lance-graph/FormatBestPractices.md`

---

## §1 State of the union (2026-04-28)

| Repo | Stage | Tests | Notes |
|---|---|---:|---|
| `lance-graph` | Tier-0 contract complete; Tier-1 LF-1/4/5/6/7/8 merged; PR #276 merged | — | DM-7 unblocked |
| `smb-office-rs` | F0–F7 complete (PR #1 + #2); blocked on LF-3 for F8 | 123 ✅ | Template for medcare |
| `medcare-rs` | F0 only (axum + MySQL); architecture review done | — | Session died on context wall |

---

## §2 Critical path — one PR unblocks both consumers

**LF-3 / DM-7** (`lance-graph-callcenter[auth]`): `JwtMiddleware + ActorContext + RlsRewriter::rewrite(LogicalPlan, &ActorContext) -> LogicalPlan`

- Unblocks SMB F8 (RBAC live)
- Preconditions medcare PostgREST RLS per `praxis_id`
- Was gated on UNKNOWN-3 (pgwire?) + UNKNOWN-4 (actor_id type?) — both resolved by PR #276
- Estimate: ~300 LOC + tests; independently mergeable

---

## §3 Build sequence

| # | PR | Repo | Unblocks |
|---|---|---|---|
| 1 | **LF-3 / DM-7** — RLS rewriter | lance-graph | SMB F8 + medcare RLS |
| 2 | **LF-90** — audit log (wire `AuditEntry`) | lance-graph | SOC2/GDPR both consumers |
| 3 | **`LanceMembrane::with_registry()`** — DatasetRegistry + RlsRule builder | lance-graph | medcare-membrane skeleton |
| 4 | **DM-8** — PostgRestHandler (3-chunk split: parser / LogicalPlan builder / axum routing) | lance-graph | 87+ medcare routes; 50+ SMB routes |
| 5 | **`StepDomain::Medcare`** — mirrors `StepDomain::Smb` | lance-graph | medcare-bridge dispatch |

PRs 1–3 + 5 are independent. PR 4 depends on PR 1.

---

## §4 Consumer scaffolding (mirrors smb-office-rs)

| smb-office-rs (template) | medcare-rs (mirror) | Delta |
|---|---|---|
| `crates/smb-bridge` | `crates/medcare-bridge` | step prefix `medcare.*`, ~30-entity ACCEPTED list |
| `crates/smb-ontology` | `crates/medcare-ontology` | clinical schemas + German clinical predicates |
| `smb_accountant / smb_auditor / smb_customer / smb_debtor` | `medcare_doctor / medcare_nurse / medcare_admin / medcare_patient` | clinical role taxonomy |
| `MongoConnector` | `MySqlConnector` | reconciler witness; MySQL never retired |

---

## §5 Data model — right representation at each scale

The natural representation for both consumers is **SPO triples + typed ontology graph**, not flat bitpacked fingerprints. Patient-data and SMB-customer-data volumes are small enough to use the rich graph shape natively; binary compression is for scale neither consumer approaches within one tenant.

| Scale | Correct shape | Incorrect shape |
|---|---|---|
| Per-tenant (1k–10k patients / 1k–50k customers) | SPO triples + ontology + lance-graph nodes/edges + Vsa16kF32 hot-path | Binary16K-only (loses graph structure; ρ≈0.1–0.3 degrades similarity) |
| Cross-tenant cohort (1M+ aggregated) | CAM-PQ 32-byte codes + Vsa16kF32 rerank top-K | direct Vsa16kF32 scan (640 GB at 10K tenants) |
| Persisted graph edges | bgz palette edge (3 bytes/edge) | Vsa16kF32 per edge (64 GB at 1M edges) |
| Live similarity / hypothesis ranking | Vsa16kF32 (N=2–16, fits L3) | i8 (precision floor 10⁻², fails epiphany margin) |

Reference: `lance-graph/FormatBestPractices.md` §5 per-workload decisions.

**For medcare specifically:**

```
Patient (node)
  ├─[hat_diagnose]──► Diagnose (node, ICD-10)
  ├─[hat_laborwert]─► Laborwert (node, LOINC)
  ├─[verschrieben]──► Medikament (node, ATC)
  └─[gehört_zu]─────► Praxis (node, tenant boundary)
```

Each edge is an SPO triple persisted as a bgz palette edge (3 bytes). Similarity search via Vsa16kF32 at query time. CAM-PQ only enters when aggregating across tenants for anonymised cohort analytics.

**For SMB analogously:**

```
Kunde (node)
  ├─[hat_rechnung]──► Rechnung (node)
  ├─[hat_mahnung]───► Mahnung (node)
  ├─[gebucht_auf]───► FiBu (node, SKR04)
  └─[gehört_zu]─────► Mandant (node, tenant boundary)
```

---

## §6 Stage mapping — medcare phases vs SMB template

| MedCare stage | SMB equivalent | Upstream dep |
|---|---|---|
| F0 (today — axum + MySQL) | pre-F0 | none |
| F1 (membrane skeleton + dual-write) | F4 (smb-bridge) | LF-3, `with_registry` |
| F2 (PostgREST live; `/api/*` parallel) | F5 (smb-ontology) | DM-8 |
| F3 (`/api/*` proxies to PostgREST; MySQL = oracle) | F6 (OrchestrationBridge) | `StepDomain::Medcare` |
| F4 (Phoenix realtime + VSA UDFs) | F7 (integration test) | LF-90, callcenter[realtime] |
| F5 (RBAC: doctor/nurse/admin/patient) | F8 | LF-3 RLS rewriter |

---

## §7 MySQL oracle — permanent, not transitional

MySQL is the parity witness at every phase. The reconciler compares SQL results from MySQL and Lance. No phase advances with nonzero drift. MySQL is never retired per explicit user directive. This is a permanent architectural feature, not a migration artifact.

---

## §8 Shared contract principle

One contract shape (`lance-graph-contract::{ontology, property, repository, rbac}`), two domain instances. The cognitive stack (`BindSpace` SoA + FreeEnergy + CausalEdge64 + NARS) is domain-agnostic — it processes SPO fingerprints whether they encode a Mahnung or a Prescription. SMB and MedCare plug into the same socket with different domain configurations.

## §9 Correction from PR #276

PR #276 framed the data layer as "Binary16K fingerprint per row". That framing was OSINT-scale defaults (10M docs) applied to per-tenant scale (10k rows). The correct shape per `FormatBestPractices.md` is SPO+graph+Vsa16kF32 hot-path at this scale, with Binary16K reserved for cold storage and CAM-PQ reserved for cross-tenant aggregation. This v1 plan replaces the Binary16K-centric framing of PR #276's plan.
