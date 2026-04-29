DRAFT — pending review (2026-04-28)

# Unified SMB-Office + MedCare Foundry Parity Roadmap

Internal coordination document. Not for client distribution.
Owner of the document: Track A (Foundry / infrastructure).
Audience: lance-graph maintainers, smb-office-rs maintainers, medcare-rs maintainers.

---

## 1. Status snapshot

- **smb-office-rs**: F0 through F7 are landed. 123 tests passing. PR #1 and PR #2 are merged.
  The SMB substrate is functionally complete for the F7 envelope (oracle parity, RBAC scaffolding,
  membrane bring-up, write path). F8 is queued behind a single lance-graph dependency.
- **medcare-rs**: RBAC layer is pending. The medcare clinic data model exists, but the row-level
  authorization rewrite has not been wired. MedCare cannot ship a multi-tenant build until that
  rewrite exists upstream.
- **lance-graph**: LF-3 / DM-7 (RLS rewriter as a DataFusion `OptimizerRule`) is the single
  critical-path blocker. Both SMB F8 and MedCare RBAC are waiting on the same upstream change.
  No other lance-graph work item currently blocks downstream consumers.

The headline: **one merge in lance-graph unblocks two downstream products simultaneously.**
This is the cheapest leverage point on the board for the next two weeks. Treat it as the highest
priority for the Foundry track.

---

## 2. PR sequence

All weeks are stated relative to PR-1 landing. No calendar dates in this section. If PR-1 slips,
the entire chain slips with it; do not re-anchor downstream PRs to dates.

### PR-1 (week 0, current) — LF-3 / DM-7

- **Scope**: implement the row-level-security rewriter as a DataFusion `OptimizerRule`, living in
  `lance-graph-callcenter::rls`.
- **Why this crate**: callcenter already owns the tenant-aware query path; co-locating the rewriter
  avoids a new top-level crate and avoids surfacing tenancy concerns into the core graph crate.
- **Behaviour**: given a tenant context (tenant_id, actor_id, role set), rewrite the logical plan
  to inject predicate filters and column-mask projections before physical planning. Deny-by-default
  on missing tenant context.
- **Unblocks**: SMB F8 (which needs tenant-scoped reads through the membrane) and MedCare RBAC
  (which needs the same machinery for clinic-scoped reads). Two consumers, one rewrite.
- **Exit criteria**: rewriter passes a property-based test that no plan reaches physical
  execution without either a tenant predicate or an explicit `system_context` bypass marker.

### PR-2 (~1 week after PR-1) — LF-90

- **Scope**: append-only audit log of RLS-rewritten queries.
- **Schema**: `tenant_id`, `actor_id`, `statement_hash`, `ts`. No statement text in the hot path;
  text is recoverable via separate statement-cache lookup keyed by hash.
- **Storage**: append-only Lance table. No updates, no deletes from application code.
- **Why immediately after PR-1**: the rewriter is the only place we can guarantee we capture every
  tenant-scoped read. Audit must hook the rewriter, not the executor, or it will miss optimizer
  short-circuits.

### PR-3 (~2 weeks) — `LanceMembrane::with_registry()` builder

- **Scope**: idempotent registration of catalog, RLS rewriter, and audit log on a single
  `LanceMembrane` instance.
- **Why a builder**: today, downstream binaries hand-wire the components in slightly different
  order, and SMB and MedCare have already drifted by one component each. A single builder removes
  that drift.
- **Idempotence requirement**: calling `with_registry()` twice on the same membrane must be a
  no-op, not a panic and not a double-registration. Downstream test harnesses will trigger this.

### PR-4 (~3 weeks) — DM-8 PostgREST handler stub

- **Scope**: HTTP shape stub that mirrors PostgREST's URL grammar (`/table?col=eq.value`,
  `Prefer: return=representation`, etc.) on top of the membrane.
- **Why a stub first**: SMB and MedCare client SDKs are being designed to a PostgREST-compatible
  surface so that an existing PostgREST-aware client can be pointed at us with minimal code change.
  Locking the URL shape early prevents two divergent client SDKs.
- **What this PR does NOT do**: it does not implement the full PostgREST feature set. It pins the
  URL/header contract, returns deterministic errors for unimplemented features, and provides a
  conformance harness.

### PR-5 (~4 weeks) — `StepDomain::Medcare` enum variant

- **Scope**: add a `Medcare` variant to the `StepDomain` enum and wire a minimal end-to-end RBAC
  trace through it.
- **Minimal e2e**: one clinic, one role, one read-modify-write cycle, with the audit log entry
  observable at the end.
- **Why last**: this PR is the first one that has nothing useful to do until PR-1 through PR-4
  are all in. It is the canary that everything below it is connected correctly.

---

## 3. Out of scope this quarter

The following items are explicitly deferred. If anyone in standup proposes pulling them forward,
push back and point at this list.

- **RaBitQ tuning**. The current RaBitQ posture is acceptable for SMB workloads. Re-tuning
  belongs to the vector-search track, not the Foundry track.
- **lancedb versioning UI**. Versioning is exposed via the API; a UI is a separate product
  decision and not on the critical path for either SMB F8 or MedCare RBAC.
- **F4-and-above dataflow features** for either SMB or MedCare. F4 in MedCare's roadmap depends on
  PR-1 through PR-5 landing. Discussing F4 details now is premature.
- **Multi-tenant federation**. Cross-tenant reads, federated audit, cross-clinic queries. Out of
  scope. The current rewriter is single-tenant-context per request, by design.

---

## 4. Cross-repo coordination

| PR    | lance-graph artifact                             | smb-office-rs unblocks                        | medcare-rs unblocks                       |
|-------|--------------------------------------------------|-----------------------------------------------|-------------------------------------------|
| PR-1  | `callcenter::rls::OptimizerRule`                 | F8 tenant-scoped reads                        | RBAC pilot build                          |
| PR-2  | `audit::append` + Lance audit table              | F8 audit-grade reads                          | RBAC audit trail                          |
| PR-3  | `LanceMembrane::with_registry()`                 | bring-up wiring                               | bring-up wiring                           |
| PR-4  | PostgREST handler stub                           | client SDK contract freeze                    | client SDK contract freeze                |
| PR-5  | `StepDomain::Medcare`                            | (no direct effect)                            | end-to-end RBAC trace lit up              |

Coordination protocol:

- Every PR in this list has a tracking issue in lance-graph and a mirror issue in each downstream
  repo it unblocks. Mirror issues link back; do not duplicate description text.
- When a PR merges, the owner is responsible for closing mirror issues and pinging the downstream
  owners in the standup channel within one working day.

---

## 5. Risk register

Risks are listed in priority order. Each risk has a current mitigation. If a mitigation is
"none yet" then it is an open question and we should not pretend otherwise.

- **Abduction-confidence calibration on RLS**. The rewriter relies on a confidence model to decide
  when to fall back to deny-by-default vs. allow-with-mask. If the confidence model is
  miscalibrated, we will either over-redact (user complaints) or under-redact (security issue).
  *Mitigation*: ship the rewriter with the confidence threshold set conservatively (favour
  over-redaction), and gate any threshold relaxation on a calibration report.
- **Audit-log retention**. Append-only is easy. Retention policy is hard, especially under
  jurisdictions with both "must retain" and "must delete on request" obligations.
  *Mitigation*: PR-2 ships with retention-policy hooks but no default retention beyond
  "infinite". Per-tenant retention is a follow-up item, not a launch blocker, but must be tracked
  before any production-grade rollout.
- **PostgREST shape stability**. PR-4 freezes a contract that two client SDKs will be built
  against. If we get the shape wrong, we will pay for it across two repos.
  *Mitigation*: PR-4 lands as a stub with a written-down conformance harness. Any future shape
  change must update the harness in the same PR. Tag the contract document with a version number
  and treat changes as semver-major.

---

## 6. Owners and escalation

| PR    | Owner            | Reviewer (primary) | Reviewer (secondary)   |
|-------|------------------|--------------------|------------------------|
| PR-1  | A1 (Track A)     | A2 (Track A)       | B-lead (Track B)       |
| PR-2  | A1 (Track A)     | A2 (Track A)       | C-lead (Track C)       |
| PR-3  | A2 (Track A)     | A1 (Track A)       | B-lead (Track B)       |
| PR-4  | B-lead (Track B) | A1 (Track A)       | A2 (Track A)           |
| PR-5  | C-lead (Track C) | A1 (Track A)       | B-lead (Track B)       |

Escalation path on review stalls:

1. If a PR has been awaiting review for more than two working days, the PR owner posts a single
   reminder in the standup channel with the PR link and the day count.
2. If still no review after one further working day, the PR owner pings the Track A lead directly.
3. Track A lead has authority to reassign the reviewer if the original reviewer is blocked or
   unavailable. Reassignment is recorded in the PR thread.
4. No silent reassignment. No skipping review. If the chain breaks, we stop the line and figure
   out why before any further PRs land.

---

## 7. Pointer note

Mirrored as one-paragraph pointer in `smb-office-rs/README` and `medcare-rs/README`. The pointer
in those READMEs links here and gives just enough context for a downstream contributor to know
why their roadmap is gated. Keep the pointer short. Detail lives in this document.

---

## 8. Document lifecycle

- This document is `DRAFT` until reviewed by the Track A lead and at least one downstream owner
  (smb-office-rs or medcare-rs).
- After review, the header becomes `ACTIVE`.
- After PR-5 lands, the header becomes `HISTORICAL` and a new roadmap document supersedes it.
- We do not edit a `HISTORICAL` document in place; we link forward to the successor.

---

## 9. Anti-goals (explicit)

The following postures are NOT what this roadmap is for. If a meeting is starting to push the
roadmap in any of these directions, the answer is "different document".

- This is not a marketing document. No claims about being "faster than Postgres" or "better than
  Palantir" appear here. Performance claims live in benchmark reports, not roadmaps.
- This is not a feature wishlist. Every entry is gated on a downstream consumer who is currently
  blocked.
- This is not a calendar. Weeks here are relative offsets from PR-1 landing. We do not commit to
  external dates from this document.

---

End of draft.
