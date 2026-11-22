# 2026-11-22 — RBAC / authorization boundary: findings recorded, work postponed

**Status:** OPEN, POSTPONED. Documentation only; no code, test or configuration changed.

DECISION (2026-11-22): no authentication or authorization change in this session. Zitadel is the identity provider and it works in production; any glitch there is not affordable.
SCOPE: lance-graph-rbac, lance-graph-contract::rbac, the DT64 lane scope, and the consumer IAM seams in report-rs and zammad-rs.
BASIS: production stability; the findings below are recorded so a later, dedicated session can act on them.
REVISIT WHEN: a session is opened for RBAC work on purpose, with the consumer owners informed.

## What the code says (read at lance-graph `138806961`, report-rs `f2e55c3`, zammad-rs `d81ac8a`)

- `lance-graph-rbac` depends only on `lance-graph-contract`; `lance-graph-dt64` is a dev-dependency for tests (`crates/lance-graph-rbac/Cargo.toml`). No dependency on any identity provider, OIDC, JWT or OGAR crate.
- `lance-graph-contract::rbac` is one file (`rbac.rs`): `ScopeSpec` (`:115`), `ScopeSet`, `Membership`, the `ClassRbac` trait (`:371`).
- `lance-graph-rbac::authorize`: `authorize_scoped(rbac, actor, class, op) -> ScopedDecision` (`authorize.rs:210`); `ScopedDecision { decision, scope, field_mask }` (`:170`) with `visible_fields()` (`:193`).
- `lance-graph-rbac::auth` is the token membrane: `AuthProvider { Store, Zitadel, Zanzibar, OryKeto }` (`auth.rs:65`) carries each provider's claim grammar as data, resolved through the contract's codebook mirror. Zitadel is selectable, not required (`Store` is the provider-agnostic base). Whether claim grammars belong in this crate or downstream is an open design question, not a defect.
- `dr::Scope` is `{ pub id: u64 }` (`lance-graph-dt64/src/dr.rs:523`). Both consumers build their own: report-rs `src/xverweis.rs:219`, zammad-rs `crates/quack-shadow/src/bound.rs:32`, each `Scope { id: 1 }`.
- report-rs reaches RBAC through `lance_graph_rbac::authorize::authorize_memberships` (`src/iam.rs:21`); its `customers_in` (`src/iam.rs:318ff`) widens an unscoped decision only for an explicit all-customers grant.
- `crates/lance-graph-rbac/tests/scoped_lane_binding.rs` (1045 lines, 22 tests, S4 / EXO-243, test-only) already covers: field intersection after Allow, no bindable field on a non-Allow decision, cross-scope refusal, the copied-scope-id finding, a sealed-scope sketch (S4b-2), row-plane sketches refused after a source move or for another decision.

## Recorded decisions this work must respect (from the Linear bridge, EXO-381)

- EXO-206: a DR/DT `Scope` is a source-partition compatibility witness, not authorization; permission changes do not invalidate bindings; the protected read boundary checks current authorization at execution.
- lance-graph does not depend on Zitadel, OIDC, `ogar-auth` or `ogar-rbac`; consumers own authentication.
- EXO-243 (field intersection and row plane) is planned (S4), not verified as implemented. Scope issuance from logins is not decided.

## Postponed items

1. **Revocation falsifier (missing).** No test re-authorizes at execution: grant A, deny B on the same valid binding; revoke A without touching the binding; the next protected read is refused; restore A and it reads again. Suggested home: the existing `scoped_lane_binding.rs` fixture, using `authorize_scoped` with the test-local `Policy` (no new evaluator).
2. **Contradiction to bring to the owner.** The test-local `Issuer` sketch (`scoped_lane_binding.rs:494-550`, test `scope_ids_issued_per_decision_separate_readers_and_follow_a_changed_decision`) issues a scope id per (actor, class, decision), so a changed decision makes `Guards::check` return `ScopeChanged`. EXO-206 says permission changes do not invalidate bindings. It is a sketch, not production; the conflict needs a ruling, not an edit.
3. **Knowledge doc** on the provider-neutral authorization boundary (owners, dependency direction, scope vs authorization), quoting EXO-206.
4. **Where claim grammars live** (`lance-graph-rbac::auth` vs downstream adapters): design question only.
5. **Scope id privacy** (`ISS-DT64-SCOPE-ID-IS-NOT-A-GRANT`): still open; both consumer constants stay as they are.
