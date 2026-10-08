# 2026-10-08 — rbac: membership-scoped authorization (additive)

**Status:** TEST-PINNED (`lance-graph-contract` rbac tests; `lance-graph-rbac` `membership_tests`). Nothing existing changed behaviour.

## MEASURED

- `authorize_scoped` intersects the row scopes of every granting role, so one more role can only narrow access: a global role plus a role scoped to org A yields org A, and two roles on different branches yield `ScopeSpec::DENY` (both pinned in `membership_tests` against `authorize_scoped`'s own output).
- `ClassRbac::row_scope(role, class)` has no actor, so one role held at two places ("editor in org A and in org B") cannot be expressed.
- A union of scopes on two branches has no single `ScopeSpec`.

## DECISION

- Contract, additive: `ScopeSpec::covers`; `ScopeSet` (union of incomparable scopes; `insert`, `admits`, `narrow` for attenuation); `Membership { role, scope }`; `ClassRbac::memberships` with a default of `actor_roles` × `row_scope` (unchanged results for every existing impl); `is_access_control_class` (the `0x0B` Auth domain).
- Kernel, additive: `authorize_memberships` → `MembershipDecision` unions the scopes of granting memberships; an unscoped granting membership means unrestricted. Its verdict equals `authorize`'s on the default path (pinned). `authorize_scoped` is unchanged.
- **SCOPE:** prepares membership-bound scope without migrating any consumer. **BASIS:** operator asked for no breaking changes, only the pieces a clean implementation needs.

## OPEN

- No consumer uses `authorize_memberships` yet; `OgarRbac` / `GrantSource` would gain a membership source to use it.
- `is_access_control_class` is a check, not enforcement; no gate consults it yet.
- No data-level row predicate is evaluated (`predicate_key` stays opaque), and authority narrowing is only `ScopeSet::narrow`, with no delegation type.
