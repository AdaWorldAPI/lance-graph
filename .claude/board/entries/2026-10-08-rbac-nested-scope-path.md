# 2026-10-08 — rbac: nested scope (`ScopePath`)

**Status:** TEST-PINNED (`lance-graph-contract` rbac tests).

## MEASURED

- `ScopeSpec` carried one flat tenant id, so axis 3 could not say "an owner at namespace `a` acts on every database inside `a`": a scope that contains every scope nested below it.

## DECISION

- `contract::rbac::ScopePath`: up to `SCOPE_PATH_DEPTH = 4` interned `u64` segments, outermost first; `contains` is prefix. `ScopeSpec` gains `path` (default `ScopePath::ROOT`, so every existing scope is unchanged) and `admits(&ScopePath)` (address containment, not policy). `intersect` keeps the narrower of two nested paths; two branches intersect to `ScopeSpec::DENY`.
- **SCOPE:** axis 3 of the keystone. **BASIS:** operator chose "extend ScopeSpec" over `predicate_key`.

## OPEN

- A scope bound to a membership (actor, role, position) is not expressible: `ClassRbac::row_scope(role, class)` has no actor, and `RoleId` is `&'static str`.
- `authorize_scoped` still never calls `roles_reaching`; a role hierarchy is expressed as explicit grants per role.
