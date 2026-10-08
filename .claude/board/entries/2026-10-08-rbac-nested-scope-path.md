# 2026-10-08 — rbac: nested scope (`ScopePath`) and the `auth_surrealdb` provider

**Status:** TEST-PINNED (`lance-graph-contract` rbac tests; `lance-graph-rbac` auth tests).

## MEASURED

- `ScopeSpec` carried one flat tenant id, so axis 3 could not say "an owner at namespace `a` acts on every database inside `a`". SurrealDB's IAM is exactly that model: `Level::{Root, Namespace, Database, Record}`, and `is_allowed_check` grants View when `resource.level().sublevel_of(actor.level())`, Edit for `Owner` likewise, and Edit for `Editor` only on 12 resource kinds (`surrealdb/core/src/iam/mod.rs`).
- SurrealDB tokens carry the subject in `ID`, roles in `RL`, the namespace in `NS`, then `DB` and `AC` (`iam/token.rs`).

## DECISION

- `contract::rbac::ScopePath`: up to `SCOPE_PATH_DEPTH = 4` interned `u64` segments, outermost first; `contains` is prefix. `ScopeSpec` gains `path` (default `ScopePath::ROOT`, so every existing scope is unchanged) and `admits(&ScopePath)` (address containment, not policy). `intersect` keeps the narrower of two nested paths; two branches intersect to `ScopeSpec::DENY`.
- `lance-graph-rbac::auth::AuthProvider::SurrealDb` = `auth_surrealdb` (`0x0B05`), grammar `ID` / `RL` / `NS`; contract mirror row added. Paired with the OGAR mint (merge together).
- A record-level SurrealDB actor holds no roles but may View; that is mapped as data (an explicit role granted View) in the OGAR adapter, so the kernel keeps "no roles ⇒ deny".
- **SCOPE:** axis 3 of the keystone. **BASIS:** operator chose "extend ScopeSpec" over `predicate_key` or harvest-only.

## OPEN

- Grants match on codebook concepts; OGAR has none for database catalog objects (namespace, table, field, index, …), which SurrealDB's Editor rule depends on.
- `authorize_scoped` still never calls `roles_reaching`; the role hierarchy (Owner ⊇ Editor ⊇ Viewer) is expressed as explicit grants per role.
