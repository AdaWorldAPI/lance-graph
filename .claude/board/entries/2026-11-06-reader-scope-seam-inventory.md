# Reader-scope seam inventory (D-LANESOCK-4)

**Date:** 2026-11-06. **Status:** in PR, docs only. Follows D-LANESOCK-3 (#1970, merged `fb27c46`).

- `ogar_auth::federation::VerifiedLogin` is the only sealed proof of login; `AuthenticatedUser` has public fields and proves nothing.
- `authorize_identity` / `authorize_user` return decisions (allow/deny, row scope, field mask) that carry no actor, class, frame or issuer: values, not witnesses.
- `dr::Scope` is a bare id compared by equality; all literal constructions are in tests, examples or probes. `FrameStore`/`Frame` is the sealing precedent; `GuardWitness` is value-derived and cannot carry authority.
- dt64 and rbac depend only on the contract; OGAR depends on lance-graph. An issuer starting from `VerifiedLogin` lives in OGAR or downstream.
- Shape: a sealed scope type next to `Scope` in dt64, issued by its own store; the issuer holds the store and issues only from `VerifiedLogin` via `authorize_identity`.
- Downstream (report-rs `83093de`, reported to its owner): `None` → all customers in `LanceGraphScope::decide_in`; `RoleScopeAuthority::scope_for` accepts a hand-built `AuthenticatedUser`.

Full text: `.claude/plans/d-lanesock-dr-dt64-lane-socket-v1.md` §8.
