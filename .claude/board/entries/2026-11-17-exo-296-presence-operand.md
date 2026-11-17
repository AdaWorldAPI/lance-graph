# EXO-296: a presence plane as an F000 operand (`Arg::Present`)

**Date:** 2026-11-17 · **Status:** MEASURED (test-pinned) · **Scope:** lance-graph-dt64 + lance-graph-r2il

zammad-rs needs `IS NULL` scopes (`ImportJob.unfinished`, `ImportJob.running`, `KnowledgeBase::Category.root`). Under decision B (EXO-294) `IS NULL` is `IntNot` of the presence plane, but an F000 effect could not name a presence plane: `Arg` was `Field | Const`, and `Field` resolves to a value lane.

Shipped:

- dt64 `Arg::Present(p)`: checked against the basis; `p` must be a declared read (a computed field has no stored presence); the binding digest tags it apart from `Field(p)`.
- r2il `FieldPresence` (field → index into `Planes::masks`) and `prepare_native_present`. `Present(p)` resolves to `Addr::Plane`, so the frozen programs read the run's plane by index and `run_native` stays at zero allocations. `prepare_native` and the interpretive `run` refuse `Present` as `PresenceUnbound`.
- No new polyfill word.

Chosen over EXO-296's option 2 (a native gate): a gate covers only predicates that fit one gate, while `running` needs a presence plane, a second presence plane and a value compare in one plan.

Evidence: `tests/integration/binding_presence.rs`, three scopes against a scalar oracle at n = 1, 63, 64, 65, 1000, 16385; six tests; four disables red.

OPEN: none.
