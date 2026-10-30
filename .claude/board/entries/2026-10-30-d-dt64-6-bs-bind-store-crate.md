# D-DT64-6-BS — the bind store as production code: `crates/lance-graph-dt64`

**Date 2026-10-30 · Status VERIFIED-IN-CODE · Plan [`d-dt64-6-routing-bindstore-behavior-v1.md`](../../plans/d-dt64-6-routing-bindstore-behavior-v1.md) Amendment A.5 · Crate `crates/lance-graph-dt64` (workspace member, depends on `lance-graph-contract` only)**

DT is the ClassId-addressed binding store (D-DRDT §1). The crate stores bindings and their version log, never business data, and evaluates nothing. It closes the four D-DRDT §4 gaps:
- `MethodId` + `SigRev`;
- a K3 guard as `set_algebra::SetExpr` + `Logic`, with `Guard::witness` as the normal form;
- an ordered `effects` list;
- one `resolve(view, classid, schema_rev)`.

Class behaviour is immutable (D5). An override is a derived classid plus a re-route. Only the route log is versioned: `routed_at(version, position)`.

Refusals are named. UNKNOWN is never empty:
- unknown class, unrouted position, unknown method;
- schema revision and field-basis drift;
- a dropped, added or mismatched guard;
- a stale signature revision.

The D-DRDT §5 falsifier passes: two methods on one cell with equal read and write sets and different effect order stay distinct, and a dropped guard refuses. 11 tests; clippy and fmt clean.

**OPEN:**
- OGAR `CapabilityAuthority` wiring (D6);
- persistence (EXO-55);
- a DR #4 router carrying `MethodRef`;
- the immutable unit (§6 Q1). The crate implements the D5 wording: the classid.
