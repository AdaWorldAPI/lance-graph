# 2026-11-03 — D-XLOG-META-WIRE-P0: the first break between M1's result and M2

**Status:** MEASURED, lab only (`.claude/probes/2026-11-03-d-xlog-meta-wire/`, `WIRING-MAP.md`, probe 18/18). No production change; dt64, r2il, mask-risc, Quack and ndarray are unchanged.

- **First break: Receipt → M2.** `r2il::binding::run` has no input for another method's result.
  - `Arg::Field` reads the same run's value or a resident I32/U32/U64 lane, else `FieldUnbound` / `LaneUnusable`.
  - A mask never crosses:
    - the receipt keeps only its field id;
    - `discard_pending` makes the slot `StaleSlot`;
    - `KEEP` is not executable.
- **Scope does not narrow M2.** `Scope { id }` is an equality-checked id with no population behind it, and M2 cannot be guarded by M1's field: WHEN → `GuardNotExecutable`.
- **The virtual XY hop is refused as a method effect:** `VIA` → `NotExecutable`, and Gather has no R2IL word (EXO-140).
- **`methods_on` (ON) and `methods_touching` (footprint) differ as documented.** Both have test-only callers, and no change event reaches them.
- **The only lawful route:** M1's mask feeding M2's fold as an in-program register inside ONE binding.
  - Oracle 177; disable 435; one support deleted 176; unrelated lane unchanged; scope and frame refused.
  - It needs a derived classid, because behaviour is immutable.

**OPEN, owner decisions:**
- WHEN evaluation as an in-program guard register (D-DT64-6);
- `VIA` / `EqU32Via` executability (EXO-140);
- a write path for change events (v3);
- composite identity via `register_derived`.

**P0b (same day, `COMPOSE-RULING.md`, `probe/src/bin/v2_compose.rs` 25/25):**
- No method-body composition exists in D-DT64-6 or OGAR. Composition today goes through a written field or stays inside one method, and `ogar-loco::Program` was rejected.
- The missing edge is a guard leaf that names another method's predicate `(MethodRef, field)`. `GuardWitness` cannot tell `Sym(1)`=M1 from `Sym(1)`=M1B.
- A lab composer emulating the proposed ruling builds one program from M1 and M2:
  - oracle 177; disabling M1's predicate gives 435;
  - wrong permit, class, sig rev, schema rev, scope, route and frame are refused;
  - cycle refused; `register_derived` + reroute succeeds.
- Corrected chain (Global WHERE → DR view → MethodRef): `where_dr3::LocalSelection` binds one selection to ONE method (`SchemaChanged`, pinned by a test). It is lab only and not lifted into dt64.
- Proposed D-DT64-6 ruling: PROPOSED / NOT IN FORCE.
