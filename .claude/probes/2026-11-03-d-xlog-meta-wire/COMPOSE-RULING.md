# D-XLOG-META-WIRE P0b: composing two MethodRefs at cold preparation, and the WHERE-scope chain

**Status:** MEASURED, lab only.
- Probe `probe/src/bin/v2_compose.rs`: 25/25 checks.
- Sections 1–2 are source-read; section 5 is source-read and tests already in the tree, nothing new run.
- No production change: `GuardNotExecutable` is untouched (check F0).
- The survival law is preserved: no mask crosses a program boundary, nothing is written to the WORLD, no Gather.

**Proposed ruling: PROPOSED / NOT IN FORCE, for operator review.**

```sh
cd .claude/probes/2026-11-03-d-xlog-meta-wire/probe
CARGO_PROFILE_DEV_DEBUG=0 cargo +1.99.0 run --release -q --bin v2_compose
```

## 1. Does D-DT64-6 / OGAR already provide method-body composition?

**No.** What exists:

| mechanism | where | what it composes | class |
|---|---|---|---|
| ordered `MethodBinding.effects`; `register` refuses `UndeclaredRead` | `dt64/src/lib.rs`; plan `d-dt64-6-routing-bindstore-behavior-v1.md` A.4 | steps inside ONE method | PRODUCTION |
| DO and ON share the property domain, so chaining is DO → field → ON | plan §3, citing `odoo-rs:docs/DT64-ON-WHEN-DO-REGISTRY.md:61-65` | methods, through a written field (the WORLD) | PROPOSED |
| `dep = do_ ∘ read_by` via `relcomp::compose`; schedule from `dep` | `tract-harvest-probe` (plan A.4 table) | a method dependency graph and a schedule | LAB |
| OGAR `KausalSpec::{Depends, StateGuard, Constrains, Onchange}` | `OGAR:crates/ogar-vocab/src/lib.rs:610-652` | triggers by field path; chaining through stored fields | other repo, PRODUCTION vocabulary |
| `ogar-loco::Program` as the method body | rejected by operator decision, 2026-10-31 (entries `2026-10-31-coresearch-mandatory-codegraph.md`, `…-d-tuple-polyfill-0-architecture.md`) | — | REJECTED |
| `register_derived` + `route` (D5) | `dt64/src/lib.rs:570, 739` | a new derived classid for new behaviour | PRODUCTION |

Every existing composition goes through a **written field** (WORLD) or stays **inside one method**. Nothing resolves two MethodRefs into one program.

## 2. The exact missing semantic edge

A WHEN leaf is `SetExpr::Sym(u32)` ("an opaque two-valued set … What carries it is the consumer's business", `contract/src/set_algebra.rs:139-142`) or `Res`, a predicate on one lane. Nothing lets a guard leaf name **another method's predicate**.

`GuardWitness` is the guard's normal form. It is identical whether `Sym(1)` stands for M1 (`a < 5`) or M1B (`a < 7`) (check **GAP**). So a reference carried as a `Sym` is invisible to `admit` and to `plan_id`:
- the referenced method's identity and signature revision cannot be checked from the contract;
- the lab composer has to carry them itself (`Composed.deps`).

**Missing edge:** a guard leaf, or a binding-side table checked by `admit`, that names `(MethodRef incl. sig_rev, predicate field)`, and a prepare step that inlines that predicate into the referencing method's program.

## 3. Bounded falsifier (lab composer emulating the proposed ruling)

The composer:
- resolves M2's WHEN `Sym(1)` through a lab table to (M1, field `m1`);
- prepends M1's predicate effects;
- gates each of M2's folds by the guard register (`u = m1 AND t; s = POP_COUNT(u)`);
- registers the composite with `register_derived` and re-routes the position;
- runs it through unchanged r2il `prepare` / `run`.

| # | requirement | result |
|---|---|---|
| F0 | production unchanged | `prepare(M2 with WHEN)` = `GuardNotExecutable` |
| 1 | M1, M2 independently registered | both admitted; distinct MethodIds, same class |
| 2 | WHEN refers to M1's predicate, not a slot | `Sym(1)` → (M1, `m1`) |
| 3 | one native fold program | `programs_run = 1`; deps `[(M1, rev 1), (M2, rev 1)]` |
| 4 | independent oracle | 177 = \|a<5 ∧ b<10\| |
| 5 | M1's predicate disabled → divergence | 435 = \|b<10\|; `Sym(1)` → M1B composes a different program |
| 6 | refusals | permit lacks M1 → `NotPermitted(M1)`; M1 of another class → `OtherClass`; M1 at another sig rev → `SignatureRevisionMismatch`; another schema rev → `SchemaRevisionMismatch`; another scope → `ScopeChanged`; permits for M1+M2 do not run the composite → `NotPermitted`; plan from before the re-route → `RouteChanged`; source moved → stale frame |
| 7 | cyclic dependency | M3 ↔ M4 → `Cycle([M3, M4, M3])`. Neighbours: a predicate computed from a fold → refused (the `FoldResultFeedsFold` rule, kept); Kleene WHEN → refused |
| 8 | `register_derived` + re-route **succeeds** | `Ok`, route → version 2; the base still resolves; adding the composite to the base → `BehaviourImmutable` |
| 9 | in-program narrowing ≠ DT64 population Scope | the narrowing lives in the program (intermediates `[m1, t, u]`); the same plan under another `Scope` id is refused. Scope is policy, not population |
| 10 | real DR-coordinate virtual relation | **not done.** Separately gated; the `a < 5 ∧ b < 10` fixture is not that test |

## 4. Proposed D-DT64-6 ruling (for operator review, not in force)

1. **Reference, not result.**
   - A WHEN may reference another registered method's *predicate*, as `(MethodRef incl. sig_rev, predicate field)`.
   - It never references a result, slot or Receipt.
   - Only two-valued guards over Distinct populations at first; Kleene stays refused.
2. **Resolved cold, executed as one program.** Preparation inlines the referenced predicate's effects ahead of the referencing method's effects, and gates its folds by the guard register. One native program; no mask leaves it.
3. **Identity.** The composite is a derived classid (`register_derived` + `route`, D5). Its plan identity covers every referenced MethodRef and signature revision. A changed reference is a new derived classid, never an in-place edit.
4. **Checks.** `admit` per referenced MethodRef; same class (or a declared derived chain); one Frame/Focus/Scope for the program; route checked at prepare and run.
5. **Permission.** Open owner question. The probe requires a permit for the composite itself and does not derive it from the permits for M1 and M2 (F6f).
6. **Acyclic, fold-free predicates.** A reference cycle refuses. A referenced predicate computed from a fold result refuses (keeps `FoldResultFeedsFold`).
7. **Not Scope.** This is in-program predicate narrowing. A DT64 population Scope bound to a WHERE view is a separate question (section 5).
8. **Unchanged laws.** No cross-program mask hand-off, no persisted intermediate mask, no silent WORLD write, no generalized Gather (EXO-140 still gates `VIA`).

## 5. The corrected chain: Global WHERE → DR view → MethodRef → Fold (operator correction, same day)

Source read only; no new run. This is the chain the operator re-prioritised to P0/P1.

| step | where | class | carries |
|---|---|---|---|
| Global WHERE | Quack typed WHERE → mask-risc `Keep`: one borrowed 8 KiB mask over 65,536 addresses ("the only materialization") | PRODUCTION (Quack) | the rows where the criteria are TRUE |
| DR addressing | `dt64::dr::{LocalMap, MapTable, FrameStore}` | PRODUCTION | global → local positions, versioned |
| projection into a view | `dr-math-binding-probe/src/where_dr3.rs` `project`: `map.len()` probes, never a pass over 65,536 rows; a frame must carry `map.digest()` | LAB, not lifted into dt64 | `LocalSelection { truth, axis, method, global frame, local frame, focus, scope, criteria }` |
| admission | `LocalSelection::admit` (`where_dr3.rs:225`) | LAB | re-checks global seal, local frame, focus, scope, axis, **method**, criteria |
| fold | `selected_count`, `selected_column`, `selected_column_k3` | LAB | Boolean folds refuse K3/Bag registers |
| reuse of one selection across programs | `drf-focus-probe/examples/p2_quack_where.rs` (R kept once, then `Plane(R) ∧ pres_p` for P properties) | LAB | works, but "Quack has no frame": stale codebook, scope and selection run silently stale (measured there) |

**First break of the corrected chain (P1):**
- The admitted selection is bound to **one** method signature: `if register.method != self.method { SchemaChanged }` (`where_dr3.rs:244`).
- The test `foreign_owner_codebook_criteria_and_schema_refuse` pins it: "Another DR method over the same axis" → `SchemaChanged`.
- Two methods can read one WHERE only by projecting it twice (O(map.len()) each, lawful: the WHERE mask is the demanded result). There is no shared, admitted scope object.
- That method type is the probe's `MethodSignature { owner_class, method_ordinal, … }`, not dt64's `MethodRef`.
- Production dt64 has `Scope { id }` and nothing that binds a scope to a WHERE view.

**Consequence:** the common case the operator describes (two methods over one global WHERE view, no `Receipt → M2`) needs no new algebra. It needs the WHERE-view binding (`LocalSelection` + `Criteria`) lifted from the lab and tied to `MethodRef`, plus a ruling on whether one admitted selection may serve several MethodRefs.
