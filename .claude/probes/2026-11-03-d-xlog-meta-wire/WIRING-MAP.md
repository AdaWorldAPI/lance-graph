# D-XLOG-META-WIRE P0: virtual XY → M1 → result → scoped M2

**Status:**
- MEASURED, lab only. A read-only source census plus one isolated falsifier (`probe/`, own workspace, 18/18 checks).
- No production change. dt64, r2il, mask-risc, Quack and ndarray are used unchanged.
- No Gather admission, no new relation, scope store, scheduler or evaluator.

**Sources:**
- lance-graph `fb5e2d35b10fc4940f5c78285bae246f90c7faad` (`origin/main`).
- OGAR `ogar-r2il` / `ogar-loco` from `main` (cargo checkout `5b82ae0`).
- ndarray `b3d7919662a3`.
- rustc 1.99.0.

```sh
cd .claude/probes/2026-11-03-d-xlog-meta-wire/probe
CARGO_PROFILE_DEV_DEBUG=0 cargo +1.99.0 run --release -q   # exits 1 on any failed check
```

## Answer

The chain `virtual XY → M1 → result → scoped M2` is **not expressible as two methods under the current contracts**.

**The first actual break is Receipt → M2.**
- `run` has no input that names another method's result.
- A field that the same run has not written is read from a resident **value** lane, or refused.
- A mask result cannot cross at all:
  1. the receipt keeps only its field position;
  2. the slot is stale once the run ends;
  3. `FieldLanes` cannot name a mask;
  4. `KEEP` is not executable.

This is the existing survival law working as written, not a missing cable.

**The one lawful route is inside one binding.** M1's mask can feed M2's fold *inside one binding* (an in-program register, Data origin, no `FoldResultFeedsFold`).
- That is not M1 → M2. It is a new method, and a registered class's behaviour is immutable.
- It needs a derived classid and a re-route (`register_derived`), which exist.

## Transitions

| # | transition | producer | consumer | contract actually carried | first missing edge or typed refusal | reusable mechanism | owner gate | falsifier (probe) |
|---|---|---|---|---|---|---|---|---|
| 1 | DR position → class | `BindStore::route` (`dt64/src/lib.rs:739`) | `resolve_at` (`:816`), `resolve` (`:794`) | classid per position at a `RouteVersion`; class schema; fields through `ClassView` | none: CONNECTED | `RouteLog`, `MapTable` | — | existing dt64 tests |
| 2 | virtual XY as a method body | `Effect { algebra, args }` (`lib.rs:128`) | r2il `prepare` (`binding.rs:276`) | words over field positions of ONE class; `Arg::Field` / `Arg::Const` | a cross-row hop has no executable word: `VIA` → `NotExecutable { opcode: VIA }`. `Gather`/`Semijoin` have no R2IL word | `EXECUTABLE` (`binding.rs:98`): single-row compares, `And/Or/Xor/Not`, folds | **EXO-140** (`VIA` lowers to `EqU32Via`, an unclassified ADMISSION B site) | T6 |
| 3 | change → eligible method | none: no change event is emitted. `run` computes effects but writes nothing back; `Cube::DeltaLog` (`xlog/src/cube.rs`) never reaches dt64 | `Resolved::methods_on` (`lib.rs:454`), whose only callers are dt64 tests | `on: WideFieldMask` | EXISTING BUT UNWIRED: no producer | `methods_on` is the trigger; `methods_touching` (`:523`) is the dependency footprint (ON ∪ reads ∪ writes) | v3 write path / mailbox owner (who writes M1's effects) | T1a–c: `methods_on(m1)` = {M2, M2_WHEN}; `methods_touching(m1)` adds M1 (it writes m1, it is not triggered by it) |
| 4 | MethodRef → native fold | `MethodRef::of` (`lib.rs:272`), `admit` (`:832`) | `prepare` / `prepare_native` (`binding_native.rs:202`) | method, sig rev, guard witness | `GuardNotExecutable` (`binding.rs:291`); `ThreeValuedNotExecutable`; sink words (`KEEP`, `GROUP_SUM`) `NotExecutable` | `ExecPermit` (lab, `binding.rs:126`) | D-DT64-6 owner (WHEN evaluation); permit issuer undecided | T5, T6 |
| 5 | fold → Receipt | `run_steps` (`binding.rs:354`) | `Receipt` (`:178`), `Executed` (`:169`) | `values: (field, i64)`, `intermediates: field positions`, `plan_id`, `MethodRef`, `FrameRevisions`; native: at most 8 outputs (`MAX_NATIVE_OUTPUTS`) | by design: a non-scalar `Val` (`Slot`, `Address`) keeps only its field id. `run` then calls `discard_pending` (`r2il/src/lib.rs:533`, epoch += 1), so a kept slot is `StaleSlot` | `Val::Slot { epoch }` (`lib.rs:200`) | the survival law (inter-program computed-mask handoff forbidden) | T2a–b, T4a–b |
| 6 | **Receipt → M2** | `Receipt.output` | `run(plan, store, frames, now, permit, fields, dialect)` (`binding.rs:330`) | — | **FIRST BREAK.** No parameter takes a receipt or a value. `Arg::Field(p)` is the same run's earlier value, else `FieldLanes` → `lane()` (`binding.rs:579`), which accepts only I32/U32/U64 lanes → `FieldUnbound` / `LaneUnusable`. `Arg::Const` is fixed at registration (immutable, D5) | none for a mask. For a scalar, only a caller-written resident lane: a WORLD write, which no seam performs | v3 write-on-behalf owner (scalar route); survival law (mask route) | T3a–b |
| 7 | result → narrower Scope | `Guards { frame, focus, scope }` (`dr.rs:509`) | `Guards::check` (`dr.rs:526`) | `Scope { id: u64 }`, `Focus { revision: u64 }`: opaque ids compared for equality | no type relates a scope to a population. Another scope id is `ScopeChanged` | `Frame`/`FrameStore` issuer identity | EXO-46 (binding membrane) | T7 |
| 8 | M1's meaning as M2's WHEN | `Guard { expr: SetExpr, logic }` (`lib.rs:152`), witness in `MethodRef` | `prepare` | the guard expression, stored and never evaluated ("The store keeps it; it evaluates nothing", `lib.rs` header) | `GuardNotExecutable` | D-DT64-6 §3: WHEN is a Quack WHERE into a guard mask G, which would be an in-program register in M2's own program (meaning crosses, no mask hand-off) | D-DT64-6 owner | T5 |
| 9 | M1 ∘ M2 fused | one binding with M1's and M2's effects | `prepare` / `run` | in-program register: Data-origin mask into a fold | lawful. Breaks only identity: behaviour is immutable (`BehaviourImmutable`), so the composite is a new derived classid | `register_derived` (`lib.rs:570`) + `route` | DT64 owner (composite identity) | T8a–e, T9 |
| 10 | consequence retention | `Receipt` (in memory) | none | — | no persistence or replay path in r2il; xlog `bag_count_admitted` (`xlog/src/lib.rs:384`) returns `Vec<i64>` outside `Receipt`, `Guards` and the permit | `FrameStore` versions; THOUGHT probes (lab only) | v3 / EXO-108 (Bag) | — |

## The lawful partial trace (T8)

FUSED = `m1 = a < 5; t = b < 10; u = m1 AND t; s = POP_COUNT(u)` in one binding:

| check | result |
|---|---|
| matches the scalar oracle | 177 = \|a<5 ∧ b<10\| |
| M1 operand removed (`s = POP_COUNT(t)`) | 435 = \|b<10\|, so the binding is load-bearing |
| one supporting row of M1 removed | 176 |
| unrelated lane `k` changed | unchanged |
| another scope | `ScopeChanged` |
| a stale frame | refused |
| tuples / stored intermediates | none; the mask is an in-program register |

This is the v4 register-gated pattern at the method level. It shows the algebra can scope M2 by M1 *inside one program*. It does not show M1's result reaching a separately registered M2.

## Crosswalk: local wiring vs upstream

Upstream facts marked † are quoted from the 2026-11-02 archaeology entry (`.claude/board/entries/2026-11-02-coresearch-xlog-souffle-crepe-dr3-archaeology.md`), not re-verified here.

| capability | class | evidence |
|---|---|---|
| position → class → fields | EXISTING AND CONNECTED | transition 1 |
| ON-trigger lookup (`methods_on`) | EXISTING BUT UNWIRED | no change producer; test-only callers |
| dependency footprint (`methods_touching`) | EXISTING BUT UNWIRED | test-only callers |
| method execution (single-row predicates + folds) | PARTIALLY IMPLEMENTED | `EXECUTABLE` subset, lab `ExecPermit` |
| cross-row hop as a method word | BLOCKED BY OWNER RULING | `VIA`/`EqU32Via` (EXO-140); no Gather word |
| WHEN evaluation | BLOCKED BY OWNER RULING | `GuardNotExecutable`; D-DT64-6 §3 names the intended shape |
| result → second method | MISSING (by law for masks; no write path for scalars) | transitions 5–6 |
| scope as a population | NOT A CONTRACT TODAY | `Scope { id }` |
| XLOG F100 lens execution | EXISTING BUT UNWIRED† | `prepare` refuses any lens that is neither POLYFILL nor the R2IL fold concept (`NotR2ilReading`, `binding.rs:297`) |
| XLOG MagicSetReport | dropped in the fork's compile.rs† (`:1496/:1511`) | owned by the D-XLOG-WIRE W1 session, not here |
| XLOG edge polarity | dropped at stratify → lower† | D-XLOG-WIRE (PR #1936) |
| XLOG recursion (`RirNode::Fixpoint`) | dormant upstream IR node†; semi-naive is runtime- and CUDA-only upstream† | EXO-73 |
| Soufflé magic sets | upstream, positive programs only† | semantic reference only |
| Crepe / `crepe_plan!` | upstream; `crepe_plan!` drifts on `let` and unsafe-rule labels (D1/D2)† | lab |
| Scallop provenance semirings | upstream capability; nothing imported | OUTSIDE scope of this chain |
| DialEgg / egglog-MLIR | offline compiler rewrites; e-graph discarded per call† | OUTSIDE scope (no runtime role) |

## Backlog (planning estimates, not deadlines)

Each item: producer → missing edge → consumer → falsifier → owner → PR scope.

**P0**
- **WHEN as in-program register.**
  - Path: M2's `Guard` (SetExpr over a field M1 *defines*) → evaluated as a Quack WHERE inside M2's own program → `prepare`/`run`.
  - Falsifier: M2_WHEN = FUSED oracle; remove the guard → 435.
  - Owner: D-DT64-6 (lift `GuardNotExecutable`).
  - Scope: one seam change, gated. **OWNER DECISION.**
- **`methods_on` / `methods_touching` negative controls.** Done here (T1); 0.5 d to move into dt64 tests. **WIRE NOW.**

**P1**
- **Change → `methods_on`.** Needs a producer of field-change events, which needs a write path (v3 mailbox, write-on-behalf). Falsifier: change a; M2 is offered, UNRELATED is not. 2–4 d after the write-path owner rules. **OWNER DECISION.**
- **Composite identity for M1∘M2.** `register_derived` + route; the falsifier is T8/T9 under a derived classid. 1–2 d. **PROBE.**
- **`VIA` executability.** EXO-140 classification first. **OWNER DECISION.**

**P2**
- XLOG compile records (MagicSetReport, polarity): already owned by D-XLOG-WIRE (session `01X6y3drw…`). **PARK** here.
- Caller-API polyfill. **PARK.**

**P3**
- Recursion (EXO-73), Bag ℕ-compose (EXO-108), cross-tile compose. **PARK.**

**SKIP**
- Carrying a `Slot` or mask across runs.
- Storing M1's mask as a plane or relation for M2.
- A scope store, scheduler or second evaluator.

## Recommendation

**OWNER DECISION** on transition 8 (WHEN evaluation as an in-program guard register).
- It is the shortest lawful route found. M2 states M1's *meaning* as its guard, and both evaluate in one program, so nothing is handed off and nothing is stored.
- Everything it needs exists except the evaluation of WHEN, which `prepare` refuses by name.
- T8 shows the same algebra already works when the effects are written into one binding.
