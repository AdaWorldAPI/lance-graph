# sql-polyfill-r2il-v1 — SQL() as a polyfill of R2IL procedures over mask-risc

**Status:** PROPOSAL (D-SQLP-0..8). Nothing beyond W1 is authorized until the
D-SQLP-3 benchmark clears its pre-registered threshold.
**Date:** 2026-10-11. **Repos:** lance-graph (crate, procedures, codegen,
quack/SAP front ends), lance-graph-java (`sql()` membrane, `plan_eval`),
OGAR (fold-band bytes, concept mint), ndarray (missing lane primitives).

## 1. Overview

One query vocabulary for every front end, with no new layer and no new
engine:

```text
Java sql()  ─Panama call─┐
quack       ─────────────┤
SAP binder  ─────────────┼─▶ SQL procedures (loco PROC_DEF bodies)
Rust caller ─────────────┘        │  bodies are R2IL fold-reading calls
                                  ▼
                         FoldDialect ──▶ mask_risc::Program ──▶ ndarray::simd
```

This is ndarray's SIMD shape one tier up. ndarray names SIMD words and its
backends implement them; here SQL procedure names are the API, R2IL calls
implement them, and mask-risc is the only engine. Java sees one Panama call,
and Valhalla value records carry the parameters and result handles; no
opcode, mask or lane crosses (lance-graph-java `CLAUDE.md`, "THE JAVA SURFACE
IS `sql()`").

It follows the 2026-09-05 lance-graph ruling that planning moves to
`ogar-loco` / `ogar-r2il` and DataFusion is in a grace period
(E-PLANNING-MIGRATES-TO-LOCO-R2IL-DATAFUSION-IS-GRACE-PERIOD-1).

## 2. What exists (VERIFIED-IN-CODE, 2026-10-11)

- **OGAR `ogar-loco`** nests behaviour already: shared-core `PROC_DEF`
  (`0x83`), `PROC_CALL` (`0x84`), `PROC_ARG` (`0x85`), per-opcode `body_refs`,
  an interpreter that follows body references with a recursion-depth limit,
  and `Vocabulary::branches` (`vocabulary.rs:261`), the predicate a cast to
  code would ask.
- **OGAR `ogar-r2il`** is one opcode table with two readings: the machine
  reading (`CONCEPT_R2IL_MACHINE = 0xC400`, 82 R2IL opcodes at `0x90..0xE1`)
  and the fold reading (`CONCEPT_R2IL_FOLD = 0xC401`) with a 12-opcode
  extension band at `0xE2..0xED` (`VIA`, `RANGE`, `SUM`, `MIN`, `MAX`,
  `GROUP_SUM`, `KEY_RUNS`, `ANY`, `ALL`, `KEEP`, `SCATTER_OR`, `BLEND`). A
  fold byte is minted only for a mask-risc `Pred`/`Terminal` that already has
  its own parity case. Both concept ids are PROVISIONAL (not minted in
  `ogar-vocab`).
- **lance-graph `FoldDialect`** executes the fold reading over mask-risc. It
  lives only in `crates/r2il-mask-abi-probe/tests/row_bridge.rs`, a test file
  in a workspace-excluded probe crate. Measured there: a join predicate plus a
  local compare runs in 2 facade passes against quack's 3; nothing
  proportional to row count reaches the interpreter stack; a scalar fold
  (`SUM`) finalizes and runs a `Program` at the point it is reached, which is
  how `SUM(revenue|m) - SUM(cost|m)` is expressed when a single-terminal
  `Program` cannot express it. Wired: `VIA`, `SUM`, `GROUP_SUM`, `RANGE`,
  `KEEP` and R2IL `Load`, `IntSub`, `IntAnd`, `IntNot`, `IntEqual`,
  `IntSLess`, `PopCount`. Refused by name: `MIN`, `MAX`, `KEY_RUNS`, `ANY`,
  `ALL`, `SCATTER_OR`, `BLEND`, `IntAdd`, and the unsigned compares
  (mask-risc has none).
- **Front ends that lower to `mask_risc::Program` today:** quack
  (`lance-graph-quack/src/lib.rs`: `lower`, `lower_fused`, `lower_group_by`,
  `lower_avg`); lance-graph-java `plan_eval` (`lgj-abi/src/exports.rs`,
  with the `lowering_convergence` test module proving its `Program` equals
  quack's); `lance-graph-sap` (CATS boundary adapters, "no SAP evaluator").
- **Column types mask-risc reads:** `LaneRef::{I32, U32, U64}` plus strided
  reads; quack's binder knows `FieldKind::{I32, Code}` only.

## 3. What does not exist

- No crate holds `FoldDialect`; nothing in production calls it.
- No SQL procedure bodies.
- No loco-body → Rust code generator. `ogar-emitter` emits OGAR classes as
  triples/DDL, and `ogar-render-askama` renders class artifacts; neither
  turns a `FunctionBody` into Rust.
- No `sql()` export in lance-graph-java.
- No float, decimal, date or i64-compare lanes in mask-risc; no ndarray i64
  or float compare-to-mask.

## 4. Decisions

- **DECISION: SQL operations are procedures, not opcodes.** Each SQL operation
  is a `PROC_DEF` body made of R2IL fold-reading calls. BASIS: the fold band's
  own admission rule (a byte needs an execution-level reason), and the
  anti-lasagne rule (a layer must hold information no other layer holds). A
  procedure that can be written from existing calls earns neither. REVISIT
  WHEN: a SQL operation cannot be written from R2IL calls; then the missing
  piece is a mask-risc capability first, an R2IL byte second.
- **DECISION: no V4 meta layer between `ogar-r2il` and `ogar-loco`.** The
  nesting it would provide is `PROC_DEF`/`PROC_CALL`, which loco already has.
- **DECISION: one spelling per operation.** Once the crate exists, every
  front end lowers through the procedures. A front end that still builds
  `mask_risc::Program` directly is a second spelling of the same query.
- **SCOPE: two execution modes, never mixed up in claims.**
  - *Declared queries* (in the schema, like SAP CDS views, or prepared): a
    build step generates Rust from the procedure bodies; LLVM sees plain
    generic code and the layer costs nothing at runtime. This is the only
    mode where "compile time" is true.
  - *Ad-hoc SQL text at runtime* (Java `sql("…")` with a new string): the
    body is built when the query arrives and runs through the interpreter and
    mask-risc. Cost is one opcode dispatch per 16,384-row tile, the same as
    mask-risc today. Still no work on the Java side.
- **DECISION: Java holds no operations.** `sql()` is one Panama downcall;
  parameters and results cross as Valhalla value records and handles; every
  public call stays O(1) in rows (lance-graph-java `CLAUDE.md`).
- **OPEN: SQL text parser.** DataFusion is in its grace period, so its parser
  is not the route. Candidates: a standalone parser crate, or a declared-query
  schema that needs no text parser for the compile-time mode. Decided in W6.
- **OPEN: the SAP cost comparison.** "Orders of magnitude cheaper than SAP" is
  a hypothesis until W3/W4 are measured against a stated reference workload.

## 5. Waves

| Wave | D-id | Repo | Work | Gate (all must pass) |
|---|---|---|---|---|
| W0 | D-SQLP-0 | lance-graph | This plan + board rows | — |
| W1 | D-SQLP-1 | lance-graph | Move `FoldDialect` and its `Val`/`FoldError` types from `r2il-mask-abi-probe/tests/row_bridge.rs` into a workspace crate `lance-graph-r2il`; the probe's tests depend on it | every existing `row_bridge.rs` test passes unchanged against the crate; still one arity table in the tree (`ogar_r2il`'s); no new opcode; the named refusals still refuse |
| W2 | D-SQLP-2 | lance-graph | First procedures as `PROC_DEF` bodies over R2IL calls: `filter_lt` (`Load` + `IntSLess` + `KEEP`), `sum`, `group_sum`, and `margin` (two `SUM` + `IntSub`) | each answer equals a scalar reference on fixtures with ≥ 3 groups and both empty and full selections; no new opcode; disable runs: drop `KEEP` from `filter_lt` → red; swap `IntSub` operands in `margin` → red |
| W3 | D-SQLP-3 | lance-graph | Benchmark `margin` three ways: direct `ndarray::simd` calls, hand-built mask-risc `Program`s, the interpreted procedure. N = 65,536 and 1,048,576; selectivity ~1 % and ~50 % | answers identical before timing; release, quiet machine, median of 101 runs, avx512 and avx2 reported separately. Pre-registered: interpreted ≤ 1.05× hand-built `Program` at both N |
| W4 | D-SQLP-4 | lance-graph | Build-time generator: a declared procedure body → Rust that calls mask-risc / `ndarray::simd` directly, with opcodes as const generics (the `ternlog::<IMM>` pattern) | generated code's answers identical to the interpreted body; generated ≤ 1.05× direct ndarray at both N; disassembly of the hot loop contains no interpreter dispatch |
| W5 | D-SQLP-5 | lance-graph, lance-graph-java | quack, `plan_eval` and the SAP binder lower through the procedures | for every case `lowering_convergence` and quack's tests pin, the `mask_risc::Program` produced through the procedures is byte-identical to today's; no other behaviour change |
| W6 | D-SQLP-6 | lance-graph-java, lance-graph | `sql()` membrane: one Panama downcall (new ABI minor), parameters and result handles as value records; parser decision (§4 OPEN) | `ApiSurfaceTest` green (no opcode, mask or FFM type in a public signature); allocation gate shows no row-proportional Java allocation; full `AllTests` + consumer mains on JDK 28 |
| W7 | D-SQLP-7 | ndarray, lance-graph, OGAR | Backlog, one item at a time, mask-risc first: `MIN`, `MAX`, `KEY_RUNS`, `ANY`, `ALL`, `SCATTER_OR`, `BLEND`, `IntAdd`; lanes for i64 compares, f64, decimal (scaled i64), date (i32 days) | per item: ndarray primitive with parity on every realization row → mask-risc `Pred`/`Terminal` with parity → `FoldDialect` wiring → procedure; each its own PR |
| W8 | D-SQLP-8 | OGAR | Mint `0xC400` / `0xC401` in `ogar-vocab` (request to OGAR; this plan does not mint) | the provisional notes in `ogar-r2il` are replaced by minted ids; no persisted GUID uses them before |

**Kill conditions.** W3 interpreted > 1.5× hand-built at either N → stop
before W4/W5; keep W1 and W2, and re-plan around where the time went. W4
generated code slower than direct ndarray beyond the threshold → ship no
generator; declared queries use the interpreted path. A W5 `Program` that is
not byte-identical is a lowering bug: fix it, never re-pin the expectation.

## 6. Order and ownership

W1 → W2 → W3 → (W4, W5) → W6. W7 runs alongside from W2 on, driven by what
the procedures need. W8 is a request that can go out now.

- lance-graph: W1–W5 (W5's quack and SAP parts).
- lance-graph-java: W5 (`plan_eval`), W6.
- ndarray: W7 primitives; needs a session with push access to
  `AdaWorldAPI/ndarray`.
- OGAR: W7 fold bytes, W8.

## 7. Non-goals

- No new opcode for a SQL operation that procedures can express.
- No meta layer or second IR between `ogar-r2il` and `ogar-loco`.
- No new DataFusion code.
- No computation, materialization or opcode on the Java side.
- No GPU or JIT backend until W3/W4 numbers show per-tile dispatch costs
  something measurable.
