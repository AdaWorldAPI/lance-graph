# 2026-11-22 — Consumer synergy: zammad-rs and report-rs against lance-graph

**Status:** MAP. Read-only for both consumers; nothing is proposed inside report-rs (in production) or zammad-rs. Trees read: report-rs `main` `f2e55c3` (#182), zammad-rs `main` `d81ac8a` (#99), lance-graph `main` `138806961`. Companion to `2026-11-22-synergy-fruits-revision-1806-2157.md` (Arc 17).

**Method.** Three Haiku mappers (zammad-rs crates/docs, report-rs src/docs, lance-graph mentions of both); every fact below that a row or amendment relies on was re-read by the orchestrator at the cited line.

## 1. What each consumer takes from lance-graph

| | report-rs | zammad-rs |
|---|---|---|
| Role | production portal (Railway); PostgreSQL in production through `work-items-pg` (`Dockerfile:49`: `--features lance,work-items-pg,licreport`) | Rails semantic compiler; "lance-graph is not on the execution path" (README); lance-graph crates are shadow runs and gates |
| lance-graph crates | contract, dt64 (`src/xverweis.rs`), report, rbac, dir-sim (5 files) | quack, mask-risc, dt64, r2il, pg-row, contract (quack-shadow); codegraph, ogar, arm-discovery (rails-codegraph); dt64, r2il, mask-risc (rails-f000); dev: xlog, pg-parity-harness |
| What lance-graph gets back | the first consumer of D-MAPPING-POLYFILL-1, the lane socket (#1969/#1970), dir-sim licence planes (#2105–#2108); report-rs #33 as the Codegraph CG-7 transfer case (#2002) | the PostgreSQL 16.15 oracle behind D-REGC (#95–#97), D-GPW (#41, #45: 57 lookups equal PostgreSQL), D-QSP-VROW (`vrow_ticket_states_pg_parity.rs`), the cube bijection (`cube_pg_bijection.rs`), the presence operand (EXO-296), CodeTable under one frame (EXO-328/329) |

The two consumers are also linked to each other: report-rs is a server-side client of zammad-rs `crates/zammad-ticket` (`src/ticket_api.rs:1-19`, `POST/GET /api/v1/tickets`, the user's ID token forwarded, `REPORT_TICKET_API_URL` unset = off); zammad-ticket deploys as one partial of the Railway project `report-rs` (`crates/zammad-ticket/deploy/railway.ts:1,40`); zammad-rs `web-login` was lifted from report-rs's sign-in (`crates/web-login/src/lib.rs:36`).

## 2. Closed ends found (lance-graph's board is behind the consumers)

- **D-REGC-10 adoption is done in zammad-rs.** `crates/quack-shadow/src/bound.rs:14-23` imports `pg_row::{AbsentAs, …}` and `lance_graph_pg_row::{ColumnSpec, LiveSource, PgRowRefusal, Reader}`; `:445` sets `absent: AbsentAs::Null`; `:772` reads through `Reader::open_selected`. STATUS_BOARD still said "zammad-rs adoption queued". Residual: `read_selection` (`:359`) and `absent_row` (`:378`) still exist; whether they are now oracles or live paths was not read.
- **The `decide_in` widening observation is stale.** `d-lanesock-dr-dt64-lane-socket-v1.md:159` says `decide_in` maps `scope: None` to all customers. On report-rs main, `customers_in` (`src/iam.rs:318ff`) widens an unscoped decision only when the actor's own entry is the explicit all-customers grant; any other unscoped decision admits no customer.
- **`REPORT_BUILD_REVISION` is set** (`Dockerfile:49`, from `RAILWAY_GIT_COMMIT_SHA`), the step `g2-dr-virtual-projection-reconcile.md:127` asked for.

## 3. Synergies (loose ends that connect)

- **C1. Both consumers build the DT64 scope id themselves.** report-rs `src/xverweis.rs:219` `const READER: Scope = Scope { id: 1 };` and zammad-rs `crates/quack-shadow/src/bound.rs:32` `const SCOPE: Scope = Scope { id: 1 };`. ISS-DT64-SCOPE-ID-IS-NOT-A-GRANT is not hypothetical: a council ruling (private id, or `bind_lanes` taking a sealed scope) breaks two call sites in two repos. report-rs #33's admission ledger item 3 names the candidate shape `VerifiedLogin → ReportAuthority → ReportScope` (as in OGAR `ogar-flow-m365`), and zammad-rs `web-login` already derives roles from the same identity provider: the sealed scope has a natural issuer in both consumers.
- **C2. report-rs #33's admission ledger is a list of lance-graph fruits.** Item 3 reader scope = ISS-DT64-SCOPE-ID-IS-NOT-A-GRANT; item 5b = ISS-DR-FRAME-TO-LANCE-VERSION-HAS-NO-WITNESS (D4 / S-INGRESS); item 6 canonical text→code authority = dt64 `CodeTable` (#1982, EXO-207), which zammad-rs already exercised (EXO-328); item 8 production XVERWEIS `MethodRef` = a same-program `UniqueU64` terminal in mask-risc (`crates/lance-graph-quack/tests/xverweis_masked_read.rs:16`); item 9 = ISS-GATHER-HOP-1616. #33 is open, not merged, "do not merge yet".
- **C3. zammad-rs is where D-GPW-5, D-REGC-11, D-OMR-7 and D-INTAKE-2F land**, all "Queued (zammad-rs side)" on STATUS_BOARD. zammad-rs already has the oracle each one needs (512 recorded SELECTs equal to PostgreSQL). Order by dependency: D-INTAKE-2F (quack-shadow onto `Table`, #95's gate unchanged) before D-REGC-11 (owner-held 512-byte rows), and D-REGC-6's predicate count over the 512 calls first.
- **C4. Two consumers, two local class ids, one governance question.** report-rs binds under a process-local `BINDINGS_VIEW = 0x7F1D`, never persisted (`xverweis.rs:210-214`); zammad-rs's laboratory id `0x050D` collides with the shipped OGAR concept `crm_campaign` (`docs/architecture/zammad-identity-governance.md` §1, mirrored at lance-graph `ogar_codebook.rs:688`). The first is safe by construction; the second is safe only while it stays in the lab. Both wait on OGAR mints, which are OGAR's call.
- **C5. Ticket intake as the first cross-consumer flow.** zammad-rs `docs/integrations/report-rs-security-ticket-intake-v0.md` proposes report-rs as the candidate compiler and zammad-rs as the adapter, with "tenant/customer enforcement must precede dedup lookup, history retrieval …" (`:81`). That is the reader-scope question again (C1), at the HTTP boundary instead of the lane socket.

## 4. Fruits

- Low-hanging: update STATUS_BOARD D-REGC-10 (done in zammad-rs) and the stale `decide_in` line (done in this commit).
- High-signal: bring C1 to the council with both call sites named; the ruling's migration is two one-line changes plus whatever issues the sealed scope.
- Heavy: D-REGC-11 and D-INTAKE-2F in zammad-rs (consumer side, not lance-graph's to land).

## ⊘ Corrections (2026-11-22, same day, from a handoff carrying the recorded EXO decisions; code facts re-verified at lance-graph `138806961`)

- **C1 is wrong in its last sentence.** A DR/DT `Scope` is a source-partition compatibility witness, not authorization (DECISION, EXO-206). It is not issued from a login, and lance-graph does not depend on any identity provider or on `ogar-auth`/`ogar-rbac`. Authorization stays in each consumer, through the provider-neutral `lance_graph_contract::rbac` contract, checked before a protected read. The two `Scope { id: 1 }` constants stay.
- **C3 ordering is not an instruction.** zammad-rs #98 already moved reads to `Reader::open_selected`. Whether moving all of `quack-shadow` onto the intake `Table` (D-INTAKE-2F) removes work or only swaps a working adapter is to be assessed first. The measured work is #96's findings (lanes checked twice, rendered cells not using the checked lanes, 546 per-row key resolutions on join rows).
- **report-rs #33 (§3 C2):** the value layer is not to be promoted by a visibility change; EXO-174 requires exact decimals and no report-rs value store.
- **Method identity is decided** (EXO-199, `MethodRef` + `Stamp`, `lance-graph-dt64/src/lib.rs:369`). **Deletes already advance the data version** (`lance-graph-intake/src/lib.rs:957`); EXO-337 is the separate abort-representation question.
