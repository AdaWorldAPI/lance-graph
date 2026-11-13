# pg-parity-harness

crate: pg-parity-harness
path: crates/pg-parity-harness
workspace: member
digest: 11e53762034e9f10
evidence legend: ../index.md#evidence-legend

Generated; the code is the truth. See ../index.md.

## Depends on


## Depended on by

- rdep lance-graph-r2il dev — crates/lance-graph-r2il/Cargo.toml:39 CONFIRMED

## Public items

- struct Refusal [pub] — crates/pg-parity-harness/src/compare.rs:10 CONFIRMED
- enum Answer [pub] — crates/pg-parity-harness/src/compare.rs:17 CONFIRMED
- enum Order [pub] — crates/pg-parity-harness/src/compare.rs:24 CONFIRMED
- enum Diff [pub] — crates/pg-parity-harness/src/compare.rs:32 CONFIRMED
- enum Outcome [pub] — crates/pg-parity-harness/src/compare.rs:106 CONFIRMED
- fn multiset_diff [pub] — crates/pg-parity-harness/src/compare.rs:115 CONFIRMED
- fn ordered_diff [pub] — crates/pg-parity-harness/src/compare.rs:148 CONFIRMED
- fn types_diff [pub] — crates/pg-parity-harness/src/compare.rs:189 CONFIRMED
- fn compare [pub] — crates/pg-parity-harness/src/compare.rs:201 CONFIRMED
- type Cell [pub] — crates/pg-parity-harness/src/lib.rs:54 CONFIRMED
- type Row [pub] — crates/pg-parity-harness/src/lib.rs:55 CONFIRMED
- struct Rows [pub] — crates/pg-parity-harness/src/lib.rs:60 CONFIRMED
- fn text [pub] — crates/pg-parity-harness/src/lib.rs:66 CONFIRMED
- fn boolean [pub] — crates/pg-parity-harness/src/lib.rs:71 CONFIRMED
- struct Skip [pub] — crates/pg-parity-harness/src/pg.rs:14 CONFIRMED
- enum PgError [pub] — crates/pg-parity-harness/src/pg.rs:17 CONFIRMED
- struct Pg [pub] — crates/pg-parity-harness/src/pg.rs:36 CONFIRMED
- fn Pg::connect [pub] — crates/pg-parity-harness/src/pg.rs:46 CONFIRMED
- fn Pg::version [pub] — crates/pg-parity-harness/src/pg.rs:74 CONFIRMED
- fn Pg::schema [pub] — crates/pg-parity-harness/src/pg.rs:78 CONFIRMED
- fn Pg::script [pub] — crates/pg-parity-harness/src/pg.rs:85 CONFIRMED
- fn Pg::query [pub] — crates/pg-parity-harness/src/pg.rs:94 CONFIRMED
- fn parse_rows [pub] — crates/pg-parity-harness/src/pg.rs:180 CONFIRMED
- fn parse_sqlstate [pub] — crates/pg-parity-harness/src/pg.rs:208 CONFIRMED
- struct Case [pub] — crates/pg-parity-harness/src/report.rs:7 CONFIRMED
- struct Report [pub] — crates/pg-parity-harness/src/report.rs:14 CONFIRMED
- fn Report::new [pub] — crates/pg-parity-harness/src/report.rs:19 CONFIRMED
- fn Report::record [pub] — crates/pg-parity-harness/src/report.rs:23 CONFIRMED
- fn Report::outcomes [pub] — crates/pg-parity-harness/src/report.rs:27 CONFIRMED
- fn Report::counts [pub] — crates/pg-parity-harness/src/report.rs:32 CONFIRMED
- fn Report::unsupported [pub] — crates/pg-parity-harness/src/report.rs:46 CONFIRMED
- fn Report::summary [pub] — crates/pg-parity-harness/src/report.rs:57 CONFIRMED
- fn Report::render [pub] — crates/pg-parity-harness/src/report.rs:67 CONFIRMED
- fn check [pub] — crates/pg-parity-harness/src/report.rs:85 CONFIRMED
- fn mutation_outcomes [pub] — crates/pg-parity-harness/src/report.rs:98 CONFIRMED
- fn check_mutation [pub] — crates/pg-parity-harness/src/report.rs:127 CONFIRMED

## Imports from in-scope crates

- none

## Calls

CALL_GRAPH=ARTIFACT_MISSING
- calls ARTIFACT_MISSING — no ruff_charon artifact for this crate UNKNOWN

## Architecture

- arch LaneBinding checked-by LaneBinding::check [CONFIRMED] validation=OK — crates/lance-graph-dt64/src/lane.rs:374 CONFIRMED

Generated from Cargo manifests — extracted relationships only.

```mermaid
flowchart LR
  n80["pg-parity-harness"]
```
