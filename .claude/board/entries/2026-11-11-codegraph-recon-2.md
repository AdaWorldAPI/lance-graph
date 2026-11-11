# 2026-11-11 — Coverage-aware impact; report-rs #33 recovered from compiler facts

**Status:** MEASURED (probe). `.claude/probes/2026-11-11-codegraph-recon-2/`. Follows
`2026-11-11-codegraph-recon-1` (#2000). Board: EXO-229, EXO-230.

`Codegraph::impact_report` (additive; `impact` unchanged) returns `impact`'s masks plus typed gaps:
external consumers unobserved, targets not extracted or undeclared, excluded regions, unresolved
dispatch, incomplete harvest, stale or unrecorded files, and `.rs` files under a read directory that
`files.tsv` does not list. Missing coverage never enters `Graded.unknown`. A missing symbol carries
its crate's gaps. 14 tests; every guard disable-verified red.

report-rs `81705d9` through Charon (lib, default features, `crate::app` opaque): CRATE `COMPLETE`,
yet 14 of 59 `src/` files unread (`iam.rs`, `main.rs`, `project_plan/*`, …). The store recovers the
first #33 finding: `workbook_delta::WorkbookSource::poll` has no library caller and no path to
`Ledger::ingest`; `Ledger::pull → source_fake::pull` is dynamic and reported UNKNOWN.

Blind arms graded strictly against the EXO-185 key (12 items): source arm 12 correct; store arm
7 correct, 3 partial, 2 unsupported; 0 wrong in both. The store's misses are configuration
(feature-gated `iam`), targets (test callers) and manifest facts (Cargo.toml). n = 1 per arm.

#2000 reproduces from its documented commands: rows equal, `machine-recovered.mmd` byte-identical,
Charon recall column identical.

OPEN: per-configuration stores (features, test targets); recording targets and opaque modules in
the store instead of declaring them.
