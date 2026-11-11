# Held-out answer key — D-CODEGRAPH-RECON-2 Gate B/D

Source: EXO-185 manual findings (2026-11-10/11), re-checked by hand against
report-rs `81705d9` (PR #33 head). This file is NOT an input to any store and
is never shown to an arm before grading.

## Q1 — the two `WorkbookSource` types (EXO-185 loose end A)

K1.1 Two distinct identities:
  - `report_rs::source_fake::WorkbookSource` (src/source_fake.rs:19)
  - `report_rs::workbook_delta::source::WorkbookSource<T: DriveTransport>` (src/workbook_delta/source.rs:210)
K1.2 Only `source_fake::WorkbookSource` implements `report_rs::source::Source`
  (src/source_fake.rs:29); its `pull` builds an `IngestBatch` (source_fake.rs:39, :61).
K1.3 `workbook_delta::source::WorkbookSource` has an inherent `poll`
  (workbook_delta/source.rs:292) returning `Poll` (`Unchanged` | `Changed(VerifiedSnapshot)`);
  it has no `Source` impl and builds no `IngestBatch`.
K1.4 `Ledger::pull(&mut dyn Source, _)` (src/source.rs:331) calls `Source::pull` (dynamic)
  and `Ledger::ingest` (source.rs:334). The lib's only `Ledger::pull` caller with a
  workbook source is `projection.rs:409`, passing `source_fake::WorkbookSource::new`.
K1.5 Missing integration: no lib (`src/`) code calls `workbook_delta::...::WorkbookSource::poll`.
  Its only caller is `tests/p20_workbook_delta.rs:304`, which feeds `Workbook::ingest`
  (tests/p20:308), not the ledger. So the real SharePoint puller never reaches
  `Ledger::ingest`. Absence scope: all of `src/` and `tests/` at 81705d9, default features.
K1.6 Correct framing: the edge `poll → IngestBatch/Ledger::ingest` is ABSENT within the
  lib target; test callers exist (p20). Claiming "poll has no callers at all" is FALSE.

## Q2 — reader scope (EXO-185 loose end B, ledger item 3)

K2.1 report-rs calls `lance_graph_rbac::authorize::authorize_memberships` only from
  `src/iam.rs:250`, and `iam` is behind feature `iam` (src/lib.rs:22-23; Cargo.toml:107),
  OFF by default.
K2.2 No call to `authorize_scoped` anywhere in report-rs `src/` or `tests/` at 81705d9.
K2.3 Correct framing for a default-features extraction: the `iam` module is UNOBSERVED
  (not compiled), so "report-rs never authorizes" is FALSE; "no `authorize_scoped` call"
  needs either the iam feature extracted or a source scope check.

## Q3 — MapLog / value persistence (EXO-185 loose end C, ledger 5a/5b)

K3.1 report-rs does not depend on `lance-graph-dt64-lance` (Cargo.toml: only
  `lance-graph-dt64`, line 46); no `MapLog` use in src/ or tests/.
K3.2 DR identity lives in memory: `IdentityFrame` holds a `MapTable`
  (src/workbook_delta/frame.rs:56), advanced in `IdentityFrame::commit`
  (frame.rs:189, `self.table.advance(` at :195).
K3.3 Correct framing: the store cannot see crates report-rs does not depend on; "no
  MapLog call" is a dependency-scope fact (Cargo.toml), not a call-graph absence.
