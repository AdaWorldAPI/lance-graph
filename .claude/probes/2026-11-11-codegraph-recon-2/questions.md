# Questions (report-rs at /home/user/rr-81705d9, commit 81705d9, default cargo features)

Q1. The crate defines two types named `WorkbookSource`.
  a. Give the fully qualified identity and defining file:line of each.
  b. Which of them implements `report_rs::source::Source`?
  c. Trace `poll`, `Source::pull`, `IngestBatch` and `Ledger::ingest`: which edges exist, with file:line witnesses?
  d. Does the workbook_delta (SharePoint drive) source reach `Ledger::ingest` from library code? Who calls its `poll`?
  e. For every "no edge / no caller" statement, state the scope you examined (files, targets, features).
Q2. Does report-rs call `lance_graph_rbac::authorize::authorize_memberships` or `authorize_scoped`? Under which build configuration? Give witnesses and scope.
Q3. Does report-rs persist DR identity through `lance-graph-dt64-lance` (`MapLog`)? Where is the identity `MapTable` held and advanced? Give witnesses and scope.

Answer format: for each sub-question, a short statement, then CONFIRMED / ABSENT-IN-SCOPE / UNOBSERVED, then witnesses (file:line), then scope. Say UNKNOWN rather than guess.
