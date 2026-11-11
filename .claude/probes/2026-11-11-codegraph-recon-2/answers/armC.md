# Arm C answers (Charon graph, report-rs 81705d9; scope S = lib target, default features, crate::app opaque)
Scope S, stated once: report_rs lib target only (tests/examples/benches NOT extracted); region crate::app opaque; features lance,iam,flow,oidc,projects,projects-pg off. Unread files per GAP: src/app.rs, src/app/plan_routes.rs, src/flow_scratch.rs, src/iam.rs, src/login.rs, src/main.rs, src/project_plan/*, src/workbook_delta/dt64/p21_lane_socket_lab.rs. External consumers of public items are UNOBSERVED. Harvest has 1 residual (HarvestIncomplete).

## Q1
a. Two types, identified by graph module paths:
 - report_rs::source_fake::WorkbookSource (non-generic; constructor `source_fake::WorkbookSource::new`)
 - report_rs::workbook_delta::source::WorkbookSource<T> (generic over a transport T; has new/into_transport/token/meta/poll)
 Defining file:line: UNKNOWN from the graph (it prints no def lines; I did not read source at unprinted locations). Files by module path: src/source_fake.rs and src/workbook_delta/source.rs. Witness inside the first: src/source_fake.rs:39 (the `fn pull` in the Source impl).
 CONFIRMED (identities), UNKNOWN (def lines).
b. source_fake::WorkbookSource implements Source (impl_Source_for_WorkbookSource::{id,name,lanes,pull} + vtable shim; src/source_fake.rs:39). The workbook_delta WorkbookSource<T> has NO impl_Source_for_* symbol in scope S: ABSENT-IN-SCOPE (scope S).
c. Edges (all CONFIRMED, scope S):
 - Ledger::pull -> Ledger::ingest: src/source.rs:334 (`self.ingest(batch)` on Ok).
 - Ledger::replay -> Ledger::ingest: src/source.rs:357.
 - projection::record_artifact -> Ledger::ingest: src/projection.rs:845, :852.
 - Projection::load_v1 -> Ledger::pull: src/projection.rs:408, :413, :416; load_v1 -> source_fake::WorkbookSource::new: src/projection.rs:409.
 - source_graph::pull_into -> Ledger::pull: src/source_graph.rs:357; load_v1 -> pull_into: src/projection.rs:429.
 - Source::pull for source_fake::WorkbookSource: only edge is its vtable method (src/source_fake.rs:39); callers are dynamic dispatch via Ledger::pull, reported UNKNOWN by the graph (trait-object call, not resolved to the impl).
 - IngestBatch: constructed/consumed by pull (return type), ingest, status/last_watermark/values closures and rules::source_basis closures; no separate "IngestBatch edge" witnessed beyond those symbols.
 - Separate, different ingest: workbook_delta::ingest::Workbook::ingest, which calls IdentityFrame::commit (src/workbook_delta/ingest.rs:264); IdentityFrame::new called from Workbook::open (src/workbook_delta/ingest.rs:117). This is NOT Ledger::ingest.
 CONFIRMED.
d. workbook_delta WorkbookSource<T>::poll has 0 callers in scope S (sites=0); its callees are meta/token/item_url/sha256_hex/status_class and ReqwestDrive::get (POSSIBLE). No edge from poll (or from WorkbookSource<T>::new, 0 refs) to Ledger::ingest or Workbook::ingest; Workbook::ingest also has 0 callers in S. Answer: ABSENT-IN-SCOPE for library code in S. Cannot say absent overall: poll/new/Workbook::ingest are PUBLIC (external callers UNOBSERVED), tests/examples not extracted, src/app.rs (opaque) and src/main.rs unread, and the p21 lab file unread, so a caller there is UNOBSERVED.
e. Scopes: as per header S for every "no caller/no edge" statement (Q1b, Q1d).

## Q2
Graph has no symbol containing authoriz, rbac, memberships, or lance_graph_rbac in the extracted scope, so no call to authorize_memberships / authorize_scoped appears: ABSENT-IN-SCOPE under scope S. Only unrelated names found (Ledger::register_scoped, Projection::scoped). The most likely home, src/iam.rs, is unread and feature iam is off, so under the `iam` feature (or in src/app.rs, opaque) the answer is UNOBSERVED. Witnesses: none (absence). Build configuration with a CONFIRMED call: none observed.

## Q3
Part 1, persistence through lance-graph-dt64-lance / MapLog: no symbol matching dt64_lance, lance_graph_dt64_lance, or MapLog: ABSENT-IN-SCOPE (scope S; features lance and others off, tests/examples not extracted, p21 lab file unread). Whether it happens under feature `lance` is UNOBSERVED.
Part 2, where the MapTable is held/advanced (CONFIRMED, scope S): lance_graph_dt64::dr::MapTable is held in the identity frame, workbook_delta::frame::IdentityFrame: MapTable::new(0,0) at src/workbook_delta/frame.rs:69 (via IdentityFrame::new, also from Default at :62), seal at src/workbook_delta/frame.rs:101, advance at src/workbook_delta/frame.rs:195 inside IdentityFrame::commit. commit is called by Workbook::ingest (src/workbook_delta/ingest.rs:264); Workbook::open builds the frame (src/workbook_delta/ingest.rs:117). The in-memory table is not persisted by any edge in S. Callers of Workbook::ingest: none in S (see Q1d), so the advance path is reached from library code only if an unobserved caller exists.
