# Arm M answers (report-rs @ /home/user/rr-81705d9)

Scope notes: default features = `default = []` (Cargo.toml `[features]`). Searched all `*.rs` under src/, tests/, examples/ (and Cargo.toml, Cargo.lock, Dockerfiles, docs/*.md for Q2/Q3). Only lib crate (src/lib.rs) and bin src/main.rs exist (no src/bin/); no workspace members.

## Q1
a. Two `WorkbookSource` types:
 - `report_rs::source_fake::WorkbookSource` — src/source_fake.rs:19 (`pub struct WorkbookSource { fixture: Fixture }`).
 - `report_rs::workbook_delta::source::WorkbookSource<T: DriveTransport>` — src/workbook_delta/source.rs:210 (`pub struct WorkbookSource<T: DriveTransport> { cfg, transport }`). Module declared at src/lib.rs:48 (`pub mod workbook_delta;`), `pub mod source;` in src/workbook_delta/mod.rs.
 CONFIRMED.

b. Only `source_fake::WorkbookSource` implements `report_rs::source::Source`: src/source_fake.rs:29 `impl Source for WorkbookSource`. The trait is src/source.rs:171. The workbook_delta type has only inherent impls (src/workbook_delta/source.rs:215 `impl<T: DriveTransport> WorkbookSource<T>`); grep of "Source for" / "IngestBatch" / "Ledger" across src/workbook_delta/** found no `Source` trait impl (only an unrelated `LaneSource` in src/workbook_delta/dt64/p21_lane_socket_lab.rs:188 and dt64.rs). CONFIRMED (the delta type is not a `Source`); scope: src/workbook_delta/*.rs, src/workbook_delta/dt64/*.rs, all of src/.

c. Edges:
 - `Source::pull` declared src/source.rs:180 (`fn pull(&mut self, since: Option<Watermark>, now_ms: i64) -> Result<IngestBatch, SourceError>`); returns `IngestBatch` (struct src/source.rs:135). Implemented at src/source_fake.rs:39 (workbook fake; builds `IngestBatch{..}`), 107, 188 (other fakes), src/source_graph.rs:257.
 - `Ledger` struct src/source.rs:259; `Ledger::ingest(&mut self, batch: IngestBatch)` src/source.rs:302.
 - `Ledger::pull(&mut self, src: &mut dyn Source, now_ms)` src/source.rs:330 calls `src.pull(since, now_ms)` (333) and on Ok `self.ingest(batch)` (334). This is the pull->IngestBatch->ingest edge.
 - Callers of `Ledger::pull`: src/projection.rs:409 (`.pull(&mut WorkbookSource::new(base.clone()), now_ms)` using source_fake::WorkbookSource, import at projection.rs:10), 414, 417 (fake sharepoint/spear); src/source_graph.rs:357 (`ledger.pull(src, now_ms)` in `pull_into`); many tests (e.g. tests/p2_scope.rs:32).
 - Other direct `Ledger::ingest` callers: src/projection.rs:845, 852 (mapping state batches); src/source.rs:357. (src/gov/mod.rs:48 / src/gov/observation.rs:263 are a different `ingest` on a different gov ledger type with `LicSnapshot`, not `source::Ledger`.)
 - `poll`: only defined at src/workbook_delta/source.rs:292 (`WorkbookSource<T>::poll(&mut self, last_etag, now_ms) -> Result<Poll, PollError>`); it returns `Poll`, not `IngestBatch`.
 - Delta-side landing is `workbook_delta::ingest::Workbook::ingest(&mut self, snap: &VerifiedSnapshot) -> Outcome` (src/workbook_delta/ingest.rs:161), a distinct method, not `Ledger::ingest`.
 CONFIRMED.

d. Does workbook_delta source reach `Ledger::ingest` from library code? No edge found. Callers of the delta `poll`: examples/workbook_sync.rs:37 (`src.poll(wb.last_etag(), now)`, then `wb.ingest(&snap)` at :46 which is `Workbook::ingest`), tests/p20_workbook_delta.rs:304,311,331,350. No call of `.poll(` anywhere in src/ outside its definition. Nothing in src/ outside src/workbook_delta references `workbook_delta` except `pub mod workbook_delta;` (src/lib.rs:48), and nothing in src/workbook_delta references `IngestBatch`/`source::Ledger`. So library code never calls delta `poll` and the delta source never reaches `source::Ledger::ingest`. ABSENT-IN-SCOPE.

e. Scope for each "no edge / no caller": grep `-rn "poll("`, `"IngestBatch"` (files containing it: src/source_graph.rs, source_fake.rs, projection.rs, source.rs, tests/p3_evidence.rs, tests/p3b_delta.rs), `"workbook_delta"`, `"Workbook::open"` over all *.rs in src/, tests/, examples/ (src/main.rs, src/app.rs, src/app/** included); features: default (none) — and the grep is feature-agnostic (cfg-gated code is still text-searched, none of these hits are cfg-gated). Not examined: external crates (git deps), dynamic dispatch/macros beyond text grep. A grep with no hits is navigation only; I opened src/workbook_delta/mod.rs, source.rs (210-300), ingest.rs (1-125,150-165,260) to confirm the doc says the delta source "returns a verified snapshot" landed via `Workbook::ingest`.

## Q2
Yes, but only when feature `iam` is on. `authorize_memberships`: src/iam.rs:21 `use lance_graph_rbac::authorize::authorize_memberships;` and call at src/iam.rs:250 inside `LanceGraphScope::decide_in` (impl at src/iam.rs:243). `authorize_scoped`: no occurrence anywhere (grep of whole repo, *.rs/*.toml/*.md): ABSENT.
Build gating: `pub mod iam;` is `#[cfg(feature = "iam")]` (src/lib.rs:22-23); `lance-graph-rbac` is `optional = true` and enabled only via `iam = ["dep:lance-graph-rbac", ...]`; `lance = ["iam","oidc"]` (Cargo.toml [features]). Reach from app: src/scope.rs:563-566 `authority()` under `#[cfg(feature = "iam")]` constructs `crate::iam::LanceGraphScope::new` (src/scope.rs:566); `#[cfg(not(feature="iam"))]` arm (568) uses `StaticScope`. `authority` is called at src/app.rs:562, 752 etc. Dockerfile:36 builds `--features lance`; Dockerfile.standalone:18 has no rbac.
- authorize_memberships: CONFIRMED under `--features iam` (or `lance`); ABSENT in default-feature build (module not compiled).
- authorize_scoped: ABSENT-IN-SCOPE (all *.rs, Cargo.toml, docs, tests, examples searched; features irrelevant as no text hit).
Also tests/p33_intake.rs:263 only mentions lance_graph_rbac in a test name.

## Q3
No. There is no use of `lance-graph-dt64-lance` or `MapLog` anywhere: grep for "dt64-lance|dt64_lance|MapLog" over *.rs, *.toml, *.md, Dockerfiles, Cargo.lock found nothing (Cargo.toml dependencies list only `lance-graph-dt64` (git, main), not a `-lance` crate; Cargo.lock has `lance-graph-dt64` at line 1366 only). ABSENT-IN-SCOPE (all features; text-wide).
Identity `MapTable` (from `lance_graph_dt64::dr`) is held in memory in `IdentityFrame { table: MapTable, tiles }` (src/workbook_delta/frame.rs:55-57, constructed `MapTable::new(0, 0)` at :69) and advanced by `IdentityFrame::commit` calling `self.table.advance(source, 0, ...)` (src/workbook_delta/frame.rs:189-203). `IdentityFrame` is a field of `workbook_delta::ingest::Workbook` (src/workbook_delta/ingest.rs:99, built at :117 in `Workbook::open`); advance occurs at ingest.rs:264 `self.frame.commit(u64::from(number), fplan)`. Planning (`plan`, frame.rs:131) advances nothing. Persistence explicitly absent: src/workbook_delta/ingest.rs:11-16 ("Persistence is not here ... It writes no file and opens no dataset"). `Workbook::open` callers are only examples/ and tests/ (examples/workbook_sync.rs:26, workbook_head_to_head.rs:183, workbook_bench.rs:21, tests/p20_workbook_delta.rs:47,635,651); no src/ caller outside the module. CONFIRMED (in-memory only) / persistence ABSENT-IN-SCOPE.
