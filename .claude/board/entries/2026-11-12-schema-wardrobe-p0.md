# 2026-11-12 — D-SCHEMA-WARDROBE-P0: the resident-write chain exists in parts, not as one wired path

**Kind:** audit + falsifier suite. No production code changed except one `cfg(doctest)` module and one dev-dependency feature.
**Grade:** VERIFIED-IN-CODE for the call graph (every cited site read at its line);
MEASURED for the suite (`cargo test -p lance-graph-supervisor --features cycle-driver --test ownership_preservation`: 9 pass;
`cargo test -p cognitive-shader-driver --doc ownership_laws`: 7 pass; 4 of 4 disable runs red).
**Plan:** `.claude/plans/schema-wardrobe-ownership-audit-v1.md`. **Linear (existing):** EXO-253.

- `MailboxSoA::cast_on_behalf` stamps the owner's own id, and has no caller outside tests.
- `BatchWriter::cast` is public and takes a plain mailbox id; a payload-only cast under a foreign id is not refused anywhere. A cross-owner MOVE is refused in `persist_cycle`.
- The only non-test submitter to the background `CycleWriter` is the DR map log, which builds its slots itself and does not pass `MailboxSoA` or `BatchWriter`.
- The wired seal path carries owned payload bytes and copies them twice before the Lance append (retry copy, freeze). The descriptor route exists in examples and tests only.
- Three knowledge documents under `.claude/v3/knowledge/` describe callers and a sink that do not match the code; listed in the plan §5, not edited here.

- Second read (`mailbox_soa.rs`, `soa_envelope.rs`, `kanban_actor.rs`): the lawful resident write is `MailboxSoA::write_row` (cycle-gated); its mailbox arm in `backing.rs` is behind a default-off feature; no caller casts after writing; `SoaEnvelope::mailbox_owner()` is unwired; `kanban_actor.rs` is read-only.

**Agent runs (grindwork tier):** four read-only runs (two fact maps, one call-site census, one quote harvest) and one drafting run for the test file. Checked by reading: `cast_on_behalf` at its stated lines, the four non-test cast sites, `seal_cycle`, `persist_cycle`, `try_submit`; the drafted test file read in full before compiling. Agent items wrong: 0. One orchestrator error caught before commit: a truncated search made `cognitive_pass` look absent; reading `cycle_driver.rs` showed it exists and calls the adapter with a fleet-resolved owner id, and the audit was corrected. One map run stopped on a missing brief field and was re-dispatched.

**Open:** the `on_behalf` enforcement seam; descriptors in `collect_casts`; the owner of a resident codegraph population (BLOCKED); the stale documents.

- **Third version, after reading the sources in full (operator correction: the audit had been written from excerpts and search hits).** The grade line above overstated the evidence: "every cited site read at its line" was true of the cited sites and false of the files, and statements about whole files were wrong because of it. Read from first to last non-test line this time: `mailbox_soa.rs`, `backing.rs`, `engine_bridge.rs`, `soa_view.rs`, `soa_envelope.rs`, `batch_writer.rs`, `owner_adapter.rs`, `persist_sink.rs`, `cycle_driver.rs`, `cycle_writer.rs`, `kanban_actor.rs`, `cycle_sink.rs`, `lance-graph-dt64-lance/src/lib.rs`, and the three V3 knowledge documents. What changed (plan §11 lists every correction):
  - `write_row`'s two callers in `src/` have no caller of their own; there is no production write into a `MailboxSoA` row.
  - A step does not need a seal: an intent-only cycle steps under `CommitOutcome::NoChange`, and `try_advance_phase` is callable by any `&mut` holder. `current_cycle` and `mailbox_id` are public fields.
  - The durable image is keyed by row and written with owner 0; two owners on one row leave one payload.
  - `BatchWriter::resolve_owner` (the delegation cache) is never called by `cast`.
  - `LanceCycleWriter` refuses any artifact that is not 512 bytes.
  - The forgeable `on_behalf` and the missing production caller were already recorded in `batch_writer.rs`'s own module doc.
  - Suite: 11 runtime tests (two `finding_` tests added), 7 doctests.

- **Outer/inner boundary (plan §1c, §1d).** The sentence "a wardrobe cannot hand `LanceCycleWriter` arbitrary source rows" was a category error and is withdrawn: a wardrobe is outer and carries a class binding, never rows. After reading lance-graph-java's sources (the `lgj-abi` Rust modules, all Java main classes, the three consumers' main sources; `exports.rs` tests in part; test suites, bench, interop and the lab sources not read):
  - the outer side there holds schema vocabulary with no instances (`Trade`, `Orders`, `Edge`), three schemas name one lane, and no second population exists;
  - a handle is an integer too; it is provenance because the registry mints and resolves it. `BatchWriter::cast` resolves nothing;
  - no export in that repo writes a row. Its write side (stage, cross as descriptors, sole writer lands) is documented and not built, so it is no precedent for an outer producer reaching the inner owner.
