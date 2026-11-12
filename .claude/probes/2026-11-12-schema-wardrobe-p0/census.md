# Call-site census (crates/**/*.rs, 2026-11-12, main at e21ae1586 + #2042)

Counts are lines matching the pattern; the audit's conclusions rest on the
sites read at their lines, not on these counts.

| pattern | hits | outside tests / examples / `#[cfg(test)]` |
|---|---|---|
| `cast_on_behalf` | 11 (5 before this PR's suite) | 1 (the definition, `mailbox_soa.rs`) |
| `.cast(` on a `BatchWriter`, in `src/` outside test modules | 3 | `mailbox_soa.rs` (inside `cast_on_behalf`), `owner_adapter.rs` (`emit_bootstrap_intent`), `cycle_driver.rs` (`restage_held`) |
| `BatchWriter::new(` | 48 | 0 |
| `.try_submit(` | 31 | 1 (`lance-graph-dt64-lance/src/lib.rs`, `MapLog::submit`) |
| `impl MailboxSoaOwner for` | 7 | 2 production types (`MailboxSoA`, `SymbiontBoard`); the rest are fakes and one probe |
| `impl WalSink for` | 17 | 2 (`LanceCycleWriter`, codegraph `LabWal`) |
| `emit_bootstrap_intent(` call lines | 20 | 1 (`cycle_driver.rs`, inside private `cognitive_pass`); the rest are examples, tests and the adapter's own tests |
| `run_cognitive_work*` / `shade_owner` callers outside `cycle_driver.rs` | 12 | 0 (four supervisor test files) |
| `SoAMailbox` / `soa_mailbox` | 0 in code | — |
