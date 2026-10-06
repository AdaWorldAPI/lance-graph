# 2026-10-06 — D-MBX-CUTOVER-0: every ShaderDriver → BindSpace edge, classified; mailbox drivers cut over

## VERIFIED-IN-CODE — classification (as of `main` @ 2a701596, before this change)

Construction sites: only this crate (`bin/serve.rs`, `bin/grpc.rs`, tests,
the StreamDto probe). No other crate builds a `ShaderDriver`.

| edge | class | evidence | after |
|---|---|---|---|
| `backing()` Singleton arm, no mailbox registered | 1 production state | default build's only substrate (lab bins) | unchanged |
| `backing()` reads singleton when no mailbox sits under id 0 | 3 compatibility fallback, **silent** | `mailboxes.get(&DEFAULT_MAILBOX)` else `Singleton` | removed: `selected` is set whenever a mailbox is registered |
| `self.bindspace.ontology()` in `run()` (`ctx_id`) | 2 immutable shared resource | the only BindSpace read inside `run()` in mailbox mode; zero production `set_ontology` callers, so dormant | driver-owned `ontology`, builder `.ontology()` |
| `row_count` / `byte_footprint` | 5 metrics fossil, **wrong in mailbox mode** | read `self.bindspace` regardless of arm; surfaced by lab `/status` and grpc | report `backing()` (`BackingStore::len`, new `MailboxSoA::byte_footprint`) |
| builder `.expect("bindspace required")` | 3 compatibility, forced a dummy | only reason the probe built `BindSpace::zeros(0)` | required only without a mailbox |
| `ShaderDriver::bindspace()` + field use in `serve.rs` / `grpc.rs` (ingest via `Arc::get_mut`, qualia read) | 3 compatibility (lab) | lab binaries always build a singleton | kept, now `Option` |
| `engine_bridge::{ingest_codebook_indices, read_qualia_decomposed}` | 3 compatibility (lab) | called from `serve.rs` / `grpc.rs` | kept |
| `engine_bridge::{dispatch_busdto, unbind_busdto, persist_cycle}` | 4 test oracle | callers: `busdto_bridge_test.rs`, `end_to_end.rs`, engine_bridge unit tests only | kept |
| `BackingStoreWrite::Singleton`, `w2_differential` singleton arm | 4 test oracle | W2 harness only; `run()` writes nothing | kept |
| `BackingStore::len` `#[allow(dead_code)]` | 6 dead code | comment "row_count routes through bindspace until W4b" | now live (row_count) |
| module/struct docs "Holds BindSpace columns (owned)"; `ShaderDriver::new(Arc<BindSpace>)` | 5 docs fossil / 3 compat constructor | | docs updated; constructor kept for singleton |

**Is OntologyRegistry the only live mailbox-mode dependency?** Inside
`run()`, yes: every column read goes through `BackingStore::Mailbox`, which
holds only `&MailboxSoA`. In mailbox mode overall, no: `row_count` /
`byte_footprint` read the singleton, the builder demanded one, and
`backing()` silently read it for any mailbox not under id 0.

## DECISION — smallest cutover

- `ShaderDriver.bindspace: Option<Arc<BindSpace>>`; mailbox drivers build
  without one.
- `selected: Option<MailboxId>` (feature-gated): set at build (the sole
  mailbox, or `.select_mailbox(id)`; several without a selection refuse to
  build); `select_mailbox` accepts only registered ids; mailboxes are never
  removed. `backing()` has no fallback.
- `with_mailbox` / `mailbox` / `mailbox_mut` are feature-gated, so a default
  build cannot register a mailbox it would not read.
- Ontology on the driver; a mailbox driver takes it only from `.ontology()`.
- No temporal change: `tick` / `current_cycle` stay on `MailboxSoA`.

SCOPE: `ShaderDispatch` carries no `MailboxId`; selection is driver state
(`select_mailbox`). REVISIT WHEN: W5 routes per dispatch, or W7 deletes
BindSpace and the lab surfaces.

## MEASURED

`tests/mailbox_cutover.rs`, 8 tests (C1–C6 plus two refusals), 8 disable
runs red: id-0 fallback, prefer an attached BindSpace, inherit ontology
from it, row count from it, zero footprint, unchecked selection, ignored
selection, silent pick of the first mailbox. Default, `mailbox-thoughtspace`,
`with-engine` and `lab` builds compile; lab clippy carries 10 pre-existing
errors, unchanged.

## OPEN

- Per-dispatch mailbox routing (W5) is not attempted.
- Lab ingest still writes the singleton only.
