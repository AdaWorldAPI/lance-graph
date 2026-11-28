# 2026-11-28 — D-TGB-7 S4a: the receipt moves into `lance-graph-thought-git`; the probe holds only its pre-registration

**Status:** TEST-PINNED (`lance-graph-thought-git/tests/receipt.rs`, 15 tests moved unchanged; the pre-registered disable, captured receipts without their seed, turns the receipt tests red). Board: EXO-379, D-TGB-4.

`receipt.rs` (D-TGB-4) and its tests moved from `crates/thought-git-probe` behind the crate's `oracle` feature. Its durable side (`VersionStore`, `commit_cycle`, `MetadataTable`, `BaseRef`) is a SIMULATED boundary standing in for Lance and `LanceCycleWriter`, so the module stays a lab reference and is not promoted into execution. The probe now holds only `PREREGISTRATION.md` and the signal checks over the contract (10 tests).

Remaining for D-TGB-7: S4b, the landing through `WalSink` / `LanceCycleWriter` (waits on EXO-148: `DurableRef`, `ThoughtCommitId`; and on where the receipt's metadata row lives), S2b and the resident squash (wait on the W0 question on EXO-379).
