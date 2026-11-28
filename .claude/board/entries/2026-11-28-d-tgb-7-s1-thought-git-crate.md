# 2026-11-28 — D-TGB-7 S1: `lance-graph-thought-git` exists; the slot pool, oracle and fixtures moved in

**Status:** TEST-PINNED (`crates/lance-graph-thought-git/tests/{slots,oracle}.rs`, 16 tests moved unchanged; 2 disables red after the move: slot handles without the generation check, a missing cell read as 0). Board: EXO-379 (inventory), EXO-489.

The crate is the D-TGB-7 implementation crate named in `thought-git-lifecycle-contract-v1.md` §6. S1 moves code, it copies nothing: `slots.rs`, `oracle.rs`, `fixtures.rs` and their tests left `crates/thought-git-probe`, which now depends on the crate for them (its other 42 tests unchanged and green). The oracle and fixtures sit behind the feature `oracle`, off by default: they are the independent reference, not execution; the crate's tests enable it through a self dev-dependency, so they cannot silently skip.

Not moved in S1, by design: the resident spaces (`ThoughtTables`, givens) still live in `cognitive-shader-driver/examples` (S2); the squash gate and protected readers run on `BTreeMap` lanes, not the resident deltas (S3); the receipt's durable half is a simulated `VersionStore` and goes onto `WalSink` / `LanceCycleWriter` (S4). `AlphaTunnel`'s ten rung lanes are superseded by 1+63 and are not migrated.

OPEN (owner): one `(slot, generation)` type (probe `Slot` vs planner `nars::warrant::ThoughtRef`); EXO-148 placement of `DurableRef` / `ThoughtCommitId`; whether the receipt's metadata row rides in the cycle commit or a separate table.
