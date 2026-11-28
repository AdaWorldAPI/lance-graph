# 2026-11-28 — D-TGB-7 S3a: the squash gate and protected readers move into `lance-graph-thought-git`

**Status:** TEST-PINNED (`lance-graph-thought-git/tests/{squash_gate,protected_readers}.rs`, 17 tests moved unchanged; disable red after the move: divergence refusal off → 1 red). Board: EXO-379, D-TGA-5, D-TGA-5D.

`squash.rs` (D-TGA-5 gate, D-TGA-5D protected readers) and its two test files moved from `crates/thought-git-probe` behind the crate's `oracle` feature. They run over the oracle's `Thought` chains (`BTreeMap` lanes), so they are the reference a squash over the resident deltas is checked against, not execution. The probe keeps `receipt.rs` and the signal checks.

OPEN (owner): the givens rule over the resident spaces. `ThoughtTables` lets a hypothesis overwrite the borrowed base (its tests rely on it), while the givens probe hydrates a separate W0 givens table no hypothesis may change. Whether W0 is the borrowed base or a hydrated givens table over it decides how R1 (contradiction with a given) applies to a resident squash.
