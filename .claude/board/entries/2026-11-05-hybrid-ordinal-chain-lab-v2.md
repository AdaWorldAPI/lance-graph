# Hybrid AD/Entra/EXO ordinal chain lab, v2: no consumer lookup, DR frame witness

**Date:** 2026-11-05. **Status:** LAB (test-only). Follow-up to merged #1962; requires the paired OGAR change `ogar_dir_core::correspond::Output::entra_mailbox`.

## What changed

- OGAR `fold` already decided the sole Exchange Online row per Entra row and discarded it. `Output::entra_mailbox` keeps it (`NO_ROW` for none or several); `Ruler::from_fold` reads it instead of a second merge. Checked against a scalar scan on every `ogar-az` fixture; the duplicate and cross-tenant cases are pinned. The randomized fixture never generates a duplicate mailbox, so only the targeted test catches a first-match regression (disable-verified).
- The lab binds every ordinal from OGAR's output: no bind-time EXO scan remains.
- The lab-local value `Stamp` is replaced by `lance_graph_dt64::dr::FrameStore`/`Frame`. A frame cannot be built from numbers, so a foreign store or a re-seal at equal revisions is refused (the value stamp aliased them).

## Measured (release, two-row fixture; reported, not gated)

Cold prepare+bind ~12 µs. Warm read ~27 ns/property (frame check + 8 identity cross-checks). Typed Quack/mask-RISC sum ~122 ns/call. **0 heap allocations** on the hot read + fold path (thread-local counting allocator; disable-verified: a 1-byte allocation turns it red).

## Disable runs

Frame check off → 3 tests red. Identity cross-check off → the "owner forgot to advance" test red. Ambiguity: the cloud plane and the `entra_mailbox == NO_ROW` refusal each refuse alone; only both off turns 2 tests red (two independent guards, not a vacuous one).

## Still open

Cross-table value-derived reads stay outside the executor; the lab reads property lanes by bound ordinal in test code, not through an admitted Gather. The coordinate does not yet carry its DT64 class route (`RouteVersion`); FrameStore and BindStore run on separate clocks in this lab. The frame's revision numbers are caller-supplied, not derived from a source receipt.
