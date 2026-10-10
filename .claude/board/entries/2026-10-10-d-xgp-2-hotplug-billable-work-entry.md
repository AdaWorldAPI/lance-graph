# 2026-10-10 — D-XGP-2 re-scoped: lockstep deprecated; the 0x0103 basis is OGAR's promoted class

**Status:** MEASURED (read-only). Plan: `.claude/plans/cross-glove-business-parity-v1.md` §C.9.

- Lockstep (paired OGAR + contract-mirror allocation) is deprecated; consumers declare `HotPlug` / `RbacPlug` and `OgarAuthority` resolves them (`lance-graph-contract/src/hotplug.rs`, `lance-graph-ogar/src/lib.rs:535`). The app prefix is not part of a plug.
- The canonical basis for `0x0103` already exists: `ogar_vocab::billable_work_entry()` (`lib.rs:3515`), `billable` + 12 family edges, lifted by `ogar-class-view`. Nothing to mint.
- `BillableWorkEntry` has no temporal role, so the date half of probe Q1 has no canonical field.
- No capability table covers `0x0103` (`capability_registry.rs:193-209`): a plug on it activates as `NoCapabilitiesFor`.

OPEN: whether `BillableWorkEntry` gains a temporal role (OGAR authority change).
