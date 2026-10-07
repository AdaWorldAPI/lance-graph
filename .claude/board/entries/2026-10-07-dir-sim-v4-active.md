# 2026-10-07 — dir-sim V4: "active" is three-valued; unknown abstains

**Status:** TEST-PINNED (`crates/lance-graph-dir-sim/tests/active_v4.rs`, `tests/sim.rs::a_missing_user_account_control_is_unknown_not_enabled`; OGAR `ogar-dir-sim` `change::tests::effective_active_is_the_v4_table`).

## DECISION

- **Rule:** active iff at least one source is known and no known source says disabled. An unknown source abstains.
- **Expression:** `(ad_known ∨ entra_known) ∧ (¬ad_known ∨ ad_enabled) ∧ (¬entra_known ∨ entra_enabled)`.
- **The one definition** is `ogar_dir_sim::effective_active(ad, entra) -> Option<bool>`. Every executor must agree with it row for row.
- **SCOPE:** directory simulation (AD, Entra).
- **BASIS:** absence of evidence never disables; negative evidence always does; unknown/unknown stays unknown.

## What changed

- **OGAR.** `NodeState::active` is `Option<bool>`.
- **dir-sim, ingest.** `from_ad` reads a missing `userAccountControl` as `None` (it used to read it as enabled).
- **dir-sim, `Population`.** It gains `active_known`, the validity plane. `active` holds known-enabled rows only, so an unknown user is not an active user, and neither the uniqueness checks nor `ImplyGroup` count it.
- **dir-sim, Quack.** `effectively_active` / `effectively_inactive` are two plane filters that lower to mask-risc unchanged.

## Gates

- **Ingest:** a missing UAC stays `None` through the snapshot and `node_state`, and the user is outside the active plane.
- **Table:** the 3×3 table is pinned in OGAR, in both source orders. The Quack filters are checked against `effective_active` on all nine rows, with stale enabled bits both set and clear.
- **Disable runs, each red:**
  - `from_ad` back to unknown-as-enabled;
  - dropping the `known_any` term;
  - an enabled bit voting outside its known plane;
  - unknown users placed in the active plane;
  - (OGAR) unknown/unknown read as enabled;
  - (OGAR) negative evidence requiring both sources.

## OPEN

- **No AD↔Entra merge in any observation path.** An Entra user is a separate node, linked by `sync_edges` as evidence. A node therefore carries one source's flag until such a merge exists.
- **Uniqueness checks no longer count users of unknown status.** Before this change an unknown-UAC user counted as active, so it was included.
