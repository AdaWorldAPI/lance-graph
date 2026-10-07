# 2026-10-07 — dir-sim: Exchange recipients are simulated state; uniqueness counts recipients, not the enabled flag

**Status:** TEST-PINNED (`crates/lance-graph-dir-sim/tests/recipients.rs`, 9 tests; `tests/alloc.rs` covers `SetRecipient`).

## DECISION

- **Vocabulary (OGAR #320).** `NodeState.recipient: Option<Recipient>` (`None` = not read) and `Change::SetRecipient`, compare-and-set through `NodeState::apply`, lowered to one `RemoteMailboxOp` lifecycle step.
- **Address owner = enabled OR a live mail recipient.** A shared, room or equipment mailbox is a disabled account and still owns its addresses. The union never drops a previously counted owner; a recipient not read falls back to the enabled flag (V4).
- **SCOPE:** directory simulation.
- **BASIS:** Exchange refuses a duplicate proxy address whatever the account's `userAccountControl` says.

Closes the OPEN item "Uniqueness counts only active owners" of `2026-10-07-dir-sim-node-properties.md`.

## What changed

- **Ingest.** `observe::from_ad` reads the msExch triplet and `targetAddress` raw into `ObservedRecipient`, stripping `SMTP:` (in). Schema ≥ 2 = read (absent attributes = not mail-enabled); older = not read.
- **Snapshot.** Raw lanes `rcp_rrt` / `rcp_display` / `rcp_details` / `rcp_target`, presence bits, a `rcp_read` plane, decoded on read by `Recipient::from_attributes` (strict; else `Other`). A base `owner` plane.
- **Overlay.** `ordinal → Option<Recipient>` overrides (net effect, dropped on delete), a created-node lane, an uninterned routing address refused.
- **diff / plan / reconcile** carry `SetRecipient`; uniqueness (`duplicates`, `smtp_duplicates`) reads `owner_users` instead of `active_users`.

## Gates

Disable runs, each red: owner = active only; owner ignores active; base owner ignores recipient overrides; `node_state` ignores the overlay; `diff_shared` misses recipient overrides; reconcile drops `SetRecipient`; "not read" collapsed to not mail-enabled; `SMTP:` kept; uninterned routing accepted; delete keeps the override; net effect off.

## OPEN

- **The routing address is not added to the proxy relation.** `Enable-RemoteMailbox` usually stamps it as an `smtp:` proxy too, so it is already counted when observed that way; a routing address held only in `targetAddress` is not.
- **Groups and contacts as recipients** are not modelled (the triplet decodes users' remote mailboxes).
- **Hybrid identity matching** (`mS-DS-ConsistencyGuid`, `msDS-ExternalDirectoryObjectId`, both 128-bit ids in OGAR #320) is not yet used to join on-premises and cloud observations.
