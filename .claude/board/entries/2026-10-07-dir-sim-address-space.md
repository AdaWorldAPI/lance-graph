# 2026-10-07 — dir-sim: UPN, SMTP, mail and routing are one address space; the routing address is the alias

**Status:** TEST-PINNED (`crates/lance-graph-dir-sim/tests/addresses.rs`, 8 tests).

## DECISION

- **Vocabulary (OGAR #321).** `Violation::AddressConflict { key, holders: [(node, AddressRole)] }`, `RoutingNotInProxies`, `RoutingMismatch`; `exchange::ROUTING` = `{alias}@{tenant}.mail.onmicrosoft.com`, alias = `mailNickname`.
- **One key space.** A UPN or SMTP address that is another object's SMTP, `mail` or routing address conflicts; so does a `mail` held by another object (an admin account whose `mail` is a password-reset target), whatever that object's enabled flag or recipient. Two holders that already collide as SMTP or as UPN are not reported again.
- **Routing.** A live remote mailbox's routing address is its own alias at the tenant, and one of its own SMTP addresses.
- **SCOPE:** directory simulation. **BASIS:** Exchange refuses an address another object holds, under any attribute; `Enable-RemoteMailbox` stamps `{alias}@{tenant}.mail.onmicrosoft.com` as both `targetAddress` and an `smtp:` proxy.

Partly closes the OPEN item "the routing address is not added to the proxy relation" of `2026-10-07-dir-sim-exchange-recipients.md`: a routing address held only in `targetAddress` now conflicts (`AddressConflict`, role `Routing`) and is reported (`RoutingNotInProxies`). The proxy rule covers only the routing address a node was **observed** with: one a version introduces (an enable, a create, a new routing address) is stamped by the operation, so the pre-actuation version passes; it stays in the address space as `Routing`. Found by Codex review of #1392: the enable path could never pass `promote_desired`, and the recipients fixture hid it by preloading the proxy.

## What changed

- **Ingest.** `ObservedNode` gains `mail` and `alias` (`mailNickname`); `from_ad` reads both; the snapshot keeps their comparison keys.
- **String fence.** Validation reads no text: `Dicts::intern` parses each value once against `ROUTING` and records the alias key (`routing_alias`).
- **Validation.** `validate::address_rules` collects `(key, holder, role)` rows (UPN, primary, secondary, routing for address owners; `mail` for every user), sorts them and reports per key. `O(n log n)` over the users — a validation pass, not the simulation hot path.

## Gates

Disable runs, each red (see the PR).

## OPEN

- `mail` and `mailNickname` cannot change in a simulation (no `Attribute` for them), and a created node has neither.
- Groups' `mail` and addresses are not part of the space (the population carries users' addresses only).
- One tenant per directory is not checked.
