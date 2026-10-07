# 2026-10-07 — dir-sim: proxyAddresses as a relation; SMTP uniqueness over distinct owners

**Status:** TEST-PINNED (`crates/lance-graph-dir-sim/tests/proxies.rs`, 11 tests).

## DECISION

- **Shape.** `proxyAddresses` are rows of a relation, not a field of `DirRecord` or a node:
  - lanes: `owner` (`UserOrdinal`), `key` (`KeyId`, the normalized address), `value` (`ValueId`, the exact spelling, egress only), `meta`;
  - `meta` is a fixed-width code: kind (`smtp` / `x500` / `sip` / `other`) in bits 0..2, and the upper-case `SMTP:` primary flag in bit 2.
- **Identity.** The prefix is metadata and never identity: `SMTP:Alice@x.de` and `smtp:alice@X.DE` share one `KeyId`.
- **Invariant.** SMTP uniqueness counts distinct active owners per normalized SMTP `KeyId`, over primary and secondary addresses. It reuses the existing `GroupReduce Count`, over one row per `(owner, key)` (the `smtp_first` plane). Conflict iff `owner_count > 1`.
- **Overrides.** A primary-SMTP override replaces the owner's observed primary row and keeps its secondaries. An override onto the owner's own secondary counts that owner once.
- **SCOPE:** directory simulation, Exchange recipient addresses.
- **BASIS:** Exchange requires every SMTP proxy, not only the primary, to be unique across recipients.

## Gates

Disable runs, each red:
- every row counted, no distinct plane;
- primary rows only;
- an override that keeps the primary row;
- no own-secondary dedupe of overrides;
- the prefix included in the key;
- a case-insensitive primary marker;
- a primary-first run order;
- secondaries counted without the active gate.

## OPEN

- **Demotion is not modelled.** `SetPrimarySmtp` drops the old primary rather than demoting it to a secondary (Exchange does demote when address policies are off). It is chosen so that renaming away from a collision clears it.
- **No rule writes the relation.** No rule adds or removes secondary proxies; only the primary is overridable.
- **Kind uniqueness.** Only SMTP is checked. X500 and SIP uniqueness are not checked.
