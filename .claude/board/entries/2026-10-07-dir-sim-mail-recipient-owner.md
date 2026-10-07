# 2026-10-07 — dir-sim: a mail recipient resolves to one directory owner by key, or to a conflict

**Status:** TEST-PINNED (`crates/lance-graph-dir-sim/tests/mail_recipient.rs`, 9 tests).

## MEASURED

The seam between mail and directory already existed except for one function. Inventory on current `main` of all four repositories:

- **Stalwart** hands a recipient over as text: `SessionAddress { address, address_lcase, domain, … }` (`smtp/src/inbound/rcpt.rs`), queued as `Recipient { address: Box<str> }`. No reference to Spear or Lance.
- **Spear** has no MIME or address parser and no recipient type: addresses are raw `Utf8` / `List<Utf8>` Lance columns (`from_addr`, `to_addrs`, …). No interned handle. `SpearBridge` is a deprecated OGIT namespace scope lock, not an address identity.
- **OGAR** defines the vocabulary (`ValueId`, `KeyId`, `normalize`, `AddressRole`, `Violation::AddressConflict`); it does not intern.
- **lance-graph** interns at ingress (`Dicts::intern`: `ValueId` → `KeyId` once) and already has a counted, never-minting ingress lookup for foreign text (`Dicts::key_lookup`). Missing: an answer to "which object holds key K across every address surface"; it existed only inside `address_rules`, as violations.

## DECISION

- `validate::address_owner(view, KeyId) -> Result<Option<Guid128>, Violation>`: `Ok(None)` no holder, `Ok(Some(owner))` exactly one (any number of roles), `Err(AddressConflict { key, holders })` two or more. No holder is ever chosen.
- It reads the same rows as `address_rules` (factored into `address_rows`), so resolution and validation cannot disagree.
- A same-attribute collision (`DuplicateSmtp` / `DuplicateUpn` to validation) is also `Err` here: either way the key names no single recipient.
- **SCOPE:** directory simulation. **BASIS:** the mail side needs a canonical identity, not a parallel one; `KeyId` is it.

## Gates

Disable runs, each red: first match wins; key ignored; `mail` dropped from the space; text work during resolution (caught by `DictCounters`, not by `string_fence`, since `key_label` has no banned token); deleted nodes still hold; ingress lookup not normalized.

## OPEN

- Spear has no step that produces a recipient: binding its `to_addrs` to a `KeyId` needs a Spear → lance-graph dependency, and Spear's `ontology` feature still pins `lancedb =0.27.2` against lance 12 on lance-graph `main`.
- Groups' addresses are not in the space, so mail to a distribution list resolves to no owner.
