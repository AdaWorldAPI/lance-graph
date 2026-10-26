# D-DT64-QB: Compatibility pointer and typed-predicate clarifications

**Status:** NON-AUTHORITATIVE ADDENDUM / docs only, 2026-10-26.

**Controlling source:** [D-DT64-Q0: source-attested binding → borrowed Quack execution](d-dt64-q0-borrowed-quack-binding-seam-v1.md). This pointer remains for previously posted EXO-50 and PR review links. Do not maintain a competing source-binding contract here. Changes to the common interface belong in the Q0 document.

## Clarifications for the eight first-lab catalogue candidates

The proposed DT64-1 slot labels are: 08 `MEMBER_OF_DIRECT`, 11 `SECURITY_GROUP`, 12 `MAIL_ENABLED_GROUP`, 13 `GROUP_GRANTS_ROLE`, 14 `EFFECTIVE_ROLE`, 17 `HAS_REMOTE_MAILBOX`, 18 `HAS_EXO_MAILBOX`, 32 `CATS_ENTRY_EMPLOYEE`. Numbers are NOT ratified production ABI.

Slots **11, 12 and 17 are unary classifications**, and must remain unary WHERE/filter predicates. Never manufacture a source→Boolean-object edge for their sake. Three directional `(s,p)→o`, `(p,o)→s`, `(s,o)→p` projections apply to **admitted typed binary relations**, not untyped arbitrary slots.

The owner-backed immutable in-memory borrow establishes the read window for a resident source; it does NOT imply a specific Lance dataset checkout. A persisted source requires an actual pinned physical reader separately from temporal `QueryReference::at`, and permission/coverage witnesses remain independent of both.

No code or runtime change is authorized by this note. For source and test gates see the authoritative Q0 document.
