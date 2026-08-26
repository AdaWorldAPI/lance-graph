## 2026-08-17 — E-IDENTITY-QUAD-4X24-RATIFIED-PERMANENT-1

**Status:** RULING `[operator]` — explicit, in-session decision, not a
code-derived inference. Closes `ISS-IDENTITY-QUAD-WIDE-CARVING-HOME`.

**Operator:** *"the 4x24 layout was already sanctioned and added as canonical
in lance-graph, look for quad 4x24"* — answering the three-option question
`ISS-IDENTITY-QUAD-WIDE-CARVING-HOME` raised (2026-08-06): ratify the
carving as permanent (option 1), mint a new sanctioned layout and migrate
(option 2), or reject it (option 3). The ruling is option 1.

**What changed.** `lance_graph_contract::identity_quad`'s 4×24 contiguous
payload — read/written through `legacy_outliers::LegacyOutlier::WideTriple`
(G2) — is now the **permanent, canonical home** for this identity tenant, not
a V1-migration waiting-room resident awaiting promotion. The module doc's own
"whether the carving deserves promotion … is an operator question, not one to
settle in code" sentence is now stale and has been replaced with the
ratification. `le-contract.md` §3a (which discourages G1–G3 wide carvings in
general, and says new classes must not be born into them) gained a one-line
cross-reference naming this tenant as the sanctioned exception, so a future
reader of §3a does not read the general discouragement as covering it.
`ISS-IDENTITY-QUAD-WIDE-CARVING-HOME` is closed with the same reasoning
inline.

**What did NOT change.** Option 2 (mint an L9-class layout, move the bit
primitives out of `legacy_outliers` into their own module) was explicitly not
taken — the code stays exactly where it was. The honest-cost accounting the
module doc already carried (no byte axis, not shift-addressable, `group_of`
does not apply) is unchanged: ratifying the carving as permanent settles
*where* it lives, not what it costs. `ISS-IDENTITY-CODEBOOK-ORDINAL-STABILITY`
and the Amendment's pinned-join-key-slot requirement are also unaffected in
substance — the join-key invariant now resolves *within* `identity_quad`
rather than in a hypothetical new layout module, since option 2 did not land.

**Also settled this session, recorded for completeness:** a separate question
about a term `WideClassView` (zero occurrences anywhere in this repo) was
raised and the operator's answer was that it was a typo with no referent —
dropped entirely, nothing to scope or build.

**What this unblocks.** A downstream consumer's identity-tenant work — the
join-key-slot pin specifically — was explicitly waiting on this ruling before
proceeding (per the Amendment in `ISS-IDENTITY-QUAD-WIDE-CARVING-HOME`, which
named the carving-home question as a precondition for pinning the join key).
That work is unblocked as of this entry.

