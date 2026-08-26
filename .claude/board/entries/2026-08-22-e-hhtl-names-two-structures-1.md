## 2026-08-22 — E-HHTL-NAMES-TWO-STRUCTURES-1 — "HHTL" is a homonym: the cascade key's `(X;Y)` tile and `NiblePath`'s subClassOf router are different objects, and conflating them produced five false structural claims in one session

**Status:** FINDING (the collision is documented on both sides; the *cost* is
measured on one session's output). **Confidence:** High — every claim below was
checked against the cited line by a council code-truth pass (inventory 17/17).

Two shipped structures wear the same name:

| "HHTL" as… | is | where |
|---|---|---|
| the cascade key's coordinate reading | `(X;Y)` per `(part_of:is_a)` tile — `part_of` = X = hi byte, `is_a` = Y = lo byte | `E-V3-BASINS-ARE-MEREOLOGY-NOT-LABELS` (:12055) |
| the Abstammung router | `NiblePath` — the `subClassOf` (P279) path, DOLCE basins 0..3, `mask-inherits-as-delta`, "walking DOWN the path is IS-A inheritance", ONE tree axis only | `hhtl.rs:1-38,56,176` |

**The first is an address; the second is a taxonomy walk.** A containment
hierarchy (book ⊃ chapter ⊃ verse) belongs to neither by default — it is
**mereology**, which `le-contract.md:56` carries in the *other byte of the same
pair*, and reading it onto the taxonomy axis is silently type-wrong.

**Measured cost.** A session addressing the KJV corpus used the name as though
it denoted one hierarchical path and produced five structural claims, every one
false: that HHTL is the Book·Chapter·Verse path; that Book·Chapter "lands
exactly" on HEEL (numerology over a self-chosen `ceil(log16 n)` carving); that
the translation lane lives in the classid; that a CAM-PQ code could ride in the
key tail; and that "MailboxId IS the NiblePath" is settled (it is flagged
DOC-ONLY, `tenants.md` §7.3). None survived a read of the definitions.

**The generalization:**

> A name that denotes two structures is not a documentation problem, it is a
> **type collision with no compiler behind it**. The two readings share a width
> (12 bytes) and a carving (`6×(8:8)`), so every syntactic check passes and only
> semantics disagree — which is precisely the class of error no gate catches.

Practical consequence: when a design sentence contains "HHTL", say **which** —
cascade-key `(X;Y)` or `NiblePath` — or the sentence is ambiguous. Nearest prior
art is a "nibble homonym" table (`handovers/2026-08-21-2200:110-114`) and a
traced collision section (`ATTENTION_MASK_AUDIT_2026_08_21.md:75`); neither
names this pair, which is why it is banked here.

Cross-ref: `E-V3-PART-OF-IS-A-TILE` (:12266), `E-FACET-8-8-ALWAYS` (:12088),
`knowledge/ast-as-partof-isa-address.md:23-28` (the 3/4/6 count distinction).

