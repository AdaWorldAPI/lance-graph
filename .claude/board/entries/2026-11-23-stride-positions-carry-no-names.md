# 2026-11-23 — Stride positions carry no names

**Status:** CORRECTION. No code changed.

The bytes of a stride are positions, like the bases of a gene. A base is not a
protein. Meaning appears only when a law reads the positions, the way a protein
appears only when a sequence is translated. Writing "byte `k` is the `part_of`
level", "offset 16 is the `part_of` axis" or "this byte is OU level 3" is the
same error as labelling a base with a protein name.

The stride is content-blind. Whether a byte reads as containment, subtype or a
DN level is decided by the law the class declares (ClassView / DT64). That
reading lives in the law, never in the bytes, and it may differ per class
over the same positions.

**What this corrects.** In the 2026-11-23 session I named positions:
`part_of` as the edge tenant's bytes, `is_a` as the key's bytes, OU levels as
byte indices. Those are readings a law may supply, not properties of the
facet. The algebra over the positions commits to no reading, positive or
negative.

OPEN: none.
