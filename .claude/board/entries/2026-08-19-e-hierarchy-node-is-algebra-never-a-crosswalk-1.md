## 2026-08-19 — E-HIERARCHY-NODE-IS-ALGEBRA-NEVER-A-CROSSWALK-1

**Status:** RULING `[operator]`, correction applied in place to
`docs/architecture/ARC-A-SOURCE-ARCHAEOLOGY.md` §3/§7/§9/§10 (⊘-marked,
original text retained, not deleted).

**The correction, in one line:** `parent_ref`, `child_or_ref_set`,
`projection_mask`, `version` must never be co-located as stored fields on
a materialized hierarchy node. Each is a cheap, register-width AND/XOR
read over an EXISTING shipped primitive — `NiblePath::parent`/`child`
(already O(1) bit-shift), a `ClassView`-projected child-position mask
ANDed against a `WideFieldMask` presence read (the same intersection
primitive `standing_mask.rs::fires()` already exercises), and a
caller-supplied `DatasetVersion`/`QueryReference` (never copied onto a
node). The "hierarchy node" at any address is a COMPUTED VIEW over
`(NiblePath, ClassView, WideFieldMask, DatasetVersion)`, evaluated fresh
each time — never a persisted crosswalk table.

**The named analogy, load-bearing:** this is exactly how Valhalla value
classes get structural operations (equality, hashCode, field access) with
zero object header and zero indirection — flat bytes, computed directly,
no materialized identity — and exactly how Panama's `MemorySegment` gives
zero-copy VIEWS into native memory with zero Java-object materialization.
The prior ARC A draft proposed a new type carrying all four fields,
reasoning "no existing type carries all four, therefore mint one" — that
reasoning was the error. Absence of a four-field type is not evidence one
is needed; per this ruling it is evidence none should exist.

**Consequence:** ARC A's §7 "proposed new types" list is now EMPTY — zero
new types across the entire dumb-storage substrate. At most one small free
function (composing `NiblePath::child` + a `ClassView`-projected mask +
plain AND) is proposed, and even its placement (beside `ClassView`,
beside `RailGraph`, or pure call-site composition) is left as a narrowed
ratification question, not a design decision made here.

**Cross-refs:** `docs/architecture/DUMB-STORAGE-RESET-CHARTER.md` §1
(the substrate must know only references/hierarchy/ClassView/
WideFieldMask/DatasetVersion — this ruling is that principle applied to
the ONE gap ARC A found); `E-ARCHITECTURE-RESET-DUMB-STORAGE-HHTL-
EPISTEMIC-1` (the parent ruling this session).

