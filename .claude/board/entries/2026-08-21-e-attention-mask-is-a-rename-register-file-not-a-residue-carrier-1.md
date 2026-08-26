## 2026-08-21 — E-ATTENTION-MASK-IS-A-RENAME-REGISTER-FILE-NOT-A-RESIDUE-CARRIER-1 — the fourth homonym collision of this arc, and the only one where the shipped type is COMPLETE for a different contract

**Status:** FINDING (D-ACR-0, report-only deliverable; every claim a read of
the file named or a stated grep). **Confidence:** High — the caller count is a
workspace-wide grep plus three sibling repos; the origin is the shipped type's
own originating plan, quoted.

`alpha-channel-rung-overlay-v1.md` §1 piece E graded
`cognitive-shader-driver/src/attention_mask.rs` + `attention_mask_actor.rs`
*"shipped; unaudited for this use"* — the use being the operator's
eye-tracking residue carrier. The audit
(`.claude/ATTENTION_MASK_AUDIT_2026_08_21.md`) returns **both** branches of the
falsifier at once, and the second is the load-bearing one.

**EXISTS-UNCALLED.** Three hits outside the two files, all non-consumers: two
`pub mod` lines in `lib.rs`, and `mailbox_soa.rs:11` — a doc comment stating
the *opposite* (*"wrap, **NO AttentionMask/LRU**, NO cross-cycle rollup"*).
`MedCare-rs` / `OGAR` / `ndarray`: 0 files each.

**And it is a different mechanism wearing the name.** The shipped type is a
complete implementation of `causaledge64-mailbox-rename-soa-v1.md` §4 —
*"AttentionMask SoA — the session-ephemeral **rename register file**"*: wide
identity (`u32` OGIT domain / `WitnessId` / `StyleId`) → scarce narrow slot
(5-bit G / 6-bit W / 8-bit style), LRU **because slots are scarce**. That is a
COMPRESSION concern. "Attention" there means *which identities are currently
resident in the slot file* — cache occupancy, not where the eye looked. A
residue carrier runs the opposite direction (visited address → recorded), is
keyed by the graph's own address rather than `MailboxId`, and is discardable
whole precisely because nothing is scarce.

Three properties settle it independently of provenance: keyed by `MailboxId`
(no `NodeGuid`/`NiblePath`/classid anywhere in the file); **not a mask** —
`Vec<AttentionMaskEntry>` with `.iter().find` / `.min_by_key` / `.filter().count`,
i.e. O(n) linear scan per operation, no bitset, no set algebra; and it records
occupancy, never a trajectory (`last_touched_cycle` is overwritten on each
`touch`, so the previous look is gone).

**Why this one is worse than the three homonyms before it.** §3a separated four
"witness" surfaces, §3k four "nibble" encodings, §3l three "hydration"
meanings — in each, the collision was between things that plainly did
different jobs once named. Here the shipped type is *finished and correct for
its own contract*, so it reads as available. The near-miss is the trap:
"shipped" invited reuse, and reuse would have folded a compression register
file into an attention overlay.

**Consequence, recorded so D-ACR-1 does not re-derive it:** piece E regrades
from *"shipped; unaudited for this use"* to ***"shipped for a DIFFERENT use;
uncalled; not a basis for piece D."*** §1's six-of-nine count is unchanged —
E existing was never the claim that E fits. `D-ACR-1` starts clean, which is a
better position than a partial fit; and the governing choice rules the shipped
shape out anyway, since a per-entry linear scan is the exact
*"64,000 objects and 64,000 crossings"* shape `lance-graph-java`'s
`Predicate.java` calls catastrophic, not a starting point that optimises into
`Mask × ClassView/WideFieldMask → Mask` compliance.

**Two dead surfaces measured in passing**, both matching shapes the
falsifiability rule already names: `plasticity_residual: u8` is declared and
initialised to `0` and **never read, never written non-zero** (two grep hits
total) — a field that only holds its zero value carries no information, the
`closed_class_guess` 150/150 shape; and `AttentionMaskMsg::BindReply` carries
three fields to a handler that returns `NoOp`.

**One superseded plan step, flagged rather than queued:** the originating §4
specifies a **singleton** actor (*"one global instance per session"*), and
`attention_mask_actor.rs:2` still calls the concrete ractor binding
*"sprint-12+ work"*. Finishing it as written would rebuild the singleton the
V3 mailbox ruling removed (`E-CE64-MB-4`, one-writer-per-mailbox). Treat that
line as superseded, not pending.

**A measured constraint this hands to D-ACR-1.** Its scope says *"composable
with `WideFieldMask`"*. Measured: `WideFieldMask` positions are `u8` —
universe capped at **256**, with a loud `UniverseExceedsSocCap` refusal above
it; its sibling `FieldMask` is `u64`/`MAX_FIELDS = 64` and **silently drops**
positions `>= 64`. A row population is neither. So "composable" cannot mean
"the same type", and §6 **Y2**'s parked `RowFocusMask × WideFieldMask` basis
collision is not a subtle semantic worry — it is a **cardinality mismatch with
a hard, loud cap on one side**. D-ACR-1 must state its basis and its
composition operator before composing, or it inherits one of two accidents:
loud refusal past 256 (borrowing the wide cap) or silent truncation past 64
(borrowing the narrow rule).
