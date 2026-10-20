# 2026-10-20 — D-TGA-SLOT-AXIS — thoughts are 64 slots, and the slot number is CE64's witness pointer

**Status:** DECISION (operator, 2026-10-20). **Basis:** VERIFIED-IN-CODE (`causal-edge/src/layout.rs:53`, `W_SHIFT = 53`, 6-bit `w_slot`).

- Thinking with a git API, at the cost of a sparse write and a merge on read.
- The axis is 64 dynamically allocated thought slots: 0 is main (the merged state the frame starts from), 1..=63 are any thought. Rung is a claim attribute, not an axis; `alpha_tunnel.rs`'s ten rung lanes are superseded. This closes the "10 rung lanes, not a 64-lane budget" conflict in `2026-10-12-ce64-hypothesis-layers-plateau.md`.
- 64 is the width of CE64 bits 53..58: a thought's slot number travels in every causal edge, so thoughts can reference each other.
- At the ceiling a new thought requires squash or land, never a 65th slot. `land` rewrites pointers to the landed slot to 0; `squash_into` rewrites them to the surviving slot; every pointer is qualified by the frame's base version (0) or the slot's generation (1..=63); `ThoughtCommitId` is never reused.
- Written into `thought-git-api-v1.md` § "Correction: the slot axis and the witness pointer", with two new D-TGA-1 falsifiers.

OPEN: the squash-or-land policy under slot pressure; the `edge.rs` "0 = no corpus anchor" doc line, to correct with D-TGA-1.
