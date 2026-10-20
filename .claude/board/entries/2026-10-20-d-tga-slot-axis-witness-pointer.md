# 2026-10-20 — D-TGA-SLOT-AXIS — thoughts are 64 slots, and the slot number is CE64's witness pointer

**Status:** DECISION (operator, 2026-10-20). **Basis:** VERIFIED-IN-CODE (`causal-edge/src/layout.rs`, const `W_SHIFT`, value 53; 6-bit `CausalEdge64::w_slot`).

- Thinking with a git API, at the cost of a sparse write and a merge on read.
- The axis is 64 dynamically allocated thought slots: 0 is main (the merged state the frame starts from), 1..=63 are any thought. Rung is a claim attribute, not an axis; `alpha_tunnel.rs`'s ten rung lanes are superseded. This closes the "10 rung lanes, not a 64-lane budget" conflict in `2026-10-12-ce64-hypothesis-layers-plateau.md`.
- 64 is the width of CE64 bits 53..58: a thought's slot number travels in every causal edge, so thoughts can reference each other.
- At the ceiling a new thought requires squash or land, never a 65th slot. `land` rewrites pointers to the landed slot to 0; `squash_into` rewrites them to the surviving slot; every pointer is qualified by the frame's base version (0) or the slot's generation (1..=63); `ThoughtCommitId` is never reused.
- Written into `thought-git-api-v1.md` § "Correction: the slot axis and the witness pointer", with two new D-TGA-1 falsifiers.

- Squash-or-land follows from the pointer graph (WORKING-MODEL): reference count per slot is one fold over `w_slot`; land in topological order; a cycle squashes into one slot first; conflicts propagate to dependants; the same graph orders `MergeQueue`.

OPEN: measuring that policy; the `edge.rs` "0 = no corpus anchor" doc line, to correct with D-TGA-1.

Correction (2026-10-20, review #1628): two readings of bits 53..58 exist in code. `deepnsm-v2/src/wire.rs` follows `w_slot` as a witness pointer that ends at 0; `MailboxSoA::apply_edges` filters on it as the mailbox's corpus and has no production caller. Which reading owns the bits is OPEN (operator); the pointer-graph policy assumes the witness reading.

Correction (2026-10-20, operator): the reading is not open. W = thought layer 0..63 was decided on 2026-10-17 (#1597, epistemic-frontier capstone; 0 givens = main, 1..63 hypotheses). The mailbox corpus filter and `deepnsm-v2`'s `WitnessTable` index are competing readers to migrate (capstone §8 row 2), not candidate readings. The "operator decision OPEN" line above is struck.
