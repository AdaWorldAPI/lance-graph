# Thinking tables: wiring to prior plans (PRs #1447–#1596)

STATUS: WORKING-MODEL, being wired. Every row below starts as OPEN and moves
to WIRED (existing plan or code carries it, with path:line) or NEW (nothing
prior carries it) only after the cited source has been read in full.

## Why this document exists

The model below was stated in session on 2026-10-17, with the observation
that everything in it already exists in the work merged as PRs #1447–#1596,
except the 64 layers themselves. This document checks that claim element by
element. It does not introduce a mechanism; it points each element at the
plan, entry or crate that already carries it, and isolates what is new.

Method: five Haiku substrate-mapper runs read every plan, knowledge note,
board entry and crate `lib.rs` changed by those PRs, in full, and copied the
lines that define, decide or measure something. Each wiring claim below is
then verified against the source itself, never against the map alone.

Separate input, same session: a ruff_cpp_spo harvest of PG-Ring (leapfrog
triejoin over compact ring indexes) with seven Haiku maps. It feeds D-ASM-11
(leapfrog as prior art to learn from, arm A seek over sealed lanes vs arm B
trie zipper), clean-room, and is cited here only where it bears on row 6/7.

## The model, element by element

| # | Element | Claim to verify | Candidate prior work | Status |
|---|---|---|---|---|
| 1 | 64 lanes are 64 tables over one identity | each lane is a whole table over the shared NodeGuid rows; sparse write, merge on read | D-HL (ce64-hypothesis-layers-v1), D-HL-5 thought lanes, D-HOLO H14c | OPEN |
| 2 | Writes are isolated per lane | a lane writes only its own dirty bytes; no lane writes another | D-HL, D-WAL (cycle writer), sparse-write rulings | OPEN |
| 3 | Reads cross lanes by presence-bit folds | `T_a[i] : T_b[i]` at the same coordinate, never a gather or foreign key | D-HOLO H14d `Thought(t)[T_a[i] : T_b[i]]`, presence planes P_l | OPEN |
| 4 | Provenance is the written-by word | per row, which lanes wrote it, as presence bits | D-HL-3 evidential base, H14c/H14d presence planes | OPEN |
| 5 | Independence for revision is decided by source bits | per-delta source bitset, not the lane number | D-HL-3, revision layer invariants (G1–G7), isa::truth | OPEN |
| 6 | Set theory is the AST | ∩ AND, ∪ OR, − AND NOT, σ aperture, π table/field choice, ⋈ layout, μ fixpoint | relalg-v1 (D-RA-1..4), metta-datalog-assimilation (D-ASM), D-TELEPORT fold_ranges | OPEN |
| 7 | Datalog fixpoint over lanes | semi-naive: Δ = new AND NOT old, done when popcount(Δ) = 0 | D-RA-3/4 (rules as algebra values, μ via the d′ driver), D-MTP-1 | OPEN |
| 8 | FIFO scheduling is stratification | a lane frees at fixpoint; FIFO order is the stratum order negation needs | D-ASM schedule, NAL schedule-depth audit, D-NALP | OPEN |
| 9 | Uses: time series, lenses, parallel NAL rules | one lane per year, per lens, or per NAL rule | D-WAL-6/7 temporal, nal-polyfill-v1, NAL audit inventory | OPEN |
| 10 | Foreign-key-shaped queries are native metacognition | who wrote it, who agrees (AND + popcount), who disagrees, as plane algebra | NAL-9 inventory, epistemic-frontier capstone (#1597), Simpson coresearch (#1596) | OPEN |
| 11 | The 64-layer table itself | 64 tables as the unit of thought, with its lifecycle | none expected | OPEN |

## Wiring (filled per element from the maps, each cited line read in source)

To be filled.

## What is new

To be filled: the elements no prior plan carries, each with the plan it
should land in.

## Open points

To be filled.
