# 2026-10-17 — PG-Ring idea catalogue (clean room, unassigned research input)

**Status:** WORKING-MODEL (research only; no code, no decision).

`.claude/research/2026-10-17-pgring-idea-catalogue.md` catalogues PG-Ring
(arXiv 2608.03840, ISWC 2026; GPL-3.0, ideas only) in 22 rows against our
substrate: 10 HAVE, 1 PARTIAL, 7 ADOPT, 1 REFUSED, 3 OPEN. It feeds no work
item: D-ASM-11 is DO NOT BUILD, and the cyclic case goes to H4 (synergy map
since #1579).

- PARTIAL: sparse per-thought deltas merged on read (D-HL); the persisted
  path (D-HL-7) is queued.

- REFUSED: the Ring's LF tracking between sort orders reads a position
  computed from another sequence's value (gather); we co-locate pairs
  instead, so its space numbers cannot be reproduced.
- ADOPT: label leaptor as next set / next unset bit; run-encoded label
  presence; per-variable reduction of the Boolean tree as a proven rewrite;
  count arrays as evidence for prefix-sum lanes at seal; adaptive variable
  order by range length; selectivity by popcount; LSQB Q2/Q3/Q6 as cyclic
  fixtures with Q1/Q4/Q5 as acyclic controls.
- OPEN: dense AND vs sparse seek (alternation complexity), which sort
  orders our workloads read, run-encoded presence vs bit scan on v4.
