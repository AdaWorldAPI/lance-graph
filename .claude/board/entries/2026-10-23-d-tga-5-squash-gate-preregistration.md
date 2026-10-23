# 2026-10-23: D-TGA-5 lab precursor — squash the chain of origin into the most-updated lane, gated on contradiction (pre-registration)

**Status:** PRE-REGISTERED. Everything above the line was fixed before any
probe code was written. Results go below it.

**Source:** an operator direction (provenance, not validation): the first
squash gate is "which thought lane was updated most". That lane's chain of
origin is squashed into one lane, gated against contradiction. Lance then
carries the delta, the compiled order and a seal reference, not every
operation. Claim: EXO-50 `session_01VncKoyG6JkpsoqhqTGTv3f/034`.
STATUS_BOARD D-TGA-5 (`squash_into` and blame) is the deliverable this
precedes.

## Model

- **A lane** is `(id ∈ 1..=63, chain: Vec<Thought>)` in origin order. Each
  thought is one batch of writes. `ops(lane)` is the total number of writes
  in its chain.
- **A lane's final state** `F_l` is its chain folded in order, with later
  writes replacing earlier ones at the same cell. A rewrite inside one lane's
  own chain is history, not a contradiction.
- **The gate** squashes a set of lanes:
  - **Target:** the lane with the most ops; ties go to the lowest id.
  - **R1, contradiction:** some `F_l` writes a cell that has a given, with a
    different value. Refuse, naming the cell, the given, the offer and the
    lane.
  - **R2, divergence:** two lanes' final states differ at one cell. Refuse,
    naming the cell and every `(value, lane)`.
  - Otherwise accept.
- **Output of an accepted squash:**
  - `cells`: the union of the final states, without values that agree with a
    given;
  - `order`: every `(lane, thought id)` in lane-ascending, chain order;
  - `blame`: for each cell, a `u64` bitset with bit `l` set for every lane
    that wrote the cell anywhere in its chain;
  - `seal`: FNV-1a 64 over the ordered history.

## Predictions (each one a test)

- **P1, equivalence.** Landing the squashed thought through the D-TGB-0
  oracle (`next_version`) gives the same `Version`, and the same `read` on
  every fixture cell, as landing every lane's final state together.
- **P2, divergence is refused.** Two lanes disagree on one cell →
  `Err(Divergence)` naming both. The naive `oracle::squash` keeps the later
  value on the same input; the test shows that difference.
- **P3, contradiction is refused.** A lane offers 999 for a given →
  `Err(Contradiction)`.
- **P4, silence twin, on non-trivial input.** Disjoint lanes are accepted. A
  lane that rewrites its own cell several times is accepted. Two lanes that
  write the same value to one cell are accepted.
- **P5, target is the most-updated lane.** The fixture puts the most ops on
  the highest lane id, so neither "lowest id" nor "first given" can pass.
- **P6, blame survives the squash.** For every cell, `blame` equals the set
  of writing lanes computed independently from the raw chains, including a
  cell two lanes wrote with the same value (both bits set).
- **P7, delta, not history.**
  - Setup: k rounds of one lane rewriting the same N cells, k ∈ {1, 2, 4,
    …, 64}.
  - `cells` stays N for every k, while the writes the history holds grow as
    k·N.
  - Stated plainly: `order` grows with the number of thoughts (k), not with
    writes per thought. The seal is one value.

## Disables (each must turn its named test red)

| | disable | test that must turn red |
|---|---|---|
| D1 | no divergence check | P2 |
| D2 | no contradiction check | P3 |
| D3 | divergence checked across every chain entry instead of across final states | P4 |
| D4 | target = lowest lane id | P5 |
| D5 | blame keeps only the target's bit | P6 |

## Kill

P1 fails, which means squash-then-land is not landing the finals. Or the gate
refuses P4's inputs, which means it fires on everything.

## Out of scope (an IDEAS row, not claimed)

- Promoting CE64 epistemic bits 59..63 (correlation → causality, MUL
  confidence) at squash time.
- An egglog / eggcc certificate that replaces the order list with a proof
  reference.
- Any Lance storage, and any timing. Under this file's rules only counts are
  reported.

---
