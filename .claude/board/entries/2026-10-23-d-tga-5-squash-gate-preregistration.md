# 2026-10-23: D-TGA-5 lab precursor — squash the chain of origin into the most-updated lane, gated on contradiction (pre-registration)

**Status:** MEASURED, all seven predictions hold. Everything above the line was
fixed before any probe code was written. The results are below it.

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

## Results (`cd crates/thought-git-probe && cargo test --test squash_gate`)

9 tests pass: P1–P7, plus a seal test and a check against the oracle's own
divergence record. All 15 + 10 + 7 + 9 tests in the crate pass. clippy
`-D warnings` and fmt are clean.

| prediction | result |
|---|---|
| P1 | the squash lands to the same `Version` as the final states, on every one of the 48 fixture cells |
| P2 | `Divergence { cell (5,2), offers [(7, lane 4), (8, lane 9)] }`; on the same input `oracle::squash` keeps 8 silently |
| P3 | `Contradiction { cell (1,0), given 101, offered 999, lane 5 }` |
| P4 | accepted; the input has 3 rewrites of one cell in one lane plus one cell two lanes agree on |
| P5 | ops per lane `(3,2) (7,4) (12,6)`, target 12; a tie goes to the lower id (11 over 20) |
| P6 | blame equals the independent reference; the shared cell carries bits 3 and 7 |
| P7 | see the table below |

P7, with N = 8:

| k | history writes | delta cells | order entries |
|---|---|---|---|
| 1 | 8 | 8 | 1 |
| 2 | 16 | 8 | 2 |
| 4 | 32 | 8 | 4 |
| 8 | 64 | 8 | 8 |
| 16 | 128 | 8 | 16 |
| 32 | 256 | 8 | 32 |
| 64 | 512 | 8 | 64 |

Every anchor was asserted before each disable, and the file was restored
after each one. Every disable turned its test red:

| disable | tests that went red |
|---|---|
| D1 | P2, plus the oracle cross-check |
| D2 | P3 |
| D3 | P4, plus P1, P5, P6, P7 and the seal test (all of them rewrite in a lane) |
| D4 | P5 |
| D5 | P6 |

**Bearing on EXO-138.** Blame survives the squash as one `u64` over lanes,
the same shape as P-E2's stamp. The lane read can therefore serve as the
stamp at squash time, provided this bitset is kept when the squash lands.

**Bearing on EXO-132 (lumpability).** Squash is a quotient: a chain of
operations collapses to one delta. P1 tests that quotient for one reader,
landing on the next version, and it holds. It does not test a reader of an
intermediate state (a mid-chain read), and by design the squash cannot
answer one. The full condition, raised by 01G22yT6: every later read,
including the 59..63 promotion, gets the same answer from the squash as
from the chain. That is a hypothesis, not tested here.
This probe is deterministic, so P1 tests deterministic congruence (the
squashed state is independent of the chain that produced it). It does not
test strong stochastic lumpability. That distinction comes from GPT /08 on
EXO-50, which 019fbZ92 adopted in /011 item 4.

Closeout: `STATUS: measured | OUTCOME: the most-updated lane absorbs the chain of origin under a contradiction/divergence gate; landing the squash equals landing the finals; blame survives; the stored delta stays N while the history grows k·N | OPEN: slot reuse (D-TGA-5's own falsifier); squash against a non-empty prior version; the order list still grows with thought count; CE64 59..63 promotion and certificates (IDEAS)`
