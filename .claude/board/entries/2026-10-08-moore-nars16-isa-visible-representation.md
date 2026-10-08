# 2026-10-08 — MooreNars16: a Moore-local representation of ISA-visible CE64 state

**Status:** MEASURED, TEST-PINNED
(`crates/lance-graph-planner/tests/moore_nars16_probe.rs`, 15 tests;
`cargo test -p lance-graph-planner --test moore_nars16_probe`). Probe only:
no CE64 layout, operation or production path changes.

## Claim tested

MooreNars16 does NOT round-trip canonical NARS24. It is a declared
Moore-local representation of the ISA-visible NARS state:

```text
lane (u16):  Pearl3 | Energy4 (raw mantissa nibble) | Plasticity3 | Polarity1 | Epi5
tenant:      Witness6 (one per tenant) + 8 lanes = 128 bits + 8 x SPOFC40
```

Correctness is ISA observational equivalence: for `forward` (both operand
positions), `learn` (both positions) and `syllogize`, a tenant operand and
its canonical edge give the same result on every field except the canonical
Direction3 sign triple.

## Measured

- **What the ISA reads.** Varying one input field at a time over 400 random
  operands per op: Direction3, Witness6 and Epi5 never change any OTHER
  output field. They are carried or dropped, never read. The weight's
  mantissa picks `forward`'s rule; plasticity gates `learn`'s S/P/O rewrites.
- **Equivalence.** 300 random tenants x 8 lanes x 5 ops: 0 observable
  divergences. In 800 `forward` results on the weight side, at least 600 differ
  in raw bits (the copied sign triple), so the equivalence is not bitwise.
- **All 16 raw mantissa nibbles** reach `forward` unchanged, including the ones
  `from_mantissa` collapses.
- **Each representation choice is load-bearing.** Storing 3 energy bits,
  dropping plasticity bit 2, dropping Epi5, restoring the wrong witness, or
  writing polarity over plasticity bit 0 each produces divergence.
- **Direction, three ways.** The 8 slots denote 8 distinct neighbours and a
  rotated slot table is detected. Polarity swaps from/to without changing the
  slot or anything `learn` sees. A probe-local sign-triple consumer refuses
  the Moore reading. The PRODUCTION consumer,
  `CausalNetwork::detect_simpsons_paradox`, cannot: it takes bare edges and
  silently reports "no pattern" for a Moore edge (pinned; Codex review on
  #1406). A Moore resident path must not reach it until it takes a tagged
  operand.
- **Witness scope.** A tenant whose lanes carry different W is refused at
  projection; a uniform W is restored on every lane.
- **Epi5.** Reserved codes 24..31 are refused at projection, decided by the
  contract's `EpistemicState5::decode`.

Disable runs (each anchor asserted), each red: witness refusal off (1),
Epi5 refusal off (1), Simpson consumer accepting Moore (1), observational
compare made bitwise (3), polarity ignored (1).

## OPEN

- **Witness ownership is not settled by the probe.** The probe enforces
  tenant scope; whether any producer NEEDS two lanes of one tenant to carry
  different W at once is answered only by code reading. The one production
  reader, `MailboxSoA::apply_edges` (`mailbox_soa.rs:355`), keeps only edges
  whose W equals the mailbox's own, so a mailbox holds one W. Gomoku's W
  addresses a pending credit event across time, not a lane. Three meanings
  still share the six bits (corpus root handle, mailbox routing, credit
  slot).
- **Polarity1 is a candidate**, not a contract: it has no canonical CE64 home.
- **Not run:** Gomoku A/B replay through tenants, Sudoku/Crossword domain
  runs, chess, timing. Sudoku/Crossword carry only Epi5 today, so they cannot
  test the other fields.
