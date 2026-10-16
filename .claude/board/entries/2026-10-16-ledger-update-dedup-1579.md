# 2026-10-16 — Recovery ledger deduplicated against #1579; DOC-WIRE (2) and N21 closed

**Status:** WORKING-MODEL (research document; no code changed).

`.claude/research/2026-10-recovery/LEDGER-UPDATE-2026-10-16b.md` maps the
overlap between the 2026-10-16 ledger revision (#1578) and #1579's synergy map,
using the ledger ids #1579 §8 adopted.

- **Closed:**
  - **C30 duplicate:** #1579 kept one #1575 arc entry.
  - **DOC-WIRE (2):** three docs still described bits 59..60 as two readings; each now carries a dated correction pointing at `causal-edge/src/layout.rs`.
  - **N21:** #1570's five commands all exit 0: `haspelmath` 6/6, `deepnsm-v2 --all-targets` 202 + 14, clippy `-D warnings` and fmt clean, and the example prints `SPO=(doctor, gave, letter)`.
- **Regraded:** N17 (relalg absorption) is no longer low. Its fix waits on D-RA-2 (rules into the contract set IR, D-CMR-1, or delete relalg).
- **New input:** #1579 §9 measured that no two label corpora fit one 256-entry codebook, so D-ASM-1's `LabelCodebook` should be one codebook per classid and space. N16 (operand registry) is related to that, not the same item.
- **Recorded by reference from #1579:** D2 preimage op, D6 SAG on a withdrawn base, X11, X19, D8 doc drift, L6 (#1544). X18/D7 was closed by #1579.

OPEN: DOC-WIRE (1), BAND-RETIRE, O30, O29, D-ASM authorization, D-RA-2.
