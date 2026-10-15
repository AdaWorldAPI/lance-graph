# 2026-10-15 — Ledger correction: CE64 is one wire contract; first counterfactual verdict

**Status:** DECISION (operator rulings R-a..R-e) + TEST-PINNED (O22, R1).

`.claude/research/2026-10-recovery/LEDGER-CORRECTION-2026-10-15.md` corrects
the 2026-10-14 ledger revision.

- **DECISION:** CausalEdge64 is a global wire format and ISA register; every
  header bit means the same thing everywhere. K1 (per-class readings of
  40..63) is withdrawn. Bits 40..42 = method (Pearl), 50..52 = the plasticity
  ladder (the Maslov pyramid), 59..63 = licence. SCOPE: every CE64 producer and
  reader. REVISIT WHEN: a measured case needs two meanings for one bit.
- **O25** is a defect with a known fix: replay composes within a trajectory;
  revision runs only across distinct `temporal.rs` coordinates. K2 narrows to
  that coordinate as the provenance root.
- **O22 done (one step):** `pearl::Reaction::verdict`,
  `Measured::counterfactual_verdict`; the first end-to-end
  `RevisionVerdict::is_acceptable()`. 3 disable runs red.
- **R1 pinned** in `kanban.rs`; disable run red.
- **Closed:** O20 (ndarray `1ecbccb`), N10 (clippy clean on `main`), N3/N4.
- **O26 downgraded:** `causal_bridge` sits behind a `wip` feature that does
  not compile; the mapping cannot reach a caller.
- **N15:** Pearl × Wang NAL S-P × OpenNARS, ONA, PLN, Jøsang, Hyperon/MeTTa as
  views on the one wire; home D-HOLO H12.

53..58 is already the witness handle in code (`w_slot`).

OPEN: 46..49 meaning; per-tenant vs global readings in `docs/architecture/ce64-semantic-upper-half.md`; the derived-packet rule; the ladder's entry
code; `learn`'s rule; the migration gate for old codes 1..6; O23.
