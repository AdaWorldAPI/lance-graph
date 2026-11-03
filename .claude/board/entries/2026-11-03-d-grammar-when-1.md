# D-GRAMMAR-WHEN-1: binding a grammatical WHEN through DT64 (2026-11-03)

**Status:** measured, lab only. Probe `crates/grammar-constraints-probe` (`src/when.rs`, `tests/when.rs`, `examples/when_cost.rs`); report `.claude/probes/2026-11-03-d-grammar-when-1/REPORT.md`. No production change, no ClassId, no `F###`.

## Measured

- **The #1947 head hop is gather-shaped.**
  - #1947 used `Pred::EqU32Via`, which reads `foreign[fk[i]]`.
  - ADMISSION B: "a position derived from another relational column or table is not admissible as a read address … Execution remains co-addressed."
  - The 2026-10-16 banner bans reads at a position taken from another lane's value.
  - Its sibling `EqU32ViaLane` is marked VIOLATION (D-ASM-11A).
  - No ruling admits `EqU32Via`; EXO-140 is still OPEN.
- **DT64 can state the dependent's own conditions, but not the head frame.**
  - The only hop-shaped leaf is an opaque `Sym`, and `admit` checks witness equality only.
  - `validate` never checks WHEN lanes against the class basis (pinned by a test).
  - No read-only "returns" DO exists.
  - R2IL `prepare` still refuses any WHEN (`GuardNotExecutable`); its integration suite passes 52/52.
- **A co-addressed form ("the join is the layout") makes the WHEN row-local in today's SetExpr.**
  - The head's frame is written as a lane at the dependent's own coordinate, so the WHEN needs no `Sym`, and the two guards keep distinct witnesses.
  - The query needs no `EqU32Via`: 10 ops, about 0.41–0.47 µs for both methods over 40 tokens, against 0.56–0.57 µs via the hop.
  - The cross-row read moves into the producer, which does it once and counted (35 lookups).
- **Lab composer (approach C): admit → check description vs witness → lower description ∧ shared WHERE into ONE program.**
  - Both forms match an independent oracle and the gold.
  - 11 negative controls behave as predicted (incl. a re-sealed same-size layout, from Codex review); 9 source disables go red.

## Open (owner)

- **EXO-140:** classify `EqU32Via`. Proposed, nonbinding: gather-class, not admitted on the query path.
- **EXO-46 / D-RMAP-1:** admit, or refuse, a witnessed sequential producer for co-addressed relation lanes.
- **D-DT64-6:** three open questions.
  - A verified description identity in place of an opaque `Sym`.
  - Checking guard lanes in `validate`.
  - A read-only DO.
- **Frame lexicon:** an owner, a revision and a language scope (`LanguageId` exists).
- **Role identity:** CE64 `Orientation` (one code: `Core`+7) vs `DativeFunction` (a set of 8 incl. Judicantis). The 7 shared functions are offset by one.
- **Candidate promotion:** a selection policy, plus the hydration / Rubicon gate.
