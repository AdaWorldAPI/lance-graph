# 2026-10-06 — D-GSO-7a: the reasoning band as a relational certification

## MEASURED

`crates/cognitive-shader-driver/examples/relational_certification_probe.rs`:
13 tests, 12 disable runs red. Supersedes the **meaning** of D-GSO-7's rungs;
#1360's probe and its board row are unchanged.

- **Inventory before the change.** `ReasoningBand` has no production writer
  (`with_reasoning_band` is called only in tests and examples). No class
  outside examples declares `BandPresence::Present`, so every production read
  refuses. The one library reader, `dismech_counterfactual::EdgeRole`, copies
  the band without deciding on it. The one raw ordinal comparison is
  `reasoning_band_probe.rs`'s `cap()`, which ranks `Perspective` / `Meta` /
  `Transcendent` above `Counterfactual`.
- **Reading, declared per class.** Bits 61..63 =
  `0 Open, 1 Associated, 2 Related, 3 Contributes, 4 CausalCandidate, 5 Causes`;
  6 and 7 refuse. `ReasoningBand` is used only as the 3-bit carrier. A class
  declared under the historical reading refuses the same bits. Working names.
- **Order.** `entails` is an explicit table and equals `code >=` on 0..=5.
  Historical `Meta` (6) and `Transcendent` (7) do not satisfy `Causes`.
- **Chain vs partial order.** Exhaustive family of 2,500 two-stratum models:
  0 implication violations with `Associated` existential over the declared
  populations; 10 with marginal association (Contributes without it; Simpson).
  Certified per code `[1100, 144, 610, 259, 387, 0]`.
- `Causes` holds against a confounded observational population when the
  randomized arms certify it, and entails the lower contracts in the trial
  population. #1360's confounding cap would demote it.
- Sibling specificity is not a rung: A and an equally effective sibling give
  `Causes` while A is not discriminative against the sibling.
- Only receipts recorded at or before the model's seal count, so a later
  receipt cannot change what an earlier seal replays to (codex review).
- Removal is not the causal test: under Y = A or B, removing A from a unit with
  B leaves Y while the population trial certifies `Causes`.

## OPEN

- Whether the reading becomes a `band_reading` lens (contract change), and the
  final names.
- `CausalCandidate` carries ordering only.
- Scoped association is only sound if strata are declared before the data
  are read; nothing enforces that yet.
- The robustness mask is an input; no CLAM/CHAODA mask has been run on real data.
- Sibling and overdetermination tests are existence witnesses, not guards.
