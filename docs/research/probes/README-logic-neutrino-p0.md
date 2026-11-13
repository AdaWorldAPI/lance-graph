# P0: u8 candidate selection versus typed semantic guarding

**Status:** cold, synthetic research fixture. **Parent:** [CE64 logic-neutrino mindmap, draft PR #2048](https://github.com/AdaWorldAPI/lance-graph/pull/2048). Documentation and isolated standard-library Python test only. No production runtime, ABI, ClassId, recipe mint, F### word, CE64 bit reassignment, evaluator, policy mutation, persisted data or CI workflow.

## Run

From the repository root:

~~~sh
python3 -m unittest -v docs/research/probes/logic_neutrino_p0.py
~~~

Locally verified on Python 3.13: **five tests pass**, including an exhaustive **1,052-observation** fixture matrix. The tests do **not** execute Rust engines or real Rails/game/MeTTa interpreters.

## What the experiment checks

| Feature | Synthetic stimulus | Positive invariant | Deliberately incorrect ablation |
|---|---|---|---|
| Family-scoped selector | two rule families reuse u8 position 1 | a byte alone is not a global ID | family omission would collide (conceptual counterexample; no separate mutation run) |
| Rails-shaped method | same method-chain string for Checklist and Ticket | both candidates may be proposed; typed receiver narrows to one | receiver check removed: two rules eligible |
| CE64-shaped role | packed CE64-like 3-bit orientation plus independently varied Epi5 code | Recipient selected regardless of valid Epi5 code; Core not selected; reserved Epi5 rejects | role check removed: Core eligible |
| Mississippi Queen budget fragment | navigation proposal with 2-coal synthetic cost, coal 0 vs 6 | insufficient resource refuses | coal guard removed: unaffordable candidate eligible |
| Source/ClassView revision | unchanged labels and bytes, changed source or class reading version | refuses stale reading | revision guard removed: stale candidate eligible |

All rules are **synthetic fixture records**. They are not copies of OGAR's u16 recipe codebook, and no result calls DT64 BindStore::admit. The term *guarded candidate* is deliberate: eligibility in this fixture does **not** mean an action is authorized or executable.

The comparator is a separate procedural fixture oracle that never reads the candidate table, candidate selection function or refusal function. The tests vary receiver identity, role (8 codes), Epi5 (32 raw codes), coal, source revision and ClassView revision. The four deletion controls each create a demonstrably incorrect result. Invalid Epi5 raw codes 24–31 refuse.

## Evidence limitations

- The packed role test implements the CE64 bit positions documented by the Rust source, but **does not invoke** the CE64 Rust decoder or Quack MatchU64. Rust parity belongs to follow-up P1.
- The Rails-shaped receiver test has no Ruff/Ore facts or independent ActiveRecord semantic oracle. It only demonstrates candidate ambiguity.
- The Mississippi Queen example is **only** the cost guard fragment for an illustrative maneuver; no complete legal-route oracle, board model or ruleset version is checked.
- No MeTTa/MORK execution, Datafrog substitutions, XLOG recursion, NNUE weights, chess castling rights, Go ko history, or R2IL/F### execution is exercised.
- The procedural oracle is **independent code on synthetic facts**, not a third-party semantic oracle. Five green tests therefore do not certify any semantic Rosetta Stone.
- No timing or performance inference is valid.

## Independent next gates

**P1:** connect an actual borrowed CE64 data lane to the existing Quack MatchU64 filter and independent scalar CE64 decoder. Include 0/1/63/64/65/256/257 rows, altered orientation, altered Epi5, mismatched ClassView and frame, and the forbidden derived-mask gather refusal.

**P2:** connect a real Ruff/Ore association and an OGAR canonical recipe to an existing DT64/F000 binding, using the original ActiveRecord oracle. Preserve incompatible receiver, missing concern and mixed source revision negatives. Do not add an interpreter.

**Decision requested:** does the existence of a tiny, family-qualified candidate index buy any meaningful normalization or executable reuse, or is it merely a convenient label on existing OGAR recipe identity? The falsifier matters more than the green count.

**Do not merge as implementation authorization.** This branch and PR are independent of the #2048 scratchpad.
