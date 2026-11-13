# P0 logic-neutrino semantic twins: executable collision audit

**Date:** 2026-11-13 · **Status:** P0 TEST-ONLY LAB, not a semantic compiler · **Source:** [CE64/ON×WHEN×DO scratchpad (draft PR #2048)](../2026-11-13-ce64-logic-neutrino-grammar-spine.md).

## Exact claim and its opposite

**Claim under test (necessity, not sufficiency):** a u8 left-corner address can be a useful **candidate index**, but no single u8 can itself authorize an ON×WHEN×DO behavior if two candidates share syntax and disagree in receiver type, orientation, source revision or resource budget.

**Strongest falsifier of the naive claim:** intentionally assign all three unrelated experimental rule candidates the **same 0x2A byte**. Remove one distinguishing WHEN guard and watch negative controls reject that implementation. This proves the *need* for a semantic witness in this fixture, not the *existence* of a general source-to-F### lowering.

## Two separately independent test surfaces

| Surface | What it executes | What it does not claim |
|---|---|---|
| [Python scalar fixture](logic_neutrino_p0.py) | Seven stdlib-only tests for same-byte candidate collision, receiver type, CE64 raw orientation, river coal, revision/reading provenance, cross-domain mismatch and a deliberately broken prefix-only admit | **Not Ruff, OGAR, ClassView, DT64 or a production evaluator.** Context and rule mapping are intentionally hand-authored |
| [Rust CE64 test](../../../crates/causal-edge/tests/logic_neutrino_p0.rs) | Real CausalEdge64::set_orientation/orientation against a separately written scalar bit oracle, all 8 orientations × 32 Epi5 raw patterns, and semantic-role twins | **Not an Epi5 declaration/certification proof.** It checks physical bit interpretation, not full ClassView eligibility |

Three logic-neutrino twins in the Python reference fixture:

| Pair | Same first byte / visible shape | Relevant differing hidden state | Oracle expectation |
|---|---|---|---|
| Rails-like source | 0x2A, items.incomplete.count candidate | Checklist::Item declared scope vs Audit::Item without it | first eligible within test; second WRONG_RECEIVER |
| CE64 | 0x2A, same low S/P/O and F/C bits | Orientation Recipient=2 vs Core=0 at bits 43..45 | first eligible within test; second WRONG_ROLE |
| Mississippi Queen | 0x2A, same heading/position/180-degree requested turn | six coal versus zero coal, test's classic movement-law fixture | first eligible; second INSUFFICIENT_COAL |

**Additional twins:** identical surface text at source revision 7 versus 8; wrong ClassView reading; no witness; unknown river ruleset; CE64 Epi5 changed with orientation unchanged. An Epi5 bit-pattern change is **not** role change.

## Run instructions

Python reference, no repository deps:

~~~sh
python3 docs/research/probes/logic_neutrino_p0.py
~~~

Rust source-backed CE64 subset on a Rust-enabled checkout:

~~~sh
cargo test --manifest-path crates/causal-edge/Cargo.toml --test logic_neutrino_p0
~~~

### Current validation receipt (2026-11-13)

- **Python reference:** 7 tests **PASS** locally, Python 3.13.5.
- **Disable run:** a temporary *uncommitted* mutant replaced the receiver/scope check with an always-false condition. Result: 7 tests executed, **2 FAIL** (test_rails_receiver_twin_and_missing_scope and test_mutant_prefix_only_admission_is_falsified). This is an active kill arm.
- **Rust:** **NOT RUN locally**. This execution environment has no cargo/rustc and no network resolution for installing it. Its three tests are committed for CI or an independent Rust-enabled review. **No claim of Rust parity yet.**
- No benchmarks, Codegraph harvest, real Rails ActiveRecord reflection, real Mississippi Queen game engine, or live ClassView/DT64 binding were executed.

### Interpretation

**P0 preliminary finding:** the *prefix-alone admission hypothesis* is falsified for the constructed fixtures. A typed WHEN witness distinguishes same-byte candidate collisions. This says **nothing** about the optimal byte assignment, universality of ON×WHEN×DO grammar, or execution speed. The toy gate is not allowed to be cited as proof of a production compiler.

**What survives:** the research path “low-byte candidate index + existing semantic identity + declared admission contract” remains a **hypothesis**, pending real integration and independent oracle. In particular, the OGAR RecipeConceptId family/entry is not assigned to the experimental 0x2A, and no F### or CE64 fields are minted.

### Refusal and ownership firewall

- Test-only scalar Context does not replace ClassView, OGAR Class or ActionDef.
- No new Ruby parser, MeTTa/MORK evaluator, Datafrog runtime, F### word, NARS opcode, game executor, or semantic class registry.
- No rewriting CE64 role bits or confusing Haspelmath orientation with river heading or Epi5 topology.
- No new mask gather; XLOG and native rule admission stay unchanged.
- No authority to execute methods, write WORLD, certify causality or change live IAM permissions.
- The 180-degree turn's two-coal cost is an explicit **test fixture assumption** under the classic Mississippi Queen movement variant; not a complete simulation or independent game-engine proof.
- The Python fixture's admit refers **only to its own simulated gate**. Actual DT64/F### authorization is not exercised.

## Next decision gate

Before proposing a real 8-bit grammar selector, find one **actual already-harvested Ruff/Ore left-corner candidate** that maps to an **existing OGAR recipe concept** and attempt a *read-only* DT64 admission with independent method/oracle evidence. Preserve the negative receiver/revision pair. If the same semantic rule cannot be shared across Zammad and OpenProject without source-specific special cases, record NOT_EXPRESSIBLE with the original source spans. Do not skip the two-hop and parameterized exists? refusals already documented for Zammad.

**No merge or runtime work is authorized by this probe.** It only closes a narrowly testable necessity question and leaves all seven research hypotheses open.
