# Epistemic writers, retraction, rule identity, and the reach/far run (2026-11-03)

**Status:** implemented in lance-graph (this PR). The XLOG fork change is in fork PR #7; runtime acceptance is still missing. The NNUE optimizer switch is blocked.

## What changed

- **Retraction (F4 closed).** `pearl::retract` is the counterpart of `pearl::revise`: it re-runs the edge's own operator and lowers the certification to what is still earned, or to Open when the remaining evidence no longer grounds the fold. It never raises; SPO/SP and projections without an operator leave the edge unchanged. The A10 probe's F4 now holds. Two disables turn it red: a no-op retract, and a retract without the never-raise check.
- **No second evidence authority.** `deepnsm_v2::wire` no longer writes CausalCandidate. The contract's obligation for code 4 is Supports plus temporal ordering over a population, and text cannot witness that. The two-source count stays a reading, `WireReport::two_source_causes`. Two disables turn the test red: restoring the promotion, and dropping the count.
- **Production writer boundary.** `causal-edge/tests/epistemic_writer_census.rs` classifies every `with_epistemic_raw5` call in `crates/*/src` by file and enclosing function. The three AUTHORIZED writers are pearl's `stamp` (behind revise / retract / hydrate), `edge_v3::rehydrate` and `deepnsm_v2::wire`; everything else is a TEST_FIXTURE. An unclassified call and a stale entry both fail; the first run, with guessed names, failed in both directions. The test-local admission probe stays test-local: per the A10 ruling it is not promoted before validity survives the publication and read boundary.
- **Rule identity.** NAL derivations name their rule with the existing owner `lance_graph_contract::nars::InferenceType`. Derivations outside NAL (elimination, constraint propagation) have no owner; the probe keeps `DerivationRule::Unowned` for them, and that is the remaining gap.

## XLOG fork: reach/far

- Fork PR AdaWorldAPI/xlog#7 is commit `ea924b3`. The integration request goes to the lead; the PR is not the integration.
- A local run outside the fork tree went through `parse_program`, `rewrite_magic_sets` and `compile`:
  - on fork `main`, reach/far is `Applied` (the hazard);
  - at `ea924b3`, `on` refuses ("rule for far reads magic_sets target reach outside the demanded queries") and `auto` declines with no magic relations in the plan;
  - without `far`, both modes still apply.
- The fork's AGENTS.md excludes local component experiments, so this run is an observation, not runtime evidence. The authorized run on the production path (RunPod GPU) has not been done. The runner was deleted.

## NNUE

- **Exact-degree pairing in the fork optimizer: BLOCKED.** xlog-stats keeps only per-column summaries (`PrefixDegreeStats`: average and maximum degree), and no production code feeds even those; the only caller of `add_prefix_degree` is `wcoj_phase_report`, with hardcoded values. The exact fold needs a per-key degree statistic and a producer on the runtime insert path. That is an XLOG wiring gap and belongs on the XLOG wiring-gap list, not in a blind optimizer change.
- **Second application:** the outcome table now exists on the consumer side, outside this repository. The measured baseline still needs per-round method counts re-derived from harvests.

## Deferral recorded

D-CODEGRAPH-0-BIN B3 (the interner) is DEFERRED until strings are outside the hot path and the XLOG wiring-gap list (6+ gaps, the polyfill among them) has reached the hot path. Any remaining string is taken off at the codebook label ("wardrobe"). The STATUS_BOARD row carries this.
