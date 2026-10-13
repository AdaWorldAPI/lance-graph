# D-SAG-CF-1 — Inverted temporal ablation and independent reasoner oracles (2026-10-12)

**Status: RFC + synthetic Boolean smoke oracle; no upstream adapters, no production path, no benchmark.**
**Parent:** [merged #1509 SAG](https://github.com/AdaWorldAPI/lance-graph/pull/1509).

- **Research plan:** [inverted-temporal-ablation-reasoner-oracles-v1.md](../../plans/inverted-temporal-ablation-reasoner-oracles-v1.md), includes the pinned-horizon two-world contract, evidence/rule/method/assumption/attention cuts, OGAR ClassView + MethodKey + schema/reading, Quack one-hop restriction, temporal vs data-causal split, Gadamer admission gap, CE64 no-new-bits rule, W0–W7 experiment sequence, and F0–F15 falsifiers.
- **Annotated sources:** [cross-reasoner arXiv map](../../../.arxiv/2026-10_inverted_temporal_ablation_reasoner_oracles.md) covers Jøsang fission, Wang internal experience, ONA vs OpenNARS, Hyperon MeTTa, actual causality with contingencies, provenance semirings, Differential Dataflow, and Soufflé.
- **Portable local oracle:** [inverted_temporal_ablation_oracle.py](../../tools/inverted_temporal_ablation_oracle.py), stdlib-only DAG/proof-term closure and 12 `unittest` checks. Run from repo root with `python3 .claude/tools/inverted_temporal_ablation_oracle.py`. Its behavior is **not** NARS truth revision, a Pearl SCM, actual causal identification, or production Quack.
- **Existing shipped foundation:** `chain_counterfactual.rs` has a detached single-step cut; `temporal.rs` deinterlaces horizon/owner trajectories; SAG #1509 represents one-hop methods and provenance. None is yet the proposed complete inverse-temporal retraction engine.
- **Biggest correctness blocker:** exact evidence-base dependence and transitive child invalidation are not generally reconstructible from bounded stamps or compressed F/C. Handle as partial/unsupported rather than silently counting redundant evidence.
- **Biggest execution blocker:** legal multi-hop relation passing in Quack; #1509 explicitly prohibits treating population intermediates as native operands.

**Next smallest PR:** pin an ONA upstream SHA and build a fully reproducible `./NAR shell` external input-suppression fixture, including identical baseline and scheduler-control runs. No irreversible writer.

**STOP:** Do not merge this research RFC as if the adapters, sparse speedups, or 64-way NARS truth replays were implemented. TruthOnly and claim-to-step mapping remain explicit semantic decisions; conditional logical importance is not automatically Pearl causality.
