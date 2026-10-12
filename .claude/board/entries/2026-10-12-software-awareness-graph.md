# D-SAG-1 — the software's own methods as graph rows (2026-10-12)

**Status:** MEASURED (one concept, one hop). Plan: `.claude/plans/software-awareness-graph-v1.md`.

- **Projection.** `crates/lance-graph-glove-parity/src/awareness.rs` projects a `MethodTable` into typed node and edge rows. Every edge carries a tier and a provenance. The edge table is a Quack table, queried by `Draft` → `lower` → mask-risc, landing on the owning method in one program.
- **Evidence chain.** Under SAP's real claims the graph explains `billable_hours`' refusal down to `claim:hours_logged:hypothesized`. `FirstOfRun` counts as demonstrated only once the compiler's `EvidenceNotIndependent` refusal is projected. IAM and SAP needs stay hypotheses.
- **H1.** The tiled `MaskedSumI32` program measures 1.6–1.8× a single-pass scalar reference:
  - native: 603.5/379.1 µs and 619.6/362.8 µs;
  - config-v3: 617.4/347.6 µs;
  - 1,048,576 rows.

  A fused mask-risc terminal is absent; `fused_ternlog` declines `Pred` and `FusedFold` is `Count`/`Any` only.
- **MDL.** Factoring one shared condition never shortens the description, so G1's `is_billable` is a semantic reuse, not a syntactic one. The first metric dropped a composite's atoms from the vocabulary, and the label-vs-identity counterexample caught it.
- **Disable runs.** 14 runs, all red. One disable first stayed green; the missing test case then exposed a kind-confusion defect in `claim` (fixed).
- **Blocker.** Multi-hop questions need a relation carrier or a factored `EqU32Via`, not a mask passed between programs.

OPEN: method identity is caller-issued (OGAR mints none); the body digest is build-local.
