# D-SAG-6-1: reference ledger and stale-status corrections for the D-SAG-6 follow-up audit

**Date:** 2026-10-28. **Status:** PROPOSED / DOCS ONLY. **Parent:** [D-SAG-6](d-sag-6-rust-codegraph-math-logic-tracert-v1.md) (#1760), stage D-SAG-6-1 (the read-only capability audit). **Not:** the audit itself, and no code, test, ABI or implementation authorization.

D-SAG-6 S0 (#1760, merged 2026-10-24) lists its entry points in §7. Since then, #1813–#1831 and the private crai-rs CR-H0/CR-H1 docs have added material the D-SAG-6-1 audit must read, and some of it now carries stale status. This plan records both, so the audit does not start from #1760's ledger alone. It adds nothing to D-SAG-6's design.

Read on 2026-10-28 at lance-graph `main` `eeaaafb`, ruff `main` `f2cb523`, OGAR `main` `6b43a48`, ndarray `master` `38dd52e`, crai-rs `main` `378a24a`. Evidence states: VERIFIED-IN-CODE (read here), CURRENT-CONTRACT (merged doc), MEASURED (command given), OPEN.

## 1. References D-SAG-6-1 needs that #1760 §7 does not list

| Reference | Why the audit needs it | State |
|---|---|---|
| ruff `crates/ruff_spo_triplet/src/ir.rs` (`ModelGraph`) | the structural model Ruff hands over; data, not execution semantics | VERIFIED-IN-CODE (exists) |
| ruff `crates/ruff_spo_triplet/src/expand.rs` | sorted, deduplicated SPO triples: event order and multiplicity are lost, so this view alone cannot separate two programs that differ only in effect order (D-SAG-6 F05) | VERIFIED-IN-CODE (exists); behaviour as documented in ruff and crai-rs CR-H0 |
| ruff `crates/ruff_spo_triplet/src/reassemble.rs` | inverse projection for C++ only, not a general reverse oracle | VERIFIED-IN-CODE (exists) |
| ruff `crates/ruff_spo_triplet/src/role.rs` (`PredicateRole`) | the 13 role bits (DESCRIBES … OPAQUE) that #1814 quotes as READS, DERIVES, EFFECTS, REQUIRES, AUTHORIZES, TRIGGERS | VERIFIED-IN-CODE |
| ruff `.claude/plans/behavioral-ore-v1.md` | the separate ordered event stream (`events.tsv`, never deduplicated); a C++ research leg, not a Rust oracle | CURRENT-CONTRACT (ruff plan) |
| ruff `.claude/plans/r2il-roundtrip-oracle-spec-v1.md` | states that SPO is not the behavioural parity oracle | CURRENT-CONTRACT (ruff plan) |
| ruff `docs/OGAR-POLYGLOT-AST-INTEGRATION.md` | "structure transpiles; behavior does not": an RFC, not a finished round trip | CURRENT-CONTRACT (ruff doc) |
| ruff has no Rust SPO frontend | frontends exist for C++, C#, Python, Ruby, MeTTa and SQLAlchemy (`crates/ruff_{cpp,csharp,python,ruby,metta,sqlalchemy}_spo`), none for Rust. D-SAG-6's "Ruff structural crosswalk" therefore has no Ruff input for Rust source today; the audit must record this as MISSING, not as a coverage gap of an existing arm | MEASURED: `ls crates \| grep spo` on ruff `f2cb523` |
| OGAR `crates/ogar-from-ruff` (`lift_model_graph`, `lift_actions`) | the existing Ruff → OGAR bridge; the audit's "smallest already-owned location for a read-only projection adapter" must be compared against it | VERIFIED-IN-CODE (crate exists; functions named in its own docs) |
| `docs/research/counterfactual-behavior/cfa-1-typed-incidence.md` (#1814) | ON×WHEN×DO over Ruff/OGAR/rust-analyzer records; "Ruff is structural crosswalk, not a Rust MIR oracle"; WHEN is a K3 guard | CURRENT-CONTRACT (merged research doc) |
| `docs/research/counterfactual-behavior/cfa-5-compiler-analysis.md` (#1818) | asks where a behavioural projection binds `ProposedMethod` without a duplicate type; cites #1760, #1747, #1792 | CURRENT-CONTRACT |
| `docs/research/counterfactual-behavior/README.md` and `thought-dependency-scratchpad.md` (#1813) | name D-SAG-6 as the source-harvesting and fixed-point owner | CURRENT-CONTRACT |
| [#1792](https://github.com/AdaWorldAPI/lance-graph/pull/1792) | escaped-`Keep` refusal: a mask may not be reused outside its borrowed scope (D-SAG-6 F07) | CURRENT-CONTRACT |
| [#1799](https://github.com/AdaWorldAPI/lance-graph/pull/1799) and the DT64 plans | DT64 versioned bindings, which crai-rs places between Ruff and the D-SAG-6 projection | CURRENT-CONTRACT |
| `crates/lance-graph-contract/src/spog_tenants.rs` | the graph of an address comes from its classid, not a stored fourth column | VERIFIED-IN-CODE (exists) |
| `crates/lance-graph-contract/src/epistemic_state5.rs` | CausalEdge64 readings need declared meaning and evidence; a syntax edge named `calls` is not causal evidence | VERIFIED-IN-CODE (exists) |
| `crates/lance-graph-cypher-quack/src/lib.rs` | experimental Cypher → Quack with a few supported shapes; other patterns are typed refusals | VERIFIED-IN-CODE (exists) |
| `.claude/plans/d-dtm-1-distributed-thought-delta-reasoning-interchange-v1.md` | "D-SAG-6 owns codegraph"; lists D-SAG-6 as the source-code reasoning Tracert owner | CURRENT-CONTRACT |
| `.claude/board/entries/2026-10-27-harvest-integration-composite-key-and-simd-scope.md` | records the read-only workbench at `mcp.exo.red` as building on D-SAG-6 | CURRENT-CONTRACT (board entry) |
| ndarray `src/simd.rs` re-exports: `*_to_mask` predicates, `popcount_raw`, `mask_ternlog_popcount`, `masked_group_count_u32*` | the concrete primitives for D-SAG-6 §4's first fixture, `count(filter(P, X))` versus `popcount(mask(P, X))` | VERIFIED-IN-CODE (re-exported names) |
| crai-rs `docs/CR-H0-*.md`, `docs/CR-H1-*.md` (private) | negative-mold falsifiers NEG-01..10, CR-F01..18 and X-01..15; several restate F01, F04, F06, F07, F11 and must be reconciled, not duplicated | CURRENT-CONTRACT (other repo, read in full) |

## 2. Corrections to stale or overstated status

| Where | What it says | What is true on 2026-10-28 |
|---|---|---|
| crai-rs `CR-H0-CHASSIS-RETROFIT-MAP.md` §2, `CR-H0-SOURCE-LEDGER.md` | #1813–#1830 are "currently open docs proposals" | all eighteen are merged into lance-graph `main` as docs; still not implementations |
| #1831 plan §2 and board entry | crai-rs CR-H0/CR-H1 is "the successor to #1760" | crai-rs keeps D-SAG-6 as the motor and forbids a new Codegraph owner in crai-rs (CR-H0 §7); its CR-H2 waits on the D-SAG-6-1 audit. crai-rs is a sidecar that consumes D-SAG-6, and D-SAG-6-1 remains unstarted in lance-graph |
| #1760 §0 and §3 | Ruff "may independently corroborate supported source structure" | true for the six languages above; for Rust there is no Ruff frontend, so there is nothing for Ruff to corroborate yet |
| #1760 §0, prompt "do not … invent new … ProposedMethod" | implies a `ProposedMethod` type exists to reuse | `ProposedMethod` appears in plans (SAG capstone, cfa-5) but has no definition in `crates/` (MEASURED: zero hits under `crates/`); the audit must treat it as a planned shape, not an existing type |

## 3. What the audit does with this

The D-SAG-6-1 deliverable stays as #1760 §6 defines it: a SHIPPED / LAB / PROPOSED / MISSING capability map and the smallest read-only adapter recommendation. This ledger adds two required rows to that map, "Ruff Rust frontend: MISSING" and "`ProposedMethod` type: PROPOSED only", and one required comparison, against `ogar-from-ruff`. The audit must also reconcile crai-rs's falsifier sets with F01–F15 rather than adopt a second numbering.

## 4. Open

1. Whether the audit should propose a `ruff_rust_spo` frontend or treat rust-analyzer / rustc MIR as the only Rust source of structure (#1760 already names them as the semantic sources).
2. Where `ProposedMethod` will be defined, and by which owner, before D-SAG-6-2 needs it.
