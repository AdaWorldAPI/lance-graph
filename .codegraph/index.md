# .codegraph

A GENERATED projection of source-backed facts (Cargo manifests, Rust item declarations, `use` trees,
ruff_charon call artifacts, and the hand-curated `architecture.tsv`). **The code is the truth**; this
is a grep-friendly index, never an authority. Calls come only from existing Charon artifacts;
without one the crate says `ARTIFACT_MISSING`.

## Freshness

```
python3 .claude/tools/codegraph_index.py --check   # exit 1 lists STALE crates and differing files
python3 .claude/tools/codegraph_index.py --write   # regenerate
```

No timestamps, absolute paths or HEAD sha are recorded. A crate digest is sha256 over sorted
(relpath, git blob id) of its Cargo.toml and `src/**/*.rs`. `--check` also fails on
ANCHOR-MISSING / FILE-MISSING rows in decisions.tsv.

## Evidence legend

- `CONFIRMED` — read directly from an authoritative artifact at the cited `path:line` (a Cargo manifest
  key, a Rust declaration or `use` statement as written, a Charon COMPILER-graded resolved call).
- `POSSIBLE` — a relationship inferred, not proven: a glob import, a re-export chain not resolved,
  a `MAY_DISPATCH` / dynamic call, a textual mention.
- `UNKNOWN` — no fact source covers it (e.g. calls when no Charon artifact).
- `LAB` / `OPEN-PR` / `PROPOSED` / `GAP` — only in architecture-derived rows; copied from
  architecture.tsv, never inferred.
- `STALE` — the cited source changed since the fact was recorded (Charon file hash mismatch).

## Files

- `crates.tsv`, `freshness.tsv`, `coverage.tsv` — crate inventory, digests, what the scanner could not see.
- `dependencies.tsv` / `reverse_dependencies.tsv` — `features_enabling` = this crate's own Cargo features
  that switch the dependency on; `internal` = target is an in-scope crate.
- `symbols.tsv` — pub / pub(crate) top-level items and `pub fn`/`const` in impl blocks (`Type::method`);
  `impl` rows carry visibility `-`. `imports.tsv` — one row per use-tree leaf, cfg kept.
- `calls.tsv` — Charon rows only. `decisions.tsv` / `gaps.tsv` — validated architecture rows.
- `crates/<crate>.index.md`, `views/<symbol>.md`.

Not scanned: macro-generated items, function-local `use`, `tests/`, `examples/`, `benches/`.

## Grep recipes (NEW / DERIVED 2026-11-11)

The original ~15 navigation commands referenced in earlier discussion were not recovered; these recipes are new and derived, not historical.

<!-- recipes -->
```
rg -n '^lance-graph-contract\t' .codegraph/reverse_dependencies.tsv   # who depends on a crate
rg -n '\tClassView\t' .codegraph/symbols.tsv                          # where a symbol is defined
rg -n 'lance_graph_contract::class_view' .codegraph/imports.tsv         # who imports a path
rg -n 'feature=' .codegraph/imports.tsv .codegraph/symbols.tsv         # cfg-gated imports and items
rg -n 'ARTIFACT_MISSING' .codegraph/gaps.tsv                                # crates with no call graph
```
<!-- /recipes -->

## Crates

| crate | workspace | path |
|---|---|---|
| [bge-m3](crates/bge-m3.index.md) | excluded | crates/bge-m3 |
| [bgz-tensor](crates/bgz-tensor.index.md) | member | crates/bgz-tensor |
| [bgz17](crates/bgz17.index.md) | excluded | crates/bgz17 |
| [causal-edge](crates/causal-edge.index.md) | excluded | crates/causal-edge |
| [cg-beh-probe](crates/cg-beh-probe.index.md) | excluded | crates/cg-beh-probe |
| [cognitive-compiler](crates/cognitive-compiler.index.md) | excluded | crates/cognitive-compiler |
| [cognitive-polyfill-probe](crates/cognitive-polyfill-probe.index.md) | excluded | crates/cognitive-polyfill-probe |
| [cognitive-shader-driver](crates/cognitive-shader-driver.index.md) | member | crates/cognitive-shader-driver |
| [cognitive-stack](crates/cognitive-stack.index.md) | excluded | crates/cognitive-stack |
| [crepe-plan-probe](crates/crepe-plan-probe.index.md) | excluded | crates/crepe-plan-probe |
| [d-diamond-1-probe](crates/d-diamond-1-probe.index.md) | excluded | crates/d-diamond-1-probe |
| [deepnsm](crates/deepnsm.index.md) | excluded | crates/deepnsm |
| [deepnsm-v2](crates/deepnsm-v2.index.md) | excluded | crates/deepnsm-v2 |
| [dr-isa-stream-lab](crates/dr-isa-stream-lab.index.md) | excluded | crates/dr-isa-stream-lab |
| [dr-select-k-lab](crates/dr-select-k-lab.index.md) | excluded | crates/dr-select-k-lab |
| [dr256-ternlog-lab](crates/dr256-ternlog-lab.index.md) | member | crates/dr256-ternlog-lab |
| [drf-focus-probe](crates/drf-focus-probe.index.md) | excluded | crates/drf-focus-probe |
| [drisa-register-probe](crates/drisa-register-probe.index.md) | excluded | crates/drisa-register-probe |
| [dt64-admission-probe](crates/dt64-admission-probe.index.md) | excluded | crates/dt64-admission-probe |
| [dt64-hotplug-probe](crates/dt64-hotplug-probe.index.md) | excluded | crates/dt64-hotplug-probe |
| [dto-class-check](crates/dto-class-check.index.md) | member | tools/dto-class-check |
| [duckdb-selvec-compat](crates/duckdb-selvec-compat.index.md) | excluded | crates/duckdb-selvec-compat |
| [elixir-template](crates/elixir-template.index.md) | excluded | crates/elixir-template |
| [foundry-parity-lab](crates/foundry-parity-lab.index.md) | excluded | crates/foundry-parity-lab |
| [gkat-probe](crates/gkat-probe.index.md) | excluded | crates/gkat-probe |
| [grammar-constraints-probe](crates/grammar-constraints-probe.index.md) | excluded | crates/grammar-constraints-probe |
| [helix](crates/helix.index.md) | excluded | crates/helix |
| [hidden-state-twins-probe](crates/hidden-state-twins-probe.index.md) | excluded | crates/hidden-state-twins-probe |
| [highheelbgz](crates/highheelbgz.index.md) | excluded | crates/highheelbgz |
| [holo-h14d-probe](crates/holo-h14d-probe.index.md) | excluded | crates/holo-h14d-probe |
| [holo-lance-masked-io-probe](crates/holo-lance-masked-io-probe.index.md) | excluded | crates/holo-lance-masked-io-probe |
| [holograph](crates/holograph.index.md) | excluded | crates/holograph |
| [jc](crates/jc.index.md) | excluded | crates/jc |
| [jc-batch-probe](crates/jc-batch-probe.index.md) | excluded | crates/jc-batch-probe |
| [lance-graph](crates/lance-graph.index.md) | member | crates/lance-graph |
| [lance-graph-act-r](crates/lance-graph-act-r.index.md) | excluded | crates/lance-graph-act-r |
| [lance-graph-archetype](crates/lance-graph-archetype.index.md) | member | crates/lance-graph-archetype |
| [lance-graph-arm-discovery](crates/lance-graph-arm-discovery.index.md) | excluded | crates/lance-graph-arm-discovery |
| [lance-graph-benches](crates/lance-graph-benches.index.md) | member | crates/lance-graph-benches |
| [lance-graph-callcenter](crates/lance-graph-callcenter.index.md) | member | crates/lance-graph-callcenter |
| [lance-graph-catalog](crates/lance-graph-catalog.index.md) | member | crates/lance-graph-catalog |
| [lance-graph-codec-research](crates/lance-graph-codec-research.index.md) | excluded | crates/lance-graph-codec-research |
| [lance-graph-codegraph](crates/lance-graph-codegraph.index.md) | excluded | crates/lance-graph-codegraph |
| [lance-graph-cognitive](crates/lance-graph-cognitive.index.md) | excluded | crates/lance-graph-cognitive |
| [lance-graph-consumer-conformance](crates/lance-graph-consumer-conformance.index.md) | member | crates/lance-graph-consumer-conformance |
| [lance-graph-contract](crates/lance-graph-contract.index.md) | member | crates/lance-graph-contract |
| [lance-graph-cypher-quack](crates/lance-graph-cypher-quack.index.md) | member | crates/lance-graph-cypher-quack |
| [lance-graph-dir-sim](crates/lance-graph-dir-sim.index.md) | excluded | crates/lance-graph-dir-sim |
| [lance-graph-dt64](crates/lance-graph-dt64.index.md) | member | crates/lance-graph-dt64 |
| [lance-graph-dt64-lance](crates/lance-graph-dt64-lance.index.md) | excluded | crates/lance-graph-dt64-lance |
| [lance-graph-glove-parity](crates/lance-graph-glove-parity.index.md) | excluded | crates/lance-graph-glove-parity |
| [lance-graph-hydrate](crates/lance-graph-hydrate.index.md) | member | crates/lance-graph-hydrate |
| [lance-graph-lane-fold](crates/lance-graph-lane-fold.index.md) | member | crates/lance-graph-lane-fold |
| [lance-graph-mailbox-lanes](crates/lance-graph-mailbox-lanes.index.md) | member | crates/lance-graph-mailbox-lanes |
| [lance-graph-mask-risc](crates/lance-graph-mask-risc.index.md) | member | crates/lance-graph-mask-risc |
| [lance-graph-metta](crates/lance-graph-metta.index.md) | excluded | crates/lance-graph-metta |
| [lance-graph-nal](crates/lance-graph-nal.index.md) | excluded | crates/lance-graph-nal |
| [lance-graph-nsl](crates/lance-graph-nsl.index.md) | excluded | crates/lance-graph-nsl |
| [lance-graph-ogar](crates/lance-graph-ogar.index.md) | excluded | crates/lance-graph-ogar |
| [lance-graph-onnx](crates/lance-graph-onnx.index.md) | member | crates/lance-graph-onnx |
| [lance-graph-ontology](crates/lance-graph-ontology.index.md) | member | crates/lance-graph-ontology |
| [lance-graph-osint](crates/lance-graph-osint.index.md) | excluded | crates/lance-graph-osint |
| [lance-graph-planner](crates/lance-graph-planner.index.md) | member | crates/lance-graph-planner |
| [lance-graph-python](crates/lance-graph-python.index.md) | excluded | crates/lance-graph-python |
| [lance-graph-quack](crates/lance-graph-quack.index.md) | member | crates/lance-graph-quack |
| [lance-graph-r2il](crates/lance-graph-r2il.index.md) | member | crates/lance-graph-r2il |
| [lance-graph-rbac](crates/lance-graph-rbac.index.md) | member | crates/lance-graph-rbac |
| [lance-graph-relalg](crates/lance-graph-relalg.index.md) | excluded | crates/lance-graph-relalg |
| [lance-graph-report](crates/lance-graph-report.index.md) | member | crates/lance-graph-report |
| [lance-graph-report-ogar](crates/lance-graph-report-ogar.index.md) | excluded | crates/lance-graph-report-ogar |
| [lance-graph-sap](crates/lance-graph-sap.index.md) | excluded | crates/lance-graph-sap |
| [lance-graph-supervisor](crates/lance-graph-supervisor.index.md) | member | crates/lance-graph-supervisor |
| [lance-graph-turbovec](crates/lance-graph-turbovec.index.md) | excluded | crates/lance-graph-turbovec |
| [lance-graph-xlog](crates/lance-graph-xlog.index.md) | member | crates/lance-graph-xlog |
| [learning](crates/learning.index.md) | excluded | crates/learning |
| [mork-pathmap-probe](crates/mork-pathmap-probe.index.md) | excluded | crates/mork-pathmap-probe |
| [neural-debug](crates/neural-debug.index.md) | member | crates/neural-debug |
| [ogar-behavior-probe](crates/ogar-behavior-probe.index.md) | excluded | crates/ogar-behavior-probe |
| [onebrc-probe](crates/onebrc-probe.index.md) | excluded | crates/onebrc-probe |
| [p64-bridge](crates/p64-bridge.index.md) | excluded | crates/p64-bridge |
| [perturbation-sim](crates/perturbation-sim.index.md) | excluded | crates/perturbation-sim |
| [pg-parity-harness](crates/pg-parity-harness.index.md) | member | crates/pg-parity-harness |
| [quack-boundary-lab](crates/quack-boundary-lab.index.md) | member | crates/quack-boundary-lab |
| [quasicryth-research](crates/quasicryth-research.index.md) | excluded | crates/quasicryth-research |
| [r2il-mask-abi-probe](crates/r2il-mask-abi-probe.index.md) | excluded | crates/r2il-mask-abi-probe |
| [reader-lm](crates/reader-lm.index.md) | excluded | crates/reader-lm |
| [rp-seal-t0-probe](crates/rp-seal-t0-probe.index.md) | excluded | crates/rp-seal-t0-probe |
| [sag-ra-probe](crates/sag-ra-probe.index.md) | excluded | crates/sag-ra-probe |
| [sigker](crates/sigker.index.md) | excluded | crates/sigker |
| [sigma-tier-router](crates/sigma-tier-router.index.md) | member | crates/sigma-tier-router |
| [soa64-u64-lab](crates/soa64-u64-lab.index.md) | excluded | crates/soa64-u64-lab |
| [surreal_container](crates/surreal_container.index.md) | member | crates/surreal_container |
| [symbiont](crates/symbiont.index.md) | excluded | crates/symbiont |
| [template-equivalence](crates/template-equivalence.index.md) | excluded | crates/template-equivalence |
| [template-runtime](crates/template-runtime.index.md) | excluded | crates/template-runtime |
| [thinking-engine](crates/thinking-engine.index.md) | excluded | crates/thinking-engine |
| [thought-git-probe](crates/thought-git-probe.index.md) | excluded | crates/thought-git-probe |
| [tract-harvest-probe](crates/tract-harvest-probe.index.md) | excluded | crates/tract-harvest-probe |
| [virtual-address-lab](crates/virtual-address-lab.index.md) | member | crates/virtual-address-lab |
| [weather-poc](crates/weather-poc.index.md) | excluded | crates/weather-poc |

## Crate dependency diagram

Generated from Cargo manifests — extracted relationships only. Internal `normal` dependencies; only crates with at least one internal edge are drawn (85 of 100).

```mermaid
flowchart LR
  n0["bge-m3"]
  n1["bgz-tensor"]
  n2["bgz17"]
  n3["causal-edge"]
  n4["cg-beh-probe"]
  n5["cognitive-compiler"]
  n6["cognitive-polyfill-probe"]
  n7["cognitive-shader-driver"]
  n8["cognitive-stack"]
  n9["crepe-plan-probe"]
  n10["d-diamond-1-probe"]
  n11["deepnsm"]
  n12["deepnsm-v2"]
  n13["dr-isa-stream-lab"]
  n14["dr-select-k-lab"]
  n15["dr256-ternlog-lab"]
  n16["drf-focus-probe"]
  n17["drisa-register-probe"]
  n18["dt64-admission-probe"]
  n19["dt64-hotplug-probe"]
  n22["elixir-template"]
  n23["foundry-parity-lab"]
  n25["grammar-constraints-probe"]
  n27["hidden-state-twins-probe"]
  n28["highheelbgz"]
  n29["holo-h14d-probe"]
  n31["holograph"]
  n32["jc"]
  n33["jc-batch-probe"]
  n34["lance-graph"]
  n36["lance-graph-archetype"]
  n37["lance-graph-arm-discovery"]
  n38["lance-graph-benches"]
  n39["lance-graph-callcenter"]
  n40["lance-graph-catalog"]
  n42["lance-graph-codegraph"]
  n43["lance-graph-cognitive"]
  n44["lance-graph-consumer-conformance"]
  n45["lance-graph-contract"]
  n46["lance-graph-cypher-quack"]
  n47["lance-graph-dir-sim"]
  n48["lance-graph-dt64"]
  n49["lance-graph-dt64-lance"]
  n50["lance-graph-glove-parity"]
  n51["lance-graph-hydrate"]
  n52["lance-graph-lane-fold"]
  n53["lance-graph-mailbox-lanes"]
  n54["lance-graph-mask-risc"]
  n55["lance-graph-metta"]
  n56["lance-graph-nal"]
  n58["lance-graph-ogar"]
  n59["lance-graph-onnx"]
  n60["lance-graph-ontology"]
  n61["lance-graph-osint"]
  n62["lance-graph-planner"]
  n63["lance-graph-python"]
  n64["lance-graph-quack"]
  n65["lance-graph-r2il"]
  n66["lance-graph-rbac"]
  n67["lance-graph-relalg"]
  n68["lance-graph-report"]
  n69["lance-graph-report-ogar"]
  n70["lance-graph-sap"]
  n71["lance-graph-supervisor"]
  n73["lance-graph-xlog"]
  n74["learning"]
  n75["mork-pathmap-probe"]
  n77["ogar-behavior-probe"]
  n78["onebrc-probe"]
  n79["p64-bridge"]
  n80["perturbation-sim"]
  n82["quack-boundary-lab"]
  n85["reader-lm"]
  n88["sigker"]
  n89["sigma-tier-router"]
  n90["soa64-u64-lab"]
  n91["surreal_container"]
  n92["symbiont"]
  n93["template-equivalence"]
  n94["template-runtime"]
  n95["thinking-engine"]
  n96["thought-git-probe"]
  n97["tract-harvest-probe"]
  n98["virtual-address-lab"]
  n99["weather-poc"]
  n0 --> n1
  n1 --> n31
  n1 --> n45
  n4 --> n23
  n4 --> n45
  n4 --> n48
  n4 --> n65
  n6 --> n48
  n6 --> n54
  n7 --> n2
  n7 --> n3
  n7 --> n11
  n7 --> n45
  n7 --> n60
  n7 --> n62
  n7 --> n79
  n7 --> n95
  n8 --> n5
  n8 --> n22
  n8 --> n34
  n8 --> n45
  n8 --> n58
  n8 --> n93
  n8 --> n94
  n9 --> n45
  n9 --> n54
  n9 --> n67
  n9 --> n75
  n10 --> n45
  n10 --> n54
  n10 --> n64
  n11 --> n43
  n11 --> n45
  n12 --> n3
  n12 --> n37
  n12 --> n45
  n13 --> n45
  n13 --> n48
  n14 --> n54
  n15 --> n45
  n15 --> n48
  n15 --> n54
  n16 --> n62
  n17 --> n45
  n17 --> n48
  n17 --> n54
  n18 --> n45
  n18 --> n48
  n19 --> n45
  n19 --> n48
  n19 --> n58
  n23 --> n45
  n23 --> n48
  n23 --> n54
  n23 --> n67
  n25 --> n45
  n25 --> n48
  n25 --> n54
  n25 --> n64
  n27 --> n45
  n27 --> n48
  n27 --> n54
  n27 --> n64
  n28 --> n1
  n29 --> n54
  n29 --> n65
  n29 --> n67
  n32 --> n88
  n33 --> n32
  n33 --> n54
  n33 --> n67
  n34 --> n1
  n34 --> n2
  n34 --> n40
  n34 --> n43
  n34 --> n45
  n34 --> n51
  n34 --> n54
  n34 --> n62
  n34 --> n64
  n34 --> n65
  n36 --> n45
  n37 --> n45
  n38 --> n15
  n38 --> n34
  n38 --> n45
  n38 --> n46
  n38 --> n48
  n38 --> n54
  n38 --> n64
  n39 --> n45
  n39 --> n60
  n39 --> n66
  n42 --> n45
  n42 --> n48
  n42 --> n54
  n42 --> n64
  n42 --> n73
  n43 --> n31
  n43 --> n45
  n44 --> n39
  n44 --> n45
  n44 --> n58
  n44 --> n60
  n44 --> n66
  n46 --> n34
  n46 --> n54
  n46 --> n64
  n47 --> n45
  n47 --> n54
  n47 --> n64
  n48 --> n45
  n49 --> n34
  n49 --> n45
  n49 --> n48
  n49 --> n62
  n49 --> n71
  n50 --> n45
  n50 --> n58
  n50 --> n60
  n50 --> n64
  n52 --> n45
  n53 --> n7
  n53 --> n45
  n53 --> n48
  n53 --> n54
  n55 --> n54
  n55 --> n64
  n56 --> n45
  n56 --> n54
  n56 --> n65
  n58 --> n45
  n58 --> n60
  n60 --> n45
  n61 --> n2
  n61 --> n3
  n61 --> n11
  n61 --> n62
  n61 --> n79
  n62 --> n2
  n62 --> n3
  n62 --> n45
  n62 --> n79
  n63 --> n34
  n64 --> n45
  n64 --> n54
  n65 --> n45
  n65 --> n48
  n65 --> n54
  n66 --> n45
  n67 --> n45
  n67 --> n54
  n67 --> n65
  n68 --> n45
  n68 --> n48
  n68 --> n54
  n68 --> n64
  n69 --> n45
  n69 --> n68
  n70 --> n45
  n70 --> n54
  n70 --> n64
  n71 --> n39
  n71 --> n45
  n71 --> n62
  n73 --> n45
  n73 --> n48
  n73 --> n54
  n73 --> n64
  n74 --> n45
  n75 --> n54
  n77 --> n45
  n77 --> n48
  n78 --> n45
  n79 --> n2
  n79 --> n3
  n80 --> n45
  n82 --> n45
  n82 --> n48
  n82 --> n54
  n82 --> n64
  n85 --> n1
  n89 --> n3
  n89 --> n45
  n90 --> n45
  n90 --> n48
  n91 --> n45
  n92 --> n34
  n92 --> n45
  n92 --> n58
  n92 --> n80
  n95 --> n1
  n95 --> n3
  n95 --> n28
  n95 --> n32
  n95 --> n45
  n96 --> n45
  n97 --> n45
  n97 --> n59
  n98 --> n15
  n98 --> n45
  n98 --> n48
  n98 --> n54
  n98 --> n64
  n98 --> n68
  n99 --> n45
  classDef shipped fill:#d8f0d8,stroke:#2e7d32,color:#111
  classDef lab fill:#fff3cd,stroke:#a67c00,color:#111
  classDef open fill:#d6e6fa,stroke:#1565c0,color:#111
  classDef gap fill:#fadadd,stroke:#c62828,stroke-dasharray:5 3,color:#111
  class n1,n7,n15,n34,n36,n38,n39,n40,n44,n45,n46,n48,n51,n52,n53,n54,n59,n60,n62,n64,n65,n66,n68,n71,n73,n82,n89,n91,n98 shipped
  class n0,n2,n3,n4,n5,n6,n8,n9,n10,n11,n12,n13,n14,n16,n17,n18,n19,n22,n23,n25,n27,n28,n29,n31,n32,n33,n37,n42,n43,n47,n49,n50,n55,n56,n58,n61,n63,n67,n69,n70,n74,n75,n77,n78,n79,n80,n85,n88,n90,n92,n93,n94,n95,n96,n97,n99 lab
```
