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
| [drf-focus-probe](crates/drf-focus-probe.index.md) | excluded | crates/drf-focus-probe |
| [drisa-register-probe](crates/drisa-register-probe.index.md) | excluded | crates/drisa-register-probe |
| [dt64-admission-probe](crates/dt64-admission-probe.index.md) | excluded | crates/dt64-admission-probe |
| [dto-class-check](crates/dto-class-check.index.md) | member | tools/dto-class-check |
| [elixir-template](crates/elixir-template.index.md) | excluded | crates/elixir-template |
| [foundry-parity-lab](crates/foundry-parity-lab.index.md) | excluded | crates/foundry-parity-lab |
| [gkat-probe](crates/gkat-probe.index.md) | excluded | crates/gkat-probe |
| [grammar-constraints-probe](crates/grammar-constraints-probe.index.md) | excluded | crates/grammar-constraints-probe |
| [helix](crates/helix.index.md) | excluded | crates/helix |
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
| [weather-poc](crates/weather-poc.index.md) | excluded | crates/weather-poc |

## Crate dependency diagram

Generated from Cargo manifests — extracted relationships only. Internal `normal` dependencies; only crates with at least one internal edge are drawn (78 of 91).

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
  n14["drf-focus-probe"]
  n15["drisa-register-probe"]
  n16["dt64-admission-probe"]
  n18["elixir-template"]
  n19["foundry-parity-lab"]
  n21["grammar-constraints-probe"]
  n23["highheelbgz"]
  n24["holo-h14d-probe"]
  n26["holograph"]
  n27["jc"]
  n28["jc-batch-probe"]
  n29["lance-graph"]
  n31["lance-graph-archetype"]
  n32["lance-graph-arm-discovery"]
  n33["lance-graph-benches"]
  n34["lance-graph-callcenter"]
  n35["lance-graph-catalog"]
  n37["lance-graph-codegraph"]
  n38["lance-graph-cognitive"]
  n39["lance-graph-consumer-conformance"]
  n40["lance-graph-contract"]
  n41["lance-graph-cypher-quack"]
  n42["lance-graph-dir-sim"]
  n43["lance-graph-dt64"]
  n44["lance-graph-dt64-lance"]
  n45["lance-graph-glove-parity"]
  n46["lance-graph-hydrate"]
  n47["lance-graph-lane-fold"]
  n48["lance-graph-mask-risc"]
  n49["lance-graph-metta"]
  n50["lance-graph-nal"]
  n52["lance-graph-ogar"]
  n53["lance-graph-onnx"]
  n54["lance-graph-ontology"]
  n55["lance-graph-osint"]
  n56["lance-graph-planner"]
  n57["lance-graph-python"]
  n58["lance-graph-quack"]
  n59["lance-graph-r2il"]
  n60["lance-graph-rbac"]
  n61["lance-graph-relalg"]
  n62["lance-graph-report"]
  n63["lance-graph-report-ogar"]
  n64["lance-graph-sap"]
  n65["lance-graph-supervisor"]
  n67["lance-graph-xlog"]
  n68["learning"]
  n69["mork-pathmap-probe"]
  n71["ogar-behavior-probe"]
  n72["onebrc-probe"]
  n73["p64-bridge"]
  n74["perturbation-sim"]
  n77["reader-lm"]
  n80["sigker"]
  n81["sigma-tier-router"]
  n82["soa64-u64-lab"]
  n83["surreal_container"]
  n84["symbiont"]
  n85["template-equivalence"]
  n86["template-runtime"]
  n87["thinking-engine"]
  n88["thought-git-probe"]
  n89["tract-harvest-probe"]
  n90["weather-poc"]
  n0 --> n1
  n1 --> n26
  n1 --> n40
  n4 --> n19
  n4 --> n40
  n4 --> n43
  n4 --> n59
  n6 --> n43
  n6 --> n48
  n7 --> n2
  n7 --> n3
  n7 --> n11
  n7 --> n40
  n7 --> n54
  n7 --> n56
  n7 --> n73
  n7 --> n87
  n8 --> n5
  n8 --> n18
  n8 --> n29
  n8 --> n40
  n8 --> n52
  n8 --> n85
  n8 --> n86
  n9 --> n40
  n9 --> n48
  n9 --> n61
  n9 --> n69
  n10 --> n40
  n10 --> n48
  n10 --> n58
  n11 --> n38
  n11 --> n40
  n12 --> n3
  n12 --> n32
  n12 --> n40
  n13 --> n40
  n13 --> n43
  n14 --> n56
  n15 --> n40
  n15 --> n43
  n15 --> n48
  n16 --> n40
  n16 --> n43
  n19 --> n40
  n19 --> n43
  n19 --> n48
  n19 --> n61
  n21 --> n40
  n21 --> n43
  n21 --> n48
  n21 --> n58
  n23 --> n1
  n24 --> n48
  n24 --> n59
  n24 --> n61
  n27 --> n80
  n28 --> n27
  n28 --> n48
  n28 --> n61
  n29 --> n1
  n29 --> n2
  n29 --> n35
  n29 --> n38
  n29 --> n40
  n29 --> n46
  n29 --> n48
  n29 --> n56
  n29 --> n58
  n29 --> n59
  n31 --> n40
  n32 --> n40
  n33 --> n29
  n33 --> n48
  n33 --> n58
  n34 --> n40
  n34 --> n54
  n34 --> n60
  n37 --> n40
  n37 --> n43
  n37 --> n48
  n37 --> n58
  n37 --> n67
  n38 --> n26
  n38 --> n40
  n39 --> n34
  n39 --> n40
  n39 --> n52
  n39 --> n54
  n39 --> n60
  n41 --> n29
  n41 --> n48
  n41 --> n58
  n42 --> n40
  n42 --> n48
  n42 --> n58
  n43 --> n40
  n44 --> n29
  n44 --> n40
  n44 --> n43
  n44 --> n56
  n44 --> n65
  n45 --> n40
  n45 --> n52
  n45 --> n54
  n45 --> n58
  n47 --> n40
  n49 --> n48
  n49 --> n58
  n50 --> n40
  n50 --> n48
  n50 --> n59
  n52 --> n40
  n52 --> n54
  n54 --> n40
  n55 --> n2
  n55 --> n3
  n55 --> n11
  n55 --> n56
  n55 --> n73
  n56 --> n2
  n56 --> n3
  n56 --> n40
  n56 --> n73
  n57 --> n29
  n58 --> n40
  n58 --> n48
  n59 --> n40
  n59 --> n43
  n59 --> n48
  n60 --> n40
  n61 --> n40
  n61 --> n48
  n61 --> n59
  n62 --> n40
  n62 --> n43
  n62 --> n48
  n62 --> n58
  n63 --> n40
  n63 --> n62
  n64 --> n40
  n64 --> n48
  n64 --> n58
  n65 --> n34
  n65 --> n40
  n65 --> n56
  n67 --> n40
  n67 --> n43
  n67 --> n48
  n67 --> n58
  n68 --> n40
  n69 --> n48
  n71 --> n40
  n71 --> n43
  n72 --> n40
  n73 --> n2
  n73 --> n3
  n74 --> n40
  n77 --> n1
  n81 --> n3
  n81 --> n40
  n82 --> n40
  n82 --> n43
  n83 --> n40
  n84 --> n29
  n84 --> n40
  n84 --> n52
  n84 --> n74
  n87 --> n1
  n87 --> n3
  n87 --> n23
  n87 --> n27
  n87 --> n40
  n88 --> n40
  n89 --> n40
  n89 --> n53
  n90 --> n40
  classDef shipped fill:#d8f0d8,stroke:#2e7d32,color:#111
  classDef lab fill:#fff3cd,stroke:#a67c00,color:#111
  classDef open fill:#d6e6fa,stroke:#1565c0,color:#111
  classDef gap fill:#fadadd,stroke:#c62828,stroke-dasharray:5 3,color:#111
  class n1,n7,n29,n31,n33,n34,n35,n39,n40,n41,n43,n46,n47,n48,n53,n54,n56,n58,n59,n60,n62,n65,n67,n81,n83 shipped
  class n0,n2,n3,n4,n5,n6,n8,n9,n10,n11,n12,n13,n14,n15,n16,n18,n19,n21,n23,n24,n26,n27,n28,n32,n37,n38,n42,n44,n45,n49,n50,n52,n55,n57,n61,n63,n64,n68,n69,n71,n72,n73,n74,n77,n80,n82,n84,n85,n86,n87,n88,n89,n90 lab
```
