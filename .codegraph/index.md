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

Generated from Cargo manifests — extracted relationships only. Internal `normal` dependencies; only crates with at least one internal edge are drawn (77 of 90).

```mermaid
flowchart LR
  n0["bge-m3"]
  n1["bgz-tensor"]
  n2["bgz17"]
  n3["causal-edge"]
  n4["cognitive-compiler"]
  n5["cognitive-polyfill-probe"]
  n6["cognitive-shader-driver"]
  n7["cognitive-stack"]
  n8["crepe-plan-probe"]
  n9["d-diamond-1-probe"]
  n10["deepnsm"]
  n11["deepnsm-v2"]
  n12["dr-isa-stream-lab"]
  n13["drf-focus-probe"]
  n14["drisa-register-probe"]
  n15["dt64-admission-probe"]
  n17["elixir-template"]
  n18["foundry-parity-lab"]
  n20["grammar-constraints-probe"]
  n22["highheelbgz"]
  n23["holo-h14d-probe"]
  n25["holograph"]
  n26["jc"]
  n27["jc-batch-probe"]
  n28["lance-graph"]
  n30["lance-graph-archetype"]
  n31["lance-graph-arm-discovery"]
  n32["lance-graph-benches"]
  n33["lance-graph-callcenter"]
  n34["lance-graph-catalog"]
  n36["lance-graph-codegraph"]
  n37["lance-graph-cognitive"]
  n38["lance-graph-consumer-conformance"]
  n39["lance-graph-contract"]
  n40["lance-graph-cypher-quack"]
  n41["lance-graph-dir-sim"]
  n42["lance-graph-dt64"]
  n43["lance-graph-dt64-lance"]
  n44["lance-graph-glove-parity"]
  n45["lance-graph-hydrate"]
  n46["lance-graph-lane-fold"]
  n47["lance-graph-mask-risc"]
  n48["lance-graph-metta"]
  n49["lance-graph-nal"]
  n51["lance-graph-ogar"]
  n52["lance-graph-onnx"]
  n53["lance-graph-ontology"]
  n54["lance-graph-osint"]
  n55["lance-graph-planner"]
  n56["lance-graph-python"]
  n57["lance-graph-quack"]
  n58["lance-graph-r2il"]
  n59["lance-graph-rbac"]
  n60["lance-graph-relalg"]
  n61["lance-graph-report"]
  n62["lance-graph-report-ogar"]
  n63["lance-graph-sap"]
  n64["lance-graph-supervisor"]
  n66["lance-graph-xlog"]
  n67["learning"]
  n68["mork-pathmap-probe"]
  n70["ogar-behavior-probe"]
  n71["onebrc-probe"]
  n72["p64-bridge"]
  n73["perturbation-sim"]
  n76["reader-lm"]
  n79["sigker"]
  n80["sigma-tier-router"]
  n81["soa64-u64-lab"]
  n82["surreal_container"]
  n83["symbiont"]
  n84["template-equivalence"]
  n85["template-runtime"]
  n86["thinking-engine"]
  n87["thought-git-probe"]
  n88["tract-harvest-probe"]
  n89["weather-poc"]
  n0 --> n1
  n1 --> n25
  n1 --> n39
  n5 --> n42
  n5 --> n47
  n6 --> n2
  n6 --> n3
  n6 --> n10
  n6 --> n39
  n6 --> n53
  n6 --> n55
  n6 --> n72
  n6 --> n86
  n7 --> n4
  n7 --> n17
  n7 --> n28
  n7 --> n39
  n7 --> n51
  n7 --> n84
  n7 --> n85
  n8 --> n39
  n8 --> n47
  n8 --> n60
  n8 --> n68
  n9 --> n39
  n9 --> n47
  n9 --> n57
  n10 --> n37
  n10 --> n39
  n11 --> n3
  n11 --> n31
  n11 --> n39
  n12 --> n39
  n12 --> n42
  n13 --> n55
  n14 --> n39
  n14 --> n42
  n14 --> n47
  n15 --> n39
  n15 --> n42
  n18 --> n39
  n18 --> n42
  n18 --> n47
  n20 --> n39
  n20 --> n42
  n20 --> n47
  n20 --> n57
  n22 --> n1
  n23 --> n47
  n23 --> n58
  n23 --> n60
  n26 --> n79
  n27 --> n26
  n27 --> n47
  n27 --> n60
  n28 --> n1
  n28 --> n2
  n28 --> n34
  n28 --> n37
  n28 --> n39
  n28 --> n45
  n28 --> n47
  n28 --> n55
  n28 --> n57
  n28 --> n58
  n30 --> n39
  n31 --> n39
  n32 --> n28
  n32 --> n47
  n32 --> n57
  n33 --> n39
  n33 --> n53
  n33 --> n59
  n36 --> n42
  n36 --> n47
  n36 --> n57
  n36 --> n66
  n37 --> n25
  n37 --> n39
  n38 --> n33
  n38 --> n39
  n38 --> n51
  n38 --> n53
  n38 --> n59
  n40 --> n28
  n40 --> n47
  n40 --> n57
  n41 --> n39
  n41 --> n47
  n41 --> n57
  n42 --> n39
  n43 --> n28
  n43 --> n39
  n43 --> n42
  n43 --> n55
  n43 --> n64
  n44 --> n39
  n44 --> n51
  n44 --> n53
  n44 --> n57
  n46 --> n39
  n48 --> n47
  n48 --> n57
  n49 --> n39
  n49 --> n47
  n49 --> n58
  n51 --> n39
  n51 --> n53
  n53 --> n39
  n54 --> n2
  n54 --> n3
  n54 --> n10
  n54 --> n55
  n54 --> n72
  n55 --> n2
  n55 --> n3
  n55 --> n39
  n55 --> n72
  n56 --> n28
  n57 --> n39
  n57 --> n47
  n58 --> n39
  n58 --> n42
  n58 --> n47
  n59 --> n39
  n60 --> n39
  n60 --> n47
  n60 --> n58
  n61 --> n39
  n61 --> n42
  n61 --> n47
  n61 --> n57
  n62 --> n39
  n62 --> n61
  n63 --> n39
  n63 --> n47
  n63 --> n57
  n64 --> n33
  n64 --> n39
  n64 --> n55
  n66 --> n39
  n66 --> n42
  n66 --> n47
  n66 --> n57
  n67 --> n39
  n68 --> n47
  n70 --> n39
  n70 --> n42
  n71 --> n39
  n72 --> n2
  n72 --> n3
  n73 --> n39
  n76 --> n1
  n80 --> n3
  n80 --> n39
  n81 --> n39
  n81 --> n42
  n82 --> n39
  n83 --> n28
  n83 --> n39
  n83 --> n51
  n83 --> n73
  n86 --> n1
  n86 --> n3
  n86 --> n22
  n86 --> n26
  n86 --> n39
  n87 --> n39
  n88 --> n39
  n88 --> n52
  n89 --> n39
  classDef shipped fill:#d8f0d8,stroke:#2e7d32,color:#111
  classDef lab fill:#fff3cd,stroke:#a67c00,color:#111
  classDef open fill:#d6e6fa,stroke:#1565c0,color:#111
  classDef gap fill:#fadadd,stroke:#c62828,stroke-dasharray:5 3,color:#111
  class n1,n6,n28,n30,n32,n33,n34,n38,n39,n40,n42,n45,n46,n47,n52,n53,n55,n57,n58,n59,n61,n64,n66,n80,n82 shipped
  class n0,n2,n3,n4,n5,n7,n8,n9,n10,n11,n12,n13,n14,n15,n17,n18,n20,n22,n23,n25,n26,n27,n31,n36,n37,n41,n43,n44,n48,n49,n51,n54,n56,n60,n62,n63,n67,n68,n70,n71,n72,n73,n76,n79,n81,n83,n84,n85,n86,n87,n88,n89 lab
```
