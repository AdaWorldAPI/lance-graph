# D-BIND-BUNDLE-MATRIX — who owns each concern

> Companion to `D-BIND-BUNDLE-0.md`. ● = owns, ○ = carries/consumes.
> "Today" names the live type. [G] read in code, [S] reasoned.

| Concern | Semantic | Bind | V4 | Bundle | Backend | Today | |
|---|---|---|---|---|---|---|---|
| field / label name | ● | ○ consumed | | | | AST, LogicalOperator | G |
| classid | | ● | ○ | | | not yet bound (labels only) | G |
| lane ordinal (`Col`) | | ● | ○ | | | `Binding` | G |
| lane width / signedness | | ● chooses | ● op meaning | | | `Kind` + `Pred::EqI32/EqU32` | G |
| codebook ordinal | | ● | ○ immediate | | | — | S |
| Local / Via / Pair | | ● | ○ | | | `Semijoin(Col, FP)` | G |
| sort order / layout guarantee | | ● (legality) | | | | `dst_ordered` | G |
| bag vs set | ● | | ○ via carrier | | | `Demand` | G |
| path identity / WALK vs TRAIL | ● | | | | | DataFusion = WALK; Quack = WALK | G |
| consumer demand | ● | ○ | ○ | | | `demand::classify` | G |
| parameter slot | | ● | ○ | | | op index of the sentinel `Pred` (probe) | G |
| parameter value | | | | | ● (in `Pred`) | patched into `Program` | G |
| result shape | ● | | ○ | | | `ResultShape` | G |
| population density / selectivity | | | | ● | | not measured at runtime | S |
| dense / sparse / runs carrier | | | | ● | ○ | dense only (R1) | G |
| gated vs fused lowering | | | | ● | | `lower` / `lower_fused` | G |
| CSR / CSR5 | | | | ● | ○ | none | S |
| push / pull | | | | ● | | none | S |
| early / late materialisation | | | | ● | ○ | gated `Pred` | G |
| morsel / extent size | | | | ● | ○ | `execute_extent` | G |
| scratch slots | | | | ○ | ● | `Scratch::for_program` | G |
| prefetch distance | | | | | ● | ndarray kernels | S |
| SIMD width / CPU caps | | | | | ● | `simd_caps()` | G |
| GPU workgroup | | | | | ● | — | S |
| JIT specialisation | | | | ● | ○ | not wired | G |
| bind generation | | ● | | | | **missing** | G |
| bundle generation | | | | ● | | **missing** | G |
