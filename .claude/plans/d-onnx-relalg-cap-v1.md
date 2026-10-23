# D-ONNX-RELALG-CAP — unified architecture and evidence gate
**Date:** 2026-10-23  
**Status:** PROPOSED / documentation only. No implementation, ABI, operator coverage, speedup or adoption is asserted.

## Mission
Reuse ONNX model/graph/function representation and versioned operator vocabulary rather than creating a competing computation graph. Use existing lance-graph-relalg as a mathematical *polyfill/planning* landing zone under the D-RA-2 decision: canonical single-table set algebra and rewrite laws belong to lance-graph-contract, while relalg retains relational planning and lowering.

Preserve ClassView, existing DR64/DR256, CausalEdge64, 64K SoA, Palette256 typed methods, Quack, R2IL, mask-risc, SIMD, evidence/certification and Thought-Git ownership. No additional general executor, new global identity layout, implicit cross-table positional gather or mandatory tensor/cube allocation.

## Convergence thesis
**One declared mathematical contract; three competing physical plans:** Tensor (dense GEMM/Conv/Attention-style kernels), DR (deterministic Boolean support/address selection + typed numeric folds), Hybrid (DR-selected support with numerical microkernels). Physical plan selection is not a mathematical rewrite certificate. Performance wins must be measured; eligibility is established before cost selection.

ONNX semantic adapter → existing contract/RelAlg mathematics → optional Global-to-Local focus/bind → {Tensor, DR, Hybrid} → existing physical backends → independent oracles and appropriately scoped certification. Avoid requiring every ONNX node to create a DR view.

The canonical holographic addressing inversion is Thought(t)[Table(a)[i] : Table(b)[i]]: operand tables are named first, a compatible coordinate selects their rows, and per-table presence controls inclusion. Not equivalent to arbitrary sparse execution, ontology equivalence or value-steered foreign gathers. Preparation-time attested same-table seeks remain subject to ADMISSION B; cross-table positional gathers remain prohibited.

Separate **LogicalShape**, **FocusWindow**, **PhysicalView**, and **ResultDemand** (Exists/Count/Rows/Tensor) conceptually. They are not new ABI types without justification. Every supported operator requires type, shape, domain, semiring, multiplicity, reduction, effect, numerical/error and binding/authority premises.

## Geometry and epistemic non-equivalence
| Logical Boolean plane | Cells | Dense bit equivalent |
|---|---:|---:|
| DR64² | 4,096 | 512 B |
| DR256² | 65,536 | 8 KiB |
| DR64³ (PROPOSED) | 262,144 | 32 KiB |
| DR256³ (PROPOSED) | 16,777,216 | 2 MiB |

DR64² / DR256² = **1/16**; full DR64³ / DR256³ = **1/64**. A 1/16 cubic-window claim needs a specifically defined two-axis projection or other explicit geometry. Logical shapes do not imply allocations or cache residency.

The existing 64-thought occupancy model M[address,thought] (65,536 × 64 bits) is **not** DR64³. Identity, property, relationship, thought and tensor axis are distinct typed dimensions unless a witnessed mapping proves otherwise.

## Architectural owners and source crosswalk
- D-RELALG-UNIFIED / PR #1641: current capabilities are partial. No general μ/preimage, universal ONNX mapper or common executor is implied.
- D-QR-CAP / PR #1663: ClassView and WideFieldMask select *structural fields*, not causal/relational identity; optional DR geometry is separate.
- D-QSP/D-RFT / PR #1667; PR #1685: lab Boolean/Distinct and Bag composition, including competing physical strategies, not dense numerical ONNX MatMul proof.
- D-DRMC / PRs #1700, #1707, #1738: narrowly typed Boolean/Distinct associativity certificates exist. Independent proof checker, general law coverage and R2IL/Quack production integration remain separate.
- PR #1740: bounded fresh-frame certificate reuse demonstrated in the lab for its admitted law family; not general rebinding.
- PR #1741: inspect latest closed/merge disposition before describing its Julia/egglog map as open or superseded; ONNX does not replace its unification/proof topics.
- Canonical: .claude/plans/dr-mathematical-compiler-v1.md, d-relalg-unified-v1.md, d-qr-classview-binding-isa-capstone-v1.md and .claude/architecture/cartesian-epistemic-self-model.md.
- D-RA-2, ADMISSION B, frame/focus/scope as three distinct guards are binding constraints.

## Stage map (separate docs-only draft PRs)
- **S0** this architecture/recovery map.
- **S1–S2** ONNX intake, typed semantic preservation, minimal RelAlg polyfills and independent oracles.
- **S3a–S3c** typed Cartesian DR64³ probe; holographic focus and expansion; optional DR256³ validation hypothesis.
- **S3d** three-arm Tensor/DR/Hybrid differential and cost benchmark.
- **S4–S5** typed Palette256/cognitive-method probe, ONNX export, cold certification and native specialization evaluation.

Stages S3a and S3b may be developed independently once common contracts exist. S3d need not wait for a DR256³ production path; S4 activation import need not wait for cubic validation. These are reviewable plans, **not implementation authorization**.

## Research ownership and workflow
Haiku: bounded inventories and evidence links. Sonnet: specifications, matrices, documentation/board proposals. Opus: formal semantic choices, adversarial counterexamples and review. Final authorization remains with operator and current code/board owners. EXO-50 for cross-agent coordination; GitHub board ledgers for durable findings.

## Open points, not presumed resolved
- Recover original DR64³ 12-stage plan: no dedicated source confirmed by this limited main/open-PR search; full branch/ledger search remains **OPEN**.
- Verify available ONNX importer(s), official current opset schemas, external-data security, operator families and effect semantics before selecting dependencies.
- Identify actual minimum RelAlg additions by counterexample; do not create an independent AST by reflex.
- Establish projection dependency closure and out-of-focus behavior: exact-local / exact-expandable / approximate / invalid / unknown.
- Establish whether DR256³ adds validation value versus independent scalar or sparse references; no theorem can follow from larger sample alone.
- Distinguish data parity, mathematical proof, binding validity and measured hardware cost.
- Independent proof checker; general snapshot rebinding beyond #1740; numerical operators and production runtime remain OPEN.

## S0 acceptance gate
Documentation-only, no shared owned file overwritten, no board closures or autogenerated indexes manually changed. Opus falsifier review must confirm no second executor, no loss of SQL K3/Bag semantics, no false ONNX compatibility claims, no unwitnessed gathers, no 1/16 cubic misstatement and no unverified speedup. Stage implementation requires separate permission.