# D-DR3-HOLO-ONNX-v1 — S3a/S3b/S3c Cartesian / Holographic register plan
**Date:** 2026-10-23 · **Status:** PROPOSED research, no DR64³/DR256³ implementation claimed
**Parents:** #1745 S0; #1746 S1–S2 · **Source owners:** D-QR-CAP, D-RELALG-UNIFIED, D-DRMC, D-RMAP/Quack, CE64.

## Thesis
A proposed 3-domain Cartesian relation A×B×C can be evaluated as (i) explicit reference semantics or (ii) a holographically focused, version-witnessed set of existing borrowed relationship views. DR64³ (64³ cells, 32 KiB dense 1-bit equivalent) and DR256³ (256³, 2 MiB dense 1-bit equivalent) are **logical comparison shapes**, not new global identities, implementation ABIs or obligatory allocations.

DR64² is 512 B; DR256² is 8 KiB. Quadratic focus ratio 1/16; full-cube ratio 1/64. A valid **cubic 1/16 focus** requires precisely defined projection, e.g. 64×64×256 / 256³, or a separate documented focus policy. Do not equate 64×64×256 with DR64³.

## Contract and invariants
- **LogicalShape**: typed domains, axes, algebra, value carrier and result demand.
- **FocusWindow**: admissible version-witnessed subset/aperture, independent of global shape.
- **PhysicalView**: optional direct borrowed operands, DR tiles, sparse selection, typed folds or fallback.
- Stable global identity and per-side coordinate witnesses; equal local ordinals alone do not establish equal identities.
- Preserve *three distinct* frame/source-seal, focus and policy scope guards. Reuse existing checks; in D-DRMC-5 Chain::new checks operand-domain compatibility/frame/focus/scope. Do not redundantly reassign these duties to ProofRef.
- Validity accompanies the relation operand. Presence != truth, NULL, Bag multiplicity, source independence or causal evidence.
- No value-derived cross-table positional gather, unauthorized foreign preimage, implicit ontology equivalence or writes through a read view.
- Preserve canonical CausalEdge64 and Palette256 semantics. No invented 18-bit DR³ ID format.
- Existing 64-thought occupancy over 65,536 addresses M[address,thought] is **distinct** from a 64³ relation cube. Require a separately witnessed projection if the two models interact.

## S3a: bounded Cartesian semantic oracle
Define an illustrative typed A×B×C shape, with a carefully selected Boolean/Distinct composition fixture and a typed numeric fixture (do not conflate them).

Required specifications:
1. Axis domains, join/contraction index and associativity premise.
2. Explicit reduction, value carrier and arithmetic/Boolean semiring.
3. Exhaustive or independent scalar evaluation for small bounded fixtures.
4. Snapshot/authority validity and result demand.
5. No eager cube allocation assumed.

Falsifiers: domain mismatch, Bag vs Distinct, a missing validity bit, noncommuting axes/projection, numeric sum-product replaced by Boolean OR-AND, overlapping local ordinals from different global identities.

**Gate S3a:** exact relational parity and declared numerical equality policy; benchmark bind/prepare/traverse/memory/materialize separately, after correctness. No production implementation from this plan.

## S3b: holographic global-to-local focus
The canonical addressing inversion is Thought(t)[Table(a)[i] : Table(b)[i]]: the method chooses operand tables; the same compatible coordinate i addresses them; presence is per operand. This is more specific than generic sparse tensor traversal.

For each focused operator classify:
- EXACT_LOCAL: all required contributors lie in the addressed focus.
- EXACT_EXPANDABLE: required contributors can legally be gathered via admitted coordinate expansion, not forbidden value-steered positional gather.
- APPROXIMATE: only under declared approximate/numerical policy.
- INVALID: focus changes operation meaning.
- UNKNOWN: no verified representation.

A local reduction is not global unless the dependency closure is complete; omitted == false is forbidden. Focus cannot override policy scope. Expansion must preserve index sealing, source identity and permission. The required output materialization remains explicit even if inputs are zero-copy.

Compare: direct register+SIMD vs R2IL lowering vs independent scalar Cartesian oracle. Account bind, index, prefetch, disjoint pages, expansion, copied payload and output materialization.

Mandatory negative fixtures:
- projection before contraction != projection after contraction;
- selected focus omits a nonzero summand;
- two separate tables expose equal ordinals but different identities;
- frame moves, scope revoked, focus crosses allocation boundary;
- no present value vs SQL NULL vs FALSE vs UNKNOWN.

**Gate S3b:** no illegal cross-table gather; source-attested global-to-local binding; exact or legally expandable semantics, else typed refusal.

## S3c: DR256³ validation necessity probe
This is an **independent hypothesis**: a broader cube may help validate cross-focus composition, but is not mathematically required for law certification merely by being larger.

Compare sparse, dense and projected validation against a trusted independent oracle. Test cross-focus consistency, expansion coverage, output type/shape, legal rebinding and larger-domain corner cases. Record limits and overheads; include 256×256×256 logical capacity but refuse an unconditional 2 MiB allocation claim.

Decisions: ADOPT for specified validation tasks / KEEP smaller scalar oracle / DEFER / REJECT. DR256³ is not made mandatory for S3d or S4 simply by naming it in the capstone.

**Gate S3c:** independently demonstrated incremental validation value with cost measurements; otherwise retain as unadopted research.

## Research sequence
Haiku: exact implementations/costs of DR64, DR256, direct register composition and probes (#1685; #1707; #1740); search all branches/ledgers for original 12-stage DR³ plan. Sonnet: target-interface sketch with only necessary fields, falsifier fixtures, stage/PR dependency map. Opus: independent proof obligations and adversarial projection semantics.

Original 12-stage plan not yet recovered by S0's limited search; do not invent historical stages. Coordinate with EXO-50 and code owners before any implementation or ABI change.

## Minimal next executable experiments (NOT implemented here)
- Boolean: three compatible DR64 relations with independent pair enumeration; invalid-domain/Bag/K3 negative cases.
- Numeric: small typed contribution-selection experiment showing that legal zero support can omit work without altering numeric outputs (including NaN, infinity, signed zero and order sensitivity).
- Focus: same operation with complete local support, missing external support, legal expansion, illegal expansion.
Record explicit oracle data, negative disables and end-to-end costs.

## Source crosswalk
#1745, #1746, #1641, #1663, #1685, #1700, #1707, #1738, #1740; .claude/architecture/cartesian-epistemic-self-model.md; .claude/plans/dr-mathematical-compiler-v1.md.
---

## 2026-10-23 Cross-session review amendment: physical bit enumeration ≠ arbitrary gather

**P0:** #1744 and #1745 define competing S0 docs at the same path. This research plan does not choose or replace the canonical source.

### Existing R1 and proposed R2
Per #1744 source reconnaissance, the *logical* 64³ index space X×Y×Z can already be traversed by the existing typed Boolean composition `R:X→Y`, `S:Y→Z` with Y contracted into `R∘S`. It is an **existing binary-composition procedure over a conceptual ternary iteration space**, not a built ternary cube or implemented DR64³ ABI.

A proposed *triangle/closing relation* adds `Q:X→Z` and `(R∘S)∧Q`; it must check both `Q.row == R.row` and `Q.column == S.column`, and frame/focus/scope validity, independently of ordinary adjacent-operand Chain compatibility. Which owner holds this extra admission check is OP-ONNX-11, not settled.

### OP-ONNX-4: bit-position scan must receive an explicit contract ruling
The current code has an apparent contradiction:
- `crates/dr-math-binding-probe/src/relcomp.rs`: `compose` uses `bits.trailing_zeros()` as a row index into an **already bound** 64×64 relation.
- `crates/dir-rmap-probe/src/compose.rs`: `ones(word)` enumerates set-bit coordinates within the DR256 support.
- `crates/dir-rmap-probe/src/qr.rs`: its documented D-RU-6 policy says no inner loop scans bits to obtain a position, instead iterating fixed slots and testing bits.

**Do not silently declare either side wrong or broaden ADMISSION B.** Investigate and request the contract owner's decision on a narrow separation:
1. **Witnessed intra-register enumeration**: indices are bounded local bit coordinates of a sealed already-bound register, with the source/target basis known and stable.
2. **Forbidden data-driven cross-table gather**: a payload value from an unrelated table selects a position in another table without an admitted sealed mapping and explicit counted preparation.

Any admissibility proposal must show the frame/coordinate identity, scope authorization, bounds, no hidden foreign read, unchanged Bag/K3 semantics and measured alternatives.

Falsification probes:
- Addressed loop and `trailing_zeros` enumeration agree on every set bit, including empty/full rows, same frame, same scope, same declared coordinate allocation.
- Stale binder or another register's row is refused, even if local ordinal values coincide.
- A relation payload value masquerading as a target table index is refused.
- Compare scan-vs-coordinate-test costs on sparse and dense masks; do not let performance alone decide legality.

### Focus completeness is a separate admission premise
A focused local view must either contain every value that contributes to the declared numerical result, be expanded through an **admitted** coordinate mapping, or refuse. Even a support-certified omission of a numerical zero may not permit omission of IEEE `0*Infinity` or other semantically exceptional contributions.

The 1/16 ratio may describe a 64×64×256 focus over a 256³ logical domain, not a DR64³ cube. Use LogicalShape / FocusWindow / PhysicalView distinctly.

**Gate S3a/b:** typed contracted-oracle parity, bounded admitted addressing, independent negative-gather tests and explicit focus coverage. S3c's larger reference domain is observational validation, not a path from finite sample to universal mathematical theorem.

---

## 2026-10-23 shared compiled-artifact identity gate

Source-backed `AdaWorldAPI/ndarray` Burn MatMul currently selects `CompiledAttention` from a d_head-keyed cache and `CompiledLinear` by matrix shape; both omit full current model/layer/weight/codebook identity verification. Existing `bgz-tensor`, `helix`, `bgz17` palettes are implemented but not all interchangeable.

**S3b holographic global→local binding must not make the same mistake.** A focus/window is not an authority grant to use a compiled table or quantizer. The relevant model graph/op/layer, source-issued weight identity, codebook/calibration generation, assignment provenance, snapshot frame, scope and value-semiring must all be attested by the correct existing owner before any borrowed DR or compiled numerical view is eligible.

Keep three distinct notions:
- Co-addressed Boolean *presence* and legal support enumeration.
- A numeric value lane with an admitted reduction/semiring.
- A registered model-specific compiled approximate artifact with a separate quality/identity contract.

**Falsifiers for the global-to-local/cube adapter:**
- Two model layers share the same dimensions and DR focus but not weights: fail closed rather than reusing another layer's compiled kernel.
- A valid local register support is rebound to a different codebook generation: refuse and force re-attestation.
- Existing 64³ Boolean relational composition is unchanged without a numeric method; only a separately validated typed mapping may connect to 256² palette lookups.
- Finite-domain Helix metric proof (e.g. 256³ exhaustive triangle check for circular distance) does not establish DR256³ relation composition, numerical MatMul or ONNX compatibility.
- Out-of-focus numerical contributions must expand/fallback/refuse even when the cached palette assignment appears complete.

**P0 coordination:** three competing S0 docs #1744/#1745/#1750 need EXO-50 owner decision. This branch changes no CausalEdge64/DR ABI or other owners' code. #1749 owns the companion reuse-and-cache-admission research plan.

---

## 2026-10-23 Optimum relationship (#1751): DR focus is optional for one-layer numerical compilation

Separate [PR #1751](https://github.com/AdaWorldAPI/lance-graph/pull/1751) proposes offline Optimum ONNX export and an existing BGZ/ndarray compiled linear candidate. **This work does not require a DR64³ or DR256³ implementation.** DR global-to-local focus remains an optional later physical view, only for operations with a witnessed relational/support dependency.

If a future Optimum-compiled BGZ sidecar is eligible for DR-assisted execution, ClassView/DR must verify source-issued model/layer/weight/codebook identity and coordinate/frame/focus/scope ownership. Neither tensor initializer names nor shape-matching palette cache entries are sufficient witnesses. Do not turn a palette-code index into a cross-table positional gather or conflate it with the identity axis.

This is a dependency clarification, **not** DR implementation authorization or a claim of numerical parity. Current S0 conflict #1744/#1745/#1750 persists.
