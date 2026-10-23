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