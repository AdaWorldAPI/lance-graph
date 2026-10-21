# QR256 / Inception: pre-registered hypothetical test paths

**Date:** 2026-10-21. **Status:** RESEARCH-ONLY TEST DESIGN; not executed. **Owners to review:** relalg, contract, R2IL, mask-risc, Quack, mork-pathmap-probe, crepe-plan-probe, Thought Git.  
**Sources:** companion notes in this .arxiv shelf: [2201.06508](2201.06508_gaussian_greedy_reversible_circuit_excerpt.md), [2610.17934](2610.17934_chordal_codes_elimination_order_excerpt.md), [2610.08701](2610.08701_gaussian_partial_pivot_growth_warning_excerpt.md), [2609.28901](2609.28901_quantum_gaussian_reversible_proof_excerpt.md), [2304.04332](2304.04332_egglog_equality_saturation_datalog_excerpt.md), [Four Russians](2026-10_four_russians_m4ri_reverse_lookup_excerpts.md).

## 0. Representation and authorization contract

- The **8 KiB 256×256 packed Boolean matrix** is the hot, resident execution state. Its axes must be *typed* (e.g. method×entity membership or source×target adjacency). One does **not** represent a general binary relation by a method×entity unary plane alone.
- A `u8` method palette supplies up to 256 **method selectors**, not 256 matrix-valued truths. A pair of u8 axes forms a 16-bit local coordinate, but its bits require storage or a separately verified virtual backing.
- The **cold sidecar** records global↔local mapping, source generations, borrowed SoA row ordinals, BEFORE/AFTER snapshots or hashes plus reconstructible snapshots, dirty deltas, typed rewrite-law IDs, proof references, and authorized writeback positions. It is not an inner-loop operand.
- During PREPARE, values may be read from the **same** source table and resolved by an attested frozen codebook; counted/witnessed same-table seeks are permitted under [ADMISSION B / #1640](https://github.com/AdaWorldAPI/lance-graph/pull/1640). **Cross-table foreign-value-directed position reads remain forbidden**, including preparation. A preallocated map does not automatically confer admission. Future exceptions demand an explicit new decision.
- Existing zero-copy index map is lab-only [#1659](https://github.com/AdaWorldAPI/lance-graph/pull/1659), co-addressed multi-side join [#1658](https://github.com/AdaWorldAPI/lance-graph/pull/1658), 256-rule CSE [#1653](https://github.com/AdaWorldAPI/lance-graph/pull/1653), Boolean tile [D-RMAP-2](../.claude/plans/d-rmap-2-context-tile-algebra-probe-v1.md). Do not substitute GraphBLAS, C/MKL, or a new runtime for already-probed native kernels.

**Cost accounting:** cold bind/build/index; source bytes touched; hot Boolean folds; cache misses; output tuple materialization; dirty update; proof DAG bytes and checker time; writeback/commit separately. The motivating 20 ms preparation + 2 s operations is a user hypothesis, **not a measurement**. For 1,000,000 operations, amortized preparation is 20 ns/operation if bind is 20 ms; for 4 billion operations it is 5 ps/operation.

## 1. Paths / hypotheses / independent oracles

| ID | Hypothesis / path | Oracle | Can-fire (required) | Can-stay-silent / kill |
|---|---|---|---|---|
| P1 | **Same-ordinal AD/Entra/EXO:** witnessed shared identity means A[i],B[i] borrow exact source rows, without joins | independent immutable GUID→row map | distinct fields across 3 sources bind correctly and reused | same row number but different external ID MUST refuse; separate deletion/reorder/stale generation |
| P2 | **Many-to-many group membership:** bound CSR/offset index plus Boolean QR register answers EXISTS without tuple enumeration | scalar multiset join and transitive memberships fixture | preserved duplicates for COUNT/BAG via ordinal sidecar | set truth does not imply multiplicity; group cycle must terminate |
| P3 | **Cross-ontology:** Mouse→Cheese→Milk→Cow and Cheese→Käserei use explicitly typed identity bridges | scalar graph path with source lineage | no repeated global resolution after bind | mismatched ontology homonym and out-of-focus escape MUST refuse/continue; no value-steered foreign read |
| P4 | **Four Russians GF(2):** block-pivot XOR LUT or reverse-index lookup amortizes linear elimination | independent GF(2) Gaussian reference | equal rank, row space and reconstructible pivot schedule | all-zero/singular inputs and tiny matrices must not claim inversion or speculative speedup |
| P5 | **Boolean tile:** bitwise OR/AND relational composition beats repeated raw joins after reuse | scalar Boolean matrix multiply and existing BitMat | 1,8,64,256 focus runs and densities 0.1%,1%,10%,50%,100%; exact tuple/mask equality | two distinct paths to same target: OR=1, XOR=0; bag answer differs from set answer |
| P6 | **Gaussian substitution / E-graph:** a=2b,b=2c,z=2a → z=8c; 256 templates × 4096 frames = 1,048,576 applications | exact symbolic substitution evaluator, Datafrog for typed relational subset | shared proof DAG and lower source-read count; exact equivalence under declared laws | f64 reassociation, u8 saturation, contradictory cycles, nonsharing source demand MUST NOT produce false exactness/compression |
| P7 | **Chordal ordering:** PEO on proven chordal primal graphs reduces intermediate fill | independent PEO/triangulation checker and GF(2) row operations | identical results, reduced peak fill, count cost | 4-cycle without chord; cyclic join that survives semijoins; arbitrary ontology not assumed chordal |
| P8 | **GEPP robustness:** f64 elimination correctness requires conditioning/growth checks | high-precision linear solver | well-conditioned solved LGS agrees | fixed absolute 1e-12 cutoff fails scale invariance; failure cannot contaminate exact QR semiring |
| P9 | **Reversible proof sidecar:** elementary GF(2) row steps invert; lossy Boolean OR needs extra BEFORE evidence | independent replay checker over raw 8 KiB state | before→rule schedule→after, inverse when legally reversible | sidecar holding only final OR matrix cannot reconstruct originals; stale source or no write acknowledgement cannot mark committed |
| P10 | **Dirty algebra:** `CandidateDirty`, `ActualDirty = BEFORE XOR AFTER`, `CommittedDirty` distinct | full recompute from clean snapshot | changed input updates only affected legal dependencies | multiple supports + single deletion must NOT delete surviving reachability; no-op delta remains silent |
| P11 | **Rotation as indexed view:** 3×3 CCW 90° through coordinate arithmetic; 256×256 QR without physical copy | explicit scalar permutation and four-rotation inverse | rotated view matches oracle; 4×90° returns original bytes | rotation of source×target directed relation may change semantics; demand axis/type witness |
| P12 | **Amortization:** reused frame over 1,000,000 folds outperforms rebind each time | identical answers and independent timing harness | break-even observed with cold binding separated from repeated folds | if cold index, staging or proof grows with every fold, reject speedup claim |

## 2. Rotation (user's exact fixture)

Input:
```text
1 2 3
4 5 6
7 8 9
```
Requested CCW 90° rotation:
```text
3 6 9
2 5 8
1 4 7
```
For N×N matrix in row-major coordinates (zero-based): `new[r,c] = old[c,N-1-r]`. Equivalently an old cell `(r,c)` moves to `(N-1-c,r)`. Check: old(0,2)=3 → new(0,0); old(2,0)=7 → new(2,2)? **No:** under this formula old(2,0)→new(2,2)=7, while the actual fixture has new(2,0)=1 and new(2,2)=7. This explicitly guards against mixing lookup and movement directions.

For a **logical/view rotation**, do not move bytes. Bind `QRView {base, axis_types, transform: RotateCCW90, generation}`; emit transformed address arithmetic at access time. Composing two quarter-turns yields 180°; four returns identity. Only physicalize if the layout economics warrant it. Track dirty positions through the *same bijection*. A lazy view is zero payload copies, not zero address arithmetic.

**Semantic hazard:** a binary directed adjacency matrix has both axes drawn from the same entity domain, but rotation is transpose plus reversal, **not generally a graph isomorphism relabeling**. A true consistent node relabeling must apply the **same permutation to both endpoints**, `R'[P(i),P(j)] = R[i,j]`; this is not a 90° image rotation. For method×entity, swapping axes changes their types, so a rotated view is only admissible as a physical/presentation transform with explicit swapped axis metadata and inverse mapping. Never interpret rotated bit positions as new facts.

## 3. Schedule / proof shape (illustrative, not ABI)

```text
BIND  frame_id, source_seals[], codebook_digest, palette_version
EXEC  method8, domain_kind, axes, QR256, operands, focus_version
RULE  proof_node_id, law_id, child_refs[], guard_ids[]
DELTA actual_dirty256x256, before_root, after_root
REPLAY inverse_coordinate_map, row_ordinal_refs, deterministic_schedule
COMMIT target_seal, authorization, store_ack  // separate from a proof of computation
```

A proof DAG compresses repeated **structure**; arbitrary independent output facts still require proportional data/entropy. Hashes prove identity *relative to retained data*, not mathematical validity. Verify each rewrite and independent execution oracle; the checker must never trust the producer's claimed `success`.

## 4. Experiment stages and gates

**S0** Documentation review: keep Gaussian REAL_F64, GF2_XOR and BOOL_OR_AND as separate contracts. Inventory the already-existing masked tile and semijoin paths; no duplicated engine.

**S1** Low-risk independent fixtures: 3×3 rotation, 16/64/256 exact Boolean bit rows, scalar closure/compose, two-path XOR-vs-OR counterexample, stale witness refusals. Run these before benchmarks.

**S2** Cold optimization research: egglog outside production, bounded fuel, rewrite/equality proof reconstruction on the exact subset, normalized DAG size and signature/provenance check.

**S3** Optional GF(2) and chordal probes: clean-room method of Four Russians, reverse-index LUT, PEO order, numerical growth adversaries. GF(2) oracle remains external if GPL constraints apply.

**S4** Million-fold amortization: 256×4096 shared-rule test plus nonsharing control; binding versus execution versus trace-check overhead, source bytes and RSS. Reject without wall-time parity and all refusal guards.

**S5** Writeback only after contract ruling: do not reinterpret a preparatory gather as admitted; no writes based on hypothetical experiments.

**Success =** correct typed operations; reproducible and verifiable BEFORE→AFTER; mutation/stale/refusal oracles; measured cost reduction net of preparation and trace. **Failure =** wrong truth bit, lost multiplicity/escape, illegal foreign address, nondeterministic witness, unverifiable derivation, silent coordinate semantic change.
