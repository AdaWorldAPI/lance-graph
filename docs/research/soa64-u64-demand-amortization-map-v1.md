# Research RFC: SOA64-U64 — demand-amortized 64 × 64-bit words over canonical addresses

**Status:** RESEARCH HYPOTHESIS / documentation only, **not** a decision, implementation, production admission, performance claim, or permission to merge.
**Date:** 2026-11-07. **Branch:** `research/soa64-u64-demand-stride-v1`.
**Scope:** AdaWorldAPI/lance-graph; AdaWorldAPI/ndarray and AdaWorldAPI/report-rs are comparators/consumers, not targets of this PR.
**Question:** Would an immutable, canonical, source-owned SoA universe *logically addressable as 64 typed 64-bit property words per identity*, with selective field demand and an inexpensive virtual transpose/stride, reduce **total answering cost** relative to narrow-coded SoA and conventional row/column designs?

## 0. Decision and admission status: do not conflate

* **Operator proposal, NOT shipped:** make SoA the invariant, NodeGuid→versioned ordinal order canonical, and 64 × u64 the logical word-addressing surface; physical width, compression and residency may differ. The previously discussed 32-lane resident budget and `CustObject64` logical basis are *not* an approved change to 64 resident physical lanes.
* **Observed today:** DT64's `ClassView` and `WideFieldMask` are field semantics, not lane addresses; `LaneSource::lane_of` returns `LaneAddr{space,lane,scope}`; `BindStore::bind_lanes` refuses mismatched `CoordSpace`, scope, changed mapping/frame/basis. `LaneBinding` holds only descriptors; `LaneSource::borrow` borrows values. Source: [dt64 lane.rs](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-dt64/src/lane.rs).
* **Observed blocker:** `AbiLanes` has only `#[cfg(test)]` constructor due to no authoritative production reader-scope issuer; it currently admits `I32/U32` lane kinds, not `U64`. Source: [lane_source.rs](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-report/src/lane_source.rs).
* **DR today:** `Extent::DR64_CUBE`, `DR256_SQ_X64`, `DR256_GLOBAL_X64` exist; last means 65,536 identity positions, 256 property positions, 64 predicates, not a shipped `256×256×16` layout. `Scope{id:u64}` is a value, not an authority. Source: [dr.rs](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-dt64/src/dr.rs).
* **Workbook ruling:** [EXO-174](https://linear.app/exo-red/issue/EXO-174/decision-rpt-xl-delta-03-the-workbook-is-a-projection-over-the-64k-soa) already specifically rules u64 tagged cells and column projections **for the workbook decision**, with open tag/codec/ingress/canonical authority gates. This research must not quietly re-decide EXO-174, or extend its workbook-specific rule to all physical kernels.
* **Frozen decisions:** [EXO-186](https://linear.app/exo-red/issue/EXO-186/decision-shared-ledgers-move-to-linear-epiphanies-agent-log-and) routes new epiphanies to P-EXO-12, agent-run messages to P-EXO-10, decisions to P-EXO-11. This document is a research plan, **not** an epiphany- or decision-ledger entry.

## 1. Exact hypothesis and null models

**H1 (demand amortization):** for a workload with requested field set `D`, the incremental traffic and computation track `|D|`, selected population, decoding and access pattern, rather than total logical field width `64`. More precisely, the *logical* 64-field basis does not force reading or hydrating all 64 physical lanes.

**H2 (stride / virtual transpose):** if each requested property is a source-owned contiguous lane ordered under the *same witnessed coordinate space*, the virtual object at identity ordinal `i` can be `(lane_A[i], ..., lane_E[i])` without constructing AoS rows. Descriptor translation may be no-alloc; actual reads, cache lines and register shuffles are not free.

**H3 (total-cost crossover):** at some query mix and demanded-field density, fixed-width u64 SoA could beat compressed/narrow-coded SoA, row AoS, and ordinary columnar layouts *after* including cold ingress, binding, decoding, persistence, cold/warm scan, cache traffic, prefetch, selected writes and egress. No crossover is guaranteed.

**H4 (epistemic amortization):** canonical WHO identity and ordinal G2L mapping, WHAT field basis, WHERE selection/presence and WHY terminal demand are validated once per compatible version/scope; subsequent positional sweeps perform no per-element identity/schema lookup. This is measurable separately from storage and SIMD effects.

**Nulls:** (N1) narrow u8/u16/u32 or compressed columns dominate because bandwidth/cache outweigh decode costs; (N2) AoS wins when virtually all 64 fields are demanded per object; (N3) sparse/unstable identities and tenancy make canonicalization more expensive than indexed reads; (N4) strong compression yields smaller total storage *and* faster scans; (N5) 64-bit tagging introduces overflow, exactness or governance costs that outweigh convenience; (N6) logical stride still incurs gather/cache-line amplification in ordinary read patterns.

### Claims NOT permitted

Never claim u64 is faster *by definition*, that typed words can encode arbitrary strings/decimals exactly, that logical width equals physical residency, that a virtual transpose is a physical transpose, that source lanes automatically align, or that a benchmark licenses production `Gather`/writer/reader-scope admission.

## 2. Ownership and facts at every layer

| Layer / question | Established owner | Current evidence | Open admission seam | Required proof |
|---|---|---|---|---|
| X: NodeGuid identity, canonical order | canonical source owner + DR address contract | DR `LocalMap`, `MapTable`, `FrameStore`; position tile math after ordinal assignment | Authoritative NodeGuid→ordinal minting and global snapshot/version, no silent renumber | permuted IDs, append, reorder, stale seal, replay |
| WHO: authorized population | RBAC/OGAR issuer; DR sealed reader scope | `Scope{id}` currently numeric; `AbiLanes` test-only | nonforgeable actor/class/field/row/frame scoped witness | foreign tenant, revoked grant, forged same integer |
| WHAT: field types and semantics | DT64 `ClassView`, `MethodBinding`, `WideFieldMask` | `MAX_FIELDS=256`; `ValueDemand::{MapOnly,Values}`, `MethodBinding::footprint` | typed u64 semantics/codec and audited field basis | schema reordered under same size, unknown field, stale revision |
| WHERE: row selection/presence | Quack set algebra / mask-RISC; source owner for presence | u64 mask words and `ndarray::simd_masking_ops` | distinguish full-range from sparse reusable selection, property-presence from row-membership | null, unknown, false, duplicates, sparse holes |
| WHY: terminal + backward dependency | admitted method and evaluator | `MethodBinding::reads`, `footprint`, `ValueDemand` | sound demanded-field closure incl. guards/effects | omitted required field must refuse, not silently skip |
| Address and frame | DR and DT64 socket | independently checked route and frame versions | no conflation of clocks; canonical source coordinate witness | shuffled B lane and equal-looking numeric IDs |
| Physical lane values | SoA owner | borrowed `LaneSource`; `AbiBatch` resident adapter | u64 `LaneRef` ingress/binding/terminal separately admitted | exact u64/decimal/text encoding, no lossy f64 |
| Algebra + fold | Quack → mask-RISC → ndarray | mask fold facilities; `ndarray::simd_stride` narrower helper | general multi-lane virtual transpose is not already implemented | scalar oracle, no intermediate row objects |
| X×Y relationship | algebra; DR admitted relation/mapping | symbolic relations expressible in principle | cross-coordinate mapping/gather explicitly NOT admitted | two sources with same labels but different row order; refusal |
| Persistence | source owner / D4-S-INGRESS | report-rs is a downstream consumer | durability + D4 writer/layout witness | crash/replay/freshness, old snapshot |
| Output | terminal consumer (e.g., report-rs) | Cockpit/Hub/Matrix are scoped lenses | do not confuse formatting with zero-cost execution | typed result, export parity, authorized fields |

References: [D-LANESOCK plan](https://github.com/AdaWorldAPI/lance-graph/blob/main/.claude/plans/d-lanesock-dr-dt64-lane-socket-v1.md); [EXO-46](https://linear.app/exo-red/issue/EXO-46/a1-decide-counted-globalfocus-binding-membrane-for-production); [EXO-140](https://linear.app/exo-red/issue/EXO-140/admission-b-call-sites-unclassified-semijoin-equ32via-gather-and-no); [EXO-135](https://linear.app/exo-red/issue/EXO-135/crumb-trail-what-may-be-built-before-execution-under-what-witness-and).

## 3. Representation axes: do not collapse them

* **Logical field width:** 64 slots in a `CustObject64`-style projection (research candidate); not a 64-field resident object. DT64 currently permits up to 256 field positions.
* **Physical lane count:** 1, 3, 8, 32, 64 explicitly varied. The prior 32-lane working-set budget is one candidate, not automatically retired.
* **Word width:** `u64` logical execution vocabulary; `u8/u16/u32/i32/u64`, bitpack, dictionary, RLE/delta and compressed physical variants are competitors.
* **Value vs reference:** signed/unsigned 64-bit, scaled integer, tagged dictionary ID, content handle, identity ordinal, relation handle, mask word. A tag budget must specify width, signedness, overflow, NaN/non-numeric, absent/unknown and collision refusal.
* **Residency:** cold remote/disk, mapped, cache-resident, register/local tile. Logical address does not promise physical residency or no fetch.
* **Selection:** contiguous range, u64-packed population mask, selection vector, logical predicate, witnessed G2L mapping. Each has a separate cost; full tile needs no compulsory survivor mask.
* **Order:** NodeGuid global order vs source-local workbook row order; must be witnessed when equal and explicitly mapped when different. No implicit numeric-`i` equality across sources.

### Baseline memory arithmetic (dense only, no headers/presence/compression)

```
bytes = identities × physical_lanes × bytes_per_word
64 lanes × 64-bit = 512 B per identity
64 lanes × 64 identities = 32 KiB
64 lanes × 256 identities = 128 KiB
64 lanes × 65,536 identities = 32 MiB
1 u64 lane × 256 identities = 2 KiB
3 u64 lanes × 256 identities = 6 KiB
32 u64 lanes × 256 identities = 64 KiB
64 u64 lanes × 256 identities = 128 KiB
```

Competing 64×u32 = 16 MiB / 65,536 identities; 64×u8 = 4 MiB; packed presence and side tables extra. Narrow physical values may win decisively on cache/bandwidth. Conversely a virtual descriptor for 64 properties can be cheap regardless of how many values live in memory.

## 4. Storage-theorem inversion: what is proposed and what would falsify it

Conventional storage optimization often minimizes bytes-per-record and compressibility; proposed optimization minimizes **cost per authorized, demanded answer**:

```
T_answer = T_ingress_amortized + T_canonicalize_amortized + T_bind
         + T_source_fetch + T_decode + T_select + T_fold + T_terminal
         + T_egress + T_invalidations
Bytes_touched = Σ_{p ∈ D} ceil(cache_lines_touched(p, selection)) × cache_line_size
              + mask/index/descriptor metadata
```

Neither formula proves a winner. Evaluate separately:
* **cold vs warm**; L1/L2/L3, main RAM, disk/remote; cache miss and bandwidth ceilings;
* **bulk scan vs point/random vs monotone sparse sweep** (prefetch helps sequential but not arbitrary accesses);
* **selectivity**, true result size, null/presence distribution, branchiness, semantic tags;
* **read-only vs update-heavy**, snapshot churn and dictionary maintenance;
* **tenant and schema cardinality**, versions, column-level source ownership, row-order stability;
* **index / mapping preparation**, memory costs of G2L, reader scope issuance, cache residency;
* **serialization** (only terminal demanded values) and no row assembly on the hot fold path;
* **SIMD architecture:** AVX-512 8×64-bit per 512-bit register, AVX2 4×64-bit, NEON 2×64-bit, scalar; actual implementation and dispatch must be independently verified. A 64-bit opcode is not a universally 64-wide SIMD lane.
* **zero-copy vs bandwidth:** borrowing avoids allocation/copy but not cache misses and memory traffic; strided address calculations are cheap, strided memory misses are not.
* **virtual transpose:** swapping descriptor interpretation is cheap; assembling cross-lane registers may still cost shuffles and fills; row-major traversals may favor AoS at high property demand.

Comparison arms, all same data/semantics/independent oracle:
A. conventional `AoS [u64;64]`, no heap row copies in the timed fold;
B. dense canonical `SoA [u64;N]×64`;
C. `SoA u8/u16/u32` with verified decode and exactness;
D. compressed/bitpacked/dictionary SoA (same semantics);
E. virtual 64-word lens over borrowed source-owned lanes, with physical widths/residency stated;
F. selective hot tile/register transpose **only if a standalone additional measured arm**; never bury the cost.

## 5. Experimental matrix with preregistered falsifiers

| Axis | Cases | Primary measurement | Negative test |
|---|---|---|---|
| identity | 64, 256, 257, 513, 65,536 | G2L count; tile arithmetic, bound time | same labels / wrong NodeGuid, shuffled row order |
| demanded fields | 1, 3, 8, 32, 64 of 64 | bytes touched; cache misses; fold ns | non-demanded property poison/panic if fetched |
| selection | full contiguous, 1%, 10%, 50%, 100% | mask/vector overhead, prefetch, count | full-range synthesized mask as hidden cost |
| access | monotone scan, random points, few-run sparse | L1/L2/LLC miss rates, bandwidth | false co-addressing masked by same ordinal |
| representation | u8/u16/u32/u64, tagged, compressed | storage size, decode ns, correctness | overflow, missing, exact decimal, invalid code |
| action | COUNT/EXISTS, masked SUM, 5-field predicate, requested 5-field terminal | hot fold ns, invocation/allocation counts | row allocation or hidden tuple builder |
| concurrency | 1/2/4 threads; mixed tenants | throughput, locality and scope refusals | cross-tenant label collision |
| mutation | stable frame, source reorder, revision/field change, revoked scope | invalidation/rebind, stale refusal | stale binding returns plausible value |
| hardware | x86 AVX2/AVX512 where present; AArch64 NEON; scalar | ISA parity, power/traffic if instrumented | unsupported ISA silently fallback without disclosure |
| egress | scalar, mask, relation ref, 5-field view | terminal bytes, time; scope parity | output includes hidden/non-demanded fields |

**Pre-register** fixture seeds, CPU/rustc/features, exact operation, time units and an independently implemented oracle *before* running any candidate arm. Benchmarks are cold/warm separately and report ns/op or µs/batch as dictated by repo's timing doctrine. Never publish a misleading millisecond-as-fold measurement. Cost report includes memory and allocation counts even when latency is low.

### Smallest decisive fixture (first executable lab, later PR)

* 513 deterministic, uniquely sourced NodeGuid identities; canonical frame snapshot and authorized WHO scope.
* A and B have two source-owned ordered lanes; B deliberately reordered in adversarial arm.
* One field basis with 64 virtual properties, of which 3 are demanded; include absent/unknown and an exact 64-bit boundary value.
* Three 256-address tiles (256+256+1); one complete and one filtered population.
* `COUNT WHERE A=k`, `EXISTS`, then `SUM(B) WHERE A=k` under checked numeric bounds; one demanded `(A,B,C)` egress view.
* Independent scalar reference; record `G2L lookup count per admitted tile sweep = 0`, hot allocations and source bytes touched.
* **One falsifier:** mark B as a different `CoordSpace` and verify typed `CoordinateSpace` refusal; no silent cross-source gather.
* **One counterexample:** compare dense u64 vs narrow/compressed on sparse 1-field workloads and publish if narrow wins.

## 6. Boundaries and regression safeguards

1. No production constructor for `AbiLanes` and no typed u64 `LaneRef` admission without an authoritative scope issuer and independent admission work.
2. Preserve ADMISSION B, Gather ban, NotASide, OP-14; symbolic X×Y expression does not license computing a hidden relational cross-gather.
3. A `WideFieldMask` is a **field** mask; row survivor mask and presence are separate planes. `None`, zero, unknown, error, false, null, codebook miss are not interchangeable.
4. Field binding uses DT64-resolved `FieldRef` basis, not a duplicate source view or field position interpreted by local assumptions.
5. Route and frame versions are a pair, not a forged single clock; row reorder/version change invalidates a read.
6. Never infer authorization from an integer `Scope`, a valid DR address or ZITADEL login alone.
7. No source-owned value migration into report-rs. Cockpit/Hub/Matrix remain terminal presentation lenses.
8. No surprise `u64` tag encoding or decimal claim: specify payload/range and independent exact-value oracle.
9. Keep `ndarray::simd` public facade as execution substrate. `simd_stride` is a positional helper, **not** proof of general cross-lane transpose or the universal mask executor.
10. A research PR may cite hypotheses and compare prior art; it may not alter normative architecture, enable a feature, introduce a second storage authority, or declare new measured results.

## 7. Research lanes: meticulous question-to-owner mapping

| Lens / angle | Key question | Required outcome | Existing issue |
|---|---|---|---|
| algebra / Datalog | Can `X×Y` be callable without pair materialization; which expressions lack lawful executor? | expression/realization/refusal matrix | [EXO-115](https://linear.app/exo-red/issue/EXO-115/quack-contract-set-algebra-route-only-the-res-expressible-filter), [EXO-108](https://linear.app/exo-red/issue/EXO-108/gap-set-semantics-distinct-over-a-projection-and-bag-count-over) |
| canonical order | Is ordinal derivation arithmetic after one witness? | identity/position proof + lookup counters | [EXO-46](https://linear.app/exo-red/issue/EXO-46/a1-decide-counted-globalfocus-binding-membrane-for-production) |
| scope / security | Can reader scope be issued only from verified login + scoped decision? | forged/foreign/revoked refuse | [EXO-174](https://linear.app/exo-red/issue/EXO-174/decision-rpt-xl-delta-03-the-workbook-is-a-projection-over-the-64k-soa), D-LANESOCK-4 |
| DT64 / method | Does `MethodRef` select only demanded fields after basis validation? | field change/semantics test | [EXO-121](https://linear.app/exo-red/issue/EXO-121/d-qr-cap-classviewwidefieldmask-relational-isa-binding-capstone) |
| SIMD / cache | Are 64-bit loads cheaper in actual demand patterns? | SIMD/ISA, cache, shuffle and traffic tables | [EXO-134](https://linear.app/exo-red/issue/EXO-134/goldstaub-count-invocation-and-allocation-boundaries-before-optimizing) |
| DR focus | What is G2L prep cost, frame invalidation and tile-rotation cost? | cold/warm separated counts | [EXO-151](https://linear.app/exo-red/issue/EXO-151/d-drf-immutable-dr64-focus-leased-foreign-cache-sparse-writeback) |
| source ingestion | Who owns canonical u64 and exact value/codebook? | provenance/codec refusal matrix | [EXO-174](https://linear.app/exo-red/issue/EXO-174/decision-rpt-xl-delta-03-the-workbook-is-a-projection-over-the-64k-soa), [EXO-185](https://linear.app/exo-red/issue/EXO-185/brief-pr-33-acid-test-dt64-simd-stride-audit-read-only) |
| workflow consumer | Do 3 UX lenses agree without separate evaluator? | source/parity witnesses | [EXO-164](https://linear.app/exo-red/issue/EXO-164/drf-rpt-lab-xverweis-binding-dr64-shootout-arms-a-oracle-b-lance-c) |
| epistemic provenance | What may be constructed in preparation, and promoted by which witness? | receipts/issuer/refusal catalog | [EXO-135](https://linear.app/exo-red/issue/EXO-135/crumb-trail-what-may-be-built-before-execution-under-what-witness-and) |
| agent coordination | How do independent experts hand off evidence without rewriting shared files? | EXO-50 separate comment envelopes | [EXO-50](https://linear.app/exo-red/issue/EXO-50/agent-message-bus-gpt-xgrok-ccsession-id), [EXO-186](https://linear.app/exo-red/issue/EXO-186/decision-shared-ledgers-move-to-linear-epiphanies-agent-log-and) |

## 8. Reuse and boundary references

* [EXO-50 agent message bus](https://linear.app/exo-red/issue/EXO-50/agent-message-bus-gpt-xgrok-ccsession-id) is **coordination only**; new append-only messages are comments with unique MESSAGE-ID and complete CC session_id. Read comments at session start; no implied push/poll delivery.
* [EXO-186 governance](https://linear.app/exo-red/issue/EXO-186/decision-shared-ledgers-move-to-linear-epiphanies-agent-log-and): independent new decisions/epiphanies belong to Linear, not `EPIPHANIES.md` or `AGENT_LOG.md`.
* [EXO-174 workbook u64 ruling](https://linear.app/exo-red/issue/EXO-174/decision-rpt-xl-delta-03-the-workbook-is-a-projection-over-the-64k-soa) is binding **within its existing scope**; `report-rs#33` remains draft and has no value-execution admission.
* [EXO-185 read-only #33 audit](https://linear.app/exo-red/issue/EXO-185/brief-pr-33-acid-test-dt64-simd-stride-audit-read-only) has an explicit no-code/no-PR mandate; this independent lance-graph research PR must not be presented as a result of that audit.
* [EXO-164 benchmark](https://linear.app/exo-red/issue/EXO-164/drf-rpt-lab-xverweis-binding-dr64-shootout-arms-a-oracle-b-lance-c) has its own gated arms, fixtures and workstream. The present candidate comparisons are a **separate hypothesis map**, not benchmark results or authority to modify EXO-164.
* [EXO-140 Gather/ADMISSION B](https://linear.app/exo-red/issue/EXO-140/admission-b-call-sites-unclassified-semijoin-equ32via-gather-and-no) remains unresolved.
* [EXO-134 invocation cost](https://linear.app/exo-red/issue/EXO-134/goldstaub-count-invocation-and-allocation-boundaries-before-optimizing) warns against optimizing kernels while ignoring invocation/alloc overhead.

## 9. Deliverables, staged gates, and explicit refusal to overclaim

**R0 — this PR:** source-linked contract map, quantitative baselines, alternatives, prompt pack, preregistered falsifiers, missing-interface register; **no code**.

**R1 — future lab (separate authorized PR):** synthetic 513-identity fixture, 64 logical fields, 1/3/8/32/64 field demand, full and sparse tile; scalar oracle, typed refusals, per-arm benchmark harness and separately counted cold/warm/egress costs. Must be approved and linked in Linear by the owner.

**R2 — future decision:** compare actual crossover and counterexamples against architecture; post findings to P-EXO-12 only if provenance/admission gate is met; post an operator decision to P-EXO-11 separately. No implied production deployment.

**Kill criteria:** unaccounted gather; identity/field scope leak; incorrect exact values; unexplained allocation or value copy on hot path; oracle mismatch; benchmark hides physical data or prep costs; required negative test stays green when guard disabled.

**Success does NOT mean u64 wins universally.** Success means we can state a reproducible, honest boundary: for each workload family, which physical representation yields the lowest end-to-end cost under the immutable address and security contract.

---
Companion: [SOA64-U64 research agent prompts](./soa64-u64-research-prompts-v1.md).
