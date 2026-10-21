# D-QR-POLY: PowerShell positional template → ordered borrowed QR256 → algebra → replay

**Date:** 2026-10-21. **Status:** DESIGN / CROSS-PR MAPPING, research only. No production implementation or admission.  
**Physical source:** [PR #1659](https://github.com/AdaWorldAPI/lance-graph/pull/1659) (open lab, branch `ccr-0455e606-wmtsor`, head `cd499c4e` at inspection).  
**Logical research:** [PR #1660](https://github.com/AdaWorldAPI/lance-graph/pull/1660) (this docs-only draft).  
**Coordination:** [Linear EXO-50](https://linear.app/exo-red/issue/EXO-50/agent-message-bus-gpt-xgrok-ccsession-id).  
**Prior rulings:** [#1640](https://github.com/AdaWorldAPI/lance-graph/pull/1640) ADMISSION B, [D-RELALG](d-relalg-unified-v1.md), [D-RMAP-1A](d-rmap-separate-tables-thought-palettes-v1.md), [D-RMAP-2](d-rmap-2-context-tile-algebra-probe-v1.md), [#1657](https://github.com/AdaWorldAPI/lance-graph/pull/1657) P0 folding gate.

## 1. Correct the history: #1559 and #1560 are not the PowerShell polyfill

- [#1559](https://github.com/AdaWorldAPI/lance-graph/pull/1559) is OGAR authorized document resolution and RBAC field/row masking.
- [#1560](https://github.com/AdaWorldAPI/lance-graph/pull/1560) is Pearl counterfactual verdict, Rubicon pin and CE64 ledger correction.
- Neither introduced the PowerShell-shaped positional wrapper. The exact `"{0}{1}" -f $laneA[$i],$laneB[$i]` analogy is explicitly documented in the code-level notes of **#1659**. EXO-50's 2026-10-21 tuple/format discussion likewise clarifies that `"{0}..{15}" -f $lane.split` is a *16-position carving instruction*, not a new `QualiaI4_16D` type or a tuple decoder.
- [#1504](https://github.com/AdaWorldAPI/lance-graph/pull/1504) previously demonstrated a separate class-qualified method declaration → Quack/R2IL algebra polyfill. It is prior art for declarative lowering, **not proof** that a reusable position-template wrapper currently exists.

**New missing seam:** a tiny declarative **position-template compiler** between a front-end's named operands and a #1659 `JoinDomain/Side/Run` or same-ordinal row cursor. It needs no string construction, row object materialization, extra interpreter or new graph engine.

## 2. Literal PowerShell model, formalized

The motivating notation is:

```powershell
"{0}{1}" -f $laneG1[$i], $laneG2[$i]
"{0}..{15}" -f $lane.split
```

Interpret the *placeholders* as compile-time slot selectors, and `$i` as a bound ordinal under one witnessed address domain. **Do not** implement PowerShell formatting at runtime. String formatting is only the metaphor.

**Same-ordinal fast path:** `Template([Operand(laneA, i), Operand(laneB, i)])`, where equality of row identity across lanes has been explicitly attested by the same sealed order, source/target identity mapping and generation. Identical numeric offsets in unrelated tables do **not** witness identity.

**Co-addressed multi-side path (#1659):** for local join slot `s`, each side supplies its *own* `off[s]..off[s+1]` and `pos[]`, and its original lane is read as `lane_side[pos_side[k]]`. This preserves source row positions without allocating copies of the source payload. Duplicate side entries remain available to bag terminals. Domain identity and offsets must be versioned.

**16-byte carving template:** a template with placeholders 0..15 selects byte positions of one contiguous borrowed value/lane with a declared 16-byte layout. It does not reinterpret the borrowed bytes as a new type or allocate 16 row objects. Bounds and field/scope authorization apply before reading.

**Important:** `"{0}{1}"` is not a collision-free key encoding for arbitrary values (e.g., `1,23` vs `12,3`). If both are `u8` and a packed key is necessary, `(u16::from(a) << 8) | u16::from(b)` is injective; otherwise use typed pairs or length framing. The intended hot path should avoid forming a key at all.

## 3. Precise ownership: address/prefetch sidecar vs QR state vs replay sidecar

The word **sidecar** describes two different objects and must never hide their cost or lifetime:

| Piece | Resident / cold | Owns | Does NOT own |
|---|---|---|---|
| `BorrowedSoA` | Original source | Immutable Arrow/Lance payload buffers, their true row ordinals, borrowed/refcounted lifetimes | New tuple payload or graph matrix copy |
| `OrderedAddressView` (working name; proposed) | **Prepared, read-side hot metadata** | `JoinDomain`, `present`, `off`, `pos`, borrowed `ScalarBuffer` owners, order/codebook/version/scope witnesses; ordinal iteration and optional one-slot-ahead prefetch | Source payload clones or truth about a foreign table |
| `QR256` | **Hot execution** | Exactly 256×256 bits = 8 KiB for one fully materialized Boolean matrix, plus independently accounted scratch/dirty planes | Payloads, global IDs, bag multiplicity, implicit ontology semantics |
| `ProofReplaySidecar` | **Cold trace/writeback** | BEFORE/AFTER reference or reconstructible deltas, mapping/version witness IDs, template/rewrite IDs, proof DAG, dirty masks, authorized writeback ordinals, commit receipts | Runtime tuple values, permission to bypass auth, automatic completion of external write |

The materialized QR **package** is `OrderedAddressView + QR256`: a borrowed pointer/index/prefetch substrate plus an explicit bit-packed relational projection. It is not 8 KiB in total; only the dense Boolean *plane* is 8 KiB. The view adds `present/off/pos`, indexes, witness metadata and buffer handles. In Rust prefer bounds-checked slices and `ScalarBuffer`/Arc ownership over dangling raw pointers.

**Two QR axis contracts (never conflate):**
1. `Method8 × LocalSlot8`: unary method/entity membership or operation schedule, indexed by a pair of `u8` coordinates (a `u16` key can select a bit).
2. `SourceSlot8 × TargetSlot8` **per predicate**: binary graph adjacency suitable for Boolean `mxv/mxm`, closure and reachability. A bank of 256 such predicates occupies up to 2 MiB of dense truth planes, not one 8 KiB plane.

Pseudocode (an explanatory declaration, **not** a committed API):

```text
template "{0}{1}"       -> ordered borrow operands (side0, side1)
resolve template         -> witnessed per-side ordinal ranges / lane positions
prepare QR[map, focus]   -> OR-aggregate relationship presence into 256² bits
BlasGraph-style OR/AND   -> QR resident mask-risc / existing ndarray kernels
emit demand Count/Exists -> masks/offsets, zero tuple allocations
emit demand Rows         -> borrowed ordinal lens; only materialize if explicitly requested
record proof             -> cold BEFORE, rule DAG, actual dirty XOR, AFTER
authorized writeback     -> separate acknowledged storage transaction
```

## 4. BlasGraph reimagined as QR, not replaced by another graph engine

Existing `crates/lance-graph/src/graph/blasgraph/{matrix,semiring}.rs` implements GraphBLAS-inspired operations using **owned CSR/CSC** and `BitVec` values. It is *not* the 8 KiB per-predicate borrowed QR tile, and its transpose/extract may allocate. Do not claim zero-copy conversion from BlasGraph.

Re-use its **operator semantics** and independent oracle values: Boolean `lor_land`, masked `mxv/mxm`, `ewise`, reductions and transposition. The hot candidate stays the **already existing** QR/bit-tile/mask-risc/ndarray code described in D-RMAP-2; do not implement a new GraphBLAS engine, import C, or duplicate kernels.

`QR_A ∘ QR_B` (two-hop existence) means `C[i,j] = OR_k(A[i,k] AND B[k,j])`, which differs from parity `XOR`. Set- and bag-count semantics must remain explicitly separate. A **logical rotation/transpose view** changes how row/column coordinates are read; treating 90° image rotation as graph edge relabeling is invalid without a separate proof.

## 5. Binding and prefetch economics

Actual #1659 evidence: `Side { generation, present, off, pos, lane }`; `Run` dereferences original values using `lane[pos[k]]`. Dense domain uses `rank+popcount` and sparse uses sorted IDs. `Run::prefetch` issues `ndarray::simd::prefetch_read_t0` for a subsequent run's first lane element. These are source-code facts, not proof that memory latency is eliminated.

**Failed #1659 gate:** on unpruned 50k-row fixtures, preparing `pos` by counting sort of unordered rows costs outside the µs budget; random count/scatter writes dominate. Proposed *but not decided*: attest already-sealed key order and derive run offsets without scatter. See [EXO-114](https://linear.app/exo-red/issue/EXO-114) for order classification and [EXO-46](https://linear.app/exo-red/issue/EXO-46) for the global→focus admission decision. A key ordered for one role may not be ordered for both sides, requiring separate declared indexes or a permutation. **Do not resort borrowed source rows.**

Amortized cost:
`T(K) = T_attest + T_index + T_QR_projection + K·T_mask_fold + T_terminal + T_replay + T_writeback`.
Report indexing/warmup, cache misses and output materialization independently. The 20 ms bind + 4 billion logical operations / 2 s motivating example is a hypothetical budget, **not a benchmark**. Per-fold nanosecond/µs gates and whole-batch elapsed time must be labeled distinctly. If a supposedly hot per-fold path measures in ms, treat it as a failed gate per #1657, not a success.

## 6. Steps, source ownership and issue links

| Phase | Deliverable | Reuse / owner | Admission |
|---|---|---|---|
| **S0: source census** | Inspect actual #1659 head and #1660 docs; identify literal `"{0}{1}"` selector without duplicating tuple decoders | #1659 `join.rs`, D-RMAP-1A, EXO-50 coordination | Docs/analysis |
| **S1: typed wrapper** | Minimal declarative projection shape `Template(selector[])` over borrowed lane+ordinal, checked matching frame/generation/auth; 16-lane position fixture | #1659 owns physical `Run`; [EXO-107](https://linear.app/exo-red/issue/EXO-107) lens gap | Lab only |
| **S2: order/inversion** | Run-offset fast path from preattested sorted key vs existing `pos[]` scatter; independent order witness including same-row identity | [EXO-114](https://linear.app/exo-red/issue/EXO-114), #1659 failed gate | Write-path ruling required |
| **S3: QR projection** | Materialize one typed 8 KiB Boolean source×target tile from borrowed ordinal views; store optional dirty mask; no row copies | D-RMAP-2, #1660 P5 | Lab only, obey #1640 |
| **S4: BlasGraph parity** | Match Boolean `AND/OR`, composition, masked `mxv/mxm`, COUNT/EXISTS without tuples | existing BlasGraph/bit-tile; [EXO-48](https://linear.app/exo-red/issue/EXO-48) | No new backend |
| **S5: rotations/views** | 3×3 `123/456/789 → 369/258/147`, 256×256 permutations and dirty maps | #1660 P11, typed axes | Refuse semantic edge changes |
| **S6: proof compression** | Palette-u8 rewrite IDs, egglog equivalence DAG, Gaussian/GF(2) typed separation, BEFORE/AFTER/dirty checker | #1660 P4/P6/P7/P9/P10 | Cold only |
| **S7: warm million-fold** | 256×4096 sharing case vs no-sharing; count source bytes and proof bytes as well as elapsed time | #1653, #1657, #1660 P12 | No production merge |
| **S8: writeback** | Separate commit receipt/transaction, inverses only when algebraically legal, generation-aware replay | Thought Git EXO-55/58/98 | Later gate |

Other overlaps: [EXO-79](https://linear.app/exo-red/issue/EXO-79) D-MPS physical join; [EXO-86](https://linear.app/exo-red/issue/EXO-86) contract/relalg; [EXO-102](https://linear.app/exo-red/issue/EXO-102) D-RU; [EXO-104](https://linear.app/exo-red/issue/EXO-104) tuple violation; [EXO-115](https://linear.app/exo-red/issue/EXO-115) Quack subset→contract; [EXO-50](https://linear.app/exo-red/issue/EXO-50) session bus. Do not claim or alter these owners without handoff.

## 7. Research probes / independent falsifiers

| Probe | Positive expectation | Mandatory negative / stop gate |
|---|---|---|
| **QR-P01: template** | `"{0}{1}"` references two borrowed lanes at the witnessed ordinal; 16 positions projected without `String`, `Vec<Row>`, or `to_vec` | Change one lane's row identity but retain equal numeric index: **refuse** |
| **QR-P02: sparse/dense domain** | `rank+popcount` and ordered-sparse path yield identical stable slots | `pos` out of bounds, duplicate rekey, stale source seal or unseen key: **refuse** |
| **QR-P03: ordered fast path** | Already sealed key order yields offsets without copied payload or re-sort; benchmark 1/8/64 reuse | Unsorted or only lexicographically LE-image sorted key: **refuse** |
| **QR-P04: prefetch** | Compare no-prefetch, 1-slot and N-slot lookahead using *identical* borrowed accesses; report cache misses and ns | No win or worse memory traffic => no promotion; hint is not a gather exception |
| **QR-P05: QR composition** | 256×256 Boolean `OR_k(AND)` matches scalar oracle + existing bit-tile / BlasGraph oracle | Two distinct paths to same target: XOR parity 0 != Boolean existence 1 |
| **QR-P06: set/bag/terminal** | `EXISTS` and `COUNT` are mask/offset folds; `Rows` is an on-demand borrowed lens preserving original ordinals | Duplicate memberships, cross-product and external-focus escapes must not be silently dropped or heap materialized |
| **QR-P07: view rotation** | 3×3 given fixture, 4×90° identity, 256×256 transpose/flip/CCW view and same-bijection dirty propagation | Method×entity axes swapped without metadata; directed graph semantics altered: **refuse** |
| **QR-P08: mathematical compression** | `a=2b; b=2c; z=2a => z=8c`; 256 templates × 4096 frames; proof DAG and optimizer cost measured | Saturating u8, float regrouping, conflicting cycles, independent expressions show no false equivalence/small-certificate guarantee |
| **QR-P09: replay/writeback** | Version-pinned BEFORE + exact dirty + AFTER reconstructs independent scalar state; actual changes traced to source ordinals | Only AFTER exists for lossy OR, ACL revoked or DB ack absent: **never claim reversible/committed** |
| **QR-P10: security/admission** | Same-table witnessed seeks, frozen-codebook lookups where admitted; cross-source co-address only after independently witnessed identity | Value from table A drives position in table B at any phase: **refuse under ADMISSION B** |

Independent reference should be *structurally different* from the candidate: scalar bit enumeration, independent GUID mapping, an external GF(2) solver or existing BlasGraph where truly equivalent. Reusing the same mask-risc code as both oracle and candidate does not certify correctness.

## 8. Decision requests before any implementation

1. **Owner assignment:** #1659's lab binder remains with its current session; #1660 remains docs/research. Who owns the tiny named-operand→position-template wrapper/lens? Coordinate through EXO-50 before code.
2. **Ordered write-path:** Is a sealed key-order witness and a dedicated key-sort at write time permissible, and who owns that index? One shared row order rarely satisfies every possible key.
3. **QR physical boundary:** Which specific typed axis scheme is approved for the first tile, source×target adjacency vs method×slot membership? No generic untyped 256×256 claim.
4. **Commit surface:** Does replay sidecar remain read-only until Thought Git writer receives separate authorization and transaction acknowledgment? Recommended yes.
5. **Admissibility:** No cross-table positional foreign gather or new interpreter is implicitly permitted by the phrase *PowerShell polyfill*.

**Success:** small declarative lens compiles to #1659 borrowed ordinal access, QR truth folding reuses existing mask-risc and BlasGraph semantics, no heap tuples, stale-order/scope guards work, cold proof replay is independently checkable, and all performance claims include binding + sidecar + output costs.  
**Failure:** source payload copied, separate graph engine, illegitimate gather, false row identity, XOR-for-OR, silently dropped multiplicity/provenance, stale commit witness or unexplained ms hot-fold path.
