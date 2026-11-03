# D-DRAM-P3: D-GLOBAL-WHERE P2 open ends integrated with the D-DRAM-0 falsifier map

> **Status: PLAN, owned by this session (operator, 2026-11-03: "integrate 1949
> ideas … and take ownership").** It merges the open ends of D-GLOBAL-WHERE P2
> (#1950) into the D-DRAM-0 plan (#1949,
> `.claude/plans/d-dram-abacus-address-memoization-v1.md`). Ownership means:
> - this session executes the lab and read-only items;
> - it prepares production changes as draft PRs, for owner review;
> - it drives each gated item to the named decision with measured inputs.
>
> Ownership does not lift any gate: EXO-46, EXO-89, EXO-151, D4 and S-INGRESS
> stay with their owners. Nothing here is authorized to merge automatically.

## 1. What P2 already measured, against #1949's gates

P2 measured some #1949 gates before #1949 was written. Each gate is credited
at the strength it was measured, no higher.

| #1949 gate | What P2 measured | Status |
|---|---|---|
| L9 (full / ALLOW / BLOCK / attested range; 0.1, 1, 10, 50%; scattered and clustered) | Every cell, 1M and 10M, plus k = 1/4/16. Mask ingress and bytes accounted separately. `BLOCKED_D4` retained. | **MEASURED-LAB**, with ids given (D4) |
| L10 (1M and 10M; several DR256 relations; identity and Bag/Boolean oracles; no dense matrix, no new paging) | `compose`: DR composition beyond 64K at 1M and 10M against an independent oracle; no dense adjacency. | **MEASURED-LAB**; Bag multiplicity not separately varied |
| L1 / L2 (locator reuse, miss → hit) | Only the *route binding* is reused, not a locator receipt (C1 correction): break-even at Q ≥ 2; 158,198 → 1,012 µs per query at Q = 256. | **PARTIAL**: binding reuse measured; receipts not built (D-DRAM-2 GATED) |
| L6 (stale / foreign refusal) | The stale-binding guard refuses after `replace_tile`; with the guard disabled, the stale binding is accepted. The Lance lane is refused for a stale version, another dataset, and a forged witness. | **PARTIAL**: no scope or palette cases |
| A3 (FieldMask 64 vs WideFieldMask 256; 32-lane mapping; noncontiguous fields) | `project`: 64 logical properties over 32 lanes, U/P/S layouts. The Lance arm resolves ClassView fields through `predicate_iri`; the ordinal and label traps fire. | **PARTIAL**: k4/k16 sparse and mixed AND cases have oracle count 0 (vacuous) |
| A4 (packed 4×8, only after codec attestation) | C2: there is **no** ordered u8-in-u32 compare; composing existing words costs one pass per 2-bit prefix (up to 5× slower than a hand loop). | **NO-EXISTING-KERNEL**, recorded; codec not attested |
| C2 (beyond 64K, independent oracle) | `compose`, as for L10. | **MEASURED-LAB** |
| C3 (DuckDB engineering lessons) | `ENGINEERING-PARITY.md`: literature-first matrix; WHEN vs WHETHER. | **DONE** (reference) |
| C4 (`BLOCKED_D4`, `S_INGRESS_COPY`) | Arm C reports `BLOCKED_D4`; ingress is measured as a copy (0.4–5.4 µs for one range, 123,162 µs for 3.26M ranges). | **DONE**; adopt the exact label `S_INGRESS_COPY` in the arm's output |

Not touched by P2: L3–L5, L7, L8, A1, A2, A5–A7, C1.

## 2. Owned work items

Ordered by signal per unit of effort. **Owner: this session** for every row.
The gate column names who must decide before anything outside the lab
changes.

| ID | Item | Gates it serves | Measured motive | Effort | Gate / where it lands |
|---|---|---|---|---|---|
| P3-1 | `ReadPolicy` weighs range count, not only coverage | L9 | Clustered 10% range sent to a full scan: 3.8× (1M), 4.6× (10M) slower | low | lab crate `crepe-plan-probe`; none |
| P3-2 | Make the `project` k4/k16 sparse and mixed AND cases non-vacuous | A3 | Oracle count 0 cannot catch a wrong answer | low | research branch; none |
| P3-3 | **D-DRAM-1 census** (read-only): the exact `ndarray::simd` 5×24 symbols and signatures, the existing DR focus/locator lease, and the one DR map owner | prerequisite for A1–A7 and L1–L8 | #1949 marks the 5×24 API UNVERIFIED; building on it unverified is a STOP condition | low | read-only; report to EXO-46 / ndarray owner |
| P3-4 | EXISTS stops at the first match per tile, plus a can-stay-silent test on an empty mask | C3 | The only DuckDB loss: 0.16× / 0.43× at 10M | low | **production mask-risc**: draft PR, mask-risc owner reviews |
| P3-5 | **D-DRAM-3**: k-of-n threshold on the shipped `half_add_mask` / `full_add_mask` vs native AND/OR/COUNT, gates A1, A2, A5, A6, A7 | A-family | No k-of-n measurement exists; P2's WHERE shapes are AND/OR/NOT only | medium | lab; after P3-3; no new opcode or public type |
| P3-6 | Ordered u8-in-u32 compare word | A4, A7 | P2 C2 measured the gap | medium | ndarray owner; proposal only, after P3-3 and a codec attestation |
| P3-7 | Route once per frame in the composition route stage | L1, L2, C1 | Routing dominates `compose` (35,914 of 50,178 µs) | medium | lab measurement now; a production binding is GATED on EXO-46 |
| P3-8 | **D-DRAM-4** end-to-end DR → Lance: ids from DR instead of given | L9, C1, C4 | The integrated path is still unmeasured | — | GATED on D4 + S-INGRESS (#1788); this session prepares the decision memo with the measured costs |
| P3-9 | **D-DRAM-2** locator receipts (L1–L8 proper) | L-family | — | — | GATED on EXO-46, EXO-89, EXO-151 |
| P3-10 | Evaluation direction from presence density | C3 | P0 W4 wrong-direction arms; #1945 | medium | Quack owner |
| P3-11 | Bind-time folding of tautologies and contradictions | C3 | `adv_contradictory_17` at PARITY | low–medium | Quack owner |

## 3. Execution order

1. **P3-1, P3-2, P3-3** first: none needs anyone else's decision. P3-3 runs
   as read-only harvest (Haiku cards, orchestrator verifies every hit).
2. **P3-4** as a draft PR on its own branch, once P3-1 is in.
3. **P3-5** after P3-3 has pinned the exact symbols. If the census finds no
   shipped 5×24 family, P3-5 uses only `half_add_mask` / `full_add_mask` and
   says so.
4. **P3-7** lab measurement; **P3-6** proposal.
5. **P3-8** decision memo to the D4 / S-INGRESS owner.

## 4. Stop conditions (from #1949 §7, kept verbatim in force)

Stop on any of:
- a new global → row-id store;
- a fresh SIMD backend in Quack;
- changed field-presence meaning;
- payload duplication by default;
- a locator outliving its lease;
- an unverified 5×24 API claimed as shipped;
- an unauthorized Gather;
- a new ClassId or F###;
- a production patch merged without its owner.

## 5. Checklist

- [x] P3-1 `ReadPolicy` range count, with a can-fire and a can-stay-silent test (disable runs: 2 and 6 tests red)
- [ ] P3-2 non-vacuous `project` cases
- [x] P3-3 D-DRAM-1 census written (`entries/2026-11-03-d-dram-1-census.md`)
- [ ] P3-4 EXISTS early exit, draft PR
- [ ] P3-5 D-DRAM-3 k-of-n bench (A1, A2, A5–A7)
- [ ] P3-6 ordered compare proposal
- [ ] P3-7 route-once lab measurement
- [ ] P3-8 D4 / S-INGRESS decision memo
- [ ] `S_INGRESS_COPY` label in the Lance arm output (C4)
