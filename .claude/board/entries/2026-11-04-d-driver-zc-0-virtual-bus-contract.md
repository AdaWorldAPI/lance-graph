# D-DRIVER-ZC-0 — virtual zero-copy driver, address-preserving XY/XZ/XX/YY/ZZ bus

**Date:** 2026-11-04. **Status:** ARCHITECTURE / FALSIFIER PRE-REGISTRATION ONLY; no new source pointer/driver/API/codec/ISA/remote server and no numeric performance result.

**Operator:** Apply the `ndarray::simd` polyfill pattern to DR's **data access**, not by auditing approximately 17,000 calls one by one. One admissible `BIND → PROJECT → STREAM → EXECUTE → RELEASE` membrane over witnessed source-owned DR coordinates should make implicit canonical value copying impossible for **migrated sealed consumer paths**. It should expose typed virtual buses `XX, XY, XZ, YY, ZZ` over *one* canonical SoA rather than storing additional bus arrays or constructing `PSCustomObject`-like rows. Foreign global→local navigation happens only under owner admission, and can amortize once per witnessed frame. Local borrowing and data-owner remote execution are separate realizations, not an exported Rust pointer.

**Plans:** [D-DRIVER-ZC master](../../plans/d-driver-zc-virtual-bus-contract-v1.md) · [staged PR charter](../../plans/d-driver-zc-staged-pr-charter-v1.md) · [47 falsifier rows](../../plans/d-driver-zc-falsifier-matrix-v1.md).

**Linear:** [EXO-154](https://linear.app/exo-red/issue/EXO-154) owner issue; [EXO-153](https://linear.app/exo-red/issue/EXO-153) existing D-DRISA stages; [EXO-50](https://linear.app/exo-red/issue/EXO-50) append-only agent handoff.

## Exact reuse

- [Zero-copy lens law](../../knowledge/zero-copy-lens-law.md) remains operator ruling. No hidden source value array, tuple enumeration or row reconstruction in the admitted fold. An array is already a ClassView projection; source stride may be borrowed where layout permits.
- [#1955](https://github.com/AdaWorldAPI/lance-graph/pull/1955) **MERGED LAB**: sealed DR register with local owned test-SoA, scope/frame/layout/source witnesses, borrow semantics, route-once. It does not prove a virtual bus or remote/disk execution.
- [#1956](https://github.com/AdaWorldAPI/lance-graph/pull/1956) **MERGED LAB**: Abakus carry-word correctness for k-of-n/Kleene/Bag, but 1.5–4.8× slower than native arms in eight measured shapes; carry is not a forced driver ALU.
- [#1950](https://github.com/AdaWorldAPI/lance-graph/pull/1950) **MERGED LAB**: real Lance mask ingress costs; prepared-binding timings include copied value lanes, and DR→Lance `BLOCKED_D4`.
- [#1953](https://github.com/AdaWorldAPI/lance-graph/pull/1953) **MERGED DOCS**: D-DRISA async register research architecture and stages; new plan does not build a second async runtime.

## Staged PRs (NONE of ZC-1..ZC-7 opened here)

| D-id | Deliverable | Status / gate |
| --- | --- | --- |
| D-DRIVER-ZC-0 | master architecture, 47 falsifiers, staged charter, board/EXO map | THIS DOCS PR, no runtime |
| D-DRIVER-ZC-1 | local borrowed-lens membrane over merged D-DRISA-1 | FUTURE LAB, source/DT64 owners |
| D-DRIVER-ZC-2 | five virtual bus roles with address-preserving lens composition | FUTURE LAB, Quack/DT64 semantic owners |
| D-DRIVER-ZC-3 | co-addressed/reused tiles; foreign global→local amortization | LOCAL LAB possible; FOREIGN BLOCKED EXO-46/89/151/140 |
| D-DRIVER-ZC-4 | async source-owner descriptor **compatibility** (not new driver) | GATED D-DRISA-2/4, security owner |
| D-DRIVER-ZC-5 | physical Lance source / disk-backed no-owned-intermediate proof | **BLOCKED D4/S-INGRESS**, EXO-114 |
| D-DRIVER-ZC-6 | typed read access firewall, compile-fail and incremental caller coverage | FUTURE scoped policy lab; EXO-104/105/109 |
| D-DRIVER-ZC-7 | "Thinking Hard Drive" representation/semantic/economic verdict | FUTURE REVIEW, not theorem certified |

## Critical guarantees and refusals

- **Resident ZC-A:** zero *newly owned source value payload bytes* in migrated admitted hot execution; legitimate scratch/selection/result bytes counted separately. A one-time source I/O/decode still costs data movement.
- **Virtual ZC-B:** no physical bus arrays for XY/XZ/XX/YY/ZZ; an already-bound fixed-size row lens costs O(1) loads in row count, but arbitrary foreign joins, K-row folds and output are not universally O(1).
- **Remote ZC-C:** source owner performs local borrow/compute; remote client sends typed instruction and receives a measured result, not a raw pointer. **No network/RDMA zero-copy claim.**
- **No universal historical certification:** the ~17k calls are operator context, not an enumerated call graph. CI can enforce a sealed migrated scope; bypass coverage remains a separate measurement.
- **Theorem remains PROPOSED:** parity + owned-payload representation bound + scope/layout/lifetime validity + exact physical I/O accounting. A zero-allocation wrong-answer path is a failure.

**Owner gates:** [EXO-46](https://linear.app/exo-red/issue/EXO-46), [EXO-89](https://linear.app/exo-red/issue/EXO-89), [EXO-151](https://linear.app/exo-red/issue/EXO-151), [EXO-140](https://linear.app/exo-red/issue/EXO-140), [EXO-121](https://linear.app/exo-red/issue/EXO-121), [EXO-114](https://linear.app/exo-red/issue/EXO-114), [EXO-104](https://linear.app/exo-red/issue/EXO-104), [EXO-105](https://linear.app/exo-red/issue/EXO-105), [EXO-109](https://linear.app/exo-red/issue/EXO-109). No owner decision is made by these plans.

**Deferred:** no audit or consolidation now of cognitive-shader-driver, former ladybug-rs, P64, StreamDTO, BusDTO, PerturbationDTO. Eventual one-canonical-surface choice belongs to a separate operator decision.

**STOP:** copying canonical payload values in the admitted hot path; inventing another hash map or row authority; foreign/local ordinal confusion; relabeling physical disk/network copies as free; new public ISA or "GPU shader"; completing full caller coverage without a defensible reachability census; automatic merge.
