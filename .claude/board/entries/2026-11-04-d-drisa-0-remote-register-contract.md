# D-DRISA-0: DR virtual register + Abakus ALU + remote async cognitive shader (pre-registered)

**Date:** 2026-11-04. **Status:** PROPOSED / DOCS ONLY; no executable remote driver, ABI, F###, ISA extension, pointer-transport, source authority or speedup claim. **Operator idea:** eliminate unnecessary hash-table-like lookups and payload transport by executing a typed program *where canonical SoA is already owned*, using a witnessed DR frame as virtual register-address space and Abakus as an optional arithmetic ALU.

**Plan (authoritative):** [D-DRISA-0](../../plans/d-drisa-remote-register-shader-v1.md) · [staged PR charter](../../plans/d-drisa-staged-pr-charter-v1.md) · [57 falsifiers](../../plans/d-drisa-falsifier-matrix-v1.md).
**Linear:** [EXO-153 program](https://linear.app/exo-red/issue/EXO-153) and [EXO-50 append-only bus](https://linear.app/exo-red/issue/EXO-50).

## Reuse, not reimplementation

- **DR:** existing `lance-graph-dt64::dr::{MapTable,Frame,LocalMap}` owns coordinates, not values. No second global→`_rowid` store.
- **Lens:** existing `ClassView`, `WideFieldMask` and immutable DT64 `MethodBinding` describe only demanded fields and authorized behavior.
- **ISA:** Quack/RelAlg/R2IL → mask-risc borrowed `Plane/Scratch/LaneRef` → ndarray SIMD. No new instruction loop, graphics backend or automatic F###.
- **Abakus:** shipped `half_add_mask` and `full_add_mask` are valid words; #1951's `U24x5`/`Abacus` are examples-only. Arithmetic thresholds are a **candidate**, native AND remains the control.
- **Source:** use canonical borrowed SoA and optional prefetch at its owner. Remote descriptor and compact result may be transported, but **remote zero-copy is not proven or promised**. Arrow C zero-copy is same-process, not an RPC guarantee.
- **Continuations:** R2IL scratch slots have epochs. Remote `MaskReceipt` cannot silently outlive its program or become a new global mask store; owner/lifetime decision required.
- **Epistemic:** grammar and XLOG/NAL polyfills are candidate consumers; THOUGHT remains speculative and WORLD writes require separate hydration/Rubicon admission.

## Staged PR sequence, not existing PRs

| Stage | Deliverable | Status / gate |
| --- | --- | --- |
| D-DRISA-0 | docs, board, EXO-153 issue, EXO-50 handoff | THIS PR; research-only |
| D-DRISA-1 | local borrowed register oracle and source/lifetime refusal | FUTURE LAB |
| D-DRISA-2 | bounded async loopback, cancellation and lease revocation | FUTURE LAB; security review |
| D-DRISA-3 | k-of-n Abakus with shipped words and negative controls | FUTURE LAB; ndarray/Quack owners |
| D-DRISA-4 | two-process data-local command vs fetch-payload baselines | FUTURE LAB; transport/security gate |
| D-DRISA-5 | source-location receipt and prefetch without copied values | **BLOCKED EXO-46/89/151** |
| D-DRISA-6 | real DR→Lance `_rowid` binding, copied mask ingress measured | **BLOCKED D4/S-INGRESS**, EXO-114 |
| D-DRISA-7 | read-only cognitive consumers, bounded result masks | **GATED EXO-140/73/78**, semantic owners |
| D-DRISA-8 | independent full-cost audit, production ruling | NOT AUTHORIZED |

## Falsifiers and cost

- **57 pre-registered rows:** R1–R16 addressing/leasing; A1–A8 ALU; N1–N14 async and transport; C1–C10 cognition/semantics; P1–P9 full-path economics. All PROPOSED/UNEXECUTED here; earlier #1948/#1950/#1951 evidence is identified as partial, never credited as remote results.
- **Whole-query benchmark:** local borrow B0, remote payload fetch B1, data-local instruction B2, binding reuse B3, memoized locator B4, prefetch B5, Abakus B6. Count cold costs, all copies, RPC bytes, source pages, authentication, p95/p99 and refusers.
- **#1950 caveat:** its route-once `Binding::prepare` retains *copied* `lane_a/lane_b`; Q≥2 improvement is not locator proof.
- **#1951 caveat:** `range_count * 4096` is a fragmentation heuristic/policy pin, not a count of physically touched pages. No planner law inferred.

## Required decisions and STOP conditions

[EXO-46 admission](https://linear.app/exo-red/issue/EXO-46), [EXO-89 relationship owner](https://linear.app/exo-red/issue/EXO-89), [EXO-151 focus lease](https://linear.app/exo-red/issue/EXO-151), [EXO-140 gather/hop classification](https://linear.app/exo-red/issue/EXO-140), [EXO-121 ClassView](https://linear.app/exo-red/issue/EXO-121), [EXO-114 ordering](https://linear.app/exo-red/issue/EXO-114), [EXO-73 recursion](https://linear.app/exo-red/issue/EXO-73), [EXO-78 thought layers](https://linear.app/exo-red/issue/EXO-78).

**STOP:** untrusted remote pointer, unauthorized field, stale frame, second row-ID map, default value gather, silently serialized payload, cross-tenant leak, runtime write, new F###, unsupported arithmetic semantics, fictitious public kernel, automatic merge.

**Reference PRs:** [#1948](https://github.com/AdaWorldAPI/lance-graph/pull/1948), [#1949](https://github.com/AdaWorldAPI/lance-graph/pull/1949), [#1950](https://github.com/AdaWorldAPI/lance-graph/pull/1950), [#1951](https://github.com/AdaWorldAPI/lance-graph/pull/1951). [STATUS_BOARD](../STATUS_BOARD.md) and [INTEGRATION_PLANS](../INTEGRATION_PLANS.md) both list this D-id.
