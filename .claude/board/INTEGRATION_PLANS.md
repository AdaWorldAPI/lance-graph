# Integration Plans — Versioned Index

> **APPEND-ONLY.** Every integration plan ever authored for this
> workspace, versioned, with status. New plans append to the top
> as new entries. Superseded plans stay — they are the design arc.
>
> Governance: same rule as `PR_ARC_INVENTORY.md`. The **Status**
> field is the only mutable field per entry. Supersedure is marked
> by adding a new top entry that references the prior version; the
> prior entry is NOT deleted.

---

## APPEND-ONLY RULE

1. **New plans PREPEND** a new section at the top.
2. **Old plan entries are IMMUTABLE** except the **Status** line.
3. **Supersedure:** if plan vN is replaced by vN+1, prepend a new
   entry for vN+1 that cites vN; update vN's **Status** to
   "Superseded by vN+1".
4. **Corrections** to plan scope during its lifetime: append a
   `**Correction (YYYY-MM-DD):**` line to the entry; do not edit
   the original scope line.
5. **Retire but never delete.** When a plan is complete or
   abandoned, update Status and move on. The entry stays.

**Per-entry format:**

- **Plan name + version**
- **Author + date**
- **Scope** — one-sentence goal (immutable)
- **Path** — workspace file location
- **Deliverables** — D-id list (immutable)
- **Status** — **mutable**: Active / Shipped / Superseded / Deferred / Abandoned
- **Confidence** — **mutable**: Working / Partial / Broken — see PR #N

---

## v1 — Categorical-Algebraic Inference (authored 2026-04-21)

**Author:** main-thread session 2026-04-21
**Scope:** Meta-architecture document proving that parsing (Kan extension), disambiguation (free-energy minimization), learning (NARS revision), memory (AriGraph commit), and awareness (method-call history) are one algebraic operation — element-wise XOR on role-indexed slices of a 10K binary VSA vector — viewed through five lenses. Grounded in Shaw 2501.05368 (category theory) + 13 supporting papers. Does not replace elegant-herding-rocket — extends it with the categorical foundation.
**Path:** `.claude/plans/categorical-algebraic-inference-v1.md` (496 lines)
**Deliverables:** This plan produces no NEW D-ids. It grounds the existing D2/D3/D5/D7/D8/D10 deliverables from elegant-herding-rocket in the categorical-algebraic framework and establishes the five-lens litmus + object-does-the-work test as architectural invariants.

**Status (2026-04-21):** Active. Companion to elegant-herding-rocket-v1, not a replacement.

**Confidence (2026-04-21):** CONJECTURE on the Kan-extension-IS-free-energy equivalence. FINDING on all other claims (grounded in shipped code + paper proofs).

---

## v1 — Codec Sweep via Lab Infra, JIT-first (authored 2026-04-20)

**Author:** main-thread session 2026-04-20
**Scope:** Operationalise PR #220's "What's Needed to Fix" list (wider codebook / residual PQ / Hadamard pre-rotation / OPQ) as a parameter sweep through the lab endpoint, with every codec candidate difference expressed as a JIT-compiled kernel rather than a cargo rebuild — one upfront API hardening rebuild, unlimited candidates afterwards.
**Path:** `.claude/plans/codec-sweep-via-lab-infra-v1.md` (396 lines)
**Deliverables:** D0.1 `CodecParams` in `WireCalibrate`, D0.2 `WireTokenAgreement` endpoint (I11 cert gate), D0.3 `WireSweep` streaming endpoint + Lance append, D0.4 surface freeze. D1.1 `CodecKernelCache` via `JitCompiler`, D1.2 rotation primitives (Identity / Hadamard / OPQ) as JIT kernels, D1.3 residual PQ via JIT composition. D2.1 reference-model loader, D2.2 decode-and-compare loop, D2.3 handler wiring. D3.1 server-side sweep handler, D3.2 curl-driven client. D4.1 DataFusion over Lance log, D4.2 Pareto frontier notebook. D5 graduation bridge (fires only on candidate passing all gates).

**Status (2026-04-20):** Active. Plan authored; no deliverables shipped yet. Depends on merge of PR #224 (three-part lab-surface framing + I11 measurability invariant) for the architectural grounding.

**Confidence (2026-04-20):** Pre-execution. Risk hot-spots: (a) JIT compile cost for residual PQ composition — needs measurement; (b) token-agreement harness load time on ref model — may dominate latency for small sweeps; (c) Lance append concurrency under streaming writes. Plan assumes these are tractable; D0 surface freeze is deliberate to prevent iterating on the DTO shape mid-sweep.

---

## v1 — Elegant Herding Rocket (authored 2026-04-19)

**Author:** main-thread session 2026-04-19
**Scope:** DeepNSM as full parser via Grammar Triangle wiring + Markov ±5 SPO+TEKAMOLO bundling + NARS-tested grammar thinking styles + coreference resolution + story-context bridge + ONNX arc emergence.
**Path:** `.claude/plans/elegant-herding-rocket-v1.md` (2,085 lines)
**Deliverables:** D0 landscape doc, D2 FailureTicket emission, D3 Triangle bridge, D4 ContextChain reasoning, D5 Markov ±5 bundler, D6 role keys, D7 grammar thinking styles, D8 story context + contradictions, D9 ONNX arc export, D10 Animal Farm validation harness, D11 bundle-perturb emergence.

**Status (2026-04-19):** Active. Phase 1 (D0 + D4 + D6) shipped in PR #210.

**Confidence (2026-04-19):** Phase 1 working (125 tests passing).
Phases 2–4 queued.

**Phases:**

- **Phase 1 — SHIPPED** (PR #210, merged): D0 landscape doc + D4
  ContextChain reasoning ops + D6 role keys. 125 tests passing.
- **Phase 2 — QUEUED:** D2 FailureTicket emission + D3 Triangle
  bridge + D5 Markov bundler + D7 grammar thinking styles.
  Estimate ~930 LOC, one PR.
- **Phase 3 — QUEUED:** D8 story-context/contradictions + D10
  Animal Farm validation harness.
- **Phase 4 — FUTURE:** D9 ONNX arc export + D11 bundle-perturb
  emergence interface.

---

## How to use this file

1. **Starting a new session:** check top entry. If Status is Active,
   that's the current plan. Read it at
   `.claude/plans/<plan-file>.md`.
2. **Proposing a new plan:** prepend a new v entry; move prior
   plan's Status to Superseded.
3. **Tracking deliverable progress:** use
   `.claude/board/STATUS_BOARD.md` for the cross-deliverable
   view (which D-ids are in which phase / PR).
4. **User requests / open threads** that aren't yet a plan: capture
   in `.claude/knowledge/OPEN_PROMPTS.md`.

## Cross-references

- **`STATUS_BOARD.md`** — deliverable-level status (D0 / D2 / D3 / …
  across all plans).
- **`OPEN_PROMPTS.md`** — outstanding user questions / threads that
  aren't yet scoped into a plan.
- **`PR_ARC_INVENTORY.md`** — shipped-PR decision history.
- **`LATEST_STATE.md`** — current-state snapshot.

## 2026-04-20 — cam-pq-production-wiring-v1
**Status:** DRAFT
**Plan:** `.claude/plans/cam-pq-production-wiring-v1.md`
**Scope:** Wire CAM-PQ as default codec for argmax-regime tensors.
**Deliverables:** D1-D7 (classifier, calibration, storage, decode, validation, E2E, fallback).
**Driver:** ICC 0.9999 at 6 B/row on Qwen3-8B (PR #218 bench).
**Effort:** ~8 person-days.
**Confidence:** HIGH.
