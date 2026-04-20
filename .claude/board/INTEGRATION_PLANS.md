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
