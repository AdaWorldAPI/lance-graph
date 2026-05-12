# Sprint-3 PR sequencing + dependency graph

> **Worker:** W10 of Sprint-3 (12 + meta CCA2A).
> **Branch:** `claude/tier-1-implementation-specs` of `AdaWorldAPI/lance-graph`.
> **Role:** sequencing / topology spec — tells the engineer which PR to land first and which can ship in parallel.
> **Predecessors:** the 11 PR-X-1 specs from W2 - W9, W11, W12. This spec composes them into a DAG.
> **Status:** Spec ready. After this lands, `git log --oneline` order on the integration branch matches the order below.

## What this spec is for

Sprint-3 produced 11 PR-shaped implementation specs (PR-A-1 ... PR-J-1, three trivia PRs, two validation specs). Each spec stands alone, but the PRs are **not independent** — landing them in the wrong order produces compile errors, missing trait imports, or smoke-test fixtures that have no resolver to call. This document is the canonical topology: critical path, parallel opportunities, reviewer load, ship order.

It is consumed by:
- The integration lead deciding which PR to merge next.
- The engineer picking up a TD-X row, who needs to know which PRs are precursors.
- Sprint-4 planning, which begins after the smoke test (step 9 below) is green.

## Topological order (week-by-week)

### Foundation (Week 1)
PRs that have **no internal sprint-3 dependencies**; an engineer can parallelize across these without blocking on anything.

```
PR-B-1 (W3) — ContextBundle typed surface           [parallelizable]
PR-CAM-DIST (W12) — register cam_distance globally   [parallelizable, 1 line]
PR-DEEPNSM-NSM-COLLAPSE (W12) — delete nsm/ shim    [parallelizable, ~30 LOC]
PR-ADJ-THINK-EXPOSE (W12) — tau_write() public API  [parallelizable, ~30 LOC]
```

### Tier-1 wiring (Week 1-2)
PRs that need **PR-B-1 to land first** because they import the resolver / typed slots.

```
PR-B-1 (W3) ──┬──► PR-A-1 (W2) — SPO-G u32 slot
              ├──► PR-J-1 (W7) — INT4-32D atoms (uses thinking_styles slot)
              └──► PR-D-1 (W9) — FMA OWL hydrator (uses OntologySlot, requires PR-A-1 too)

PR-A-1 (W2) ──► PR-C-1 (W4) — GenericBridge (also requires PR-B-1)
PR-B-1 + PR-A-1 ──► PR-D-1
```

### Tier-2 supervised mesh (Week 2-3)
PRs that need **PR-B-1 + PR-C-1 + manifests** because the supervisor types the slot it owns.

```
PR-B-1 ──► PR-E-1 (W5) — manifest.yaml + build script
PR-C-1 + PR-E-1 ──► PR-F-1 (W6) — ractor supervisor
```

### Validation (Week 3-4)
After Tier-2 lands, the end-to-end smoke test exercises the whole stack; the consumer-template dry-run validates the wiring is reproducible for a new crate.

```
PR-A-1 + PR-B-1 + PR-C-1 + PR-E-1 + PR-F-1 ──► Smoke test (W11 spec)
PR-A-1 + PR-B-1 + PR-C-1 + PR-E-1 + PR-F-1 ──► Consumer template dry-run (W8 spec; hubspo-rs)
```

## Critical path

```
PR-B-1 → PR-A-1 → PR-C-1 → PR-E-1 → PR-F-1 → smoke test
```

≈ 1 + 2 + 2 + 2 + 3 + 1 days = **~11 working days minimum** (assuming serial execution; **~6 days if parallelized** across two engineers — see "Parallel-sprint opportunities" below).

## Bottlenecks

1. **PR-B-1 is the foundation.** Every Tier-1 / Tier-2 PR depends on it. Land it first or parallel sprints stall — the trivia bundle is the only thing that can ship without it.
2. **PR-F-1 is the largest** (~400 LOC, ractor supervisor + I-2 enforcement). Flag for senior eng review; do not let a junior reviewer rubber-stamp this one.
3. **PR-D-1 has external dependency** — FMA TTL download (~30 MB), license review, and parser crate selection (`oxigraph` vs `sophia`). Start the license / download work in parallel with PR-B-1 so the parser crate is chosen by the time PR-D-1 enters review.

## Parallel-sprint opportunities

While the critical path executes, the **trivia bundle** (W12: PR-CAM-DIST + PR-ADJ-THINK-EXPOSE + PR-DEEPNSM-NSM-COLLAPSE) can ship by a different engineer in **<1 day total**. Net entropy delta from those alone: **−3 ledger rows** in `ARCHITECTURE_ENTROPY_LEDGER.md`.

Concretely, if two engineers are available:

- **Engineer A** (critical path): PR-B-1 → PR-A-1 → PR-C-1 → PR-E-1 → PR-F-1 → smoke test.
- **Engineer B** (parallel): trivia bundle (day 1), PR-J-1 (day 2-3, after PR-B-1), PR-D-1 (day 4-7, after PR-A-1; download FMA TTL on day 1 in background), consumer-template dry-run with hubspo-rs (day 8, after PR-F-1).

Wall-clock: ~6 days for the full sprint instead of ~11.

## Quick PR review matrix

| PR | Files | LOC | Tests | Reviewer load |
|---|---|---|---|---|
| PR-B-1 | 3 | ~200 | 3 | Light |
| PR-A-1 | 5 | ~300 + migration | 4 | Medium |
| PR-C-1 | 5 | ~200 + 2 wrapper updates | 4 | Medium (touches medcare/smb wrappers) |
| PR-E-1 | 8 | ~330 (incl. 6 manifests) | 4 | Medium |
| PR-F-1 | 5 | ~400 | 5 | **Heavy** (ractor + I-2 enforcement) |
| PR-J-1 | 3 | ~120 | 4 | Light |
| PR-D-1 | 4 + TTL | ~600 | 4 | **Heavy** (parser + 30 MB TTL) |
| PR-CAM-DIST | 1 | 1 line | 1 | Trivial |
| PR-ADJ-THINK | 1 | ~30 | 1 | Light |
| PR-DEEPNSM-NSM | 1 (delete dir) | ~30 + 5 deletes | (existing tests) | Light |

## Recommended ship order (sequential view)

For a single engineer or a single integration branch, this is the order:

1. **PR-B-1** (foundation — every other PR depends on this typed surface)
2. **PR-CAM-DIST + PR-ADJ-THINK + PR-DEEPNSM-NSM** (trivia, can interleave on the same day)
3. **PR-A-1** (SPO-G u32 slot — requires PR-B-1)
4. **PR-C-1** (GenericBridge — requires PR-A-1 + PR-B-1)
5. **PR-J-1** (INT4-32D atoms — requires PR-B-1; can ship in parallel with PR-C-1)
6. **PR-E-1** (manifest + build — requires PR-B-1)
7. **PR-D-1** (FMA OWL hydrator — requires PR-A-1 + PR-B-1; download TTL early)
8. **PR-F-1** (ractor supervisor — requires PR-C-1 + PR-E-1; **largest review load**)
9. **Smoke test (W11)** — exercises every PR above end-to-end
10. **Consumer template dry-run with hubspo-rs (W8)** — validates the whole stack is reproducible for a new crate

After step 10: the architecture is **END-TO-END VALIDATED**. Sprint-4 begins with the anatomy demo PRs (PR-ANATOMY-2 through PR-ANATOMY-7, planned in `.claude/plans/anatomy-realtime-v1.md`).

## Decisions logged

1. **PR-B-1 is the unconditional first PR.** Everything else either depends on the typed surface (`ContextBundle`, `OntologySlot`, `ConsumerPointer`) or imports the resolver. Attempting PR-A-1 before PR-B-1 would inline the slot definitions into `lance-graph` itself and create a circular dep when PR-B-1 lands — exactly the anti-pattern PR #359 corrected.
2. **PR-D-1 is sequenced after PR-A-1, not in parallel.** PR-D-1 hydrates SPO quads with `g = FMA_ROOT_G`, which only exists once `SpoQuad.g` is a real `u32` field. Running them in parallel forces PR-D-1 to mock the slot, then rewrite the test fixtures after PR-A-1 lands — wasted work.
3. **Trivia bundle ships standalone in one PR-train, not three separate PRs.** Each is <50 LOC; review overhead per-PR dominates the actual change. A single review covers all three with one CI run.
4. **PR-F-1 review is a senior-eng gate.** I-2 (single-writer) enforcement is the load-bearing invariant that protects every consumer; a regression here is a multi-day debugging session. Junior reviewer + senior approver, not junior alone.
5. **Smoke test (W11) and consumer dry-run (W8) are sequential, not parallel.** The smoke test must be green before the dry-run, because the dry-run trusts the smoke test exit code as the "stack is healthy" signal. Reversing order means a hubspo-rs scaffolding bug could mask a stack-level smoke failure.

## Cross-references

- `.claude/specs/sprint-3-execution-plan.md` (W1 master plan — the week-by-week parent of this PR graph)
- `.claude/specs/pr-a-1-spo-g-u32-slot.md` (W2 — first concrete Tier-1 spec; this graph schedules it)
- All 11 PR-X-1 specs from sister workers (W3 - W9, W11, W12) — when those land, this graph is the index that orders them
- `.claude/board/TECH_DEBT.md` (TD-X rows being closed by these PRs — cross-check after each PR ships)
- `.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md` (entropy delta tracking — trivia bundle alone removes 3 rows)
- `.claude/plans/anatomy-realtime-v1.md` (sprint-4+ continuation — begins after step 10 above is green)

## Open questions for the engineer

1. **Two-engineer split or single-engineer serial?** The "Parallel-sprint opportunities" section assumes two engineers; integration lead picks based on availability. Recommendation: two engineers if available — wall-clock drops from ~11 to ~6 days.
2. **PR-D-1 parser crate — `oxigraph` or `sophia`?** Decision needs to be made before PR-D-1 review (day 1, in parallel with PR-B-1 work). Recommendation: `oxigraph` (SPARQL store needs it anyway; one less dep).
3. **Do we squash the trivia bundle into one commit?** Three commits with one PR is fine; one squashed commit is also fine. Recommendation: squash, because `git log` reads cleaner and the three changes are all "trivia closure" anyway.
