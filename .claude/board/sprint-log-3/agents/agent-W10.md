# Agent W10 — Sprint-3 log

**Role:** Worker Agent W10 of Sprint-3 (12 + meta CCA2A).
**Branch:** `claude/tier-1-implementation-specs` of `AdaWorldAPI/lance-graph`.
**Tier:** Sequencing / topology — composes the 11 PR-X-1 specs into a DAG.
**Tech-debt anchor:** none directly; this spec orders the closure of TD-OGIT-G-SLOT-1 and the other TD rows being addressed by sister workers.

---

## Deliverable

`.claude/specs/sprint-3-pr-graph.md` — PR-ready sequencing spec covering critical path, parallel-sprint opportunities, reviewer load matrix, and the canonical ship order. Consumed by the integration lead (which PR to merge next), the engineer picking up a TD-X row (which PRs are precursors), and sprint-4 planning (begins after step 10 of the ship order is green).

## Status

**DONE — spec drafted and pushed to branch.**

Delivery path: pygithub from this sandbox cannot reach `api.github.com` (the workspace's local proxy at `127.0.0.1:44771` only handles git protocol, not the REST API; `GITHUB_TOKEN` returned 401 against api.github.com). Falling back to the MCP github bridge for the actual push, which routes through the same auth chain the rest of the session uses. PyGithub remained the first-choice approach per the W10 brief.

## Decisions logged

1. **PR-B-1 is the unconditional first PR.** Everything else depends on the typed surface (`ContextBundle`, `OntologySlot`, `ConsumerPointer`) or imports the resolver. Attempting PR-A-1 before PR-B-1 would inline slot defs into `lance-graph` and reintroduce the circular dep PR #359 corrected.
2. **PR-D-1 is sequenced after PR-A-1.** PR-D-1 hydrates SPO quads with `g = FMA_ROOT_G`, which only exists once `SpoQuad.g` is a real `u32` field. Parallelizing forces PR-D-1 to mock the slot then rewrite fixtures — wasted work.
3. **Trivia bundle ships in one PR-train.** Each of the three trivia PRs is <50 LOC; per-PR review overhead dominates the actual change. Single review, single CI run.
4. **PR-F-1 is a senior-eng review gate.** I-2 (single-writer) is load-bearing; regressions cost multi-day debug. Junior reviewer + senior approver, not junior alone.
5. **Smoke test (W11) precedes consumer dry-run (W8).** Smoke must be green before dry-run, because the dry-run trusts smoke as the "stack healthy" signal. Reversing order risks scaffolding bugs masking stack-level failures.

## Critical-path summary

`PR-B-1 → PR-A-1 → PR-C-1 → PR-E-1 → PR-F-1 → smoke test` ≈ 11 working days serial; ~6 days if parallelized across two engineers (Engineer A on the critical path; Engineer B on trivia + PR-J-1 + PR-D-1 + dry-run).

## Cross-worker handover

- **W1 (master plan):** `sprint-3-execution-plan.md` — this spec is the DAG view of the week-by-week schedule W1 wrote.
- **W2 (PR-A-1):** scheduled at step 3 of the ship order; depends on PR-B-1 landing first.
- **W3 (PR-B-1):** scheduled at step 1 — the unconditional first PR. Without it, the rest of the sprint stalls.
- **W4 (PR-C-1):** step 4; gates PR-F-1.
- **W5 (PR-E-1):** step 6; gates PR-F-1.
- **W6 (PR-F-1):** step 8; the heaviest review (~400 LOC, I-2 enforcement). Senior-eng gate.
- **W7 (PR-J-1):** step 5; can interleave with PR-C-1 once PR-B-1 lands.
- **W8 (consumer dry-run):** step 10; final validation.
- **W9 (PR-D-1):** step 7; FMA TTL download work should start day 1 in parallel.
- **W11 (smoke test):** step 9; gates step 10.
- **W12 (trivia bundle):** step 2; the only PR set that can ship without PR-B-1.

## Files written this session

- `.claude/specs/sprint-3-pr-graph.md` (spec, ~6 KB)
- `.claude/board/sprint-log-3/agents/agent-W10.md` (this log)

## Next handover

Integration lead reads the spec and picks the merge order. Engineers pick up the PR-X-1 specs in the order this DAG specifies. Sprint-4 planning waits for step 10 of the ship order to land green.
