# Ripple Architecture Blackboard Changelog

## 2026-04-02 / 01

Created:
- `docs/integrated-architecture-map.md`
- `.claude/blackboard-ripple-architecture-20260402-01.MD`

### What was established

- The repo is best read as an emerging cognition stack rather than a plain token pipeline.
- The central architectural failure mode is premature flattening of unlike functions.
- The strongest current hierarchy is:
  - HEEL = basin
  - HIP = family
  - BRANCH = contradiction or polarity split
  - TWIG = local neighborhood
  - LEAF = exact member
- The clean DTO stack is:
  - `StreamDto`
  - `ResonanceDto`
  - `BusDto`
  - `ThoughtStruct` above them as accountable thought object
- `p64 + Blumenstrauß` should be treated as the explicit reasoning bus.
- Thinking style should be treated as collapse policy, not metadata.

### Most important distinctions locked in

1. Field is not sweep.
2. Sweep is not bus.
3. bgz17 is HEEL, not universal truth.
4. QK / V / Gate / UD should remain behaviorally split.
5. Explicit thought should compile to structure before text.

### Highest-priority open questions

1. Minimal schemas for `StreamDto`, `ResonanceDto`, `BusDto`, and `ThoughtStruct`.
2. Exact representation for HIP and BRANCH.
3. What TWIG and LEAF should materially store.
4. What object is searched in the large field and what object is emitted by sweep.
5. What the smallest viable host-model glove should be.

### Suggested next implementation order

1. Add `StreamDto`, `BusDto`, and `ThoughtStruct`.
2. Wire them into existing hydration, blackboard, and p64 flow.
3. Add HIP family layer with role-aware codebooks.
4. Add BRANCH contradiction handling.
5. Add searchable resonance field and `ResonanceDto`.
6. Add host-model glove.

### Notes

This changelog should remain cumulative.
Each new blackboard file should append:
- what changed
- what got clarified
- what remains uncertain
- what should be tested next
