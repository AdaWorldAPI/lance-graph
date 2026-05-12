# Sprint-3 Execution Plan — Tier-1 Implementation Specs

> **Status:** Specs ready for engineer execution. After this sprint, picking up TD-OGIT-G-SLOT-1 (or any TD-X) means reading one PR-shaped spec, not designing from scratch.
>
> **Predecessors:** sprint-2 produced the 15-pattern synthesis + 11 TD entries. Sprint-3 converts those TD entries into PR-ready specs.
>
> **Architectural reference:** all pattern letters use the canonical W1 master assignment from `.claude/plans/unified-ogit-architecture-v1.md` (and the post-PR-#359 corrected `.claude/knowledge/tier-0-pattern-recognition.md`).

## What this plan covers

12 PR-shaped implementation specs covering:

1. **Tier-1 wiring** (Patterns A + B + C): SPO-G u32 slot, ContextBundle typed surface, GenericBridge dispatcher
2. **Tier-2 supervised consumer mesh** (Patterns E + F): manifest.yaml + build script + ractor supervisor port
3. **Tier-3 specialization** (Patterns J): INT4-32D thinking atoms for new-domain bootstrap
4. **Pattern D first hydrator**: FMA OWL hydrator (also opens the anatomy-realtime-v1 demo path)
5. **Trivia closures** (TD-9, TD-10, TD-11): one-liner + small fixes
6. **Consumer template**: how to add the Nth consumer crate (hubspo-rs as worked example)
7. **Sequencing + smoke test**: PR dependency graph + end-to-end validation

## Execution path overview (4 weeks)

### Week 1 — Foundation (Tier 1 types)
1. PR-B-1 (W3 spec) — `ContextBundle` typed surface + slot stubs
2. PR-A-1 (W2 spec) — SPO-G u32 slot in quad store + writer
3. PR-C-1 (W4 spec) — GenericBridge consuming ConsumerPointer
4. Trivia (W12 spec): PR-CAM-DIST + PR-ADJ-THINK-EXPOSE + PR-DEEPNSM-NSM-COLLAPSE

### Week 2 — Supervised consumer mesh (Tier 2)
5. PR-E-1 (W5 spec) — `/modules/<name>/manifest.yaml` + build script
6. PR-F-1 (W6 spec) — ractor supervisor port from gRPC trait shape

### Week 3 — Specialization + first hydrator
7. PR-J-1 (W7 spec) — INT4-32D thinking atoms
8. PR-D-1 (W9 spec) — FMA OWL hydrator (Pattern D first concrete)

### Week 4 — Validation + new consumer dry-run
9. Smoke test (W11 spec) — end-to-end OGIT-G validation
10. Consumer template dry-run (W8 spec) — scaffold hubspo-rs in <1 day

## PR-by-PR summary table

| PR | Spec doc | Pattern | TD | Effort | Depends on |
|---|---|---|---|---|---|
| PR-A-1 | `pr-a-1-spo-g-u32-slot.md` (W2) | A | TD-OGIT-G-SLOT-1 | medium ~300 LOC | PR-B-1 |
| PR-B-1 | `pr-b-1-context-bundle.md` (W3) | B | TD-CONTEXT-BUNDLE-2 | small ~200 LOC | (foundation) |
| PR-C-1 | `pr-c-1-generic-bridge.md` (W4) | C | TD-GENERIC-BRIDGE-3 | medium ~200 LOC | PR-A-1 + PR-B-1 |
| PR-E-1 | `pr-e-1-manifest-modules.md` (W5) | E | TD-MANIFEST-MODULES-4 | medium ~330 LOC | PR-B-1 |
| PR-F-1 | `pr-f-1-ractor-supervisor.md` (W6) | F | TD-RACTOR-SUPERVISOR-5 | large ~400 LOC | PR-C-1 + PR-E-1 |
| PR-J-1 | `pr-j-1-int4-32d-atoms.md` (W7) | J | TD-INT4-32D-ATOMS-6 | small ~120 LOC | PR-B-1 |
| PR-D-1 | `pr-d-1-fma-owl-hydrator.md` (W9) | D | (anatomy demo) | medium ~600 LOC | PR-A-1 + PR-B-1 |
| PR-CAM-DIST | `trivia-prs-bundle.md` (W12) | (CAM-DIST-1 reframe) | TD-CAM-DIST-REGISTRATION-9 | trivial 1 line | — |
| PR-ADJ-THINK | `trivia-prs-bundle.md` (W12) | (ADJ-THINK-1 reframe) | TD-ADJ-THINK-EXPOSE-10 | trivial ~30 LOC | — |
| PR-DEEPNSM-NSM | `trivia-prs-bundle.md` (W12) | (DEEPNSM-NSM-1 closure) | TD-DEEPNSM-NSM-COLLAPSE-11 | small ~30 LOC + 5 deletes | — |

Plus Sprint-3 supporting docs: PR sequencing graph (W10) + smoke test design (W11) + consumer template (W8).

## Pattern letter status (canonical, post-PR #359)

| Pattern | Status | This sprint's progress |
|---|---|---|
| A SPO-G u32 slot | DESIGN PHASE | PR-A-1 spec'd by W2 |
| B Context Bundle | DESIGN PHASE | PR-B-1 spec'd by W3 |
| C Generic Bridge | DESIGN PHASE | PR-C-1 spec'd by W4 |
| D Meta-Structure Hydration | DESIGN PHASE | PR-D-1 (FMA hydrator) spec'd by W9 |
| E Compile-Time Consumer Binding | DESIGN PHASE | PR-E-1 spec'd by W5 |
| F ractor Supervisor | DESIGN PHASE | PR-F-1 spec'd by W6 |
| G Best-Practice Thinking Inheritance | DESIGN PHASE | (deferred — needs A+B+C foundation first) |
| H Switchable Cognitive Vessel | SHIPPED | (extension only via G-parameter wiring after Tier 1) |
| I Implicit Cognition | SHIPPED | (no work needed) |
| J INT4-32D Atoms | DESIGN PHASE | PR-J-1 spec'd by W7 |
| K Circular Compilation | ASPIRATIONAL | (deferred to post-Tier-2) |
| L SPO-Chain Narrative | PARTIALLY SHIPPED | (deferred — needs A+L glue after Tier 1) |
| M Wave-Particle Bimodal | PRIMITIVES SHIPPED | (no work this sprint) |
| N Fingerprint-as-Codebook | SHIPPED | (no work needed) |
| O Phenomenological Memory | SHIPPED | (no work needed) |

## Risk callouts

1. **PR-B-1 must land first** — every other Tier-1 PR depends on the `ContextBundle` type. Single point of bottleneck for Week 1.
2. **PR-F-1 is large (~400 LOC)** — the ractor supervisor port from the gRPC trait shape is mechanical but spans 7 actor handlers. Risk of scope creep into Tier-3 territory.
3. **PR-D-1 FMA hydrator** has external dependency (rio_xml or similar OWL parser); pin the version early.
4. **Consumer template dry-run (W8 spec)** is the validation that the architecture reduces per-consumer cost from ~800 LOC to ~30 LOC. If this dry-run produces ~300 LOC, we have a regression in the ConsumerPointer design.

## Acceptance criteria for sprint-3

- [ ] All 12 spec docs exist on branch
- [ ] Each spec includes: file paths, code sketches, test plan, dependencies
- [ ] PR sequencing graph (W10 spec) shows topological order
- [ ] Smoke test design (W11 spec) covers A+B+C+E+F end-to-end
- [ ] No new code written this sprint (specs only)
- [ ] PR opened against main with all 12 specs

## Cross-references

- `.claude/plans/unified-ogit-architecture-v1.md` (sprint-2 master)
- `.claude/plans/ogit-g-context-bundle-v1.md` (Tier-1 sub-plan)
- `.claude/plans/compile-time-consumer-binding-v1.md` (Tier-2 sub-plan)
- `.claude/plans/anatomy-realtime-v1.md` (proof-of-vision)
- `.claude/knowledge/tier-0-pattern-recognition.md` (canonical pattern letters, post-PR #359)
- `.claude/board/TECH_DEBT.md` (the 11 TD rows being spec'd)
- `.claude/patterns.md` (Pattern Recognition Framework)
