# Ripple Architecture File Index

## Purpose

This file helps a parallel session enter the current architecture work without re-deriving the whole map.

The goal is:

- reduce communication debt
- preserve additive progress
- keep later coding integration separate from current architectural synthesis

## Read first

### 1. Main architecture spine
- `docs/integrated-architecture-map.md`

What it does:
- epiphanies first
- existing repo anchors
- architecture reframe
- layer hierarchy
- DTO stack
- implementation phases

### 2. Core knowledge spine
- `.claude/knowledge.md`
- `.claude/ripple-project-readme.mde`

What they do:
- compress doctrine
- identify key distinctions
- name the intended layer model

### 3. Blackboard memory
- `.claude/blackboard-ripple-architecture-20260402-01.MD`
- `.claude/blackboard-ripple-architecture-20260402-02.MD`
- `.claude/blackboard-ripple-architecture-changelog.md`

What they do:
- hold evolving synthesis
- preserve open questions
- track what has already been established
- connect perspective, trajectory, and reinforcement ideas into the main thread

## Contracts

### DTO and object contracts
- `.claude/contracts/ripple-dto-contracts.md`

Focus:
- `StreamDto`
- `ResonanceDto`
- `BusDto`
- `ThoughtStruct`

### Representation contracts
- `.claude/contracts/ripple-representation-contract.md`

Focus:
- HEEL
- HIP
- BRANCH
- TWIG
- LEAF
- role-aware family split

### Perspective contracts
- `.claude/contracts/user-agent-topic-perspective-contract.md`

Focus:
- `UserModel`
- `AgentModel`
- `TopicModel`
- `AngleModel`
- `PerspectiveFrame`
- `SharedGestalt`
- `TrajectoryArc`

## Knowledge extensions

- `.claude/knowledge/user-agent-topic-ripple-model.md`

Focus:
- user / agent / topic / angle interplay
- graph vs ripple vs trajectory views
- shared gestalt
- mirror-style loops
- reinforcement over traversal quality

## Orchestration

- `.claude/agent2agent-orchestrator-prompt.md`
- `.claude/agents/README.md`

What they do:
- define agent-to-agent discipline
- keep unlike architectural concerns from collapsing into one voice

## Agent cards

### Existing
- `.claude/agents/container-architect.md`

### Added for ripple architecture
- `.claude/agents/ripple-architect.md`
- `.claude/agents/resonance-cartographer.md`
- `.claude/agents/bus-compiler.md`
- `.claude/agents/contradiction-cartographer.md`
- `.claude/agents/thought-struct-scribe.md`
- `.claude/agents/family-codec-smith.md`
- `.claude/agents/host-glove-designer.md`
- `.claude/agents/perspective-weaver.md`
- `.claude/agents/trajectory-cartographer.md`
- `.claude/agents/mirror-kernel-synthesist.md`

## Best next pressure points for the other session

1. Minimal exact schemas for:
   - `ResonanceDto`
   - `BusDto`
   - `ThoughtStruct`
   - `PerspectiveFrame`
   - `TrajectoryArc`
2. Exact numerical form for:
   - HIP
   - BRANCH
   - contradiction preservation
   - shared gestalt overlap/tension
3. Smallest benchmarkable host-model glove.
4. Reinforcement targets for `p64 + Blumenstrauß` traversal quality.
5. Whether perspective-aware traversal materially beats topic-only traversal.

## Additive collaboration rule

For now, new work should be additive.

That means:
- add notes
- add contracts
- add schemas
- add benchmarks
- add pressure tests
- add questions
- add blackboard updates

Avoid for now:
- broad integration coding
- invasive refactors
- collapsing several unresolved objects into one implementation prematurely

## One-line handoff

The repo already contains the anatomy, bus, blackboard, and trajectory hints of a cognition stack. The current mission is to deepen the object model and preserve distinctions, not to pay integration debt too early.
