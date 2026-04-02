# Contract Proposal: User / Agent / Topic / Angle Perspective Modeling

## Purpose

This document proposes a contract for modeling cognition as an intertwining of:

- user
- agent
- topic
- angle
- perspective
- episodic trajectory
- causal graph state

The goal is to stop treating these as loose metadata attached after the fact.
They should instead become interacting objects inside the cognitive stack.

## Core insight

A thought event is rarely about a topic alone.
It is usually a structured relation among:

- who is perceiving
- who is responding
- what is being considered
- from which angle
- under which memory and causal trajectory
- with what tensions, hypotheses, and revisions

So the system should model not just knowledge graph content, but perspective-conditioned traversal.

## Primary objects

### UserModel
Represents the current user as:
- preference attractors
- active concerns
- discourse style
- long-range themes
- episodic anchors
- contradiction sensitivity
- trust or novelty appetite

### AgentModel
Represents the active agent stance as:
- current mode or style
- inference posture
- explanation density
- exploration vs crystallization preference
- self-observed uncertainty
- policy for collapse

### TopicModel
Represents the topic as:
- basin
- family
- branch tensions
- linked graph entities
- active hypotheses
- frontier status

### AngleModel
Represents a perspective slice on the topic as:
- causal angle
- ethical angle
- technical angle
- emotional angle
- strategic angle
- temporal angle
- identity or persona angle

### PerspectiveFrame
Represents a relation among:
- one subject position
- one topic region
- one or more angles
- one memory trace or trajectory
- one style or collapse policy

### TrajectoryArc
Represents a causal or episodic path through time:
- event sequence
- graph revision chain
- support accumulation
- contradiction accumulation
- branch divergence
- stabilization or collapse

## Working interpretation

AriGraph and persona modeling should not be separate subsystems.
They should be treated as complementary views of the same process:

- AriGraph = structural graph of entities, edges, and causal paths
- persona modeling = viewpoint and interpretive bias surface
- episodic memory = time-indexed trajectory of state and revision

Together they form causality trajectories.

## Two geometries at once

The architecture should be able to represent cognition as both:

### Spine trajectory
A durable causal chain or directed path:
- who said what
- what changed
- which support accumulated
- which contradiction revised the graph
- what stabilized enough to become a thought object

### Ripple field
A superpositional local field of:
- competing angles
- hypothesis bundles
- resonance pressure
- perspective interference
- local branch activation

These are not competing metaphors.
They are complementary views:

- spine = path
- ripple = local field
- graph = topology
- thought = temporary stabilization across both

## Suggested contracts

### `PerspectiveFrame`

```rust
pub struct PerspectiveFrame {
    pub subject_kind: SubjectKind,      // user / agent / shared / external actor
    pub subject_ref: SubjectRef,
    pub topic_anchor: TopicAnchor,
    pub angle_mask: AngleMask,
    pub style_mix: StyleMix,
    pub emotional_weight: f32,
    pub epistemic_confidence: f32,
    pub novelty_pull: f32,
    pub contradiction_pull: f32,
    pub episodic_refs: Vec<EpisodicRef>,
}
```

### `TrajectoryArc`

```rust
pub struct TrajectoryArc {
    pub arc_id: ArcId,
    pub nodes: Vec<TrajectoryNode>,
    pub support_accum: f32,
    pub contradiction_accum: f32,
    pub branch_divergence: f32,
    pub stabilization_score: f32,
    pub last_event_idx: u64,
}
```

### `SharedGestalt`

```rust
pub struct SharedGestalt {
    pub user_frame: PerspectiveFrame,
    pub agent_frame: PerspectiveFrame,
    pub topic_anchor: TopicAnchor,
    pub overlap_mask: AngleMask,
    pub tension_mask: AngleMask,
    pub synthesis_pressure: f32,
    pub unresolved_pressure: f32,
}
```

## Mirror-neuron style modeling

This does not require anthropomorphic claims.
A defensible architectural interpretation is:

- simulate likely user perspective shifts
- predict what interpretation the user may be moving toward
- compare that to the agent's current frame
- preserve both overlap and mismatch
- use the gap as a guide for synthesis, clarification, or challenge

This is a perspective-mirroring loop, not a claim about biological neurons.

## Semantic kernel interpretation

A semantic kernel here can be treated as a compact mediation layer that:

- bundles topic, angle, and perspective cues
- interfaces between resonance and explicit traversal
- conditions synthesis and tool use
- provides a small stable surface for host-model guidance

## BNN / reinforcement interpretation

A useful framing is:

- p64 + Blumenstrauß = explicit traversal and composition substrate
- reinforcement = path preference shaping over traversals and collapse policies
- BNN-like dynamics = sparse or bounded state updates over structured topology

This suggests a future direction where reinforcement is not only token reward,
but reward over:

- trajectory coherence
- contradiction handling
- perspective alignment
- novelty without derailment
- stable synthesis across user-agent-topic frames

## Design rules

1. User, agent, topic, and angle should all be first-class modelable objects.
2. Perspective should be representable before textualization.
3. Episodic memory should be stored as causal trajectory, not only as retrieval chunks.
4. Shared gestalt should preserve overlap and tension simultaneously.
5. Hypothesis testing should traverse both graph spine and ripple field.

## Open questions

1. What minimal schema should `UserModel` and `AgentModel` have first?
2. Should `PerspectiveFrame` live inside `ResonanceDto`, `ThoughtStruct`, or both?
3. What is the cleanest representation for shared gestalt and perspective mismatch?
4. How should episodic arcs be revised without identity collapse?
5. What reward signals would make traversal reinforcement meaningful beyond token success?
