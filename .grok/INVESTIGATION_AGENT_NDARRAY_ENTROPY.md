# INVESTIGATION AGENT + NDARRAY + ENTROPY WORK

**Version**: 0.2 (Updated with pre-integrated ndarray 20-200 ns random access + explicit Entropy handling)  
**Status**: Authoritative spec for the second vertical

## 1. Core Insight (Reinforced)

The Investigation Agent is **not** a layer on top of the substrate. It **is** the substrate performing a particular operation:

- Enter SoA at a GUID (already 20–200 ns via ndarray-backed cursor)
- Follow CausalEdge64 (ndarray-accelerated)
- Accumulate context while updating an AwarenessColumn that explicitly tracks **entropy** (ambiguity, density, contradictions, sparse vs rich regions)
- Stabilize the signature
- MUL gate reads both the signature **and** the packed OWL+DOLCE schema invariants
- Output typed EscalationMessage (tokens only)

Because ndarray is **already integrated everywhere**, every traversal step, gather, and AwarenessColumn update inherits the 20–200 ns random access performance. This makes the agent practical even when it walks dozens or hundreds of related entities.

## 2. Entropy Work — What It Means Here

"Entropy work" = deliberately modeling, measuring, and acting on **uncertainty and ambiguity** in the data rather than pretending perfect knowledge.

HIRO has none of this: rules fire or they don’t; multi-match is resolved by ordering; misroutings surface only after human complaint.

Our system makes entropy first-class:

### 2.1 AwarenessColumn as Entropy Accumulator
The column (256-byte signature) is updated on every visited row. It encodes:
- Information density along traversed paths
- Number and severity of contradictions (multiple high-confidence but incompatible edges)
- Sparse regions (data that should exist but is missing)
- Convergence / divergence of multiple paths
- Semantic embedding augmentation (via embedanything + ndarray) when lexical signals are weak

When the signature **stabilizes** (further traversal changes it below a threshold), the agent has either built a coherent picture or has hit the ceiling of what the data supports. High remaining entropy → MUL is more likely to escalate.

### 2.2 MUL Gate + Schema Invariants (Free Entropy Signals)
The packed schema (produced by glue layer) gives MUL **free, high-quality priors** at almost zero cost:
- Functional property returning >1 candidate → strong "this data has entropy / is inconsistent" signal → hard veto path in `is_unskilled_overconfident()`.
- DOLCE Endurant vs Perdurant → different intervention/counterfactual semantics (Pearl masks).
- Domain/range violations (should never reach MUL because validator already rejected them).

MUL therefore spends its budget on **genuine epistemic uncertainty** rather than on defending against structurally impossible data.

### 2.3 Drift Detection Under Noise
Drift signatures are AwarenessColumn shapes that correlated with past successful escalations. Because the column explicitly tracks entropy, drift detection works even on noisy or partial data:
- Rising entropy in authentication + patch + disk pressure patterns → early warning before ticket arrives.
- The same ndarray fast paths + Bgz tensor storage make continuous background scanning feasible.

### 2.4 Calibration & Brier Work
Every MUL outcome (confident action vs hold-for-human) is recorded with the final AwarenessColumn signature. Over time this builds a calibration dataset. The system learns:
- "When entropy signature looks like X, my confident decisions are only 70% correct → I should raise the bar."

This is the feedback loop HIRO fundamentally lacks.

## 3. Performance Reality Check (ndarray Integration)

Because ndarray is pre-integrated:
- Single SoA row access / column gather: **20–200 ns**
- Typical investigation (30–80 related entities): still comfortably sub-millisecond on modern hardware
- AwarenessColumn update + signature math: dominated by the same ndarray primitives
- embedanything GGUF call (when used): also rides on ndarray backend

This is why the investigation agent can be the **second** vertical instead of the tenth — the heavy lifting for fast random access was already done.

## 4. Integration Points (Updated)

- **SoACursor / SoA views**: Already ndarray-backed → use directly.
- **PackedSchema validation**: Called on every visited row or batch; uses ndarray SIMD kernels for the hot paths.
- **AwarenessColumn**: Stored via Bgz tensor (compressed, random-access) inside Lance; updated with ndarray ops.
- **embed_anything DTO**: Called opportunistically for semantic features when lexical/structural entropy is high.
- **MUL gate**: Receives both stabilized signature and schema-derived priors in one call.
- **Ractor supervision**: One `InvestigationActor` per active investigation (or process group); crashes are isolated.

## 5. Entropy Work Checklist for Implementation

When coding the agent, ensure every major component explicitly surfaces or consumes entropy:

- [ ] AwarenessColumn update function documents what entropy signals it captures.
- [ ] MUL assessment call site passes both signature **and** relevant schema property characteristics.
- [ ] Drift signature matcher has a "minimum entropy reduction" threshold before firing an escalation.
- [ ] EscalationMessage includes a human-readable `entropy_notes` or `ambiguity_summary` field.
- [ ] Calibration loop (Brier) is wired from day one on the Routing vertical’s MUL outcomes.
- [ ] Documentation (this file + CLAUDE.md / .claude/) explicitly calls out the entropy-handling philosophy.

## 6. Relation to HIRO Replacement & Foundry Assimilation

- HIRO: No entropy model → misrouting is discovered late by humans.
- Our agent: Entropy is modeled continuously → problems are escalated **before** they become customer-visible tickets, with context attached.
- Foundry: Strong operational ontology + ML integration. We assimilate both **plus** native causal + uncertainty reasoning on a single ultra-fast ndarray + Lance substrate.

The investigation agent is the concrete deliverable that makes the "preemptive operational twin" promise real.

---

This spec, together with the Routing example, the code sketch, and the four earlier fanout documents, gives a complete, entropy-aware picture of how the first two verticals deliver a dramatically better system than HIRO while assimilating Foundry-class capabilities on our own terms.

**Documentation awareness updated. Entropy work explicitly called out and required in implementation.**