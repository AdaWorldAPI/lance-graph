# D-STAUNEN-1 | Learning ISA and ownership | draft
**Docs only, no byte/ABI edits.** Define the thinnest built-in DT64 behavior to bind candidate learning methods over the existing class-agnostic ndarray polyfill and immutable WORLD / sparse THOUGHT rules.

## First pass
- Audit `causal-edge/src/layout.rs`, `isa.rs`, CE64 capstone and all producer/consumer readings of 46..49, 50..52, W and Epi5. O2 50..52 remains unresolved, migration under owner only.
- A CE64 word is a typed learning/reasoning *reading*, not a synapse-state container or hypervector.
- Owner routing: DR/LocalMap maps identities; DT64 method binding `ON×WHEN×DO` names class contract, signatures and effects; Atlas owns numeric neuron/synapse state and schedules; XLOG owns source logic modes; Thought-Git owns epistemic deltas.
- Distinguish biological plasticity / symbolic revision / epistemic certification / metaplasticity. Define state transitions only after independent checks.
- No new ClassId, register, global tenant or opcode. Moore8 is a view, not the full adjacency.

## Research gates
C0 classify current live field readers and contradicting declarations; C1 enum/typed view accepts good and refuses legacy provenance; C2 W slot-generation and source-version witness; C3 write/merge staged vs sealed; C4 novelty signal has explicit source/time/eligibility and cannot certify; C5 signatures/guard/authorization; C6 replay and conservation of evidence roots.

**Prereq:** D-STAUNEN-0, #1597, #1628, #1839, #1869. **Blocking:** O2 writer ruling; ISA legacy migration, thought-generation lifetime.

## Round 2 | Stage-by-stage CC landing specification

### Current contracts to check, not redesign
- CE64 upper half on `main` explicitly documents a **user decision** for activation i4 (46..49) while legacy `isa.rs` still interprets a mantissa/opcode; this is a migration gap. Do **not** silently read both.
- Bits 50..52 are `PlasticityState` in code, while historical docs propose a novelty-to-wisdom rung. O2 has not been ratified. Treat a novelty *event* as separately typed until the owning contract is settled; do not steal three bits for an enum.
- W 53..58: live thought slot; receipt must include base/version and generation. On durable seal the bits remain, but the slot may be reused; recovered evidence identity is **not W**.
- EpistemicState5 = Topology2 × Certification3, including `Open`. A rule join, a spike, a high probability or an activation cannot upgrade to `Causes`; actual intervention evidence is a separate permission/replay contract.
- F/C, Pearl3 and Epi5 remain distinct; CE64 witness/audit is not a foreign key or truth fallback. Keep evidence source-root disjointness outside the word.
- Existing contracts: `lance_graph_contract::lane_codec::EpiLaneClass` / `ClassView` / WideFieldMask (#1839), DT64 `MethodId/SigRev`, `MethodBinding`, K3 WHEN guard (#1869), DR source+frame seals; do not mint a second catalog.

### Proposed minimum learning operation (signature, not new opcode)
```text
learn_attempt(
  borrowed_source_frame, current_thought_generation, learner_method_binding,
  spike_or_derived_event_ref, eligibility_ref, novelty_or_reward_signal_ref,
  evidence_root_set
) -> TypedResult<ProposedDelta | Refusal>
```
- Routes through `DT64::resolve/admit` only after ON/WHEN/DO read and effect checks; DO cannot bypass OGAR authorization.
- Valid candidate is a **sparse THOUGHT delta**, never an immediate WORLD edit. A neural-owned mutable instance may change only via Atlas-authorized commands, with side effects committed at defined simulator boundaries; no rollback illusion.
- A delta records *who* proposed it, *what rule* and *which independently identified source observations* backed it, plus numeric profile and input version.
- Separate read-only inspection from effects. A new `MethodBinding` is derived/versioned rather than mutating the canonical class.
- Explicit `UnsupportedModel`, `SourceMoved`, `FrameChanged`, `UnknownMethod`, `MissingEvidence`, `EffectNotDeclared`, `ThoughtRecycled`, `UnknownEpistemicRail`, `UnauthorizedEffect` refusal classes are candidate typed outcomes; reuse existing native types instead of creating duplicates.

### CC gates with disable-to-red probes
| Gate | Fixture / expected positive | Negative control / failure |
|---|---|---|
| C0 | Generate a *current* producer-consumer table for 46..49, 50..52, W, Epi5 | Refuse conflicting legacy readers rather than infer semantics |
| C1 | Same CE64 word classified under explicit class+rail | Wrong provenance, unknown rail or wrong orientation refuses |
| C2 | W=7 generation A and generation B distinct despite equal bits | Slot-reuse contamination must fail |
| C3 | Sparse hypothesis change passes `write_epi` presence rules | NaN/absent must not overwrite known sealed givens; -0 is not absent |
| C4 | Signed novelty/reward source and predicted comparison timestamp present | Source-free novelty may alter nothing and certifies nothing |
| C5 | DT64 method with proper K3 guard and declared writes can be admitted | UNKNOWN guard, stale signature, unavailable catalog, unauthorized DO refuse |
| C6 | Reset + replay reproduces tagged candidate and preserves protected readers | W-only lineage, duplicate-root confidence inflation and lost intermediate writes fail |

**CC acceptance artifacts:** a single source authority ledger, golden vectors for source/version/refusal/read/write, explicit O2 ruling request, rust source/ABI diagram, bounded in-crate test results, 6 disable checks (one per gate). No benchmark result substitutes for semantic parity. Stop at C0 if bit semantics disagree and an owner is unavailable.

### Research and repository links
- CE64 semantic upper half: https://github.com/AdaWorldAPI/lance-graph/blob/main/docs/architecture/ce64-semantic-upper-half.md
- epistemic capstone: https://github.com/AdaWorldAPI/lance-graph/blob/main/.claude/knowledge/epistemic-frontier-capstone.md
- PR #1597 activation decision: https://github.com/AdaWorldAPI/lance-graph/pull/1597
- PR #1628 thought identity vs durable slot: https://github.com/AdaWorldAPI/lance-graph/pull/1628
- PR #1839 class-dependent U9: https://github.com/AdaWorldAPI/lance-graph/pull/1839
- PR #1869 binding/catalogue: https://github.com/AdaWorldAPI/lance-graph/pull/1869
- Abraham & Bear metaplasticity (not a CE bit map): https://doi.org/10.1016/S0166-2236(96)80018-X
- Miconi et al learning-to-learn plastic networks: https://arxiv.org/abs/1804.02464

**No source edit / no ABI promotion** is authorized by this doc. The first implementer PR should be a lab-only contract probe, after C0 scope and O2 owner decision.
