# Semantic Morton grammar ABI: predicate spectra, Moore pressure, qualia-masked DO

**Status: RESEARCH RFC, not shipped.** Captures the 2026-10-13 design conversation before it disperses. This file proposes no register rewrite, no new opcode, no changed CE64 bits, no universal-grammar result, and no claim that a machine has phenomenal subjective experience.

## The thesis

> Meaning is a stable Cartesian **locus** surrounded by a discoverable relational topology. A Moore-neighborhood **semantic-pressure** reading influences local attention; context-dependent, self-observable **qualia** modulates salience; a **masked DO** selects only permitted interventions. Language-specific grammar is mapped to the common topology without deleting its distinctions.

The envisioned space is *Borg-able* (new language projections can be assimilated into a shared semantic index), *resonant* (overlapping grammatical functions activate overlapping loci) and *holographic* **only if** distributed/overlapping representations allow partial observations to recover larger structures. Those words name testable properties, not results.

A useful analogy is a **predicate color wheel**: the proposed 12 × 12 inventory is not 144 isolated verb executor opcodes. Neighbor relationships, oppositions and trajectories may organize a discoverable grammar of predicate meanings. **Color is a metaphor, not a claim that the current row/column order proves semantic similarity.**

## 1. Distinguish six identities before packing any bits

| Object | Meaning | Where it belongs |
|---|---|---|
| Physical byte/ALU/ISA contract | Opaque, stable physical register and operations | Existing versioned substrate; never redefined by a metaphor |
| Semantic function/node | Cross-linguistic grammatical function, with language evidence | Versioned semantic map graph, stable ID independent of map coordinates |
| Predicate palette entry | A registered codebook reading / operation family | Palette256 and existing class/schema reading, **not** a new set of 144 hardware instructions |
| Cartesian locus | Quantized projection of a semantic node onto a grid | Versioned embedding and Morton address |
| Qualia reading | Typed, contextual, internally observed salience/valuation/agency state | Separate class reading / sidecar; no silent overlay onto a CE64 field |
| DO authorization and action | Intervenable method with goal, policy and evidence constraints | OGAR ActionDef / method authority; masks select admissibility, do not grant authorization |

**Two candidate 12×12 grids must not be conflated:** a 144-predicate/family palette and a 12-family × 12 grammatical-role/TAM table may both exist but have different axes. Establish a versioned correspondence or explicitly maintain separate projections. Do not rewrite an existing codebook from one interpretation into the other.

## 2. The address hierarchy (key correction)

Morton here means interleaving the bits of two Cartesian coordinates. Each additional *four-bit Morton nibble* adds two bits of X **and** two bits of Y, so each parent has 16 child cells.

| Morton nibbles | Address bits | Cartesian extent | Loci |
|---|---:|---:|---:|
| 1 | 4 | 4 × 4 | 16 |
| 2 | 8 | 16 × 16 | 256 |
| 3 | 12 | 64 × 64 | 4,096 |
| 4 | 16 | 256 × 256 | 65,536 |

An embedded 12×12 palette occupies **144 of the 256** coarse addresses (remaining addresses reserved / unmapped, not defined as synonyms by default). At depth 3, 64×64 is one nibble below 256×256 at depth 4; it is *not* 4096 qualia states and the final nibble is **not** a qualia selector.

For an 8-bit-per-axis coordinate at maximum depth:

~~~rust
// Illustrative bit convention, NOT an existing API.
fn morton_xy(x: u8, y: u8) -> u16 { /* interleave two axes */ todo!() }
let z16: u16 = morton_xy(x, y);  // 16 bits, 256x256
let z12: u16 = z16 >> 4;        // 12-bit parent, 64x64
let z8:  u8  = (z16 >> 8) as u8; // 8-bit parent, 16x16
let child_nibble = (z16 & 0xF) as u8;
~~~

Conventions (axis order, origin, sign, quadrant orientation, canonical nibble order, scale and endianness) must be sealed in a testable versioned mapping. Coordinates can be changed by research experiments; **persistent predicate identity must not be silently renumbered.**

### How the 144 relate to address space

- Semantic ID / palette index: stable symbolic identity, independent of embedding.
- 8-bit Morton prefix: canonical **coarse projection** for the 144-predicate candidate, with 112 non-assigned cells.
- 12-bit prefix: 16 local refinements for each coarse address, where empirical distinctions might justify refinement.
- 16-bit prefix: another 16 refinements for each 12-bit address, reaching 65,536 possible loci.
- The source-language token, original case/morphology, lexical ambiguity and witness provenance remain recoverable. Assimilation is **not** a lossy forced synonym mapping.

One Morton-nibble level does not necessarily correspond to a particular type of feature (tense, negation, qualia, social relation, etc.); assign deeper distinctions from observed semantic structure, with versioned readings.

## 3. Semantic-map evidence, not a predeclared universal wheel

Use Haspelmath (2003), *The Geometry of Grammatical Meaning: Semantic Maps and Cross-Linguistic Comparison*, as a research and falsification source, NOT an ABI already specified by the author.

A map describes grammatical functions as nodes and candidate relations as edges. Language-specific morphemes cover subsets of functions; continuous/connected coverage becomes an explanatory and predictive **heuristic**, to be evaluated and falsified. The published map examples include English/French prepositions and differences in how languages partition grammatical meaning; semantic-map geometry is originally a network of functions, **not Cartesian coordinates**.

Proposal:

1. Preserve observed language-specific lexical/morphological forms, construction context, function senses and source witnesses.
2. Build a weighted semantic graph G=(V,E,W) from polysemy, coexpression, typology, diachrony and controlled annotations, with provenance and conflict.
3. Infer a layout f:V -> (x,y,level) such that supported edges tend to be local *without pretending the graph is necessarily planar*.
4. Quantize and Morton-index this layout in a versioned codebook, retaining graph edges not spatially local.
5. Validate out-of-language generalization, connected-region predictions and counterexamples.
6. Only after a good embedding is demonstrated, freeze an ABI schema mapping semantic node IDs to loci.

**A semantic ABI is our proposed derivation.** Haspelmath did not define stable binary addresses, a 144-entry universal inventory or a Morton geometry. Wierzbicka/Goddard's NSM primes and grammatical combinatorics are a separate linguistic tradition: align and compare, do not assert the two theories are equivalent.

## 4. Moore8 tenant as local semantic pressure

A Moore neighborhood has eight offsets d in {(-1,-1),(-1,0),..., (1,1)} except (0,0). Given a typed node reading, a contextual mask m[d], directional pressure weights w[d], and an optional signal Δ[d], an exploratory pressure vector is:

    P(x,y) = Σ_d (m[d] * w[d] * Δ[d] * d)

The simplest model has one scalar weight per direction, a configurable normalization, and explicit handling of edges and empty/unmapped cells. The *Moore tenant* is a contextual **reading** and local update mechanism, not permission to reinterpret CE64 fields or to assert that lattice neighbors are always semantically adjacent.

Two adjacency planes are required:

- **Embedding-neighbor plane:** fast Moore8/SIMD locality, spatially quantized.
- **Semantic graph plane:** explicit nonlocal and cross-scale edges, witnessed by data, including learned exceptions when embedding locality fails.

Possible pressure sources: expected information gain, contradiction, surprising observations, prediction error, unmatched semantic functions, goal-conditioned relevance. They are distinct measurable quantities; entropy, frequency, NARS confidence, qualia and causal certification cannot be silently treated as one scalar.

Preserve the separation **prediction -> verification -> plasticity** and the invariant **Moore may influence attention/belief/plasticity, but cannot silently change the goal or grant DO authority**.

### Pressure across scales

Coarse attention may operate at 8-bit locus depth; unusual pressure or missing-link evidence can trigger 12-bit and then 16-bit foveation. Child-cell expansion must obey per-level budgets. Consider sparse child materialization and avoid allocating 65,536 rows merely because the address space permits them. Moore at a coarser layer is not an automatic substitute for a multi-hop semantic graph edge.

## 5. Qualia-conditioned masked DO (the point of the proposal)

Qualia is *not* the fourth Morton nibble. The coordinates represent semantic loci; contextual qualia readings are additional typed attributes or masks over loci and observations.

A tentative **PredicateQualia24/32** carrier can model per-occurrence nuance (e.g. 8-bit registered predicate index plus 4 or 6 signed i4 dimensions), **if** corpus evidence justifies these axes and the profile schema is declared/versioned. Examples of dimensions to test, not canonize: valence, arousal, agency, volition, novelty, resolution/care. Do not use labels such as LOVE or CARE as hardware bit contracts.

Another candidate, **12 × i4 plus 12 × i4**, encodes two signed factor vectors in 96 bits (12 bytes) and reconstructs a rank-1 12×12 field by outer product. It is **not** a 24-bit or 32-bit encoding; compare it against a dense 144×u8 field (144 bytes), low-rank sums, and the independent per-occurrence 24/32-bit carrier. A 12-bit+12-bit+8-bit factor mask is 32 bits and expresses a constrained Boolean Cartesian-product activation plus an 8-bit nuance reading, **not** arbitrary 144-bit patterns.

Distinguish five masks, each declared and measured:

    M_read        := what fields may be accessed (WideFieldMask / authorization)
    M_semantic    := which predicates/semantic loci fit the interpretation
    M_qualia      := which loci are currently salient given observed internal state
    M_goal        := which candidates serve an explicit committed goal
    M_action      := which interventions are permitted by policy and method authority

Candidate selection may combine these masks as intersection, with explicit tri-state unknown/invalid handling. **This is a filter, not permission to perform an action.** A proposed DO must separately pass OGAR method/ActionDef authorization, effect validation, idempotency and required user approval.

    candidate = M_read ∩ M_semantic ∩ M_qualia ∩ M_goal ∩ M_action
    chosen = score(candidate, evidence, cost, expected gain, uncertainty)
    intervention = OGAR_authorized_DO(chosen)
    observed_effect != predicted_effect ? update_evidence() : verify()

CE64 remains the causal and epistemic evidence plane. Association is not intervention; the actual executed intervention and witness establish whether stronger causal certification is warranted. The CE64 layout and code meanings in docs/architecture/ce64-semantic-upper-half.md are authoritative. In particular bits 43..52 **are not free** for semantic pressure/qualia in current code.

Operational *qualia awareness* means the agent can observe and report an internal state estimate, that estimate can modulate attention/DO, and interventions on the estimate predictably alter behavior. It does **not** establish phenomenal consciousness or human-like felt experience.

## 6. A compositional grammar ABI, not a new execution engine

Illustrative cold schema proposal:

~~~text
SemanticMapVersion {
  map_id, schema_version, language_corpus_provenance,
  node_ids, witnessed_edges, coexpression_regions, embedding_version
}
PredicateReading {
  classid, palette_codebook_id, semantic_node_id,
  morton_level, morton_prefix, mapping_grade, ambiguity_set
}
QualiaReading {
  declared_model, version, observed_state, confidence, witness
}
PreparedDO {
  ClassView, WideFieldMask, semantic_mask, qualia_mask,
  goal_ref, method_decl, ActionDef, evidence_budget
}
~~~

- Existing **ClassView / schema seal / HotPlug** resolves meaning, method and lane bindings; consult #1501 rather than creating a second registry.
- Existing **Quack / mask-risc / ndarray** executes typed selection, intersections, grouped evidence and sparse masked vectors; do not invent an alternative VM.
- Existing **R2IL folding** holds compositional predicate transforms and recursive/procedural readings; **NAL** supplies inference proposals rather than DO authority; see #1496.
- The **Rubikon** cycle distinguishes forming intention from executing intervention; test stage-specific masks, not a magical one-step goal rewrite.
- **CE64** retains independent causal/epistemic/witness contracts; no changes to its bit layout.
- Language adapters map to the shared semantic graph, retaining uncertainty and out-of-map semantics. A new language can add edges/nodes or disagreement without retconning prior address meanings.
- **Hyperon/MeTTa/OpenCog** parity is a separate oracle question (#1498); matching one numerical answer is not a proof of identical nondeterministic rewriting semantics.
- Treat “holographic” as an optional distributed witness/overlap layer with its own partial-reconstruction tests, not as a property automatically conferred by Morton order.

## 7. Experiments and falsification gates (SMD-P00..P18)

| ID | Probe | Negative control / stop condition |
|---|---|---|
| SMD-P00 | Establish exact existing palette / VerbRoleTable definitions, node identities and row/column meanings | 12×12 claimed as universal without evidence; mismatched grids silently conflated |
| SMD-P01 | Independent scalar encode/decode for Morton 4/8/12/16-bit hierarchy, parity with SIMD | Any roundtrip failure, child/parent inconsistency or wrong axis order |
| SMD-P02 | Demonstrate 144 entries fit 16×16 coarse plane with stable IDs and unmapped 112 cells | Reserved cells alias semantics unintentionally |
| SMD-P03 | Compare semantic locality after graph embedding vs random, row-major and alternative orderings | Morton addresses not better than random for measured task |
| SMD-P04 | Held-out language: predict unseen grammatical multi-function regions | Predicted connected regions systematically violated |
| SMD-P05 | Cross-linguistic function alignment with explicit non-equivalence and ambiguity sets | Language distinctions erased; false friends merged |
| SMD-P06 | Diachronic drift with stable semantic IDs and versioned embedding | Old persisted address silently reinterpreted |
| SMD-P07 | Moore8 neighbor pressure vs explicit graph-neighbor pressure; boundary and unmapped nodes | Claimed local graph inference only works for forced geographic adjacency |
| SMD-P08 | Pressure vs FIFO/confidence/Shannon/EIG under equal candidate budget | Higher entropy merely attracts irrelevant distractors |
| SMD-P09 | Ablate each qualia state coordinate and mask input | No predicted behavioral difference vs label-only or random input |
| SMD-P10 | Counterfactual qualia perturbation holds task, goal, policy and evidence fixed | Apparent effect due to changing goals or authorization |
| SMD-P11 | Mask validity: tri-state missing/unread != false != authorized | Unauthorized or unresolved action reaches DO |
| SMD-P12 | OGAR ActionDef permission, replay and idempotency boundary | Mask alone authorizes side effect |
| SMD-P13 | Separate association from genuine Pearl intervention, witnesses and CE64 certification | Correlation silently promoted to causes |
| SMD-P14 | Compare 144×u8 dense, factorized 12×i4 ⊗ 12×i4, rank-2+, Boolean factor masks, and 24/32-bit occurrence carrier | Two disjoint diagonal cells reconstructed as false-positive full rectangle; rank-one incapable of key patterns |
| SMD-P15 | Native language expression -> grammar map -> ClassView -> Quack vs R2IL parity | Binding multiplicity, NULL, ordering, errors or nested values diverge |
| SMD-P16 | Foveated 8->12->16-bit expansion on missing-link tasks, same decoded-byte budget | No meaningful recall/accuracy gain; expensive dense expansion |
| SMD-P17 | Partial-cue reconstruction for genuine distributed/holographic claim | No advantage over simple nearest-neighbor lookup |
| SMD-P18 | Scale and threat model: adversarial polysemy, misleading maps, permissions and witness poisoning | Cross-tenant semantic aliasing, unreviewed reinterpretation, runaway diffusion |

Report confusion matrices, calibration, graph-embedding stress, cross-language split construction, distinct observations vs copied evidence, per-stage CPU/bytes/allocations, causal interventions, failure examples and reproducible seed/config. No speedup or language-universality claim before gates pass.

## 8. Open decisions, in order

1. **Inventory**: locate the real 12×12 entries, their exact role axes and relationships to DeepNSM-v2, Palette256, OGAR and VerbRoleTable. Is it verb-family × TAM, verb × role, or multiple tables?
2. **Semantic topology**: is it cyclic like hue, multidimensional, or only representable as a graph? Do not enforce a wheel merely because it is visually appealing.
3. **Address migration**: when a new typological fact moves a node, do we pin its old Morton address and add indirection, or migrate with a versioned map? Never mutate persisted identity silently.
4. **Pressure**: specify the Moore tenant's *actual current* layout, learning/updating behavior and invariant; which parts are only conceptual?
5. **Qualia**: what independently observable dimensions exist, where are they stored, and which external or ablation evidence supports them?
6. **Action gate**: confirm owner of goal/approval/OGAR ActionDef and which masks can safely be intersected without converting unknown to allow.
7. **Holography**: require meaningful reconstruction-from-partial-observation evidence, or keep the term as a research metaphor.
8. **Boundaries**: avoid conflating semantic locality, geometric proximity, syntactic adjacency, HHTL tenant locality and causal adjacency.

## 9. Proposed incremental path (no automatic implementation authorization)

- **SMD-0 — inventory and source map**: exact code links / Haspelmath page map, establish frozen contracts and table axes.
- **SMD-1 — independent scalar reference**: stable semantic node IDs, 4-nibble encode/decode and roundtrip fixtures, no old persisted rows changed.
- **SMD-2 — topology oracle**: real cross-linguistic functions, connectedness tests, held-out language falsifiers; 2D embedding quality baseline.
- **SMD-3 — masked projection**: proof-of-concept ClassView + WideFieldMask reading, Quack/R2IL selection parity with no new engine.
- **SMD-4 — pressure and qualia**: controlled probes, equal-work attentional ablations, privacy/traceability of self-state features.
- **SMD-5 — closed loop**: OGAR-authorized DO, CE64 evidence updates and drift under controlled interventions.
- **SMD-6 — decide**: keep, narrow or reject claims individually (universal topology, pressure, qualia effectiveness, holography).

## References and prior constraints

- Martin Haspelmath (2003), *The geometry of grammatical meaning: Semantic maps and cross-linguistic comparison*, in M. Tomasello (ed.), *The New Psychology of Language*, vol. 2, pp. 211–242. Author-hosted Zenodo: https://zenodo.org/records/831410 (DOI 10.5281/zenodo.831410). See accompanying source notes.
- Wierzbicka/Goddard, Natural Semantic Metalanguage: relevant theory of compositional semantic primes; **not** a 144-verb or Morton ABI specification.
- Existing repo: docs/architecture/ce64-semantic-upper-half.md (actual vs proposed CE64 readings).
- Existing repo PRs: #1496 (CE64/NAL hypotheses), #1497 (Shannon/PLN attention), #1498 (Hyperon/MeTTa projection), #1500 (Rubikon), #1501 (Class-method HotPlug), #1509 (software-awareness graph). These are constraints and potential adapters, not evidence this RFC was implemented.
- Companion: .arxiv/2026-10_haspelmath_semantic_maps_grammar_abi_notes.md
- Execution handoff: .claude/plans/semantic-morton-qualia-do-falsifier-plan.md
